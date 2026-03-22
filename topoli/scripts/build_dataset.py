"""Build the TopoLI-Retrieval dataset from scratch.

Usage:
    python -m topoli.scripts.build_dataset --output-dir ./topoli-retrieval-v1
    python -m topoli.scripts.build_dataset --num-passages 1000 --skip-queries

Extracts passages from permissive-license sources, generates queries
via doc2query (Qwen3-8B Apache-2.0), filters, deduplicates, mines
hard negatives, and outputs JSONL with full provenance.

All data sources are commercially usable (Apache-2.0, MIT, CC-BY, CC0).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from topoli.data.dataset_builder import build_manifest, write_passages_jsonl
from topoli.data.hard_negatives import BM25NegativeMiner
from topoli.data.loader import (
    DataLoaderConfig,
    get_passage_sources,
    load_miracl_pairs,
    load_passages_from_hf,
    load_triviaqa_pairs,
)
from topoli.data.quality_filter import deduplicate_queries, filter_pair
from topoli.data.query_generator_impl import (
    QueryGenPipeline,
    build_hf_generate_fn,
    build_vllm_generate_fn,
)
from topoli.data.source_config import License, get_source_registry

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build TopoLI-Retrieval dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./topoli-retrieval-v1"),
    )
    parser.add_argument(
        "--num-passages",
        type=int,
        default=10000,
        help="Passages per source (default: 10000, production: 500000)",
    )
    parser.add_argument(
        "--query-model",
        type=str,
        default="Qwen/Qwen3-8B",
    )
    parser.add_argument(
        "--query-batch-size",
        type=int,
        default=512,
        help="Batch size for query generation (controls prompt chunking)",
    )
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="Use vLLM for inference (requires native CUDA, not GPU-over-TCP)",
    )
    parser.add_argument(
        "--n-negatives",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--skip-queries",
        action="store_true",
        help="Skip query generation (for testing passage extraction only)",
    )
    parser.add_argument(
        "--skip-existing-qa",
        action="store_true",
        help="Skip loading TriviaQA/MIRACL",
    )
    return parser.parse_args()


def main() -> None:
    """Build the TopoLI-Retrieval dataset."""
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ─── Verify all sources ──────────────────────────────────────────────
    registry = get_source_registry()
    logger.info("Verified %d data sources (all commercially usable):", len(registry))
    for src in registry:
        logger.info("  %s (%s) — %s", src.name, src.license.value, src.domain.value)

    # ─── Step 1: Extract passages from HuggingFace sources ───────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info(
        "STEP 1: Extracting passages (target: %d per source)", args.num_passages
    )
    logger.info("=" * 60)

    loader_config = DataLoaderConfig(num_passages_per_source=args.num_passages)
    all_passages = []

    for source in get_passage_sources():
        logger.info("Loading from %s...", source.name)
        try:
            passages = load_passages_from_hf(source, loader_config)
            logger.info("  Got %d passages from %s", len(passages), source.name)
            all_passages.extend(passages)
        except Exception:
            logger.exception("  FAILED to load %s, skipping", source.name)

    logger.info("Total passages extracted: %d", len(all_passages))

    # Write passages
    passages_path = output_dir / "passages.jsonl"
    write_passages_jsonl(all_passages, passages_path)
    logger.info("Wrote passages to %s", passages_path)

    # ─── Step 2: Load existing QA pairs ──────────────────────────────────
    existing_pairs: list[dict[str, str]] = []
    if not args.skip_existing_qa:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 2: Loading existing QA pairs (TriviaQA, MIRACL)")
        logger.info("=" * 60)

        triviaqa_pairs = load_triviaqa_pairs(max_pairs=min(args.num_passages, 50000))
        logger.info("TriviaQA: %d pairs", len(triviaqa_pairs))
        existing_pairs.extend(triviaqa_pairs)

        miracl_pairs = load_miracl_pairs(max_pairs=min(args.num_passages, 30000))
        logger.info("MIRACL: %d pairs", len(miracl_pairs))
        existing_pairs.extend(miracl_pairs)

        existing_path = output_dir / "existing_qa_pairs.jsonl"
        with existing_path.open("w") as f:
            for pair in existing_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        logger.info("Wrote %d existing pairs to %s", len(existing_pairs), existing_path)

    # ─── Step 3: Generate queries via doc2query ──────────────────────────
    generated_pairs: list[dict[str, str]] = []
    if not args.skip_queries:
        logger.info("")
        logger.info("=" * 60)
        engine = "vLLM" if args.vllm else "HuggingFace"
        logger.info(
            "STEP 3: Generating queries with %s via %s", args.query_model, engine
        )
        logger.info("=" * 60)

        if args.vllm:
            generate_fn = build_vllm_generate_fn(
                model_name=args.query_model,
                max_new_tokens=64,
                temperature=0.7,
                gpu_memory_utilization=0.90,
            )
        else:
            generate_fn = build_hf_generate_fn(
                model_name=args.query_model,
                max_new_tokens=64,
                temperature=0.7,
            )
        pipeline = QueryGenPipeline(
            generate_fn=generate_fn,
            model_name=args.query_model,
            model_license=License.APACHE_2_0,
            batch_size=args.query_batch_size,
        )
        generated_pairs = pipeline.generate(all_passages)
        logger.info("Generated %d query-passage pairs", len(generated_pairs))

        # Filter
        logger.info("Filtering...")
        filtered = [
            p
            for p in generated_pairs
            if filter_pair(query=p["query"], passage=p["passage_text"])
        ]
        logger.info("After filtering: %d pairs", len(filtered))

        # Deduplicate
        queries = [p["query"] for p in filtered]
        deduped = set(deduplicate_queries(queries))
        filtered = [p for p in filtered if p["query"] in deduped]
        logger.info("After dedup: %d pairs", len(filtered))
        generated_pairs = filtered

        gen_path = output_dir / "generated_pairs.jsonl"
        with gen_path.open("w") as f:
            for pair in generated_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        logger.info("Wrote generated pairs to %s", gen_path)
    else:
        logger.info("Skipping query generation (--skip-queries)")

    # ─── Step 4: Mine hard negatives ─────────────────────────────────────
    if generated_pairs and all_passages:
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 4: Mining BM25 hard negatives")
        logger.info("=" * 60)

        passage_map = {p.passage_id: p.text for p in all_passages}
        miner = BM25NegativeMiner(passage_map)

        queries_for_mining = [
            {"query": p["query"], "passage_id": p["passage_id"]}
            for p in generated_pairs
        ]
        hard_negs = miner.batch_mine(queries_for_mining, n_negatives=args.n_negatives)

        neg_path = output_dir / "hard_negatives.jsonl"
        with neg_path.open("w") as f:
            for pair, negs in zip(generated_pairs, hard_negs, strict=True):
                record = {
                    "query_id": pair.get("passage_id", ""),
                    "query": pair["query"],
                    "positive_id": pair["passage_id"],
                    "negative_ids": negs,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Wrote hard negatives to %s", neg_path)

    # ─── Step 5: Write manifest ──────────────────────────────────────────
    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 5: Writing manifest")
    logger.info("=" * 60)

    manifest = build_manifest(all_passages)
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(manifest.model_dump_json(indent=2))
    logger.info("Manifest: %s", manifest.model_dump())

    elapsed = time.time() - t0
    logger.info("")
    logger.info("=" * 60)
    logger.info("DONE in %.1f minutes", elapsed / 60)
    logger.info("Output: %s", output_dir)
    logger.info(
        "Passages: %d | Existing QA: %d | Generated: %d",
        len(all_passages),
        len(existing_pairs),
        len(generated_pairs),
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
