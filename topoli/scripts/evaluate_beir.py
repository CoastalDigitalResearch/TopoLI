"""Evaluate TopoLI-1B on BEIR benchmark.

Usage:
    python -m topoli.scripts.evaluate_beir \
        --checkpoint /path/to/finetuned/checkpoint \
        --datasets nfcorpus fiqa scifact
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

BEIR_DATASETS = (
    "msmarco",
    "nfcorpus",
    "fiqa",
    "arguana",
    "touche-2020",
    "dbpedia-entity",
    "scidocs",
    "fever",
    "climate-fever",
    "scifact",
    "nq",
    "hotpotqa",
    "signal-1m",
    "bioasq",
    "trec-covid",
    "cqadupstack",
    "quora",
    "robust04",
)


def main() -> None:
    """Run BEIR evaluation."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    logger.info("BEIR evaluation pipeline:")
    logger.info("1. Load fine-tuned TopoLI-1B checkpoint")
    logger.info("2. For each BEIR dataset:")
    logger.info("   a. Download corpus + queries + qrels")
    logger.info("   b. Encode all passages with model")
    logger.info("   c. Build index (brute-force or FAISS)")
    logger.info("   d. Retrieve top-1000 for each query")
    logger.info("   e. Compute MRR@10, NDCG@10, Recall@100, Recall@1000")
    logger.info("3. Aggregate results across datasets")
    logger.info("")
    logger.info("Available BEIR datasets: %s", ", ".join(BEIR_DATASETS))


if __name__ == "__main__":
    main()
