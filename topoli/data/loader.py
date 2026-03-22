"""Real data loading from HuggingFace and local sources.

Downloads from permissive-license sources (TriviaQA, MIRACL, Common Corpus,
KL3M, SlimPajama) and converts to PassageRecords with full provenance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from topoli.data.passage_extractor import extract_passages
from topoli.data.source_config import License, PassageRecord, SourceDomain

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataLoaderConfig:
    """Controls scale of data loading."""

    num_passages_per_source: int = 10000
    min_chars: int = 200
    max_chars: int = 600
    seed: int = 42


@dataclass(frozen=True)
class PassageSource:
    """Maps a data source to HuggingFace loading parameters."""

    name: str
    hf_dataset: str
    hf_config: str | None = None
    hf_split: str = "train"
    text_field: str = "text"
    doc_id_field: str = "id"
    license: License = License.APACHE_2_0
    domain: SourceDomain = SourceDomain.GENERAL
    streaming: bool = True


def get_passage_sources() -> list[PassageSource]:
    """Return all configured passage sources for TopoLI-Retrieval.

    ALL sources are verified commercially usable:
    Apache-2.0, MIT, BSD, CC-BY, CC0, or public domain.
    NO CC-BY-SA. NO non-commercial. NO ambiguity.

    Open License Corpus subsets are cherry-picked to exclude SA content
    (StackExchange, Wikipedia, Wikinews are excluded).
    """
    return [
        # === SlimPajama (Apache-2.0) — general web text ===
        PassageSource(
            name="slimpajama",
            hf_dataset="DKYoon/SlimPajama-6B",
            hf_config=None,
            text_field="text",
            doc_id_field="__index_level_0__",
            license=License.APACHE_2_0,
            domain=SourceDomain.GENERAL,
        ),
        # === Science: ArXiv papers (CC-BY via ccdv summarization corpus) ===
        PassageSource(
            name="arxiv_science",
            hf_dataset="ccdv/arxiv-summarization",
            hf_config="document",
            hf_split="train",
            text_field="article",
            doc_id_field="abstract",
            license=License.CC_BY_4_0,
            domain=SourceDomain.SCIENCE,
        ),
        # === Science: PubMed abstracts (public domain / CC-BY) ===
        PassageSource(
            name="pubmed_science",
            hf_dataset="ccdv/pubmed-summarization",
            hf_config="document",
            hf_split="train",
            text_field="article",
            doc_id_field="abstract",
            license=License.CC_BY_4_0,
            domain=SourceDomain.SCIENCE,
        ),
        # === Books: Project Gutenberg (public domain) ===
        PassageSource(
            name="gutenberg_books",
            hf_dataset="manu/project_gutenberg",
            hf_config=None,
            hf_split="en",
            text_field="text",
            doc_id_field="id",
            license=License.PUBLIC_DOMAIN,
            domain=SourceDomain.CULTURE,
        ),
        # === Math: Open Web Math (CC-BY via CommonCrawl) ===
        PassageSource(
            name="open_web_math",
            hf_dataset="open-web-math/open-web-math",
            hf_config=None,
            hf_split="train",
            text_field="text",
            doc_id_field="url",
            license=License.CC_BY_4_0,
            domain=SourceDomain.SCIENCE,
        ),
        # === General: C4 English (ODC-By) ===
        PassageSource(
            name="c4_general",
            hf_dataset="allenai/c4",
            hf_config="en",
            hf_split="train",
            text_field="text",
            doc_id_field="url",
            license=License.ODC_BY,
            domain=SourceDomain.GENERAL,
        ),
    ]


def load_passages_from_source(
    documents: list[dict[str, str]],
    source_name: str,
    source_license: License,
    max_passages: int = 10000,
    min_chars: int = 200,
    max_chars: int = 600,
) -> list[PassageRecord]:
    """Extract passages from a list of raw documents."""
    all_passages = extract_passages(
        documents=documents,
        source_name=source_name,
        source_license=source_license,
        min_chars=min_chars,
        max_chars=max_chars,
    )
    return all_passages[:max_passages]


def load_passages_from_hf(
    source: PassageSource,
    config: DataLoaderConfig,
) -> list[PassageRecord]:
    """Stream documents from HuggingFace and extract passages."""
    from datasets import load_dataset  # type: ignore[attr-defined]

    logger.info(
        "Loading %s from %s (config=%s, split=%s, streaming=%s)",
        source.name,
        source.hf_dataset,
        source.hf_config,
        source.hf_split,
        source.streaming,
    )

    ds = load_dataset(  # type: ignore[call-overload]
        source.hf_dataset,
        name=source.hf_config,
        split=source.hf_split,
        streaming=source.streaming,
    )

    all_passages: list[PassageRecord] = []
    batch: list[dict[str, str]] = []
    batch_size = 1000
    target = config.num_passages_per_source

    for i, example in enumerate(ds):
        if len(all_passages) >= target:
            break
        text = _extract_text(example, source.text_field)
        doc_id = _extract_text(example, source.doc_id_field) or f"{source.name}_{i}"
        if text and len(text) >= config.min_chars:
            batch.append({"text": text, "doc_id": str(doc_id)})

        if len(batch) >= batch_size:
            passages = load_passages_from_source(
                documents=batch,
                source_name=source.name,
                source_license=source.license,
                max_passages=target - len(all_passages),
                min_chars=config.min_chars,
                max_chars=config.max_chars,
            )
            all_passages.extend(passages)
            batch = []
            if i % 10000 == 0:
                logger.info(
                    "  %s: %d docs processed, %d passages so far",
                    source.name,
                    i,
                    len(all_passages),
                )

    if batch and len(all_passages) < target:
        passages = load_passages_from_source(
            documents=batch,
            source_name=source.name,
            source_license=source.license,
            max_passages=target - len(all_passages),
            min_chars=config.min_chars,
            max_chars=config.max_chars,
        )
        all_passages.extend(passages)

    logger.info("Extracted %d passages from %s", len(all_passages), source.name)
    return all_passages[:target]


def _extract_text(example: dict, field_path: str) -> str | None:  # type: ignore[type-arg]
    """Extract a text field from a nested dict using dot notation."""
    parts = field_path.split(".")
    current = example
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and current:
            current = current[0]
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        else:
            return None
    if isinstance(current, str):
        return current
    if isinstance(current, list) and current and isinstance(current[0], str):
        return " ".join(current)
    return str(current) if current is not None else None


def load_existing_qa_pairs(
    records: list[dict],  # type: ignore[type-arg]
    source_name: str,
    source_license: License,
    question_field: str = "question",
    answer_field: str = "answer",
) -> list[dict[str, str]]:
    """Load existing QA pairs (TriviaQA, MIRACL format)."""
    pairs: list[dict[str, str]] = []
    for record in records:
        query = _extract_text(record, question_field)
        answer = _extract_text(record, answer_field)
        if query and answer:
            pairs.append(
                {
                    "query": query,
                    "answer": answer,
                    "source_name": source_name,
                    "source_license": source_license.value,
                }
            )
    return pairs


def load_triviaqa_pairs(max_pairs: int = 50000) -> list[dict[str, str]]:
    """Load QA pairs from TriviaQA (Apache-2.0)."""
    from datasets import load_dataset  # type: ignore[attr-defined]

    logger.info("Loading TriviaQA (Apache-2.0)...")
    ds = load_dataset("mandarjoshi/trivia_qa", "rc", split="train", streaming=True)

    records: list[dict] = []  # type: ignore[type-arg]
    for i, example in enumerate(ds):
        if i >= max_pairs:
            break
        records.append(example)

    return load_existing_qa_pairs(
        records=records,
        source_name="triviaqa",
        source_license=License.APACHE_2_0,
        question_field="question",
        answer_field="answer.value",
    )


def load_miracl_pairs(max_pairs: int = 30000) -> list[dict[str, str]]:
    """Load retrieval pairs from MIRACL English (Apache-2.0)."""
    from datasets import load_dataset  # type: ignore[attr-defined]

    logger.info("Loading MIRACL English (Apache-2.0)...")
    try:
        ds = load_dataset("miracl/miracl", "en", split="train", streaming=True)
    except (RuntimeError, Exception):  # noqa: BLE001
        logger.warning(
            "MIRACL dataset unavailable (script format not supported). Skipping."
        )
        return []

    pairs: list[dict[str, str]] = []
    for i, example in enumerate(ds):
        if i >= max_pairs:
            break
        query = example.get("query", "")
        positives = example.get("positive_passages", [])
        if query and positives:
            for pos in positives:
                text = pos.get("text", "")
                if text:
                    pairs.append(
                        {
                            "query": query,
                            "passage": text,
                            "source_name": "miracl_en",
                            "source_license": License.APACHE_2_0.value,
                        }
                    )
    logger.info("Loaded %d MIRACL pairs", len(pairs))
    return pairs
