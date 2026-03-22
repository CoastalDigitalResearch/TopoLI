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
    """Return all configured passage sources for TopoLI-Retrieval."""
    return [
        PassageSource(
            name="common_corpus_government",
            hf_dataset="PleIAs/Common-Corpus",
            hf_config="default",
            text_field="text",
            doc_id_field="id",
            license=License.CC0,
            domain=SourceDomain.GOVERNMENT,
        ),
        PassageSource(
            name="kl3m_government",
            hf_dataset="alea-institute/kl3m-data-govinfo-crec",
            hf_config=None,
            text_field="text",
            doc_id_field="identifier",
            license=License.CC_BY_4_0,
            domain=SourceDomain.GOVERNMENT,
        ),
        PassageSource(
            name="slimpajama",
            hf_dataset="cerebras/SlimPajama-627B",
            hf_config=None,
            text_field="text",
            doc_id_field="meta",
            license=License.APACHE_2_0,
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
        trust_remote_code=True,
    )

    documents: list[dict[str, str]] = []
    target = config.num_passages_per_source * 3

    for i, example in enumerate(ds):
        if i >= target:
            break
        text = _extract_text(example, source.text_field)
        doc_id = _extract_text(example, source.doc_id_field) or f"{source.name}_{i}"
        if text and len(text) >= config.min_chars:
            documents.append({"text": text, "doc_id": str(doc_id)})

    logger.info("Collected %d documents from %s", len(documents), source.name)

    return load_passages_from_source(
        documents=documents,
        source_name=source.name,
        source_license=source.license,
        max_passages=config.num_passages_per_source,
        min_chars=config.min_chars,
        max_chars=config.max_chars,
    )


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
    ds = load_dataset("miracl/miracl", "en", split="train", streaming=True)

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
