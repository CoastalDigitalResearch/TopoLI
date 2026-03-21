"""Dataset builder for TopoLI-Retrieval.

Orchestrates the full pipeline: extract passages, generate queries,
filter, deduplicate, mine hard negatives, and output JSONL with
full provenance tracking.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from topoli.data.source_config import PassageRecord


class DatasetManifest(BaseModel):
    """Manifest documenting dataset composition and provenance."""

    model_config = ConfigDict(frozen=True)

    dataset_name: str = "topoli-retrieval"
    version: str = "0.1.0"
    total_passages: int
    source_counts: dict[str, int]
    license_summary: dict[str, int]
    dataset_license: str = "Apache-2.0"


def build_manifest(
    passages: list[PassageRecord],
    dataset_name: str = "topoli-retrieval",
    version: str = "0.1.0",
) -> DatasetManifest:
    """Build a manifest from a list of passage records."""
    source_counts = dict(Counter(p.source_name for p in passages))
    license_counts = dict(Counter(p.source_license.value for p in passages))

    return DatasetManifest(
        dataset_name=dataset_name,
        version=version,
        total_passages=len(passages),
        source_counts=source_counts,
        license_summary=license_counts,
    )


def write_passages_jsonl(
    passages: list[PassageRecord],
    output_path: Path,
) -> None:
    """Write passages to JSONL with full provenance per record."""
    with output_path.open("w") as f:
        for passage in passages:
            record = {
                "passage_id": passage.passage_id,
                "text": passage.text,
                "source_name": passage.source_name,
                "source_license": passage.source_license.value,
                "source_doc_id": passage.source_doc_id,
                "char_count": passage.char_count,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
