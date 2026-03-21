"""Tests for dataset builder orchestration."""

from __future__ import annotations

import json
from pathlib import Path

from topoli.data.dataset_builder import (
    build_manifest,
    write_passages_jsonl,
)
from topoli.data.source_config import License, PassageRecord


def _sample_passages() -> list[PassageRecord]:
    return [
        PassageRecord(
            passage_id=f"p_{i}",
            text=f"This is passage number {i} with enough text to be valid.",
            source_name="test_source",
            source_license=License.APACHE_2_0,
            source_doc_id=f"doc_{i}:chunk_0",
            char_count=50,
        )
        for i in range(5)
    ]


class TestDatasetManifest:
    """Manifest tracks dataset composition and provenance."""

    def test_build_manifest(self) -> None:
        passages = _sample_passages()
        manifest = build_manifest(
            passages=passages,
            dataset_name="topoli-retrieval-test",
            version="0.1.0",
        )
        assert manifest.dataset_name == "topoli-retrieval-test"
        assert manifest.total_passages == 5
        assert "test_source" in manifest.source_counts
        assert manifest.source_counts["test_source"] == 5

    def test_manifest_license_summary(self) -> None:
        passages = _sample_passages()
        manifest = build_manifest(passages=passages)
        assert License.APACHE_2_0.value in manifest.license_summary

    def test_manifest_is_frozen(self) -> None:
        manifest = build_manifest(passages=_sample_passages())
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            manifest.total_passages = 99  # type: ignore[misc]


class TestWritePassagesJsonl:
    """JSONL output is valid and complete."""

    def test_writes_correct_number_of_lines(self, tmp_path: Path) -> None:
        passages = _sample_passages()
        out_path = tmp_path / "passages.jsonl"
        write_passages_jsonl(passages, out_path)

        lines = out_path.read_text().strip().split("\n")
        assert len(lines) == 5

    def test_each_line_is_valid_json(self, tmp_path: Path) -> None:
        passages = _sample_passages()
        out_path = tmp_path / "passages.jsonl"
        write_passages_jsonl(passages, out_path)

        for line in out_path.read_text().strip().split("\n"):
            record = json.loads(line)
            assert "passage_id" in record
            assert "text" in record
            assert "source_name" in record
            assert "source_license" in record

    def test_preserves_provenance(self, tmp_path: Path) -> None:
        passages = _sample_passages()
        out_path = tmp_path / "passages.jsonl"
        write_passages_jsonl(passages, out_path)

        first = json.loads(out_path.read_text().strip().split("\n")[0])
        assert first["source_license"] == "Apache-2.0"
        assert first["source_name"] == "test_source"
