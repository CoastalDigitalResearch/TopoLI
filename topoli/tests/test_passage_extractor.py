"""Tests for passage extraction from permissive-license sources."""

from __future__ import annotations

from topoli.data.passage_extractor import (
    chunk_document,
    clean_text,
    extract_passages,
)
from topoli.data.source_config import License


class TestCleanText:
    """Text cleaning preserves content, removes noise."""

    def test_removes_excessive_whitespace(self) -> None:
        result = clean_text("hello   world\n\n\nfoo")
        assert result == "hello world\nfoo"

    def test_strips_leading_trailing(self) -> None:
        result = clean_text("  hello world  ")
        assert result == "hello world"

    def test_preserves_single_newlines(self) -> None:
        result = clean_text("line one\nline two")
        assert result == "line one\nline two"

    def test_removes_null_bytes(self) -> None:
        result = clean_text("hello\x00world")
        assert result == "helloworld"

    def test_empty_string(self) -> None:
        result = clean_text("")
        assert result == ""


class TestChunkDocument:
    """Document chunking produces correct passage sizes."""

    def test_short_doc_single_chunk(self) -> None:
        text = "This is a short document with a few words."
        chunks = chunk_document(text, min_chars=10, max_chars=200, overlap_chars=20)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_doc_multiple_chunks(self) -> None:
        words = ["word"] * 200
        text = " ".join(words)
        chunks = chunk_document(text, min_chars=100, max_chars=400, overlap_chars=50)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) >= 100
            assert len(chunk) <= 400

    def test_chunks_have_overlap(self) -> None:
        text = "A " * 500
        chunks = chunk_document(text, min_chars=100, max_chars=300, overlap_chars=50)
        if len(chunks) >= 2:
            end_of_first = chunks[0][-50:]
            assert end_of_first in chunks[1]

    def test_too_short_doc_returns_empty(self) -> None:
        text = "hi"
        chunks = chunk_document(text, min_chars=100, max_chars=400, overlap_chars=50)
        assert len(chunks) == 0

    def test_respects_sentence_boundaries(self) -> None:
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = chunk_document(text, min_chars=10, max_chars=40, overlap_chars=5)
        for chunk in chunks:
            assert chunk.endswith(".") or chunk.endswith(". ")


class TestExtractPassages:
    """Extract passages creates PassageRecords with provenance."""

    def test_extracts_from_documents(self) -> None:
        docs = [
            {"text": "A " * 200, "doc_id": "doc_001"},
            {"text": "B " * 200, "doc_id": "doc_002"},
        ]
        passages = extract_passages(
            documents=docs,
            source_name="test_source",
            source_license=License.APACHE_2_0,
            min_chars=100,
            max_chars=300,
        )
        assert len(passages) > 0
        for p in passages:
            assert p.source_name == "test_source"
            assert p.source_license == License.APACHE_2_0
            assert len(p.text) >= 100
            assert p.char_count == len(p.text)

    def test_passage_ids_are_unique(self) -> None:
        docs = [{"text": "word " * 100, "doc_id": f"doc_{i}"} for i in range(5)]
        passages = extract_passages(
            documents=docs,
            source_name="test",
            source_license=License.CC0,
            min_chars=50,
            max_chars=200,
        )
        ids = [p.passage_id for p in passages]
        assert len(ids) == len(set(ids))

    def test_preserves_source_doc_id(self) -> None:
        docs = [{"text": "word " * 100, "doc_id": "my_doc_42"}]
        passages = extract_passages(
            documents=docs,
            source_name="test",
            source_license=License.MIT,
            min_chars=50,
            max_chars=600,
        )
        assert all("my_doc_42" in p.source_doc_id for p in passages)
