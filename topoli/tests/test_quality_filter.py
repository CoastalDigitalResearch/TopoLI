"""Tests for query-passage quality filtering."""

from __future__ import annotations

from topoli.data.quality_filter import (
    compute_token_overlap,
    deduplicate_queries,
    filter_pair,
)


class TestTokenOverlap:
    """Token overlap detects verbatim copying."""

    def test_identical_strings(self) -> None:
        assert compute_token_overlap("hello world", "hello world") == 1.0

    def test_no_overlap(self) -> None:
        assert compute_token_overlap("hello world", "foo bar") == 0.0

    def test_partial_overlap(self) -> None:
        overlap = compute_token_overlap("the cat sat", "the dog sat")
        assert 0.3 < overlap < 0.8

    def test_empty_strings(self) -> None:
        assert compute_token_overlap("", "") == 0.0


class TestFilterPair:
    """Filter detects bad query-passage pairs."""

    def test_valid_pair_passes(self) -> None:
        assert filter_pair(
            query="What is the federal interest rate?",
            passage="The Federal Reserve raised interest rates by 25 basis points today.",
        )

    def test_high_overlap_rejected(self) -> None:
        text = "The Federal Reserve raised interest rates"
        assert not filter_pair(query=text, passage=text, max_overlap=0.8)

    def test_short_query_rejected(self) -> None:
        assert not filter_pair(
            query="hi",
            passage="A long passage about economics and policy.",
            min_query_tokens=5,
        )

    def test_long_query_rejected(self) -> None:
        long_q = " ".join(["word"] * 40)
        assert not filter_pair(
            query=long_q,
            passage="A passage.",
            max_query_tokens=30,
        )


class TestDeduplication:
    """Query deduplication removes exact and near-duplicates."""

    def test_exact_duplicates_removed(self) -> None:
        queries = ["What is GDP?", "What is GDP?", "How does inflation work?"]
        result = deduplicate_queries(queries)
        assert len(result) == 2

    def test_unique_queries_preserved(self) -> None:
        queries = ["What is GDP?", "How does inflation work?", "Federal Reserve policy"]
        result = deduplicate_queries(queries)
        assert len(result) == 3

    def test_empty_list(self) -> None:
        assert deduplicate_queries([]) == []
