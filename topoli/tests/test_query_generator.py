"""Tests for doc2query generation from passages."""

from __future__ import annotations

from topoli.data.query_generator import (
    DOC2QUERY_PROMPT,
    QueryGeneratorConfig,
    build_prompt,
    parse_query_response,
)
from topoli.data.source_config import License, PassageRecord


def _sample_passage() -> PassageRecord:
    return PassageRecord(
        passage_id="test_001",
        text="The Federal Reserve raised interest rates by 25 basis points.",
        source_name="kl3m_government",
        source_license=License.CC_BY_4_0,
        source_doc_id="fed_001:chunk_0",
        char_count=62,
    )


class TestQueryGeneratorConfig:
    """Config is frozen and validated."""

    def test_default_model(self) -> None:
        cfg = QueryGeneratorConfig()
        assert cfg.model_name == "Qwen/Qwen3-8B"

    def test_default_license(self) -> None:
        cfg = QueryGeneratorConfig()
        assert cfg.model_license == License.APACHE_2_0

    def test_frozen(self) -> None:
        cfg = QueryGeneratorConfig()
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            cfg.model_name = "other"  # type: ignore[misc]

    def test_max_query_tokens(self) -> None:
        cfg = QueryGeneratorConfig()
        assert cfg.max_query_tokens == 30

    def test_min_query_tokens(self) -> None:
        cfg = QueryGeneratorConfig()
        assert cfg.min_query_tokens == 5


class TestBuildPrompt:
    """Prompt construction for doc2query."""

    def test_contains_passage_text(self) -> None:
        passage = _sample_passage()
        prompt = build_prompt(passage.text)
        assert passage.text in prompt

    def test_contains_instruction(self) -> None:
        prompt = build_prompt("Some text here.")
        assert "search" in prompt.lower() or "query" in prompt.lower()

    def test_uses_template(self) -> None:
        prompt = build_prompt("test text")
        assert DOC2QUERY_PROMPT.split("{")[0].strip() in prompt


class TestParseQueryResponse:
    """Parsing LLM output into clean queries."""

    def test_clean_simple_query(self) -> None:
        result = parse_query_response("What is the federal interest rate?")
        assert result == "What is the federal interest rate?"

    def test_strips_whitespace(self) -> None:
        result = parse_query_response("  What is inflation?  \n", min_tokens=2)
        assert result == "What is inflation?"

    def test_strips_query_prefix(self) -> None:
        result = parse_query_response("Query: What is inflation?", min_tokens=2)
        assert result == "What is inflation?"

    def test_returns_none_for_empty(self) -> None:
        result = parse_query_response("")
        assert result is None

    def test_returns_none_for_too_long(self) -> None:
        long_query = " ".join(["word"] * 50)
        result = parse_query_response(long_query, max_tokens=30)
        assert result is None

    def test_returns_none_for_too_short(self) -> None:
        result = parse_query_response("hi", min_tokens=5)
        assert result is None

    def test_takes_first_line_only(self) -> None:
        result = parse_query_response("First query\nSecond query\nThird", min_tokens=2)
        assert result == "First query"
