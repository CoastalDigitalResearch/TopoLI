"""Tests for real query generation implementation."""

from __future__ import annotations

from topoli.data.query_generator import (
    build_prompt,
    parse_query_response,
)
from topoli.data.query_generator_impl import (
    QueryBatch,
    QueryGenPipeline,
    batch_passages,
)
from topoli.data.source_config import License, PassageRecord


def _passages(n: int) -> list[PassageRecord]:
    return [
        PassageRecord(
            passage_id=f"p_{i}",
            text=f"The government regulation number {i} states that all financial institutions must report quarterly earnings within 45 days of the fiscal quarter end.",
            source_name="test",
            source_license=License.APACHE_2_0,
            source_doc_id=f"doc_{i}:chunk_0",
            char_count=130,
        )
        for i in range(n)
    ]


class TestBatchPassages:
    """Batch passages for efficient GPU inference."""

    def test_batches_correct_size(self) -> None:
        passages = _passages(10)
        batches = batch_passages(passages, batch_size=3)
        assert len(batches) == 4  # 3+3+3+1
        assert len(batches[0].passages) == 3
        assert len(batches[-1].passages) == 1

    def test_batches_contain_prompts(self) -> None:
        passages = _passages(5)
        batches = batch_passages(passages, batch_size=2)
        for batch in batches:
            assert len(batch.prompts) == len(batch.passages)
            for prompt in batch.prompts:
                assert "Query:" in prompt

    def test_empty_input(self) -> None:
        batches = batch_passages([], batch_size=4)
        assert len(batches) == 0


class TestQueryGenPipeline:
    """Pipeline generates queries from passages using a mock model."""

    def test_generates_queries_from_passages(self) -> None:
        passages = _passages(4)

        def mock_generate(prompts: list[str]) -> list[str]:
            return [f"what are the quarterly earnings reporting deadlines for passage {i}" for i in range(len(prompts))]

        pipeline = QueryGenPipeline(
            generate_fn=mock_generate,
            model_name="mock-model",
            model_license=License.APACHE_2_0,
            batch_size=2,
        )
        results = pipeline.generate(passages)
        assert len(results) == 4
        assert all("quarterly" in r["query"] for r in results)
        assert all(r["passage_id"].startswith("p_") for r in results)
        assert all(r["generator_model"] == "mock-model" for r in results)
        assert all(r["generator_license"] == "Apache-2.0" for r in results)

    def test_filters_bad_queries(self) -> None:
        passages = _passages(4)

        def mock_generate(prompts: list[str]) -> list[str]:
            return ["good query about economics", "hi", "another good search query", ""]

        pipeline = QueryGenPipeline(
            generate_fn=mock_generate,
            model_name="mock",
            model_license=License.APACHE_2_0,
            batch_size=4,
            min_query_tokens=3,
        )
        results = pipeline.generate(passages)
        assert len(results) == 2  # "hi" and "" filtered out

    def test_tracks_provenance(self) -> None:
        passages = _passages(2)

        def mock_generate(prompts: list[str]) -> list[str]:
            return ["what are quarterly earnings deadlines", "financial reporting requirements"]

        pipeline = QueryGenPipeline(
            generate_fn=mock_generate,
            model_name="Qwen/Qwen3-8B",
            model_license=License.APACHE_2_0,
            batch_size=4,
        )
        results = pipeline.generate(passages)
        for r in results:
            assert "passage_id" in r
            assert "source_name" in r
            assert "source_license" in r
            assert "generator_model" in r
            assert "generator_license" in r
