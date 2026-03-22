"""Tests for real data loading from HuggingFace sources."""

from __future__ import annotations

from topoli.data.loader import (
    DataLoaderConfig,
    PassageSource,
    load_existing_qa_pairs,
    load_passages_from_source,
)
from topoli.data.source_config import License, SourceDomain


class TestDataLoaderConfig:
    """Loader config controls scale and checkpointing."""

    def test_defaults(self) -> None:
        cfg = DataLoaderConfig()
        assert cfg.num_passages_per_source == 10000
        assert cfg.seed == 42

    def test_small_scale_for_testing(self) -> None:
        cfg = DataLoaderConfig(num_passages_per_source=100)
        assert cfg.num_passages_per_source == 100


class TestPassageSource:
    """PassageSource maps source configs to HuggingFace loading logic."""

    def test_triviaqa_source(self) -> None:
        src = PassageSource(
            name="triviaqa",
            hf_dataset="mandarjoshi/trivia_qa",
            hf_config="rc",
            hf_split="train",
            text_field="search_results.search_context",
            doc_id_field="question_id",
            license=License.APACHE_2_0,
            domain=SourceDomain.QA,
        )
        assert src.name == "triviaqa"
        assert src.license == License.APACHE_2_0

    def test_miracl_source(self) -> None:
        src = PassageSource(
            name="miracl_en",
            hf_dataset="miracl/miracl",
            hf_config="en",
            hf_split="train",
            text_field="positive_passages.text",
            doc_id_field="query_id",
            license=License.APACHE_2_0,
            domain=SourceDomain.RETRIEVAL,
        )
        assert src.name == "miracl_en"


class TestLoadPassagesFromSource:
    """Loading passages from a source produces PassageRecords."""

    def test_load_from_simple_texts(self) -> None:
        docs = [
            {"text": "word " * 80, "doc_id": "d1"},
            {"text": "word " * 80, "doc_id": "d2"},
        ]
        passages = load_passages_from_source(
            documents=docs,
            source_name="test",
            source_license=License.APACHE_2_0,
            max_passages=10,
            min_chars=100,
            max_chars=600,
        )
        assert len(passages) > 0
        assert all(p.source_name == "test" for p in passages)
        assert all(p.source_license == License.APACHE_2_0 for p in passages)

    def test_respects_max_passages(self) -> None:
        docs = [{"text": "word " * 80, "doc_id": f"d{i}"} for i in range(100)]
        passages = load_passages_from_source(
            documents=docs,
            source_name="test",
            source_license=License.CC0,
            max_passages=5,
        )
        assert len(passages) <= 5

    def test_empty_docs(self) -> None:
        passages = load_passages_from_source(
            documents=[],
            source_name="test",
            source_license=License.MIT,
            max_passages=10,
        )
        assert len(passages) == 0


class TestLoadExistingQAPairs:
    """Loading existing QA pairs from TriviaQA/MIRACL format."""

    def test_triviaqa_format(self) -> None:
        records = [
            {
                "question": "What is the capital of France?",
                "answer": {"value": "Paris"},
                "question_id": "q1",
            },
            {
                "question": "Who wrote Hamlet?",
                "answer": {"value": "Shakespeare"},
                "question_id": "q2",
            },
        ]
        pairs = load_existing_qa_pairs(
            records=records,
            source_name="triviaqa",
            source_license=License.APACHE_2_0,
            question_field="question",
            answer_field="answer.value",
        )
        assert len(pairs) == 2
        assert pairs[0]["query"] == "What is the capital of France?"
        assert pairs[0]["source_license"] == License.APACHE_2_0.value
