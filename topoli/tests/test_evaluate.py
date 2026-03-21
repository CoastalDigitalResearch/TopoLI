"""Tests for evaluation pipeline: encoding, indexing, retrieval, metrics."""

from __future__ import annotations

import numpy as np
import torch

from topoli.evaluate.retrieval_eval import (
    BruteForceIndex,
    encode_corpus,
    evaluate_retrieval_metrics,
    maxsim_rerank,
)


class TestBruteForceIndex:
    """Brute-force index for exact MaxSim retrieval."""

    def test_add_and_search(self) -> None:
        index = BruteForceIndex(dim=8)
        doc_embs = [np.random.randn(4, 8).astype(np.float32) for _ in range(10)]
        for i, emb in enumerate(doc_embs):
            index.add(i, emb)

        query = np.random.randn(3, 8).astype(np.float32)
        results = index.search(query, top_k=5)
        assert len(results) == 5
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

    def test_search_returns_sorted_by_score(self) -> None:
        index = BruteForceIndex(dim=4)
        index.add(0, np.array([[1, 0, 0, 0]], dtype=np.float32))
        index.add(1, np.array([[0, 1, 0, 0]], dtype=np.float32))
        index.add(2, np.array([[1, 1, 0, 0]], dtype=np.float32))

        query = np.array([[1, 0, 0, 0]], dtype=np.float32)
        results = index.search(query, top_k=3)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_limits_results(self) -> None:
        index = BruteForceIndex(dim=4)
        for i in range(20):
            index.add(i, np.random.randn(2, 4).astype(np.float32))
        query = np.random.randn(2, 4).astype(np.float32)
        results = index.search(query, top_k=5)
        assert len(results) == 5

    def test_empty_index(self) -> None:
        index = BruteForceIndex(dim=4)
        query = np.random.randn(2, 4).astype(np.float32)
        results = index.search(query, top_k=5)
        assert len(results) == 0


class TestMaxSimRerank:
    """MaxSim reranking of candidate documents."""

    def test_returns_sorted_results(self) -> None:
        query = np.random.randn(4, 8).astype(np.float32)
        docs = [np.random.randn(6, 8).astype(np.float32) for _ in range(5)]
        results = maxsim_rerank(query, docs, doc_ids=[0, 1, 2, 3, 4])
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)


class TestEncodeCorpus:
    """Corpus encoding produces correct shapes."""

    def test_encode_returns_list_of_arrays(self) -> None:
        model = _DummyModel(hidden=64, colbert_dim=8)
        texts = ["hello world", "foo bar baz"]

        def tokenize(text: str) -> torch.Tensor:
            return torch.randint(0, 100, (1, 8))

        embeddings = encode_corpus(model, texts, tokenize)
        assert len(embeddings) == 2
        assert all(isinstance(e, np.ndarray) for e in embeddings)
        assert all(e.shape[1] == 8 for e in embeddings)


class TestEvaluateMetrics:
    """End-to-end metric computation."""

    def test_perfect_retrieval(self) -> None:
        results = [(0, 1.0), (1, 0.9), (2, 0.8)]
        qrels = {0}
        metrics = evaluate_retrieval_metrics(results, qrels)
        assert metrics["mrr@10"] == 1.0
        assert metrics["recall@100"] == 1.0

    def test_no_relevant_docs(self) -> None:
        results = [(0, 1.0), (1, 0.9)]
        qrels: set[int] = set()
        metrics = evaluate_retrieval_metrics(results, qrels)
        assert metrics["mrr@10"] == 0.0

    def test_relevant_at_rank_3(self) -> None:
        results = [(0, 1.0), (1, 0.9), (2, 0.8)]
        qrels = {2}
        metrics = evaluate_retrieval_metrics(results, qrels)
        assert abs(metrics["mrr@10"] - 1.0 / 3) < 1e-6


class _DummyModel:
    """Dummy model for testing encode_corpus."""

    def __init__(self, hidden: int, colbert_dim: int) -> None:
        self.colbert_dim = colbert_dim

    def encode(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, seq = input_ids.shape
        embs = torch.randn(batch, seq, self.colbert_dim)
        scores = torch.rand(batch, seq, 1)
        return embs, scores
