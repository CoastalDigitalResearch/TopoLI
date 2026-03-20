"""Tests for retrieval evaluation metrics."""

from __future__ import annotations

import numpy as np

from topoli.eval import evaluate_retrieval


class TestMRR:
    """Mean Reciprocal Rank @ k."""

    def test_perfect_ranking_gives_1(self) -> None:
        results = [(0, 10.0), (1, 9.0), (2, 8.0)]
        qrels = {0}
        metrics = evaluate_retrieval(results, qrels)
        np.testing.assert_almost_equal(metrics["mrr@10"], 1.0)

    def test_relevant_at_rank_2(self) -> None:
        results = [(5, 10.0), (0, 9.0), (2, 8.0)]
        qrels = {0}
        metrics = evaluate_retrieval(results, qrels)
        np.testing.assert_almost_equal(metrics["mrr@10"], 0.5)

    def test_no_relevant_gives_0(self) -> None:
        results = [(5, 10.0), (6, 9.0)]
        qrels = {0}
        metrics = evaluate_retrieval(results, qrels)
        np.testing.assert_almost_equal(metrics["mrr@10"], 0.0)


class TestNDCG:
    """Normalized Discounted Cumulative Gain @ 10."""

    def test_perfect_ranking(self) -> None:
        results = [(0, 10.0), (1, 9.0), (2, 8.0)]
        qrels = {0, 1, 2}
        metrics = evaluate_retrieval(results, qrels)
        np.testing.assert_almost_equal(metrics["ndcg@10"], 1.0)

    def test_inverted_ranking(self) -> None:
        results = [(3, 10.0), (4, 9.0), (0, 8.0), (1, 7.0), (2, 6.0)]
        qrels = {0, 1, 2}
        metrics = evaluate_retrieval(results, qrels)
        assert metrics["ndcg@10"] < 1.0
        assert metrics["ndcg@10"] > 0.0

    def test_no_relevant_gives_0(self) -> None:
        results = [(5, 10.0), (6, 9.0)]
        qrels = {0}
        metrics = evaluate_retrieval(results, qrels)
        np.testing.assert_almost_equal(metrics["ndcg@10"], 0.0)


class TestRecall:
    """Recall @ k."""

    def test_perfect_recall_at_100(self) -> None:
        results = [(i, float(100 - i)) for i in range(100)]
        qrels = {0, 1, 2}
        metrics = evaluate_retrieval(results, qrels)
        np.testing.assert_almost_equal(metrics["recall@100"], 1.0)

    def test_partial_recall(self) -> None:
        results = [(0, 10.0), (5, 9.0)]
        qrels = {0, 1, 2}
        metrics = evaluate_retrieval(results, qrels)
        np.testing.assert_almost_equal(metrics["recall@100"], 1.0 / 3.0)

    def test_recall_at_1000_includes_all(self) -> None:
        results = [(i, float(1000 - i)) for i in range(500)]
        qrels = {100, 200, 300}
        metrics = evaluate_retrieval(results, qrels)
        np.testing.assert_almost_equal(metrics["recall@1000"], 1.0)


class TestEdgeCases:
    """Edge cases for evaluation."""

    def test_empty_results(self) -> None:
        metrics = evaluate_retrieval([], {0, 1})
        assert metrics["mrr@10"] == 0.0
        assert metrics["ndcg@10"] == 0.0
        assert metrics["recall@100"] == 0.0

    def test_empty_qrels(self) -> None:
        results = [(0, 10.0)]
        metrics = evaluate_retrieval(results, set())
        assert metrics["mrr@10"] == 0.0
        assert metrics["ndcg@10"] == 0.0

    def test_all_metrics_present(self) -> None:
        results = [(0, 10.0)]
        qrels = {0}
        metrics = evaluate_retrieval(results, qrels)
        assert "mrr@10" in metrics
        assert "ndcg@10" in metrics
        assert "recall@100" in metrics
        assert "recall@1000" in metrics
