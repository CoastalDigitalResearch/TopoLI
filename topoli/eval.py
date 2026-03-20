"""Retrieval evaluation metrics."""

from __future__ import annotations

import math


def evaluate_retrieval(
    results: list[tuple[int, float]],
    qrels: set[int],
) -> dict[str, float]:
    """Compute standard retrieval metrics.

    Args:
        results: Ranked list of (doc_id, score), descending by score.
        qrels: Set of relevant document IDs.

    Returns:
        Dictionary with mrr@10, ndcg@10, recall@100, recall@1000.
    """
    return {
        "mrr@10": _mrr(results, qrels, k=10),
        "ndcg@10": _ndcg(results, qrels, k=10),
        "recall@100": _recall(results, qrels, k=100),
        "recall@1000": _recall(results, qrels, k=1000),
    }


def _mrr(results: list[tuple[int, float]], qrels: set[int], k: int) -> float:
    """Mean Reciprocal Rank @ k."""
    for i, (doc_id, _) in enumerate(results[:k]):
        if doc_id in qrels:
            return 1.0 / (i + 1)
    return 0.0


def _dcg(relevances: list[float], k: int) -> float:
    """Discounted Cumulative Gain."""
    total = 0.0
    for i, rel in enumerate(relevances[:k]):
        total += rel / math.log2(i + 2)
    return total


def _ndcg(results: list[tuple[int, float]], qrels: set[int], k: int) -> float:
    """Normalized DCG @ k."""
    if not qrels:
        return 0.0

    actual = [1.0 if doc_id in qrels else 0.0 for doc_id, _ in results[:k]]
    ideal = sorted(actual, reverse=True)

    idcg = _dcg(ideal, k)
    if idcg == 0.0:
        return 0.0

    return _dcg(actual, k) / idcg


def _recall(results: list[tuple[int, float]], qrels: set[int], k: int) -> float:
    """Recall @ k."""
    if not qrels:
        return 0.0

    retrieved_relevant = sum(1 for doc_id, _ in results[:k] if doc_id in qrels)
    return retrieved_relevant / len(qrels)
