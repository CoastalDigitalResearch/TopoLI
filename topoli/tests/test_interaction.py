"""Tests for late interaction scoring (MaxSim)."""

from __future__ import annotations

import numpy as np

from topoli.interaction import maxsim, score_documents


class TestMaxSim:
    """ColBERT MaxSim: max_j(sim(q_i, d_j)) summed over query tokens."""

    def test_identical_embeddings_score_highest(self) -> None:
        q = np.array([[1.0, 0.0], [0.0, 1.0]])
        d_same = q.copy()
        d_diff = np.array([[0.5, 0.5], [0.5, 0.5]])
        assert maxsim(q, d_same) > maxsim(q, d_diff)

    def test_cosine_identical_gives_n_query_tokens(self) -> None:
        q = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        d = q.copy()
        score = maxsim(q, d, similarity="cosine")
        np.testing.assert_almost_equal(score, 3.0)

    def test_dot_product_mode(self) -> None:
        q = np.array([[2.0, 0.0]])
        d = np.array([[3.0, 0.0]])
        score = maxsim(q, d, similarity="dot")
        np.testing.assert_almost_equal(score, 6.0)

    def test_l2_mode_identical_is_zero(self) -> None:
        q = np.array([[1.0, 2.0]])
        d = q.copy()
        score = maxsim(q, d, similarity="l2")
        np.testing.assert_almost_equal(score, 0.0)

    def test_l2_mode_farther_scores_higher(self) -> None:
        q = np.array([[0.0, 0.0]])
        d_near = np.array([[0.1, 0.0]])
        d_far = np.array([[10.0, 0.0]])
        assert maxsim(q, d_near, similarity="l2") < maxsim(q, d_far, similarity="l2")

    def test_single_token_query_and_doc(self) -> None:
        q = np.array([[1.0, 0.0, 0.0]])
        d = np.array([[0.0, 1.0, 0.0]])
        score = maxsim(q, d, similarity="cosine")
        np.testing.assert_almost_equal(score, 0.0)


class TestScoreDocuments:
    """Score and rank multiple documents against a query."""

    def test_ranking_order(self) -> None:
        q = np.array([[1.0, 0.0], [0.0, 1.0]])
        docs = [
            np.array([[0.5, 0.5], [0.5, 0.5]]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
            np.array([[0.1, 0.9], [0.9, 0.1]]),
        ]
        ranked = score_documents(q, docs)
        assert ranked[0][0] == 1

    def test_returns_all_documents(self) -> None:
        q = np.array([[1.0, 0.0]])
        docs = [np.array([[1.0, 0.0]]) for _ in range(5)]
        ranked = score_documents(q, docs)
        assert len(ranked) == 5

    def test_scores_are_descending(self) -> None:
        q = np.array([[1.0, 0.0], [0.0, 1.0]])
        rng = np.random.default_rng(42)
        docs = [rng.standard_normal((3, 2)) for _ in range(10)]
        ranked = score_documents(q, docs)
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_empty_doc_list(self) -> None:
        q = np.array([[1.0, 0.0]])
        ranked = score_documents(q, [])
        assert ranked == []

    def test_different_doc_lengths(self) -> None:
        q = np.array([[1.0, 0.0]])
        docs = [
            np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
            np.array([[0.9, 0.1]]),
        ]
        ranked = score_documents(q, docs)
        assert len(ranked) == 2
