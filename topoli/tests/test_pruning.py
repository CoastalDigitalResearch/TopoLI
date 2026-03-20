"""Tests for token pruning strategies."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from topoli.config import BaselinePruneConfig, HybridPruneConfig, TopoPruneConfig
from topoli.pruning import prune_tokens


def _sample_embeddings(n: int = 20, dim: int = 16) -> NDArray[np.float64]:
    rng = np.random.default_rng(42)
    return rng.standard_normal((n, dim))


def _sample_scores(n: int = 20) -> NDArray[np.float64]:
    rng = np.random.default_rng(42)
    return np.abs(rng.standard_normal(n))


class TestTopoPrune:
    """Topological pruning keeps highest-scoring tokens."""

    def test_correct_number_retained(self) -> None:
        emb = _sample_embeddings(20)
        scores = _sample_scores(20)
        config = TopoPruneConfig(
            pruning_ratio=0.5,
            homology_dims=(0, 1),
            scoring="birth_death_gap",
        )
        pruned, indices = prune_tokens(emb, scores, config)
        assert pruned.shape[0] == 10
        assert len(indices) == 10

    def test_highest_scoring_tokens_kept(self) -> None:
        emb = _sample_embeddings(20)
        scores = _sample_scores(20)
        config = TopoPruneConfig(
            pruning_ratio=0.5,
            homology_dims=(0, 1),
            scoring="birth_death_gap",
        )
        _, indices = prune_tokens(emb, scores, config)
        top_10 = set(np.argsort(scores)[-10:].tolist())
        assert set(indices.tolist()) == top_10

    def test_pruned_embeddings_match_original(self) -> None:
        emb = _sample_embeddings(20)
        scores = _sample_scores(20)
        config = TopoPruneConfig(
            pruning_ratio=0.3,
            homology_dims=(0, 1),
            scoring="birth_death_gap",
        )
        pruned, indices = prune_tokens(emb, scores, config)
        np.testing.assert_array_equal(pruned, emb[indices])

    def test_zero_ratio_keeps_all(self) -> None:
        emb = _sample_embeddings(10)
        scores = _sample_scores(10)
        config = TopoPruneConfig(
            pruning_ratio=0.0,
            homology_dims=(0, 1),
            scoring="birth_death_gap",
        )
        pruned, _indices = prune_tokens(emb, scores, config)
        assert pruned.shape[0] == 10

    def test_indices_are_sorted(self) -> None:
        emb = _sample_embeddings(20)
        scores = _sample_scores(20)
        config = TopoPruneConfig(
            pruning_ratio=0.5,
            homology_dims=(0, 1),
            scoring="birth_death_gap",
        )
        _, indices = prune_tokens(emb, scores, config)
        np.testing.assert_array_equal(indices, np.sort(indices))


class TestBaselinePrune:
    """Baseline pruning strategies."""

    def test_none_method_keeps_all(self) -> None:
        emb = _sample_embeddings(15)
        scores = _sample_scores(15)
        config = BaselinePruneConfig(pruning_ratio=0.5, method="none")
        pruned, _indices = prune_tokens(emb, scores, config)
        assert pruned.shape[0] == 15

    def test_top_k_keeps_highest_norm(self) -> None:
        emb = _sample_embeddings(20)
        scores = _sample_scores(20)
        config = BaselinePruneConfig(pruning_ratio=0.5, method="top_k")
        pruned, _ = prune_tokens(emb, scores, config)
        assert pruned.shape[0] == 10

    def test_random_keeps_correct_count(self) -> None:
        emb = _sample_embeddings(20)
        scores = _sample_scores(20)
        config = BaselinePruneConfig(pruning_ratio=0.5, method="random")
        pruned, _ = prune_tokens(emb, scores, config)
        assert pruned.shape[0] == 10

    def test_random_is_deterministic_with_same_input(self) -> None:
        emb = _sample_embeddings(20)
        scores = _sample_scores(20)
        config = BaselinePruneConfig(pruning_ratio=0.5, method="random")
        _, idx1 = prune_tokens(emb, scores, config)
        _, idx2 = prune_tokens(emb, scores, config)
        np.testing.assert_array_equal(idx1, idx2)


class TestHybridPrune:
    """Hybrid pruning combines TDA scores with IDF-like weights."""

    def test_correct_count(self) -> None:
        emb = _sample_embeddings(20)
        tda_scores = _sample_scores(20)
        config = HybridPruneConfig(
            pruning_ratio=0.5,
            homology_dims=(0, 1),
            scoring="birth_death_gap",
            topo_weight=0.7,
            idf_weight=0.3,
        )
        idf_scores = np.abs(np.random.default_rng(99).standard_normal(20))
        pruned, _ = prune_tokens(emb, tda_scores, config, idf_scores=idf_scores)
        assert pruned.shape[0] == 10

    def test_hybrid_without_idf_raises(self) -> None:
        emb = _sample_embeddings(20)
        scores = _sample_scores(20)
        config = HybridPruneConfig(
            pruning_ratio=0.5,
            homology_dims=(0, 1),
            scoring="birth_death_gap",
            topo_weight=0.7,
            idf_weight=0.3,
        )
        with pytest.raises(ValueError, match="idf_scores"):
            prune_tokens(emb, scores, config)


class TestEdgeCases:
    """Edge cases for pruning."""

    def test_single_token_not_pruned(self) -> None:
        emb = np.array([[1.0, 2.0]])
        scores = np.array([1.0])
        config = TopoPruneConfig(
            pruning_ratio=0.9,
            homology_dims=(0,),
            scoring="birth_death_gap",
        )
        pruned, _indices = prune_tokens(emb, scores, config)
        assert pruned.shape[0] >= 1

    def test_all_equal_scores(self) -> None:
        emb = _sample_embeddings(10)
        scores = np.ones(10)
        config = TopoPruneConfig(
            pruning_ratio=0.5,
            homology_dims=(0,),
            scoring="birth_death_gap",
        )
        pruned, _ = prune_tokens(emb, scores, config)
        assert pruned.shape[0] == 5
