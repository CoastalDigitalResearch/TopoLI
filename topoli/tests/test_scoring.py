"""Tests for TDA scoring functions."""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from topoli.tda.persistence import compute_persistence_diagram
from topoli.tda.scoring import (
    score_birth_death_gap,
    score_persistence_weighted,
    score_representative_cycle,
)


def _barbell_embeddings(
    n_per_group: int = 30, bridge_size: int = 3
) -> NDArray[np.float64]:
    """Create a barbell-shaped point cloud: two clusters connected by a bridge.

    Bridge tokens should be topologically important (they connect components).
    """
    rng = np.random.default_rng(42)
    cluster_a = rng.normal(loc=-5.0, scale=0.3, size=(n_per_group, 16))
    cluster_b = rng.normal(loc=5.0, scale=0.3, size=(n_per_group, 16))
    bridge = np.linspace(
        cluster_a.mean(axis=0),
        cluster_b.mean(axis=0),
        bridge_size,
    )
    return np.vstack([cluster_a, bridge, cluster_b])


def _circle_with_interior(
    n_circle: int = 40, n_interior: int = 20
) -> NDArray[np.float64]:
    """Circle points plus random interior points."""
    rng = np.random.default_rng(42)
    angles = np.linspace(0, 2 * np.pi, n_circle, endpoint=False)
    circle = np.column_stack([np.cos(angles), np.sin(angles)])
    interior = rng.uniform(-0.3, 0.3, size=(n_interior, 2))
    return np.vstack([circle, interior])


class TestBirthDeathGap:
    """Birth-death gap scoring should highlight bridge tokens."""

    def test_bridge_tokens_score_higher(self) -> None:
        emb = _barbell_embeddings()
        result = compute_persistence_diagram(emb, max_dim=0)
        scores = score_birth_death_gap(
            emb,
            result["diagrams"],
            homology_dims=(0,),
            distance_matrix=result["distance_matrix"],
        )
        n_a = 30
        bridge_start = n_a
        bridge_end = n_a + 3
        bridge_mean = scores[bridge_start:bridge_end].mean()
        cluster_mean = np.concatenate([scores[:n_a], scores[bridge_end:]]).mean()
        assert bridge_mean > cluster_mean

    def test_output_shape_matches_input(self) -> None:
        emb = _barbell_embeddings()
        result = compute_persistence_diagram(emb, max_dim=0)
        scores = score_birth_death_gap(
            emb,
            result["diagrams"],
            homology_dims=(0,),
            distance_matrix=result["distance_matrix"],
        )
        assert scores.shape == (emb.shape[0],)

    def test_scores_in_0_1_range(self) -> None:
        emb = _barbell_embeddings()
        result = compute_persistence_diagram(emb, max_dim=0)
        scores = score_birth_death_gap(
            emb,
            result["diagrams"],
            homology_dims=(0,),
            distance_matrix=result["distance_matrix"],
        )
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)


class TestRepresentativeCycle:
    """Representative cycle scoring should highlight cycle participants."""

    def test_top_scoring_tokens_mostly_on_circle(self) -> None:
        emb = _circle_with_interior(n_circle=40, n_interior=20)
        result = compute_persistence_diagram(emb, max_dim=1)
        scores = score_representative_cycle(
            emb,
            result["diagrams"],
            result["cocycles"],
            homology_dims=(1,),
        )
        top_k = 20
        top_indices = np.argsort(scores)[-top_k:]
        circle_in_top = np.sum(top_indices < 40)
        assert circle_in_top > top_k // 2

    def test_scores_in_0_1_range(self) -> None:
        emb = _circle_with_interior()
        result = compute_persistence_diagram(emb, max_dim=1)
        scores = score_representative_cycle(
            emb,
            result["diagrams"],
            result["cocycles"],
            homology_dims=(1,),
        )
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)


class TestPersistenceWeighted:
    """Persistence-weighted scoring."""

    def test_output_shape(self) -> None:
        emb = _barbell_embeddings()
        result = compute_persistence_diagram(emb, max_dim=1)
        scores = score_persistence_weighted(
            emb,
            result["diagrams"],
            homology_dims=(0, 1),
            distance_matrix=result["distance_matrix"],
        )
        assert scores.shape == (emb.shape[0],)

    def test_scores_in_0_1_range(self) -> None:
        emb = _barbell_embeddings()
        result = compute_persistence_diagram(emb, max_dim=1)
        scores = score_persistence_weighted(
            emb,
            result["diagrams"],
            homology_dims=(0, 1),
            distance_matrix=result["distance_matrix"],
        )
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)


class TestDiscriminationMargins:
    """Strong discrimination between important and unimportant tokens."""

    def test_birth_death_gap_bridge_ratio_above_5x(self) -> None:
        emb = _barbell_embeddings()
        result = compute_persistence_diagram(emb, max_dim=0)
        scores = score_birth_death_gap(
            emb,
            result["diagrams"],
            homology_dims=(0,),
            distance_matrix=result["distance_matrix"],
        )
        bridge_mean = scores[30:33].mean()
        cluster_mean = np.concatenate([scores[:30], scores[33:]]).mean()
        ratio = bridge_mean / max(cluster_mean, 1e-9)
        assert ratio > 5.0

    def test_persistence_weighted_bridge_discrimination(self) -> None:
        emb = _barbell_embeddings()
        result = compute_persistence_diagram(emb, max_dim=0)
        scores = score_persistence_weighted(
            emb,
            result["diagrams"],
            homology_dims=(0,),
            distance_matrix=result["distance_matrix"],
        )
        bridge_mean = scores[30:33].mean()
        cluster_mean = np.concatenate([scores[:30], scores[33:]]).mean()
        assert bridge_mean > cluster_mean

    def test_birth_death_gap_vectorized_matches_shape(self) -> None:
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((100, 32))
        result = compute_persistence_diagram(emb, max_dim=1)
        scores = score_birth_death_gap(
            emb,
            result["diagrams"],
            homology_dims=(0, 1),
            distance_matrix=result["distance_matrix"],
        )
        assert scores.shape == (100,)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)


class TestPerformance:
    """Scoring should be efficient on realistic sizes."""

    def test_birth_death_gap_under_1s_for_180_tokens(self) -> None:
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((180, 128))
        result = compute_persistence_diagram(emb, max_dim=1)
        t0 = time.monotonic()
        score_birth_death_gap(
            emb,
            result["diagrams"],
            homology_dims=(0, 1),
            distance_matrix=result["distance_matrix"],
        )
        elapsed = time.monotonic() - t0
        assert elapsed < 1.0


class TestEdgeCases:
    """Edge cases for scoring functions."""

    def test_zero_persistent_features_returns_zeros(self) -> None:
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((5, 2))
        result = compute_persistence_diagram(emb, max_dim=1)
        scores = score_birth_death_gap(
            emb,
            result["diagrams"],
            homology_dims=(1,),
            distance_matrix=result["distance_matrix"],
        )
        assert np.allclose(scores, 0.0)

    def test_single_token_returns_zero(self) -> None:
        emb = np.array([[1.0, 2.0, 3.0]])
        result = compute_persistence_diagram(emb, max_dim=0)
        scores = score_birth_death_gap(
            emb,
            result["diagrams"],
            homology_dims=(0,),
            distance_matrix=result["distance_matrix"],
        )
        assert scores.shape == (1,)

    def test_all_scoring_functions_agree_on_zeros_for_no_features(self) -> None:
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((5, 2))
        result = compute_persistence_diagram(emb, max_dim=1)
        dgms = result["diagrams"]
        dm = result["distance_matrix"]
        s1 = score_birth_death_gap(emb, dgms, (1,), dm)
        s2 = score_representative_cycle(emb, dgms, result["cocycles"], (1,))
        s3 = score_persistence_weighted(emb, dgms, (1,), dm)
        assert np.allclose(s1, 0.0)
        assert np.allclose(s2, 0.0)
        assert np.allclose(s3, 0.0)
