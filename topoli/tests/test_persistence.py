"""Tests for TDA persistence computation."""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from topoli.tda.persistence import compute_persistence_diagram


def _circle_point_cloud(n: int = 100) -> NDArray[np.float64]:
    """Generate points on a unit circle in 2D."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _two_clusters(
    n_per_cluster: int = 50, separation: float = 10.0
) -> NDArray[np.float64]:
    """Generate two well-separated clusters."""
    rng = np.random.default_rng(42)
    c1 = rng.normal(loc=0.0, scale=0.1, size=(n_per_cluster, 2))
    c2 = rng.normal(loc=separation, scale=0.1, size=(n_per_cluster, 2))
    return np.vstack([c1, c2])


class TestCirclePointCloud:
    """A circle should have exactly 1 significant H1 feature."""

    def test_circle_has_one_h1_feature(self) -> None:
        pts = _circle_point_cloud(100)
        result = compute_persistence_diagram(pts, max_dim=1)
        h1 = result["diagrams"][1]
        persistent = h1[h1[:, 1] - h1[:, 0] > 0.5]
        assert len(persistent) == 1

    def test_circle_h1_has_high_persistence(self) -> None:
        pts = _circle_point_cloud(100)
        result = compute_persistence_diagram(pts, max_dim=1)
        h1 = result["diagrams"][1]
        max_persistence = np.max(h1[:, 1] - h1[:, 0])
        assert max_persistence > 0.5


class TestTwoClusters:
    """Two separated clusters should show a significant H0 gap."""

    def test_two_clusters_have_persistent_h0_gap(self) -> None:
        pts = _two_clusters()
        result = compute_persistence_diagram(pts, max_dim=0)
        h0 = result["diagrams"][0]
        finite = h0[np.isfinite(h0[:, 1])]
        max_persistence = np.max(finite[:, 1] - finite[:, 0])
        assert max_persistence > 0.05


class TestPerformance:
    """Persistence on moderately sized arrays completes quickly."""

    def test_180x128_under_2_seconds(self) -> None:
        rng = np.random.default_rng(42)
        pts = rng.standard_normal((180, 128))
        start = time.monotonic()
        compute_persistence_diagram(pts, max_dim=1)
        elapsed = time.monotonic() - start
        assert elapsed < 2.0


class TestSubsampling:
    """Subsampling should preserve topological features."""

    def test_subsampled_circle_preserves_h1(self) -> None:
        pts = _circle_point_cloud(200)
        full = compute_persistence_diagram(pts, max_dim=1)
        sub = compute_persistence_diagram(pts, max_dim=1, n_subsample=50)

        full_max = np.max(full["diagrams"][1][:, 1] - full["diagrams"][1][:, 0])
        sub_max = np.max(sub["diagrams"][1][:, 1] - sub["diagrams"][1][:, 0])
        assert abs(full_max - sub_max) < 0.5

    def test_subsample_indices_returned(self) -> None:
        pts = _circle_point_cloud(200)
        result = compute_persistence_diagram(pts, max_dim=1, n_subsample=50)
        assert result["subsample_indices"] is not None
        assert len(result["subsample_indices"]) == 50


class TestReturnStructure:
    """Verify the result dictionary has expected keys."""

    def test_result_has_required_keys(self) -> None:
        pts = _circle_point_cloud(50)
        result = compute_persistence_diagram(pts, max_dim=1)
        assert "diagrams" in result
        assert "cocycles" in result
        assert "distance_matrix" in result
        assert "subsample_indices" in result
