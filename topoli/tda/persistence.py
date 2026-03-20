"""Persistent homology computation for token embeddings."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from ripser import ripser
from scipy.spatial.distance import pdist, squareform


def _greedy_permutation(
    distance_matrix: NDArray[np.float64], n: int
) -> NDArray[np.intp]:
    """Select n landmarks via greedy permutation (farthest point sampling)."""
    n_points = distance_matrix.shape[0]
    if n >= n_points:
        return np.arange(n_points, dtype=np.intp)

    rng = np.random.default_rng(0)
    indices = np.empty(n, dtype=np.intp)
    indices[0] = rng.integers(n_points)

    min_dists = distance_matrix[indices[0]].copy()
    for i in range(1, n):
        indices[i] = np.argmax(min_dists)
        np.minimum(min_dists, distance_matrix[indices[i]], out=min_dists)

    return indices


def compute_persistence_diagram(
    embeddings: NDArray[np.float64],
    max_dim: int = 1,
    max_edge_length: float = math.inf,
    n_subsample: int | None = None,
) -> dict[str, Any]:
    """Compute persistence diagrams from token embeddings.

    Args:
        embeddings: Array of shape (n_tokens, embedding_dim).
        max_dim: Maximum homology dimension to compute.
        max_edge_length: Maximum edge length for Vietoris-Rips complex.
        n_subsample: If set, subsample to this many points via greedy permutation.

    Returns:
        Dictionary with keys: diagrams, cocycles, distance_matrix, subsample_indices.
    """
    condensed = pdist(embeddings, metric="cosine")
    dist_matrix: NDArray[np.float64] = squareform(condensed)

    subsample_indices: NDArray[np.intp] | None = None
    working_matrix = dist_matrix

    if n_subsample is not None and n_subsample < embeddings.shape[0]:
        subsample_indices = _greedy_permutation(dist_matrix, n_subsample)
        working_matrix = dist_matrix[np.ix_(subsample_indices, subsample_indices)]

    result = ripser(
        working_matrix,
        maxdim=max_dim,
        thresh=max_edge_length,
        distance_matrix=True,
        do_cocycles=True,
    )

    return {
        "diagrams": result["dgms"],
        "cocycles": result["cocycles"],
        "distance_matrix": dist_matrix,
        "subsample_indices": subsample_indices,
    }
