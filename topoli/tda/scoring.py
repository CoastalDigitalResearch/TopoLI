"""TDA-based token scoring functions."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def _normalize_scores(scores: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize scores to [0, 1]."""
    max_val = scores.max()
    if max_val == 0.0:
        return scores
    result: NDArray[np.float64] = scores / max_val
    return result


def score_birth_death_gap(
    embeddings: NDArray[np.float64],
    diagrams: list[NDArray[np.float64]],
    homology_dims: tuple[int, ...],
    distance_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Score tokens by involvement in persistent features via birth-death gap.

    For each persistent feature, tokens incident to edges at the death scale
    (the critical edges that merge/kill the feature) score highest, weighted
    by the feature's persistence.

    Args:
        embeddings: (n_tokens, dim) array.
        diagrams: List of persistence diagrams per dimension.
        homology_dims: Which homology dimensions to consider.
        distance_matrix: (n_tokens, n_tokens) pairwise distance matrix.

    Returns:
        (n_tokens,) array of scores in [0, 1].
    """
    n_tokens = embeddings.shape[0]
    scores = np.zeros(n_tokens, dtype=np.float64)

    for dim in homology_dims:
        if dim >= len(diagrams):
            continue
        dgm = diagrams[dim]
        if len(dgm) == 0:
            continue

        finite_mask = np.isfinite(dgm[:, 1])
        finite_dgm = dgm[finite_mask]
        if len(finite_dgm) == 0:
            continue

        persistences = finite_dgm[:, 1] - finite_dgm[:, 0]
        deaths = finite_dgm[:, 1]

        for death, persistence in zip(deaths, persistences, strict=True):
            epsilon = 0.1 * persistence if persistence > 0 else 0.01
            for i in range(n_tokens):
                for j in range(i + 1, n_tokens):
                    d_ij = distance_matrix[i, j]
                    if abs(d_ij - death) < epsilon:
                        scores[i] += persistence
                        scores[j] += persistence

    return _normalize_scores(scores)


def score_representative_cycle(
    embeddings: NDArray[np.float64],
    diagrams: list[NDArray[np.float64]],
    cocycles: list[list[Any]],
    homology_dims: tuple[int, ...],
    persistence_threshold: float = 0.1,
) -> NDArray[np.float64]:
    """Score tokens by membership in representative cocycles.

    Tokens appearing in cocycles of persistent features score higher.
    Uses cocycle edge data from ripser: each cocycle entry is an array where
    each row represents a simplex. For H1, rows are [vertex_i, vertex_j, coeff].

    Args:
        embeddings: (n_tokens, dim) array.
        diagrams: List of persistence diagrams per dimension.
        cocycles: List of cocycle lists per dimension from ripser.
        homology_dims: Which homology dimensions to consider.
        persistence_threshold: Minimum persistence to consider a feature.

    Returns:
        (n_tokens,) array of scores in [0, 1].
    """
    n_tokens = embeddings.shape[0]
    scores = np.zeros(n_tokens, dtype=np.float64)

    for dim in homology_dims:
        if dim >= len(diagrams) or dim >= len(cocycles):
            continue
        dgm = diagrams[dim]
        dim_cocycles = cocycles[dim]

        if len(dgm) == 0 or len(dim_cocycles) == 0:
            continue

        for i, cocycle in enumerate(dim_cocycles):
            if i >= len(dgm):
                break
            persistence = dgm[i, 1] - dgm[i, 0]
            if not np.isfinite(persistence) or persistence < persistence_threshold:
                continue

            cocycle_arr = np.asarray(cocycle)
            if cocycle_arr.ndim < 2 or cocycle_arr.shape[0] == 0:
                continue

            vertex_indices = np.unique(cocycle_arr[:, :-1].flatten())
            valid = vertex_indices[vertex_indices < n_tokens]
            if len(valid) > 0:
                np.add.at(scores, valid.astype(np.intp), persistence)

    return _normalize_scores(scores)


def score_persistence_weighted(
    embeddings: NDArray[np.float64],
    diagrams: list[NDArray[np.float64]],
    homology_dims: tuple[int, ...],
    distance_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Score tokens by persistence-weighted proximity to topological features.

    For each persistent feature, score each token by how close its nearest-neighbor
    distance is to the birth/death scale of that feature, weighted by persistence.

    Args:
        embeddings: (n_tokens, dim) array.
        diagrams: List of persistence diagrams per dimension.
        homology_dims: Which homology dimensions to consider.
        distance_matrix: (n_tokens, n_tokens) pairwise distance matrix.

    Returns:
        (n_tokens,) array of scores in [0, 1].
    """
    n_tokens = embeddings.shape[0]
    scores = np.zeros(n_tokens, dtype=np.float64)

    for dim in homology_dims:
        if dim >= len(diagrams):
            continue
        dgm = diagrams[dim]
        if len(dgm) == 0:
            continue

        finite_mask = np.isfinite(dgm[:, 1])
        finite_dgm = dgm[finite_mask]
        if len(finite_dgm) == 0:
            continue

        persistences = finite_dgm[:, 1] - finite_dgm[:, 0]
        midpoints = (finite_dgm[:, 0] + finite_dgm[:, 1]) / 2.0

        min_dists = distance_matrix.min(axis=1)
        for mid, p in zip(midpoints, persistences, strict=True):
            proximity = 1.0 / (1.0 + np.abs(min_dists - mid))
            scores += p * proximity

    return _normalize_scores(scores)
