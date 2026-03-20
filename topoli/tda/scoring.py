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
    by the feature's persistence. Fully vectorized for performance.

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

    upper_tri = np.triu_indices(n_tokens, k=1)
    edge_dists = distance_matrix[upper_tri]

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
            near_death = np.abs(edge_dists - death) < epsilon
            edge_scores = near_death.astype(np.float64) * persistence

            row_idx = upper_tri[0]
            col_idx = upper_tri[1]
            np.add.at(scores, row_idx, edge_scores)
            np.add.at(scores, col_idx, edge_scores)

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
    Weights by persistence^2 / cocycle_size to reward tokens in small,
    highly persistent cycles (topologically distinctive).

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
                weight = persistence * persistence / len(valid)
                np.add.at(scores, valid.astype(np.intp), weight)

    return _normalize_scores(scores)


def score_persistence_weighted(
    embeddings: NDArray[np.float64],
    diagrams: list[NDArray[np.float64]],
    homology_dims: tuple[int, ...],
    distance_matrix: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Score tokens by persistence-weighted proximity to critical edges.

    For each persistent feature, scores each token by how many of its edges
    are near the death threshold — the critical scale where the topological
    feature is destroyed. Bridge/boundary tokens have edges at unusual
    intermediate distances near the death scale. Weighted by persistence.

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
            bandwidth = 0.2 * persistence if persistence > 0 else 0.01
            proximity = np.exp(-((distance_matrix - death) ** 2) / (2 * bandwidth**2))
            token_scores: NDArray[np.float64] = proximity.sum(axis=1)
            scores += persistence * token_scores

    return _normalize_scores(scores)
