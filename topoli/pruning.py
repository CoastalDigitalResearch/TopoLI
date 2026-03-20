"""Token pruning strategies for TopoLI."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from topoli.config import (
    BaselinePruneConfig,
    HybridPruneConfig,
    TopoPruneConfig,
)


def prune_tokens(
    embeddings: NDArray[np.float64],
    scores: NDArray[np.float64],
    config: TopoPruneConfig | HybridPruneConfig | BaselinePruneConfig,
    *,
    idf_scores: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    """Prune token embeddings based on the given pruning config.

    Args:
        embeddings: (n_tokens, dim) array of token embeddings.
        scores: (n_tokens,) array of TDA importance scores.
        config: Pruning configuration determining strategy and ratio.
        idf_scores: (n_tokens,) IDF scores, required for HybridPruneConfig.

    Returns:
        Tuple of (pruned_embeddings, kept_indices) where indices are sorted.
    """
    if isinstance(config, BaselinePruneConfig):
        return _prune_baseline(embeddings, scores, config)
    if isinstance(config, HybridPruneConfig):
        if idf_scores is None:
            msg = "idf_scores required for HybridPruneConfig"
            raise ValueError(msg)
        return _prune_hybrid(embeddings, scores, config, idf_scores)
    return _prune_topo(embeddings, scores, config)


def _keep_count(n_tokens: int, pruning_ratio: float) -> int:
    keep = max(1, int(n_tokens * (1.0 - pruning_ratio)))
    return min(keep, n_tokens)


def _prune_topo(
    embeddings: NDArray[np.float64],
    scores: NDArray[np.float64],
    config: TopoPruneConfig,
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    n_tokens = embeddings.shape[0]
    keep = _keep_count(n_tokens, config.pruning_ratio)
    top_indices = np.argsort(scores)[-keep:]
    kept = np.sort(top_indices).astype(np.intp)
    return embeddings[kept], kept


def _prune_hybrid(
    embeddings: NDArray[np.float64],
    tda_scores: NDArray[np.float64],
    config: HybridPruneConfig,
    idf_scores: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    n_tokens = embeddings.shape[0]
    keep = _keep_count(n_tokens, config.pruning_ratio)

    tda_norm = tda_scores / max(tda_scores.max(), 1e-9)
    idf_norm = idf_scores / max(idf_scores.max(), 1e-9)
    combined = config.topo_weight * tda_norm + config.idf_weight * idf_norm

    top_indices = np.argsort(combined)[-keep:]
    kept = np.sort(top_indices).astype(np.intp)
    return embeddings[kept], kept


def _prune_baseline(
    embeddings: NDArray[np.float64],
    _scores: NDArray[np.float64],
    config: BaselinePruneConfig,
) -> tuple[NDArray[np.float64], NDArray[np.intp]]:
    n_tokens = embeddings.shape[0]

    if config.method == "none":
        return embeddings, np.arange(n_tokens, dtype=np.intp)

    keep = _keep_count(n_tokens, config.pruning_ratio)

    if config.method == "top_k":
        norms = np.linalg.norm(embeddings, axis=1)
        top_indices = np.argsort(norms)[-keep:]
    else:
        rng = np.random.default_rng(hash(embeddings.tobytes()) % (2**32))
        top_indices = rng.choice(n_tokens, size=keep, replace=False)

    kept = np.sort(top_indices).astype(np.intp)
    return embeddings[kept], kept
