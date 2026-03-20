"""Multi-stage retrieval pipeline for TopoLI."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from topoli.config import (
    BaselinePruneConfig,
    HybridPruneConfig,
    TopoLIConfig,
    TopoPruneConfig,
)
from topoli.interaction import score_documents
from topoli.pruning import prune_tokens
from topoli.tda.persistence import compute_persistence_diagram
from topoli.tda.scoring import (
    score_birth_death_gap,
    score_persistence_weighted,
    score_representative_cycle,
)


def _compute_tda_scores(
    doc_embs: NDArray[np.float64],
    config: TopoPruneConfig | HybridPruneConfig,
) -> NDArray[np.float64]:
    """Compute TDA importance scores for document tokens."""
    max_dim = max(config.homology_dims)
    result = compute_persistence_diagram(
        doc_embs,
        max_dim=max_dim,
        n_subsample=(
            config.n_subsample if isinstance(config, TopoPruneConfig) else None
        ),
    )

    if config.scoring == "birth_death_gap":
        return score_birth_death_gap(
            doc_embs,
            result["diagrams"],
            config.homology_dims,
            result["distance_matrix"],
        )
    if config.scoring == "representative_cycle":
        return score_representative_cycle(
            doc_embs,
            result["diagrams"],
            result["cocycles"],
            config.homology_dims,
            config.persistence_threshold,
        )
    return score_persistence_weighted(
        doc_embs,
        result["diagrams"],
        config.homology_dims,
        result["distance_matrix"],
    )


def _prune_document(
    doc_embs: NDArray[np.float64],
    config: TopoPruneConfig | HybridPruneConfig | BaselinePruneConfig,
) -> NDArray[np.float64]:
    """Apply pruning to a single document's token embeddings."""
    if isinstance(config, BaselinePruneConfig):
        dummy_scores = np.zeros(doc_embs.shape[0])
        pruned, _ = prune_tokens(doc_embs, dummy_scores, config)
        return pruned

    tda_scores = _compute_tda_scores(doc_embs, config)

    if isinstance(config, HybridPruneConfig):
        idf_scores = np.linalg.norm(doc_embs, axis=1)
        pruned, _ = prune_tokens(doc_embs, tda_scores, config, idf_scores=idf_scores)
        return pruned

    pruned, _ = prune_tokens(doc_embs, tda_scores, config)
    return pruned


def execute_pipeline(
    query_embs: NDArray[np.float64],
    doc_embs_list: list[NDArray[np.float64]],
    config: TopoLIConfig,
) -> list[tuple[int, float]]:
    """Execute the multi-stage retrieval pipeline.

    Each stage prunes document tokens, then re-ranks by MaxSim,
    keeping only top_k candidates for the next stage.

    Args:
        query_embs: (n_query, dim) query token embeddings.
        doc_embs_list: List of document token embedding arrays.
        config: Full TopoLI configuration.

    Returns:
        List of (doc_index, score) sorted by descending score.
    """
    if not doc_embs_list:
        return []

    candidates: list[tuple[int, NDArray[np.float64]]] = [
        (i, emb) for i, emb in enumerate(doc_embs_list)
    ]

    similarity = config.interaction.similarity

    for stage in config.pipeline.stages:
        pruned_docs: list[NDArray[np.float64]] = []
        original_indices: list[int] = []

        for doc_idx, doc_emb in candidates:
            pruned = _prune_document(doc_emb, stage.pruning)
            pruned_docs.append(pruned)
            original_indices.append(doc_idx)

        ranked = score_documents(query_embs, pruned_docs, similarity)

        top_k = min(stage.top_k, len(ranked))
        candidates = [
            (original_indices[rank_idx], pruned_docs[rank_idx])
            for rank_idx, _ in ranked[:top_k]
        ]

    final_docs = [emb for _, emb in candidates]
    final_indices = [idx for idx, _ in candidates]
    final_ranked = score_documents(query_embs, final_docs, similarity)

    return [(final_indices[rank_idx], score) for rank_idx, score in final_ranked]
