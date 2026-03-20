"""Late interaction scoring (ColBERT MaxSim)."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray


def maxsim(
    query_embs: NDArray[np.float64],
    doc_embs: NDArray[np.float64],
    similarity: Literal["cosine", "dot", "l2"] = "cosine",
) -> float:
    """Compute ColBERT MaxSim between query and document embeddings.

    For each query token, find maximum similarity to any doc token, then sum.

    Args:
        query_embs: (n_query, dim) query token embeddings.
        doc_embs: (n_doc, dim) document token embeddings.
        similarity: Similarity function to use.

    Returns:
        Scalar MaxSim score.
    """
    if similarity == "cosine":
        q_norm = query_embs / np.linalg.norm(query_embs, axis=1, keepdims=True).clip(
            1e-9
        )
        d_norm = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True).clip(1e-9)
        sim_matrix = q_norm @ d_norm.T
    elif similarity == "dot":
        sim_matrix = query_embs @ doc_embs.T
    else:
        diffs = query_embs[:, np.newaxis, :] - doc_embs[np.newaxis, :, :]
        sim_matrix = np.sqrt((diffs**2).sum(axis=2))

    max_sims = sim_matrix.max(axis=1)
    result: float = float(max_sims.sum())
    return result


def score_documents(
    query_embs: NDArray[np.float64],
    doc_embs_list: list[NDArray[np.float64]],
    similarity: Literal["cosine", "dot", "l2"] = "cosine",
) -> list[tuple[int, float]]:
    """Score and rank documents against a query using MaxSim.

    Args:
        query_embs: (n_query, dim) query token embeddings.
        doc_embs_list: List of (n_doc_i, dim) document embeddings.
        similarity: Similarity function to use.

    Returns:
        List of (doc_index, score) tuples sorted by descending score.
    """
    if not doc_embs_list:
        return []

    scored = [
        (i, maxsim(query_embs, doc, similarity)) for i, doc in enumerate(doc_embs_list)
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored
