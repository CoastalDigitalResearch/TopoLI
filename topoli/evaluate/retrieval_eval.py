"""Retrieval evaluation: encoding, indexing, reranking, metrics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
import torch
from numpy.typing import NDArray

from topoli.eval import evaluate_retrieval


class EncoderProtocol(Protocol):
    """Protocol for models that encode token IDs to embeddings."""

    def encode(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


class BruteForceIndex:
    """Brute-force ColBERT MaxSim index for exact retrieval."""

    def __init__(self, dim: int) -> None:
        self.dim = dim
        self.doc_ids: list[int] = []
        self.doc_embs: list[NDArray[np.float32]] = []

    def add(self, doc_id: int, embeddings: NDArray[np.float32]) -> None:
        self.doc_ids.append(doc_id)
        self.doc_embs.append(embeddings)

    def search(
        self,
        query_embs: NDArray[np.float32],
        top_k: int,
    ) -> list[tuple[int, float]]:
        if not self.doc_embs:
            return []
        return maxsim_rerank(query_embs, self.doc_embs, self.doc_ids)[:top_k]


def maxsim_rerank(
    query_embs: NDArray[np.float32],
    doc_embs_list: list[NDArray[np.float32]],
    doc_ids: list[int],
) -> list[tuple[int, float]]:
    """Rerank documents by ColBERT MaxSim score."""
    scored: list[tuple[int, float]] = []
    q_norm = query_embs / np.linalg.norm(
        query_embs,
        axis=1,
        keepdims=True,
    ).clip(1e-9)

    for doc_id, doc_embs in zip(doc_ids, doc_embs_list, strict=True):
        d_norm = doc_embs / np.linalg.norm(
            doc_embs,
            axis=1,
            keepdims=True,
        ).clip(1e-9)
        sim = q_norm @ d_norm.T
        score = float(sim.max(axis=1).sum())
        scored.append((doc_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def encode_corpus(
    model: EncoderProtocol,
    texts: list[str],
    tokenize_fn: Callable[[str], torch.Tensor],
) -> list[NDArray[np.float32]]:
    """Encode a corpus of texts to ColBERT token embeddings."""
    all_embs: list[NDArray[np.float32]] = []
    for text in texts:
        input_ids = tokenize_fn(text)
        with torch.no_grad():
            embs, _ = model.encode(input_ids)
        all_embs.append(embs.squeeze(0).cpu().numpy().astype(np.float32))
    return all_embs


def evaluate_retrieval_metrics(
    results: list[tuple[int, float]],
    qrels: set[int],
) -> dict[str, float]:
    """Compute standard retrieval metrics using existing topoli.eval."""
    return evaluate_retrieval(results, qrels)
