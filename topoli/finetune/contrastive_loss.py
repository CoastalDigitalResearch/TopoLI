"""ColBERT contrastive loss with differentiable MaxSim.

Implements InfoNCE loss using ColBERT's late interaction scoring
(sum of per-query-token maximum cosine similarities).
"""

from __future__ import annotations

import torch
from torch import Tensor


def maxsim(query_embs: Tensor, doc_embs: Tensor) -> Tensor:
    """Compute ColBERT MaxSim score between queries and documents.

    query_embs: (batch, q_len, dim)
    doc_embs: (batch, d_len, dim)
    Returns: (batch,) MaxSim scores
    """
    sim_matrix = torch.bmm(query_embs, doc_embs.transpose(1, 2))
    max_sim_per_token = sim_matrix.max(dim=2).values
    return max_sim_per_token.sum(dim=1)  # (batch,)


def colbert_info_nce(
    query_embs: Tensor,
    pos_doc_embs: Tensor,
    neg_doc_embs: Tensor,
    temperature: float = 0.05,
) -> Tensor:
    """InfoNCE loss with ColBERT MaxSim scoring.

    query_embs: (batch, q_len, dim)
    pos_doc_embs: (batch, d_len, dim)
    neg_doc_embs: (batch, n_neg, d_len, dim)
    """
    n_neg = neg_doc_embs.size(1)

    pos_scores = maxsim(query_embs, pos_doc_embs) / temperature  # (batch,)

    # Score each negative
    neg_scores_list: list[Tensor] = []
    for i in range(n_neg):
        neg_score = maxsim(query_embs, neg_doc_embs[:, i]) / temperature
        neg_scores_list.append(neg_score)
    neg_scores = torch.stack(neg_scores_list, dim=1)  # (batch, n_neg)

    all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)
    log_softmax = torch.log_softmax(all_scores, dim=1)
    return -log_softmax[:, 0].mean()
