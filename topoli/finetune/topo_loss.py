"""Differentiable topological structure losses.

These are differentiable proxies for persistent homology that encourage
token embeddings to form well-separated clusters with clear bridge structure.
This is the key novel contribution of TopoLI-1B.
"""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional


def pairwise_variance_loss(token_embs: Tensor) -> Tensor:
    """Encourage bimodal pairwise distance distribution.

    Maximizes variance of pairwise cosine distances (proxy for well-separated
    clusters with clear bridges). Lower loss = more topological structure.

    token_embs: (n_tokens, dim) or (batch, n_tokens, dim)
    """
    if token_embs.dim() == 3:
        losses = torch.stack([pairwise_variance_loss(e) for e in token_embs])
        return losses.mean()

    normed = functional.normalize(token_embs, dim=-1)
    cos_sim = normed @ normed.T
    cos_dist = 1.0 - cos_sim

    upper_tri_mask = torch.triu(torch.ones_like(cos_dist, dtype=torch.bool), diagonal=1)
    distances = cos_dist[upper_tri_mask]

    local_density = _knn_mean_distance(cos_dist, k=5)

    return -distances.var() + 0.5 * local_density.mean()


def _knn_mean_distance(dist_matrix: Tensor, k: int = 5) -> Tensor:
    """Mean distance to k nearest neighbors per point."""
    k_clamped = min(k, dist_matrix.size(0) - 1)
    if k_clamped <= 0:
        return torch.tensor(0.0, device=dist_matrix.device)
    topk_dists = torch.topk(dist_matrix, k=k_clamped + 1, dim=-1, largest=False)
    return topk_dists.values[:, 1:].mean(dim=-1)


def dtm_topo_loss(token_embs: Tensor, k: int = 5) -> Tensor:
    """Distance-to-Measure based topological loss.

    DTM captures H0 structure: points with high DTM are bridges/outliers,
    points with low DTM are in dense clusters. We maximize DTM variance
    to encourage clear cluster/bridge distinction.

    token_embs: (n_tokens, dim)
    """
    if token_embs.dim() == 3:
        losses = torch.stack([dtm_topo_loss(e, k=k) for e in token_embs])
        return losses.mean()

    normed = functional.normalize(token_embs, dim=-1)
    dists = torch.cdist(normed, normed)

    k_clamped = min(k, dists.size(0) - 1)
    if k_clamped <= 0:
        return torch.tensor(0.0, device=token_embs.device, requires_grad=True)

    topk_sq = torch.topk(dists.pow(2), k=k_clamped + 1, dim=-1, largest=False)
    dtm = torch.sqrt(topk_sq.values[:, 1:].mean(dim=-1) + 1e-8)

    return -dtm.var()


def topo_structure_loss(
    token_embs: Tensor,
    pairwise_weight: float = 0.5,
    dtm_weight: float = 0.5,
) -> Tensor:
    """Combined topological structure loss.

    Weighted combination of pairwise variance and DTM losses.
    token_embs: (n_tokens, dim) or (batch, n_tokens, dim)
    """
    pw_loss = pairwise_variance_loss(token_embs)
    dtm_loss_val = dtm_topo_loss(token_embs)
    return pairwise_weight * pw_loss + dtm_weight * dtm_loss_val
