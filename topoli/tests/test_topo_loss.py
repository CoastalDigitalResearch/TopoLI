"""Tests for differentiable topological structure loss."""

from __future__ import annotations

import torch

from topoli.finetune.topo_loss import (
    dtm_topo_loss,
    pairwise_variance_loss,
    topo_structure_loss,
)


class TestPairwiseVarianceLoss:
    """Pairwise variance loss encourages bimodal distance distribution."""

    def test_returns_scalar(self) -> None:
        embs = torch.randn(16, 128)
        loss = pairwise_variance_loss(embs)
        assert loss.shape == ()

    def test_clustered_embeddings_lower_loss(self) -> None:
        cluster_a = torch.randn(8, 128) * 0.01 + torch.tensor([1.0] * 128)
        cluster_b = torch.randn(8, 128) * 0.01 + torch.tensor([-1.0] * 128)
        clustered = torch.cat([cluster_a, cluster_b])
        loss_clustered = pairwise_variance_loss(clustered)

        uniform = torch.randn(16, 128)
        loss_uniform = pairwise_variance_loss(uniform)

        assert loss_clustered.item() < loss_uniform.item()

    def test_gradient_flows(self) -> None:
        embs = torch.randn(16, 128, requires_grad=True)
        loss = pairwise_variance_loss(embs)
        loss.backward()
        assert embs.grad is not None
        assert embs.grad.abs().sum() > 0


class TestDTMTopoLoss:
    """DTM-based topological loss captures cluster/bridge structure."""

    def test_returns_scalar(self) -> None:
        embs = torch.randn(16, 128)
        loss = dtm_topo_loss(embs, k=3)
        assert loss.shape == ()

    def test_gradient_flows(self) -> None:
        embs = torch.randn(16, 128, requires_grad=True)
        loss = dtm_topo_loss(embs, k=3)
        loss.backward()
        assert embs.grad is not None

    def test_different_k_values(self) -> None:
        embs = torch.randn(20, 128)
        loss_k3 = dtm_topo_loss(embs, k=3)
        loss_k5 = dtm_topo_loss(embs, k=5)
        assert loss_k3.shape == ()
        assert loss_k5.shape == ()


class TestTopoStructureLoss:
    """Combined topological structure loss."""

    def test_returns_scalar(self) -> None:
        embs = torch.randn(16, 128)
        loss = topo_structure_loss(embs)
        assert loss.shape == ()

    def test_gradient_flows(self) -> None:
        embs = torch.randn(16, 128, requires_grad=True)
        loss = topo_structure_loss(embs)
        loss.backward()
        assert embs.grad is not None

    def test_batched_input(self) -> None:
        embs = torch.randn(4, 16, 128)
        loss = topo_structure_loss(embs)
        assert loss.shape == ()
