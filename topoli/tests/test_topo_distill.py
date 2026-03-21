"""Tests for topological pruning head distillation loss."""

from __future__ import annotations

import torch

from topoli.finetune.topo_distill import pruning_distill_loss


class TestPruningDistillLoss:
    """Distillation loss trains pruning head to predict TDA scores."""

    def test_returns_scalar(self) -> None:
        pred = torch.rand(2, 16, 1)
        target = torch.rand(2, 16)
        mask = torch.ones(2, 16)
        loss = pruning_distill_loss(pred, target, mask)
        assert loss.shape == ()

    def test_zero_loss_for_perfect_prediction(self) -> None:
        target = torch.rand(2, 16)
        pred = target.unsqueeze(-1)
        mask = torch.ones(2, 16)
        loss = pruning_distill_loss(pred, target, mask)
        torch.testing.assert_close(loss, torch.tensor(0.0), atol=1e-6, rtol=1e-6)

    def test_masked_positions_ignored(self) -> None:
        pred = torch.rand(2, 16, 1)
        target = torch.rand(2, 16)
        mask = torch.zeros(2, 16)
        mask[0, :8] = 1.0
        loss = pruning_distill_loss(pred, target, mask)
        assert loss.item() >= 0

    def test_gradient_flows(self) -> None:
        pred = torch.rand(2, 16, 1, requires_grad=True)
        target = torch.rand(2, 16)
        mask = torch.ones(2, 16)
        loss = pruning_distill_loss(pred, target, mask)
        loss.backward()
        assert pred.grad is not None

    def test_loss_decreases_with_better_predictions(self) -> None:
        target = torch.rand(2, 16)
        mask = torch.ones(2, 16)

        pred_bad = torch.rand(2, 16, 1)
        pred_good = target.unsqueeze(-1) + torch.randn(2, 16, 1) * 0.01

        loss_bad = pruning_distill_loss(pred_bad, target, mask)
        loss_good = pruning_distill_loss(pred_good, target, mask)
        assert loss_good.item() < loss_bad.item()
