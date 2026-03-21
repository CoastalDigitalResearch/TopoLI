"""Tests for topological pruning head."""

from __future__ import annotations

import torch

from topoli.model.pruning_head import PruningHead


class TestPruningHeadForward:
    """Pruning head produces per-token importance scores in [0,1]."""

    def test_output_shape(self) -> None:
        head = PruningHead(hidden_size=64, hidden_dim=32)
        x = torch.randn(2, 16, 64)
        out = head(x)
        assert out.shape == (2, 16, 1)

    def test_output_range_0_to_1(self) -> None:
        head = PruningHead(hidden_size=64, hidden_dim=32)
        x = torch.randn(4, 32, 64)
        out = head(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_output_range_with_extreme_inputs(self) -> None:
        head = PruningHead(hidden_size=64, hidden_dim=32)
        x = torch.randn(2, 8, 64) * 100.0
        out = head(x)
        assert out.min() >= 0.0
        assert out.max() <= 1.0

    def test_default_hidden_dim_is_256(self) -> None:
        head = PruningHead(hidden_size=1536)
        x = torch.randn(1, 4, 1536)
        out = head(x)
        assert out.shape == (1, 4, 1)


class TestPruningHeadGradients:
    """Gradients flow through the pruning head."""

    def test_gradient_flows(self) -> None:
        head = PruningHead(hidden_size=64, hidden_dim=32)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = head(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
