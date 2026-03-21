"""Tests for ColBERT projection head."""

from __future__ import annotations

import torch

from topoli.model.colbert_head import ColBERTHead


class TestColBERTHeadForward:
    """ColBERT head projects and normalizes correctly."""

    def test_output_shape(self) -> None:
        head = ColBERTHead(hidden_size=64, colbert_dim=128)
        x = torch.randn(2, 16, 64)
        out = head(x)
        assert out.shape == (2, 16, 128)

    def test_output_is_l2_normalized(self) -> None:
        head = ColBERTHead(hidden_size=64, colbert_dim=128)
        x = torch.randn(2, 16, 64)
        out = head(x)
        norms = torch.norm(out, dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)

    def test_default_colbert_dim_is_128(self) -> None:
        head = ColBERTHead(hidden_size=1536)
        x = torch.randn(1, 8, 1536)
        out = head(x)
        assert out.shape == (1, 8, 128)

    def test_no_bias(self) -> None:
        head = ColBERTHead(hidden_size=64, colbert_dim=128)
        assert head.projection.bias is None


class TestColBERTHeadGradients:
    """Gradients flow through the ColBERT head."""

    def test_gradient_flows(self) -> None:
        head = ColBERTHead(hidden_size=64, colbert_dim=128)
        x = torch.randn(2, 8, 64, requires_grad=True)
        out = head(x)
        out.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0
