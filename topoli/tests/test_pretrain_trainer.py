"""Tests for pre-training trainer components."""

from __future__ import annotations

import torch

from topoli.model.encoder import TopoLIEncoder
from topoli.model.model_config import EncoderConfig
from topoli.pretrain.trainer import (
    CosineWithWarmup,
    PretrainStep,
    build_mlm_head,
)


def _tiny_encoder() -> TopoLIEncoder:
    cfg = EncoderConfig(
        hidden_size=64,
        num_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        vocab_size=100,
    )
    return TopoLIEncoder(cfg)


class TestMLMHead:
    """MLM head maps encoder output to vocab logits."""

    def test_output_shape(self) -> None:
        head = build_mlm_head(hidden_size=64, vocab_size=100)
        x = torch.randn(2, 16, 64)
        logits = head(x)
        assert logits.shape == (2, 16, 100)

    def test_gradient_flows(self) -> None:
        head = build_mlm_head(hidden_size=64, vocab_size=100)
        x = torch.randn(2, 8, 64, requires_grad=True)
        logits = head(x)
        logits.sum().backward()
        assert x.grad is not None


class TestCosineWithWarmup:
    """LR schedule with linear warmup and cosine decay."""

    def test_starts_at_zero(self) -> None:
        scheduler = CosineWithWarmup(
            peak_lr=6e-4,
            min_lr=6e-5,
            warmup_steps=100,
            total_steps=1000,
        )
        assert scheduler.get_lr(0) < 1e-6

    def test_reaches_peak_at_warmup_end(self) -> None:
        scheduler = CosineWithWarmup(
            peak_lr=6e-4,
            min_lr=6e-5,
            warmup_steps=100,
            total_steps=1000,
        )
        lr = scheduler.get_lr(100)
        assert abs(lr - 6e-4) < 1e-6

    def test_decays_after_warmup(self) -> None:
        scheduler = CosineWithWarmup(
            peak_lr=6e-4,
            min_lr=6e-5,
            warmup_steps=100,
            total_steps=1000,
        )
        lr_peak = scheduler.get_lr(100)
        lr_mid = scheduler.get_lr(550)
        lr_end = scheduler.get_lr(999)
        assert lr_peak > lr_mid > lr_end

    def test_never_below_min(self) -> None:
        scheduler = CosineWithWarmup(
            peak_lr=6e-4,
            min_lr=6e-5,
            warmup_steps=100,
            total_steps=1000,
        )
        for step in range(1001):
            assert scheduler.get_lr(step) >= 0.0


class TestPretrainStep:
    """Single pre-training step (forward + backward)."""

    def test_returns_loss(self) -> None:
        encoder = _tiny_encoder()
        mlm_head = build_mlm_head(hidden_size=64, vocab_size=100)
        step_fn = PretrainStep(encoder, mlm_head)

        input_ids = torch.randint(0, 100, (2, 16))
        labels = torch.full((2, 16), -100, dtype=torch.long)
        labels[0, 3:6] = torch.randint(0, 100, (3,))
        labels[1, 7:10] = torch.randint(0, 100, (3,))

        loss = step_fn(input_ids, labels)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_gradient_flows_to_encoder(self) -> None:
        encoder = _tiny_encoder()
        mlm_head = build_mlm_head(hidden_size=64, vocab_size=100)
        step_fn = PretrainStep(encoder, mlm_head)

        input_ids = torch.randint(0, 100, (2, 8))
        labels = torch.full((2, 8), -100, dtype=torch.long)
        labels[0, 2:5] = torch.randint(0, 100, (3,))

        loss = step_fn(input_ids, labels)
        loss.backward()
        assert encoder.token_embedding.weight.grad is not None

    def test_accepts_attention_mask(self) -> None:
        encoder = _tiny_encoder()
        mlm_head = build_mlm_head(hidden_size=64, vocab_size=100)
        step_fn = PretrainStep(encoder, mlm_head)

        input_ids = torch.randint(0, 100, (2, 8))
        labels = torch.full((2, 8), -100, dtype=torch.long)
        labels[0, 2:4] = torch.randint(0, 100, (2,))
        mask = torch.ones(2, 8, dtype=torch.bool)
        mask[0, 6:] = False

        loss = step_fn(input_ids, labels, attention_mask=mask)
        assert loss.item() > 0
