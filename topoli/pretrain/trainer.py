"""Pre-training trainer: MLM with FSDP, checkpointing, LR scheduling."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from topoli.pretrain.mlm_objective import mlm_loss


def build_mlm_head(hidden_size: int, vocab_size: int) -> nn.Sequential:
    """Build the MLM prediction head (hidden -> vocab logits)."""
    return nn.Sequential(
        nn.Linear(hidden_size, hidden_size, bias=False),
        nn.GELU(),
        nn.LayerNorm(hidden_size),
        nn.Linear(hidden_size, vocab_size, bias=False),
    )


class CosineWithWarmup:
    """Linear warmup followed by cosine decay to min_lr."""

    def __init__(
        self,
        peak_lr: float,
        min_lr: float,
        warmup_steps: int,
        total_steps: int,
    ) -> None:
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def get_lr(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.peak_lr * step / max(1, self.warmup_steps)
        decay_steps = self.total_steps - self.warmup_steps
        progress = min(1.0, (step - self.warmup_steps) / max(1, decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine


class PretrainStep(nn.Module):
    """Single pre-training forward pass: encoder + MLM head -> loss."""

    def __init__(self, encoder: nn.Module, mlm_head: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.mlm_head = mlm_head

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        hidden = self.encoder(input_ids, attention_mask=attention_mask)
        logits = self.mlm_head(hidden)
        return mlm_loss(logits, labels)


def clip_grad_norm(
    model: nn.Module,
    max_norm: float,
) -> float:
    """Clip gradients and return the total norm before clipping."""
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return float(total_norm)
