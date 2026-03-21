"""Pruning head distillation from TDA scores.

Trains the pruning head to predict topological importance scores
computed by Ripser (from topoli.tda.scoring), using MSE loss on
valid (non-padded) positions.
"""

from __future__ import annotations

from torch import Tensor


def pruning_distill_loss(
    pruning_head_output: Tensor,
    tda_target_scores: Tensor,
    attention_mask: Tensor,
) -> Tensor:
    """MSE between predicted and actual TDA importance scores.

    pruning_head_output: (batch, seq_len, 1) — model predictions [0,1]
    tda_target_scores: (batch, seq_len) — Ripser-computed scores
    attention_mask: (batch, seq_len) — 1.0 for valid positions, 0.0 for padding
    """
    pred = pruning_head_output.squeeze(-1)
    diff = (pred - tda_target_scores) * attention_mask
    n_valid = attention_mask.sum().clamp(min=1.0)
    return (diff.pow(2).sum()) / n_valid
