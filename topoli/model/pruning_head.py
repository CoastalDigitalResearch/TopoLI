"""Topological pruning head predicting per-token importance."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PruningHead(nn.Module):
    """MLP predicting topological importance scores per token."""

    def __init__(
        self,
        hidden_size: int,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        swiglu_out = hidden_dim // 2
        self.gate_proj = nn.Linear(hidden_size, hidden_dim, bias=False)
        self.up_proj = nn.Linear(hidden_size, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, swiglu_out, bias=False)
        self.score_proj = nn.Linear(swiglu_out, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        h = nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        h = self.down_proj(h)
        return torch.sigmoid(self.score_proj(h))
