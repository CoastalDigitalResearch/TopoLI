"""ColBERT projection head for late interaction."""

from __future__ import annotations

from torch import Tensor, nn
from torch.nn import functional


class ColBERTHead(nn.Module):
    """Linear projection to ColBERT embedding dim with L2 normalization."""

    def __init__(
        self,
        hidden_size: int,
        colbert_dim: int = 128,
    ) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, colbert_dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return functional.normalize(self.projection(x), dim=-1)
