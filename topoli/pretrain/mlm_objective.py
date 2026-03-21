"""MLM loss computation."""

from __future__ import annotations

from torch import Tensor
from torch.nn import functional


def mlm_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Compute cross-entropy loss on masked positions only.

    logits: (batch, seq_len, vocab_size)
    labels: (batch, seq_len) with -100 for unmasked positions
    """
    return functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
        ignore_index=-100,
    )
