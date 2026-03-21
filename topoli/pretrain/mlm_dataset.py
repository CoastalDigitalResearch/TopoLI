"""MLM dataset utilities: span masking and sequence packing."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass(frozen=True)
class MaskingParams:
    """Parameters for span masking."""

    mask_token_id: int
    vocab_size: int
    mask_ratio: float = 0.3
    mean_span_length: float = 3.0
    special_token_ids: frozenset[int] = field(default_factory=frozenset)


def apply_span_masking(
    input_ids: Tensor,
    params: MaskingParams,
) -> tuple[Tensor, Tensor]:
    """Apply span masking to a token sequence.

    Returns (masked_input_ids, labels) where labels=-100 for unmasked positions.
    Follows 80/10/10 split: 80% [MASK], 10% random, 10% unchanged.
    """
    seq_len = len(input_ids)
    labels = torch.full_like(input_ids, -100)
    masked_ids = input_ids.clone()

    eligible = torch.ones(seq_len, dtype=torch.bool)
    for sid in params.special_token_ids:
        eligible &= input_ids != sid

    n_eligible = eligible.sum().item()
    if n_eligible == 0:
        return masked_ids, labels

    n_to_mask = max(1, int(n_eligible * params.mask_ratio))
    masked_set = _select_span_positions(
        eligible,
        n_to_mask,
        seq_len,
        params.mean_span_length,
    )
    _apply_mask_replacements(
        masked_ids,
        labels,
        input_ids,
        sorted(masked_set),
        params,
    )
    return masked_ids, labels


def _select_span_positions(
    eligible: Tensor,
    n_to_mask: int,
    seq_len: int,
    mean_span_length: float,
) -> set[int]:
    """Select positions to mask using geometric-distributed spans."""
    eligible_indices = eligible.nonzero(as_tuple=True)[0].tolist()
    masked_set: set[int] = set()
    geom = torch.distributions.Geometric(1.0 / mean_span_length)

    while len(masked_set) < n_to_mask and eligible_indices:
        rand_idx = int(torch.randint(0, len(eligible_indices), (1,)).item())
        start_idx = eligible_indices[rand_idx]
        span_len = max(1, int(geom.sample().item()))  # type: ignore[no-untyped-call]
        for offset in range(span_len):
            pos = start_idx + offset
            if pos < seq_len and eligible[pos]:
                masked_set.add(pos)
            if len(masked_set) >= n_to_mask:
                break

    return masked_set


def _apply_mask_replacements(
    masked_ids: Tensor,
    labels: Tensor,
    input_ids: Tensor,
    positions: list[int],
    params: MaskingParams,
) -> None:
    """Apply 80/10/10 replacement to selected positions."""
    for pos in positions:
        labels[pos] = input_ids[pos]
        rand_val = torch.rand(1).item()
        if rand_val < 0.8:
            masked_ids[pos] = params.mask_token_id
        elif rand_val < 0.9:
            masked_ids[pos] = torch.randint(0, params.vocab_size, (1,)).item()


def pack_sequences(
    sequences: list[Tensor],
    max_length: int,
    sep_token_id: int,
) -> list[Tensor]:
    """Pack multiple short sequences into fixed-length sequences.

    Concatenates documents with [SEP] separators and pads/truncates to max_length.
    """
    if not sequences:
        return []

    packed: list[Tensor] = []
    current_tokens: list[int] = []

    for seq in sequences:
        seq_list = seq.tolist()

        if current_tokens and len(current_tokens) + 1 + len(seq_list) > max_length:
            # Pad or truncate current
            packed.append(_finalize(current_tokens, max_length, sep_token_id))
            current_tokens = []

        if current_tokens:
            current_tokens.append(sep_token_id)
        current_tokens.extend(seq_list)

        if len(current_tokens) >= max_length:
            packed.append(_finalize(current_tokens, max_length, sep_token_id))
            current_tokens = []

    if current_tokens:
        packed.append(_finalize(current_tokens, max_length, sep_token_id))

    return packed


def _finalize(tokens: list[int], max_length: int, pad_id: int) -> Tensor:
    """Truncate or pad a token list to exact max_length."""
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    elif len(tokens) < max_length:
        tokens = tokens + [pad_id] * (max_length - len(tokens))
    return torch.tensor(tokens, dtype=torch.long)
