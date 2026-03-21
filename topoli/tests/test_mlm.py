"""Tests for MLM masking, sequence packing, and loss computation."""

from __future__ import annotations

import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from topoli.pretrain.mlm_dataset import (
    MaskingParams,
    apply_span_masking,
    pack_sequences,
)
from topoli.pretrain.mlm_objective import mlm_loss


class TestSpanMasking:
    """Span masking creates contiguous masked regions."""

    def test_mask_ratio_approximately_correct(self) -> None:
        input_ids = torch.randint(10, 1000, (512,))
        params = MaskingParams(mask_token_id=1, vocab_size=1000, mask_ratio=0.3)
        masked_ids, labels = apply_span_masking(input_ids, params)
        n_masked = (labels != -100).sum().item()
        ratio = n_masked / len(input_ids)
        assert 0.15 < ratio < 0.45

    def test_labels_match_original_at_masked_positions(self) -> None:
        input_ids = torch.randint(10, 1000, (256,))
        params = MaskingParams(mask_token_id=1, vocab_size=1000, mask_ratio=0.3)
        masked_ids, labels = apply_span_masking(input_ids, params)
        mask_positions = labels != -100
        torch.testing.assert_close(labels[mask_positions], input_ids[mask_positions])

    def test_unmasked_positions_have_ignore_label(self) -> None:
        input_ids = torch.randint(10, 1000, (128,))
        params = MaskingParams(mask_token_id=1, vocab_size=1000, mask_ratio=0.3)
        masked_ids, labels = apply_span_masking(input_ids, params)
        unmasked = labels == -100
        torch.testing.assert_close(masked_ids[unmasked], input_ids[unmasked])

    def test_80_10_10_split(self) -> None:
        torch.manual_seed(42)
        input_ids = torch.randint(10, 1000, (2048,))
        params = MaskingParams(mask_token_id=1, vocab_size=1000, mask_ratio=0.3)
        masked_ids, labels = apply_span_masking(input_ids, params)
        mask_positions = labels != -100
        n_masked = mask_positions.sum().item()
        if n_masked > 0:
            n_mask_token = (masked_ids[mask_positions] == 1).sum().item()
            assert n_mask_token / n_masked > 0.5

    @given(seed=st.integers(min_value=0, max_value=10000))
    @settings(max_examples=10)
    def test_never_masks_special_tokens(self, seed: int) -> None:
        torch.manual_seed(seed)
        input_ids = torch.randint(0, 1000, (128,))
        input_ids[0] = 0  # CLS
        input_ids[-1] = 2  # SEP
        params = MaskingParams(
            mask_token_id=1,
            vocab_size=1000,
            mask_ratio=0.3,
            special_token_ids=frozenset({0, 1, 2, 3}),
        )
        masked_ids, labels = apply_span_masking(input_ids, params)
        for special_id in [0, 2]:
            special_positions = input_ids == special_id
            assert (labels[special_positions] == -100).all()


class TestSequencePacking:
    """Sequence packing concatenates documents without padding waste."""

    def test_packs_to_target_length(self) -> None:
        sequences = [torch.randint(10, 1000, (50,)) for _ in range(10)]
        packed = pack_sequences(sequences, max_length=128, sep_token_id=2)
        for seq in packed:
            assert len(seq) == 128

    def test_short_sequences_combined(self) -> None:
        sequences = [torch.randint(10, 1000, (30,)) for _ in range(10)]
        packed = pack_sequences(sequences, max_length=128, sep_token_id=2)
        # 10 sequences of 30 tokens ~ 300 tokens -> should produce ~2-3 packed sequences
        assert len(packed) < 10

    def test_separator_between_documents(self) -> None:
        seq_a = torch.full((20,), 100)
        seq_b = torch.full((20,), 200)
        packed = pack_sequences([seq_a, seq_b], max_length=50, sep_token_id=2)
        # The separator token should appear between documents
        assert 2 in packed[0].tolist()

    def test_empty_input(self) -> None:
        packed = pack_sequences([], max_length=128, sep_token_id=2)
        assert len(packed) == 0


class TestMLMLoss:
    """MLM loss computation on masked positions only."""

    def test_loss_is_scalar(self) -> None:
        logits = torch.randn(2, 16, 100)
        labels = torch.full((2, 16), -100, dtype=torch.long)
        labels[0, 3:6] = torch.randint(0, 100, (3,))
        labels[1, 7:10] = torch.randint(0, 100, (3,))
        loss = mlm_loss(logits, labels)
        assert loss.shape == ()

    def test_loss_is_positive(self) -> None:
        logits = torch.randn(2, 16, 100)
        labels = torch.full((2, 16), -100, dtype=torch.long)
        labels[0, 3:6] = torch.randint(0, 100, (3,))
        loss = mlm_loss(logits, labels)
        assert loss.item() > 0

    def test_loss_decreases_with_correct_predictions(self) -> None:
        labels = torch.full((1, 8), -100, dtype=torch.long)
        labels[0, 2:5] = torch.tensor([10, 20, 30])

        # Random logits
        logits_random = torch.randn(1, 8, 100)
        loss_random = mlm_loss(logits_random, labels)

        # Logits that strongly predict the correct tokens
        logits_correct = torch.randn(1, 8, 100) * 0.01
        logits_correct[0, 2, 10] = 10.0
        logits_correct[0, 3, 20] = 10.0
        logits_correct[0, 4, 30] = 10.0
        loss_correct = mlm_loss(logits_correct, labels)

        assert loss_correct.item() < loss_random.item()

    def test_gradient_flows(self) -> None:
        logits = torch.randn(2, 8, 100, requires_grad=True)
        labels = torch.full((2, 8), -100, dtype=torch.long)
        labels[0, 2:4] = torch.randint(0, 100, (2,))
        loss = mlm_loss(logits, labels)
        loss.backward()
        assert logits.grad is not None
