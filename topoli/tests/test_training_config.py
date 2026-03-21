"""Tests for pre-training and fine-tuning configurations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from topoli.finetune.finetune_config import FinetuneConfig, TopoLossSchedule
from topoli.pretrain.pretrain_config import PretrainConfig


class TestPretrainConfig:
    """Pre-training config is frozen and validated."""

    def test_defaults(self) -> None:
        cfg = PretrainConfig()
        assert cfg.batch_size == 2048
        assert cfg.max_length == 512
        assert cfg.learning_rate == 6e-4
        assert cfg.weight_decay == 0.1
        assert cfg.warmup_steps == 2000
        assert cfg.total_steps == 200_000
        assert cfg.mask_ratio == 0.3

    def test_frozen(self) -> None:
        cfg = PretrainConfig()
        with pytest.raises(ValidationError):
            cfg.batch_size = 1024  # type: ignore[misc]

    def test_learning_rate_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            PretrainConfig(learning_rate=0.0)

    def test_checkpoint_interval(self) -> None:
        cfg = PretrainConfig()
        assert cfg.checkpoint_every == 5000


class TestFinetuneConfig:
    """Fine-tuning config is frozen and validated."""

    def test_defaults(self) -> None:
        cfg = FinetuneConfig()
        assert cfg.batch_size == 128
        assert cfg.learning_rate == 3e-5
        assert cfg.total_steps == 120_000
        assert cfg.n_negatives == 7
        assert cfg.temperature == 0.05
        assert cfg.query_max_length == 32
        assert cfg.doc_max_length == 256

    def test_frozen(self) -> None:
        cfg = FinetuneConfig()
        with pytest.raises(ValidationError):
            cfg.batch_size = 64  # type: ignore[misc]

    def test_temperature_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            FinetuneConfig(temperature=0.0)


class TestTopoLossSchedule:
    """Topo loss schedule ramps alpha/beta correctly."""

    def test_warmup_phase_zero(self) -> None:
        schedule = TopoLossSchedule()
        alpha, beta = schedule.get_weights(step=1000)
        assert alpha == 0.0
        assert beta == 0.0

    def test_ramp_phase_interpolates(self) -> None:
        schedule = TopoLossSchedule()
        alpha, beta = schedule.get_weights(step=10_000)
        assert 0.0 < alpha < 0.05
        assert 0.0 < beta < 0.1

    def test_full_phase_at_max(self) -> None:
        schedule = TopoLossSchedule()
        alpha, beta = schedule.get_weights(step=50_000)
        assert alpha == 0.05
        assert beta == 0.1

    def test_anneal_phase_decreases(self) -> None:
        schedule = TopoLossSchedule()
        alpha, beta = schedule.get_weights(step=110_000)
        assert alpha < 0.05
        assert beta < 0.1
        assert alpha > 0.0
        assert beta > 0.0
