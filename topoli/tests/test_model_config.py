"""Tests for TopoLI-1B model configuration."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from topoli.model.model_config import (
    EncoderConfig,
    ModelConfig,
    colbert_head_config,
    pruning_head_config,
    topoli_1b,
    topoli_150m,
    topoli_400m,
)


class TestEncoderConfigDefaults:
    """Default encoder config matches TopoLI-1B spec."""

    def test_1b_hidden_size(self) -> None:
        cfg = EncoderConfig()
        assert cfg.hidden_size == 1536

    def test_1b_num_layers(self) -> None:
        cfg = EncoderConfig()
        assert cfg.num_layers == 28

    def test_1b_num_heads(self) -> None:
        cfg = EncoderConfig()
        assert cfg.num_attention_heads == 12

    def test_head_dim_is_128(self) -> None:
        cfg = EncoderConfig()
        assert cfg.hidden_size // cfg.num_attention_heads == 128

    def test_1b_intermediate_size(self) -> None:
        cfg = EncoderConfig()
        assert cfg.intermediate_size == 4096

    def test_vocab_size(self) -> None:
        cfg = EncoderConfig()
        assert cfg.vocab_size == 32000

    def test_max_position(self) -> None:
        cfg = EncoderConfig()
        assert cfg.max_position_embeddings == 8192

    def test_no_bias(self) -> None:
        cfg = EncoderConfig()
        assert cfg.bias is False

    def test_no_dropout(self) -> None:
        cfg = EncoderConfig()
        assert cfg.dropout == 0.0

    def test_activation_is_swiglu(self) -> None:
        cfg = EncoderConfig()
        assert cfg.activation == "swiglu"

    def test_norm_is_rmsnorm(self) -> None:
        cfg = EncoderConfig()
        assert cfg.norm_type == "rmsnorm"

    def test_rope_theta(self) -> None:
        cfg = EncoderConfig()
        assert cfg.rope_theta == 10000.0


class TestEncoderConfigValidation:
    """Invalid encoder configs raise ValidationError."""

    def test_hidden_size_not_divisible_by_heads(self) -> None:
        with pytest.raises(ValidationError, match="divisible"):
            EncoderConfig(hidden_size=1000, num_attention_heads=12)

    def test_num_layers_must_be_positive(self) -> None:
        with pytest.raises(ValidationError, match="num_layers"):
            EncoderConfig(num_layers=0)

    def test_vocab_size_must_be_positive(self) -> None:
        with pytest.raises(ValidationError, match="vocab_size"):
            EncoderConfig(vocab_size=0)

    def test_dropout_above_1_rejected(self) -> None:
        with pytest.raises(ValidationError, match="dropout"):
            EncoderConfig(dropout=1.5)

    def test_dropout_below_0_rejected(self) -> None:
        with pytest.raises(ValidationError, match="dropout"):
            EncoderConfig(dropout=-0.1)

    def test_invalid_activation(self) -> None:
        with pytest.raises(ValidationError):
            EncoderConfig(activation="relu")  # type: ignore[arg-type]

    def test_invalid_norm_type(self) -> None:
        with pytest.raises(ValidationError):
            EncoderConfig(norm_type="batchnorm")  # type: ignore[arg-type]


class TestModelConfig:
    """Top-level model config composes encoder + head configs."""

    def test_default_colbert_dim(self) -> None:
        cfg = ModelConfig()
        assert cfg.colbert_dim == 128

    def test_default_pruning_head_enabled(self) -> None:
        cfg = ModelConfig()
        assert cfg.pruning_head_hidden == 256

    def test_colbert_dim_must_be_positive(self) -> None:
        with pytest.raises(ValidationError, match="colbert_dim"):
            ModelConfig(colbert_dim=0)

    def test_pruning_head_hidden_must_be_positive(self) -> None:
        with pytest.raises(ValidationError, match="pruning_head_hidden"):
            ModelConfig(pruning_head_hidden=0)


class TestPresetFactories:
    """Preset model configs are valid and have expected param counts."""

    def test_topoli_1b_is_valid(self) -> None:
        cfg = topoli_1b()
        assert isinstance(cfg, ModelConfig)
        assert cfg.encoder.hidden_size == 1536
        assert cfg.encoder.num_layers == 28

    def test_topoli_150m_is_valid(self) -> None:
        cfg = topoli_150m()
        assert isinstance(cfg, ModelConfig)
        assert cfg.encoder.hidden_size == 768
        assert cfg.encoder.num_layers == 12

    def test_topoli_400m_is_valid(self) -> None:
        cfg = topoli_400m()
        assert isinstance(cfg, ModelConfig)
        assert cfg.encoder.hidden_size == 1024
        assert cfg.encoder.num_layers == 24

    def test_colbert_head_config_factory(self) -> None:
        cfg = colbert_head_config()
        assert cfg.colbert_dim == 128

    def test_pruning_head_config_factory(self) -> None:
        cfg = pruning_head_config()
        assert cfg.pruning_head_hidden == 256


class TestConfigFrozen:
    """Model configs are immutable."""

    def test_cannot_mutate_encoder(self) -> None:
        cfg = topoli_1b()
        with pytest.raises(ValidationError):
            cfg.encoder = cfg.encoder  # type: ignore[misc]

    def test_cannot_mutate_colbert_dim(self) -> None:
        cfg = topoli_1b()
        with pytest.raises(ValidationError):
            cfg.colbert_dim = 64  # type: ignore[misc]


class TestConfigRoundTrip:
    """Config serializes and deserializes identically."""

    def test_round_trip_1b(self) -> None:
        cfg = topoli_1b()
        dumped = cfg.model_dump()
        restored = ModelConfig.model_validate(dumped)
        assert restored == cfg

    def test_round_trip_150m(self) -> None:
        cfg = topoli_150m()
        dumped = cfg.model_dump()
        restored = ModelConfig.model_validate(dumped)
        assert restored == cfg


class TestConfigPropertyBased:
    """Hypothesis-driven tests for encoder config."""

    @given(
        hidden=st.sampled_from([768, 1024, 1536, 2048]),
        heads=st.sampled_from([8, 12, 16]),
    )
    @settings(max_examples=30)
    def test_valid_hidden_head_combos(self, hidden: int, heads: int) -> None:
        if hidden % heads == 0:
            cfg = EncoderConfig(
                hidden_size=hidden,
                num_attention_heads=heads,
            )
            assert cfg.hidden_size == hidden
            assert cfg.num_attention_heads == heads
        else:
            with pytest.raises(ValidationError):
                EncoderConfig(
                    hidden_size=hidden,
                    num_attention_heads=heads,
                )

    @given(dropout=st.floats(min_value=0.0, max_value=0.5))
    @settings(max_examples=20)
    def test_valid_dropout_accepted(self, dropout: float) -> None:
        cfg = EncoderConfig(dropout=dropout)
        assert cfg.dropout == dropout
