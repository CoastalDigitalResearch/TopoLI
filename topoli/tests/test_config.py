"""Tests for TopoLI configuration models."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from topoli.config import (
    BaselinePruneConfig,
    DimReduceConfig,
    HybridPruneConfig,
    PipelineConfig,
    QuantizeConfig,
    StageConfig,
    TopoLIConfig,
    TopoPruneConfig,
    baseline_colbertv2,
    hybrid_topo_idf,
    topo_aggressive,
)


class TestPresetConfigs:
    """Preset factory functions produce valid, frozen configs."""

    def test_baseline_colbertv2_is_valid(self) -> None:
        cfg = baseline_colbertv2()
        assert isinstance(cfg, TopoLIConfig)
        assert cfg.backbone.name == "colbertv2"

    def test_topo_aggressive_is_valid(self) -> None:
        cfg = topo_aggressive()
        assert isinstance(cfg, TopoLIConfig)
        assert isinstance(cfg.pipeline.stages[0].pruning, TopoPruneConfig)

    def test_hybrid_topo_idf_is_valid(self) -> None:
        cfg = hybrid_topo_idf()
        assert isinstance(cfg, TopoLIConfig)
        assert isinstance(cfg.pipeline.stages[0].pruning, HybridPruneConfig)


class TestConfigValidation:
    """Invalid configs raise ValidationError."""

    def test_pruning_ratio_above_09_rejected(self) -> None:
        with pytest.raises(ValidationError, match="pruning_ratio"):
            TopoPruneConfig(
                pruning_ratio=0.95,
                homology_dims=(0, 1),
                scoring="birth_death_gap",
            )

    def test_pruning_ratio_below_0_rejected(self) -> None:
        with pytest.raises(ValidationError, match="pruning_ratio"):
            TopoPruneConfig(
                pruning_ratio=-0.1,
                homology_dims=(0, 1),
                scoring="birth_death_gap",
            )

    def test_homology_dim_3_rejected(self) -> None:
        with pytest.raises(ValidationError, match="homology_dim"):
            TopoPruneConfig(
                pruning_ratio=0.5,
                homology_dims=(0, 3),
                scoring="birth_death_gap",
            )

    def test_target_dim_not_in_valid_set(self) -> None:
        with pytest.raises(ValidationError, match="target_dim"):
            DimReduceConfig(method="pca", target_dim=48)

    def test_n_buckets_not_power_of_2(self) -> None:
        with pytest.raises(ValidationError, match="n_buckets"):
            QuantizeConfig(enabled=True, n_buckets=12, n_bits=8)

    def test_topo_weight_outside_0_1(self) -> None:
        with pytest.raises(ValidationError, match="topo_weight"):
            HybridPruneConfig(
                pruning_ratio=0.5,
                homology_dims=(0, 1),
                scoring="birth_death_gap",
                topo_weight=1.5,
                idf_weight=0.5,
            )

    def test_pipeline_stages_not_decreasing_k_rejected(self) -> None:
        stage1 = StageConfig(
            name="stage1",
            top_k=100,
            pruning=BaselinePruneConfig(pruning_ratio=0.3, method="top_k"),
        )
        stage2 = StageConfig(
            name="stage2",
            top_k=200,
            pruning=BaselinePruneConfig(pruning_ratio=0.5, method="top_k"),
        )
        with pytest.raises(ValidationError, match="decreasing"):
            PipelineConfig(stages=(stage1, stage2))


class TestConfigRoundTrip:
    """Config serializes and deserializes identically."""

    def test_round_trip_baseline(self) -> None:
        cfg = baseline_colbertv2()
        dumped = cfg.model_dump()
        restored = TopoLIConfig.model_validate(dumped)
        assert restored == cfg

    def test_round_trip_topo_aggressive(self) -> None:
        cfg = topo_aggressive()
        dumped = cfg.model_dump()
        restored = TopoLIConfig.model_validate(dumped)
        assert restored == cfg

    def test_round_trip_hybrid(self) -> None:
        cfg = hybrid_topo_idf()
        dumped = cfg.model_dump()
        restored = TopoLIConfig.model_validate(dumped)
        assert restored == cfg


class TestConfigFrozen:
    """Configs are immutable."""

    def test_cannot_mutate_top_level(self) -> None:
        cfg = baseline_colbertv2()
        with pytest.raises(ValidationError):
            cfg.backbone = cfg.backbone  # type: ignore[misc]


class TestConfigPropertyBased:
    """Hypothesis-driven boundary tests."""

    @given(ratio=st.floats(min_value=0.0, max_value=0.9))
    @settings(max_examples=50)
    def test_valid_pruning_ratios_accepted(self, ratio: float) -> None:
        cfg = TopoPruneConfig(
            pruning_ratio=ratio,
            homology_dims=(0, 1),
            scoring="birth_death_gap",
        )
        assert cfg.pruning_ratio == ratio

    @given(ratio=st.floats(min_value=0.91, max_value=100.0))
    @settings(max_examples=20)
    def test_invalid_pruning_ratios_rejected(self, ratio: float) -> None:
        with pytest.raises(ValidationError):
            TopoPruneConfig(
                pruning_ratio=ratio,
                homology_dims=(0, 1),
                scoring="birth_death_gap",
            )

    @given(weight=st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=50)
    def test_valid_topo_weights_accepted(self, weight: float) -> None:
        cfg = HybridPruneConfig(
            pruning_ratio=0.5,
            homology_dims=(0, 1),
            scoring="birth_death_gap",
            topo_weight=weight,
            idf_weight=1.0 - weight,
        )
        assert cfg.topo_weight == weight
