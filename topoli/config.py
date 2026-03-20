"""TopoLI configuration models."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BackboneConfig(BaseModel):
    """Configuration for the backbone encoder."""

    model_config = ConfigDict(frozen=True)

    name: str = "colbertv2"
    checkpoint: str = "colbert-ir/colbertv2.0"
    max_length: int = 512
    dim: int = 128


class DimReduceConfig(BaseModel):
    """Configuration for dimensionality reduction."""

    model_config = ConfigDict(frozen=True)

    method: Literal["pca", "random_projection", "none"] = "none"
    target_dim: int = 128

    @field_validator("target_dim")
    @classmethod
    def target_dim_must_be_valid(cls, v: int) -> int:
        valid = {32, 64, 96, 128}
        if v not in valid:
            msg = f"target_dim must be one of {valid}, got {v}"
            raise ValueError(msg)
        return v


class QuantizeConfig(BaseModel):
    """Configuration for quantization."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    n_buckets: int = 256
    n_bits: int = 8

    @field_validator("n_buckets")
    @classmethod
    def n_buckets_must_be_power_of_2(cls, v: int) -> int:
        if v < 1 or (v & (v - 1)) != 0:
            msg = f"n_buckets must be a power of 2, got {v}"
            raise ValueError(msg)
        return v


class ProjectConfig(BaseModel):
    """Configuration for projection layer."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    target_dim: int = 64


class TokenReprConfig(BaseModel):
    """Configuration for token representation."""

    model_config = ConfigDict(frozen=True)

    dim_reduce: DimReduceConfig = DimReduceConfig()
    quantize: QuantizeConfig = QuantizeConfig()
    project: ProjectConfig = ProjectConfig()


class TopoPruneConfig(BaseModel):
    """Topological pruning configuration."""

    model_config = ConfigDict(frozen=True)

    pruning_ratio: Annotated[float, Field(ge=0.0, le=0.9)]
    homology_dims: tuple[int, ...] = (0, 1)
    scoring: Literal[
        "birth_death_gap",
        "representative_cycle",
        "persistence_weighted",
    ] = "birth_death_gap"
    persistence_threshold: float = 0.1
    max_edge_length: float = math.inf
    n_subsample: int | None = None

    @field_validator("homology_dims")
    @classmethod
    def homology_dims_max_2(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        for d in v:
            if d > 2:
                msg = f"homology_dims entries must be <= 2, got {d}"
                raise ValueError(msg)
        return v


class HybridPruneConfig(BaseModel):
    """Hybrid topological + IDF pruning configuration."""

    model_config = ConfigDict(frozen=True)

    pruning_ratio: Annotated[float, Field(ge=0.0, le=0.9)]
    homology_dims: tuple[int, ...] = (0, 1)
    scoring: Literal[
        "birth_death_gap",
        "representative_cycle",
        "persistence_weighted",
    ] = "birth_death_gap"
    topo_weight: Annotated[float, Field(ge=0.0, le=1.0)]
    idf_weight: Annotated[float, Field(ge=0.0, le=1.0)]
    persistence_threshold: float = 0.1

    @field_validator("homology_dims")
    @classmethod
    def homology_dims_max_2(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        for d in v:
            if d > 2:
                msg = f"homology_dims entries must be <= 2, got {d}"
                raise ValueError(msg)
        return v


class BaselinePruneConfig(BaseModel):
    """Baseline (non-topological) pruning configuration."""

    model_config = ConfigDict(frozen=True)

    pruning_ratio: Annotated[float, Field(ge=0.0, le=0.9)]
    method: Literal["top_k", "random", "none"] = "top_k"


PruneConfig = TopoPruneConfig | HybridPruneConfig | BaselinePruneConfig


class InteractionConfig(BaseModel):
    """Configuration for late interaction scoring."""

    model_config = ConfigDict(frozen=True)

    similarity: Literal["cosine", "l2", "dot"] = "cosine"


class QueryExpandConfig(BaseModel):
    """Configuration for query expansion."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    n_expand: int = 0


class StageConfig(BaseModel):
    """Configuration for a single pipeline stage."""

    model_config = ConfigDict(frozen=True)

    name: str
    top_k: int
    pruning: PruneConfig


class PipelineConfig(BaseModel):
    """Multi-stage retrieval pipeline configuration."""

    model_config = ConfigDict(frozen=True)

    stages: tuple[StageConfig, ...]

    @model_validator(mode="after")
    def stages_must_have_decreasing_k(self) -> PipelineConfig:
        for i in range(1, len(self.stages)):
            if self.stages[i].top_k >= self.stages[i - 1].top_k:
                msg = "Pipeline stages must have decreasing top_k values"
                raise ValueError(msg)
        return self


class IndexConfig(BaseModel):
    """Configuration for the index."""

    model_config = ConfigDict(frozen=True)

    n_partitions: int = 128
    n_probe: int = 10


class TopoLIConfig(BaseModel):
    """Top-level TopoLI configuration."""

    model_config = ConfigDict(frozen=True)

    backbone: BackboneConfig = BackboneConfig()
    token_repr: TokenReprConfig = TokenReprConfig()
    pipeline: PipelineConfig
    interaction: InteractionConfig = InteractionConfig()
    query_expand: QueryExpandConfig = QueryExpandConfig()
    index: IndexConfig = IndexConfig()


def baseline_colbertv2() -> TopoLIConfig:
    """Standard ColBERTv2 config without topological pruning."""
    return TopoLIConfig(
        backbone=BackboneConfig(name="colbertv2"),
        pipeline=PipelineConfig(
            stages=(
                StageConfig(
                    name="retrieve",
                    top_k=1000,
                    pruning=BaselinePruneConfig(pruning_ratio=0.0, method="none"),
                ),
                StageConfig(
                    name="rerank",
                    top_k=100,
                    pruning=BaselinePruneConfig(pruning_ratio=0.0, method="none"),
                ),
            ),
        ),
    )


def topo_aggressive() -> TopoLIConfig:
    """Aggressive topological pruning config."""
    return TopoLIConfig(
        backbone=BackboneConfig(name="colbertv2"),
        pipeline=PipelineConfig(
            stages=(
                StageConfig(
                    name="topo_prune",
                    top_k=1000,
                    pruning=TopoPruneConfig(
                        pruning_ratio=0.7,
                        homology_dims=(0, 1),
                        scoring="birth_death_gap",
                    ),
                ),
                StageConfig(
                    name="rerank",
                    top_k=100,
                    pruning=TopoPruneConfig(
                        pruning_ratio=0.3,
                        homology_dims=(0, 1),
                        scoring="representative_cycle",
                    ),
                ),
            ),
        ),
    )


def hybrid_topo_idf() -> TopoLIConfig:
    """Hybrid topological + IDF pruning config."""
    return TopoLIConfig(
        backbone=BackboneConfig(name="colbertv2"),
        pipeline=PipelineConfig(
            stages=(
                StageConfig(
                    name="hybrid_prune",
                    top_k=1000,
                    pruning=HybridPruneConfig(
                        pruning_ratio=0.5,
                        homology_dims=(0, 1),
                        scoring="birth_death_gap",
                        topo_weight=0.7,
                        idf_weight=0.3,
                    ),
                ),
                StageConfig(
                    name="rerank",
                    top_k=100,
                    pruning=BaselinePruneConfig(pruning_ratio=0.0, method="none"),
                ),
            ),
        ),
    )
