"""TopoLI-1B model configuration."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EncoderConfig(BaseModel):
    """Transformer encoder architecture configuration."""

    model_config = ConfigDict(frozen=True)

    hidden_size: int = 1536
    num_layers: Annotated[int, Field(gt=0)] = 28
    num_attention_heads: int = 12
    intermediate_size: int = 4096
    vocab_size: Annotated[int, Field(gt=0)] = 32000
    max_position_embeddings: int = 8192
    activation: Literal["swiglu", "gelu"] = "swiglu"
    norm_type: Literal["rmsnorm", "layernorm"] = "rmsnorm"
    rope_theta: float = 10000.0
    bias: bool = False
    dropout: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0

    @model_validator(mode="after")
    def hidden_size_divisible_by_heads(self) -> EncoderConfig:
        if self.hidden_size % self.num_attention_heads != 0:
            msg = (
                f"hidden_size ({self.hidden_size}) must be divisible "
                f"by num_attention_heads ({self.num_attention_heads})"
            )
            raise ValueError(msg)
        return self


class ModelConfig(BaseModel):
    """Top-level TopoLI-1B model configuration."""

    model_config = ConfigDict(frozen=True)

    encoder: EncoderConfig = EncoderConfig()
    colbert_dim: Annotated[int, Field(gt=0)] = 128
    pruning_head_hidden: Annotated[int, Field(gt=0)] = 256


def topoli_1b() -> ModelConfig:
    """TopoLI-1B: 1.05B parameter encoder."""
    return ModelConfig(
        encoder=EncoderConfig(
            hidden_size=1536,
            num_layers=28,
            num_attention_heads=12,
            intermediate_size=4096,
        ),
    )


def topoli_400m() -> ModelConfig:
    """TopoLI-400M: 400M parameter encoder for ablation."""
    return ModelConfig(
        encoder=EncoderConfig(
            hidden_size=1024,
            num_layers=24,
            num_attention_heads=16,
            intermediate_size=2816,
        ),
    )


def topoli_150m() -> ModelConfig:
    """TopoLI-150M: 150M parameter encoder for ablation."""
    return ModelConfig(
        encoder=EncoderConfig(
            hidden_size=768,
            num_layers=12,
            num_attention_heads=12,
            intermediate_size=2048,
        ),
    )


def colbert_head_config() -> ModelConfig:
    """Default config emphasizing ColBERT head settings."""
    return ModelConfig(colbert_dim=128)


def pruning_head_config() -> ModelConfig:
    """Default config emphasizing pruning head settings."""
    return ModelConfig(pruning_head_hidden=256)
