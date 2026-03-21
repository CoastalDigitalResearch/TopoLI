"""Pre-training configuration."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class PretrainConfig(BaseModel):
    """Configuration for MLM pre-training."""

    model_config = ConfigDict(frozen=True)

    batch_size: int = 2048
    max_length: int = 512
    learning_rate: Annotated[float, Field(gt=0)] = 6e-4
    min_learning_rate: float = 6e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    warmup_steps: int = 2000
    total_steps: int = 200_000
    gradient_clip: float = 1.0
    mask_ratio: float = 0.3
    mean_span_length: float = 3.0
    checkpoint_every: int = 5000
    keep_checkpoints: int = 3
    log_every: int = 100
    precision: str = "bf16"
