"""Fine-tuning configuration with topological loss scheduling."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class FinetuneConfig(BaseModel):
    """Configuration for ColBERT retrieval fine-tuning."""

    model_config = ConfigDict(frozen=True)

    batch_size: int = 128
    learning_rate: Annotated[float, Field(gt=0)] = 3e-5
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    total_steps: int = 120_000
    gradient_clip: float = 1.0
    n_negatives: int = 7
    temperature: Annotated[float, Field(gt=0)] = 0.05
    query_max_length: int = 32
    doc_max_length: int = 256
    hard_neg_remine_every: int = 10_000
    hard_neg_start_step: int = 20_000
    checkpoint_every: int = 5000
    log_every: int = 100
    precision: str = "bf16"


class TopoLossSchedule(BaseModel):
    """Schedule for ramping topological loss weights."""

    model_config = ConfigDict(frozen=True)

    warmup_end: int = 5_000
    ramp_end: int = 15_000
    full_end: int = 100_000
    anneal_end: int = 120_000
    alpha_max: float = 0.05
    alpha_min: float = 0.01
    beta_max: float = 0.1
    beta_min: float = 0.05

    def get_weights(self, step: int) -> tuple[float, float]:
        """Get (alpha, beta) for the given training step."""
        if step < self.warmup_end:
            return 0.0, 0.0

        if step < self.ramp_end:
            progress = (step - self.warmup_end) / (self.ramp_end - self.warmup_end)
            return self.alpha_max * progress, self.beta_max * progress

        if step < self.full_end:
            return self.alpha_max, self.beta_max

        if step < self.anneal_end:
            progress = (step - self.full_end) / (self.anneal_end - self.full_end)
            alpha = self.alpha_max - (self.alpha_max - self.alpha_min) * progress
            beta = self.beta_max - (self.beta_max - self.beta_min) * progress
            return alpha, beta

        return self.alpha_min, self.beta_min
