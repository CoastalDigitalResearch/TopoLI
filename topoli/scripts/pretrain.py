"""Pre-training entry point for TopoLI-1B.

Usage:
    torchrun --nproc_per_node=8 -m topoli.scripts.pretrain
"""

from __future__ import annotations

import logging
import sys

from topoli.model.encoder import TopoLIEncoder
from topoli.model.model_config import topoli_1b
from topoli.pretrain.pretrain_config import PretrainConfig
from topoli.pretrain.trainer import (
    PretrainStep,
    build_mlm_head,
)

logger = logging.getLogger(__name__)


def main() -> None:
    """Run MLM pre-training."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    model_cfg = topoli_1b()
    train_cfg = PretrainConfig()
    encoder_cfg = model_cfg.encoder

    logger.info(
        "Building encoder: %d layers, %d hidden",
        encoder_cfg.num_layers,
        encoder_cfg.hidden_size,
    )
    encoder = TopoLIEncoder(encoder_cfg)
    mlm_head = build_mlm_head(encoder_cfg.hidden_size, encoder_cfg.vocab_size)
    step_fn = PretrainStep(encoder, mlm_head)

    total_params = sum(p.numel() for p in step_fn.parameters())
    logger.info("Total parameters: %s", f"{total_params:,}")

    logger.info("Starting pre-training for %d steps", train_cfg.total_steps)
    logger.info(
        "Optimizer: AdamW(lr=%e, wd=%f)",
        train_cfg.learning_rate,
        train_cfg.weight_decay,
    )
    logger.info(
        "Schedule: cosine warmup=%d, total=%d",
        train_cfg.warmup_steps,
        train_cfg.total_steps,
    )
    logger.info(
        "Connect to data sources, set up FSDP wrapping, "
        "and implement the training loop here."
    )
    logger.info(
        "This script provides the scaffolding. "
        "The actual data loading and distributed setup "
        "depend on the compute environment (Thunder Compute)."
    )


if __name__ == "__main__":
    main()
