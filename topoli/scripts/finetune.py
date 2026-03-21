"""Fine-tuning entry point for TopoLI-1B.

Usage:
    torchrun --nproc_per_node=4 -m topoli.scripts.finetune \
        --checkpoint /path/to/pretrained/checkpoint
"""

from __future__ import annotations

import logging
import sys

from topoli.finetune.finetune_config import FinetuneConfig, TopoLossSchedule
from topoli.finetune.trainer import TopoLIForRetrieval
from topoli.model.colbert_head import ColBERTHead
from topoli.model.encoder import TopoLIEncoder
from topoli.model.model_config import topoli_1b
from topoli.model.pruning_head import PruningHead

logger = logging.getLogger(__name__)


def main() -> None:
    """Run ColBERT fine-tuning with topo-aware losses."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    model_cfg = topoli_1b()
    ft_cfg = FinetuneConfig()
    topo_schedule = TopoLossSchedule()

    encoder = TopoLIEncoder(model_cfg.encoder)
    colbert_head = ColBERTHead(model_cfg.encoder.hidden_size, model_cfg.colbert_dim)
    pruning_head = PruningHead(
        model_cfg.encoder.hidden_size, model_cfg.pruning_head_hidden
    )

    model = TopoLIForRetrieval(encoder, colbert_head, pruning_head)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Total parameters: %s", f"{total_params:,}")
    logger.info("Fine-tuning config: %s", ft_cfg.model_dump())
    logger.info(
        "Topo loss schedule: warmup=%d, ramp=%d, full=%d, anneal=%d",
        topo_schedule.warmup_end,
        topo_schedule.ramp_end,
        topo_schedule.full_end,
        topo_schedule.anneal_end,
    )
    logger.info(
        "Load pretrained checkpoint, connect TopoLI-Retrieval dataset, "
        "set up optimizer, and implement the training loop here."
    )


if __name__ == "__main__":
    main()
