"""Export TopoLI-1B checkpoint to HuggingFace format.

Usage:
    python -m topoli.scripts.export_hf \
        --checkpoint /path/to/checkpoint \
        --output-dir ./topoli-1b-hf
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)


def main() -> None:
    """Export model checkpoint to HuggingFace format."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    logger.info("Export pipeline:")
    logger.info("1. Load FSDP/DDP checkpoint")
    logger.info("2. Consolidate sharded state dict")
    logger.info("3. Save as HuggingFace-compatible format:")
    logger.info("   - config.json (architecture)")
    logger.info("   - model.safetensors (weights)")
    logger.info("   - tokenizer files")
    logger.info("4. Create model card with:")
    logger.info("   - License: Apache-2.0")
    logger.info("   - Training data: TopoLI-Retrieval (published dataset)")
    logger.info("   - Architecture details")
    logger.info("   - Evaluation results")


if __name__ == "__main__":
    main()
