"""Build the TopoLI-Retrieval dataset from scratch.

Usage:
    python -m topoli.scripts.build_dataset --output-dir ./topoli-retrieval-v1

Extracts passages from permissive-license sources, generates queries
via doc2query (Qwen3-8B Apache-2.0), filters, deduplicates, mines
hard negatives, and outputs JSONL with full provenance.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from topoli.data.source_config import get_source_registry

logger = logging.getLogger(__name__)


def main() -> None:
    """Build the TopoLI-Retrieval dataset."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    output_dir = Path("./topoli-retrieval-v1")
    output_dir.mkdir(parents=True, exist_ok=True)

    registry = get_source_registry()
    logger.info("Registered %d data sources:", len(registry))
    for src in registry:
        logger.info("  %s (%s) — %s", src.name, src.license.value, src.domain.value)
        if not src.commercially_usable:
            msg = f"Source {src.name} is not commercially usable!"
            raise ValueError(msg)

    logger.info("\nDataset build pipeline:")
    logger.info(
        "1. Extract passages from each source (Common Corpus, KL3M, SlimPajama)"
    )
    logger.info("2. Generate queries via Qwen3-8B (Apache-2.0)")
    logger.info("3. Filter: length, overlap, dedup")
    logger.info("4. Mine hard negatives (BM25)")
    logger.info("5. Output JSONL with per-record provenance")
    logger.info("")
    logger.info(
        "This script provides the orchestration scaffolding. "
        "Each step requires downloading the actual data sources "
        "and running inference on a GPU for query generation."
    )
    logger.info("Output directory: %s", output_dir)


if __name__ == "__main__":
    main()
