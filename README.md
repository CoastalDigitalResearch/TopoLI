# TopoLI

**Topological Late Interaction** — a ColBERTv2 variant that uses persistent homology to intelligently prune token embeddings for efficient, high-quality neural retrieval.

## Overview

ColBERT-style models represent documents as bags of token embeddings and score queries via MaxSim (maximum similarity per query token, summed). This is powerful but storage-heavy. TopoLI asks: *which tokens actually matter for retrieval?*

Using **persistent homology** from topological data analysis, TopoLI identifies tokens that are structurally important — bridge tokens connecting semantic clusters, boundary tokens forming topological cycles — and prunes the rest. The result is fewer stored tokens with minimal retrieval quality loss.

## Quick Start

```bash
# Install
uv sync --all-extras

# Run tests
uv run pytest topoli/tests/ -q

# Lint + type check
uv run ruff check topoli/
uv run mypy --strict topoli/
```

## Usage

```python
from topoli.config import topo_aggressive
from topoli.pipeline import execute_pipeline

# Load a preset config (70% topological pruning)
config = topo_aggressive()

# query_embs: (n_query_tokens, 128) numpy array
# doc_embs_list: list of (n_doc_tokens, 128) numpy arrays
results = execute_pipeline(query_embs, doc_embs_list, config)

# results: [(doc_index, score), ...] sorted by descending score
```

### Preset Configurations

```python
from topoli.config import baseline_colbertv2, topo_aggressive, hybrid_topo_idf

# Standard ColBERT, no pruning (baseline)
baseline_colbertv2()

# Aggressive topological pruning (70% reduction)
topo_aggressive()

# Hybrid: 50% pruning combining TDA scores + IDF weighting
hybrid_topo_idf()
```

### Custom Configuration

```python
from topoli.config import (
    TopoLIConfig, PipelineConfig, StageConfig, TopoPruneConfig, BaselinePruneConfig
)

config = TopoLIConfig(
    pipeline=PipelineConfig(
        stages=(
            StageConfig(
                name="topo_prune",
                top_k=1000,
                pruning=TopoPruneConfig(
                    pruning_ratio=0.5,
                    homology_dims=(0, 1),
                    scoring="birth_death_gap",
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
```

## How It Works

1. **Compute persistence** — Build a Vietoris-Rips filtration on document token embeddings using cosine distance
2. **Score tokens** — Map persistent features back to tokens via birth-death gap, representative cycle, or persistence-weighted scoring
3. **Prune** — Keep the top (1 - ratio) tokens by importance score
4. **Retrieve** — Score pruned documents against queries using ColBERT MaxSim

See [docs/THEORY.md](docs/THEORY.md) for the mathematical foundations and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for system design.

## Scoring Functions

| Function | What it detects | Best for |
|----------|----------------|----------|
| `birth_death_gap` | Bridge tokens connecting clusters | General-purpose pruning |
| `representative_cycle` | Tokens in topological loops | Structural document analysis |
| `persistence_weighted` | Tokens at topologically significant scales | Soft scoring with wide coverage |

## Test Suite

86 tests covering:
- Configuration validation with Hypothesis property-based testing
- Topological correctness (circle H1, cluster H0)
- Scoring discrimination (18.6x bridge vs cluster ratio)
- Pipeline end-to-end with synthetic benchmarks
- Performance budgets (<1s for 180-token persistence + scoring)

## Project Structure

```
topoli/
  config.py              # Pydantic v2 config DSL (frozen, validated)
  pruning.py             # Token pruning strategies (topo, hybrid, baseline)
  interaction.py         # MaxSim late interaction scoring
  pipeline.py            # Multi-stage retrieval pipeline
  eval.py                # MRR, NDCG, Recall metrics
  tda/
    persistence.py       # Ripser wrapper with cosine distance + subsampling
    scoring.py           # TDA token importance scoring functions
  tests/
    test_config.py       # Config validation + hypothesis
    test_persistence.py  # Topological correctness
    test_scoring.py      # Scoring accuracy + performance
    test_pruning.py      # Pruning strategies
    test_interaction.py  # MaxSim correctness
    test_pipeline.py     # Pipeline integration
    test_eval.py         # Metric correctness
    test_benchmarks.py   # End-to-end topo vs baseline
```

## Requirements

- Python 3.12+
- numpy, scipy, ripser, pydantic v2
- Dev: pytest, hypothesis, ruff, mypy

## License

Apache-2.0
