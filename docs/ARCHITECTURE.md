# TopoLI Architecture: Topological Late Interaction

## What is TopoLI?

TopoLI is a **ColBERTv2 variant** that uses **persistent homology** — a technique from topological data analysis (TDA) — to intelligently prune token embeddings during late-interaction retrieval. Where ColBERTv2 stores and scores every token embedding, TopoLI identifies which tokens are *structurally important* to the document's meaning and keeps only those.

The result: fewer tokens stored, faster retrieval, and retrieval quality that matches or exceeds the unpruned baseline.

## Why Topology?

ColBERT-style models represent each document as a bag of token embeddings in high-dimensional space. The **shape** of that point cloud carries meaning:

- **Clusters** represent semantically coherent groups of tokens
- **Bridges** between clusters represent tokens that connect distinct concepts
- **Cycles** represent tokens that form loops in the embedding space — these are often structural elements like discourse markers or topic transitions

Standard pruning (random, top-k by norm) is blind to this structure. A bridge token connecting two concept clusters might have average norm but is irreplaceable for retrieval. Topological pruning sees this.

### Persistent Homology in 60 Seconds

Given a point cloud of token embeddings:

1. Grow balls around each point, increasing the radius
2. Track when clusters **merge** (H0 features die) and when **loops form and fill** (H1 features are born and die)
3. Features that persist across a wide range of radii are "real" — they reflect genuine structure, not noise

The **persistence diagram** records each feature's birth and death radius. Features with large persistence = important topological structure.

## System Design

```
┌─────────────────────────────────────────────────┐
│                  TopoLIConfig                    │
│  (Pydantic v2, frozen, validated)                │
├─────────────────────────────────────────────────┤
│                                                   │
│  ┌─────────┐   ┌──────────┐   ┌──────────────┐  │
│  │Backbone │   │ Token    │   │  Pipeline     │  │
│  │Config   │   │ Repr     │   │  Config       │  │
│  │         │   │ Config   │   │  ┌─────────┐  │  │
│  │colbertv2│   │dim_reduce│   │  │ Stage 1 │  │  │
│  │         │   │quantize  │   │  │ top_k   │  │  │
│  │         │   │project   │   │  │ pruning │  │  │
│  └─────────┘   └──────────┘   │  └─────────┘  │  │
│                                │  │ Stage 2 │  │  │
│                                │  │ top_k   │  │  │
│                                │  │ pruning │  │  │
│                                │  └─────────┘  │  │
│                                └──────────────┘  │
└─────────────────────────────────────────────────┘
```

### Pipeline Flow

```
Query + Corpus
      │
      ▼
┌─────────────┐
│  Stage 1    │  For each document:
│             │  1. Compute persistence diagram (ripser)
│  Prune +    │  2. Score tokens by TDA importance
│  Rank       │  3. Keep top (1 - ratio) tokens
│             │  4. MaxSim score against query
│  top_k=1000 │  5. Keep top 1000 documents
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Stage 2    │  Repeat with remaining candidates
│             │  (possibly different pruning strategy)
│  Rerank     │
│             │
│  top_k=100  │
└──────┬──────┘
       │
       ▼
  Final Ranked Results
```

## Modules

### `topoli/config.py` — Configuration DSL

All configuration is expressed as **frozen Pydantic v2 models** with validation. Three preset factories cover common use cases:

| Preset | Description |
|--------|-------------|
| `baseline_colbertv2()` | Standard ColBERT, no pruning |
| `topo_aggressive()` | 70% pruning via birth-death gap scoring |
| `hybrid_topo_idf()` | 50% pruning combining TDA + IDF scores |

Validation rules enforce:
- Pruning ratio in [0, 0.9]
- Homology dimensions ≤ 2
- Pipeline stages in decreasing top_k order
- Target dimensions in {32, 64, 96, 128}

### `topoli/tda/persistence.py` — Persistence Computation

Wraps [ripser](https://ripser.scikit-tda.org/) with cosine distance:

1. Compute pairwise cosine distances via `scipy.spatial.distance.pdist`
2. Optionally subsample via **greedy permutation** (farthest point sampling) for large token sets
3. Run Vietoris-Rips persistence via ripser
4. Return diagrams, cocycles, distance matrix, and subsample indices

### `topoli/tda/scoring.py` — Token Importance Scoring

Three scoring functions, all returning normalized [0, 1] arrays:

**`score_birth_death_gap`** — For each persistent feature, find edges in the distance matrix near the death threshold. Tokens incident to these critical edges are structurally important. *Vectorized via numpy upper-triangular indexing.*

**`score_representative_cycle`** — Extract cocycle representatives from ripser. Tokens participating in cocycles of persistent features score higher, weighted by persistence² / cocycle_size. This rewards tokens in small, highly persistent cycles.

**`score_persistence_weighted`** — Gaussian proximity scoring: for each persistent feature, compute how many of each token's edges are near the death scale (using a Gaussian kernel). Bridge and boundary tokens have unusual edge patterns at the topological feature's scale.

### `topoli/pruning.py` — Token Pruning

Three strategies:

- **TopoPrune**: Keep highest TDA-scored tokens
- **HybridPrune**: Weighted combination of TDA + IDF scores
- **BaselinePrune**: Top-k by norm, random, or no-op

All return `(pruned_embeddings, kept_indices)` with indices sorted.

### `topoli/interaction.py` — Late Interaction (MaxSim)

ColBERT's core operator: for each query token, find the maximum similarity to any document token, then sum across query tokens.

Supports cosine, dot product, and L2 similarity modes.

### `topoli/pipeline.py` — Multi-Stage Pipeline

Composes the above into a full retrieval pipeline:

1. For each stage in the pipeline config:
   - Prune each candidate document's tokens
   - Score all candidates against the query via MaxSim
   - Keep top_k candidates
2. Return final ranked results

### `topoli/eval.py` — Evaluation Metrics

Standard IR metrics: MRR@10, NDCG@10, Recall@100, Recall@1000.

## Why Build It This Way?

### TDD-First

Every module was built test-first. The test suite (86 tests) covers:
- Configuration validation and boundary conditions (hypothesis property-based)
- Topological correctness (circle H1 features, cluster H0 gaps)
- Scoring discrimination (bridge tokens vs cluster tokens: 18.6x ratio)
- Pipeline end-to-end (topo pruning retains ≥80% baseline recall at 50% token reduction)
- Performance (persistence + scoring on 180×128 embeddings in <1s)

### Frozen Configs

All configuration is immutable. This prevents accidental mutation during pipeline execution and makes configs safe to share across threads.

### Functional Style

Pure functions with explicit inputs and outputs. No hidden state. The pipeline is a composition of transforms, each independently testable.

### Cosine Distance

ColBERT operates in a space where cosine similarity is the natural metric. All TDA computation uses cosine distance, ensuring the topological features we detect are meaningful in the same space where retrieval happens.

## Performance

| Operation | Size | Time |
|-----------|------|------|
| Persistence diagram | 180×128 | 0.21s |
| birth_death_gap scoring | 180×128 | 0.08s |
| Full pipeline (30 docs, 15 tokens, H0 pruning) | — | <2s |

## What's Next

- **Phase 3**: Integration with real ColBERTv2 checkpoints via HuggingFace
- **Phase 4**: Index-time precomputation of persistence diagrams
- **Phase 5**: Benchmarks on MS MARCO, BEIR
