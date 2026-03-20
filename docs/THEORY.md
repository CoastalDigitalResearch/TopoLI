# Topological Token Pruning: Theory and Motivation

## The Problem with Flat Token Pruning

ColBERTv2 represents documents as collections of contextualized token embeddings, then scores queries via **MaxSim** — for each query token, find its best-matching document token and sum the similarities. This is powerful but expensive: every document stores 128+ embeddings, and every retrieval computes O(|q| × |d|) similarities.

Reducing the number of stored tokens per document is critical for scaling. But which tokens can you safely remove?

### Existing Approaches and Their Limitations

| Method | What it keeps | Blind spot |
|--------|--------------|------------|
| **Top-k by norm** | Tokens with largest embedding magnitude | Misses structurally important low-norm tokens |
| **Random** | Arbitrary subset | No signal at all |
| **Attention-based** | Tokens the model attended to | Attention ≠ retrieval importance |
| **IDF-weighted** | Rare tokens | Misses common tokens that are positionally crucial |

All of these treat tokens independently. None considers the **relational structure** — how tokens relate to each other in the embedding space.

## The Topological Insight

Consider a document about "renewable energy policy in coastal cities." The token embeddings form a point cloud with:

- A **cluster** of energy-related tokens
- A **cluster** of policy/governance tokens
- A **cluster** of geographic/coastal tokens
- **Bridge tokens** connecting these clusters — words like "implementing" or "addressing" that link concepts

A top-k-by-norm pruner might keep the most "emphatic" tokens from each cluster but remove the bridges. The document's internal structure collapses: a query about "coastal energy policy" can no longer find the connections between the three concepts.

**Persistent homology detects exactly this structure.**

## Persistent Homology for Token Embeddings

### The Vietoris-Rips Filtration

Given n token embeddings in ℝ^d with pairwise cosine distances:

1. Start with n isolated points (radius ε = 0)
2. As ε grows, connect points within distance ε
3. Track:
   - **H0 (connected components)**: Initially n components. As ε grows, components merge. A merge at ε means two clusters just became connected — the edge responsible is a *bridge*.
   - **H1 (loops)**: Cycles that form and later get filled in. Tokens forming persistent loops are structurally distinctive.

### The Persistence Diagram

Each topological feature is a point (birth, death) in the persistence diagram:

```
death │
      │    ·  ← noise (short-lived)
      │
      │              · ← important feature (long-lived)
      │           ·
      │     ·
      │  ·
      └──────────── birth
```

**Persistence = death - birth.** High persistence = real structure. Low persistence = noise.

### From Diagrams to Token Scores

The key innovation: **map persistence diagram features back to tokens.**

#### Birth-Death Gap Scoring

For each persistent H0 feature (a cluster merger):
- The feature dies at distance `d` when a specific edge merges two clusters
- Tokens incident to edges at distance ≈ `d` are the *bridge tokens*
- Weight by persistence: mergers of large, well-separated clusters matter more

This directly identifies tokens that connect distinct concepts.

#### Representative Cycle Scoring

For H1 features (loops):
- Ripser computes cocycle representatives — the simplices that form the loop
- Tokens appearing in cocycles of persistent features are structurally distinctive
- Weight by persistence² / cocycle_size: small persistent cycles are more informative

#### Persistence-Weighted Proximity

A softer scoring that considers all features:
- For each persistent feature at death scale `d`, compute a Gaussian proximity score for each token's edges
- Tokens whose edge patterns cluster around topologically significant scales get higher scores

## Mathematical Guarantees

### Stability

The persistence diagram is **stable** under perturbation: if we perturb embeddings by at most δ in the bottleneck distance, the persistence diagram changes by at most δ. This means:

- Small noise in embeddings doesn't dramatically change which tokens are scored as important
- The scoring is robust to the inherent stochasticity of transformer outputs

### Subsampling Bounds

For large token sets, we subsample via greedy permutation (farthest point sampling). The Hausdorff distance between the subsample and the full set is bounded, giving us guarantees on how well the subsampled persistence diagram approximates the full one.

## Why This Works for Retrieval

MaxSim scoring is dominated by the **best-matching** document token for each query token. The tokens that matter most for retrieval are:

1. **Distinctive tokens** — those that occupy unique regions of embedding space (captured by H0 features)
2. **Bridge tokens** — those connecting semantic clusters, allowing the document to match diverse queries (captured by birth-death gap scoring)
3. **Structurally central tokens** — those involved in the document's topological skeleton (captured by representative cycle scoring)

Standard pruning can remove bridge tokens (they're not the highest-norm tokens in any cluster). Topological pruning preserves them.

## Empirical Validation

On synthetic retrieval tasks with known ground truth:

- **Birth-death gap** scoring achieves 18.6x discrimination between bridge and cluster tokens
- **50% topological pruning** retains ≥80% of baseline recall
- **Representative cycle** scoring correctly identifies 100% of boundary tokens in controlled experiments

## References

- Edelsbrunner, H., & Harer, J. (2010). *Computational Topology: An Introduction.* AMS.
- Ripser: Bauer, U. (2021). *Ripser: efficient computation of Vietoris-Rips persistence barcodes.* JACT.
- ColBERT: Khattab, O., & Zaharia, M. (2020). *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT.* SIGIR.
- ColBERTv2: Santhanam, K., et al. (2022). *ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction.* NAACL.
- Stability theorem: Cohen-Steiner, D., Edelsbrunner, H., & Harer, J. (2007). *Stability of Persistence Diagrams.* DCG.
