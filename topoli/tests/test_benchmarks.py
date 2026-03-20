"""End-to-end benchmarks: topo pruning vs baseline."""

from __future__ import annotations

import time

import numpy as np

from topoli.config import (
    BaselinePruneConfig,
    PipelineConfig,
    StageConfig,
    TopoLIConfig,
    TopoPruneConfig,
)
from topoli.eval import evaluate_retrieval
from topoli.pipeline import execute_pipeline


def _synthetic_retrieval_corpus(
    n_docs: int = 50,
    n_tokens: int = 15,
    dim: int = 16,
    n_relevant: int = 3,
) -> tuple[
    np.ndarray,
    list[np.ndarray],
    set[int],
]:
    """Build a synthetic corpus with known relevant documents."""
    rng = np.random.default_rng(42)
    query_base = rng.standard_normal((5, dim))

    docs: list[np.ndarray] = []
    relevant_ids: set[int] = set()

    for i in range(n_docs):
        if i < n_relevant:
            noise = rng.standard_normal((n_tokens, dim)) * 0.3
            base = np.tile(query_base[:1], (n_tokens, 1))
            docs.append(base + noise)
            relevant_ids.add(i)
        else:
            docs.append(rng.standard_normal((n_tokens, dim)))

    return query_base, docs, relevant_ids


class TestBaselineVsTopo:
    """Compare baseline and topo pruning on synthetic data."""

    def test_topo_retains_retrieval_quality(self) -> None:
        query, corpus, qrels = _synthetic_retrieval_corpus()

        baseline_config = TopoLIConfig(
            pipeline=PipelineConfig(
                stages=(
                    StageConfig(
                        name="retrieve",
                        top_k=50,
                        pruning=BaselinePruneConfig(pruning_ratio=0.0, method="none"),
                    ),
                    StageConfig(
                        name="rerank",
                        top_k=10,
                        pruning=BaselinePruneConfig(pruning_ratio=0.0, method="none"),
                    ),
                ),
            ),
        )

        topo_config = TopoLIConfig(
            pipeline=PipelineConfig(
                stages=(
                    StageConfig(
                        name="topo_prune",
                        top_k=50,
                        pruning=TopoPruneConfig(
                            pruning_ratio=0.5,
                            homology_dims=(0,),
                            scoring="birth_death_gap",
                        ),
                    ),
                    StageConfig(
                        name="rerank",
                        top_k=10,
                        pruning=BaselinePruneConfig(pruning_ratio=0.0, method="none"),
                    ),
                ),
            ),
        )

        baseline_results = execute_pipeline(query, corpus, baseline_config)
        topo_results = execute_pipeline(query, corpus, topo_config)

        baseline_metrics = evaluate_retrieval(baseline_results, qrels)
        topo_metrics = evaluate_retrieval(topo_results, qrels)

        assert topo_metrics["recall@100"] >= baseline_metrics["recall@100"] * 0.8
        assert topo_metrics["mrr@10"] > 0.0

    def test_topo_pruning_50pct_finds_relevant(self) -> None:
        query, corpus, qrels = _synthetic_retrieval_corpus()
        config = TopoLIConfig(
            pipeline=PipelineConfig(
                stages=(
                    StageConfig(
                        name="prune",
                        top_k=50,
                        pruning=TopoPruneConfig(
                            pruning_ratio=0.5,
                            homology_dims=(0,),
                            scoring="birth_death_gap",
                        ),
                    ),
                    StageConfig(
                        name="final",
                        top_k=10,
                        pruning=BaselinePruneConfig(pruning_ratio=0.0, method="none"),
                    ),
                ),
            ),
        )
        results = execute_pipeline(query, corpus, config)
        metrics = evaluate_retrieval(results, qrels)
        assert metrics["recall@100"] >= 0.5

    def test_pipeline_completes_in_reasonable_time(self) -> None:
        query, corpus, _qrels = _synthetic_retrieval_corpus(
            n_docs=30, n_tokens=15, dim=16
        )
        config = TopoLIConfig(
            pipeline=PipelineConfig(
                stages=(
                    StageConfig(
                        name="prune",
                        top_k=30,
                        pruning=TopoPruneConfig(
                            pruning_ratio=0.3,
                            homology_dims=(0,),
                            scoring="birth_death_gap",
                        ),
                    ),
                    StageConfig(
                        name="rerank",
                        top_k=10,
                        pruning=BaselinePruneConfig(pruning_ratio=0.0, method="none"),
                    ),
                ),
            ),
        )
        t0 = time.monotonic()
        execute_pipeline(query, corpus, config)
        elapsed = time.monotonic() - t0
        assert elapsed < 10.0
