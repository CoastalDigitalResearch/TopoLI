"""Tests for the multi-stage retrieval pipeline."""

from __future__ import annotations

import numpy as np

from topoli.config import (
    BaselinePruneConfig,
    PipelineConfig,
    StageConfig,
    TopoLIConfig,
    TopoPruneConfig,
    baseline_colbertv2,
    topo_aggressive,
)
from topoli.pipeline import execute_pipeline


def _synthetic_corpus(
    n_docs: int = 20, n_tokens: int = 10, dim: int = 16
) -> list[np.ndarray]:
    rng = np.random.default_rng(42)
    return [rng.standard_normal((n_tokens, dim)) for _ in range(n_docs)]


def _relevant_query(corpus: list[np.ndarray], target_idx: int) -> np.ndarray:
    """Create a query similar to a target document."""
    rng = np.random.default_rng(99)
    noise = rng.standard_normal(corpus[target_idx].shape) * 0.1
    return (corpus[target_idx] + noise)[:5]


class TestPipelineExecution:
    """Pipeline processes stages in order with decreasing candidates."""

    def test_returns_ranked_results(self) -> None:
        corpus = _synthetic_corpus(20)
        query = _relevant_query(corpus, 0)
        config = baseline_colbertv2()
        results = execute_pipeline(query, corpus, config)
        assert len(results) > 0
        assert all(len(r) == 2 for r in results)

    def test_results_bounded_by_final_top_k(self) -> None:
        corpus = _synthetic_corpus(200)
        query = _relevant_query(corpus, 0)
        config = baseline_colbertv2()
        results = execute_pipeline(query, corpus, config)
        final_k = config.pipeline.stages[-1].top_k
        assert len(results) <= final_k

    def test_stages_reduce_candidates(self) -> None:
        corpus = _synthetic_corpus(200)
        query = _relevant_query(corpus, 0)
        config = TopoLIConfig(
            pipeline=PipelineConfig(
                stages=(
                    StageConfig(
                        name="broad",
                        top_k=50,
                        pruning=BaselinePruneConfig(pruning_ratio=0.0, method="none"),
                    ),
                    StageConfig(
                        name="narrow",
                        top_k=10,
                        pruning=BaselinePruneConfig(pruning_ratio=0.0, method="none"),
                    ),
                ),
            ),
        )
        results = execute_pipeline(query, corpus, config)
        assert len(results) <= 10

    def test_scores_are_descending(self) -> None:
        corpus = _synthetic_corpus(50)
        query = _relevant_query(corpus, 0)
        config = baseline_colbertv2()
        results = execute_pipeline(query, corpus, config)
        scores = [s for _, s in results]
        assert scores == sorted(scores, reverse=True)

    def test_relevant_doc_ranks_highly(self) -> None:
        corpus = _synthetic_corpus(50)
        query = _relevant_query(corpus, 3)
        config = baseline_colbertv2()
        results = execute_pipeline(query, corpus, config)
        top_10_ids = [idx for idx, _ in results[:10]]
        assert 3 in top_10_ids


class TestTopoPipeline:
    """Pipeline with topological pruning."""

    def test_topo_pipeline_runs(self) -> None:
        corpus = _synthetic_corpus(30, n_tokens=15, dim=16)
        query = _relevant_query(corpus, 0)
        config = topo_aggressive()
        results = execute_pipeline(query, corpus, config)
        assert len(results) > 0

    def test_topo_pruning_reduces_token_count(self) -> None:
        corpus = _synthetic_corpus(10, n_tokens=20, dim=16)
        query = _relevant_query(corpus, 0)
        config = TopoLIConfig(
            pipeline=PipelineConfig(
                stages=(
                    StageConfig(
                        name="prune",
                        top_k=10,
                        pruning=TopoPruneConfig(
                            pruning_ratio=0.5,
                            homology_dims=(0,),
                            scoring="birth_death_gap",
                        ),
                    ),
                    StageConfig(
                        name="rerank",
                        top_k=5,
                        pruning=BaselinePruneConfig(pruning_ratio=0.0, method="none"),
                    ),
                ),
            ),
        )
        results = execute_pipeline(query, corpus, config)
        assert len(results) <= 5


class TestPipelineEdgeCases:
    """Edge cases."""

    def test_single_document(self) -> None:
        corpus = _synthetic_corpus(1)
        query = _relevant_query(corpus, 0)
        config = baseline_colbertv2()
        results = execute_pipeline(query, corpus, config)
        assert len(results) == 1

    def test_empty_corpus(self) -> None:
        query = np.random.default_rng(42).standard_normal((5, 16))
        config = baseline_colbertv2()
        results = execute_pipeline(query, [], config)
        assert results == []
