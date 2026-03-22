"""Tests for BM25 hard negative mining."""

from __future__ import annotations

from topoli.data.hard_negatives import BM25NegativeMiner


class TestBM25NegativeMiner:
    """BM25 mines hard negatives from a passage corpus."""

    def test_builds_index(self) -> None:
        passages = {
            "p0": "machine learning algorithms for classification",
            "p1": "deep neural networks and backpropagation",
            "p2": "cooking recipes for Italian pasta dishes",
            "p3": "natural language processing with transformers",
            "p4": "gardening tips for growing tomatoes",
        }
        miner = BM25NegativeMiner(passages)
        assert miner.corpus_size == 5

    def test_mines_negatives_for_query(self) -> None:
        passages = {
            "p0": "machine learning algorithms for classification",
            "p1": "deep neural networks and backpropagation",
            "p2": "cooking recipes for Italian pasta dishes",
            "p3": "natural language processing with transformers",
            "p4": "gardening tips for growing tomatoes",
        }
        miner = BM25NegativeMiner(passages)
        negatives = miner.mine(
            query="what is deep learning",
            positive_id="p1",
            n_negatives=3,
        )
        assert len(negatives) <= 3
        assert "p1" not in negatives

    def test_excludes_positive(self) -> None:
        passages = {
            "p0": "deep learning for computer vision",
            "p1": "deep learning neural networks",
            "p2": "deep learning NLP transformers",
        }
        miner = BM25NegativeMiner(passages)
        negatives = miner.mine(
            query="deep learning",
            positive_id="p1",
            n_negatives=5,
        )
        assert "p1" not in negatives

    def test_returns_passage_ids(self) -> None:
        passages = {
            f"p{i}": f"document number {i} about topic {i % 3}"
            for i in range(20)
        }
        miner = BM25NegativeMiner(passages)
        negatives = miner.mine(
            query="topic about documents",
            positive_id="p0",
            n_negatives=7,
        )
        assert all(neg_id.startswith("p") for neg_id in negatives)
        assert all(neg_id in passages for neg_id in negatives)

    def test_batch_mine(self) -> None:
        passages = {
            f"p{i}": f"passage about subject {i} with interesting details"
            for i in range(10)
        }
        miner = BM25NegativeMiner(passages)
        queries = [
            {"query": "interesting subject details", "passage_id": "p0"},
            {"query": "passage about subject", "passage_id": "p5"},
        ]
        results = miner.batch_mine(queries, n_negatives=3)
        assert len(results) == 2
        assert all(len(r) <= 3 for r in results)
