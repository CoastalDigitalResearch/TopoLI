"""BM25 hard negative mining for retrieval training."""

from __future__ import annotations

import logging

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25NegativeMiner:
    """Mines hard negatives from a passage corpus using BM25."""

    def __init__(self, passages: dict[str, str]) -> None:
        self._ids = list(passages.keys())
        self._texts = list(passages.values())
        tokenized = [text.lower().split() for text in self._texts]
        self._bm25 = BM25Okapi(tokenized)
        self._id_to_idx = {pid: i for i, pid in enumerate(self._ids)}
        logger.info("Built BM25 index over %d passages", len(self._ids))

    @property
    def corpus_size(self) -> int:
        return len(self._ids)

    def mine(
        self,
        query: str,
        positive_id: str,
        n_negatives: int = 7,
        min_rank: int = 1,
        max_rank: int = 200,
    ) -> list[str]:
        """Mine hard negatives for a query, excluding the positive."""
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        ranked_indices = scores.argsort()[::-1]

        negatives: list[str] = []
        for idx in ranked_indices[min_rank:max_rank]:
            pid = self._ids[idx]
            if pid == positive_id:
                continue
            negatives.append(pid)
            if len(negatives) >= n_negatives:
                break

        return negatives

    def batch_mine(
        self,
        queries: list[dict[str, str]],
        n_negatives: int = 7,
    ) -> list[list[str]]:
        """Mine hard negatives for a batch of queries."""
        results: list[list[str]] = []
        for q in queries:
            negs = self.mine(
                query=q["query"],
                positive_id=q["passage_id"],
                n_negatives=n_negatives,
            )
            results.append(negs)
        return results
