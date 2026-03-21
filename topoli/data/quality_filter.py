"""Quality filtering for generated query-passage pairs."""

from __future__ import annotations


def compute_token_overlap(text_a: str, text_b: str) -> float:
    """Compute Jaccard token overlap between two texts."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def filter_pair(
    query: str,
    passage: str,
    min_query_tokens: int = 3,
    max_query_tokens: int = 30,
    max_overlap: float = 0.8,
) -> bool:
    """Return True if the query-passage pair passes quality checks."""
    query_tokens = len(query.split())
    if query_tokens < min_query_tokens or query_tokens > max_query_tokens:
        return False

    return compute_token_overlap(query, passage) <= max_overlap


def deduplicate_queries(queries: list[str]) -> list[str]:
    """Remove exact duplicate queries preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for q in queries:
        normalized = q.strip().lower()
        if normalized not in seen:
            seen.add(normalized)
            result.append(q)
    return result
