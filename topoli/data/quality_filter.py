"""Quality filtering for passages and query-passage pairs."""

from __future__ import annotations

import hashlib
import re

STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "shall",
        "can",
        "need",
        "dare",
        "ought",
        "to",
        "of",
        "in",
        "for",
        "on",
        "with",
        "at",
        "by",
        "from",
        "as",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "and",
        "but",
        "or",
        "nor",
        "not",
        "so",
        "yet",
        "both",
        "either",
        "neither",
        "each",
        "every",
        "all",
        "any",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "no",
        "only",
        "own",
        "same",
        "than",
        "too",
        "very",
        "just",
        "because",
        "if",
        "when",
        "while",
        "where",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "i",
        "you",
        "he",
        "she",
        "we",
        "they",
        "me",
        "him",
        "her",
        "us",
        "them",
        "my",
        "your",
    }
)


def compute_token_overlap(text_a: str, text_b: str) -> float:
    """Compute Jaccard token overlap between two texts."""
    tokens_a = set(text_a.lower().split())
    tokens_b = set(text_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def filter_passage(
    text: str,
    min_chars: int = 100,
    max_chars: int = 2000,
    max_stopword_ratio: float = 0.70,
    max_repeat_ratio: float = 0.50,
) -> bool:
    """Return True if passage passes quality checks."""
    if len(text) < min_chars or len(text) > max_chars:
        return False

    words = text.lower().split()
    if not words:
        return False

    stopword_count = sum(1 for w in words if w in STOPWORDS)
    if stopword_count / len(words) > max_stopword_ratio:
        return False

    unique_words = set(words)
    if len(unique_words) / len(words) < (1.0 - max_repeat_ratio):
        return False

    return not re.search(
        r"(cookie|privacy policy|terms of service|click here|subscribe)",
        text.lower(),
    )


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


def minhash_fingerprint(text: str, n_shingles: int = 3) -> str:
    """Compute a simple MinHash-style fingerprint for near-dedup."""
    words = text.lower().split()
    if len(words) < n_shingles:
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    shingles = {
        " ".join(words[i : i + n_shingles]) for i in range(len(words) - n_shingles + 1)
    }
    min_hash = min(
        hashlib.sha256(s.encode()).hexdigest()
        for s in shingles
    )
    return min_hash[:16]


def deduplicate_passages_by_content(
    passages: list[dict[str, str]],
    text_key: str = "text",
) -> list[dict[str, str]]:
    """Remove near-duplicate passages using MinHash fingerprints."""
    seen: set[str] = set()
    result: list[dict[str, str]] = []
    for p in passages:
        fp = minhash_fingerprint(p[text_key])
        if fp not in seen:
            seen.add(fp)
            result.append(p)
    return result
