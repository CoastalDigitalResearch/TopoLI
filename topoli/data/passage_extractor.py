"""Extract and chunk passages from permissive-license corpora."""

from __future__ import annotations

import hashlib
import re

from topoli.data.source_config import License, PassageRecord


def clean_text(text: str) -> str:
    """Remove noise while preserving meaningful content."""
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n", text)
    return text.strip()


def _overlap_start(items: list[str], overlap_chars: int) -> int:
    """Find the start index to keep overlap_chars worth of trailing items."""
    collected = 0
    start = len(items)
    for i in range(len(items) - 1, -1, -1):
        collected += len(items[i])
        start = i
        if collected >= overlap_chars:
            break
    return start


def chunk_document(
    text: str,
    min_chars: int = 200,
    max_chars: int = 600,
    overlap_chars: int = 50,
) -> list[str]:
    """Split document into overlapping passages respecting sentence boundaries."""
    if len(text) < min_chars:
        return []

    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r"(?<=[.!?])\s+", text)

    if len(sentences) == 1 and len(text) > max_chars:
        return _chunk_by_words(text, min_chars, max_chars, overlap_chars)

    return _chunk_by_sentences(sentences, min_chars, max_chars, overlap_chars)


def _chunk_by_sentences(
    sentences: list[str],
    min_chars: int,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    """Chunk a list of sentences into passages with overlap."""
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > max_chars and current:
            chunk = " ".join(current)
            if len(chunk) >= min_chars:
                chunks.append(chunk)
            idx = _overlap_start(current, overlap_chars)
            current = current[idx:]
            current_len = sum(len(s) for s in current) + max(0, len(current) - 1)

        current.append(sentence)
        current_len += len(sentence) + (1 if len(current) > 1 else 0)

    if current:
        chunk = " ".join(current)
        if len(chunk) >= min_chars:
            chunks.append(chunk)

    return chunks


def _chunk_by_words(
    text: str,
    min_chars: int,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    """Fallback chunker that splits on word boundaries."""
    words = text.split()
    chunks: list[str] = []
    start = 0

    while start < len(words):
        current: list[str] = []
        current_len = 0

        i = start
        while i < len(words) and current_len + len(words[i]) + 1 <= max_chars:
            current.append(words[i])
            current_len += len(words[i]) + (1 if current_len > 0 else 0)
            i += 1

        chunk = " ".join(current)
        if len(chunk) >= min_chars:
            chunks.append(chunk)

        if not current:
            break

        # Advance with overlap, but always move forward by at least 1 word
        overlap_words = 0
        overlap_len = 0
        for j in range(len(current) - 1, -1, -1):
            overlap_len += len(current[j])
            overlap_words += 1
            if overlap_len >= overlap_chars:
                break
        advance = max(1, len(current) - overlap_words)
        start = start + advance

    return chunks


def _passage_id(source_name: str, doc_id: str, chunk_idx: int) -> str:
    """Generate deterministic passage ID."""
    raw = f"{source_name}:{doc_id}:{chunk_idx}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def extract_passages(
    documents: list[dict[str, str]],
    source_name: str,
    source_license: License,
    min_chars: int = 200,
    max_chars: int = 600,
) -> list[PassageRecord]:
    """Extract passage records with full provenance from a list of documents."""
    records: list[PassageRecord] = []
    for doc in documents:
        text = clean_text(doc["text"])
        doc_id = doc.get("doc_id", "unknown")
        chunks = chunk_document(text, min_chars=min_chars, max_chars=max_chars)

        for i, chunk in enumerate(chunks):
            pid = _passage_id(source_name, doc_id, i)
            records.append(
                PassageRecord(
                    passage_id=pid,
                    text=chunk,
                    source_name=source_name,
                    source_license=source_license,
                    source_doc_id=f"{doc_id}:chunk_{i}",
                    char_count=len(chunk),
                )
            )

    return records
