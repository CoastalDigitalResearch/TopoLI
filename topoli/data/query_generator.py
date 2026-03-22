"""Doc2query generation using open-weight LLMs.

Generates diverse search queries from passages using permissively-licensed
models (Qwen3.5-9B Apache-2.0). Multiple prompt templates cover different
search intents for richer training signal.
"""

from __future__ import annotations

import random

from pydantic import BaseModel, ConfigDict

from topoli.data.source_config import License

QUERY_TEMPLATES: tuple[str, ...] = (
    (
        "Given the following passage, write a natural search engine query "
        "that a user would type to find this information. "
        "Output ONLY the query, nothing else.\n\n"
        "Passage: {passage}\n\nQuery:"
    ),
    (
        "A user wants to find the following information online. "
        "Write the search query they would type into Google. "
        "Output ONLY the query.\n\n"
        "Information: {passage}\n\nSearch query:"
    ),
    (
        "Read this passage and write a question that it answers. "
        "The question should be specific and natural-sounding. "
        "Output ONLY the question.\n\n"
        "Passage: {passage}\n\nQuestion:"
    ),
    (
        "What factual question does this passage answer? "
        "Write a clear, concise question. "
        "Output ONLY the question.\n\n"
        "Passage: {passage}\n\nQuestion:"
    ),
    (
        "Imagine someone needs the information in this passage. "
        "Write the keyword search they would use. "
        "Output ONLY the search terms.\n\n"
        "Passage: {passage}\n\nKeywords:"
    ),
    (
        "Write a how-to or explanatory query that this passage "
        "would be a good result for. "
        "Output ONLY the query.\n\n"
        "Passage: {passage}\n\nQuery:"
    ),
)

RELEVANCE_SCORING_PROMPT = (
    "Rate how well this search query matches this passage on a scale of 1-5.\n"
    "1 = completely irrelevant\n"
    "3 = somewhat relevant\n"
    "5 = perfectly relevant\n\n"
    "Query: {query}\n"
    "Passage: {passage}\n\n"
    "Score (1-5):"
)


class QueryGeneratorConfig(BaseModel):
    """Configuration for doc2query generation."""

    model_config = ConfigDict(frozen=True)

    model_name: str = "Qwen/Qwen3-8B"
    model_license: License = License.APACHE_2_0
    max_new_tokens: int = 64
    temperature: float = 0.7
    min_query_tokens: int = 5
    max_query_tokens: int = 30
    batch_size: int = 512
    queries_per_passage: int = 2


def build_prompt(passage_text: str, template_idx: int | None = None) -> str:
    """Build a doc2query prompt using a random or specified template."""
    if template_idx is None:
        template_idx = random.randint(0, len(QUERY_TEMPLATES) - 1)  # noqa: S311
    template = QUERY_TEMPLATES[template_idx % len(QUERY_TEMPLATES)]
    return template.format(passage=passage_text)


def build_scoring_prompt(query: str, passage: str) -> str:
    """Build a relevance scoring prompt for quality filtering."""
    return RELEVANCE_SCORING_PROMPT.format(query=query, passage=passage)


def parse_query_response(
    response: str,
    min_tokens: int = 5,
    max_tokens: int = 30,
) -> str | None:
    """Parse and validate LLM response into a clean query."""
    text = response.strip()
    if not text:
        return None

    text = text.split("\n")[0].strip()

    for prefix in (
        "Query:",
        "query:",
        "Q:",
        "Search:",
        "Search query:",
        "Question:",
        "question:",
        "Keywords:",
        "keywords:",
    ):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()

    if not text:
        return None

    token_count = len(text.split())
    if token_count < min_tokens or token_count > max_tokens:
        return None

    return text


def parse_relevance_score(response: str) -> float | None:
    """Parse a 1-5 relevance score from model output."""
    text = response.strip()
    for char in text:
        if char.isdigit():
            score = int(char)
            if 1 <= score <= 5:
                return float(score)
    return None
