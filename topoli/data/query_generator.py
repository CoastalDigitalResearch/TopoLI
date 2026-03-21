"""Doc2query generation using open-weight LLMs.

Generates search queries from passages using permissively-licensed models
(Qwen3-8B Apache-2.0). All generated queries inherit the passage source
license for provenance tracking.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from topoli.data.source_config import License

DOC2QUERY_PROMPT = (
    "Given the following passage, write a natural search engine query that "
    "a user would type to find this information. Output ONLY the query, "
    "nothing else.\n\n"
    "Passage: {passage}\n\n"
    "Query:"
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
    batch_size: int = 32


def build_prompt(passage_text: str) -> str:
    """Build the doc2query prompt for a given passage."""
    return DOC2QUERY_PROMPT.format(passage=passage_text)


def parse_query_response(
    response: str,
    min_tokens: int = 5,
    max_tokens: int = 30,
) -> str | None:
    """Parse and validate LLM response into a clean query."""
    text = response.strip()
    if not text:
        return None

    # Take first line only
    text = text.split("\n")[0].strip()

    # Strip common prefixes
    for prefix in ("Query:", "query:", "Q:", "Search:"):
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()

    if not text:
        return None

    token_count = len(text.split())
    if token_count < min_tokens or token_count > max_tokens:
        return None

    return text
