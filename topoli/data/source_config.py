"""Data source configuration and provenance for TopoLI-Retrieval.

Every data source used in training is registered here with its exact license.
Only unambiguously commercially-usable licenses are permitted:
Apache-2.0, MIT, CC-BY, CC0, ODC-By, Public Domain.

NO CC-BY-SA. NO non-commercial. NO ambiguity.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field


class License(Enum):
    """Permitted licenses — all commercially usable, no ShareAlike."""

    APACHE_2_0 = "Apache-2.0"
    MIT = "MIT"
    CC_BY_4_0 = "CC-BY-4.0"
    CC_BY_3_0 = "CC-BY-3.0"
    CC0 = "CC0-1.0"
    ODC_BY = "ODC-By-1.0"
    PUBLIC_DOMAIN = "Public Domain"
    BSD_3 = "BSD-3-Clause"
    BSD_2 = "BSD-2-Clause"


class SourceDomain(Enum):
    """Content domain categories."""

    GOVERNMENT = "government"
    LEGAL = "legal"
    SCIENCE = "science"
    CULTURE = "culture"
    GENERAL = "general"
    QA = "qa"
    RETRIEVAL = "retrieval"


class DataSourceConfig(BaseModel):
    """Configuration for a single data source with license provenance."""

    model_config = ConfigDict(frozen=True)

    name: str
    huggingface_id: str
    license: License
    domain: SourceDomain
    description: str
    url: str = ""

    @property
    def commercially_usable(self) -> bool:
        return True


class PassageRecord(BaseModel):
    """A single passage with full provenance chain."""

    model_config = ConfigDict(frozen=True)

    passage_id: str
    text: str
    source_name: str
    source_license: License
    source_doc_id: str
    char_count: Annotated[int, Field(gt=0)]


class QueryRecord(BaseModel):
    """A generated query with provenance."""

    model_config = ConfigDict(frozen=True)

    query_id: str
    text: str
    passage_id: str
    generator_model: str
    generator_license: License


class RetrievalPair(BaseModel):
    """A query-passage pair with relevance and provenance."""

    model_config = ConfigDict(frozen=True)

    query_id: str
    passage_id: str
    relevance: Annotated[float, Field(ge=0.0, le=1.0)]
    source_name: str
    source_license: License


def get_source_registry() -> tuple[DataSourceConfig, ...]:
    """Return all verified permissive data sources for TopoLI-Retrieval.

    ALL sources commercially usable. NO CC-BY-SA. NO non-commercial.
    """
    return (
        # === Existing QA datasets (Apache-2.0) ===
        DataSourceConfig(
            name="triviaqa",
            huggingface_id="mandarjoshi/trivia_qa",
            license=License.APACHE_2_0,
            domain=SourceDomain.QA,
            description="95K factoid QA pairs from trivia questions",
            url="https://github.com/mandarjoshi90/triviaqa",
        ),
        # === Open License Corpus (Apache-2.0 compilation) ===
        # SA subsets (StackExchange, Wikipedia, Wikinews) excluded
        DataSourceConfig(
            name="olc_legal",
            huggingface_id="kernelmachine/open-license-corpus",
            license=License.APACHE_2_0,
            domain=SourceDomain.LEGAL,
            description=(
                "Pile of Law + Case Law Access Project. "
                "6.5M court decisions, public domain / CC-BY."
            ),
            url="https://huggingface.co/datasets/kernelmachine/open-license-corpus",
        ),
        DataSourceConfig(
            name="olc_science",
            huggingface_id="kernelmachine/open-license-corpus",
            license=License.APACHE_2_0,
            domain=SourceDomain.SCIENCE,
            description=("ArXiv abstracts (public domain) + S2ORC CC-BY papers."),
        ),
        DataSourceConfig(
            name="olc_books",
            huggingface_id="kernelmachine/open-license-corpus",
            license=License.PUBLIC_DOMAIN,
            domain=SourceDomain.CULTURE,
            description="Project Gutenberg public domain books.",
        ),
        DataSourceConfig(
            name="olc_math",
            huggingface_id="kernelmachine/open-license-corpus",
            license=License.APACHE_2_0,
            domain=SourceDomain.SCIENCE,
            description="DeepMind Mathematics + AMPS datasets (Apache).",
        ),
        DataSourceConfig(
            name="olc_conversation",
            huggingface_id="kernelmachine/open-license-corpus",
            license=License.APACHE_2_0,
            domain=SourceDomain.GENERAL,
            description=(
                "HackerNews (MIT) + Ubuntu IRC (Apache). "
                "Excludes CC-BY-SA StackExchange."
            ),
        ),
        # === SlimPajama (Apache-2.0) ===
        DataSourceConfig(
            name="slimpajama",
            huggingface_id="DKYoon/SlimPajama-6B",
            license=License.APACHE_2_0,
            domain=SourceDomain.GENERAL,
            description="General web text. Apache-2.0.",
        ),
        # === Pre-training only ===
        DataSourceConfig(
            name="ettin_pretraining",
            huggingface_id="jhu-clsp/ettin-pretraining-data",
            license=License.MIT,
            domain=SourceDomain.GENERAL,
            description="1.7T tokens of curated pre-training data. MIT license.",
        ),
    )
