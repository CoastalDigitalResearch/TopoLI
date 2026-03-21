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
    """Return all verified permissive data sources for TopoLI-Retrieval."""
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
        DataSourceConfig(
            name="miracl_en",
            huggingface_id="miracl/miracl",
            license=License.APACHE_2_0,
            domain=SourceDomain.RETRIEVAL,
            description="32K English retrieval queries with human annotations",
            url="https://project-miracl.github.io/",
        ),
        # === Common Corpus subsets (CC-BY, CC0, public domain) ===
        DataSourceConfig(
            name="common_corpus_government",
            huggingface_id="PleIAs/Common-Corpus",
            license=License.CC0,
            domain=SourceDomain.GOVERNMENT,
            description=(
                "US/EU government documents, legislation, regulations. "
                "406B tokens. Public domain / CC0."
            ),
        ),
        DataSourceConfig(
            name="common_corpus_science",
            huggingface_id="PleIAs/Common-Corpus",
            license=License.CC_BY_4_0,
            domain=SourceDomain.SCIENCE,
            description=(
                "ArXiv CC-BY papers, PubMed Central OA CC-BY subset. 281B tokens."
            ),
        ),
        DataSourceConfig(
            name="common_corpus_culture",
            huggingface_id="PleIAs/Common-Corpus",
            license=License.PUBLIC_DOMAIN,
            domain=SourceDomain.CULTURE,
            description=("Project Gutenberg, public domain books. 886B tokens."),
        ),
        # === KL3M (CC-BY 4.0, copyright-verified) ===
        DataSourceConfig(
            name="kl3m_government",
            huggingface_id="alea-institute/kl3m-data-govinfo-crec",
            license=License.CC_BY_4_0,
            domain=SourceDomain.GOVERNMENT,
            description=(
                "Federal Register, CFR, Congressional Record. "
                "Copyright-clean, legally verified."
            ),
            url="https://github.com/alea-institute/kl3m-data",
        ),
        DataSourceConfig(
            name="kl3m_legal",
            huggingface_id="alea-institute/kl3m-data-caselaw",
            license=License.CC_BY_4_0,
            domain=SourceDomain.LEGAL,
            description=(
                "US case law, court opinions, SEC filings. "
                "Copyright-clean, legally verified."
            ),
            url="https://github.com/alea-institute/kl3m-data",
        ),
        # === SlimPajama (Apache-2.0) ===
        DataSourceConfig(
            name="slimpajama",
            huggingface_id="cerebras/SlimPajama-627B",
            license=License.APACHE_2_0,
            domain=SourceDomain.GENERAL,
            description="627B tokens of cleaned web text. Apache-2.0.",
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
