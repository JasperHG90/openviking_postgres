"""Validated settings for the pgvector backend.

``VectorDBBackendConfig.custom_params`` is an untyped ``dict[str, Any]`` that
arrives straight from ``ov.conf``. Parsing it by hand defers every failure to
the first place a key is read, and silently ignores a typo. Everything in this
module exists so the dict becomes a validated model at the point it arrives.
"""

from __future__ import annotations

import os
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

DistanceMetric = Literal["cosine", "l2", "ip"]
IndexMethod = Literal["flat", "hnsw", "ivfflat", "auto"]
IterativeScan = Literal["off", "strict_order", "relaxed_order"]
TimezonePolicy = Literal["local", "utc"]

DSN_ENV_VARS: tuple[str, ...] = (
    "OPENVIKING_POSTGRES_DSN",
    "OPENVIKING_PG_DSN",
    "DATABASE_URL",
)

DEFAULT_KEYWORD_FIELDS: tuple[str, ...] = (
    "name",
    "description",
    "abstract",
    "tags",
    "search_tags",
)


class VectorDBConfigLike(Protocol):
    """The parts of OpenViking's vectordb config this backend reads.

    Typed structurally so the adapter does not import
    ``VectorDBBackendConfig``: any object carrying these attributes works,
    which keeps the dependency one-directional and makes the adapter testable
    with a plain stub.

    Attributes
    ----------
    name : str | None
        Collection name.
    index_name : str | None
        Index bundle name.
    url : str | None
        Fallback connection string.
    distance_metric : str
        Distance metric, unless ``custom_params.distance`` overrides it.
    sparse_weight : float
        Weight of the sparse term in hybrid scoring.
    custom_params : dict[str, Any]
        Backend-specific settings, parsed into :class:`PgVectorParams`.
    """

    name: str | None
    index_name: str | None
    url: str | None
    distance_metric: str
    sparse_weight: float
    custom_params: dict[str, Any]


class PgVectorParams(BaseModel):
    """Backend settings read from ``storage.vectordb.custom_params``.

    Unknown keys are rejected rather than ignored, so a misspelled option
    fails at startup instead of silently leaving the default in place.

    Attributes
    ----------
    dsn : str | None
        libpq connection string. When absent, the caller falls back to the
        parent config's ``url`` field and then to the environment.
    db_schema : str
        PostgreSQL schema holding collection tables, registry tables, and the
        helper functions. Accepted under the config key ``schema``.
    table_prefix : str
        Prefix applied to each collection's table name.
    index_method : IndexMethod
        ``flat`` for exact search with no ANN index, ``hnsw`` or ``ivfflat``
        for approximate search, ``auto`` to follow the requested index type.
    index_options : dict[str, Any]
        Passed through to ``CREATE INDEX ... WITH (...)``.
    create_extension : bool
        Whether to run ``CREATE EXTENSION IF NOT EXISTS vector`` at startup.
        Set False on managed PostgreSQL where the role cannot create
        extensions and an administrator has installed it already.
    iterative_scan : IterativeScan
        Recovery mode when an ANN index under-returns beneath a selective
        filter. Ignored for exact search and on pgvector below 0.8.
    distance : DistanceMetric | None
        Overrides the parent config's ``distance_metric`` when set.
    keyword_fields : list[str] | None
        Text columns included in the full-text index. ``None`` selects
        :data:`DEFAULT_KEYWORD_FIELDS`.
    text_search_config : str
        PostgreSQL text search configuration used to build tsvectors.
    tz_policy : TimezonePolicy
        Timezone applied to naive timestamps. Must match OpenViking's own
        setting or the two backends will store different epoch values.
    min_pool_size : int
        Connections held open by the pool.
    max_pool_size : int
        Upper bound on pooled connections.
    connect_timeout : float
        Seconds to wait for a connection from the pool.
    application_name : str
        Reported to PostgreSQL as ``application_name``.
    """

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    dsn: str | None = Field(
        default=None, description="libpq connection string for the target database."
    )
    db_schema: str = Field(
        default="public",
        alias="schema",
        min_length=1,
        description="PostgreSQL schema holding tables and helper functions.",
    )
    table_prefix: str = Field(
        default="ov_", description="Prefix applied to collection table names."
    )
    index_method: IndexMethod = Field(
        default="flat",
        description="Vector index strategy; 'flat' means exact search.",
    )
    index_options: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra options for CREATE INDEX ... WITH (...).",
    )
    create_extension: bool = Field(
        default=True,
        description=(
            "Run CREATE EXTENSION IF NOT EXISTS vector at startup. Set false on "
            "managed PostgreSQL where the role lacks the privilege."
        ),
    )
    iterative_scan: IterativeScan = Field(
        default="relaxed_order",
        description=(
            "How an ANN index recovers when a selective filter leaves it short "
            "of the requested row count. Needs pgvector 0.8 or newer."
        ),
    )
    distance: DistanceMetric | None = Field(
        default=None, description="Overrides the parent config's distance_metric."
    )
    keyword_fields: list[str] | None = Field(
        default=None, description="Text columns to include in the full-text index."
    )
    text_search_config: str = Field(
        default="simple",
        min_length=1,
        description="PostgreSQL text search configuration for tsvectors.",
    )
    tz_policy: TimezonePolicy = Field(
        default="local", description="Timezone applied to naive timestamps."
    )
    min_pool_size: int = Field(
        default=1, ge=0, description="Connections held open by the pool."
    )
    max_pool_size: int = Field(
        default=8, ge=1, description="Upper bound on pooled connections."
    )
    connect_timeout: float = Field(
        default=10.0, gt=0, description="Seconds to wait for a pooled connection."
    )
    application_name: str = Field(
        default="openviking", description="Reported to PostgreSQL as application_name."
    )

    @model_validator(mode="after")
    def _check_pool_bounds(self) -> PgVectorParams:
        """Reject a pool whose minimum exceeds its maximum."""
        if self.min_pool_size > self.max_pool_size:
            raise ValueError(
                f"min_pool_size ({self.min_pool_size}) cannot exceed "
                f"max_pool_size ({self.max_pool_size})"
            )
        return self

    def resolved_keyword_fields(self) -> list[str]:
        """Return the configured keyword fields, or the defaults.

        Returns
        -------
        list[str]
            Field names to include in the full-text index.
        """
        if self.keyword_fields is None:
            return list(DEFAULT_KEYWORD_FIELDS)
        return list(self.keyword_fields)


def resolve_dsn(params: PgVectorParams, url: str | None) -> str:
    """Find the connection string, preferring explicit configuration.

    Order is ``custom_params.dsn``, then the parent config's ``url``, then the
    environment. Keeping the DSN in an environment variable avoids writing a
    database password into ``ov.conf``.

    Parameters
    ----------
    params :
        Parsed backend settings.
    url :
        The parent ``VectorDBBackendConfig.url``, if set.

    Returns
    -------
    str
        A libpq connection string.

    Raises
    ------
    ValueError
        If no connection string is configured anywhere.
    """
    if params.dsn:
        return params.dsn
    if url:
        return url
    for name in DSN_ENV_VARS:
        value = os.environ.get(name)
        if value:
            return value
    raise ValueError(
        "pgvector backend requires a connection string. Set "
        "storage.vectordb.custom_params.dsn, storage.vectordb.url, or one of "
        f"these environment variables: {', '.join(DSN_ENV_VARS)}"
    )
