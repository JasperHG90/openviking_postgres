"""Shared fixtures.

Integration tests run against a real PostgreSQL with pgvector, started as a
throwaway container via testcontainers.  One container is shared by the whole
session; each test gets its own PostgreSQL schema inside it.

Set ``OV_POSTGRES_TEST_DSN`` to point at an existing server instead, which
skips the container entirely::

    export OV_POSTGRES_TEST_DSN=postgresql://localhost/openviking_test

If neither a container runtime nor a DSN is available, the integration tests
skip rather than fail, so the unit suite stays runnable anywhere.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

import pytest

TEST_DSN_ENV = "OV_POSTGRES_TEST_DSN"

# Official pgvector image: PostgreSQL with the extension already built in.
POSTGRES_IMAGE = os.environ.get("OV_POSTGRES_TEST_IMAGE", "pgvector/pgvector:pg17")


@pytest.fixture(scope="session")
def dsn() -> Iterator[str]:
    """Yield a connection string to a PostgreSQL that provides pgvector."""
    explicit = os.environ.get(TEST_DSN_ENV)
    if explicit:
        # This function is a generator (it yields below), so it must yield
        # here too -- a bare `return explicit` would end it without ever
        # producing a value, breaking every test that uses an external server.
        yield explicit
        return

    try:
        # Moved package in newer testcontainers; keep the old path as fallback.
        try:
            from testcontainers.community.postgres import PostgresContainer
        except ImportError:
            from testcontainers.postgres import PostgresContainer
    except ImportError:  # pragma: no cover - depends on the environment
        pytest.skip(
            "Neither $OV_POSTGRES_TEST_DSN nor testcontainers is available; "
            "run `uv sync` to install the dev dependency group"
        )

    try:
        container = PostgresContainer(POSTGRES_IMAGE, driver=None)
        container.start()
    except Exception as exc:  # pragma: no cover - depends on the environment
        pytest.skip(
            f"Could not start a PostgreSQL container ({exc.__class__.__name__}: {exc})"
        )

    try:
        yield container.get_connection_url()
    finally:
        container.stop()


@pytest.fixture(scope="session")
def _verify_pgvector(dsn: str) -> bool:
    """Fail loudly if the server cannot provide the extension."""
    import psycopg

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
            if cur.fetchone() is None:
                pytest.fail(
                    "The 'vector' extension is not available on this server. "
                    "Use the pgvector/pgvector image, or install pgvector on the "
                    "server $OV_POSTGRES_TEST_DSN points at."
                )
    return True


@pytest.fixture
def test_schema(dsn: str, _verify_pgvector: bool) -> Iterator[str]:
    """An isolated PostgreSQL schema, dropped when the test finishes."""
    import psycopg
    from psycopg import sql

    name = f"ovtest_{uuid.uuid4().hex[:12]}"
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(name)))
    try:
        yield name
    finally:
        with psycopg.connect(dsn, autocommit=True) as conn:
            conn.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(sql.Identifier(name))
            )


# Dimension the test collections declare. Must match `TEST_DIMENSION` in
# tests/test_integration.py.
TEST_DIMENSION = 8


@dataclass(frozen=True)
class _StubEmbedding:
    """The one embedding attribute the adapter reads from global config."""

    dimension: int


@dataclass(frozen=True)
class _StubConfig:
    """Stand-in for the global OpenViking config during tests."""

    embedding: _StubEmbedding


@pytest.fixture(autouse=True)
def _fixed_embedding_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the embedding dimension the adapter reads from global config.

    ``CollectionAdapter.query`` builds a random vector for filter-only queries,
    sized from ``get_openviking_config().embedding.dimension`` -- the single
    global-config read in that module. Left alone the suite would consult the
    developer's real ``~/.openviking/ov.conf``, making results depend on a file
    no test wrote. ``EmbeddingConfig.dimension`` is a read-only property, so a
    stub is used rather than a mutated copy.
    """
    from openviking.storage.vectordb_adapters import base

    monkeypatch.setattr(
        base,
        "get_openviking_config",
        lambda: _StubConfig(embedding=_StubEmbedding(dimension=TEST_DIMENSION)),
    )
