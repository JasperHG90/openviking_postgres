"""Integration coverage for the configurable code paths.

The main integration suite exercises the default configuration (exact search,
cosine distance, no sparse vectors).  These tests cover the options that change
which SQL gets generated: ANN index methods, distance metrics, and hybrid
dense+sparse scoring.
"""

from __future__ import annotations

import contextlib
import copy
import math
import threading
from typing import Any, cast

import psycopg
import pytest
from openviking.storage.vectordb.index.cuvs_index import matches_filter
from openviking_cli.utils.config.vectordb_config import VectorDBBackendConfig
from psycopg import sql
from pydantic import ValidationError

from ov_postgres import ddl
from ov_postgres.adapter import PgVectorCollectionAdapter
from ov_postgres.collection import NULL_BUCKET
from ov_postgres.filters import UnsupportedFilterError
from ov_postgres.schema import CollectionSchema

from .test_integration import (
    DIM,
    FIELD_TYPES,
    META,
    as_reference_row,
    vec,
)

pytestmark = pytest.mark.integration


def build(
    dsn: str,
    schema: str,
    *,
    sparse_weight: float = 0.0,
    meta: dict[str, Any] | None = None,
    **params: object,
) -> PgVectorCollectionAdapter:
    """Create an adapter and its collection with the given options."""
    config = VectorDBBackendConfig(
        backend="ov_postgres.adapter.PgVectorCollectionAdapter",
        name="context",
        index_name="default",
        distance_metric=params.pop("distance_metric", "cosine"),
        sparse_weight=sparse_weight,
        custom_params={"dsn": dsn, "schema": schema, **params},
    )
    adapter = PgVectorCollectionAdapter.from_config(config)
    adapter.create_collection(
        "context",
        META if meta is None else meta,
        distance=config.distance_metric,
        sparse_weight=sparse_weight,
        index_name="default",
    )
    return adapter


def build_existing(dsn: str, schema: str, **params: object) -> PgVectorCollectionAdapter:
    """Open an already-created collection without calling create_collection."""
    config = VectorDBBackendConfig(
        backend="ov_postgres.adapter.PgVectorCollectionAdapter",
        name="context",
        index_name="default",
        custom_params={"dsn": dsn, "schema": schema, **params},
    )
    return PgVectorCollectionAdapter.from_config(config)


def indexes_on(dsn: str, schema: str, table: str = "ov_context") -> dict[str, str]:
    """Return the indexes present on a table, keyed by name."""
    import psycopg

    with psycopg.connect(dsn) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT indexname, indexdef FROM pg_indexes "
                "WHERE schemaname = %s AND tablename = %s",
                (schema, table),
            )
            return {row[0]: row[1] for row in cur.fetchall()}


def test_flat_creates_no_ann_index(dsn: str, test_schema: str) -> None:
    """`flat` means exact search, which in pgvector is simply no index."""
    adapter = build(dsn, test_schema, index_method="flat")
    try:
        defs = indexes_on(dsn, test_schema)
        assert not any("hnsw" in d or "ivfflat" in d for d in defs.values())
    finally:
        adapter.close()


def test_hnsw_index_is_created_and_queryable(dsn: str, test_schema: str) -> None:
    """An HNSW index is built with its options and still returns rows."""
    adapter = build(
        dsn,
        test_schema,
        index_method="hnsw",
        index_options={"m": 8, "ef_construction": 32},
    )
    try:
        defs = indexes_on(dsn, test_schema)
        hnsw = [d for d in defs.values() if "hnsw" in d]
        assert hnsw, f"no hnsw index created; got {defs}"
        assert "vector_cosine_ops" in hnsw[0]
        assert "m='8'" in hnsw[0] or "m=8" in hnsw[0]

        adapter.upsert([{"id": f"r{i}", "vector": vec(i)} for i in range(20)])
        results = adapter.query(query_vector=vec(3), limit=5)
        assert len(results) == 5
    finally:
        adapter.close()


def test_ivfflat_index_is_created(dsn: str, test_schema: str) -> None:
    """An IVFFlat index is built when configured."""
    adapter = build(dsn, test_schema, index_method="ivfflat", index_options={"lists": 4})
    try:
        defs = indexes_on(dsn, test_schema)
        assert any("ivfflat" in d for d in defs.values()), defs
    finally:
        adapter.close()


def test_unknown_index_method_is_rejected(dsn: str, test_schema: str) -> None:
    """An unknown index method is refused when the config is parsed.

    Catching it at the boundary means the error names the offending field and
    lists the valid values, instead of surfacing later from DDL generation.
    """
    with pytest.raises(ValidationError, match="index_method"):
        build(dsn, test_schema, index_method="bogus")


def test_unknown_custom_param_is_rejected(dsn: str, test_schema: str) -> None:
    """A misspelled custom_params key fails loudly rather than being ignored.

    Silently dropping it would leave the default in place, which is the
    failure mode this backend's config model exists to prevent.
    """
    with pytest.raises(ValidationError, match="taable_prefix"):
        build(dsn, test_schema, taable_prefix="oops_")


def test_scalar_and_fulltext_indexes_exist(dsn: str, test_schema: str) -> None:
    """Scalar, array, path-prefix and full-text indexes are all created."""
    adapter = build(dsn, test_schema)
    try:
        defs = indexes_on(dsn, test_schema)
        assert any(name.endswith("__level_idx") for name in defs)
        # list<string> fields get a GIN index, not a btree.
        tags = [d for n, d in defs.items() if n.endswith("__search_tags_idx")]
        assert tags and "gin" in tags[0].lower()
        assert any(name.endswith("__fts_idx") for name in defs)
        # Path fields additionally get a prefix-searchable index.
        assert any(name.endswith("__uri_prefix_idx") for name in defs)
    finally:
        adapter.close()


@pytest.mark.parametrize("metric", ["cosine", "l2", "ip"])
def test_distance_metrics_rank_nearest_first(
    dsn: str, test_schema: str, metric: str
) -> None:
    """Every metric ranks the identical vector first."""
    adapter = build(dsn, test_schema, distance_metric=metric)
    try:
        target = [1.0] + [0.0] * (DIM - 1)
        adapter.upsert(
            [
                {"id": "same", "vector": target},
                {"id": "orthogonal", "vector": [0.0, 1.0] + [0.0] * (DIM - 2)},
                {"id": "opposite", "vector": [-1.0] + [0.0] * (DIM - 1)},
            ]
        )
        ranked = [r["id"] for r in adapter.query(query_vector=target, limit=3)]
        assert ranked[0] == "same", f"{metric} ranked {ranked}"
        assert ranked.index("opposite") > ranked.index("same")
    finally:
        adapter.close()


def test_scores_are_finite_and_descending(dsn: str, test_schema: str) -> None:
    """Scores are finite and monotonically ordered."""
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert([{"id": f"r{i}", "vector": vec(i)} for i in range(10)])
        scores = [r["_score"] for r in adapter.query(query_vector=vec(0), limit=10)]
        assert all(math.isfinite(s) for s in scores)
        assert scores == sorted(scores, reverse=True)
    finally:
        adapter.close()


@pytest.mark.parametrize("index_method", ["flat", "hnsw"])
def test_unknown_distance_metric_is_rejected(
    dsn: str, test_schema: str, index_method: str
) -> None:
    """Must fail on the exact path too, not only when building an ANN index.

    Falling back to cosine for an unrecognised metric would silently return
    wrong rankings rather than surfacing the misconfiguration.
    """
    with pytest.raises(ValueError, match="distance"):
        build(dsn, test_schema, distance_metric="manhattan", index_method=index_method)


def test_sparse_weight_shifts_ranking(dsn: str, test_schema: str) -> None:
    """With a sparse term weighted in, a sparse-matching row should win.

    Both rows sit at the same dense distance from the query, so any change in
    ordering is attributable to the sparse contribution.
    """
    adapter = build(dsn, test_schema, sparse_weight=10.0)
    try:
        shared = [1.0] + [0.0] * (DIM - 1)
        adapter.upsert(
            [
                {"id": "sparse_hit", "vector": shared, "sparse_vector": {"42": 1.0}},
                {"id": "sparse_miss", "vector": shared, "sparse_vector": {"7": 1.0}},
            ]
        )
        dense_only = adapter.query(query_vector=shared, limit=2)
        assert {r["id"] for r in dense_only} == {"sparse_hit", "sparse_miss"}

        hybrid = adapter.query(
            query_vector=shared, sparse_query_vector={"42": 1.0}, limit=2
        )
        assert hybrid[0]["id"] == "sparse_hit"
        assert hybrid[0]["_score"] > hybrid[1]["_score"]
    finally:
        adapter.close()


def test_sparse_ignored_when_weight_is_zero(dsn: str, test_schema: str) -> None:
    """Zero sparse weight leaves dense scores untouched."""
    adapter = build(dsn, test_schema, sparse_weight=0.0)
    try:
        shared = [1.0] + [0.0] * (DIM - 1)
        adapter.upsert(
            [
                {"id": "a", "vector": shared, "sparse_vector": {"42": 1.0}},
                {"id": "b", "vector": shared, "sparse_vector": {"7": 1.0}},
            ]
        )
        results = adapter.query(
            query_vector=shared, sparse_query_vector={"42": 1.0}, limit=2
        )
        # Identical dense vectors and no sparse weighting -> identical scores.
        assert results[0]["_score"] == pytest.approx(results[1]["_score"])
    finally:
        adapter.close()


def test_sparse_dot_function_matches_python(dsn: str, test_schema: str) -> None:
    """Differential check on the SQL sparse dot product."""
    import json

    import psycopg

    adapter = build(dsn, test_schema, sparse_weight=1.0)
    try:
        cases = [
            ({"1": 2.0, "2": 3.0}, {"1": 4.0, "2": 5.0}),
            ({"1": 1.0}, {"2": 1.0}),
            ({}, {"1": 1.0}),
            ({"a": 0.5, "b": -1.5}, {"a": 2.0, "b": 2.0, "c": 9.0}),
        ]
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SET search_path TO {}, public").format(
                        sql.Identifier(test_schema)
                    )
                )
                for left, right in cases:
                    expected = sum(v * right[k] for k, v in left.items() if k in right)
                    cur.execute(
                        "SELECT ov_sparse_dot(%s::jsonb, %s::jsonb)",
                        (json.dumps(left), json.dumps(right)),
                    )
                    row = cur.fetchone()
                    assert row is not None
                    assert row[0] == pytest.approx(expected)
    finally:
        adapter.close()


def test_search_by_id_excludes_the_seed_row(dsn: str, test_schema: str) -> None:
    """Neighbour search never returns the row it started from."""
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert([{"id": f"r{i}", "vector": vec(i)} for i in range(5)])
        result = adapter.get_collection().search_by_id("default", "r0", limit=10)
        ids = [item.id for item in result.data]
        assert "r0" not in ids
        assert len(ids) == 4
    finally:
        adapter.close()


def test_search_by_id_on_missing_row_is_empty(dsn: str, test_schema: str) -> None:
    """Neighbour search on an unknown id is empty."""
    adapter = build(dsn, test_schema)
    try:
        result = adapter.get_collection().search_by_id("default", "ghost")
        assert result.data == []
    finally:
        adapter.close()


def test_search_by_random_respects_filters(dsn: str, test_schema: str) -> None:
    """Random sampling still honours the filter."""
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert(
            [{"id": f"r{i}", "level": i % 2, "vector": vec(i)} for i in range(10)]
        )
        result = adapter.get_collection().search_by_random(
            "default",
            limit=100,
            filters={"op": "must", "field": "level", "conds": [1]},
        )
        assert len(result.data) == 5
    finally:
        adapter.close()


def test_multimodal_raises_not_implemented(dsn: str, test_schema: str) -> None:
    """Multimodal search reports it is unsupported."""
    adapter = build(dsn, test_schema)
    try:
        with pytest.raises(NotImplementedError):
            adapter.get_collection().search_by_multimodal(
                "default", text="x", image=None, video=None
            )
    finally:
        adapter.close()


def test_dsn_falls_back_to_environment(
    monkeypatch: pytest.MonkeyPatch, dsn: str, test_schema: str
) -> None:
    """A DSN can come from the environment."""
    monkeypatch.setenv("OPENVIKING_POSTGRES_DSN", dsn)
    config = VectorDBBackendConfig(
        backend="ov_postgres.adapter.PgVectorCollectionAdapter",
        name="context",
        custom_params={"schema": test_schema},
    )
    adapter = PgVectorCollectionAdapter.from_config(config)
    try:
        assert adapter.collection_exists() is False  # connects successfully
    finally:
        adapter.close()


def test_missing_dsn_raises_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing DSN names every place one could be set."""
    for var in ("OPENVIKING_POSTGRES_DSN", "OPENVIKING_PG_DSN", "DATABASE_URL"):
        monkeypatch.delenv(var, raising=False)
    config = VectorDBBackendConfig(
        backend="ov_postgres.adapter.PgVectorCollectionAdapter", name="context"
    )
    with pytest.raises(ValueError, match="connection string"):
        PgVectorCollectionAdapter.from_config(config)


def test_helpers_live_in_the_configured_schema(dsn: str, test_schema: str) -> None:
    """Helper functions must not leak into `public`.

    They are referenced schema-qualified, so if they were created unqualified
    the queries would either fail or silently resolve to a stale copy in
    `public` left by another deployment.
    """
    import psycopg

    adapter = build(dsn, test_schema)
    try:
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT n.nspname, p.proname FROM pg_proc p "
                    "JOIN pg_namespace n ON n.oid = p.pronamespace "
                    "WHERE p.proname LIKE 'ov\\_%'"
                )
                found = {(row[0], row[1]) for row in cur.fetchall()}

        expected = {"ov_path_matches", "ov_sparse_dot", "ov_array_to_text"}
        in_test_schema = {name for ns, name in found if ns == test_schema}
        in_public = {name for ns, name in found if ns == "public"}

        assert expected <= in_test_schema, f"missing from {test_schema}: {found}"
        assert not (expected & in_public), f"helpers leaked into public: {in_public}"
    finally:
        adapter.close()


def test_path_filter_works_without_public_on_search_path(
    dsn: str, test_schema: str
) -> None:
    """A qualified call must not depend on search_path including the schema."""
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert(
            [
                {"id": "a", "uri": "viking://a", "vector": vec(1)},
                {"id": "b", "uri": "viking://a/b", "vector": vec(2)},
            ]
        )
        results = adapter.query(
            filter={"op": "must", "field": "uri", "conds": ["/a"], "para": "-d=0"},
            limit=10,
        )
        assert {r["id"] for r in results} == {"a"}
    finally:
        adapter.close()


def test_table_prefix_is_honoured(dsn: str, test_schema: str) -> None:
    """The configured table prefix is used."""
    adapter = build(dsn, test_schema, table_prefix="custom_")
    try:
        defs = indexes_on(dsn, test_schema, table="custom_context")
        assert defs, "expected the table to be created with the configured prefix"
    finally:
        adapter.close()


GEO_FIELD_TYPES: dict[str, str] = {}

GEO_META: dict[str, Any] = {
    "CollectionName": "context",
    "Description": "geo and float test collection",
    "Fields": [
        {"FieldName": "id", "FieldType": "string", "IsPrimaryKey": True},
        {"FieldName": "name", "FieldType": "string"},
        {"FieldName": "loc", "FieldType": "geo_point"},
        {"FieldName": "score", "FieldType": "float32"},
        {"FieldName": "counts", "FieldType": "list<int64>"},
        {"FieldName": "flag", "FieldType": "bool"},
        {"FieldName": "vector", "FieldType": "vector", "Dim": DIM},
    ],
    "ScalarIndex": ["name", "score", "counts", "flag"],
}

GEO_FIELD_TYPES.update({f["FieldName"]: f["FieldType"] for f in GEO_META["Fields"]})


def build_geo(dsn: str, schema: str) -> PgVectorCollectionAdapter:
    """Create an adapter over a collection declaring geo and float fields."""
    config = VectorDBBackendConfig(
        backend="ov_postgres.adapter.PgVectorCollectionAdapter",
        name="context",
        index_name="default",
        custom_params={"dsn": dsn, "schema": schema},
    )
    adapter = PgVectorCollectionAdapter.from_config(config)
    adapter.create_collection(
        "context", GEO_META, distance="cosine", sparse_weight=0.0, index_name="default"
    )
    return adapter


def test_geo_point_collection_is_readable(dsn: str, test_schema: str) -> None:
    """A collection declaring geo_point must still be readable.

    The field is stored as two real columns, so projecting the declared name
    would reference a column that does not exist and fail every default read.
    """
    adapter = build_geo(dsn, test_schema)
    try:
        adapter.upsert({"id": "a", "name": "n", "loc": "4.9,52.4", "vector": vec(1)})

        fetched = adapter.get(["a"])[0]
        assert fetched["name"] == "n"
        assert fetched["loc"] == "4.9,52.4"

        searched = adapter.query(query_vector=vec(1), limit=1)[0]
        assert searched["loc"] == "4.9,52.4"

        assert adapter.get_collection().search_by_random("default", limit=1).data
    finally:
        adapter.close()


def test_geo_point_absent_value_is_omitted(dsn: str, test_schema: str) -> None:
    """A row with no location reads back without the field, not as a partial."""
    adapter = build_geo(dsn, test_schema)
    try:
        adapter.upsert({"id": "a", "name": "n", "vector": vec(1)})
        assert "loc" not in adapter.get(["a"])[0]
    finally:
        adapter.close()


def test_float32_equality_matches(dsn: str, test_schema: str) -> None:
    """Equality on a float32 column must match the value that was stored.

    The column is `real`; an uncast operand binds as `double precision`, and
    `0.1::real = 0.1::float8` is false.
    """
    adapter = build_geo(dsn, test_schema)
    try:
        adapter.upsert({"id": "a", "score": 0.1, "vector": vec(1)})
        node = {"op": "must", "field": "score", "conds": [0.1]}
        assert {r["id"] for r in adapter.query(filter=node, limit=10)} == {"a"}
    finally:
        adapter.close()


def test_float32_range_bounds(dsn: str, test_schema: str) -> None:
    """Range bounds on a float32 column compare at the stored precision."""
    adapter = build_geo(dsn, test_schema)
    try:
        adapter.upsert(
            [
                {"id": "lo", "score": 0.1, "vector": vec(1)},
                {"id": "hi", "score": 9.5, "vector": vec(2)},
            ]
        )
        node = {"op": "range", "field": "score", "gte": 0.1, "lte": 1.0}
        assert {r["id"] for r in adapter.query(filter=node, limit=10)} == {"lo"}
    finally:
        adapter.close()


def test_bool_and_int_list_filters(dsn: str, test_schema: str) -> None:
    """Boolean and list<int64> columns filter without type errors."""
    adapter = build_geo(dsn, test_schema)
    try:
        adapter.upsert(
            [
                {"id": "t", "flag": True, "counts": [1, 2], "vector": vec(1)},
                {"id": "f", "flag": False, "counts": [3], "vector": vec(2)},
            ]
        )
        flag = {"op": "must", "field": "flag", "conds": [True]}
        assert {r["id"] for r in adapter.query(filter=flag, limit=10)} == {"t"}

        counts = {"op": "must", "field": "counts", "conds": [3]}
        assert {r["id"] for r in adapter.query(filter=counts, limit=10)} == {"f"}
    finally:
        adapter.close()


def test_search_by_id_accepts_a_filter_wrapper(dsn: str, test_schema: str) -> None:
    """A `{"filter": ...}` wrapper survives being nested for neighbour search.

    ``search_by_id`` wraps the caller's filter in an ``and``, so the unwrapping
    must work at any depth, not only for a bare one-key node.
    """
    adapter = build_geo(dsn, test_schema)
    try:
        adapter.upsert(
            [{"id": f"r{i}", "name": "keep", "vector": vec(i)} for i in range(4)]
        )
        wrapped = {"filter": {"op": "must", "field": "name", "conds": ["keep"]}}
        result = adapter.get_collection().search_by_id("default", "r0", filters=wrapped)
        ids = [item.id for item in result.data]
        assert "r0" not in ids
        assert len(ids) == 3
    finally:
        adapter.close()


def test_table_names_do_not_collide_after_sanitising(dsn: str, test_schema: str) -> None:
    """Names that sanitise alike must not share one table.

    `my-collection` and `my_collection` both sanitise to `my_collection`;
    without a disambiguator the second would silently bind to the first's
    table and read its rows.
    """
    adapters = []
    try:
        for name, marker in (("my-collection", "dash"), ("my_collection", "under")):
            config = VectorDBBackendConfig(
                backend="ov_postgres.adapter.PgVectorCollectionAdapter",
                name=name,
                index_name="default",
                custom_params={"dsn": dsn, "schema": test_schema},
            )
            adapter = PgVectorCollectionAdapter.from_config(config)
            adapter.create_collection(
                name, GEO_META, distance="cosine", sparse_weight=0.0, index_name="default"
            )
            adapter.upsert({"id": marker, "name": marker, "vector": vec(1)})
            adapters.append(adapter)

        # Each collection sees only its own row.
        assert [r["id"] for r in adapters[0].query(query_vector=vec(1))] == ["dash"]
        assert [r["id"] for r in adapters[1].query(query_vector=vec(1))] == ["under"]
    finally:
        for adapter in adapters:
            adapter.close()


def test_index_distance_survives_reopen(dsn: str, test_schema: str) -> None:
    """A reopened collection queries with the metric its index was built for.

    An ANN index is built for one operator class. Falling back to the
    configured default after a restart would leave the index unusable and rank
    by a different metric than the one requested.
    """
    first = build(dsn, test_schema, distance_metric="l2", index_method="hnsw")
    first.upsert([{"id": f"r{i}", "vector": vec(i)} for i in range(5)])
    first.close()

    # Reopen with the default metric configured, not l2.
    config = VectorDBBackendConfig(
        backend="ov_postgres.adapter.PgVectorCollectionAdapter",
        name="context",
        index_name="default",
        distance_metric="cosine",
        custom_params={"dsn": dsn, "schema": test_schema, "index_method": "hnsw"},
    )
    second = PgVectorCollectionAdapter.from_config(config)
    try:
        assert second.collection_exists()
        assert second._stored_index_distance() == "l2"
        assert second.query(query_vector=vec(0), limit=5)
    finally:
        second.close()


def test_declared_null_column_is_not_shadowed_by_extra(
    dsn: str, test_schema: str
) -> None:
    """A declared column that is NULL must not be filled in from `extra`.

    `extra` can hold a stale copy written before the field joined the schema;
    returning it would misreport the row.
    """
    adapter = build_geo(dsn, test_schema)
    try:
        # Write `name` into extra by bypassing the schema-aware split.
        collection = adapter.get_collection()
        collection.upsert_data([{"id": "a", "vector": vec(1)}])
        with psycopg.connect(dsn) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("UPDATE {}.{} SET extra = %s WHERE id = %s").format(
                        sql.Identifier(test_schema), sql.Identifier("ov_context")
                    ),
                    ('{"loc": "9.9,9.9", "name": "STALE"}', "a"),
                )
                conn.commit()

        record = adapter.get(["a"])[0]
        # `loc` is genuinely NULL -- geo_point takes no default -- so this is
        # the case the guard exists for: a declared column must win even when
        # its value is absent, or a stale copy in `extra` would be reported as
        # the row's location.
        assert "loc" not in record, f"stale extra leaked: {record}"
        assert record["name"] == "", f"stale extra leaked: {record}"
    finally:
        adapter.close()


@pytest.mark.parametrize(
    "text,expected",
    [
        ("0.8.2", (0, 8, 2)),
        ("0.5.0", (0, 5, 0)),
        ("0.7", (0, 7)),
        ("1.0.0-beta", (1, 0, 0)),
        ("weird", (0,)),
    ],
)
def test_extension_version_parsing(text: str, expected: tuple[int, ...]) -> None:
    """Versions parse to comparable tuples; junk degrades instead of raising."""
    assert ddl.parse_extension_version(text) == expected


def test_bootstrap_can_skip_create_extension() -> None:
    """`create_extension=false` omits the statement a managed role cannot run."""
    with_ext = ddl.bootstrap_statements("s", create_extension=True)
    without = ddl.bootstrap_statements("s", create_extension=False)
    assert len(with_ext) - len(without) == 1
    rendered = " ".join(s.as_string(None) for s in without)
    assert "CREATE EXTENSION" not in rendered


def test_create_extension_false_works_when_already_installed(
    dsn: str, test_schema: str
) -> None:
    """With the extension present, skipping creation is a no-op."""
    adapter = build(dsn, test_schema, create_extension=False)
    try:
        adapter.upsert({"id": "a", "vector": vec(1)})
        assert adapter.count() == 1
    finally:
        adapter.close()


def test_detected_version_gates_iterative_scan(dsn: str, test_schema: str) -> None:
    """The GUC is emitted only for an ANN index, a filter, and pgvector >= 0.8."""
    adapter = build(dsn, test_schema, index_method="hnsw")
    try:
        collection = adapter.get_collection()
        inner = collection._Collection__collection
        assert inner._pgvector_version >= ddl.MIN_VERSION_ITERATIVE_SCAN

        filtered = inner._iterative_scan_setup()
        assert len(filtered) == 1
        assert "hnsw.iterative_scan" in filtered[0].as_string(None)

        # Needed for an unfiltered search too: an index scan visits at most
        # hnsw.ef_search candidates, so a bare LIMIT 200 returned 40 rows.
        assert len(inner._iterative_scan_setup()) == 1

        # An older pgvector must not be sent a GUC it does not know.
        inner._pgvector_version = (0, 7, 0)
        assert inner._iterative_scan_setup() == []
    finally:
        adapter.close()


def test_iterative_scan_not_used_for_exact_search(dsn: str, test_schema: str) -> None:
    """Exact search scans every row, so the GUC would only add cost."""
    adapter = build(dsn, test_schema, index_method="flat")
    try:
        inner = adapter.get_collection()._Collection__collection
        assert inner._iterative_scan_setup() == []
    finally:
        adapter.close()


def test_iterative_scan_off_disables_the_guc(dsn: str, test_schema: str) -> None:
    """Setting the option to `off` suppresses the statement entirely."""
    adapter = build(dsn, test_schema, index_method="hnsw", iterative_scan="off")
    try:
        inner = adapter.get_collection()._Collection__collection
        assert inner._iterative_scan_setup() == []
    finally:
        adapter.close()


def test_ivfflat_degrades_strict_order(dsn: str, test_schema: str) -> None:
    """Ivfflat has no strict_order mode, so it degrades rather than erroring."""
    adapter = build(
        dsn,
        test_schema,
        index_method="ivfflat",
        index_options={"lists": 4},
        iterative_scan="strict_order",
    )
    try:
        inner = adapter.get_collection()._Collection__collection
        rendered = inner._iterative_scan_setup()[0].as_string(None)
        assert "ivfflat.iterative_scan" in rendered
        assert "relaxed_order" in rendered
    finally:
        adapter.close()


def test_filtered_hnsw_search_returns_a_full_page(dsn: str, test_schema: str) -> None:
    """A selective filter over an HNSW index still fills the requested limit.

    Without iterative scan the index visits a fixed candidate pool and then
    filters, so a selective predicate silently returns short.
    """
    adapter = build(dsn, test_schema, index_method="hnsw")
    try:
        # One row in 20 matches, spread across the whole vector space.
        adapter.upsert(
            [
                {
                    "id": f"r{i}",
                    "level": 1 if i % 20 == 0 else 0,
                    "vector": vec(i),
                }
                for i in range(400)
            ]
        )
        node = {"op": "must", "field": "level", "conds": [1]}
        results = adapter.query(query_vector=vec(7), filter=node, limit=20)
        assert len(results) == 20, f"index returned short: {len(results)}"
    finally:
        adapter.close()


@pytest.mark.parametrize(
    "field,conds,expect",
    [
        ("counts", [1, 1.5], {"t"}),
        ("counts", [1.5], set()),
        ("counts", [-0.4], set()),
        ("counts", [2.0], {"t"}),
        ("score", [0.5, 1], {"t"}),
    ],
)
def test_mixed_and_fractional_numeric_operands(
    dsn: str, test_schema: str, field: str, conds: list[Any], expect: set[str]
) -> None:
    """Numeric operand lists must bind as one type and never be rounded.

    psycopg refuses a heterogeneous list outright, and the ``::bigint[]`` cast
    on an integer array silently rounds a fractional operand -- ``ARRAY[1.5]``
    becomes ``{2}`` and matches a value the reference does not.
    """
    adapter = build_geo(dsn, test_schema)
    try:
        adapter.upsert({"id": "t", "counts": [1, 2], "score": 0.5, "vector": vec(1)})
        node = {"op": "must", "field": field, "conds": conds}
        assert {r["id"] for r in adapter.query(filter=node, limit=10)} == expect
    finally:
        adapter.close()


@pytest.mark.parametrize("key", ["lte", "lt", "gte", "gt"])
def test_nan_range_bounds_match_nothing(dsn: str, test_schema: str, key: str) -> None:
    """A NaN bound must not select the whole table.

    PostgreSQL sorts NaN above every number, so ``score <= NaN`` was true for
    every row, while the reference's ``_in_range`` matches none of them.
    """
    adapter = build_geo(dsn, test_schema)
    try:
        adapter.upsert(
            [
                {"id": "a", "score": 1.0, "vector": vec(1)},
                {"id": "b", "score": 5.0, "vector": vec(2)},
            ]
        )
        node = {"op": "range", "field": "score", key: float("nan")}
        assert adapter.query(filter=node, limit=10) == []
    finally:
        adapter.close()


@pytest.mark.parametrize(
    "key,bound,expect",
    [
        ("lte", float("inf"), {"a", "b"}),
        ("lt", float("inf"), {"a", "b"}),
        ("gte", float("-inf"), {"a", "b"}),
        ("gt", float("-inf"), {"a", "b"}),
        ("gte", float("inf"), set()),
        ("lte", float("-inf"), set()),
    ],
)
def test_infinite_range_bounds_order_normally(
    dsn: str, test_schema: str, key: str, bound: float, expect: set[str]
) -> None:
    """Infinity is a real bound and must not be dropped.

    Python orders against infinity exactly as PostgreSQL does, so ``lte=inf``
    matches everything. Rejecting it alongside NaN made these match nothing.
    """
    adapter = build_geo(dsn, test_schema)
    try:
        adapter.upsert(
            [
                {"id": "a", "score": 1.0, "vector": vec(1)},
                {"id": "b", "score": 5.0, "vector": vec(2)},
            ]
        )
        node = {"op": "range", "field": "score", key: bound}
        assert {r["id"] for r in adapter.query(filter=node, limit=10)} == expect
    finally:
        adapter.close()


def test_schema_declared_default_beats_the_type_default(
    dsn: str, test_schema: str
) -> None:
    """``DefaultValue`` in the schema wins, as it does on the built-in engine."""
    meta = {
        "CollectionName": "context",
        "Fields": [
            {"FieldName": "id", "FieldType": "string", "IsPrimaryKey": True},
            {"FieldName": "level", "FieldType": "int64", "DefaultValue": 5},
            {"FieldName": "vector", "FieldType": "vector", "Dim": DIM},
        ],
        "ScalarIndex": ["level"],
    }
    config = VectorDBBackendConfig(
        backend="ov_postgres.adapter.PgVectorCollectionAdapter",
        name="context",
        index_name="default",
        custom_params={"dsn": dsn, "schema": test_schema},
    )
    adapter = PgVectorCollectionAdapter.from_config(config)
    adapter.create_collection(
        "context", meta, distance="cosine", sparse_weight=0.0, index_name="default"
    )
    try:
        adapter.upsert({"id": "a", "vector": vec(1)})
        assert adapter.get(["a"])[0]["level"] == 5
    finally:
        adapter.close()


def test_empty_datetime_round_trips_from_the_local_backend(
    dsn: str, test_schema: str
) -> None:
    """``created_at: ""`` is the engine's own default and must not raise.

    ``LocalCollection.fetch_data`` returns it for any record without a
    timestamp, so rejecting it made such a record impossible to copy here.
    """
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert({"id": "a", "created_at": "", "vector": vec(1)})
        assert "created_at" not in adapter.get(["a"])[0]
    finally:
        adapter.close()


def test_ann_index_is_actually_used(dsn: str, test_schema: str) -> None:
    """The emitted statement must be planned as an index scan, not a seq scan.

    Anything wrapped around the distance operator -- the score template, a
    COALESCE, an extra leading ORDER BY key -- stops PostgreSQL matching the
    expression against the HNSW operator class, and it silently falls back to
    scanning every row. That made ``index_method`` and the whole iterative-scan
    mechanism inert while the README advertised them.
    """
    import random

    import psycopg

    adapter = build(dsn, test_schema, index_method="hnsw")
    try:
        rng = random.Random(7)
        adapter.upsert(
            [
                {
                    "id": f"r{i}",
                    "level": i % 3,
                    "vector": [rng.uniform(-1, 1) for _ in range(DIM)],
                }
                for i in range(2000)
            ]
        )

        captured: dict[str, Any] = {}
        collection = adapter.get_collection()._Collection__collection
        original = collection._execute

        def spy(stmt: Any, params: Any = None, **kw: Any) -> Any:
            text = stmt.as_string(None) if hasattr(stmt, "as_string") else str(stmt)
            if "_score" in text and text.lstrip().startswith("SELECT"):
                captured["sql"] = text
                captured["params"] = list(params or [])
            return original(stmt, params, **kw)

        collection._execute = spy
        adapter.query(query_vector=vec(1), limit=10)
        collection._execute = original

        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(f'ANALYZE "{test_schema}".ov_context')
            cur.execute("EXPLAIN " + captured["sql"], captured["params"])
            plan = "\n".join(row[0] for row in cur.fetchall())

        assert "Index Scan" in plan, plan
        assert "hnsw" in plan.lower(), plan
        assert "Seq Scan" not in plan, plan
    finally:
        adapter.close()


def test_sparse_vector_takes_no_default(dsn: str, test_schema: str) -> None:
    """``sparse_vector`` must stay unset, as the engine leaves it.

    ``LocalCollection._write_data_list`` pops the sparse-vector key before
    serialising, so defaulting it to ``{}`` here inverted every filter that
    tests it for absence.
    """
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert({"id": "a", "vector": vec(1)})
        assert "sparse_vector" not in adapter.get(["a"])[0]
        node = {"op": "must", "field": "sparse_vector", "conds": [None]}
        assert {r["id"] for r in adapter.query(filter=node, limit=10)} == {"a"}
    finally:
        adapter.close()


@pytest.mark.parametrize(
    "declared,expected",
    [
        (5, 5),
        (0, 0),
        ("n/a", 0),
    ],
)
def test_declared_default_is_validated(
    dsn: str, test_schema: str, declared: object, expected: object
) -> None:
    """A mistyped ``DefaultValue`` falls back to the type default.

    The engine never validates ``DefaultValue`` -- pydantic skips defaults --
    so a string on an integer column is accepted there. Writing it verbatim
    here would make every upsert fail on the column type.
    """
    meta = {
        "CollectionName": "context",
        "Fields": [
            {"FieldName": "id", "FieldType": "string", "IsPrimaryKey": True},
            {"FieldName": "level", "FieldType": "int64", "DefaultValue": declared},
            {"FieldName": "vector", "FieldType": "vector", "Dim": DIM},
        ],
        "ScalarIndex": ["level"],
    }
    config = VectorDBBackendConfig(
        backend="ov_postgres.adapter.PgVectorCollectionAdapter",
        name="context",
        index_name="default",
        custom_params={"dsn": dsn, "schema": test_schema},
    )
    adapter = PgVectorCollectionAdapter.from_config(config)
    adapter.create_collection(
        "context", meta, distance="cosine", sparse_weight=0.0, index_name="default"
    )
    try:
        adapter.upsert({"id": "a", "vector": vec(1)})
        assert adapter.get(["a"])[0]["level"] == expected
    finally:
        adapter.close()


def test_declared_default_does_not_stop_backfill_converging(
    dsn: str, test_schema: str
) -> None:
    """A timestamp default must not make the backfill rewrite rows forever.

    ``DefaultValue: ""`` on a ``date_time`` resolved to NULL, so the column
    joined the repair predicate with a no-op assignment and every row was
    rewritten on every pass.
    """
    meta = {
        "CollectionName": "context",
        "Fields": [
            {"FieldName": "id", "FieldType": "string", "IsPrimaryKey": True},
            {"FieldName": "when", "FieldType": "date_time", "DefaultValue": ""},
            {"FieldName": "level", "FieldType": "int64"},
            {"FieldName": "vector", "FieldType": "vector", "Dim": DIM},
        ],
        "ScalarIndex": ["level"],
    }
    config = VectorDBBackendConfig(
        backend="ov_postgres.adapter.PgVectorCollectionAdapter",
        name="context",
        index_name="default",
        custom_params={"dsn": dsn, "schema": test_schema},
    )
    adapter = PgVectorCollectionAdapter.from_config(config)
    adapter.create_collection(
        "context", meta, distance="cosine", sparse_weight=0.0, index_name="default"
    )
    try:
        adapter.upsert([{"id": f"r{i}", "vector": vec(i)} for i in range(3)])
        assert adapter.backfill_defaults() == 0
        assert adapter.backfill_defaults() == 0
    finally:
        adapter.close()


@pytest.mark.parametrize(
    "key,bound,expect",
    [
        ("lte", 2, {"t", "f"}),
        ("gte", 0.5, {"t"}),
        ("lt", 0.5, {"f"}),
    ],
)
def test_range_bounds_on_a_bool_column(
    dsn: str, test_schema: str, key: str, bound: object, expect: set[str]
) -> None:
    """A boolean column is orderable, so a range bound need not be 0 or 1.

    Equality needs an exact 0 or 1 -- the reference finds ``True == 2`` false --
    but ``True >= 0.5`` is true, so ordering accepts any number.
    """
    adapter = build_geo(dsn, test_schema)
    try:
        adapter.upsert(
            [
                {"id": "t", "flag": True, "vector": vec(1)},
                {"id": "f", "flag": False, "vector": vec(2)},
            ]
        )
        node = {"op": "range", "field": "flag", key: bound}
        assert {r["id"] for r in adapter.query(filter=node, limit=10)} == expect
    finally:
        adapter.close()


@pytest.mark.parametrize(
    "field,conds", [("counts", [10**19]), ("score", [10**400]), ("level", [10**19])]
)
def test_out_of_range_operands_match_nothing(
    dsn: str, test_schema: str, field: str, conds: list[Any]
) -> None:
    """An operand no column value could equal must not crash the query.

    ``10**19`` exceeds bigint and raised ``NumericValueOutOfRange``; the
    reference simply matches nothing.
    """
    adapter = build_geo(dsn, test_schema)
    try:
        adapter.upsert({"id": "a", "counts": [1], "score": 1.0, "vector": vec(1)})
        node = {"op": "must", "field": field, "conds": conds}
        assert adapter.query(filter=node, limit=10) == []
    finally:
        adapter.close()


def test_fractional_value_is_refused_by_an_int_column(dsn: str, test_schema: str) -> None:
    """Writing 1.7 to an integer column must not silently store 2.

    The engine's validator raises ``int_from_float`` rather than rounding.
    """
    adapter = build(dsn, test_schema)
    try:
        with pytest.raises(ValueError, match="not an integer"):
            adapter.upsert({"id": "a", "level": 1.7, "vector": vec(1)})
    finally:
        adapter.close()


@pytest.mark.parametrize("limit", [40, 100, 200])
def test_ann_search_returns_the_full_page(dsn: str, test_schema: str, limit: int) -> None:
    """An ANN search must return as many rows as asked for.

    An index scan visits at most ``hnsw.ef_search`` candidates (40 by default),
    so once the index was genuinely being used a bare ``LIMIT 200`` silently
    returned 40 rows. Iterative scan is needed for every ANN search, not only
    a filtered one.
    """
    import random

    import psycopg

    adapter = build(dsn, test_schema, index_method="hnsw")
    try:
        rng = random.Random(11)
        adapter.upsert(
            [
                {"id": f"r{i}", "vector": [rng.uniform(-1, 1) for _ in range(DIM)]}
                for i in range(1500)
            ]
        )
        # Without statistics the planner picks a sequential scan, which honours
        # any LIMIT and hides the truncation entirely. Autovacuum would analyse
        # the table in production, so the test must too or it proves nothing.
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(f'ANALYZE "{test_schema}".ov_context')

        assert len(adapter.query(query_vector=vec(1), limit=limit)) == limit
    finally:
        adapter.close()


@pytest.mark.parametrize(
    "key,bound,expect",
    [
        ("gte", 2, set()),
        ("gte", 1.5, set()),
        ("lte", -1, set()),
        ("lt", -1, set()),
        ("lte", 0.5, {"f"}),
        ("lt", 2, {"f", "t"}),
    ],
)
def test_bool_range_bounds_order_numerically(
    dsn: str, test_schema: str, key: str, bound: object, expect: set[str]
) -> None:
    """A bound outside 0..1 orders numerically, as Python does.

    Converting the bound with ``bool()`` moved the comparison into the boolean
    domain, so ``flag >= 2`` became ``flag >= TRUE`` and matched the true row.
    """
    adapter = build_geo(dsn, test_schema)
    try:
        adapter.upsert(
            [
                {"id": "t", "flag": True, "vector": vec(1)},
                {"id": "f", "flag": False, "vector": vec(2)},
            ]
        )
        node = {"op": "range", "field": "flag", key: bound}
        assert {r["id"] for r in adapter.query(filter=node, limit=10)} == expect
    finally:
        adapter.close()


@pytest.mark.parametrize(
    "key,bound,expect",
    [
        ("lte", 10**19, {"a"}),
        ("lt", 10**19, {"a"}),
        ("gte", 10**19, set()),
        ("gte", -(10**19), {"a"}),
        ("lte", -(10**19), set()),
    ],
)
def test_out_of_range_bounds_saturate(
    dsn: str, test_schema: str, key: str, bound: int, expect: set[str]
) -> None:
    """A bound beyond the column's range is satisfied or not, never an error.

    Dropping it as "incomparable" answered backwards: every stored value is
    below ``10**19``, so an upper bound is met by every row.
    """
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert({"id": "a", "level": 1, "vector": vec(1)})
        node = {"op": "range", "field": "level", key: bound}
        assert {r["id"] for r in adapter.query(filter=node, limit=10)} == expect
    finally:
        adapter.close()


@pytest.mark.parametrize("conds", [[1e300], [10**400]])
def test_huge_float_operands_match_nothing(
    dsn: str, test_schema: str, conds: list[Any]
) -> None:
    """A float beyond a real's range must not reach the cast and overflow."""
    adapter = build_geo(dsn, test_schema)
    try:
        adapter.upsert({"id": "a", "score": 1.0, "vector": vec(1)})
        node = {"op": "must", "field": "score", "conds": conds}
        assert adapter.query(filter=node, limit=10) == []
    finally:
        adapter.close()


def test_fractional_list_element_is_refused(dsn: str, test_schema: str) -> None:
    """A fractional element of a list<int64> must not be truncated."""
    adapter = build_geo(dsn, test_schema)
    try:
        with pytest.raises(ValueError, match="not an integer"):
            adapter.upsert({"id": "a", "counts": [1.7, 2.9], "vector": vec(1)})
    finally:
        adapter.close()


@pytest.mark.parametrize("bad", [0, -1])
def test_backfill_rejects_a_useless_batch_size(
    dsn: str, test_schema: str, bad: int
) -> None:
    """``batch_size=0`` looped forever issuing ``UPDATE ... LIMIT 0``."""
    adapter = build(dsn, test_schema)
    try:
        with pytest.raises(ValueError, match="at least 1"):
            adapter.backfill_defaults(batch_size=bad)
    finally:
        adapter.close()


def test_backfill_rejects_a_fractional_batch_size(dsn: str, test_schema: str) -> None:
    """A fractional batch size truncated to a different value than it compared.

    The LIMIT used ``int(batch_size)`` while the termination test compared
    against the unconverted value, so 10.5 repaired 10 rows, returned 10 and
    stopped with rows still unrepaired.
    """
    adapter = build(dsn, test_schema)
    try:
        with pytest.raises(ValueError, match="at least 1"):
            # Deliberately the wrong type: this guards runtime callers, which
            # the type checker does not police.
            adapter.backfill_defaults(batch_size=cast(int, 0.5))
    finally:
        adapter.close()


@pytest.mark.parametrize("batch_size", [1, 3, 100])
def test_backfill_repairs_every_row_whatever_the_batch_size(
    dsn: str, test_schema: str, batch_size: int
) -> None:
    """The batching loop must repair every row and then stop.

    A batch smaller than the backlog forces several iterations, so this also
    covers the loop's termination: it stops when a pass repairs fewer rows
    than the batch size.
    """
    adapter = build(dsn, test_schema)
    try:
        collection = adapter.get_collection()
        inner = collection._Collection__collection
        for i in range(7):
            inner._execute(
                sql.SQL("INSERT INTO {}.{} (id, vector) VALUES (%s, %s::vector)").format(
                    sql.Identifier(inner._db_schema), sql.Identifier(inner._table)
                ),
                (f"old{i}", "[" + ",".join(["0.1"] * DIM) + "]"),
            )

        assert adapter.backfill_defaults(batch_size=batch_size) == 7
        assert adapter.backfill_defaults(batch_size=batch_size) == 0

        node = {"op": "must", "field": "level", "conds": [0]}
        assert len(adapter.query(filter=node, limit=50)) == 7
    finally:
        adapter.close()


def test_auto_index_method_still_gets_the_scan_guc(dsn: str, test_schema: str) -> None:
    """``index_method="auto"`` must not lose the iterative-scan GUC.

    ``create_index`` resolved ``auto`` to a concrete method locally, so the
    gate still saw the literal string and emitted nothing -- leaving the same
    40-row truncation the GUC exists to prevent.
    """
    adapter = build(dsn, test_schema, index_method="auto")
    try:
        inner = adapter.get_collection()._Collection__collection
        assert inner._resolved_index_method() in ("flat", "hnsw", "ivfflat")
    finally:
        adapter.close()

    # A second process only *opens* the collection and never runs create_index,
    # so the resolution has to come from the registry rather than instance state.
    reopened = build_existing(dsn, test_schema, index_method="auto")
    try:
        inner = reopened.get_collection()._Collection__collection
        assert inner._index_method == "auto"
        assert inner._resolved_index_method() in ("flat", "hnsw", "ivfflat")
    finally:
        reopened.close()


@pytest.mark.parametrize("bound", [True, False])
@pytest.mark.parametrize("key", ["gt", "gte", "lt", "lte"])
def test_boolean_bounds_on_a_bool_range(
    dsn: str, test_schema: str, key: str, bound: bool
) -> None:
    """A boolean bound must survive the numeric ordering cast.

    Casting the column to int while leaving the bound a PostgreSQL boolean
    made every one of these raise ``integer >= boolean``.
    """
    adapter = build_geo(dsn, test_schema)
    try:
        records: list[dict[str, Any]] = [
            {"id": "t", "flag": True, "vector": vec(1)},
            {"id": "f", "flag": False, "vector": vec(2)},
        ]
        adapter.upsert(records)
        node = {"op": "range", "field": "flag", key: bound}
        expected = {
            r["id"]
            for r in (as_reference_row(GEO_META, rec) for rec in records)
            if matches_filter(r, node, GEO_FIELD_TYPES)
        }
        assert {r["id"] for r in adapter.query(filter=node, limit=10)} == expected
    finally:
        adapter.close()


@pytest.mark.parametrize(
    "key,bound,expect",
    [
        ("lte", float("inf"), {"a"}),
        ("lt", float("inf"), {"a"}),
        ("gte", float("-inf"), {"a"}),
        ("gte", float("inf"), set()),
        ("lte", 2**63, {"a"}),
        ("gte", 2**63, set()),
    ],
)
def test_infinite_and_boundary_bounds_on_an_int_column(
    dsn: str, test_schema: str, key: str, bound: float, expect: set[str]
) -> None:
    """Infinity and the exact bigint boundary saturate rather than match nothing.

    ``float(2**63 - 1)`` rounds up to ``2**63``, so a float comparison missed
    the very boundary the saturation logic exists for.
    """
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert({"id": "a", "level": 1, "vector": vec(1)})
        node = {"op": "range", "field": "level", key: bound}
        assert {r["id"] for r in adapter.query(filter=node, limit=10)} == expect
    finally:
        adapter.close()


def test_legacy_vectorless_row_can_be_rewritten(dsn: str, test_schema: str) -> None:
    """A row left without an embedding must not become permanently unwritable.

    OpenViking's read-modify-write paths re-upsert exactly what they fetched,
    and a legacy row reads back with no vector key. Refusing that would leave
    it stuck forever, since ``backfill_defaults`` cannot invent an embedding.
    """
    adapter = build(dsn, test_schema)
    try:
        inner = adapter.get_collection()._Collection__collection
        inner._execute(
            sql.SQL("INSERT INTO {}.{} (id, level) VALUES (%s, %s)").format(
                sql.Identifier(inner._db_schema), sql.Identifier(inner._table)
            ),
            ("legacy", 1),
        )
        record = adapter.get(["legacy"])[0]
        assert "vector" not in record

        adapter.upsert(record | {"level": 2})
        assert adapter.get(["legacy"])[0]["level"] == 2

        # A genuinely new record still needs one.
        with pytest.raises(ValueError, match="embedding is required"):
            adapter.upsert({"id": "brand-new", "level": 1})
    finally:
        adapter.close()


def test_hybrid_order_agrees_with_the_reported_score(dsn: str, test_schema: str) -> None:
    """A vectorless row must not sit at position 0 holding the lowest score.

    ``coalesce(dense, 0.0)`` is the best possible dense term for l2 and ip, so
    ordering by the raw score put such a row first while its reported score was
    floored below every real one.
    """
    config = VectorDBBackendConfig(
        backend="ov_postgres.adapter.PgVectorCollectionAdapter",
        name="context",
        index_name="default",
        distance_metric="l2",
        sparse_weight=1.0,
        custom_params={"dsn": dsn, "schema": test_schema},
    )
    adapter = PgVectorCollectionAdapter.from_config(config)
    adapter.create_collection(
        "context", META, distance="l2", sparse_weight=1.0, index_name="default"
    )
    try:
        adapter.upsert({"id": "real", "vector": vec(1), "sparse_vector": {"7": 1.0}})
        inner = adapter.get_collection()._Collection__collection
        inner._execute(
            sql.SQL(
                "INSERT INTO {}.{} (id, sparse_vector) VALUES (%s, %s::jsonb)"
            ).format(sql.Identifier(inner._db_schema), sql.Identifier(inner._table)),
            ("legacy", '{"7": 1.0}'),
        )

        results = adapter.query(
            query_vector=vec(9), sparse_query_vector={"7": 1.0}, limit=10
        )
        assert results[0]["id"] == "real"
        scores = [r["_score"] for r in results]
        assert scores == sorted(scores, reverse=True), results
    finally:
        adapter.close()


@pytest.mark.parametrize(
    "key,bound,expect",
    [
        ("lt", "cat", {"slash", "Zed"}),
        ("gt", "cat", {"zed"}),
        ("gte", "Zed", {"zed", "Zed"}),
        ("lt", "a", {"slash", "Zed"}),
    ],
)
def test_string_ranges_use_code_point_order(
    dsn: str, test_schema: str, key: str, bound: str, expect: set[str]
) -> None:
    """Text is compared by code point, as the reference does.

    PostgreSQL orders text by the database collation, which on the usual
    ``en_US.utf8`` image ignores punctuation and case in a way Python does not:
    ``'/x/y' < 'c_d'`` is false there and true in Python. Without an explicit C
    collation this diverged on roughly one filter in a hundred.
    """
    adapter = build(dsn, test_schema)
    try:
        records: list[dict[str, Any]] = [
            {"id": "slash", "name": "/x/y/z", "vector": vec(1)},
            {"id": "zed", "name": "zed", "vector": vec(2)},
            {"id": "Zed", "name": "Zed", "vector": vec(3)},
        ]
        adapter.upsert(records)
        node = {"op": "range", "field": "name", key: bound}
        assert {r["id"] for r in adapter.query(filter=node, limit=10)} == expect

        # And the reference agrees with that expectation.
        reference = {
            r["id"]
            for r in (as_reference_row(META, rec) for rec in records)
            if matches_filter(r, node, FIELD_TYPES)
        }
        assert reference == expect
    finally:
        adapter.close()


def test_scalar_sort_uses_code_point_order(dsn: str, test_schema: str) -> None:
    """``search_by_scalar`` on text must order as a Python sort would."""
    adapter = build(dsn, test_schema)
    try:
        names = ["/x/y/z", "Zed", "_under", "alpha", "zed"]
        adapter.upsert(
            [{"id": n, "name": n, "vector": vec(i)} for i, n in enumerate(names)]
        )
        ordered = [
            r["id"] for r in adapter.query(order_by="name", order_desc=False, limit=10)
        ]
        assert ordered == sorted(names)
    finally:
        adapter.close()


@pytest.mark.parametrize("key", ["lt", "lte", "gt", "gte"])
@pytest.mark.parametrize("bound", [float("inf"), float("-inf")])
def test_infinite_bounds_on_a_text_column_match_nothing(
    dsn: str, test_schema: str, key: str, bound: float
) -> None:
    """Infinity against a string column matches nothing, as the reference does.

    ``"alpha" < inf`` raises TypeError in Python and ``_in_range`` returns
    False. The saturation shortcut answered before the column type was
    consulted, so it reported "satisfied by everything" instead.
    """
    adapter = build(dsn, test_schema)
    try:
        records: list[dict[str, Any]] = [{"id": "a", "name": "alpha", "vector": vec(1)}]
        adapter.upsert(records)
        node = {"op": "range", "field": "name", key: bound}
        expected = {
            r["id"]
            for r in (as_reference_row(META, rec) for rec in records)
            if matches_filter(r, node, FIELD_TYPES)
        }
        assert {r["id"] for r in adapter.query(filter=node, limit=10)} == expected
    finally:
        adapter.close()


def test_must_not_on_an_undeclared_field_honours_none(dsn: str, test_schema: str) -> None:
    """``must_not`` with ``None`` in conds must exclude a row with no such field.

    An absent field reads as ``None``, so ``must`` matches and its negation
    must not. Only the ``must`` branch handled it.
    """
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert({"id": "a", "vector": vec(1)})
        assert (
            adapter.query(
                filter={"op": "must_not", "field": "nosuch", "conds": [None]}, limit=10
            )
            == []
        )
        assert (
            len(
                adapter.query(
                    filter={"op": "must", "field": "nosuch", "conds": [None]}, limit=10
                )
            )
            == 1
        )
    finally:
        adapter.close()


def test_update_data_cannot_clear_an_embedding(dsn: str, test_schema: str) -> None:
    """Nulling a vector would recreate the row no ANN index can return."""
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert({"id": "a", "vector": vec(1)})
        with pytest.raises(ValueError, match="embedding is required"):
            adapter.get_collection().update_data([{"id": "a", "vector": None}])
    finally:
        adapter.close()


def test_auto_index_method_emits_the_scan_guc(dsn: str, test_schema: str) -> None:
    """``auto`` must not lose the iterative-scan GUC and truncate at 40 rows.

    The gate read the raw configured string, so ``auto`` returned early and the
    resolver added in the previous round was never reached — the fix was dead
    code and the truncation it was written for was still live.
    """
    import random

    creator = build(dsn, test_schema, index_method="hnsw")
    try:
        rng = random.Random(23)
        creator.upsert(
            [
                {"id": f"r{i}", "vector": [rng.uniform(-1, 1) for _ in range(DIM)]}
                for i in range(1500)
            ]
        )
    finally:
        creator.close()

    # A second process only opens the collection, so the method must come from
    # the registry rather than from instance state.
    reopened = build_existing(dsn, test_schema, index_method="auto")
    try:
        inner = reopened.get_collection()._Collection__collection
        assert inner._resolved_index_method() == "hnsw"
        assert len(inner._iterative_scan_setup()) == 1

        import psycopg

        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(f'ANALYZE "{test_schema}".ov_context')
        assert len(reopened.query(query_vector=vec(1), limit=200)) == 200
    finally:
        reopened.close()


def test_null_and_empty_groups_are_counted_separately(dsn: str, test_schema: str) -> None:
    """A NULL column and an empty string are different groups.

    Folding NULL onto ``""`` collided with the real empty-string bucket, so one
    count was silently dropped — and which one depended on GROUP BY row order.
    """
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert(
            [
                {"id": "e1", "context_type": "", "vector": vec(1)},
                {"id": "e2", "context_type": "", "vector": vec(2)},
                {"id": "q", "context_type": "q", "vector": vec(3)},
            ]
        )
        inner = adapter.get_collection()._Collection__collection
        inner._execute(
            sql.SQL("INSERT INTO {}.{} (id, vector) VALUES (%s, %s::vector)").format(
                sql.Identifier(inner._db_schema), sql.Identifier(inner._table)
            ),
            ("nul", "[" + ",".join(["0.1"] * DIM) + "]"),
        )

        agg = inner.aggregate_data(
            index_name="default", op="count", field="context_type"
        ).agg
        assert agg == {NULL_BUCKET: 1, "": 2, "q": 1}
        assert sum(agg.values()) == adapter.count()
    finally:
        adapter.close()


def test_overlapping_concurrent_upserts_do_not_deadlock(
    dsn: str, test_schema: str
) -> None:
    """Two writers with the same ids in opposite orders must not deadlock.

    ``executemany`` takes row locks in batch order, so opposing orders deadlock
    and PostgreSQL rolls one batch back entirely. Sorting each batch by primary
    key gives every writer the same lock order.
    """
    import threading

    adapter = build(dsn, test_schema)
    try:
        ids = [f"k{i}" for i in range(30)]
        errors: list[str] = []

        def writer(forward: bool) -> None:
            order = ids if forward else list(reversed(ids))
            for _ in range(5):
                try:
                    adapter.upsert(
                        [{"id": i, "level": 1, "vector": vec(3)} for i in order]
                    )
                except Exception as exc:
                    errors.append(type(exc).__name__)

        threads = [threading.Thread(target=writer, args=(i % 2 == 0,)) for i in range(6)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert errors == [], errors
        assert adapter.count() == len(ids)
    finally:
        adapter.close()


def test_aggregate_conditions_are_validated(dsn: str, test_schema: str) -> None:
    """An unsupported or ungrouped aggregate condition is refused, not ignored.

    ``cond`` was dropped entirely when ungrouped, and unknown keys silently, so
    a caller asking for groups above a threshold got every group back with no
    way to notice.
    """
    adapter = build(dsn, test_schema)
    try:
        collection = adapter.get_collection()._Collection__collection
        adapter.upsert({"id": "a", "context_type": "x", "vector": vec(1)})

        with pytest.raises(ValueError, match="require a grouping field"):
            collection.aggregate_data(index_name="default", cond={"gt": 1000})
        with pytest.raises(ValueError, match="Unsupported aggregate condition"):
            collection.aggregate_data(
                index_name="default", field="context_type", cond={"eq": 2}
            )
        with pytest.raises(ValueError, match="must be a number"):
            collection.aggregate_data(
                index_name="default", field="context_type", cond={"gt": "many"}
            )
        with pytest.raises(ValueError, match="Cannot group on a vector"):
            collection.aggregate_data(index_name="default", field="vector")
    finally:
        adapter.close()


def test_scalar_sort_does_not_leak_the_sort_column(dsn: str, test_schema: str) -> None:
    """An explicit projection must not gain the sort field.

    ``LocalCollection.search_by_scalar`` pops it when the caller did not ask
    for it; this returned it, so the same call produced different shapes on the
    two backends.
    """
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert({"id": "a", "context_type": "x", "level": 3, "vector": vec(1)})
        record = adapter.query(
            order_by="level", order_desc=True, output_fields=["context_type"], limit=1
        )[0]
        assert record["context_type"] == "x"
        assert "level" not in record
    finally:
        adapter.close()


def test_keyword_terms_are_matched_literally(dsn: str, test_schema: str) -> None:
    """Caller text must not act as tsquery operators.

    Raw terms went into ``websearch_to_tsquery`` unquoted, so ``-fox`` was read
    as negation and returned the complement of what was asked for.
    """
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert(
            [
                {"id": "a", "name": "quick fox", "vector": vec(1)},
                {"id": "b", "name": "slow bear", "vector": vec(2)},
            ]
        )
        assert {r["id"] for r in adapter.search_by_keywords(keywords=["fox"])} == {"a"}
        # Quoted, the leading dash is punctuation and the term tokenises to
        # `fox`. Unquoted it was read as negation and returned {'b'} -- the
        # complement of what the caller asked for.
        assert {r["id"] for r in adapter.search_by_keywords(keywords=["-fox"])} == {"a"}
        # An embedded operator is likewise matched as text, not obeyed.
        assert adapter.search_by_keywords(keywords=["quick OR bear"]) == []
    finally:
        adapter.close()


@pytest.mark.parametrize(
    "node,match",
    [
        ({"op": "must", "field": "ghost", "conds": "hello"}, "conds must be a list"),
        (
            {"op": "must_not", "field": "ghost", "conds": ["/a"], "para": "-d=0"},
            "only supported for path fields",
        ),
    ],
)
def test_undeclared_fields_are_still_validated(
    dsn: str, test_schema: str, node: dict[str, Any], match: str
) -> None:
    """A malformed node is refused whether or not the column exists.

    The undeclared-field branch returned before the checks a declared field
    gets, so a bad node either matched every row or leaked a raw TypeError.
    """
    adapter = build(dsn, test_schema)
    try:
        with pytest.raises(UnsupportedFilterError, match=match):
            adapter.query(filter=node, limit=10)
    finally:
        adapter.close()


def test_equality_and_range_both_use_an_index(dsn: str, test_schema: str) -> None:
    """A collated index must not displace the plain one.

    PostgreSQL matches an expression index syntactically, so replacing the
    plain btree with ``(col COLLATE "C")`` left a bare ``col = ANY(...)`` --
    by far the commonest filter shape -- with no index at all.
    """
    import random

    import psycopg

    adapter = build(dsn, test_schema)
    try:
        rng = random.Random(31)
        adapter.upsert(
            [
                {
                    "id": f"r{i}",
                    "name": f"nm{i:07d}",
                    "vector": [rng.uniform(-1, 1) for _ in range(DIM)],
                }
                for i in range(5000)
            ]
        )
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(f'ANALYZE "{test_schema}".ov_context')
            for query in (
                f'SELECT id FROM "{test_schema}".ov_context '
                "WHERE name = ANY(ARRAY['nm0000777'])",
                f'SELECT id FROM "{test_schema}".ov_context '
                "WHERE name COLLATE \"C\" > 'nm0004990'",
            ):
                cur.execute("EXPLAIN " + query)
                plan = "\n".join(row[0] for row in cur.fetchall())
                assert "Index" in plan, plan
                assert "Seq Scan" not in plan, plan
    finally:
        adapter.close()


def test_an_upgraded_database_recovers_index_behaviour(
    dsn: str, test_schema: str
) -> None:
    """The index fixes must reach a collection created by an older version.

    ``CollectionAdapter.create_collection`` returns early when the collection
    exists, so ``create_index`` never runs again: the resolved method was never
    recorded and the collated indexes were never built. Resolution now reads
    ``pg_indexes``, and ``ensure_indexes`` supplies the missing indexes.
    """
    import random

    import psycopg

    creator = build(dsn, test_schema, index_method="hnsw")
    try:
        rng = random.Random(33)
        creator.upsert(
            [
                {
                    "id": f"r{i}",
                    "name": f"n{i:06d}",
                    "vector": [rng.uniform(-1, 1) for _ in range(DIM)],
                }
                for i in range(1500)
            ]
        )
    finally:
        creator.close()

    # Make it look like a database written before round 10.
    with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT indexname FROM pg_indexes "
            "WHERE schemaname = %s AND indexname LIKE %s",
            (test_schema, "%_c_idx"),
        )
        legacy = [row[0] for row in cur.fetchall()]
        for name in legacy:
            cur.execute(
                sql.SQL("DROP INDEX {}.{}").format(
                    sql.Identifier(test_schema), sql.Identifier(name)
                )
            )
        cur.execute(
            sql.SQL(
                "UPDATE {}.ov_indexes SET meta = meta - 'ResolvedIndexMethod'"
            ).format(sql.Identifier(test_schema))
        )
    assert legacy, "expected collated indexes to exist before the downgrade"

    reopened = build_existing(dsn, test_schema, index_method="auto")
    try:
        inner = reopened.get_collection()._Collection__collection
        # Resolved from the indexes that exist, not from a registry key an
        # older database never wrote.
        assert inner._resolved_index_method() == "hnsw"
        assert len(inner._iterative_scan_setup()) == 1

        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(f'ANALYZE "{test_schema}".ov_context')
        assert len(reopened.query(query_vector=vec(1), limit=200)) == 200

        repaired = reopened.ensure_indexes()
        assert set(legacy) <= set(repaired), (repaired, legacy)
        assert reopened.ensure_indexes() == []
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "query,expect",
    [
        ("quick fox", {"d1"}),
        ("fox", {"d1"}),
        ("-fox", {"d1"}),
        ("quick OR bear", set()),
    ],
)
def test_keyword_search_is_literal_and_not_a_phrase(
    dsn: str, test_schema: str, query: str, expect: set[str]
) -> None:
    """Multi-word search must AND its terms, not require adjacency.

    Quoting each term stopped ``-fox`` acting as negation but turned every
    multi-word query into a phrase query, so ``"quick fox"`` no longer matched
    "the quick brown fox". ``plainto_tsquery`` ANDs terms and treats
    punctuation as text.
    """
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert(
            [
                {"id": "d1", "name": "the quick brown fox", "vector": vec(1)},
                {"id": "d2", "name": "slow bear", "vector": vec(2)},
            ]
        )
        assert {r["id"] for r in adapter.search_by_keywords(query=query)} == expect
    finally:
        adapter.close()


def test_mixed_upsert_and_update_do_not_deadlock(dsn: str, test_schema: str) -> None:
    """Sorting upsert batches alone was not enough.

    ``update_data`` still locked in caller order, so a workload where every
    caller agreed on an order -- previously safe -- began deadlocking once
    upsert reordered to sorted.
    """
    import threading

    adapter = build(dsn, test_schema)
    try:
        ids = [f"k{i}" for i in range(40)]
        adapter.upsert([{"id": i, "level": 0, "vector": vec(3)} for i in ids])
        errors: list[str] = []

        def upserter(forward: bool) -> None:
            order = ids if forward else list(reversed(ids))
            for _ in range(5):
                try:
                    adapter.upsert(
                        [{"id": i, "level": 1, "vector": vec(3)} for i in order]
                    )
                except Exception as exc:
                    errors.append(type(exc).__name__)

        def updater(forward: bool) -> None:
            order = ids if forward else list(reversed(ids))
            for _ in range(5):
                try:
                    adapter.get_collection().update_data(
                        [{"id": i, "level": 2} for i in order]
                    )
                except Exception as exc:
                    errors.append(type(exc).__name__)

        threads = [
            threading.Thread(target=(upserter if i % 2 else updater), args=(i % 4 < 2,))
            for i in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == [], errors
    finally:
        adapter.close()


def test_update_data_writes_nothing_when_a_key_is_missing(
    dsn: str, test_schema: str
) -> None:
    """An unknown key must not leave the batch's other records written.

    The check ran after the transaction committed, so the good records landed
    and then the call raised. ``LocalCollection`` validates before writing.
    """
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert({"id": "a", "level": 1, "vector": vec(1)})
        with pytest.raises(ValueError, match="not found"):
            adapter.get_collection().update_data(
                [{"id": "a", "level": 77}, {"id": "ghost", "level": 1}]
            )
        assert adapter.get(["a"])[0]["level"] == 1
    finally:
        adapter.close()


def test_null_array_still_sorts_last(dsn: str, test_schema: str) -> None:
    """Collating a text array must not turn a NULL column into an empty one.

    ``array(SELECT unnest(NULL))`` is the empty array, so a NULL column stopped
    sorting last and became indistinguishable from a row holding ``[]``.
    """
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert(
            [
                {"id": "empty", "search_tags": [], "vector": vec(1)},
                {"id": "a", "search_tags": ["a"], "vector": vec(2)},
                {"id": "under", "search_tags": ["_x"], "vector": vec(3)},
            ]
        )
        inner = adapter.get_collection()._Collection__collection
        inner._execute(
            sql.SQL(
                "INSERT INTO {}.{} (id, vector, search_tags) "
                "VALUES (%s, %s::vector, NULL)"
            ).format(sql.Identifier(inner._db_schema), sql.Identifier(inner._table)),
            ("nullrow", "[" + ",".join(["0.1"] * DIM) + "]"),
        )
        ordered = [
            r["id"]
            for r in adapter.query(order_by="search_tags", order_desc=False, limit=10)
        ]
        assert ordered[-1] == "nullrow", ordered
        # And element order is code-point, so "_x" sorts before "a".
        assert ordered.index("under") < ordered.index("a"), ordered
    finally:
        adapter.close()


def test_null_group_key_avoids_a_real_value(dsn: str, test_schema: str) -> None:
    """The NULL sentinel must not merge with a row that literally holds it."""
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert(
            [
                {"id": "lit", "context_type": NULL_BUCKET, "vector": vec(1)},
                {"id": "real", "context_type": "x", "vector": vec(2)},
            ]
        )
        inner = adapter.get_collection()._Collection__collection
        inner._execute(
            sql.SQL("INSERT INTO {}.{} (id, vector) VALUES (%s, %s::vector)").format(
                sql.Identifier(inner._db_schema), sql.Identifier(inner._table)
            ),
            ("nul", "[" + ",".join(["0.1"] * DIM) + "]"),
        )
        agg = inner.aggregate_data(
            index_name="default", op="count", field="context_type"
        ).agg
        assert agg[NULL_BUCKET] == 1, agg
        assert agg[NULL_BUCKET + "_"] == 1, agg
        assert sum(agg.values()) == adapter.count()
    finally:
        adapter.close()


LONG_FIELD = "a_very_long_field_name_that_users_might_plausibly_declare"

LONG_META: dict[str, Any] = {
    "CollectionName": "context",
    "Fields": [
        {"FieldName": "id", "FieldType": "string", "IsPrimaryKey": True},
        {"FieldName": LONG_FIELD, "FieldType": "string"},
        {"FieldName": "vector", "FieldType": "vector", "Dim": DIM},
    ],
    "ScalarIndex": [LONG_FIELD],
}


def test_long_field_names_get_distinct_indexes(dsn: str, test_schema: str) -> None:
    """Index names must stay distinct after PostgreSQL truncates them.

    Identifiers are silently cut at 63 bytes, so ``…_idx`` and ``…_c_idx`` for
    a long field collapsed to the same name. ``CREATE INDEX IF NOT EXISTS``
    matches by name, so the collated index was skipped without complaint and
    every text range on that column scanned the table.
    """
    config = VectorDBBackendConfig(
        backend="ov_postgres.adapter.PgVectorCollectionAdapter",
        name="context",
        index_name="default",
        custom_params={"dsn": dsn, "schema": test_schema},
    )
    adapter = PgVectorCollectionAdapter.from_config(config)
    adapter.create_collection(
        "context", LONG_META, distance="cosine", sparse_weight=0.0, index_name="default"
    )
    try:
        built = indexes_on(dsn, test_schema)
        on_long_field = [
            name for name, definition in built.items() if LONG_FIELD in definition
        ]
        # Both the plain and the collated index must exist as separate objects.
        assert len(on_long_field) == 2, built
        assert len(set(on_long_field)) == 2
        assert all(len(name.encode("utf-8")) <= 63 for name in on_long_field)

        collated = [d for d in built.values() if 'COLLATE "C"' in d]
        assert collated, built

        adapter.upsert({"id": "a", LONG_FIELD: "value", "vector": vec(1)})
        node = {"op": "must", "field": LONG_FIELD, "conds": ["value"]}
        assert {r["id"] for r in adapter.query(filter=node, limit=10)} == {"a"}
    finally:
        adapter.close()


def test_short_index_names_are_unchanged(dsn: str, test_schema: str) -> None:
    """A name that already fits must not be rewritten.

    Existing databases keep the index names they have, so the collision-safe
    naming must be a no-op below the limit.
    """
    assert ddl.index_name("ov_context", "name", "idx") == "ov_context__name_idx"
    assert ddl.index_name("ov_context", "fts_idx") == "ov_context__fts_idx"

    adapter = build(dsn, test_schema)
    try:
        assert "ov_context__name_idx" in indexes_on(dsn, test_schema)
    finally:
        adapter.close()


def test_ensure_indexes_rebuilds_a_wrong_definition(dsn: str, test_schema: str) -> None:
    """An index with the right name but the wrong definition must be rebuilt.

    An earlier version built the *collated* index under the plain index's name.
    ``CREATE INDEX IF NOT EXISTS`` matches on name alone, so the migration
    skipped the plain index and reported success while leaving equality -- the
    commonest filter -- with no usable index.
    """
    import random

    import psycopg

    adapter = build(dsn, test_schema)
    try:
        rng = random.Random(41)
        adapter.upsert(
            [
                {
                    "id": f"r{i}",
                    "name": f"nm{i:07d}",
                    "vector": [rng.uniform(-1, 1) for _ in range(DIM)],
                }
                for i in range(4000)
            ]
        )

        # Recreate the old shape: a collated index wearing the plain one's name.
        with psycopg.connect(dsn, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL("DROP INDEX {}.{}").format(
                    sql.Identifier(test_schema),
                    sql.Identifier("ov_context__name_idx"),
                )
            )
            cur.execute(
                sql.SQL('CREATE INDEX {} ON {}.{} (name COLLATE "C")').format(
                    sql.Identifier("ov_context__name_idx"),
                    sql.Identifier(test_schema),
                    sql.Identifier("ov_context"),
                )
            )

        assert "ov_context__name_idx" in adapter.ensure_indexes()

        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(f'ANALYZE "{test_schema}".ov_context')
            cur.execute(
                f'EXPLAIN SELECT id FROM "{test_schema}".ov_context '
                "WHERE name = ANY(ARRAY['nm0000777'])"
            )
            plan = "\n".join(row[0] for row in cur.fetchall())
        assert "Index" in plan, plan
        assert "Seq Scan" not in plan, plan
        assert adapter.ensure_indexes() == []
    finally:
        adapter.close()


@pytest.mark.parametrize(
    "kwargs,expect",
    [
        ({"keywords": ["fox", "dog"]}, {"d1", "d2"}),
        ({"keywords": ["fox"]}, {"d1"}),
        ({"query": "quick fox"}, {"d1"}),
        ({"query": "-fox"}, {"d1"}),
        ({"keywords": ["fox"], "query": "dog"}, {"d1", "d2"}),
    ],
)
def test_multiple_keywords_are_alternatives(
    dsn: str, test_schema: str, kwargs: dict[str, Any], expect: set[str]
) -> None:
    """Separate keywords are alternatives; words inside one term are required.

    A single ``plainto_tsquery`` over all terms joined ANDs them, so asking for
    two keywords matched only documents containing both.
    """
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert(
            [
                {"id": "d1", "name": "the quick brown fox", "vector": vec(1)},
                {"id": "d2", "name": "a lazy dog", "vector": vec(2)},
                {"id": "d3", "name": "unrelated text", "vector": vec(3)},
            ]
        )
        assert {r["id"] for r in adapter.search_by_keywords(**kwargs)} == expect
    finally:
        adapter.close()


def test_create_index_invalidates_the_resolved_method(dsn: str, test_schema: str) -> None:
    """Building an index must not leave a stale cached resolution.

    The resolver caches what ``pg_indexes`` held at first use; ``create_index``
    changes exactly that, so a search before it kept answering ``flat`` and the
    scan GUC was never emitted -- truncating results at 40 rows.
    """
    adapter = build(dsn, test_schema, index_method="auto")
    try:
        inner = adapter.get_collection()._Collection__collection
        assert inner._resolved_index_method() == "flat"

        inner.create_index(
            "default",
            {
                "IndexName": "default",
                "VectorIndex": {"IndexType": "hnsw", "Distance": "cosine"},
                "ScalarIndex": [],
            },
        )
        assert inner._resolved_index_method() == "hnsw"
        assert len(inner._iterative_scan_setup()) == 1
    finally:
        adapter.close()


def test_a_mistyped_key_is_refused_like_the_reference(dsn: str, test_schema: str) -> None:
    """An integer offered for a string key must be refused, not coerced.

    The native engine validates every record with pydantic, which rejects
    ``3`` for a string field. PostgreSQL instead writes ``"3"`` into the text
    column, and every read then binds the key as an integer and raises
    ``operator does not exist: text = smallint`` -- a row that can be written
    and never read, updated or deleted.
    """
    adapter = build(dsn, test_schema)
    try:
        collection = adapter.get_collection()
        with pytest.raises(ValueError, match="declared 'string'"):
            adapter.upsert({"id": 3, "level": 1, "vector": vec(1)})
        with pytest.raises(ValueError, match="declared 'string'"):
            collection.update_data([{"id": 3, "level": 2}])
        assert adapter.count() == 0

        # Reads are lenient, because the engine's are: `fetch_data` and
        # `delete_data` hash `str(key)`, so an integer finds the record stored
        # under its string form rather than raising.
        adapter.upsert({"id": "3", "level": 1, "vector": vec(1)})
        assert [r["id"] for r in adapter.get([3])] == ["3"]
        assert collection.search_by_id("default", 3).data == []
        assert adapter.delete(ids=[3]) == 1
        assert adapter.count() == 0
    finally:
        adapter.close()


def int_key_meta() -> dict[str, Any]:
    """Return META with an int64 primary key."""
    meta = copy.deepcopy(META)
    meta["Fields"] = [
        {"FieldName": "id", "FieldType": "int64", "IsPrimaryKey": True},
        *[f for f in meta["Fields"] if f["FieldName"] != "id"],
    ]
    return meta


@pytest.mark.parametrize(
    ("given", "stored"),
    [(7, 7), ("7", 7), (" 7 ", 7), ("7.0", 7), (7.0, 7), (True, 1)],
)
def test_an_integer_key_is_written_as_the_reference_validates_it(
    dsn: str, test_schema: str, given: object, stored: int
) -> None:
    """On write, an int64 key accepts whatever pydantic's lax mode accepts.

    The engine's validator is lax for integers where it is strict for strings,
    so refusing these would diverge in the opposite direction.
    """
    adapter = build(dsn, test_schema, meta=int_key_meta())
    try:
        adapter.upsert({"id": given, "level": 1, "vector": vec(1)})
        assert [r["id"] for r in adapter.get([stored])] == [stored]
        adapter.get_collection().update_data([{"id": given, "level": 5}])
        assert adapter.get([stored])[0]["level"] == 5
        assert adapter.delete(ids=[stored]) == 1
    finally:
        adapter.close()


@pytest.mark.parametrize(
    ("given", "matches"),
    [(7, True), ("7", True), (" 7 ", False), ("7.0", False), (7.0, False), (True, False)],
)
def test_an_integer_key_is_read_by_its_string_form(
    dsn: str, test_schema: str, given: object, matches: bool
) -> None:
    """On read, a key matches only when its string form does.

    The engine keys rows on a hash of ``str(key)``, so ``7.0`` and ``" 7 "``
    find nothing there even though both validate as the integer 7. Coercing
    them here would find the row and diverge.
    """
    adapter = build(dsn, test_schema, meta=int_key_meta())
    try:
        adapter.upsert({"id": 7, "level": 1, "vector": vec(1)})
        assert bool(adapter.get([given])) is matches
        result = adapter.get_collection().fetch_data([given])
        assert result.ids_not_exist == ([] if matches else [given])
    finally:
        adapter.close()


@pytest.mark.parametrize("given", ["x", 7.5, None, [], "0x1f"])
def test_an_integer_key_refuses_what_the_reference_refuses(
    dsn: str, test_schema: str, given: object
) -> None:
    """Values pydantic cannot read as an integer are refused, not passed on."""
    adapter = build(dsn, test_schema, meta=int_key_meta())
    try:
        with pytest.raises((TypeError, ValueError)):
            adapter.upsert({"id": given, "level": 1, "vector": vec(1)})
    finally:
        adapter.close()


def test_update_data_returns_ids_in_input_order(dsn: str, test_schema: str) -> None:
    """The caller is told which keys were written, in the order they gave.

    Sorting for lock ordering must not leak into the return value;
    ``upsert_data`` and ``LocalCollection`` both preserve input order.
    """
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert([{"id": k, "level": 0, "vector": vec(1)} for k in ("c", "a", "b")])
        result = adapter.get_collection().update_data(
            [{"id": "c", "level": 1}, {"id": "a", "level": 1}, {"id": "b", "level": 1}]
        )
        assert result.ids == ["c", "a", "b"]
        assert result.updated_count == 3
    finally:
        adapter.close()


def test_ensure_indexes_is_idempotent_with_full_text(dsn: str, test_schema: str) -> None:
    """A full-text index must not be rebuilt on every run.

    Its key is an expression PostgreSQL rewrites heavily, so it can never match
    the statement textually. Matching on a fingerprint recorded at creation
    lets it be reconciled without rebuilding for ever.
    """
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert({"id": "a", "name": "text", "vector": vec(1)})
        assert adapter.ensure_indexes() == []
        assert adapter.ensure_indexes() == []
        assert "ov_context__fts_idx" in indexes_on(dsn, test_schema)
    finally:
        adapter.close()


def test_ensure_indexes_rebuilds_the_full_text_index_after_a_config_change(
    dsn: str, test_schema: str
) -> None:
    """Changing ``text_search_config`` must re-key the full-text index.

    The index stores lexemes produced by one configuration. A query parsed
    under another cannot use it, so the recommended switch to ``english``
    would otherwise leave every keyword search scanning the table.
    """
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert({"id": "a", "name": "running", "vector": vec(1)})
        assert "'simple'" in indexes_on(dsn, test_schema)["ov_context__fts_idx"]
    finally:
        adapter.close()

    adapter = build_existing(dsn, test_schema, text_search_config="english")
    try:
        assert adapter.ensure_indexes() == ["ov_context__fts_idx"]
        assert "'english'" in indexes_on(dsn, test_schema)["ov_context__fts_idx"]
        assert adapter.ensure_indexes() == []
    finally:
        adapter.close()


def test_ensure_indexes_leaves_indexes_it_did_not_create_alone(
    dsn: str, test_schema: str
) -> None:
    """A user's own index must survive reconciliation.

    Reconciliation drops and recreates, so touching an index this package did
    not create would silently destroy someone's tuning.
    """
    import psycopg

    adapter = build(dsn, test_schema)
    try:
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(
                f'CREATE INDEX mine_idx ON "{test_schema}".ov_context (level, name)'
            )
            conn.commit()
        assert adapter.ensure_indexes() == []
        assert "mine_idx" in indexes_on(dsn, test_schema)
    finally:
        adapter.close()


def test_a_field_named_fts_does_not_collide_with_the_full_text_index(
    dsn: str, test_schema: str
) -> None:
    """Two indexes must never resolve to one name.

    Joining name parts with underscores is ambiguous: a scalar index on a field
    called ``fts`` yields ``ov_context__fts_idx``, which is also the fixed name
    of the full-text index. Both would then be created under that name, each
    run destroying the other's.
    """
    meta = copy.deepcopy(META)
    meta["Fields"].append({"FieldName": "fts", "FieldType": "string"})
    meta["ScalarIndex"].append("fts")
    adapter = build(dsn, test_schema, meta=meta)
    try:
        names = indexes_on(dsn, test_schema)
        fts_index = names["ov_context__fts_idx"]
        assert "gin" in fts_index and "to_tsvector" in fts_index
        scalar = [
            definition
            for name, definition in names.items()
            if name.startswith("ov_context__fts_idx_") and "to_tsvector" not in definition
        ]
        assert len(scalar) == 1, names
        assert adapter.ensure_indexes() == []
    finally:
        adapter.close()


def test_concurrent_vectorless_upserts_do_not_deadlock(
    dsn: str, test_schema: str
) -> None:
    """Row locks must be taken in the order the rows are written.

    A record arriving without an embedding is checked against the stored row
    under ``FOR UPDATE``; every record is then written in ``sorted()`` order,
    which is Python's code-point order. The lock query used a bare ``ORDER
    BY``, so it followed the database collation instead -- and under
    ``en_US.utf8`` ``a_1`` sorts before ``a-b`` while in Python it sorts after.
    A writer sending vectorless records therefore locked those two rows in the
    opposite order to a writer sending records with embeddings, and the pair
    deadlocked.
    """
    keys = ["a_1", "a-b", "a_2", "a-c", "a_3", "a-d"]
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert([{"id": k, "level": 0, "vector": vec(1)} for k in keys])
        errors: list[BaseException] = []

        def churn(seed: int, *, with_vectors: bool) -> None:
            batch: list[dict[str, Any]] = [{"id": k, "level": seed} for k in keys]
            if with_vectors:
                for record in batch:
                    record["vector"] = vec(1)
            try:
                for _ in range(40):
                    adapter.upsert(batch)
            except BaseException as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=churn, args=(n,), kwargs={"with_vectors": n % 2 == 0})
            for n in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == [], errors[0]
    finally:
        adapter.close()


def test_ensure_indexes_skips_an_index_a_constraint_owns(
    dsn: str, test_schema: str
) -> None:
    """A constraint's index cannot be dropped, so it must not be attempted.

    ``DROP INDEX`` on it raises ``DependentObjectsStillExist``, which would
    abort the run with the indexes reconciled before it already rebuilt.
    """
    adapter = build(dsn, test_schema)
    try:
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(f'DROP INDEX "{test_schema}".ov_context__level_idx')
            cur.execute(
                f'ALTER TABLE "{test_schema}".ov_context '
                "ADD CONSTRAINT ov_context__level_idx UNIQUE (level)"
            )
            conn.commit()
        assert adapter.ensure_indexes() == []
        assert "ov_context__level_idx" in indexes_on(dsn, test_schema)
    finally:
        adapter.close()


def test_keyword_paging_is_stable_when_scores_tie(dsn: str, test_schema: str) -> None:
    """Paging a tied result set must not repeat or drop rows.

    ``ts_rank`` gives every row the same score here, and without a tiebreaker
    PostgreSQL is free to order the run differently per query -- so a page
    boundary inside it returns some rows twice and never returns others.
    """
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert(
            [
                {"id": f"r{n:04d}", "name": "quick brown fox", "vector": vec(1)}
                for n in range(300)
            ]
        )
        collection = adapter.get_collection()
        seen: list[str] = []
        for page in range(6):
            result = collection.search_by_keywords(
                "default", keywords=["fox"], limit=50, offset=page * 50
            )
            seen.extend(str(item.id) for item in result.data)
        assert len(seen) == 300
        assert len(set(seen)) == 300
    finally:
        adapter.close()


def test_scalar_paging_is_stable_when_the_sort_column_ties(
    dsn: str, test_schema: str
) -> None:
    """The same tie problem applies when sorting on a column of equal values."""
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert(
            [{"id": f"r{n:04d}", "level": 7, "vector": vec(1)} for n in range(300)]
        )
        collection = adapter.get_collection()
        seen: list[str] = []
        for page in range(6):
            result = collection.search_by_scalar(
                "default", "level", limit=50, offset=page * 50, order="desc"
            )
            seen.extend(str(item.id) for item in result.data)
        assert len(seen) == 300
        assert len(set(seen)) == 300
    finally:
        adapter.close()


def test_a_very_long_keyword_list_does_not_exhaust_the_parser(
    dsn: str, test_schema: str
) -> None:
    """Thousands of keywords must still run.

    One ``plainto_tsquery`` per term joined flat parses as a left-deep tree,
    and the parser recurses per level: past roughly 4200 terms PostgreSQL
    raised ``stack depth limit exceeded``.
    """
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert({"id": "a", "name": "needle", "vector": vec(1)})
        keywords = [f"term{n}" for n in range(8000)]
        keywords[4000] = "needle"
        result = adapter.get_collection().search_by_keywords(
            "default", keywords=keywords, limit=10
        )
        assert [str(item.id) for item in result.data] == ["a"]
    finally:
        adapter.close()


def test_the_index_method_cache_is_cleared_by_a_rebuild(
    dsn: str, test_schema: str
) -> None:
    """A cached resolver answer must not outlive the indexes it read.

    ``create_index`` clears the cache because it just changed the indexes. A
    resolver that read ``pg_indexes`` before that and wrote afterwards would
    reinstate the stale answer permanently, and with it the truncated ANN
    results the scan GUC exists to prevent.
    """
    adapter = build(dsn, test_schema, index_method="auto")
    try:
        collection = cast(Any, adapter.get_collection())._Collection__collection
        assert collection._resolved_index_method() == "flat"
        collection.create_index(
            "default", {"VectorIndex": {"Distance": "cosine", "IndexType": "hnsw"}}
        )
        assert collection._resolved_method is None
        assert collection._resolved_index_method() == "hnsw"
    finally:
        adapter.close()


def test_concurrent_int_key_upserts_do_not_deadlock(dsn: str, test_schema: str) -> None:
    """An integer key must lock rows in numeric order, as its column sorts.

    The write order came from ``str(key)``, which puts ``10`` before ``7``,
    while the lock query's ``ORDER BY`` on a bigint column is numeric. The two
    disagreed for any key set spanning a digit-count boundary.
    """
    keys = [7, 8, 9, 10, 11, 12]
    adapter = build(dsn, test_schema, meta=int_key_meta())
    try:
        adapter.upsert([{"id": k, "level": 0, "vector": vec(1)} for k in keys])
        errors: list[BaseException] = []

        def churn(seed: int, *, with_vectors: bool) -> None:
            batch: list[dict[str, Any]] = [{"id": k, "level": seed} for k in keys]
            if with_vectors:
                for record in batch:
                    record["vector"] = vec(1)
            try:
                for _ in range(40):
                    adapter.upsert(batch)
            except BaseException as exc:
                errors.append(exc)

        threads = [
            threading.Thread(target=churn, args=(n,), kwargs={"with_vectors": n % 2 == 0})
            for n in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == [], errors[0]
    finally:
        adapter.close()


def test_writers_with_different_vectorless_subsets_do_not_deadlock(
    dsn: str, test_schema: str
) -> None:
    """The pre-write lock must cover the whole batch, not part of it.

    Locking only the records that arrived without an embedding answers the
    question it was asked, but leaves two writers whose vectorless subsets
    differ claiming the shared rows in incompatible orders.
    """
    keys = ["k0", "k1", "k2", "k3", "k4", "k5"]
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert([{"id": k, "level": 0, "vector": vec(1)} for k in keys])
        errors: list[BaseException] = []

        def churn(seed: int, *, omit: set[str]) -> None:
            batch: list[dict[str, Any]] = []
            for key in keys:
                record: dict[str, Any] = {"id": key, "level": seed}
                if key not in omit:
                    record["vector"] = vec(1)
                batch.append(record)
            try:
                for _ in range(60):
                    adapter.upsert(batch)
            except BaseException as exc:
                errors.append(exc)

        threads = [
            threading.Thread(
                target=churn,
                args=(n,),
                kwargs={"omit": {"k4", "k5"} if n % 2 else {"k0", "k1"}},
            )
            for n in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == [], errors[0]
    finally:
        adapter.close()


def test_a_stale_resolver_read_does_not_overwrite_a_newer_answer(
    dsn: str, test_schema: str
) -> None:
    """A cached index method must not outlive the indexes it described.

    Clearing the cache is not enough on its own: a resolver that read
    ``pg_indexes`` before a rebuild and wrote after it found the cache empty
    -- exactly the post-clear state -- and reinstated its stale answer for
    good. The scan setting is then never emitted and a filtered ANN search
    silently returns fewer rows than asked for.

    The rebuild is injected into the resolver's own catalog query, so the
    interleaving under test happens for real rather than being described.
    """
    adapter = build(dsn, test_schema, index_method="auto")
    try:
        collection = cast(Any, adapter.get_collection())._Collection__collection
        assert collection._resolved_index_method() == "flat"
        with collection._lock:
            collection._resolved_method = None

        original = collection._execute
        injected: list[bool] = []

        def execute_then_rebuild(statement: Any, *args: Any, **kwargs: Any) -> Any:
            rows = original(statement, *args, **kwargs)
            if not injected and "indexdef" in statement.as_string(None):
                # The resolver has its answer -- `flat` -- and has not cached
                # it yet. The rebuild lands in that window.
                injected.append(True)
                collection.create_index(
                    "default",
                    {"VectorIndex": {"Distance": "cosine", "IndexType": "hnsw"}},
                )
            return rows

        collection._execute = execute_then_rebuild
        try:
            assert collection._resolved_index_method() == "hnsw"
        finally:
            collection._execute = original
        assert injected == [True]
        assert collection._resolved_index_method() == "hnsw"
    finally:
        adapter.close()


def test_a_quoted_field_name_survives_collection_creation(
    dsn: str, test_schema: str
) -> None:
    """A field name containing a quote must not break index bookkeeping.

    psycopg doubles a quote inside an identifier, so reading the index name
    back out of the rendered statement stopped at the doubled quote and the
    follow-up ``COMMENT ON INDEX`` addressed a table that does not exist.
    """
    meta = copy.deepcopy(META)
    meta["Fields"].append({"FieldName": 'we"ird', "FieldType": "string"})
    meta["ScalarIndex"].append('we"ird')
    adapter = build(dsn, test_schema, meta=meta)
    try:
        assert 'ov_context__we"ird_idx' in indexes_on(dsn, test_schema)
        assert adapter.ensure_indexes() == []
        adapter.upsert({"id": "a", 'we"ird': "x", "vector": vec(1)})
        assert adapter.get(["a"])[0]['we"ird'] == "x"
    finally:
        adapter.close()


def test_vector_paging_is_stable_when_distances_tie(dsn: str, test_schema: str) -> None:
    """Identical embeddings must not make a page boundary lose rows.

    Duplicate text is ordinary in a memory store, and identical embeddings
    score identically. Without a tiebreaker the run of equal distances may be
    ordered differently per query, so paging repeats some rows and skips
    others.
    """
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert(
            [{"id": f"r{n:04d}", "level": n, "vector": vec(1)} for n in range(300)]
        )
        collection = adapter.get_collection()
        seen: list[str] = []
        for page in range(6):
            result = collection.search_by_vector(
                "default", dense_vector=vec(1), limit=50, offset=page * 50
            )
            seen.extend(str(item.id) for item in result.data)
        assert len(seen) == 300
        assert len(set(seen)) == 300
    finally:
        adapter.close()


def test_create_index_does_not_stamp_an_index_it_skipped(
    dsn: str, test_schema: str
) -> None:
    """A fingerprint must describe the index that is there, not the one asked for.

    ``CREATE INDEX IF NOT EXISTS`` silently skips an existing name. Stamping
    regardless marks a stale index as current, and reconciliation then never
    looks at it again.
    """
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert({"id": "a", "name": "running", "vector": vec(1)})
    finally:
        adapter.close()

    adapter = build_existing(dsn, test_schema, text_search_config="english")
    try:
        collection = cast(Any, adapter.get_collection())._Collection__collection
        collection.create_index("default", {})
        # The index is still the `simple` one, so reconciliation must still
        # see it as out of date.
        assert "'simple'" in indexes_on(dsn, test_schema)["ov_context__fts_idx"]
        assert adapter.ensure_indexes() == ["ov_context__fts_idx"]
        assert "'english'" in indexes_on(dsn, test_schema)["ov_context__fts_idx"]
    finally:
        adapter.close()


def test_ensure_indexes_drops_a_full_text_index_it_no_longer_wants(
    dsn: str, test_schema: str
) -> None:
    """Narrowing ``keyword_fields`` to nothing must retire the index.

    Left in place it is rebuilt on every write while no query can reach it --
    `search_by_keywords` reports no text fields and returns nothing.
    """
    adapter = build(dsn, test_schema)
    try:
        assert "ov_context__fts_idx" in indexes_on(dsn, test_schema)
    finally:
        adapter.close()

    adapter = build_existing(dsn, test_schema, keyword_fields=["level"])
    try:
        assert adapter.ensure_indexes() == ["ov_context__fts_idx"]
        assert "ov_context__fts_idx" not in indexes_on(dsn, test_schema)
        assert adapter.ensure_indexes() == []
    finally:
        adapter.close()


def test_too_many_keywords_is_reported_not_hit(dsn: str, test_schema: str) -> None:
    """Past the parameter limit the caller gets a message, not a driver error."""
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert({"id": "a", "name": "needle", "vector": vec(1)})
        collection = adapter.get_collection()
        with pytest.raises(ValueError, match="at most 16384 distinct terms"):
            collection.search_by_keywords(
                "default", keywords=[f"t{n}" for n in range(20000)]
            )
        # Repeats cost parameters and change nothing, so they are collapsed.
        result = collection.search_by_keywords(
            "default", keywords=["needle"] * 30000, limit=5
        )
        assert [str(item.id) for item in result.data] == ["a"]
    finally:
        adapter.close()


def test_upsert_and_update_report_keys_the_same_way(dsn: str, test_schema: str) -> None:
    """Both must return the key as stored, as the engine's validator does."""
    adapter = build(dsn, test_schema, meta=int_key_meta())
    try:
        collection = adapter.get_collection()
        assert collection.upsert_data([{"id": 10, "level": 1, "vector": vec(1)}]).ids == [
            10
        ]
        assert collection.update_data([{"id": 10, "level": 2}]).ids == [10]
        assert [r["id"] for r in adapter.get([10])] == [10]
    finally:
        adapter.close()


@pytest.mark.parametrize("method", ["hnsw", "ivfflat"])
def test_ensure_indexes_keeps_the_ann_index(
    dsn: str, test_schema: str, method: str
) -> None:
    """Reconciliation must not touch the vector index.

    Its shape depends on the distance metric and method recorded at creation,
    so it is deliberately outside the reconcilable set -- which makes a
    fingerprint on it read as an index this version no longer wants. The sweep
    that retires those would drop it, and since `create_collection` returns
    early for an existing collection it is never rebuilt. Searches keep
    working, sequentially, so nothing reports the loss.
    """
    adapter = build(dsn, test_schema, index_method=method)
    try:
        adapter.upsert(
            [
                {"id": f"r{n:03d}", "level": n, "vector": vec(n % 20 + 1)}
                for n in range(60)
            ]
        )
        name = f"ov_context__vector_{method}_idx"
        assert name in indexes_on(dsn, test_schema)
        assert adapter.ensure_indexes() == []
        assert name in indexes_on(dsn, test_schema)
        assert adapter.ensure_indexes() == []
        assert name in indexes_on(dsn, test_schema)
    finally:
        adapter.close()


def test_ensure_indexes_survives_a_concurrent_drop(dsn: str, test_schema: str) -> None:
    """Two processes reconciling at once must not abort each other.

    A rolling upgrade runs this from every instance, and the loser of the race
    finds the index already gone. A bare ``DROP INDEX`` raises there, leaving
    the run half-applied -- the state the constraint handling exists to avoid.

    Asserted on the statement rather than raced for: the window is a few
    microseconds wide and reproduces perhaps one run in three, which is no use
    as a regression test.
    """
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert({"id": "a", "name": "text", "vector": vec(1)})
    finally:
        adapter.close()

    adapter = build_existing(dsn, test_schema, text_search_config="english")
    try:
        collection = cast(Any, adapter.get_collection())._Collection__collection
        seen: list[str] = []
        original = collection._pool.connection

        class RecordingCursor:
            """Cursor wrapper that keeps the SQL text of every statement."""

            def __init__(self, inner: Any) -> None:
                self._inner = inner

            def execute(self, statement: Any, *args: Any, **kwargs: Any) -> Any:
                seen.append(
                    statement.as_string(None)
                    if hasattr(statement, "as_string")
                    else str(statement)
                )
                return self._inner.execute(statement, *args, **kwargs)

            def __getattr__(self, item: str) -> Any:
                return getattr(self._inner, item)

        @contextlib.contextmanager
        def recording_connection(*args: Any, **kwargs: Any) -> Any:
            with original(*args, **kwargs) as conn:
                real_cursor = conn.cursor

                @contextlib.contextmanager
                def cursor(*a: Any, **k: Any) -> Any:
                    with real_cursor(*a, **k) as cur:
                        yield RecordingCursor(cur)

                conn.cursor = cursor
                yield conn

        collection._pool.connection = recording_connection
        try:
            assert collection.ensure_indexes() == ["ov_context__fts_idx"]
        finally:
            collection._pool.connection = original

        drops = [text for text in seen if text.startswith("DROP INDEX")]
        assert drops, seen
        assert all("IF EXISTS" in text for text in drops), drops
    finally:
        adapter.close()


def test_delete_claims_rows_in_the_order_writers_claim_them(
    dsn: str, test_schema: str
) -> None:
    """A delete must lock rows in the same order upserts do.

    A bare ``DELETE ... WHERE pk = ANY(...)`` locks in scan order, while
    writes lock in ``_sort_key`` order, so an overlapping batch deadlocks.
    The ordering is asserted on the plan rather than raced for: a deadlock
    reproduces about one run in five, which is no use as a regression test.
    """
    adapter = build(dsn, test_schema)
    try:
        keys = [f"k{n}" for n in range(8)]
        adapter.upsert([{"id": k, "level": 0, "vector": vec(1)} for k in keys])
        collection = cast(Any, adapter.get_collection())._Collection__collection
        seen: list[str] = []
        original = collection._execute

        def record(statement: Any, *args: Any, **kwargs: Any) -> Any:
            seen.append(statement.as_string(None))
            return original(statement, *args, **kwargs)

        collection._execute = record
        try:
            collection.delete_data(list(reversed(keys)))
        finally:
            collection._execute = original

        deletes = [text for text in seen if text.startswith("DELETE")]
        assert len(deletes) == 1
        assert "FOR UPDATE" in deletes[0]
        assert 'ORDER BY "id" COLLATE "C"' in deletes[0]
        assert adapter.count() == 0
    finally:
        adapter.close()


def test_deletes_and_upserts_do_not_deadlock(dsn: str, test_schema: str) -> None:
    """The ordering above, exercised under real contention."""
    keys = [f"k{n}" for n in range(8)]
    adapter = build(dsn, test_schema)
    try:
        errors: list[BaseException] = []

        def write(seed: int) -> None:
            batch = [{"id": k, "level": seed, "vector": vec(1)} for k in keys]
            try:
                for _ in range(150):
                    adapter.upsert(batch)
            except BaseException as exc:
                errors.append(exc)

        def wipe() -> None:
            try:
                for _ in range(150):
                    adapter.delete(ids=list(keys))
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=write, args=(n,)) for n in range(4)]
        threads += [threading.Thread(target=wipe) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert errors == [], errors[0]
    finally:
        adapter.close()


def test_update_data_reports_a_missing_primary_key(dsn: str, test_schema: str) -> None:
    """The record-level check must run before the key is coerced for sorting."""
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert({"id": "a", "level": 0, "vector": vec(1)})
        with pytest.raises(ValueError, match="update_data record is missing 'id'"):
            adapter.get_collection().update_data([{"level": 9}])
    finally:
        adapter.close()


def path_key_meta() -> dict[str, Any]:
    """Return META with a path-typed primary key."""
    meta = copy.deepcopy(META)
    meta["Fields"] = [
        {"FieldName": "id", "FieldType": "path", "IsPrimaryKey": True},
        *[f for f in meta["Fields"] if f["FieldName"] != "id"],
    ]
    return meta


def test_search_by_id_on_a_path_key_excludes_only_the_seed_row(
    dsn: str, test_schema: str
) -> None:
    """Excluding the seed row must not exclude everything beneath it.

    A path field's ``must_not`` is a subtree test, so the self-exclusion took
    the whole subtree with it: neighbours of ``/a`` lost ``/a/b`` and
    ``/a/b/c`` as well.
    """
    keys = ["/a", "/a/b", "/a/b/c", "/z"]
    adapter = build(dsn, test_schema, meta=path_key_meta())
    try:
        adapter.upsert([{"id": k, "level": 0, "vector": vec(1)} for k in keys])
        result = adapter.get_collection().search_by_id("default", "/a", limit=10)
        assert sorted(str(item.id) for item in result.data) == ["/a/b", "/a/b/c", "/z"]
    finally:
        adapter.close()


def test_a_float_primary_key_is_usable(dsn: str, test_schema: str) -> None:
    """A key type the schema parser accepts must actually work.

    Without a validator the values reached the lock array unchecked, and
    psycopg refused the resulting mixed list with ``cannot dump lists of mixed
    types``.
    """
    # Mixed on purpose: without a validator these reach the lock array as
    # float, str and int, and psycopg refuses a heterogeneous list.
    given: list[Any] = [1.5, "-2.25", 3]
    keys: list[Any] = [1.5, -2.25, 3.0]
    meta = copy.deepcopy(META)
    meta["Fields"] = [
        {"FieldName": "id", "FieldType": "float32", "IsPrimaryKey": True},
        *[f for f in meta["Fields"] if f["FieldName"] != "id"],
    ]
    adapter = build(dsn, test_schema, meta=meta)
    try:
        adapter.upsert([{"id": k, "level": 0, "vector": vec(1)} for k in given])
        assert sorted(str(r["id"]) for r in adapter.get(keys)) == sorted(
            str(k) for k in keys
        )
        # Rewriting rows that have no embedding takes the pre-write lock,
        # whose key array is where unvalidated mixed types are refused with
        # `cannot dump lists of mixed types`.
        with psycopg.connect(dsn) as conn, conn.cursor() as cur:
            cur.execute(
                sql.SQL("UPDATE {}.ov_context SET vector = NULL").format(
                    sql.Identifier(test_schema)
                )
            )
            conn.commit()
        adapter.upsert([{"id": k, "level": 1} for k in given])
        assert adapter.get([keys[0]])[0]["level"] == 1
        assert adapter.delete(ids=keys) == len(keys)
    finally:
        adapter.close()


@pytest.mark.parametrize("ov_type", ["bool", "vector", "geo_point"])
def test_an_unusable_primary_key_type_is_refused_up_front(ov_type: str) -> None:
    """A key that cannot work must fail when declared, not on the first write.

    A `bool` key never reaches the database at all: OpenViking's own wrapper
    replaces a falsy id with a generated one, so `False` arrives as a UUID.
    """
    meta = copy.deepcopy(META)
    meta["Fields"] = [
        {"FieldName": "id", "FieldType": ov_type, "IsPrimaryKey": True, "Dim": DIM},
        *[f for f in meta["Fields"] if f["FieldName"] != "id"],
    ]
    with pytest.raises(ValueError, match="primary key"):
        CollectionSchema.from_meta(meta)


def test_a_non_finite_primary_key_is_refused(dsn: str, test_schema: str) -> None:
    """NaN equals nothing, itself included, so a row keyed on it is lost."""
    meta = copy.deepcopy(META)
    meta["Fields"] = [
        {"FieldName": "id", "FieldType": "float32", "IsPrimaryKey": True},
        *[f for f in meta["Fields"] if f["FieldName"] != "id"],
    ]
    adapter = build(dsn, test_schema, meta=meta)
    try:
        with pytest.raises(ValueError, match="cannot store"):
            adapter.upsert({"id": float("nan"), "level": 0, "vector": vec(1)})
    finally:
        adapter.close()


def test_reads_report_an_unmatchable_key_as_missing(dsn: str, test_schema: str) -> None:
    """A key no stored row can equal is not found, not an error.

    ``LocalCollection`` hashes ``str(key)`` and simply misses, so raising here
    would turn a lookup that returns nothing into a crash.
    """
    adapter = build(dsn, test_schema)
    try:
        adapter.upsert({"id": "a", "level": 0, "vector": vec(1)})
        collection = adapter.get_collection()
        assert collection.fetch_data([None]).ids_not_exist == [None]
        assert collection.delete_data([None]) is True
        assert adapter.count() == 1
        assert collection.search_by_id("default", None).data == []
        assert collection.search_by_id("default", "   ").data == []
    finally:
        adapter.close()


def test_an_integer_key_read_that_cannot_match_is_not_found(
    dsn: str, test_schema: str
) -> None:
    """A lookup no bigint can equal is a miss, not a crash.

    The engine hashes ``str(key)``, so ``"nope"`` and ``None`` simply find
    nothing there. Raising would turn an empty result into an error.
    """
    adapter = build(dsn, test_schema, meta=int_key_meta())
    try:
        adapter.upsert({"id": 1, "level": 0, "vector": vec(1)})
        collection = adapter.get_collection()
        assert collection.fetch_data([None]).ids_not_exist == [None]
        assert collection.fetch_data(["nope"]).ids_not_exist == ["nope"]
        assert collection.fetch_data([1]).ids_not_exist == []
        assert collection.search_by_id("default", "nope").data == []
        assert collection.delete_data(["nope"]) is True
        assert adapter.count() == 1
    finally:
        adapter.close()
