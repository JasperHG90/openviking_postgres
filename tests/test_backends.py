"""Integration coverage for the configurable code paths.

The main integration suite exercises the default configuration (exact search,
cosine distance, no sparse vectors).  These tests cover the options that change
which SQL gets generated: ANN index methods, distance metrics, and hybrid
dense+sparse scoring.
"""

from __future__ import annotations

import math
from typing import Any

import psycopg
import pytest
from openviking_cli.utils.config.vectordb_config import VectorDBBackendConfig
from psycopg import sql
from pydantic import ValidationError

from ov_postgres import ddl
from ov_postgres.adapter import PgVectorCollectionAdapter

from .test_integration import DIM, META, vec

pytestmark = pytest.mark.integration


def build(
    dsn: str, schema: str, *, sparse_weight: float = 0.0, **params: object
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
        META,
        distance=config.distance_metric,
        sparse_weight=sparse_weight,
        index_name="default",
    )
    return adapter


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
                    ('{"name": "STALE"}', "a"),
                )
                conn.commit()

        record = adapter.get(["a"])[0]
        # The declared column wins: it holds the engine default, not the stale
        # copy that was planted in `extra`.
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

        filtered = inner._iterative_scan_setup(filtered=True)
        assert len(filtered) == 1
        assert "hnsw.iterative_scan" in filtered[0].as_string(None)

        # No filter means the index already returns a full page.
        assert inner._iterative_scan_setup(filtered=False) == []

        # An older pgvector must not be sent a GUC it does not know.
        inner._pgvector_version = (0, 7, 0)
        assert inner._iterative_scan_setup(filtered=True) == []
    finally:
        adapter.close()


def test_iterative_scan_not_used_for_exact_search(dsn: str, test_schema: str) -> None:
    """Exact search scans every row, so the GUC would only add cost."""
    adapter = build(dsn, test_schema, index_method="flat")
    try:
        inner = adapter.get_collection()._Collection__collection
        assert inner._iterative_scan_setup(filtered=True) == []
    finally:
        adapter.close()


def test_iterative_scan_off_disables_the_guc(dsn: str, test_schema: str) -> None:
    """Setting the option to `off` suppresses the statement entirely."""
    adapter = build(dsn, test_schema, index_method="hnsw", iterative_scan="off")
    try:
        inner = adapter.get_collection()._Collection__collection
        assert inner._iterative_scan_setup(filtered=True) == []
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
        rendered = inner._iterative_scan_setup(filtered=True)[0].as_string(None)
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
