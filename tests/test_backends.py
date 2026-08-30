"""Integration coverage for the configurable code paths.

The main integration suite exercises the default configuration (exact search,
cosine distance, no sparse vectors).  These tests cover the options that change
which SQL gets generated: ANN index methods, distance metrics, and hybrid
dense+sparse scoring.
"""

from __future__ import annotations

import math
from typing import Any, cast

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

        filtered = inner._iterative_scan_setup(filtered=True)
        assert len(filtered) == 1
        assert "hnsw.iterative_scan" in filtered[0].as_string(None)

        # Needed for an unfiltered search too: an index scan visits at most
        # hnsw.ef_search candidates, so a bare LIMIT 200 returned 40 rows.
        assert len(inner._iterative_scan_setup(filtered=False)) == 1

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

    adapter = build(dsn, test_schema, index_method="hnsw")
    try:
        rng = random.Random(11)
        adapter.upsert(
            [
                {"id": f"r{i}", "vector": [rng.uniform(-1, 1) for _ in range(DIM)]}
                for i in range(1500)
            ]
        )
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
