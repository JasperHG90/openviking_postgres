"""End-to-end tests against a live PostgreSQL + pgvector.

Run with::

    OV_POSTGRES_TEST_DSN=postgresql://localhost/openviking_test \
        .venv/bin/python -m pytest tests/test_integration.py -v

The most valuable test here is ``test_filter_semantics_match_reference``: it
generates random records and random filters, evaluates each filter both in
PostgreSQL (through the adapter) and in Python (through OpenViking's own
``matches_filter``), and asserts the two agree on every row.  That is what
pins this backend to native behaviour rather than to my reading of it.
"""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterator
from typing import Any

import pytest
from openviking.storage.vectordb.index.cuvs_index import (
    UnsupportedCuVSFilterError,
    matches_filter,
)
from openviking_cli.utils.config.vectordb_config import VectorDBBackendConfig

from ov_postgres.adapter import PgVectorCollectionAdapter

pytestmark = pytest.mark.integration

# `CollectionAdapter.query` synthesises a random vector for filter-only queries,
# sized from `get_openviking_config().embedding.dimension` rather than from the
# collection, so the two must agree or those queries fail on a dimension
# mismatch. Rather than read the developer's real ~/.openviking config -- which
# would make the suite machine-dependent -- the `_fixed_embedding_dimension`
# fixture in conftest.py pins that lookup to this value.
DIM = 8

META: dict[str, Any] = {
    "CollectionName": "context",
    "Description": "test collection",
    "Fields": [
        {"FieldName": "id", "FieldType": "string", "IsPrimaryKey": True},
        {"FieldName": "uri", "FieldType": "path"},
        {"FieldName": "context_type", "FieldType": "string"},
        {"FieldName": "name", "FieldType": "string"},
        {"FieldName": "description", "FieldType": "string"},
        {"FieldName": "level", "FieldType": "int64"},
        {"FieldName": "active_count", "FieldType": "int64"},
        {"FieldName": "created_at", "FieldType": "date_time"},
        {"FieldName": "search_tags", "FieldType": "list<string>"},
        {"FieldName": "vector", "FieldType": "vector", "Dim": DIM},
        {"FieldName": "sparse_vector", "FieldType": "sparse_vector"},
    ],
    "ScalarIndex": [
        "uri",
        "context_type",
        "name",
        "level",
        "active_count",
        "created_at",
        "search_tags",
    ],
}

FIELD_TYPES = {f["FieldName"]: f["FieldType"] for f in META["Fields"]}


def make_adapter(dsn: str, schema: str, **params: object) -> PgVectorCollectionAdapter:
    """Build an adapter pointed at a throwaway schema."""
    config = VectorDBBackendConfig(
        backend="ov_postgres.adapter.PgVectorCollectionAdapter",
        name="context",
        index_name="default",
        distance_metric="cosine",
        custom_params={"dsn": dsn, "schema": schema, **params},
    )
    return PgVectorCollectionAdapter.from_config(config)


@pytest.fixture
def adapter(dsn: str, test_schema: str) -> Iterator[PgVectorCollectionAdapter]:
    """Yield an adapter with the context collection already created."""
    inst = make_adapter(dsn, test_schema)
    inst.create_collection(
        "context", META, distance="cosine", sparse_weight=0.0, index_name="default"
    )
    try:
        yield inst
    finally:
        inst.close()


def vec(seed: int) -> list[float]:
    """Return a deterministic pseudo-random vector for the given seed."""
    rng = random.Random(seed)
    return [rng.uniform(-1, 1) for _ in range(DIM)]


def test_create_is_idempotent_and_reports_existence(dsn: str, test_schema: str) -> None:
    """Creating an existing collection is a no-op, not an error."""
    inst = make_adapter(dsn, test_schema)
    assert inst.collection_exists() is False
    assert (
        inst.create_collection(
            "context", META, distance="cosine", sparse_weight=0.0, index_name="default"
        )
        is True
    )
    assert inst.collection_exists() is True
    # Second call must be a no-op, not an error.
    assert (
        inst.create_collection(
            "context", META, distance="cosine", sparse_weight=0.0, index_name="default"
        )
        is False
    )
    inst.close()


def test_reopen_finds_existing_collection(dsn: str, test_schema: str) -> None:
    """A fresh adapter binds to a collection created by an earlier one."""
    first = make_adapter(dsn, test_schema)
    first.create_collection(
        "context", META, distance="cosine", sparse_weight=0.0, index_name="default"
    )
    first.upsert({"id": "a", "uri": "viking://x", "vector": vec(1)})
    first.close()

    second = make_adapter(dsn, test_schema)
    assert second.collection_exists() is True
    assert len(second.get(["a"])) == 1
    second.close()


def test_drop_collection(adapter: PgVectorCollectionAdapter) -> None:
    """Dropping removes the collection and reports it gone."""
    adapter.upsert({"id": "a", "vector": vec(1)})
    assert adapter.drop_collection() is True
    assert adapter.collection_exists() is False


def test_get_collection_info_roundtrips_meta(adapter: PgVectorCollectionAdapter) -> None:
    """Stored metadata comes back exactly as it was declared."""
    info = adapter.get_collection_info()
    assert info["CollectionName"] == "context"
    assert len(info["Fields"]) == len(META["Fields"])


def test_index_registry(adapter: PgVectorCollectionAdapter) -> None:
    """Index bundles are listed, found, and readable by name."""
    coll = adapter.get_collection()
    assert coll.list_indexes() == ["default"]
    assert coll.has_index("default") is True
    assert coll.has_index("nope") is False
    assert coll.get_index_meta_data("default")["IndexName"] == "default"


def test_upsert_get_roundtrip(adapter: PgVectorCollectionAdapter) -> None:
    """Written fields read back with their types intact."""
    adapter.upsert(
        {
            "id": "a",
            "uri": "viking://user/default/notes",
            "name": "note",
            "level": 2,
            "created_at": "2026-03-22T08:39:45Z",
            "search_tags": ["x", "y"],
            "vector": vec(1),
        }
    )
    records = adapter.get(["a"])
    assert len(records) == 1
    record = records[0]
    assert record["id"] == "a"
    assert record["name"] == "note"
    assert record["level"] == 2
    assert record["search_tags"] == ["x", "y"]
    # URIs are stored path-style and decoded back on read.
    assert record["uri"] == "viking://user/default/notes"


def test_upsert_generates_ids_when_absent(adapter: PgVectorCollectionAdapter) -> None:
    """A record with no primary key is assigned one."""
    ids = adapter.upsert([{"name": "x", "vector": vec(1)}])
    assert len(ids) == 1 and ids[0]
    assert adapter.get(ids)[0]["name"] == "x"


def test_upsert_updates_existing_row(adapter: PgVectorCollectionAdapter) -> None:
    """Upserting a known key replaces rather than inserts."""
    adapter.upsert({"id": "a", "name": "before", "vector": vec(1)})
    adapter.upsert({"id": "a", "name": "after", "vector": vec(2)})
    assert adapter.get(["a"])[0]["name"] == "after"
    assert adapter.count() == 1


def test_unknown_fields_survive_in_extra(adapter: PgVectorCollectionAdapter) -> None:
    """Fields outside the schema round-trip through the JSONB column."""
    adapter.upsert({"id": "a", "custom_thing": {"k": 1}, "vector": vec(1)})
    assert adapter.get(["a"])[0]["custom_thing"] == {"k": 1}


def test_partial_update_leaves_other_columns(adapter: PgVectorCollectionAdapter) -> None:
    """``update_data`` writes only the columns it was given."""
    adapter.upsert({"id": "a", "name": "keep", "level": 1, "vector": vec(1)})
    adapter.update_data([{"id": "a", "level": 9}])
    record = adapter.get(["a"])[0]
    assert record["level"] == 9
    assert record["name"] == "keep"


def test_fetch_reports_missing_ids(adapter: PgVectorCollectionAdapter) -> None:
    """Keys that match nothing are reported back separately."""
    adapter.upsert({"id": "a", "vector": vec(1)})
    result = adapter.get_collection().fetch_data(["a", "ghost"])
    assert [item.id for item in result.items] == ["a"]
    assert result.ids_not_exist == ["ghost"]


def test_delete_by_id_and_by_filter(adapter: PgVectorCollectionAdapter) -> None:
    """Rows delete by explicit key and by filter."""
    adapter.upsert(
        [
            {"id": "a", "level": 1, "vector": vec(1)},
            {"id": "b", "level": 2, "vector": vec(2)},
            {"id": "c", "level": 2, "vector": vec(3)},
        ]
    )
    assert adapter.delete(ids=["a"]) == 1
    assert adapter.count() == 2
    assert adapter.delete(filter={"op": "must", "field": "level", "conds": [2]}) == 2
    assert adapter.count() == 0


def test_clear(adapter: PgVectorCollectionAdapter) -> None:
    """Clearing empties the table without dropping it."""
    adapter.upsert([{"id": str(i), "vector": vec(i)} for i in range(5)])
    assert adapter.clear() is True
    assert adapter.count() == 0


def test_vector_search_orders_by_similarity(adapter: PgVectorCollectionAdapter) -> None:
    """Nearer vectors score higher and rank first."""
    target = [1.0] + [0.0] * (DIM - 1)
    near = [0.99, 0.01] + [0.0] * (DIM - 2)
    far = [-1.0] + [0.0] * (DIM - 1)
    adapter.upsert(
        [
            {"id": "near", "vector": near},
            {"id": "far", "vector": far},
        ]
    )
    results = adapter.query(query_vector=target, limit=10)
    assert [r["id"] for r in results] == ["near", "far"]
    assert results[0]["_score"] > results[1]["_score"]


def test_vector_search_respects_filter(adapter: PgVectorCollectionAdapter) -> None:
    """A filter restricts the candidate set before ranking."""
    adapter.upsert(
        [
            {"id": "a", "level": 1, "vector": vec(1)},
            {"id": "b", "level": 2, "vector": vec(2)},
        ]
    )
    results = adapter.query(
        query_vector=vec(1),
        filter={"op": "must", "field": "level", "conds": [2]},
        limit=10,
    )
    assert [r["id"] for r in results] == ["b"]


def test_rows_without_vectors_rank_last_in_vector_search(
    adapter: PgVectorCollectionAdapter,
) -> None:
    """A row with no vector sorts last rather than disappearing.

    Excluding it looked right for a dense ranking, but the same code path
    serves filter-only queries, so the row also vanished from
    ``delete(filter=...)`` while ``count()`` still counted it.
    """
    adapter.upsert([{"id": "novec", "name": "x"}, {"id": "hasvec", "vector": vec(1)}])
    results = adapter.query(query_vector=vec(1), limit=10)
    assert [r["id"] for r in results] == ["hasvec", "novec"]


def test_limit_and_offset(adapter: PgVectorCollectionAdapter) -> None:
    """Paging returns disjoint, ordered pages."""
    adapter.upsert([{"id": f"r{i}", "level": i, "vector": vec(i)} for i in range(10)])
    page1 = adapter.query(order_by="level", order_desc=False, limit=3, offset=0)
    page2 = adapter.query(order_by="level", order_desc=False, limit=3, offset=3)
    assert [r["id"] for r in page1] == ["r0", "r1", "r2"]
    assert [r["id"] for r in page2] == ["r3", "r4", "r5"]


def test_scalar_ordering_both_directions(adapter: PgVectorCollectionAdapter) -> None:
    """Scalar sort honours ascending and descending."""
    adapter.upsert([{"id": f"r{i}", "level": i, "vector": vec(i)} for i in range(4)])
    asc = adapter.query(order_by="level", order_desc=False, limit=10)
    desc = adapter.query(order_by="level", order_desc=True, limit=10)
    assert [r["id"] for r in asc] == ["r0", "r1", "r2", "r3"]
    assert [r["id"] for r in desc] == ["r3", "r2", "r1", "r0"]


def test_output_fields_projection(adapter: PgVectorCollectionAdapter) -> None:
    """Only the requested columns come back."""
    adapter.upsert({"id": "a", "name": "n", "description": "d", "vector": vec(1)})
    record = adapter.query(query_vector=vec(1), output_fields=["name"], limit=1)[0]
    assert record["name"] == "n"
    assert "description" not in record


def test_keyword_search(adapter: PgVectorCollectionAdapter) -> None:
    """Full-text search finds a row by a word in its name."""
    adapter.upsert(
        [
            {"id": "a", "name": "postgres vector search", "vector": vec(1)},
            {"id": "b", "name": "completely unrelated", "vector": vec(2)},
        ]
    )
    results = adapter.search_by_keywords(query="postgres", limit=10)
    assert [r["id"] for r in results] == ["a"]


def test_keyword_search_covers_array_fields(adapter: PgVectorCollectionAdapter) -> None:
    """Array text fields are searchable too."""
    adapter.upsert({"id": "a", "search_tags": ["kubernetes"], "vector": vec(1)})
    assert [r["id"] for r in adapter.search_by_keywords(keywords=["kubernetes"])] == ["a"]


def test_count_with_and_without_filter(adapter: PgVectorCollectionAdapter) -> None:
    """Counting honours the filter it is given."""
    adapter.upsert(
        [
            {"id": "a", "level": 1, "vector": vec(1)},
            {"id": "b", "level": 2, "vector": vec(2)},
            {"id": "c", "level": 2, "vector": vec(3)},
        ]
    )
    assert adapter.count() == 3
    assert adapter.count({"op": "must", "field": "level", "conds": [2]}) == 2


def test_grouped_aggregate(adapter: PgVectorCollectionAdapter) -> None:
    """Grouped counts bucket rows by a scalar column."""
    adapter.upsert(
        [
            {"id": "a", "context_type": "memory", "vector": vec(1)},
            {"id": "b", "context_type": "memory", "vector": vec(2)},
            {"id": "c", "context_type": "skill", "vector": vec(3)},
        ]
    )
    result = adapter.get_collection().aggregate_data(
        index_name="default", op="count", field="context_type"
    )
    assert result.agg == {"memory": 2, "skill": 1}


@pytest.fixture
def path_data(adapter: PgVectorCollectionAdapter) -> list[str]:
    """Populate a small URI hierarchy and return the URIs."""
    uris = [
        "viking://a",
        "viking://a/b",
        "viking://a/b/c",
        "viking://a/b/c/d",
        "viking://other",
    ]
    adapter.upsert([{"id": u, "uri": u, "vector": vec(i)} for i, u in enumerate(uris)])
    return uris


@pytest.mark.parametrize(
    "depth,expected",
    [
        (0, {"viking://a"}),
        (1, {"viking://a", "viking://a/b"}),
        (2, {"viking://a", "viking://a/b", "viking://a/b/c"}),
        (-1, {"viking://a", "viking://a/b", "viking://a/b/c", "viking://a/b/c/d"}),
    ],
)
def test_path_scope_depth(
    adapter: PgVectorCollectionAdapter,
    path_data: list[str],
    depth: int,
    expected: set[str],
) -> None:
    """Depth scoping selects exactly the subtree within budget."""
    results = adapter.query(
        filter={
            "op": "must",
            "field": "uri",
            "conds": ["/a"],
            "para": f"-d={depth}",
        },
        limit=100,
    )
    assert {r["id"] for r in results} == expected


def test_path_scope_root_matches_everything(
    adapter: PgVectorCollectionAdapter, path_data: list[str]
) -> None:
    """Scoping at the root matches every row."""
    results = adapter.query(
        filter={"op": "must", "field": "uri", "conds": ["/"], "para": "-d=-1"},
        limit=100,
    )
    assert len(results) == len(path_data)


def _random_record(rng: random.Random, index: int) -> dict[str, Any]:
    """Build one pseudo-random record spanning every filterable field.

    Some fields are omitted at random so the sweep covers NULL columns, and the
    URI set includes ``_`` and ``%`` so LIKE-metacharacter handling is exercised
    against the reference rather than assumed.
    """
    record: dict[str, Any] = {
        "id": f"r{index}",
        "uri": rng.choice(
            [
                "viking://a",
                "viking://a/b",
                "viking://a/b/c",
                "viking://z",
                "viking://",
                "viking://a_b",
                "viking://aXb/child",
                "viking://a%b",
                "viking://a-b/child",
            ]
        ),
        "context_type": rng.choice(["resource", "memory", "skill"]),
        "name": rng.choice(["alpha", "beta", "gamma", "alpha beta", "50%_x"]),
        "level": rng.choice([0, 1, 2]),
        "active_count": rng.randint(0, 20),
        "search_tags": rng.sample(["x", "y", "z"], k=rng.randint(0, 3)),
        "vector": vec(index),
    }
    # Drop a field now and then so NULL columns are part of the comparison.
    for field in ("name", "level", "search_tags", "context_type"):
        if rng.random() < 0.2:
            del record[field]
    return record


def _candidate_filters() -> list[dict[str, Any]]:
    """Return every leaf filter plus each pairwise and/or combination."""
    leaves: list[dict[str, Any]] = [
        {"op": "must", "field": "context_type", "conds": ["memory"]},
        {"op": "must", "field": "context_type", "conds": ["memory", "skill"]},
        {"op": "must_not", "field": "context_type", "conds": ["memory"]},
        {"op": "must", "field": "level", "conds": [1]},
        {"op": "must_not", "field": "level", "conds": [0, 2]},
        {"op": "range", "field": "active_count", "gte": 5},
        {"op": "range", "field": "active_count", "gt": 3, "lte": 15},
        {"op": "range_out", "field": "active_count", "gte": 5, "lt": 10},
        {"op": "contains", "field": "name", "substring": "alpha"},
        {"op": "contains", "field": "name", "substring": "zzz"},
        {"op": "must", "field": "search_tags", "conds": ["x"]},
        {"op": "must", "field": "search_tags", "conds": ["x", "z"]},
        {"op": "must_not", "field": "search_tags", "conds": ["y"]},
        {"op": "must", "field": "uri", "conds": ["/a"], "para": "-d=0"},
        {"op": "must", "field": "uri", "conds": ["/a"], "para": "-d=1"},
        {"op": "must", "field": "uri", "conds": ["/a"], "para": "-d=-1"},
        {"op": "must_not", "field": "uri", "conds": ["/a"], "para": "-d=1"},
        # LIKE metacharacters in a scoped path must stay literal.
        {"op": "must", "field": "uri", "conds": ["/a_b"], "para": "-d=-1"},
        {"op": "must", "field": "uri", "conds": ["/a%b"], "para": "-d=-1"},
        {"op": "must", "field": "uri", "conds": ["/a-b"], "para": "-d=-1"},
        # A missing value: `None in conds` matches, everything else does not.
        {"op": "must", "field": "name", "conds": [None]},
        {"op": "must_not", "field": "name", "conds": [None]},
        # Undeclared fields evaluate against a missing value.
        {"op": "must", "field": "nope", "conds": ["x"]},
        {"op": "must_not", "field": "nope", "conds": ["x"]},
        {"op": "contains", "field": "nope", "substring": "x"},
        # Empty operand lists.
        {"op": "must", "field": "context_type", "conds": []},
        {"op": "must_not", "field": "context_type", "conds": []},
        # Operands whose type cannot match the column.
        {"op": "must", "field": "level", "conds": ["not-a-number"]},
        {"op": "must", "field": "name", "conds": [7]},
        {"op": "contains", "field": "level", "substring": "1"},
        {"op": "range", "field": "name", "gte": 5},
        # Substring containing LIKE metacharacters.
        {"op": "contains", "field": "name", "substring": "50%_x"},
        # A wrapper around a leaf.
        {"filter": {"op": "must", "field": "context_type", "conds": ["memory"]}},
    ]
    combos: list[dict[str, Any]] = list(leaves)
    for left, right in itertools.combinations(leaves, 2):
        combos.append({"op": "and", "conds": [left, right]})
        combos.append({"op": "or", "conds": [left, right]})
    return combos


def test_filter_semantics_match_reference(adapter: PgVectorCollectionAdapter) -> None:
    """Every filter must select the same rows in SQL as in Python.

    ``matches_filter`` is the evaluator OpenViking's cuVS path uses, and it is
    the authoritative statement of what the DSL means.  Anywhere the SQL
    disagrees, the SQL is wrong.
    """
    rng = random.Random(20260829)
    records = [_random_record(rng, i) for i in range(60)]
    adapter.upsert(records)

    # The adapter path-encodes `uri` on write, so the Python evaluator must see
    # the same encoded value the database stores.
    encoded = []
    for record in records:
        row = dict(record)
        row["uri"] = PgVectorCollectionAdapter._encode_uri_field_value(row["uri"])
        encoded.append(row)

    checked = 0
    skipped: list[dict[str, Any]] = []
    for node in _candidate_filters():
        try:
            expected = {
                row["id"] for row in encoded if matches_filter(row, node, FIELD_TYPES)
            }
        except UnsupportedCuVSFilterError:
            # The reference refuses some nodes (date_time, geo_point). Those
            # cannot be compared against anything, so they are counted and
            # asserted on below rather than quietly dropped.
            skipped.append(node)
            continue

        actual = {r["id"] for r in adapter.query(filter=node, limit=1000)}
        assert actual == expected, (
            f"filter {node!r}\n  sql-only: {sorted(actual - expected)}"
            f"\n  py-only:  {sorted(expected - actual)}"
        )
        checked += 1

    assert checked > 100, f"expected a broad sweep, only compared {checked} filters"
    # Every candidate must actually be compared. A new filter the reference
    # cannot evaluate would otherwise pass this test without being verified.
    assert not skipped, f"{len(skipped)} filters were never compared: {skipped[:3]}"


@pytest.mark.parametrize(
    "scoped,sibling",
    [
        ("my_notes", "myXnotes"),
        ("my_notes", "my-notes"),
        ("a%b", "aQQb"),
        ("under_score", "underXscore"),
    ],
)
def test_path_scope_does_not_leak_across_siblings(
    adapter: PgVectorCollectionAdapter, scoped: str, sibling: str
) -> None:
    """`_` and `%` in a scoped path must not act as LIKE wildcards.

    The reference uses ``startswith``; an unescaped LIKE would pull in sibling
    subtrees, which breaks the containment that path scope exists to provide.
    """
    adapter.upsert(
        [
            {
                "id": "in",
                "uri": f"viking://user/default/{scoped}/private",
                "vector": vec(1),
            },
            {
                "id": "out",
                "uri": f"viking://user/default/{sibling}/other",
                "vector": vec(2),
            },
        ]
    )
    node = {
        "op": "must",
        "field": "uri",
        "conds": [f"/user/default/{scoped}"],
        "para": "-d=-1",
    }
    assert {r["id"] for r in adapter.query(filter=node, limit=50)} == {"in"}

    negated = dict(node, op="must_not")
    assert {r["id"] for r in adapter.query(filter=negated, limit=50)} == {"out"}


def test_path_scope_ignores_surrounding_whitespace(
    adapter: PgVectorCollectionAdapter,
) -> None:
    """Whitespace is stripped as Python's ``str.strip()`` does, not just spaces."""
    adapter.upsert({"id": "a", "uri": "viking://user/default/notes", "vector": vec(1)})
    node = {
        "op": "must",
        "field": "uri",
        "conds": ["\t/user/default/notes\n"],
        "para": "-d=-1",
    }
    assert {r["id"] for r in adapter.query(filter=node, limit=10)} == {"a"}


@pytest.mark.parametrize(
    "node",
    [
        {"op": "contains", "field": "level", "substring": "1"},
        {"op": "range", "field": "name", "gte": 5},
        {"op": "range", "field": "search_tags", "gte": 1},
        {"op": "must", "field": "name", "conds": ["alpha", 1]},
        {"op": "must", "field": "name", "conds": [1]},
        {"op": "must", "field": "uri", "conds": [123]},
        {"op": "must", "field": "level", "conds": ["not-a-number"]},
    ],
)
def test_type_mismatched_operands_match_nothing(
    adapter: PgVectorCollectionAdapter, node: dict[str, Any]
) -> None:
    """A mismatched operand selects no rows instead of raising.

    The reference compares with ``in``/``==``/``<`` and catches ``TypeError``,
    so it yields an empty result. Binding the operand raw would instead make
    PostgreSQL raise ``UndefinedFunction`` and fail the whole query.
    """
    adapter.upsert(
        [
            {
                "id": "a",
                "name": "alpha",
                "level": 1,
                "uri": "viking://a",
                "vector": vec(1),
            },
            {"id": "b", "name": "beta", "level": 2, "vector": vec(2)},
        ]
    )
    results = adapter.query(filter=node, limit=50)
    assert {r["id"] for r in results} <= {"a"}


def test_output_fields_excludes_out_of_schema_data(
    adapter: PgVectorCollectionAdapter,
) -> None:
    """An explicit projection must not smuggle `extra` back in."""
    adapter.upsert(
        {"id": "a", "name": "n", "vector": vec(1), "secret": "LEAK", "bulk": [1] * 50}
    )
    record = adapter.query(query_vector=vec(1), output_fields=["name"], limit=1)[0]
    assert record["name"] == "n"
    assert "secret" not in record
    assert "bulk" not in record


def test_extra_is_returned_when_no_projection_given(
    adapter: PgVectorCollectionAdapter,
) -> None:
    """Without a projection, out-of-schema fields still round-trip."""
    adapter.upsert({"id": "a", "custom": {"k": 1}, "vector": vec(1)})
    assert adapter.get(["a"])[0]["custom"] == {"k": 1}


def test_upsert_replaces_the_whole_row(adapter: PgVectorCollectionAdapter) -> None:
    """Upsert replaces, matching LocalCollection's whole-document write."""
    adapter.upsert({"id": "z", "name": "A", "level": 1, "vector": vec(1)})
    adapter.upsert({"id": "z", "name": "B", "vector": vec(1)})
    record = adapter.get(["z"])[0]
    assert record["name"] == "B"
    assert "level" not in record


def test_upsert_batch_applies_in_input_order(
    adapter: PgVectorCollectionAdapter,
) -> None:
    """A repeated primary key in one batch ends on its last occurrence."""
    adapter.upsert(
        [
            {"id": "w", "name": "N1", "vector": vec(1)},
            {"id": "w", "name": "N2", "level": 5, "vector": vec(1)},
            {"id": "w", "name": "N3", "vector": vec(1)},
        ]
    )
    record = adapter.get(["w"])[0]
    assert record["name"] == "N3"
    assert "level" not in record


def test_update_data_rejects_unknown_primary_key(
    adapter: PgVectorCollectionAdapter,
) -> None:
    """Updating a row that does not exist is an error, as on the local backend."""
    with pytest.raises(ValueError, match="not found"):
        adapter.get_collection().update_data([{"id": "ghost", "name": "x"}])


def test_conds_containing_none_matches_missing_values(
    adapter: PgVectorCollectionAdapter,
) -> None:
    """``None in conds`` matches a row whose column is absent."""
    adapter.upsert(
        [
            {"id": "named", "name": "alpha", "vector": vec(1)},
            {"id": "unnamed", "vector": vec(2)},
        ]
    )
    node = {"op": "must", "field": "name", "conds": [None]}
    assert {r["id"] for r in adapter.query(filter=node, limit=10)} == {"unnamed"}


def test_range_out_on_unknown_field_matches_everything(
    adapter: PgVectorCollectionAdapter,
) -> None:
    """A negated test against an undeclared field excludes nothing."""
    adapter.upsert([{"id": f"r{i}", "vector": vec(i)} for i in range(3)])
    node = {"op": "range_out", "field": "nope", "gte": 1}
    assert len(adapter.query(filter=node, limit=10)) == 3


def test_read_modify_write_preserves_vectors(
    adapter: PgVectorCollectionAdapter,
) -> None:
    """A fetched record fed back into upsert must keep its embedding.

    ``upsert_data`` replaces the whole row, and OpenViking's
    ``increment_active_count`` and ``update_uri_mapping`` both do
    ``upsert(record | {...})`` on a record from ``get()``. If ``get()`` omitted
    the vector, the upsert would write NULL over it and the row would vanish
    from every vector search while still existing in the table.
    """
    adapter.upsert(
        {
            "id": "a",
            "name": "doc",
            "active_count": 0,
            "sparse_vector": {"7": 1.0},
            "vector": vec(1),
        }
    )
    assert [r["id"] for r in adapter.query(query_vector=vec(1), limit=5)] == ["a"]

    record = adapter.get(["a"])[0]
    assert "vector" in record, "get() must return vectors for read-modify-write"
    assert "sparse_vector" in record

    adapter.upsert(record | {"active_count": 1})

    assert [r["id"] for r in adapter.query(query_vector=vec(1), limit=5)] == ["a"]
    after = adapter.get(["a"])[0]
    assert after["active_count"] == 1
    assert after["vector"] == record["vector"]
    assert after["sparse_vector"] == {"7": 1.0}


def test_search_projection_still_excludes_vectors(
    adapter: PgVectorCollectionAdapter,
) -> None:
    """Search results stay lean; only ``get()`` returns the full row.

    Guards the fix above from being widened into ``_output_columns``, which
    would put a 512-float vector in every search result.
    """
    adapter.upsert({"id": "a", "name": "doc", "vector": vec(1)})
    hit = adapter.query(query_vector=vec(1), limit=1)[0]
    assert "vector" not in hit
    assert "sparse_vector" not in hit


WIDE_META: dict[str, Any] = {
    "CollectionName": "context",
    "Description": "collection covering the field types the context schema omits",
    "Fields": [
        {"FieldName": "id", "FieldType": "string", "IsPrimaryKey": True},
        {"FieldName": "flag", "FieldType": "bool"},
        {"FieldName": "score", "FieldType": "float32"},
        {"FieldName": "counts", "FieldType": "list<int64>"},
        {"FieldName": "body", "FieldType": "text"},
        {"FieldName": "vector", "FieldType": "vector", "Dim": DIM},
    ],
    "ScalarIndex": ["flag", "score", "counts", "body"],
}

WIDE_FIELD_TYPES = {f["FieldName"]: f["FieldType"] for f in WIDE_META["Fields"]}


@pytest.fixture
def wide_adapter(dsn: str, test_schema: str) -> Iterator[PgVectorCollectionAdapter]:
    """Yield an adapter over a collection declaring the less common types."""
    inst = make_adapter(dsn, test_schema)
    inst.create_collection(
        "context", WIDE_META, distance="cosine", sparse_weight=0.0, index_name="default"
    )
    try:
        yield inst
    finally:
        inst.close()


def _wide_record(rng: random.Random, index: int) -> dict[str, Any]:
    """Build one pseudo-random record over bool, float32, list<int64> and text."""
    record: dict[str, Any] = {
        "id": f"w{index}",
        "flag": rng.choice([True, False]),
        "score": rng.choice([0.1, 0.5, 1.0, 2.5, 9.75]),
        "counts": rng.sample([1, 2, 3, 5, 8], k=rng.randint(0, 3)),
        "body": rng.choice(["alpha text", "beta text", "gamma", "50%_x"]),
        "vector": vec(index),
    }
    for field in ("flag", "score", "counts", "body"):
        if rng.random() < 0.2:
            del record[field]
    return record


def _wide_candidate_filters() -> list[dict[str, Any]]:
    """Return leaf filters over the wide schema, plus pairwise combinations."""
    leaves: list[dict[str, Any]] = [
        {"op": "must", "field": "flag", "conds": [True]},
        {"op": "must", "field": "flag", "conds": [False]},
        {"op": "must_not", "field": "flag", "conds": [True]},
        {"op": "must", "field": "score", "conds": [0.1]},
        {"op": "must", "field": "score", "conds": [0.1, 9.75]},
        {"op": "must_not", "field": "score", "conds": [0.5]},
        {"op": "range", "field": "score", "gte": 0.5},
        {"op": "range", "field": "score", "gt": 0.1, "lte": 2.5},
        {"op": "range_out", "field": "score", "gte": 0.5, "lt": 2.5},
        {"op": "must", "field": "counts", "conds": [3]},
        {"op": "must", "field": "counts", "conds": [1, 8]},
        {"op": "must_not", "field": "counts", "conds": [2]},
        {"op": "must", "field": "body", "conds": ["gamma"]},
        {"op": "must_not", "field": "body", "conds": ["gamma"]},
        {"op": "contains", "field": "body", "substring": "text"},
        {"op": "contains", "field": "body", "substring": "50%_x"},
        {"op": "range", "field": "counts", "gte": 2},
        {"op": "must", "field": "flag", "conds": [None]},
        {"op": "must", "field": "score", "conds": [None]},
        {"op": "must", "field": "flag", "conds": ["not-a-bool"]},
        {"op": "must", "field": "score", "conds": ["not-a-number"]},
        {"op": "must", "field": "counts", "conds": ["not-an-int"]},
        {"op": "contains", "field": "score", "substring": "0"},
        {"op": "must", "field": "body", "conds": [7]},
    ]
    combos: list[dict[str, Any]] = list(leaves)
    for left, right in itertools.combinations(leaves, 2):
        combos.append({"op": "and", "conds": [left, right]})
        combos.append({"op": "or", "conds": [left, right]})
    return combos


def test_filter_semantics_match_reference_for_remaining_types(
    wide_adapter: PgVectorCollectionAdapter,
) -> None:
    """Extend the differential comparison to bool, float32, list<int64> and text.

    The main sweep runs against the ``context`` schema, which declares only
    ``string``, ``path``, ``int64`` and ``list<string>``. The casts this
    backend applies for the other types -- ``::real[]`` for float32,
    ``::bigint[]`` for list<int64> -- and the deliberate bool/int64 exclusion in
    ``_is_comparable`` were previously asserted only against hand-written
    expectations. This checks them against the reference evaluator instead.
    """
    rng = random.Random(20260830)
    records = [_wide_record(rng, i) for i in range(60)]
    wide_adapter.upsert(records)

    checked = 0
    skipped: list[dict[str, Any]] = []
    for node in _wide_candidate_filters():
        try:
            expected = {
                row["id"]
                for row in records
                if matches_filter(row, node, WIDE_FIELD_TYPES)
            }
        except UnsupportedCuVSFilterError:
            skipped.append(node)
            continue

        actual = {r["id"] for r in wide_adapter.query(filter=node, limit=1000)}
        assert actual == expected, (
            f"filter {node!r}\n  sql-only: {sorted(actual - expected)}"
            f"\n  py-only:  {sorted(expected - actual)}"
        )
        checked += 1

    assert checked > 400, f"expected a broad sweep, only compared {checked} filters"
    assert not skipped, f"{len(skipped)} filters were never compared: {skipped[:3]}"


@pytest.mark.parametrize("field", ["search_tags", "level"])
@pytest.mark.parametrize("op", ["range", "range_out"])
def test_range_without_bounds_matches_present_values(
    adapter: PgVectorCollectionAdapter, field: str, op: str
) -> None:
    """A range carrying no bounds tests only for a present value.

    ``Range(field)`` with every bound left as None compiles to a bare
    ``{"op": "range", "field": ...}``. The reference rejects only a missing
    *value*, so a present one passes every absent check and matches. Guarding
    array columns unconditionally made this return nothing.
    """
    records: list[dict[str, Any]] = [
        {"id": "has", "level": 3, "search_tags": ["x"], "vector": vec(1)},
        {"id": "empty", "vector": vec(2)},
    ]
    adapter.upsert(records)

    node = {"op": op, "field": field}
    expected = {r["id"] for r in records if matches_filter(r, node, FIELD_TYPES)}
    actual = {r["id"] for r in adapter.query(filter=node, limit=50)}
    assert actual == expected


@pytest.mark.parametrize(
    "node",
    [
        {"op": "must", "field": "flag", "conds": [1]},
        {"op": "must", "field": "flag", "conds": [0]},
        {"op": "must_not", "field": "flag", "conds": [1]},
        {"op": "range", "field": "flag", "gte": 0},
        {"op": "must", "field": "flag", "conds": [2]},
        {"op": "must", "field": "score", "conds": [True]},
        {"op": "range", "field": "score", "gte": True},
    ],
)
def test_bool_and_int_operands_interoperate(
    wide_adapter: PgVectorCollectionAdapter, node: dict[str, Any]
) -> None:
    """Python compares ``True == 1``; the SQL must agree in both directions.

    PostgreSQL has no implicit boolean/bigint comparison, so these operands are
    converted rather than dropped. ``2`` against a boolean column still matches
    nothing, because the reference finds ``True == 2`` false as well.
    """
    records: list[dict[str, Any]] = [
        {"id": "t", "flag": True, "score": 1.0, "vector": vec(1)},
        {"id": "f", "flag": False, "score": 0.0, "vector": vec(2)},
        {"id": "none", "vector": vec(3)},
    ]
    wide_adapter.upsert(records)

    expected = {r["id"] for r in records if matches_filter(r, node, WIDE_FIELD_TYPES)}
    actual = {r["id"] for r in wide_adapter.query(filter=node, limit=50)}
    assert actual == expected


def test_rows_without_vectors_remain_findable_and_deletable(
    adapter: PgVectorCollectionAdapter,
) -> None:
    """A record with no embedding must not become invisible to filters.

    ``CollectionAdapter.query`` synthesises a random vector for filter-only
    queries, so excluding vectorless rows from vector search also hid them from
    ``delete(filter=...)`` and ``scroll`` -- while ``count()`` still counted
    them, leaving a row that could be neither found nor removed.
    """
    adapter.upsert(
        [
            {"id": "novec", "level": 1, "vector": None},
            {"id": "hasvec", "level": 1, "vector": vec(1)},
        ]
    )
    assert adapter.count() == 2

    node = {"op": "must", "field": "level", "conds": [1]}
    assert {r["id"] for r in adapter.query(filter=node, limit=50)} == {
        "novec",
        "hasvec",
    }

    assert adapter.delete(filter=node) == 2
    assert adapter.count() == 0


def test_dense_search_still_ranks_vectors_first(
    adapter: PgVectorCollectionAdapter,
) -> None:
    """Sorting vectorless rows last must not disturb a real ranking."""
    adapter.upsert(
        [
            {"id": "novec"},
            {"id": "near", "vector": [1.0] + [0.0] * (DIM - 1)},
            {"id": "far", "vector": [-1.0] + [0.0] * (DIM - 1)},
        ]
    )
    ranked = [r["id"] for r in adapter.query(query_vector=[1.0] + [0.0] * (DIM - 1))]
    assert ranked[:2] == ["near", "far"]
    assert ranked[-1] == "novec"
