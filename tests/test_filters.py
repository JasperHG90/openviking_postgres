"""Unit tests for the filter compiler. No database required.

Two kinds of check here:

1.  Differential tests against OpenViking's own implementations
    (``DataProcessor.parse_datetime_to_epoch_ms``, ``cuvs_index._parse_depth``)
    so value conversion cannot silently drift from the native engine.
2.  Structural tests on the emitted SQL, asserting the shape and the bound
    parameters rather than exact whitespace.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from openviking.storage.vectordb.index import cuvs_index
from openviking.storage.vectordb.utils.data_processor import DataProcessor

from ov_postgres.filters import (
    FilterCompiler,
    UnsupportedFilterError,
    parse_datetime_to_epoch_ms,
    parse_depth,
)
from ov_postgres.schema import CollectionSchema

META = {
    "CollectionName": "context",
    "Fields": [
        {"FieldName": "id", "FieldType": "string", "IsPrimaryKey": True},
        {"FieldName": "uri", "FieldType": "path"},
        {"FieldName": "name", "FieldType": "string"},
        {"FieldName": "level", "FieldType": "int64"},
        {"FieldName": "created_at", "FieldType": "date_time"},
        {"FieldName": "search_tags", "FieldType": "list<string>"},
        {"FieldName": "vector", "FieldType": "vector", "Dim": 4},
    ],
    "ScalarIndex": ["uri", "name", "level", "created_at", "search_tags"],
}


@pytest.fixture
def schema() -> CollectionSchema:
    """Return the parsed context-collection schema used by these tests."""
    return CollectionSchema.from_meta(META)


@pytest.fixture
def compiler(schema: CollectionSchema) -> FilterCompiler:
    """Return a filter compiler bound to the test schema."""
    return FilterCompiler(schema, tz_policy="local")


def render(
    compiler: FilterCompiler, node: Mapping[str, Any] | None
) -> tuple[str, list[Any]]:
    """Compile a filter node into its SQL text and bound parameters."""
    predicate, params = compiler.compile(node)
    return predicate.as_string(None), params


@pytest.mark.parametrize(
    "value",
    [
        "2026-03-22T08:39:45",
        "2026-03-22T08:39:45Z",
        "2026-03-22T08:39:45+02:00",
        "2026-08-29T17:38:52.865Z",
        1745000000000,
        1745000000000.0,
    ],
)
def test_datetime_matches_openviking(value: object) -> None:
    """Timestamp conversion agrees with OpenViking's own parser."""
    reference = DataProcessor(fields_dict={}, tz_policy="local")
    assert parse_datetime_to_epoch_ms(value, "local") == (
        reference.parse_datetime_to_epoch_ms(value)
    )


@pytest.mark.parametrize("value", ["not-a-date", "", "   "])
def test_datetime_rejects_garbage(value: object) -> None:
    """Unparseable timestamps raise rather than defaulting."""
    with pytest.raises(ValueError):
        parse_datetime_to_epoch_ms(value)


@pytest.mark.parametrize("para", [None, "", "-d=0", "-d=1", "-d=-1", " -d=3 ", "-d=10"])
def test_parse_depth_matches_openviking(para: object) -> None:
    """Depth parsing agrees with OpenViking's ``_parse_depth``."""
    assert parse_depth(para) == cuvs_index._parse_depth(para)


@pytest.mark.parametrize("para", ["-depth=1", "junk", "-d=", "-d=x"])
def test_parse_depth_rejects_garbage(para: object) -> None:
    """A malformed depth parameter is rejected."""
    with pytest.raises((UnsupportedFilterError, cuvs_index.UnsupportedCuVSFilterError)):
        parse_depth(para)


def test_empty_filter_is_true(compiler: FilterCompiler) -> None:
    """An absent filter matches every row."""
    text, params = render(compiler, None)
    assert text == "TRUE"
    assert params == []


def test_eq_becomes_any(compiler: FilterCompiler) -> None:
    """Equality compiles to an indexable ``= ANY`` test."""
    text, params = render(compiler, {"op": "must", "field": "name", "conds": ["a"]})
    assert "= ANY(" in text
    assert params == [["a"]]


def test_in_binds_all_values(compiler: FilterCompiler) -> None:
    """Every value of an ``In`` filter travels as one bound list."""
    text, params = render(compiler, {"op": "must", "field": "level", "conds": [0, 1, 2]})
    assert "= ANY(" in text
    assert params == [[0, 1, 2]]


def test_array_field_uses_overlap(compiler: FilterCompiler) -> None:
    """Array fields use ``&&``, matching the reference's any-of semantics."""
    text, params = render(
        compiler, {"op": "must", "field": "search_tags", "conds": ["x", "y"]}
    )
    assert "&&" in text
    assert params == [["x", "y"]]


def test_must_not_is_null_safe(compiler: FilterCompiler) -> None:
    """NULL must count as 'not matching', so a bare NOT would be wrong."""
    text, _ = render(compiler, {"op": "must_not", "field": "name", "conds": ["a"]})
    assert "COALESCE" in text.upper()


def test_path_uses_helper_function_with_depth(compiler: FilterCompiler) -> None:
    """Path scoping calls the SQL helper with the parsed depth."""
    text, params = render(
        compiler,
        {"op": "must", "field": "uri", "conds": ["/a/b"], "para": "-d=2"},
    )
    assert "ov_path_matches" in text
    assert params == ["/a/b", 2]


def test_path_without_para_is_unlimited_depth(compiler: FilterCompiler) -> None:
    """A path filter with no ``-d=`` parameter searches at unlimited depth."""
    _, params = render(compiler, {"op": "must", "field": "uri", "conds": ["/a"]})
    assert params == ["/a", -1]


def test_para_rejected_on_non_path_field(compiler: FilterCompiler) -> None:
    """A depth parameter on a non-path field is an error, not silently ignored."""
    with pytest.raises(UnsupportedFilterError):
        render(
            compiler,
            {"op": "must", "field": "name", "conds": ["a"], "para": "-d=1"},
        )


def test_range_converts_datetime_bounds(compiler: FilterCompiler) -> None:
    """Range bounds on a timestamp column convert to epoch milliseconds."""
    expected = parse_datetime_to_epoch_ms("2026-01-01T00:00:00")
    _, params = render(
        compiler,
        {"op": "range", "field": "created_at", "gte": "2026-01-01T00:00:00"},
    )
    assert params == [expected]


def test_time_range_is_treated_as_range(compiler: FilterCompiler) -> None:
    """``time_range`` compiles like ``range``."""
    text, _ = render(
        compiler, {"op": "time_range", "field": "created_at", "gte": 0, "lt": 10}
    )
    assert ">=" in text and "<" in text


def test_range_excludes_nulls(compiler: FilterCompiler) -> None:
    """`_in_range` returns False for a None value; SQL must agree."""
    text, _ = render(compiler, {"op": "range", "field": "level", "gt": 1})
    assert "IS NOT NULL" in text


def test_range_out_negates(compiler: FilterCompiler) -> None:
    """``range_out`` is the complement of ``range``."""
    text, _ = render(compiler, {"op": "range_out", "field": "level", "gt": 1})
    assert "NOT" in text.upper()


def test_contains_escapes_like_metacharacters(compiler: FilterCompiler) -> None:
    """``%`` and ``_`` in a substring are escaped, not treated as wildcards."""
    _, params = render(
        compiler, {"op": "contains", "field": "name", "substring": "50%_x"}
    )
    assert params == ["%50\\%\\_x%"]


def test_contains_on_non_text_is_false(compiler: FilterCompiler) -> None:
    """A non-textual column contains no substring."""
    text, params = render(
        compiler, {"op": "contains", "field": "level", "substring": "1"}
    )
    assert text == "FALSE"
    assert params == []


def test_and_or_nesting(compiler: FilterCompiler) -> None:
    """Nested boolean groups compile with their parameters in order."""
    node = {
        "op": "and",
        "conds": [
            {"op": "must", "field": "name", "conds": ["a"]},
            {
                "op": "or",
                "conds": [
                    {"op": "must", "field": "level", "conds": [1]},
                    {"op": "range", "field": "level", "gte": 5},
                ],
            },
        ],
    }
    text, params = render(compiler, node)
    assert " AND " in text and " OR " in text
    assert params == [["a"], [1], 5]


def test_empty_and_is_true_empty_or_is_false(compiler: FilterCompiler) -> None:
    """An empty ``and`` matches everything; an empty ``or`` matches nothing."""
    assert render(compiler, {"op": "and", "conds": []})[0] == "TRUE"
    assert render(compiler, {"op": "or", "conds": []})[0] == "FALSE"


def test_unknown_field_matches_nothing(compiler: FilterCompiler) -> None:
    """A ``must`` on an undeclared field matches no row."""
    assert render(compiler, {"op": "must", "field": "nope", "conds": ["x"]})[0] == "FALSE"


def test_unknown_field_must_not_matches_everything(compiler: FilterCompiler) -> None:
    """A ``must_not`` on an undeclared field excludes no row."""
    assert (
        render(compiler, {"op": "must_not", "field": "nope", "conds": ["x"]})[0] == "TRUE"
    )


def test_filter_wrapper_is_unwrapped(compiler: FilterCompiler) -> None:
    """A node wrapped in ``{'filter': ...}`` compiles like the bare node."""
    wrapped = {"filter": {"op": "must", "field": "name", "conds": ["a"]}}
    assert render(compiler, wrapped) == render(
        compiler, {"op": "must", "field": "name", "conds": ["a"]}
    )


def test_unsupported_op_raises(compiler: FilterCompiler) -> None:
    """An unknown operator is rejected rather than ignored."""
    with pytest.raises(UnsupportedFilterError):
        render(compiler, {"op": "wat", "field": "name", "conds": ["a"]})


def test_geo_range_raises_clearly(compiler: FilterCompiler) -> None:
    """``geo_range`` reports that it is unimplemented."""
    with pytest.raises(UnsupportedFilterError, match="geo_range"):
        render(compiler, {"op": "geo_range", "field": "name", "radius": "1km"})


def test_no_value_is_interpolated_into_sql(compiler: FilterCompiler) -> None:
    """Every operand must travel as a bound parameter, never inline."""
    text, params = render(
        compiler,
        {"op": "must", "field": "name", "conds": ["'; DROP TABLE ov_context; --"]},
    )
    assert "DROP TABLE" not in text
    assert params == [["'; DROP TABLE ov_context; --"]]
