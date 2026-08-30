"""Compile OpenViking's filter DSL into parameterised SQL.

The DSL is the one emitted by ``CollectionAdapter._compile_filter``
(openviking/storage/vectordb_adapters/base.py) and evaluated in pure Python by
``matches_filter`` in openviking/storage/vectordb/index/cuvs_index.py.  That
Python evaluator is the reference semantics this module reproduces in SQL --
where the two disagree, the Python one is right and this is a bug.

Nodes have the shape::

    {"op": "and" | "or", "conds": [ ...nodes... ]}
    {"op": "must" | "must_not", "field": str, "conds": [...], "para": "-d=N"?}
    {"op": "contains", "field": str, "substring": str}
    {"op": "prefix",   "field": str, "conds": [...]}
    {"op": "regex",    "field": str, "conds": [...]}
    {"op": "range" | "range_out" | "time_range",
     "field": str, "gt"/"gte"/"lt"/"lte": value}

A node may also be wrapped as ``{"filter": <node>}``.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from psycopg import sql

from .schema import CollectionSchema, FieldSpec

_DEPTH_RE = re.compile(r"\s*-d=(-?\d+)\s*")

# Ops that carry their operand in a "conds" list.
_COND_OPS = {"must", "must_not", "prefix", "regex"}
_RANGE_OPS = {"range", "range_out", "time_range"}


class UnsupportedFilterError(ValueError):
    """Raised for a filter node this backend cannot express in SQL."""


def parse_depth(para: object) -> int | None:
    """Parse the ``-d=N`` path-scope parameter. Mirrors cuvs_index._parse_depth."""
    if para in (None, ""):
        return None
    if not isinstance(para, str):
        raise UnsupportedFilterError(f"Unsupported path filter parameter: {para!r}")
    match = _DEPTH_RE.fullmatch(para)
    if not match:
        raise UnsupportedFilterError(f"Unsupported path filter parameter: {para!r}")
    return int(match.group(1))


def parse_datetime_to_epoch_ms(value: object, tz_policy: str = "local") -> int:
    """Mirror of ``DataProcessor.parse_datetime_to_epoch_ms``.

    Naive timestamps take the configured policy; OpenViking's own default is
    ``local``, so the same string lands on the same integer on both backends.
    """
    if isinstance(value, bool):
        raise ValueError("date_time value must be string or number, got bool")
    if isinstance(value, (int, float)):
        return int(value)
    if not isinstance(value, str):
        raise ValueError(
            f"date_time value must be string or number, got {type(value).__name__}"
        )
    raw = value.strip()
    if not raw:
        raise ValueError("date_time value is empty")
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"invalid date_time format: {value}") from exc
    if dt.tzinfo is None:
        if tz_policy == "local":
            dt = dt.replace(tzinfo=datetime.now().astimezone().tzinfo)
        elif tz_policy == "utc":
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            raise ValueError(f"unknown tz_policy: {tz_policy}")
    return int(dt.timestamp() * 1000)


class FilterCompiler:
    """Compiles one filter tree into a SQL predicate and its bound parameters.

    Parameters
    ----------
    schema :
        Parsed collection schema, used to resolve field names to column types.
    tz_policy :
        Timezone applied to naive timestamps. Must match OpenViking's own
        setting or the two backends will compare different epoch values.
    db_schema :
        PostgreSQL schema holding the ``ov_path_matches`` helper, which is
        called schema-qualified.
    """

    def __init__(
        self,
        schema: CollectionSchema,
        tz_policy: str = "local",
        db_schema: str = "public",
    ) -> None:
        self._schema = schema
        self._tz_policy = tz_policy
        self._db_schema = db_schema

    def compile(self, node: Mapping[str, Any] | None) -> tuple[sql.Composable, list[Any]]:
        """Return ``(predicate, params)``; predicate is ``TRUE`` when empty."""
        if not node:
            return sql.SQL("TRUE"), []
        if not isinstance(node, Mapping):
            raise UnsupportedFilterError(f"Filter node must be an object: {node!r}")

        # `{"filter": {...}}` wrapper, and the `{"filter":..., "sorter":...}`
        # form that search_by_scalar builds.  The sorter is handled by the
        # caller, so only the filter half is compiled here.
        if "filter" in node and "op" not in node:
            nested = node.get("filter")
            if nested is None:
                return sql.SQL("TRUE"), []
            if not isinstance(nested, Mapping):
                raise UnsupportedFilterError("The filter wrapper must contain an object")
            return self.compile(nested)

        params: list[Any] = []
        predicate = self._node(node, params)
        return predicate, params

    def _node(self, node: Mapping[str, Any], params: list[Any]) -> sql.Composable:
        """Compile one filter node, dispatching on its operator."""
        if not isinstance(node, Mapping):
            raise UnsupportedFilterError(f"Filter node must be an object: {node!r}")

        if "filter" in node and "op" not in node:
            nested = node.get("filter")
            if nested is None:
                return sql.SQL("TRUE")
            if not isinstance(nested, Mapping):
                raise UnsupportedFilterError("The filter wrapper must contain an object")
            return self._node(nested, params)

        op = str(node.get("op", "")).lower()

        if op in ("and", "or"):
            children = node.get("conds", [])
            if not isinstance(children, list):
                raise UnsupportedFilterError(f"{op} filter conds must be a list")
            if not children:
                # An empty AND is vacuously true; an empty OR matches nothing.
                return sql.SQL("TRUE") if op == "and" else sql.SQL("FALSE")
            parts = [self._node(child, params) for child in children]
            joiner = sql.SQL(" AND ") if op == "and" else sql.SQL(" OR ")
            return sql.SQL("({})").format(joiner.join(parts))

        field_name = node.get("field")
        if not isinstance(field_name, str):
            raise UnsupportedFilterError(f"Filter field must be a string: {node!r}")

        spec = self._schema.by_name(field_name)
        if spec is None:
            # An undeclared field has no value, so the reference evaluates
            # against None: `must` and `contains` fail, while the negated
            # forms (`must_not`, `range_out`) succeed.
            if op in ("must_not", "range_out"):
                return sql.SQL("TRUE")
            if op == "must" and None in (node.get("conds") or []):
                # `None in conds` matches a missing value.
                return sql.SQL("TRUE")
            return sql.SQL("FALSE")

        if op in _COND_OPS:
            return self._cond_op(op, spec, node, params)
        if op == "contains":
            return self._contains(spec, node, params)
        if op in _RANGE_OPS:
            return self._range(op, spec, node, params)
        if op == "geo_range":
            raise UnsupportedFilterError(
                "geo_range filters are not implemented by the pgvector backend"
            )
        raise UnsupportedFilterError(f"Unsupported filter operation: {op!r}")

    def _cond_op(
        self,
        op: str,
        spec: FieldSpec,
        node: Mapping[str, Any],
        params: list[Any],
    ) -> sql.Composable:
        """Compile a `conds`-carrying operator, negating for ``must_not``."""
        conds = node.get("conds", [])
        if not isinstance(conds, list):
            raise UnsupportedFilterError(f"{op} filter conds must be a list")

        col = _col(spec.name)

        if spec.is_path and op in ("must", "must_not"):
            depth = parse_depth(node.get("para"))
            matched = self._path_match(col, conds, depth, params)
        else:
            if node.get("para") not in (None, ""):
                raise UnsupportedFilterError(
                    f"Filter parameters are only supported for path fields: {node!r}"
                )
            if op == "prefix":
                matched = self._prefix(spec, col, conds, params)
            elif op == "regex":
                matched = self._regex(spec, col, conds, params)
            else:
                matched = self._value_match(spec, col, conds, params)

        if op == "must_not":
            # NULL columns must count as "not matching", so a plain NOT is
            # wrong -- NOT NULL is NULL, which filters the row out.
            return sql.SQL("(NOT COALESCE({}, FALSE))").format(matched)
        return matched

    def _value_match(
        self,
        spec: FieldSpec,
        col: sql.Composable,
        conds: Sequence[Any],
        params: list[Any],
    ) -> sql.Composable:
        """Compile a set-membership test over ``conds``.

        Array columns use ``&&`` (any element in common) and scalars use
        ``= ANY``, mirroring ``_value_matches`` in the reference evaluator.

        Parameters
        ----------
        spec :
            The column being tested.
        col :
            The rendered column reference.
        conds :
            Candidate values.
        params :
            Bound-parameter accumulator, appended to in place.

        Returns
        -------
        sql.Composable
            A boolean predicate.
        """
        if not conds:
            return sql.SQL("FALSE")

        # Operands whose Python type cannot match this column are dropped:
        # the reference evaluator compares with `in`/`==`, which is False for a
        # mismatched type rather than an error. Binding them would instead make
        # PostgreSQL raise UndefinedFunction and fail the whole query.
        values = [
            self._coerce(spec, value) for value in conds if _is_comparable(spec, value)
        ]
        matches_null = any(value is None for value in conds)

        if not values:
            if matches_null:
                return sql.SQL("({} IS NULL)").format(col)
            return sql.SQL("FALSE")

        params.append(values)
        if spec.is_array:
            predicate = sql.SQL("({} && {})").format(col, self._operand(spec))
        else:
            predicate = sql.SQL("({} = ANY({}))").format(col, self._operand(spec))

        if matches_null:
            # `None in conds` matches a row whose value is absent.
            return sql.SQL("({} OR {} IS NULL)").format(predicate, col)
        return predicate

    def _operand(self, spec: FieldSpec) -> sql.Composable:
        """Render a placeholder cast to the column's type where needed.

        ``float32`` columns are ``real``; an unqualified numeric literal binds
        as ``double precision``, and ``0.1::real = 0.1::float8`` is false. The
        cast keeps equality and range bounds behaving like the reference.

        Parameters
        ----------
        spec :
            The column the operand is compared against.

        Returns
        -------
        sql.Composable
            A placeholder, cast when the column type requires it.
        """
        if spec.ov_type == "float32":
            return sql.SQL("{}::real[]").format(sql.Placeholder())
        if spec.ov_type == "list<int64>":
            return sql.SQL("{}::bigint[]").format(sql.Placeholder())
        return sql.Placeholder()

    def _scalar_operand(self, spec: FieldSpec) -> sql.Composable:
        """Render a single-value placeholder, cast to the column's type.

        The list-valued counterpart of :meth:`_operand`, used for range bounds.

        Parameters
        ----------
        spec :
            The column the operand is compared against.

        Returns
        -------
        sql.Composable
            A placeholder, cast when the column type requires it.
        """
        if spec.ov_type == "float32":
            return sql.SQL("{}::real").format(sql.Placeholder())
        return sql.Placeholder()

    def _path_match(
        self,
        col: sql.Composable,
        conds: Sequence[Any],
        depth: int | None,
        params: list[Any],
    ) -> sql.Composable:
        """Compile a path-scope test against the ``ov_path_matches`` helper.

        Parameters
        ----------
        col :
            The rendered column reference.
        conds :
            Candidate ancestor paths.
        depth :
            Depth budget; ``None`` means unlimited, encoded as ``-1``.
        params :
            Bound-parameter accumulator, appended to in place.

        Returns
        -------
        sql.Composable
            A boolean predicate.
        """
        # `_path_matches` returns False when either side is not a string, so a
        # non-string operand simply cannot match; passing it would make the
        # helper call fail to resolve on type.
        conds = [c for c in conds if isinstance(c, str)]
        if not conds:
            return sql.SQL("FALSE")
        parts: list[sql.Composable] = []
        for expected in conds:
            params.append(expected)
            params.append(depth if depth is not None else -1)
            parts.append(
                sql.SQL("{}.ov_path_matches({}, {}, {})").format(
                    sql.Identifier(self._db_schema),
                    col,
                    sql.Placeholder(),
                    sql.Placeholder(),
                )
            )
        return sql.SQL("({})").format(sql.SQL(" OR ").join(parts))

    def _prefix(
        self,
        spec: FieldSpec,
        col: sql.Composable,
        conds: Sequence[Any],
        params: list[Any],
    ) -> sql.Composable:
        """Compile a prefix test over ``conds``.

        Parameters
        ----------
        spec :
            The column being tested.
        col :
            The rendered column reference.
        conds :
            Candidate prefixes.
        params :
            Bound-parameter accumulator, appended to in place.

        Returns
        -------
        sql.Composable
            A boolean predicate; ``FALSE`` for a non-textual column.
        """
        conds = [c for c in conds if isinstance(c, str)]
        if not conds or not spec.is_textual:
            return sql.SQL("FALSE")
        parts: list[sql.Composable] = []
        for value in conds:
            params.append(_like_prefix(value))
            if spec.is_array:
                parts.append(
                    sql.SQL(
                        "EXISTS (SELECT 1 FROM unnest({}) AS _e(v) "
                        "WHERE _e.v LIKE {} ESCAPE '\\')"
                    ).format(col, sql.Placeholder())
                )
            else:
                parts.append(
                    sql.SQL("({} LIKE {} ESCAPE '\\')").format(col, sql.Placeholder())
                )
        return sql.SQL("({})").format(sql.SQL(" OR ").join(parts))

    def _regex(
        self,
        spec: FieldSpec,
        col: sql.Composable,
        conds: Sequence[Any],
        params: list[Any],
    ) -> sql.Composable:
        """Compile a POSIX-regex test over ``conds``.

        Parameters
        ----------
        spec :
            The column being tested.
        col :
            The rendered column reference.
        conds :
            Candidate patterns.
        params :
            Bound-parameter accumulator, appended to in place.

        Returns
        -------
        sql.Composable
            A boolean predicate; ``FALSE`` for a non-textual column.
        """
        conds = [c for c in conds if isinstance(c, str)]
        if not conds or not spec.is_textual:
            return sql.SQL("FALSE")
        parts: list[sql.Composable] = []
        for value in conds:
            params.append(value)
            if spec.is_array:
                parts.append(
                    sql.SQL(
                        "EXISTS (SELECT 1 FROM unnest({}) AS _e(v) WHERE _e.v ~ {})"
                    ).format(col, sql.Placeholder())
                )
            else:
                parts.append(sql.SQL("({} ~ {})").format(col, sql.Placeholder()))
        return sql.SQL("({})").format(sql.SQL(" OR ").join(parts))

    def _contains(
        self,
        spec: FieldSpec,
        node: Mapping[str, Any],
        params: list[Any],
    ) -> sql.Composable:
        """Compile a substring test, matching only textual columns."""
        substring = node.get("substring")
        # `_contains` returns False for a non-string needle, and for a value
        # that is neither text nor a list of text. Checking before binding
        # keeps a mismatched column from reaching PostgreSQL as `bigint ~~ text`.
        if not isinstance(substring, str) or not spec.is_textual:
            return sql.SQL("FALSE")
        col = _col(spec.name)
        params.append(_like_contains(substring))
        if spec.is_array:
            return sql.SQL(
                "EXISTS (SELECT 1 FROM unnest({}) AS _e(v) "
                "WHERE _e.v LIKE {} ESCAPE '\\')"
            ).format(col, sql.Placeholder())
        return sql.SQL("({} LIKE {} ESCAPE '\\')").format(col, sql.Placeholder())

    def _range(
        self,
        op: str,
        spec: FieldSpec,
        node: Mapping[str, Any],
        params: list[Any],
    ) -> sql.Composable:
        """Compile a bounded comparison, negating for ``range_out``.

        A range with no bounds at all is not a no-op: the reference rejects
        only a missing *value*, so a present value passes every absent check
        and matches. That case falls through to the ``IS NOT NULL`` test below,
        for arrays as much as for scalars.

        A *bounded* range on an array column short-circuits to no match, since
        an array has no ordering against a scalar bound.
        """
        col = _col(spec.name)
        bounded = any(node.get(key) is not None for key in ("gt", "gte", "lt", "lte"))

        # An array column has no ordering against a scalar bound. The reference
        # evaluates `list >= 2`, catches the TypeError and returns False, so a
        # *bounded* range on an array must match nothing rather than reach
        # PostgreSQL as `bigint[] >= smallint`. Checking the element type is not
        # enough: list<int64> against an int bound has a compatible element type
        # and still cannot be ordered.
        if spec.is_array and bounded:
            return sql.SQL("TRUE") if op == "range_out" else sql.SQL("FALSE")

        comparisons = (
            ("gt", sql.SQL(">")),
            ("gte", sql.SQL(">=")),
            ("lt", sql.SQL("<")),
            ("lte", sql.SQL("<=")),
        )

        parts: list[sql.Composable] = []
        for key, operator in comparisons:
            value = node.get(key)
            if value is None:
                continue
            if not _is_comparable(spec, value):
                # `_in_range` catches TypeError and returns False, so a bound
                # of the wrong type excludes every row rather than raising.
                return sql.SQL("TRUE") if op == "range_out" else sql.SQL("FALSE")
            params.append(self._coerce(spec, value))
            parts.append(
                sql.SQL("{} {} {}").format(col, operator, self._scalar_operand(spec))
            )

        if not parts:
            # No bounds: `_in_range` still rejects NULL values.
            inner = sql.SQL("({} IS NOT NULL)").format(col)
        else:
            inner = sql.SQL("({} IS NOT NULL AND {})").format(
                col, sql.SQL(" AND ").join(parts)
            )

        if op == "range_out":
            return sql.SQL("(NOT COALESCE({}, FALSE))").format(inner)
        return inner

    def _coerce(self, spec: FieldSpec, value: object) -> object:
        """Apply the same type conversion the native engine applies on write.

        Also converts a boolean/integer operand to the column's own type, so
        the ``True == 1`` equivalence Python gives the reference survives into
        SQL, where the two types have no implicit comparison.
        """
        if spec.is_datetime and value is not None:
            return parse_datetime_to_epoch_ms(value, self._tz_policy)
        return _coerce_operand(spec, value)


# Python types that can meaningfully compare against each column type. The
# reference evaluator compares with `in`, `==` and `<`, which are False (or
# TypeError-caught-as-False) across types; PostgreSQL instead raises
# UndefinedFunction, so operands are filtered before they reach a query.
_COMPARABLE_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "text": (str,),
    "path": (str,),
    "int64": (int,),
    "float32": (int, float),
    "bool": (bool,),
    "list<string>": (str,),
    "list<int64>": (int,),
    "date_time": (str, int, float),
}


def _is_comparable(spec: FieldSpec, value: object) -> bool:
    """Return whether ``value`` can be compared against ``spec``'s column.

    Python compares ``True == 1``, so the reference matches a boolean against
    an integer operand and vice versa. PostgreSQL has no implicit
    boolean/bigint comparison, so those operands are converted by
    :func:`_coerce_operand` rather than dropped. Only ``0`` and ``1`` convert:
    the reference finds no match for ``True == 2`` either, so a wider integer
    against a boolean column is correctly not comparable.

    Parameters
    ----------
    spec :
        The column the value would be compared against.
    value :
        The candidate operand.

    Returns
    -------
    bool
        True when the comparison is expressible in SQL.
    """
    allowed = _COMPARABLE_TYPES.get(spec.ov_type)
    if allowed is None:
        return False
    if spec.ov_type == "bool":
        # bool(2) would be True, but the reference says `True == 2` is False.
        return isinstance(value, bool) or (isinstance(value, int) and value in (0, 1))
    if isinstance(value, bool):
        # A boolean against a numeric column compares as 1/0, as Python does.
        return any(t in allowed for t in (int, float))
    return isinstance(value, allowed)


def _coerce_operand(spec: FieldSpec, value: object) -> object:
    """Convert an operand to the column's own type where Python would compare.

    Parameters
    ----------
    spec :
        The column the operand is compared against.
    value :
        An operand already accepted by :func:`_is_comparable`.

    Returns
    -------
    object
        The operand as the column's type.
    """
    if spec.ov_type == "bool" and not isinstance(value, bool):
        return bool(value)
    if isinstance(value, bool) and spec.ov_type in ("int64", "float32", "list<int64>"):
        return int(value)
    return value


def _col(name: str) -> sql.Composable:
    """Render a column reference as a quoted identifier."""
    return sql.Identifier(name)


def _escape_like(value: str) -> str:
    """Escape LIKE metacharacters so ``value`` matches literally."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _like_prefix(value: str) -> str:
    """Build a LIKE pattern matching strings that start with ``value``."""
    return _escape_like(value) + "%"


def _like_contains(value: str) -> str:
    """Build a LIKE pattern matching strings that contain ``value``."""
    return "%" + _escape_like(value) + "%"
