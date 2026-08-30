"""DDL: extension bootstrap, helper functions, registry and collection tables."""

from __future__ import annotations

import hashlib
from typing import Any

from psycopg import sql

from .schema import CollectionSchema, FieldSpec

# What psycopg's execute() accepts.
Statement = sql.SQL | sql.Composed

# Registry tables track collection and index metadata so that
# ``get_meta_data`` / ``list_indexes`` return exactly what was declared.
REGISTRY_COLLECTIONS = "ov_collections"
REGISTRY_INDEXES = "ov_indexes"

# PostgreSQL truncates identifiers at this many bytes, silently.
_MAX_IDENTIFIER_BYTES = 63


# Minimum pgvector versions for features that are not always available.
# https://github.com/pgvector/pgvector -- HNSW arrived in 0.5.0, and the
# iterative-scan GUCs that rescue a filtered ANN search arrived in 0.8.0.
MIN_VERSION_HNSW: tuple[int, ...] = (0, 5, 0)
MIN_VERSION_ITERATIVE_SCAN: tuple[int, ...] = (0, 8, 0)


def parse_extension_version(text: str) -> tuple[int, ...]:
    """Parse a pgvector ``extversion`` string into comparable integers.

    Avoids a dependency on ``packaging`` for what is always a simple dotted
    numeric version. Non-numeric parts degrade to zero rather than raising, so
    an unexpected suffix disables gated features instead of breaking startup.

    Parameters
    ----------
    text :
        The value of ``pg_extension.extversion``, such as ``"0.8.2"``.

    Returns
    -------
    tuple[int, ...]
        Version components, comparable with the ``MIN_VERSION_*`` constants.
    """
    parts: list[int] = []
    for chunk in text.split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def advisory_lock_key(schema_name: str) -> int:
    """Derive a stable 63-bit advisory-lock key from a schema name.

    ``hash()`` is salted per process and would give each server a different
    key, so a deterministic digest is used instead.

    Parameters
    ----------
    schema_name :
        The schema being bootstrapped.

    Returns
    -------
    int
        A key that fits PostgreSQL's ``bigint`` advisory-lock argument.
    """
    digest = hashlib.sha256(f"ov_postgres.bootstrap.{schema_name}".encode()).digest()
    return int.from_bytes(digest[:8], "big") >> 1


def bootstrap_statements(
    schema_name: str, *, create_extension: bool = True
) -> list[Statement]:
    """Build the statements that make a database usable.

    The statements run in one transaction, and the first takes a PostgreSQL
    advisory lock keyed on the schema name. ``IF NOT EXISTS`` is check-then-act
    rather than atomic, so without that lock two servers starting together race
    and lose: PostgreSQL reports ``UniqueViolation`` on ``pg_namespace`` or
    ``pg_proc``, or ``tuple concurrently updated`` from ``CREATE EXTENSION``.
    A per-process ``threading.Lock`` cannot help, since the racing parties are
    usually different processes.

    Parameters
    ----------
    schema_name :
        PostgreSQL schema to create the helpers and registry tables in.
    create_extension :
        Whether to include ``CREATE EXTENSION``. False on managed PostgreSQL,
        where the role cannot create extensions and one already exists.

    Returns
    -------
    list[Statement]
        Statements to execute in order, inside a single transaction.
    """
    ns = sql.Identifier(schema_name)
    extension: list[Statement] = (
        [sql.SQL("CREATE EXTENSION IF NOT EXISTS vector")] if create_extension else []
    )
    return [
        # Held until the transaction ends, serialising concurrent bootstraps.
        sql.SQL("SELECT pg_advisory_xact_lock({})").format(
            sql.Literal(advisory_lock_key(schema_name))
        ),
        sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(ns),
        *extension,
        _path_matches_fn(schema_name),
        _sparse_dot_fn(schema_name),
        _array_to_text_fn(schema_name),
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {}.{} (
                name        text PRIMARY KEY,
                table_name  text NOT NULL,
                meta        jsonb NOT NULL,
                created_at  timestamptz NOT NULL DEFAULT now()
            )
            """
        ).format(ns, sql.Identifier(REGISTRY_COLLECTIONS)),
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {}.{} (
                collection  text NOT NULL,
                index_name  text NOT NULL,
                meta        jsonb NOT NULL,
                created_at  timestamptz NOT NULL DEFAULT now(),
                PRIMARY KEY (collection, index_name)
            )
            """
        ).format(ns, sql.Identifier(REGISTRY_INDEXES)),
    ]


def index_name(table: str, *parts: str) -> str:
    """Build an index name that stays distinct after PostgreSQL truncates it.

    PostgreSQL silently truncates an identifier at 63 bytes, so
    ``ov_ctx__<long field>_idx`` and ``ov_ctx__<long field>_c_idx`` collapse to
    the same name. ``CREATE INDEX IF NOT EXISTS`` then matches by name and
    skips the second index without complaint -- leaving, for instance, no
    collated index and every text range scanning the table.

    Names that already fit are returned unchanged, so existing databases keep
    the index names they have.

    Parameters
    ----------
    table :
        Table the index is on.
    parts :
        Name components, joined with underscores after the table.

    Returns
    -------
    str
        An index name of at most 63 bytes, distinct for distinct inputs.
    """
    candidate = f"{table}__{'_'.join(parts)}"
    if len(candidate.encode("utf-8")) <= _MAX_IDENTIFIER_BYTES:
        return candidate
    digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()[:8]
    suffix = f"_{digest}"
    keep = _MAX_IDENTIFIER_BYTES - len(suffix)
    trimmed = candidate.encode("utf-8")[:keep].decode("utf-8", "ignore")
    return f"{trimmed}{suffix}"


def _path_matches_fn(schema_name: str) -> Statement:
    """Build the path-scope helper, a port of ``_path_matches``.

    Kept equivalent to the Python reference in
    ``openviking/storage/vectordb/index/cuvs_index.py``: normalise both sides
    to a leading slash and no trailing slash, compute the relative depth of
    *value* beneath *expected*, and compare against the depth budget (negative
    or NULL means unlimited).

    Parameters
    ----------
    schema_name :
        Schema to create the function in.

    Returns
    -------
    Statement
        A ``CREATE OR REPLACE FUNCTION`` statement.
    """
    return sql.SQL(
        """
        CREATE OR REPLACE FUNCTION {ns}.ov_path_matches(
            value text, expected text, depth integer
        ) RETURNS boolean
        LANGUAGE plpgsql IMMUTABLE PARALLEL SAFE AS $fn$
        DECLARE
            v text; e text; suffix text; rel integer;
        BEGIN
            IF value IS NULL OR expected IS NULL THEN
                RETURN false;
            END IF;

            -- btrim(text) alone strips spaces only; Python's str.strip()
            -- strips every ASCII whitespace character, so name them.
            v := btrim(value, E' \t\n\r\f\v');
            IF left(v, 1) <> '/' THEN v := '/' || v; END IF;
            v := rtrim(v, '/');
            IF v = '' THEN v := '/'; END IF;

            e := btrim(expected, E' \t\n\r\f\v');
            IF left(e, 1) <> '/' THEN e := '/' || e; END IF;
            e := rtrim(e, '/');
            IF e = '' THEN e := '/'; END IF;

            IF v = e THEN
                rel := 0;
            ELSIF e = '/' THEN
                rel := coalesce(
                    array_length(array_remove(string_to_array(v, '/'), ''), 1), 0);
            -- Plain equality on the prefix, never LIKE: the expected path is
            -- caller data, and `_` and `%` in it would otherwise act as
            -- wildcards and match sibling subtrees. Mirrors the reference's
            -- value_path.startswith(expected_path + "/").
            ELSIF substr(v, 1, length(e) + 1) = e || '/' THEN
                suffix := substr(v, length(e) + 2);
                rel := coalesce(
                    array_length(array_remove(string_to_array(suffix, '/'), ''), 1), 0);
            ELSE
                RETURN false;
            END IF;

            IF depth IS NULL OR depth < 0 THEN
                RETURN true;
            END IF;
            RETURN rel <= depth;
        END;
        $fn$
        """
    ).format(ns=sql.Identifier(schema_name))


def _sparse_dot_fn(schema_name: str) -> Statement:
    """Build the dot-product helper for two JSONB sparse vectors.

    Parameters
    ----------
    schema_name :
        Schema to create the function in.

    Returns
    -------
    Statement
        A ``CREATE OR REPLACE FUNCTION`` statement.
    """
    return sql.SQL(
        """
        CREATE OR REPLACE FUNCTION {ns}.ov_sparse_dot(vec_a jsonb, vec_b jsonb)
        RETURNS double precision
        LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $fn$
            SELECT coalesce(
                sum(ea.value::double precision * eb.value::double precision), 0.0)
            FROM jsonb_each_text(coalesce(vec_a, '{{}}'::jsonb)) AS ea
            JOIN jsonb_each_text(coalesce(vec_b, '{{}}'::jsonb)) AS eb
              ON ea.key = eb.key
        $fn$
        """
    ).format(ns=sql.Identifier(schema_name))


def _array_to_text_fn(schema_name: str) -> Statement:
    """Build an IMMUTABLE text[] flattener for use in index expressions.

    ``array_to_string`` is only STABLE -- in general it depends on the output
    function of the element type -- so PostgreSQL refuses it inside an index
    expression. Fixed to ``text[]`` and a constant separator the operation is
    genuinely immutable, so wrapping it is sound rather than a fib to the
    planner.

    Parameters
    ----------
    schema_name :
        Schema to create the function in.

    Returns
    -------
    Statement
        A ``CREATE OR REPLACE FUNCTION`` statement.
    """
    return sql.SQL(
        """
        CREATE OR REPLACE FUNCTION {ns}.ov_array_to_text(arr text[])
        RETURNS text
        LANGUAGE sql IMMUTABLE PARALLEL SAFE AS $fn$
            SELECT coalesce(array_to_string(arr, ' '), '')
        $fn$
        """
    ).format(ns=sql.Identifier(schema_name))


# Column types as SQL literals, keyed by OpenViking field type.  `vector` is
# absent because its type carries a dimension and is built separately.
_COLUMN_TYPES: dict[str, sql.SQL] = {
    "string": sql.SQL("text"),
    "text": sql.SQL("text"),
    "path": sql.SQL("text"),
    "int64": sql.SQL("bigint"),
    "float32": sql.SQL("real"),
    "bool": sql.SQL("boolean"),
    "list<string>": sql.SQL("text[]"),
    "list<int64>": sql.SQL("bigint[]"),
    "date_time": sql.SQL("bigint"),
    "sparse_vector": sql.SQL("jsonb"),
}


def _column_type(spec: FieldSpec) -> sql.Composable:
    """Return the PostgreSQL column type for a field.

    Parameters
    ----------
    spec :
        The parsed field.

    Returns
    -------
    sql.Composable
        The column type, ready to compose into DDL.

    Raises
    ------
    ValueError
        If a vector field has no dimension, or the type has no column form.
    """
    if spec.is_vector:
        if not spec.dim:
            raise ValueError(f"Vector field {spec.name!r} has no dimension")
        # A bound integer literal, so no string reaches SQL composition.
        return sql.SQL("vector({})").format(sql.Literal(int(spec.dim)))
    column_type = _COLUMN_TYPES.get(spec.ov_type)
    if column_type is None:
        raise ValueError(
            f"No PostgreSQL column type for field {spec.name!r} of type {spec.ov_type!r}"
        )
    return column_type


def create_table(schema_name: str, table: str, coll: CollectionSchema) -> Statement:
    """Build the CREATE TABLE statement for a collection.

    Parameters
    ----------
    schema_name :
        Schema to create the table in.
    table :
        Table name.
    coll :
        Parsed collection schema.

    Returns
    -------
    Statement
        An idempotent ``CREATE TABLE IF NOT EXISTS``.
    """
    columns: list[sql.Composable] = []
    for spec in coll.fields:
        if spec.is_geo:
            for suffix in ("_lon", "_lat"):
                columns.append(
                    sql.SQL("{} real").format(sql.Identifier(spec.name + suffix))
                )
            continue
        column = sql.SQL("{} {}").format(sql.Identifier(spec.name), _column_type(spec))
        if spec.is_primary:
            column = sql.SQL("{} PRIMARY KEY").format(column)
        columns.append(column)

    # Anything written but not declared in the schema lands here instead of
    # being dropped on the floor.
    columns.append(
        sql.SQL("{} jsonb NOT NULL DEFAULT '{{}}'::jsonb").format(sql.Identifier("extra"))
    )

    return sql.SQL("CREATE TABLE IF NOT EXISTS {}.{} ({})").format(
        sql.Identifier(schema_name),
        sql.Identifier(table),
        sql.SQL(", ").join(columns),
    )


def scalar_index_statements(
    schema_name: str, table: str, coll: CollectionSchema
) -> list[Statement]:
    """Build an index for every field named in ``ScalarIndex``.

    Array fields get a GIN index; everything else gets a B-tree. Path fields
    additionally get a ``text_pattern_ops`` index so prefix matches can use it.

    Parameters
    ----------
    schema_name :
        Schema holding the table.
    table :
        Table name.
    coll :
        Parsed collection schema.

    Returns
    -------
    list[Statement]
        Idempotent ``CREATE INDEX`` statements.
    """
    statements: list[Statement] = []
    for name in coll.scalar_index:
        spec = coll.by_name(name)
        if spec is None or spec.is_vector or spec.is_sparse or spec.is_geo:
            continue
        plain_name = index_name(table, name, "idx")
        method = sql.SQL("gin") if spec.is_array else sql.SQL("btree")
        statements.append(
            sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{} USING {} ({})").format(
                sql.Identifier(plain_name),
                sql.Identifier(schema_name),
                sql.Identifier(table),
                method,
                sql.Identifier(name),
            )
        )
        # A *second*, collated index for range comparisons, which run under
        # COLLATE "C" to match the reference's code-point ordering. PostgreSQL
        # matches an expression index syntactically, so a collated index cannot
        # serve a bare `col = ANY(...)`; replacing the plain one turned every
        # equality filter -- by far the commonest shape -- into a seq scan.
        if spec.is_textual and not spec.is_array:
            statements.append(
                sql.SQL('CREATE INDEX IF NOT EXISTS {} ON {}.{} ({} COLLATE "C")').format(
                    sql.Identifier(index_name(table, name, "c_idx")),
                    sql.Identifier(schema_name),
                    sql.Identifier(table),
                    sql.Identifier(name),
                )
            )
        # Path fields are also queried by prefix; a text_pattern_ops index lets
        # `LIKE 'prefix%'` use an index instead of scanning.
        if spec.is_path:
            statements.append(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {} ON {}.{} ({} text_pattern_ops)"
                ).format(
                    sql.Identifier(index_name(table, name, "prefix_idx")),
                    sql.Identifier(schema_name),
                    sql.Identifier(table),
                    sql.Identifier(name),
                )
            )
    return statements


# pgvector operator classes by distance metric, and the supported ANN methods.
# Both are `sql.SQL` literals so nothing derived from user configuration is
# ever spliced into SQL as raw text.
_VECTOR_OPS: dict[str, sql.SQL] = {
    "cosine": sql.SQL("vector_cosine_ops"),
    "l2": sql.SQL("vector_l2_ops"),
    "ip": sql.SQL("vector_ip_ops"),
}

_ANN_METHODS: dict[str, sql.SQL] = {
    "hnsw": sql.SQL("hnsw"),
    "ivfflat": sql.SQL("ivfflat"),
}


def vector_index_statement(
    schema_name: str,
    table: str,
    spec: FieldSpec,
    distance: str,
    method: str,
    options: dict[str, Any] | None = None,
) -> Statement | None:
    """Build the ANN index statement, or ``None`` for exact search.

    pgvector has no "flat" index type -- omitting the index *is* exact search,
    which is what OpenViking's ``flat``/``flat_hybrid`` index types ask for.

    Parameters
    ----------
    schema_name :
        Schema holding the table.
    table :
        Table name.
    spec :
        The vector field to index.
    distance :
        Distance metric, selecting the pgvector operator class.
    method :
        ``flat`` for no index, otherwise ``hnsw`` or ``ivfflat``.
    options :
        Extra settings for the index's ``WITH`` clause.

    Returns
    -------
    Statement | None
        The statement, or ``None`` when ``method`` asks for exact search.

    Raises
    ------
    ValueError
        If the distance metric or index method is unknown.
    """
    if method in ("flat", "none", None):
        return None

    opclass = _VECTOR_OPS.get(distance)
    if opclass is None:
        raise ValueError(
            f"Unsupported distance metric for pgvector: {distance!r}. "
            f"Expected one of: {', '.join(sorted(_VECTOR_OPS))}"
        )
    ann_method = _ANN_METHODS.get(method)
    if ann_method is None:
        raise ValueError(
            f"Unsupported pgvector index method: {method!r}. "
            f"Expected 'flat', or one of: {', '.join(sorted(_ANN_METHODS))}"
        )

    statement = sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{} USING {} ({} {})").format(
        sql.Identifier(index_name(table, spec.name, method, "idx")),
        sql.Identifier(schema_name),
        sql.Identifier(table),
        ann_method,
        sql.Identifier(spec.name),
        opclass,
    )
    if options:
        rendered = sql.SQL(", ").join(
            sql.SQL("{} = {}").format(sql.Identifier(key), sql.Literal(value))
            for key, value in options.items()
        )
        statement = sql.SQL("{} WITH ({})").format(statement, rendered)
    return statement


def fulltext_index_statement(
    schema_name: str, table: str, specs: list[FieldSpec], regconfig: str
) -> Statement | None:
    """Build the GIN index backing keyword search.

    Parameters
    ----------
    schema_name :
        Schema holding the table.
    table :
        Table name.
    specs :
        Text columns to include.
    regconfig :
        PostgreSQL text search configuration.

    Returns
    -------
    Statement | None
        The statement, or ``None`` when no text columns are available.
    """
    if not specs:
        return None
    return sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {}.{} USING gin ({})").format(
        sql.Identifier(index_name(table, "fts_idx")),
        sql.Identifier(schema_name),
        sql.Identifier(table),
        tsvector_expr(specs, regconfig, schema_name),
    )


def tsvector_expr(
    specs: list[FieldSpec], regconfig: str, schema_name: str
) -> sql.Composable:
    """Build the tsvector expression over ``specs``.

    The ``regconfig`` is cast explicitly so the whole expression is IMMUTABLE
    and therefore indexable -- the one-argument form of ``to_tsvector`` depends
    on a session GUC and cannot be used in an index.

    Array columns are flattened so ``list<string>`` fields such as
    ``search_tags`` are searchable alongside plain text.

    Parameters
    ----------
    specs :
        Text columns to concatenate.
    regconfig :
        PostgreSQL text search configuration.
    schema_name :
        Schema holding the array-flattening helper.

    Returns
    -------
    sql.Composable
        A ``to_tsvector(...)`` expression.
    """
    parts: list[sql.Composable] = []
    for spec in specs:
        parts.append(sql.SQL("coalesce({}, '')").format(_text_of(spec, schema_name)))
    joined = sql.SQL(" || ' ' || ").join(parts)
    return sql.SQL("to_tsvector({}::regconfig, {})").format(
        sql.Literal(regconfig), joined
    )


def _text_of(spec: FieldSpec, schema_name: str) -> sql.Composable:
    """Render a column as plain text, flattening arrays to a joined string.

    The helper is schema-qualified so this expression is textually identical
    whether built for the index or for a query -- a mismatch would leave the
    GIN index unused.

    Parameters
    ----------
    spec :
        Column to render.
    schema_name :
        Schema holding the array-flattening helper.

    Returns
    -------
    sql.Composable
        An expression of type ``text``.
    """
    col = sql.Identifier(spec.name)
    if spec.is_array:
        return sql.SQL("{}.ov_array_to_text({})").format(sql.Identifier(schema_name), col)
    return col
