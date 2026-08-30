"""``ICollection`` implementation backed by PostgreSQL + pgvector."""

from __future__ import annotations

import json
import logging
import math
import re
import threading
from collections.abc import Mapping, Sequence
from typing import Any, Literal, overload

from openviking.storage.vectordb.collection.collection import ICollection
from openviking.storage.vectordb.collection.result import (
    AggregateResult,
    DataItem,
    FetchDataInCollectionResult,
    SearchItemResult,
    SearchResult,
    UpdateResult,
    UpsertDataResult,
)
from psycopg import Cursor, sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from . import ddl
from .filters import FilterCompiler, parse_datetime_to_epoch_ms
from .schema import (
    GEO_LAT_SUFFIX,
    GEO_LON_SUFFIX,
    CollectionSchema,
    FieldSpec,
    default_for,
    fulltext_candidates,
)

# What psycopg's execute()/executemany() actually accept.
Statement = sql.SQL | sql.Composed

logger = logging.getLogger(__name__)

# Distance metric -> (pgvector operator, template turning a distance into a
# similarity score where higher is better).  OpenViking ranks descending by
# score, so every metric is normalised to "bigger is more similar".
#
# Held as `sql.SQL` literals rather than plain strings so no caller-supplied
# text can reach SQL composition -- `sql.SQL` accepts only literal strings by
# design, and that guarantee is worth preserving here.
_DISTANCE: dict[str, tuple[sql.SQL, sql.SQL]] = {
    # (operator, score template with {} for the distance expression)
    "cosine": (sql.SQL("<=>"), sql.SQL("1.0 - ({})")),
    "l2": (sql.SQL("<->"), sql.SQL("-({})")),
    "ip": (sql.SQL("<#>"), sql.SQL("-({})")),
}

DEFAULT_KEYWORD_FIELDS = ("name", "description", "abstract", "tags", "search_tags")

# Key for the NULL group in an aggregate. A column that is NULL and one that
# holds "" are different groups, and folding them together lost one of them.
NULL_BUCKET = "__ov_null__"

# Reported score for a row with no embedding. This is a *display* floor, not a
# ranking sentinel: ordering never uses it, so it cannot swamp a sparse term.
# OpenViking's HierarchicalRetriever re-sorts candidates by `_score` in Python
# and applies an absolute threshold, so a vectorless row reporting 0.0 would
# overtake a genuine but distant match and pass a threshold of zero.
# Below any score pgvector can produce: distances are float4, so the
# largest magnitude a negated `ip` term can reach is ~3.4e38.
NO_VECTOR_SCORE = -1e308

# GUC namespaces for iterative scan, as SQL literals so nothing derived from
# configuration is spliced into a statement as raw text.
_SCAN_GUC: dict[str, sql.SQL] = {
    "hnsw": sql.SQL("hnsw"),
    "ivfflat": sql.SQL("ivfflat"),
}


class PgVectorCollection(ICollection):  # type: ignore[misc]  # ICollection is untyped
    """A single OpenViking collection stored as one PostgreSQL table.

    Instances are safe to share across threads for the operations OpenViking
    performs: every statement runs on its own pooled connection, and the lock
    serialises writes to the cached schema and the active distance metric.

    One caveat: :meth:`update` mutates the cached schema in place over two
    assignments, so a reader calling :meth:`get_meta_data` concurrently with it
    can observe the new ``raw`` alongside the old ``description``. Collection
    metadata is written once at creation in normal use, so this is noted rather
    than locked against on every read.

    Parameters
    ----------
    pool :
        Connection pool to run statements on.
    db_schema :
        PostgreSQL schema holding the table, registry, and helper functions.
    collection_name :
        OpenViking's name for the collection.
    table_name :
        Name of the backing table.
    coll_schema :
        Parsed collection schema.
    distance :
        Distance metric: ``cosine``, ``l2``, or ``ip``.
    sparse_weight :
        Weight of the sparse term in hybrid scoring; zero disables it.
    index_method :
        ``flat`` for exact search, ``hnsw``/``ivfflat`` for approximate,
        ``auto`` to follow the requested index type.
    index_options :
        Extra settings for the ANN index's ``WITH`` clause.
    keyword_fields :
        Text columns included in the full-text index.
    text_search_config :
        PostgreSQL text search configuration for tsvectors.
    tz_policy :
        Timezone applied to naive timestamps.
    iterative_scan :
        Recovery mode for an ANN index that under-returns beneath a selective
        filter. ``off`` disables it.
    pgvector_version :
        Installed pgvector version, used to gate features. An empty tuple
        disables every gated feature.
    index_name :
        Name of the index bundle, used to resolve an ``auto`` index method
        from the registry.
    owns_pool :
        Whether :meth:`close` should also close the pool. False when the
        adapter owns it and shares it across collections.

    Raises
    ------
    ValueError
        If ``distance`` is not a supported metric.
    """

    def __init__(
        self,
        *,
        pool: ConnectionPool,
        db_schema: str,
        collection_name: str,
        table_name: str,
        coll_schema: CollectionSchema,
        distance: str = "cosine",
        sparse_weight: float = 0.0,
        index_method: str = "flat",
        index_options: dict[str, Any] | None = None,
        keyword_fields: Sequence[str] = DEFAULT_KEYWORD_FIELDS,
        text_search_config: str = "simple",
        tz_policy: str = "local",
        iterative_scan: str = "relaxed_order",
        pgvector_version: tuple[int, ...] = (),
        index_name: str = "default",
        owns_pool: bool = False,
    ) -> None:
        super().__init__()
        if distance not in _DISTANCE:
            raise ValueError(
                f"Unsupported distance metric for pgvector: {distance!r}. "
                f"Expected one of: {', '.join(sorted(_DISTANCE))}"
            )
        self._pool = pool
        self._owns_pool = owns_pool
        self._db_schema = db_schema
        self._name = collection_name
        self._table = table_name
        self._schema = coll_schema
        self._distance = distance
        self._sparse_weight = sparse_weight
        self._index_method = index_method
        self._index_options = dict(index_options or {})
        self._keyword_fields = list(keyword_fields)
        self._text_search_config = text_search_config
        self._tz_policy = tz_policy
        self._iterative_scan = iterative_scan
        self._pgvector_version = pgvector_version
        self._index_name = index_name
        self._resolved_method: str | None = None
        self._closed = False
        self._lock = threading.RLock()
        self._compiler = FilterCompiler(
            coll_schema, tz_policy=tz_policy, db_schema=db_schema
        )

    @property
    def _qualified(self) -> Statement:
        """The schema-qualified collection table."""
        return sql.SQL("{}.{}").format(
            sql.Identifier(self._db_schema), sql.Identifier(self._table)
        )

    def _registry(self, table: str) -> Statement:
        """Return a schema-qualified reference to a registry table."""
        return sql.SQL("{}.{}").format(
            sql.Identifier(self._db_schema), sql.Identifier(table)
        )

    def _check_open(self) -> None:
        """Raise if the collection has been closed."""
        if self._closed:
            raise RuntimeError(f"Collection {self._name!r} is closed")

    @overload
    def _execute(
        self,
        statement: Statement,
        params: Sequence[Any] | None = ...,
        *,
        fetch: Literal["none"] = ...,
        setup: Sequence[Statement] = ...,
    ) -> None: ...

    @overload
    def _execute(
        self,
        statement: Statement,
        params: Sequence[Any] | None = ...,
        *,
        fetch: Literal["one"],
        setup: Sequence[Statement] = ...,
    ) -> dict[str, Any] | None: ...

    @overload
    def _execute(
        self,
        statement: Statement,
        params: Sequence[Any] | None = ...,
        *,
        fetch: Literal["all"],
        setup: Sequence[Statement] = ...,
    ) -> list[dict[str, Any]]: ...

    def _execute(
        self,
        statement: Statement,
        params: Sequence[Any] | None = None,
        *,
        fetch: Literal["none", "one", "all"] = "none",
        setup: Sequence[Statement] = (),
    ) -> dict[str, Any] | list[dict[str, Any]] | None:
        """Run one statement on a pooled connection.

        Parameters
        ----------
        statement :
            The composed SQL to run.
        params :
            Values bound to the statement's placeholders.
        fetch :
            ``"none"`` discards the result, ``"one"`` returns a single row,
            ``"all"`` returns every row.
        setup :
            Statements run first, on the same connection and inside the same
            transaction. Needed for ``SET LOCAL``, whose effect is scoped to
            the transaction and would be lost on a separate connection.

        Returns
        -------
        dict[str, Any] | list[dict[str, Any]] | None
            The rows requested by ``fetch``.
        """
        self._check_open()
        with self._pool.connection() as conn:
            with conn.cursor(row_factory=dict_row) as cur:
                for prelude in setup:
                    cur.execute(prelude)
                cur.execute(statement, params or ())
                if fetch == "one":
                    return cur.fetchone()
                if fetch == "all":
                    return cur.fetchall()
                return None

    def update(
        self, fields: dict[str, Any] | None = None, description: str | None = None
    ) -> bool:
        """Merge changes into the collection's stored metadata.

        Parameters
        ----------
        fields :
            Metadata keys to add or overwrite.
        description :
            Replacement description.

        Returns
        -------
        bool
            Always ``True``; a failure raises instead.
        """
        with self._lock:
            meta = dict(self._schema.raw)
            if description is not None:
                meta["Description"] = description
            if fields:
                meta.update(fields)
            self._execute(
                sql.SQL("UPDATE {} SET meta = %s WHERE name = %s").format(
                    self._registry(ddl.REGISTRY_COLLECTIONS)
                ),
                (json.dumps(meta), self._name),
            )
            self._schema.raw = meta
            if description is not None:
                self._schema.description = description
        return True

    def get_meta_data(self) -> dict[str, Any]:
        """Return the collection metadata recorded in the registry.

        Returns
        -------
        dict[str, Any]
            The stored schema, or the in-memory copy if the registry row is
            missing.
        """
        row = self._execute(
            sql.SQL("SELECT meta FROM {} WHERE name = %s").format(
                self._registry(ddl.REGISTRY_COLLECTIONS)
            ),
            (self._name,),
            fetch="one",
        )
        if not row:
            return dict(self._schema.raw)
        meta = row["meta"]
        return meta if isinstance(meta, dict) else json.loads(meta)

    def close(self) -> None:
        """Mark the collection closed, releasing the pool if it owns one."""
        if self._closed:
            return
        self._closed = True
        if self._owns_pool:
            try:
                self._pool.close()
            except Exception:  # pragma: no cover - shutdown best effort
                logger.debug("Failed to close pgvector pool", exc_info=True)

    def drop(self) -> bool:
        """Drop the table and forget the collection and its indexes.

        Returns
        -------
        bool
            Always ``True``; a failure raises instead.
        """
        self._check_open()
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(self._qualified)
                )
                cur.execute(
                    sql.SQL("DELETE FROM {} WHERE collection = %s").format(
                        self._registry(ddl.REGISTRY_INDEXES)
                    ),
                    (self._name,),
                )
                cur.execute(
                    sql.SQL("DELETE FROM {} WHERE name = %s").format(
                        self._registry(ddl.REGISTRY_COLLECTIONS)
                    ),
                    (self._name,),
                )
        return True

    def create_index(self, index_name: str, meta_data: dict[str, Any]) -> _IndexHandle:
        """Create the ANN, scalar and full-text indexes for this collection.

        OpenViking models an "index" as a named bundle of vector + scalar
        index settings.  Postgres has no such grouping, so the bundle is
        recorded in the registry and its parts are created as real indexes.
        """
        self._check_open()
        vector_meta = meta_data.get("VectorIndex", {}) or {}
        distance = str(vector_meta.get("Distance") or self._distance)
        if distance not in _DISTANCE:
            raise ValueError(
                f"Unsupported distance metric for pgvector: {distance!r}. "
                f"Expected one of: {', '.join(sorted(_DISTANCE))}"
            )
        index_type = str(vector_meta.get("IndexType") or "flat").lower()

        # `CollectionAdapter._build_default_index_meta` always asks for
        # `flat`/`flat_hybrid`, so honouring IndexType alone would make the
        # configured ANN method unreachable.  The explicit `index_method`
        # option wins; `auto` defers to whatever the index meta requested.
        if self._index_method == "auto":
            method = "flat" if index_type.startswith("flat") else "hnsw"
        else:
            method = self._index_method

        statements: list[Statement] = list(
            ddl.scalar_index_statements(self._db_schema, self._table, self._schema)
        )

        vector_field = self._schema.vector_field
        if vector_field is not None:
            vector_stmt = ddl.vector_index_statement(
                self._db_schema,
                self._table,
                vector_field,
                distance,
                method,
                self._index_options,
            )
            if vector_stmt is not None:
                statements.append(vector_stmt)

        fts_stmt = ddl.fulltext_index_statement(
            self._db_schema,
            self._table,
            self._fulltext_specs(),
            self._text_search_config,
        )
        if fts_stmt is not None:
            statements.append(fts_stmt)

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                for statement in statements:
                    cur.execute(statement)
                # Record the method actually built, not the one requested:
                # OpenViking always asks for `flat`/`flat_hybrid`, so a
                # collection configured for hnsw would otherwise be recorded as
                # flat and every process that reopened it would resolve `auto`
                # to the wrong answer.
                recorded = dict(meta_data)
                recorded["ResolvedIndexMethod"] = method
                cur.execute(
                    sql.SQL(
                        "INSERT INTO {} (collection, index_name, meta) "
                        "VALUES (%s, %s, %s) "
                        "ON CONFLICT (collection, index_name) "
                        "DO UPDATE SET meta = EXCLUDED.meta"
                    ).format(self._registry(ddl.REGISTRY_INDEXES)),
                    (self._name, index_name, json.dumps(recorded)),
                )

        with self._lock:
            self._distance = distance
            # The resolver reads pg_indexes; this call just changed them.
            self._resolved_method = None
        return _IndexHandle(index_name, meta_data)

    def has_index(self, index_name: str) -> bool:
        """Return whether an index bundle of this name is registered.

        Parameters
        ----------
        index_name :
            Bundle name to look for.

        Returns
        -------
        bool
            True when the bundle exists.
        """
        row = self._execute(
            sql.SQL(
                "SELECT 1 AS present FROM {} WHERE collection = %s AND index_name = %s"
            ).format(self._registry(ddl.REGISTRY_INDEXES)),
            (self._name, index_name),
            fetch="one",
        )
        return row is not None

    def get_index(self, index_name: str) -> _IndexHandle | None:
        """Return a handle for a registered index bundle.

        Parameters
        ----------
        index_name :
            Bundle name to look up.

        Returns
        -------
        _IndexHandle | None
            The handle, or ``None`` when the bundle is unknown.
        """
        meta = self.get_index_meta_data(index_name)
        if meta is None:
            return None
        return _IndexHandle(index_name, meta)

    def get_index_meta_data(self, index_name: str) -> dict[str, Any] | None:
        """Return the stored metadata for an index bundle.

        Parameters
        ----------
        index_name :
            Bundle name to look up.

        Returns
        -------
        dict[str, Any] | None
            The stored metadata, or ``None`` when the bundle is unknown.
        """
        row = self._execute(
            sql.SQL(
                "SELECT meta FROM {} WHERE collection = %s AND index_name = %s"
            ).format(self._registry(ddl.REGISTRY_INDEXES)),
            (self._name, index_name),
            fetch="one",
        )
        if not row:
            return None
        meta = row["meta"]
        return meta if isinstance(meta, dict) else json.loads(meta)

    def list_indexes(self) -> list[str]:
        """Return the names of every registered index bundle.

        Returns
        -------
        list[str]
            Bundle names for this collection.
        """
        rows = self._execute(
            sql.SQL("SELECT index_name FROM {} WHERE collection = %s").format(
                self._registry(ddl.REGISTRY_INDEXES)
            ),
            (self._name,),
            fetch="all",
        )
        return [row["index_name"] for row in rows or []]

    def update_index(
        self,
        index_name: str,
        scalar_index: dict[str, Any] | None = None,
        description: str | None = None,
    ) -> bool:
        """Update the stored metadata for an index bundle.

        Parameters
        ----------
        index_name :
            Bundle to update.
        scalar_index :
            Replacement scalar-index settings.
        description :
            Replacement description.

        Returns
        -------
        bool
            Always ``True``; a failure raises instead.
        """
        meta = self.get_index_meta_data(index_name) or {}
        if scalar_index is not None:
            meta["ScalarIndex"] = scalar_index
        if description is not None:
            meta["Description"] = description
        self._execute(
            sql.SQL(
                "UPDATE {} SET meta = %s WHERE collection = %s AND index_name = %s"
            ).format(self._registry(ddl.REGISTRY_INDEXES)),
            (json.dumps(meta), self._name, index_name),
        )
        return True

    def drop_index(self, index_name: str) -> bool:
        """Forget an index bundle.

        The physical indexes are deliberately left in place: they are shared
        by every bundle over the same table, and dropping the table drops them.

        Parameters
        ----------
        index_name :
            Bundle to forget.

        Returns
        -------
        bool
            Always ``True``; a failure raises instead.
        """
        self._execute(
            sql.SQL("DELETE FROM {} WHERE collection = %s AND index_name = %s").format(
                self._registry(ddl.REGISTRY_INDEXES)
            ),
            (self._name, index_name),
        )
        return True

    def upsert_data(
        self, data_list: list[dict[str, Any]], ttl: int = 0
    ) -> UpsertDataResult:
        """Insert or replace whole records, keyed by primary key.

        A record replaces the stored row outright: any column it omits is
        written as NULL and any previous ``extra`` is discarded. This matches
        ``LocalCollection``, which stores each record as one JSON document, so
        the two backends stay swappable. Use :meth:`update_data` to change
        selected columns and leave the rest alone.

        Parameters
        ----------
        data_list :
            Records to write. Each must carry the primary key.
        ttl :
            Ignored; PostgreSQL has no row expiry here. A non-zero value logs
            a warning so the caller knows it had no effect.

        Returns
        -------
        UpsertDataResult
            The primary keys written, in input order.

        Raises
        ------
        ValueError
            If a record omits the primary key.
        """
        if not data_list:
            return UpsertDataResult(ids=[])
        if ttl:
            logger.warning("pgvector backend ignores ttl=%s (not implemented)", ttl)

        self._check_open()
        columns, rows, ids, vectorless = self._replacement_rows(data_list)
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                # Checked inside the writing transaction: on its own connection
                # the check would commit before the insert begins, and a
                # concurrent DELETE in that window would let a vectorless row
                # through.
                if vectorless:
                    self._reject_new_vectorless(cur, vectorless)
                cur.executemany(self._upsert_statement(columns), rows)
        return UpsertDataResult(ids=ids)

    def _upsert_statement(self, columns: list[str]) -> Statement:
        """Build the INSERT ... ON CONFLICT statement for a column set."""
        pk = self._schema.primary_key.name
        assignments = [
            sql.SQL("{} = EXCLUDED.{}").format(sql.Identifier(c), sql.Identifier(c))
            for c in columns
            if c != pk
        ]
        conflict = (
            sql.SQL("DO UPDATE SET {}").format(sql.SQL(", ").join(assignments))
            if assignments
            else sql.SQL("DO NOTHING")
        )
        return sql.SQL("INSERT INTO {} ({}) VALUES ({}) ON CONFLICT ({}) {}").format(
            self._qualified,
            sql.SQL(", ").join(sql.Identifier(c) for c in columns),
            sql.SQL(", ").join(sql.Placeholder() * len(columns)),
            sql.Identifier(pk),
            conflict,
        )

    def update_data(self, data_list: list[dict[str, Any]]) -> UpdateResult:
        """Update only the columns present in each record.

        Absent columns keep their stored value, and ``extra`` is merged rather
        than replaced. This is the partial counterpart to :meth:`upsert_data`,
        which replaces the whole row.

        Parameters
        ----------
        data_list :
            Partial records; each must carry the primary key.

        Returns
        -------
        UpdateResult
            Primary keys of the rows that matched and were updated.

        Raises
        ------
        ValueError
            If a record omits the primary key, or names one that does not
            exist. ``LocalCollection.update_data`` also refuses unknown keys,
            so a caller cannot use update as a silent insert on one backend
            and get a no-op on the other.
        """
        if not data_list:
            return UpdateResult(ok=True, ids=[], updated_count=0)

        pk = self._schema.primary_key.name
        vector_field = self._schema.vector_field
        updated: list[str] = []
        missing: list[str] = []
        self._check_open()
        # Checked before writing anything: LocalCollection.update_data raises
        # before it writes, so a batch naming an unknown key must not commit
        # its other records first.
        keys = [str(r[pk]) for r in data_list if pk in r]
        known = {
            str(row[pk])
            for row in self._execute(
                sql.SQL("SELECT {pk} FROM {table} WHERE {pk} = ANY(%s)").format(
                    pk=sql.Identifier(pk), table=self._qualified
                ),
                (keys,),
                fetch="all",
            )
            or []
        }
        absent = [k for k in keys if k not in known]
        if absent:
            raise ValueError(
                f"record not found for primary key(s): {', '.join(sorted(absent))}"
            )

        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                # Sorted by primary key for the same reason upsert sorts:
                # every writer must take row locks in one order or two batches
                # touching the same rows deadlock. Upsert sorting alone was not
                # enough -- it made previously-safe mixed workloads deadlock.
                # Sorted for lock ordering only; `updated` is re-ordered to
                # match the caller's input below, as upsert_data does.
                for record in sorted(data_list, key=lambda r: str(r.get(pk))):
                    if pk not in record:
                        raise ValueError(f"update_data record is missing {pk!r}")
                    if (
                        vector_field is not None
                        and vector_field.name in record
                        and record[vector_field.name] is None
                    ):
                        # Clearing a vector would leave a row that no ANN index
                        # can return, which is what the upsert guard prevents.
                        raise ValueError(
                            f"update_data cannot clear {vector_field.name!r} on "
                            f"{record[pk]!r}: an embedding is required"
                        )
                    columns, values, extra = self._split_record(record)
                    setters = [
                        sql.SQL("{} = {}").format(sql.Identifier(c), sql.Placeholder())
                        for c in columns
                        if c != pk
                    ]
                    params = [v for c, v in zip(columns, values, strict=True) if c != pk]
                    if extra:
                        setters.append(
                            sql.SQL("extra = extra || {}").format(sql.Placeholder())
                        )
                        params.append(json.dumps(extra))
                    if not setters:
                        continue
                    params.append(record[pk])
                    cur.execute(
                        sql.SQL("UPDATE {} SET {} WHERE {} = {}").format(
                            self._qualified,
                            sql.SQL(", ").join(setters),
                            sql.Identifier(pk),
                            sql.Placeholder(),
                        ),
                        params,
                    )
                    if cur.rowcount:
                        updated.append(str(record[pk]))
                    else:
                        missing.append(str(record[pk]))
        if missing:
            raise ValueError(f"record not found for primary key(s): {', '.join(missing)}")
        # Records are written in sorted order for lock safety, but the caller
        # is told which keys landed in the order they supplied them -- as
        # upsert_data and LocalCollection both do.
        written = set(updated)
        ordered = [key for key in keys if key in written]
        return UpdateResult(ok=True, ids=ordered, updated_count=len(ordered))

    def delete_data(self, primary_keys: list[Any]) -> bool:
        """Delete rows by primary key.

        Parameters
        ----------
        primary_keys :
            Keys to remove. Unknown keys are ignored.

        Returns
        -------
        bool
            Always ``True``; a failure raises instead.
        """
        if not primary_keys:
            return True
        pk = self._schema.primary_key.name
        self._execute(
            sql.SQL("DELETE FROM {} WHERE {} = ANY(%s)").format(
                self._qualified, sql.Identifier(pk)
            ),
            (list(primary_keys),),
        )
        return True

    def backfill_defaults(self, *, batch_size: int = 5000) -> int:
        """Set the engine default on every column left NULL by an older write.

        Defaults are applied on write, so rows stored before this backend began
        applying them keep NULL scalars. One collection then holds two
        populations, and a filter such as ``level == 0`` finds the newer rows
        and misses the older ones. Running this once after upgrading makes the
        table consistent.

        Safe to repeat: it only touches columns that are NULL, and never the
        primary key, vectors, timestamps or geo points -- the fields the engine
        itself leaves unset. Safe alongside a concurrent writer: each row is
        locked and re-checked, so a value written meanwhile is preserved.

        Work is committed in batches so a large table does not hold row locks
        for the whole run, produce one enormous WAL burst, or lose all progress
        on interruption.

        Parameters
        ----------
        batch_size :
            Rows to repair per transaction.

        Returns
        -------
        int
            Number of rows updated.
        """
        self._check_open()
        batch = int(batch_size)
        if batch < 1:
            raise ValueError(f"batch_size must be at least 1, got {batch_size!r}")
        assignments: list[Statement] = []
        params: list[Any] = []
        predicates: list[Statement] = []
        for spec in self._schema.fields:
            if spec.is_geo:
                continue
            default = default_for(spec)
            if default is None:
                continue
            column = sql.Identifier(spec.name)
            assignments.append(
                sql.SQL("{} = coalesce({}, {})").format(column, column, sql.Placeholder())
            )
            params.append(self._adapt_value(spec, default))
            predicates.append(sql.SQL("{} IS NULL").format(column))

        if not assignments:
            return 0

        pk = sql.Identifier(self._schema.primary_key.name)
        needs_repair = sql.SQL("({})").format(sql.SQL(" OR ").join(predicates))
        statement = sql.SQL(
            "UPDATE {table} SET {sets} WHERE {pk} IN ("
            "  SELECT {pk} FROM {table} WHERE {needs} LIMIT {limit}"
            ")"
        ).format(
            table=self._qualified,
            sets=sql.SQL(", ").join(assignments),
            pk=pk,
            needs=needs_repair,
            limit=sql.Literal(batch),
        )

        total = 0
        while True:
            with self._pool.connection() as conn, conn.cursor() as cur:
                cur.execute(statement, params)
                repaired = int(cur.rowcount)
            total += repaired
            if repaired < batch:
                return total

    def delete_all_data(self) -> bool:
        """Remove every row, leaving the table and its indexes in place.

        Returns
        -------
        bool
            Always ``True``; a failure raises instead.
        """
        self._execute(sql.SQL("TRUNCATE TABLE {}").format(self._qualified))
        return True

    def fetch_data(self, primary_keys: list[Any]) -> FetchDataInCollectionResult:
        """Fetch whole records by primary key.

        Parameters
        ----------
        primary_keys :
            Keys to fetch.

        Returns
        -------
        FetchDataInCollectionResult
            The records found, plus the keys that matched nothing.
        """
        if not primary_keys:
            return FetchDataInCollectionResult(items=[], ids_not_exist=[])
        pk = self._schema.primary_key.name
        # Every column, vectors included -- not the search projection.
        # `upsert_data` replaces the whole row, and OpenViking's read-modify-write
        # paths (`increment_active_count`, `update_uri_mapping`) feed a fetched
        # record straight back into it. A projection that omitted vectors would
        # write NULL over them and silently drop the row out of every search.
        columns = [spec.name for spec in self._schema.fields]
        rows = self._execute(
            sql.SQL("SELECT {} FROM {} WHERE {} = ANY(%s)").format(
                self._select_list(columns), self._qualified, sql.Identifier(pk)
            ),
            (list(primary_keys),),
            fetch="all",
        )
        items: list[DataItem] = []
        found: set[object] = set()
        for row in rows or []:
            record = self._row_to_record(row, columns)
            identifier = record.pop(pk, None)
            found.add(identifier)
            items.append(DataItem(id=identifier, fields=record))
        missing = [key for key in primary_keys if key not in found]
        return FetchDataInCollectionResult(items=items, ids_not_exist=missing)

    def search_by_vector(
        self,
        index_name: str,
        dense_vector: list[float] | None = None,
        limit: int = 10,
        offset: int = 0,
        filters: dict[str, Any] | None = None,
        sparse_vector: dict[str, float] | None = None,
        output_fields: list[str] | None = None,
    ) -> SearchResult:
        """Rank rows by similarity to a dense and/or sparse query vector.

        Parameters
        ----------
        index_name :
            Accepted for interface compatibility; PostgreSQL picks the index.
        dense_vector :
            Dense query vector.
        limit :
            Maximum rows to return.
        offset :
            Rows to skip.
        filters :
            Compiled filter DSL restricting the candidate set.
        sparse_vector :
            Sparse query vector, used only when ``sparse_weight`` is positive.
        output_fields :
            Columns to return; ``None`` returns all non-vector columns.

        Returns
        -------
        SearchResult
            Matching rows, highest score first.

        Raises
        ------
        ValueError
            If neither a dense nor a sparse vector is supplied.
        """
        if dense_vector is None and sparse_vector is None:
            raise ValueError("search_by_vector requires a dense or sparse vector")

        score_expr, score_params = self._score_expression(dense_vector, sparse_vector)
        predicate, filter_params = self._compiler.compile(filters)
        columns = self._output_columns(output_fields)
        include_extra = not output_fields

        # A row with no vector scores NULL. It is sorted last rather than
        # filtered out: `CollectionAdapter.query` synthesises a random vector
        # for filter-only queries, so excluding those rows would also hide them
        # from `delete(filter=...)`, `scroll` and `fetch_by_uri` -- while
        # `count()` still counted them. That left a record written without an
        # embedding both unfindable and undeletable.
        # Order by the bare distance operator whenever it can decide the
        # ranking on its own. Anything wrapped around it -- the score template,
        # a COALESCE, an extra leading key -- makes the expression unmatchable
        # against the HNSW/IVFFlat operator class, and PostgreSQL falls back to
        # a sequential scan plus a top-N sort. That silently made index_method
        # and the whole iterative-scan mechanism inert.
        #
        # ASC on a distance is also NULLS LAST by default, so a row without an
        # embedding lands behind every row that has one for free.
        order_expr, order_params = self._order_expression(dense_vector, sparse_vector)

        statement = sql.SQL(
            "SELECT {cols}, {score} AS _score, {vec_present} AS __ov_has_vector "
            "FROM {table} WHERE {pred} ORDER BY {order} LIMIT %s OFFSET %s"
        ).format(
            cols=self._select_list(columns, include_extra=include_extra),
            score=score_expr,
            vec_present=self._has_vector_expression(),
            table=self._qualified,
            pred=predicate,
            order=order_expr,
        )
        # Placeholder order follows the statement: SELECT, then WHERE, then
        # ORDER BY.
        params = [*score_params, *filter_params, *order_params, limit, offset]
        rows = self._execute(
            statement,
            params,
            fetch="all",
            setup=self._iterative_scan_setup(),
        )
        # Floored whenever a dense vector was asked for, hybrid included. The
        # dense half of a hybrid score is fabricated for a row with no
        # embedding -- `coalesce(..., 0.0)` is the best possible value for l2
        # and ip, where every real term is negative -- so such a row could
        # outrank genuine matches on a score it did not earn.
        return self._rows_to_search_result(
            rows, columns, floor_vectorless=dense_vector is not None
        )

    def search_by_id(
        self,
        index_name: str,
        id: object,
        limit: int = 10,
        offset: int = 0,
        filters: dict[str, Any] | None = None,
        output_fields: list[str] | None = None,
    ) -> SearchResult:
        """Find neighbours of an existing row, excluding the row itself."""
        vector_field = self._schema.vector_field
        if vector_field is None:
            raise ValueError("Collection has no vector field")
        pk = self._schema.primary_key.name

        row = self._execute(
            sql.SQL("SELECT {} FROM {} WHERE {} = %s").format(
                sql.Identifier(vector_field.name), self._qualified, sql.Identifier(pk)
            ),
            (id,),
            fetch="one",
        )
        if not row or row[vector_field.name] is None:
            return SearchResult(data=[])

        vector = _parse_vector(row[vector_field.name])
        exclusion = {"op": "must_not", "field": pk, "conds": [id]}
        combined: dict[str, Any] = (
            {"op": "and", "conds": [filters, exclusion]} if filters else exclusion
        )
        return self.search_by_vector(
            index_name,
            dense_vector=vector,
            limit=limit,
            offset=offset,
            filters=combined,
            output_fields=output_fields,
        )

    def search_by_keywords(
        self,
        index_name: str,
        keywords: list[str] | None = None,
        query: str | None = None,
        limit: int = 10,
        offset: int = 0,
        filters: dict[str, Any] | None = None,
        output_fields: list[str] | None = None,
    ) -> SearchResult:
        """Lexical search over the indexed text columns via Postgres FTS.

        The native local backend raises NotImplementedError here because it
        cannot vectorise text itself.  Postgres can rank lexically without an
        embedding model, so this is implemented rather than refused.
        """
        terms: list[str] = []
        if keywords:
            terms.extend(str(k) for k in keywords if str(k).strip())
        if query and str(query).strip():
            terms.append(str(query).strip())
        if not terms:
            return SearchResult(data=[])

        specs = self._fulltext_specs()
        if not specs:
            logger.warning(
                "search_by_keywords: no text fields available on collection %s",
                self._name,
            )
            return SearchResult(data=[])

        tsvector = ddl.tsvector_expr(specs, self._text_search_config, self._db_schema)
        # websearch_to_tsquery accepts free-form user input without throwing on
        # punctuation, unlike to_tsquery.
        # One plainto_tsquery per term, OR'd together.
        #
        # plainto_tsquery rather than websearch_to_tsquery: the latter honours
        # `-`, `OR` and quotes as operators, so `-fox` returned the complement
        # of what was asked for. Quoting each term instead turned every
        # multi-word query into a phrase query. But a single plainto over all
        # terms joined ANDs them, so `keywords=["fox", "dog"]` stopped matching
        # either one. Per-term queries keep both properties: words inside a
        # term are ANDed, separate terms are alternatives.
        per_term = sql.SQL(" || ").join(
            sql.SQL("plainto_tsquery({}::regconfig, %s)").format(
                sql.Literal(self._text_search_config)
            )
            for _ in terms
        )
        tsquery = sql.SQL("({})").format(per_term)
        pass  # query text is built per term below

        predicate, filter_params = self._compiler.compile(filters)
        columns = self._output_columns(output_fields)
        include_extra = not output_fields

        statement = sql.SQL(
            "SELECT {cols}, ts_rank({tsv}, {tsq}) AS _score FROM {table} "
            "WHERE {pred} AND {tsv} @@ {tsq} "
            "ORDER BY _score DESC LIMIT %s OFFSET %s"
        ).format(
            cols=self._select_list(columns, include_extra=include_extra),
            tsv=tsvector,
            tsq=tsquery,
            table=self._qualified,
            pred=predicate,
        )
        # Placeholder order: rank's tsquery terms, filter params, then the
        # WHERE clause's copy of the same terms.
        params = [*terms, *filter_params, *terms, limit, offset]
        rows = self._execute(statement, params, fetch="all")
        return self._rows_to_search_result(rows, columns)

    def search_by_multimodal(
        self,
        index_name: str,
        text: str | None,
        image: object | None,
        video: object | None,
        limit: int = 10,
        offset: int = 0,
        filters: dict[str, Any] | None = None,
        output_fields: list[str] | None = None,
    ) -> SearchResult:
        """Reject multimodal search, which needs an embedding model.

        This layer has no vectoriser, so there is nothing to turn text, an
        image, or a video into a query vector. The ``local`` backend declines
        for the same reason.

        Raises
        ------
        NotImplementedError
            Always.
        """
        raise NotImplementedError(
            "PgVectorCollection does not vectorise multimodal input; "
            "call search_by_vector with caller-provided vectors"
        )

    def search_by_random(
        self,
        index_name: str,
        limit: int = 10,
        offset: int = 0,
        filters: dict[str, Any] | None = None,
        output_fields: list[str] | None = None,
    ) -> SearchResult:
        """Return an arbitrary sample of matching rows.

        Parameters
        ----------
        index_name :
            Accepted for interface compatibility; unused.
        limit :
            Maximum rows to return.
        offset :
            Rows to skip.
        filters :
            Compiled filter DSL restricting the candidate set.
        output_fields :
            Columns to return; ``None`` returns all non-vector columns.

        Returns
        -------
        SearchResult
            Randomly ordered rows; scores are not meaningful and are zero.
        """
        predicate, filter_params = self._compiler.compile(filters)
        columns = self._output_columns(output_fields)
        include_extra = not output_fields
        statement = sql.SQL(
            "SELECT {cols}, 0.0::double precision AS _score FROM {table} "
            "WHERE {pred} ORDER BY random() LIMIT %s OFFSET %s"
        ).format(
            cols=self._select_list(columns, include_extra=include_extra),
            table=self._qualified,
            pred=predicate,
        )
        rows = self._execute(statement, [*filter_params, limit, offset], fetch="all")
        return self._rows_to_search_result(rows, columns)

    def search_by_scalar(
        self,
        index_name: str,
        field: str,
        order: str | None = "desc",
        limit: int = 10,
        offset: int = 0,
        filters: dict[str, Any] | None = None,
        output_fields: list[str] | None = None,
    ) -> SearchResult:
        """Return rows ordered by a scalar column.

        Parameters
        ----------
        index_name :
            Accepted for interface compatibility; unused.
        field :
            Column to sort by.
        order :
            ``"desc"`` or ``"asc"``.
        limit :
            Maximum rows to return.
        offset :
            Rows to skip.
        filters :
            Compiled filter DSL restricting the candidate set.
        output_fields :
            Columns to return; ``None`` returns all non-vector columns.

        Returns
        -------
        SearchResult
            Rows in the requested order, with the sort value as the score.
            NULLs sort last in both directions.

        Raises
        ------
        ValueError
            If ``field`` is not declared in the schema.
        """
        spec = self._schema.by_name(field)
        if spec is None:
            raise ValueError(f"Unknown sort field: {field!r}")
        if spec.is_vector or spec.is_sparse:
            raise ValueError(f"Cannot sort on a {spec.ov_type} field: {field!r}")

        direction = sql.SQL("DESC") if str(order).lower() == "desc" else sql.SQL("ASC")
        # Same reason as the range comparisons: PostgreSQL's default collation
        # is not code-point order, so a text sort would disagree with the
        # built-in backend's Python sort.
        sort_col: sql.Composable = sql.Identifier(field)
        if spec.ov_type in ("string", "text", "path"):
            sort_col = sql.SQL('{} COLLATE "C"').format(sql.Identifier(field))
        elif spec.ov_type == "list<string>":
            # An array of text is compared element-wise under the database
            # collation as well: ARRAY['_x'] < ARRAY['a'] is false under
            # en_US.utf8 and true in Python. The CASE keeps NULL as NULL --
            # array(SELECT unnest(NULL)) is the *empty* array, which would make
            # a NULL column sort with the empty ones instead of last.
            sort_col = sql.SQL(
                "CASE WHEN {col} IS NULL THEN NULL "
                'ELSE array(SELECT unnest({col}) COLLATE "C") END'
            ).format(col=sql.Identifier(field))
        predicate, filter_params = self._compiler.compile(filters)
        columns = self._output_columns(output_fields)
        include_extra = not output_fields
        # Selected so it can be used as the score, then dropped again if the
        # caller did not ask for it -- LocalCollection.search_by_scalar does
        # the same.
        borrowed_sort_column = field not in columns
        if borrowed_sort_column:
            columns = [*columns, field]

        statement = sql.SQL(
            "SELECT {cols} FROM {table} WHERE {pred} "
            "ORDER BY {sort} {dir} NULLS LAST LIMIT %s OFFSET %s"
        ).format(
            cols=self._select_list(columns, include_extra=include_extra),
            table=self._qualified,
            pred=predicate,
            sort=sort_col,
            dir=direction,
        )
        rows = self._execute(statement, [*filter_params, limit, offset], fetch="all")

        # The scalar sort key doubles as the score, matching the local backend.
        result = self._rows_to_search_result(rows, columns, score_key=None)
        for item, row in zip(result.data, rows or [], strict=True):
            value = row.get(field)
            item.score = float(value) if isinstance(value, (int, float)) else 0.0
            if borrowed_sort_column and item.fields is not None:
                item.fields.pop(field, None)
        return result

    def aggregate_data(
        self,
        index_name: str,
        op: str = "count",
        field: str | None = None,
        filters: dict[str, Any] | None = None,
        cond: dict[str, Any] | None = None,
    ) -> AggregateResult:
        """Count rows, optionally grouped by a scalar column.

        Parameters
        ----------
        index_name :
            Accepted for interface compatibility; unused.
        op :
            Only ``"count"`` is supported.
        field :
            Column to group by; ``None`` returns a single total.
        filters :
            Compiled filter DSL applied before aggregation.
        cond :
            Post-aggregation bounds on the group counts, such as
            ``{"gt": 10}``.

        Returns
        -------
        AggregateResult
            ``{"_total": n}`` when ungrouped, else one entry per group.

        Raises
        ------
        ValueError
            If ``op`` is not ``"count"``, or ``field`` is undeclared.
        """
        if op != "count":
            raise ValueError(f"Unsupported aggregate op: {op!r}")

        predicate, filter_params = self._compiler.compile(filters)

        if cond:
            # Rejected rather than ignored: a caller asking for groups above a
            # threshold and silently getting all of them has no way to notice.
            unknown = set(cond) - {"gt", "gte", "lt", "lte"}
            if unknown:
                raise ValueError(
                    f"Unsupported aggregate condition(s): {sorted(unknown)}. "
                    "Expected any of: gt, gte, lt, lte"
                )
            for key, bound in cond.items():
                if bound is not None and (
                    isinstance(bound, bool) or not isinstance(bound, (int, float))
                ):
                    raise ValueError(
                        f"Aggregate condition {key!r} must be a number, got {bound!r}"
                    )
            if field is None and any(v is not None for v in cond.values()):
                raise ValueError(
                    "Aggregate conditions apply to groups, so they require a "
                    "grouping field"
                )

        if field is None:
            row = self._execute(
                sql.SQL("SELECT count(*) AS total FROM {} WHERE {}").format(
                    self._qualified, predicate
                ),
                filter_params,
                fetch="one",
            )
            total = int(row["total"]) if row else 0
            return AggregateResult(agg={"_total": total}, op=op, field=None)

        spec = self._schema.by_name(field)
        if spec is None:
            raise ValueError(f"Unknown aggregate field: {field!r}")
        if spec.is_vector or spec.is_sparse:
            raise ValueError(f"Cannot group on a {spec.ov_type} field: {field!r}")

        having: Statement = sql.SQL("")
        having_params: list[Any] = []
        if cond:
            clauses = []
            comparisons = (
                ("gt", sql.SQL(">")),
                ("gte", sql.SQL(">=")),
                ("lt", sql.SQL("<")),
                ("lte", sql.SQL("<=")),
            )
            for key, operator in comparisons:
                if cond.get(key) is not None:
                    clauses.append(
                        sql.SQL("count(*) {} {}").format(operator, sql.Placeholder())
                    )
                    having_params.append(cond[key])
            if clauses:
                having = sql.SQL(" HAVING {}").format(sql.SQL(" AND ").join(clauses))

        statement = sql.SQL(
            "SELECT {group} AS bucket, count(*) AS total FROM {table} "
            "WHERE {pred} GROUP BY {group}{having}"
        ).format(
            group=sql.Identifier(field),
            table=self._qualified,
            pred=predicate,
            having=having,
        )
        rows = self._execute(statement, [*filter_params, *having_params], fetch="all")
        buckets = [(row["bucket"], int(row["total"])) for row in rows or []]
        # The NULL group needs a key of its own -- folding it onto "" collided
        # with the real empty-string group and silently dropped one count. The
        # sentinel is extended until it collides with nothing the column
        # actually holds, so a row literally containing it stays its own group.
        present = {str(b) for b, _ in buckets if b is not None}
        null_key = NULL_BUCKET
        while null_key in present:
            null_key += "_"

        agg: dict[str, Any] = {}
        for bucket, total in buckets:
            agg[null_key if bucket is None else str(bucket)] = total
        return AggregateResult(agg=agg, op=op, field=field)

    def _has_vector_expression(self) -> Statement:
        """Project whether the row carries a dense vector.

        Reported separately from the score so ordering and reporting can use
        different expressions; folding the distinction into the score is what
        broke ranking in both directions previously.

        Returns
        -------
        Statement
            A boolean expression, constant TRUE when the schema has no vector.
        """
        vector_field = self._schema.vector_field
        if vector_field is None:
            return sql.SQL("TRUE")
        return sql.SQL("({} IS NOT NULL)").format(sql.Identifier(vector_field.name))

    def _order_expression(
        self,
        dense_vector: list[float] | None,
        sparse_vector: dict[str, float] | None,
    ) -> tuple[Statement, list[Any]]:
        """Build the ORDER BY, preferring a form an ANN index can serve.

        A pure dense search orders by the bare ``vector <op> query`` distance,
        which is exactly what the HNSW and IVFFlat operator classes match, so
        the planner can use the index. Hybrid search cannot: the sparse term is
        not in any index, so it falls back to ordering by the combined score.

        Parameters
        ----------
        dense_vector :
            The dense query vector, if any.
        sparse_vector :
            The sparse query vector, if any.

        Returns
        -------
        tuple[Statement, list[Any]]
            The ORDER BY expression and the parameters it binds.
        """
        vector_field = self._schema.vector_field
        hybrid = bool(
            sparse_vector and self._schema.sparse_field and self._sparse_weight > 0
        )
        if dense_vector is not None and vector_field is not None and not hybrid:
            operator, _ = _DISTANCE[self._distance]
            return (
                sql.SQL("{} {} {}::vector ASC").format(
                    sql.Identifier(vector_field.name), operator, sql.Placeholder()
                ),
                [_format_vector(dense_vector)],
            )
        # Hybrid, or no dense vector at all: rank by the combined score. When a
        # dense vector was supplied, rows without an embedding are pushed last
        # explicitly -- their dense term is `coalesce(..., 0.0)`, which for l2
        # and ip is the best possible value and would otherwise outrank genuine
        # matches whose score is reported below theirs.
        if dense_vector is not None and vector_field is not None:
            return (
                sql.SQL("({} IS NOT NULL) DESC, _score DESC NULLS LAST").format(
                    sql.Identifier(vector_field.name)
                ),
                [],
            )
        return sql.SQL("_score DESC NULLS LAST"), []

    def _score_expression(
        self,
        dense_vector: list[float] | None,
        sparse_vector: dict[str, float] | None,
    ) -> tuple[sql.Composable, list[Any]]:
        """Build the ranking expression and its parameters.

        Returns
        -------
        tuple[sql.Composable, list[Any]]
            The score expression and the parameters it binds.
        """
        params: list[Any] = []
        terms: list[sql.Composable] = []

        vector_field = self._schema.vector_field
        if dense_vector is not None and vector_field is not None:
            operator, score_template = _DISTANCE[self._distance]
            distance = sql.SQL("({} {} {}::vector)").format(
                sql.Identifier(vector_field.name),
                operator,
                sql.Placeholder(),
            )
            params.append(_format_vector(dense_vector))
            # A missing vector contributes nothing rather than NULL, which
            # would swallow the sparse term as well (NULL + x = NULL). Ordering
            # keeps such a row behind every vectored one regardless.
            terms.append(
                sql.SQL("coalesce({}, 0.0)").format(score_template.format(distance))
            )

        sparse_field = self._schema.sparse_field
        if sparse_vector and sparse_field is not None and self._sparse_weight > 0:
            terms.append(
                sql.SQL("({} * {}.ov_sparse_dot({}, {}::jsonb))").format(
                    sql.Literal(float(self._sparse_weight)),
                    sql.Identifier(self._db_schema),
                    sql.Identifier(sparse_field.name),
                    sql.Placeholder(),
                )
            )
            params.append(json.dumps(sparse_vector))

        if not terms:
            return sql.SQL("0.0::double precision"), []
        return sql.SQL("({})").format(sql.SQL(" + ").join(terms)), params

    def _iterative_scan_setup(self) -> list[Statement]:
        """Build the ``SET LOCAL`` prelude for a filtered ANN search.

        An ANN index visits a fixed candidate pool and only then applies the
        filter, so a selective filter can leave fewer rows than ``limit`` --
        silently, as a short result rather than an error. pgvector 0.8 added
        iterative scan, which keeps widening the search until enough rows
        match. It applies to approximate indexes only: exact search already
        considers every row.

        ``relaxed_order`` allows slight ordering deviation in exchange for
        speed; ``strict_order`` preserves exact distance order. ivfflat
        supports only ``relaxed_order``, so ``strict_order`` degrades to it
        rather than erroring.

        Applies to every ANN search, not only a filtered one: an index scan
        visits at most ``hnsw.ef_search`` candidates (40 by default), so a bare
        ``LIMIT 200`` silently returned 40 rows once the index was actually
        being used.

        Returns
        -------
        list[Statement]
            Zero or one ``SET LOCAL`` statement.
        """
        if self._iterative_scan == "off":
            return []
        method = self._resolved_index_method()
        if method not in ("hnsw", "ivfflat"):
            return []
        if self._pgvector_version < ddl.MIN_VERSION_ITERATIVE_SCAN:
            logger.debug(
                "iterative scan needs pgvector %s; installed %s -- skipping",
                ".".join(str(p) for p in ddl.MIN_VERSION_ITERATIVE_SCAN),
                ".".join(str(p) for p in self._pgvector_version) or "unknown",
            )
            return []

        guc = _SCAN_GUC.get(method)
        if guc is None:
            return []
        mode = self._iterative_scan
        if method == "ivfflat" and mode == "strict_order":
            mode = "relaxed_order"
        return [
            sql.SQL("SET LOCAL {}.iterative_scan = {}").format(guc, sql.Literal(mode))
        ]

    def _resolved_index_method(self) -> str:
        """Return the index method actually in force for this collection.

        Read from ``pg_indexes`` rather than the registry: a collection created
        by an earlier version has no recorded resolution, and
        ``CollectionAdapter.create_collection`` returns early for an existing
        collection, so ``create_index`` never runs again to write one. The
        indexes themselves are the only reliable record of what was built.

        Cached after the first lookup, since the answer cannot change without
        ``create_index`` running.

        Returns
        -------
        str
            ``flat``, ``hnsw`` or ``ivfflat``.
        """
        if self._index_method != "auto":
            return self._index_method
        if self._resolved_method is not None:
            return self._resolved_method

        rows = self._execute(
            sql.SQL(
                "SELECT indexdef FROM pg_indexes WHERE schemaname = %s AND tablename = %s"
            ),
            (self._db_schema, self._table),
            fetch="all",
        )
        resolved = "flat"
        for row in rows or []:
            definition = str(row["indexdef"]).lower()
            if " using hnsw " in definition:
                resolved = "hnsw"
                break
            if " using ivfflat " in definition:
                resolved = "ivfflat"
                break
        self._resolved_method = resolved
        return resolved

    def ensure_indexes(self) -> list[str]:
        """Bring the collection's indexes in line with what this version expects.

        ``CollectionAdapter.create_collection`` returns early when the
        collection already exists, so ``create_index`` never runs again and a
        database created by an earlier version keeps the indexes it had.

        Creating them is not enough on its own. An earlier version built the
        *collated* index under the plain index's name, so the name is already
        taken and ``CREATE INDEX IF NOT EXISTS`` silently skips the plain one --
        reporting success while leaving equality, the commonest filter, with no
        usable index. Any index whose stored definition differs from the one
        this version would create is therefore dropped and rebuilt.

        Each statement commits on its own so the table is locked for one index
        at a time rather than for the whole run; ``CREATE INDEX`` still blocks
        writers for the duration of each.

        Returns
        -------
        list[str]
            Names of indexes that were created or rebuilt.
        """
        self._check_open()
        wanted = ddl.scalar_index_statements(self._db_schema, self._table, self._schema)
        fts = ddl.fulltext_index_statement(
            self._db_schema,
            self._table,
            self._fulltext_specs(),
            self._text_search_config,
        )
        if fts is not None:
            wanted.append(fts)

        existing = {
            str(row["indexname"]): str(row["indexdef"])
            for row in self._execute(
                sql.SQL(
                    "SELECT indexname, indexdef FROM pg_indexes "
                    "WHERE schemaname = %s AND tablename = %s"
                ),
                (self._db_schema, self._table),
                fetch="all",
            )
            or []
        }

        # Only the scalar indexes are reconciled. Their key is a bare column,
        # optionally collated, so comparing definitions is reliable -- and that
        # is exactly where the damage is, a collated index wearing the plain
        # one's name. A full-text key is an expression PostgreSQL rewrites
        # heavily (COALESCE casing, ::text casts, added parentheses), so it can
        # never be matched textually and any attempt would rebuild it forever.
        reconcilable = {
            _index_name_of(st.as_string(None))
            for st in ddl.scalar_index_statements(
                self._db_schema, self._table, self._schema
            )
        }

        changed: list[str] = []
        for statement in wanted:
            rendered = statement.as_string(None)
            name = _index_name_of(rendered)
            if name is None:
                continue
            current = existing.get(name)
            if current is not None and (
                name not in reconcilable or _definitions_agree(current, rendered)
            ):
                continue
            with self._pool.connection() as conn, conn.cursor() as cur:
                if current is not None:
                    # Wrong definition under the right name: the only way to
                    # correct it, since IF NOT EXISTS matches on name alone.
                    cur.execute(
                        sql.SQL("DROP INDEX {}.{}").format(
                            sql.Identifier(self._db_schema), sql.Identifier(name)
                        )
                    )
                cur.execute(statement)
            changed.append(name)
        return sorted(changed)

    def _fulltext_specs(self) -> list[FieldSpec]:
        """Return the schema fields backing keyword search.

        Returns
        -------
        list[FieldSpec]
            Declared, textual columns among the configured keyword fields.
        """
        specs: list[FieldSpec] = []
        for name in fulltext_candidates(self._schema, self._keyword_fields):
            spec = self._schema.by_name(name)
            if spec is not None:
                specs.append(spec)
        return specs

    def _output_columns(self, output_fields: list[str] | None) -> list[str]:
        """Resolve which schema columns a query should return."""
        pk = self._schema.primary_key.name
        if output_fields:
            names = [
                name for name in output_fields if self._schema.by_name(name) is not None
            ]
        else:
            names = self._schema.selectable_names()
        if pk not in names:
            names = [pk, *names]
        return names

    def _select_list(
        self, columns: list[str], *, include_extra: bool = True
    ) -> sql.Composable:
        """Render the SELECT list for the given columns.

        Parameters
        ----------
        columns :
            Schema field names to project.
        include_extra :
            Whether to also select the JSONB overflow column. False when the
            caller asked for an explicit projection, so that out-of-schema
            fields do not slip past it.

        Returns
        -------
        sql.Composable
            A comma-separated select list.
        """
        parts: list[sql.Composable] = []
        for name in columns:
            spec = self._schema.by_name(name)
            if spec is None:
                parts.append(sql.Identifier(name))
            elif spec.is_vector:
                # Render vectors as text; psycopg has no native vector type.
                parts.append(
                    sql.SQL("{}::text AS {}").format(
                        sql.Identifier(name), sql.Identifier(name)
                    )
                )
            elif spec.is_geo:
                # Stored as two real columns; reassembled in _row_to_record.
                parts.append(sql.Identifier(spec.name + GEO_LON_SUFFIX))
                parts.append(sql.Identifier(spec.name + GEO_LAT_SUFFIX))
            else:
                parts.append(sql.Identifier(name))
        if include_extra:
            parts.append(sql.Identifier("extra"))
        return sql.SQL(", ").join(parts)

    def _row_to_record(
        self, row: Mapping[str, Any], columns: list[str]
    ) -> dict[str, Any]:
        """Convert one result row into an OpenViking record."""
        record: dict[str, Any] = {}
        declared: set[str] = set()
        for name in columns:
            spec = self._schema.by_name(name)
            if spec is not None and spec.is_geo:
                declared.add(name)
                lon = row.get(name + GEO_LON_SUFFIX)
                lat = row.get(name + GEO_LAT_SUFFIX)
                if lon is not None and lat is not None:
                    record[name] = f"{lon},{lat}"
                continue
            if name not in row:
                continue
            declared.add(name)
            value = row[name]
            if value is None:
                continue
            if spec is not None and spec.is_vector and isinstance(value, str):
                value = _parse_vector(value)
            record[name] = value
        extra = row.get("extra")
        if isinstance(extra, dict):
            for key, value in extra.items():
                # A declared column wins even when its value is NULL: `extra`
                # may still hold a stale copy written before the field existed
                # in the schema, and returning that would misreport the row.
                if key not in declared:
                    record.setdefault(key, value)
        return record

    def _rows_to_search_result(
        self,
        rows: Sequence[Mapping[str, Any]] | None,
        columns: list[str],
        score_key: str | None = "_score",
        *,
        floor_vectorless: bool = False,
    ) -> SearchResult:
        """Wrap result rows in a SearchResult."""
        pk = self._schema.primary_key.name
        items: list[SearchItemResult] = []
        for row in rows or []:
            record = self._row_to_record(row, columns)
            identifier = record.pop(pk, None)
            score = None
            if score_key is not None:
                raw = row.get(score_key)
                score = float(raw) if isinstance(raw, (int, float)) else 0.0
                # A row with no embedding has no comparable similarity, so it
                # is reported below every real score: OpenViking's retriever
                # re-sorts by `_score` and applies an absolute threshold, and
                # 0.0 would let it overtake a genuine match. Hybrid included --
                # the dense half of its score is fabricated by the COALESCE.
                if floor_vectorless and row.get("__ov_has_vector") is False:
                    score = NO_VECTOR_SCORE
            items.append(SearchItemResult(id=identifier, fields=record, score=score))
        return SearchResult(data=items)

    def _split_record(
        self, record: Mapping[str, Any]
    ) -> tuple[list[str], list[Any], dict[str, Any]]:
        """Split a record into known columns, their values, and JSONB extras."""
        columns: list[str] = []
        values: list[Any] = []
        extra: dict[str, Any] = {}

        for key, value in record.items():
            spec = self._schema.by_name(key)
            if spec is None:
                extra[key] = value
                continue
            if spec.is_geo:
                # geo_point expands into two real columns, as the native
                # engine does; unparseable values fall through to `extra`.
                try:
                    lon, lat = _parse_geo_point(value)
                except (TypeError, ValueError, OverflowError):
                    extra[key] = value
                    continue
                columns.extend([spec.name + "_lon", spec.name + "_lat"])
                values.extend([lon, lat])
                continue
            columns.append(key)
            values.append(self._adapt_value(spec, value))
        return columns, values, extra

    def _replacement_rows(
        self, data_list: list[dict[str, Any]]
    ) -> tuple[list[str], list[list[Any]], list[str], list[str]]:
        """Build one uniform row per record for a replacing upsert.

        Every row binds the same column list -- every writable schema column
        plus ``extra`` -- so a record that omits a field clears it. Because
        there is a single statement, ``executemany`` applies the rows in input
        order, and a batch containing the same primary key twice ends on the
        last occurrence.

        Parameters
        ----------
        data_list :
            Records to write.

        Returns
        -------
        tuple[list[str], list[list[Any]], list[str], list[str]]
            The column list, one value row per record, the primary keys, and
            the keys of records that arrived without an embedding.

        Raises
        ------
        ValueError
            If a record omits the primary key.
        """
        pk = self._schema.primary_key.name
        pk_is_text = self._schema.primary_key.ov_type in ("string", "text", "path")
        vector_field = self._schema.vector_field
        columns: list[str] = []
        for spec in self._schema.fields:
            if spec.is_geo:
                columns.append(spec.name + GEO_LON_SUFFIX)
                columns.append(spec.name + GEO_LAT_SUFFIX)
            else:
                columns.append(spec.name)
        columns.append("extra")

        # The native engine's validator substitutes a per-type default for any
        # omitted field, so an absent `level` is 0 there rather than missing.
        # Storing NULL instead would make the same filter match on one backend
        # and not the other.
        defaults: dict[str, Any] = {}
        for spec in self._schema.fields:
            if spec.is_geo:
                continue
            value = default_for(spec)
            if value is not None:
                defaults[spec.name] = self._adapt_value(spec, value)

        rows: list[list[Any]] = []
        ids: list[str] = []
        vectorless: list[str] = []
        for record in data_list:
            names, values, extra = self._split_record(record)
            mapping: dict[str, Any] = dict(defaults)
            mapping.update(dict(zip(names, values, strict=True)))
            mapping["extra"] = json.dumps(extra) if extra else "{}"
            if mapping.get(pk) is None:
                raise ValueError(f"upsert record is missing primary key {pk!r}")
            if pk_is_text:
                # PostgreSQL writes 3 into a text column happily, but every
                # read path then binds it as an integer and raises
                # `operator does not exist: text = smallint` -- a row that can
                # be written and never read, updated or deleted.
                mapping[pk] = str(mapping[pk])
            if vector_field is not None and mapping.get(vector_field.name) is None:
                vectorless.append(str(mapping[pk]))
            ids.append(str(mapping[pk]))
            rows.append([mapping.get(name) for name in columns])

        # Sorted by primary key so concurrent batches touching the same rows
        # take their locks in the same order. Without it, two writers with
        # overlapping ids in opposite orders deadlock and PostgreSQL rolls one
        # batch back entirely. `sorted` is stable, so the documented
        # last-occurrence-wins behaviour for a repeated key is preserved.
        # `ids` keeps input order, because the caller is told which keys were
        # written, not in which order they hit the database.
        rows = [row for _, row in sorted(zip(ids, rows, strict=True), key=lambda p: p[0])]
        return columns, rows, ids, vectorless

    def _reject_new_vectorless(self, cur: Cursor[Any], ids: list[str]) -> None:
        """Refuse a *new* record with no embedding, but allow rewriting an old one.

        The engine's validator marks the vector Required, and pgvector's index
        builds skip NULL vectors, so a row without one is both impossible there
        and invisible to an index scan here. A database written by an earlier
        version may still hold such rows, and OpenViking's read-modify-write
        paths re-upsert exactly what they fetched -- which for those rows has no
        vector. Rejecting that would leave them permanently unwritable with no
        way to repair them, so an existing row is allowed to be rewritten as it
        already is.

        Parameters
        ----------
        cur :
            Cursor inside the writing transaction, so the check and the insert
            cannot be separated by a concurrent delete.
        ids :
            Primary keys of records that arrived without an embedding.

        Raises
        ------
        ValueError
            If any of them names a row that does not already exist.
        """
        vector_field = self._schema.vector_field
        if vector_field is None:
            return
        pk = sql.Identifier(self._schema.primary_key.name)
        cur.execute(
            sql.SQL(
                "SELECT {pk} FROM {table} WHERE {pk} = ANY(%s) ORDER BY {pk} FOR UPDATE"
            ).format(pk=pk, table=self._qualified),
            (sorted(ids),),
        )
        existing = {str(row[0]) for row in cur.fetchall()}
        new = [i for i in ids if i not in existing]
        if new:
            raise ValueError(
                f"record(s) {', '.join(sorted(new)[:5])!r} have no "
                f"{vector_field.name!r}: an embedding is required, as the "
                "built-in backend requires one"
            )

    def _adapt_value(self, spec: FieldSpec, value: object) -> object:
        """Convert a Python value into something psycopg can bind."""
        if value is None:
            return None
        if spec.is_vector:
            return _format_vector(value)
        if spec.is_sparse:
            return json.dumps(value) if not isinstance(value, str) else value
        if spec.ov_type in ("float32", "int64"):
            # PostgreSQL sorts NaN above every number, while the reference's
            # `_in_range` returns False for every comparison against it, so a
            # stored NaN would diverge on every range filter. pgvector rejects
            # it in a vector column for the same reason; do it for scalars too.
            # Checked after coercion to float, because PostgreSQL accepts the
            # strings "NaN" and "Infinity" for a real column just as readily.
            try:
                numeric = float(value)  # type: ignore[arg-type]
            except (TypeError, ValueError, OverflowError):
                numeric = 0.0
            if math.isnan(numeric) or math.isinf(numeric):
                raise ValueError(
                    f"{spec.ov_type} field {spec.name!r} cannot store {value!r}: "
                    "NaN and infinity have no consistent ordering"
                )
            # Checked after the non-finite guard, since NaN is not integral
            # either and deserves the clearer message.
            if (
                spec.ov_type == "int64"
                and isinstance(value, float)
                and not value.is_integer()
            ):
                # The engine's validator raises int_from_float here; rounding
                # silently would store a value the caller never sent.
                raise ValueError(
                    f"int64 field {spec.name!r} cannot store {value!r}: "
                    "a fractional value is not an integer"
                )
        if spec.is_datetime:
            # The engine's own default for a timestamp is "", and it drops the
            # field rather than storing one. A record copied from the built-in
            # backend carries that value, so it must round-trip to NULL.
            if value == "":
                return None
            return parse_datetime_to_epoch_ms(value, self._tz_policy)
        if spec.is_array:
            parts: list[Any]
            if isinstance(value, str):
                # `;`-joined strings are accepted for list fields, matching
                # DataProcessor._split_str_list.
                parts = value.split(";")
            elif isinstance(value, (list, tuple)):
                parts = list(value)
            else:
                raise ValueError(
                    f"{spec.ov_type} field {spec.name!r} must be a list "
                    "or a ';'-joined string"
                )
            if spec.ov_type == "list<int64>":
                converted: list[int] = []
                for part in parts:
                    if isinstance(part, float) and not part.is_integer():
                        # Matches the scalar path and the engine's validator,
                        # which raises int_from_float rather than truncating.
                        raise ValueError(
                            f"list<int64> field {spec.name!r} cannot store "
                            f"{part!r}: a fractional value is not an integer"
                        )
                    converted.append(int(part))
                return converted
            return [str(p) for p in parts]
        return value


class _IndexHandle:
    """Lightweight stand-in for an index object.

    ``CollectionAdapter.create_collection`` discards this return value; it
    exists so callers that inspect the result get a name and its metadata.

    Parameters
    ----------
    name :
        Index bundle name.
    meta :
        The bundle's stored metadata.

    Attributes
    ----------
    name : str
        Index bundle name.
    meta : dict[str, Any]
        The bundle's stored metadata.
    """

    def __init__(self, name: str, meta: dict[str, Any]) -> None:
        self.name = name
        self.meta = meta

    def get_meta_data(self) -> dict[str, Any]:
        """Return the index bundle's stored metadata."""
        return self.meta

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        """Return a debugging representation naming the index."""
        return f"<PgVectorIndex {self.name!r}>"


_INDEX_NAME_RE = re.compile(r'CREATE INDEX(?: IF NOT EXISTS)? "([^"]+)"')


def _index_name_of(statement: str) -> str | None:
    """Extract the index name from a rendered CREATE INDEX statement.

    Parameters
    ----------
    statement :
        A rendered ``CREATE INDEX`` statement.

    Returns
    -------
    str | None
        The index name, or ``None`` if the statement is not a CREATE INDEX.
    """
    match = _INDEX_NAME_RE.search(statement)
    return match.group(1) if match else None


def _definitions_agree(stored: str, wanted: str) -> bool:
    """Compare an existing index definition with the one we would create.

    ``pg_indexes.indexdef`` is PostgreSQL's own normalised rendering, so it
    never matches our SQL textually. Comparing the parenthesised key, with
    whitespace and quoting stripped, is enough to tell a plain index from a
    collated one -- the distinction that matters here.

    Only sound for a simple column key. PostgreSQL rewrites an expression key
    (function casing, added casts and parentheses) far enough that no textual
    comparison survives, so callers must not use this on one.

    Parameters
    ----------
    stored :
        ``indexdef`` as PostgreSQL reports it.
    wanted :
        The statement this version would run.

    Returns
    -------
    bool
        True when the two describe the same index key.
    """

    def key(text: str) -> str:
        # The *outermost* parenthesised group after USING: a full-text index
        # key contains nested calls, so taking the last open paren would
        # compare only the innermost fragment and never match.
        start = text.find("(", text.find(" USING ") if " USING " in text else 0)
        if start == -1:
            return ""
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    inner = text[start + 1 : i]
                    return (
                        "".join(inner.split()).replace('"', "").replace("'", "").lower()
                    )
        return ""

    stored_key, wanted_key = key(stored), key(wanted)
    if stored_key == wanted_key:
        return True
    # PostgreSQL renders a regconfig literal with an explicit cast that our SQL
    # does not carry; ignore that difference rather than rebuild every run.
    return stored_key.replace("::regconfig", "") == wanted_key.replace("::regconfig", "")


def _format_vector(value: object) -> str:
    """Render a float sequence in pgvector's literal form.

    Parameters
    ----------
    value :
        A sequence of numbers, or an already-formatted literal string.

    Returns
    -------
    str
        A ``[1.0,2.0]``-style literal.

    Raises
    ------
    TypeError
        If ``value`` is neither a string nor a sequence of numbers.
    """
    if isinstance(value, str):
        return value
    if not isinstance(value, Sequence):
        raise TypeError(f"vector value must be a sequence, got {type(value).__name__}")
    return "[" + ",".join(repr(float(v)) for v in value) + "]"


def _parse_vector(value: object) -> list[float]:
    """Parse a pgvector value into a list of floats.

    Parameters
    ----------
    value :
        A sequence of numbers, or pgvector's ``[1.0,2.0]`` text form.

    Returns
    -------
    list[float]
        The vector's components.
    """
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    if not text:
        return []
    return [float(part) for part in text.split(",")]


def _parse_geo_point(value: object) -> tuple[float, float]:
    """Parse ``"lon,lat"``. Mirrors DataProcessor.parse_geo_point."""
    if not isinstance(value, str):
        raise ValueError(f"geo_point value must be string, got {type(value).__name__}")
    raw = value.strip()
    if not raw:
        raise ValueError("geo_point value is empty")
    parts = raw.split(",")
    if len(parts) != 2:
        raise ValueError("geo_point must be in 'lon,lat' format")
    lon, lat = float(parts[0].strip()), float(parts[1].strip())
    if not (-180.0 <= lon <= 180.0):
        raise ValueError("geo_point longitude out of range [-180, 180]")
    if not (-90.0 <= lat <= 90.0):
        raise ValueError("geo_point latitude out of range [-90, 90]")
    return lon, lat
