"""``ICollection`` implementation backed by PostgreSQL + pgvector."""

from __future__ import annotations

import json
import logging
import math
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
from psycopg import sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from . import ddl
from .filters import FilterCompiler, parse_datetime_to_epoch_ms
from .schema import (
    GEO_LAT_SUFFIX,
    GEO_LON_SUFFIX,
    CollectionSchema,
    FieldSpec,
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

# GUC namespaces for iterative scan, as SQL literals so nothing derived from
# configuration is spliced into a statement as raw text.
_SCAN_GUC: dict[str, sql.SQL] = {
    "hnsw": sql.SQL("hnsw"),
    "ivfflat": sql.SQL("ivfflat"),
}


class PgVectorCollection(ICollection):  # type: ignore[misc]  # ICollection is untyped
    """A single OpenViking collection stored as one PostgreSQL table.

    Instances are safe to share across threads: every statement runs on its own
    pooled connection. The lock serialises *writes* to the cached schema and the
    active distance metric; those attributes and the closed flag are read
    without it, which is sound because each is a single reference assignment --
    a racing reader sees the old or the new value, never a torn one.

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
                cur.execute(
                    sql.SQL(
                        "INSERT INTO {} (collection, index_name, meta) "
                        "VALUES (%s, %s, %s) "
                        "ON CONFLICT (collection, index_name) "
                        "DO UPDATE SET meta = EXCLUDED.meta"
                    ).format(self._registry(ddl.REGISTRY_INDEXES)),
                    (self._name, index_name, json.dumps(meta_data)),
                )

        with self._lock:
            self._distance = distance
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
        columns, rows, ids = self._replacement_rows(data_list)
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
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
        updated: list[str] = []
        missing: list[str] = []
        self._check_open()
        with self._pool.connection() as conn:
            with conn.cursor() as cur:
                for record in data_list:
                    if pk not in record:
                        raise ValueError(f"update_data record is missing {pk!r}")
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
        return UpdateResult(ok=True, ids=updated, updated_count=len(updated))

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
        statement = sql.SQL(
            "SELECT {cols}, {score} AS _score FROM {table} WHERE {pred} "
            "ORDER BY _score DESC NULLS LAST LIMIT %s OFFSET %s"
        ).format(
            cols=self._select_list(columns, include_extra=include_extra),
            score=score_expr,
            table=self._qualified,
            pred=predicate,
        )
        params = [*score_params, *filter_params, limit, offset]
        rows = self._execute(
            statement,
            params,
            fetch="all",
            setup=self._iterative_scan_setup(filtered=bool(filters)),
        )
        return self._rows_to_search_result(rows, columns)

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
        tsquery = sql.SQL("websearch_to_tsquery({}::regconfig, %s)").format(
            sql.Literal(self._text_search_config)
        )
        query_text = " OR ".join(terms) if len(terms) > 1 else terms[0]

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
        # Placeholder order: rank's tsquery, filter params, WHERE's tsquery.
        params = [query_text, *filter_params, query_text, limit, offset]
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

        direction = sql.SQL("DESC") if str(order).lower() == "desc" else sql.SQL("ASC")
        predicate, filter_params = self._compiler.compile(filters)
        columns = self._output_columns(output_fields)
        include_extra = not output_fields
        if field not in columns:
            columns = [*columns, field]

        statement = sql.SQL(
            "SELECT {cols} FROM {table} WHERE {pred} "
            "ORDER BY {sort} {dir} NULLS LAST LIMIT %s OFFSET %s"
        ).format(
            cols=self._select_list(columns, include_extra=include_extra),
            table=self._qualified,
            pred=predicate,
            sort=sql.Identifier(field),
            dir=direction,
        )
        rows = self._execute(statement, [*filter_params, limit, offset], fetch="all")

        # The scalar sort key doubles as the score, matching the local backend.
        result = self._rows_to_search_result(rows, columns, score_key=None)
        for item, row in zip(result.data, rows or [], strict=True):
            value = row.get(field)
            item.score = float(value) if isinstance(value, (int, float)) else 0.0
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
        agg = {
            ("" if row["bucket"] is None else str(row["bucket"])): int(row["total"])
            for row in rows or []
        }
        return AggregateResult(agg=agg, op=op, field=field)

    def _score_expression(
        self,
        dense_vector: list[float] | None,
        sparse_vector: dict[str, float] | None,
    ) -> tuple[sql.Composable, list[Any]]:
        """Build the ranking expression and its parameters."""
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
            terms.append(score_template.format(distance))

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

    def _iterative_scan_setup(self, *, filtered: bool) -> list[Statement]:
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

        Parameters
        ----------
        filtered :
            Whether the query carries a filter. Unfiltered ANN search returns
            a full page already, so the GUC would only add cost.

        Returns
        -------
        list[Statement]
            Zero or one ``SET LOCAL`` statement.
        """
        if not filtered or self._iterative_scan == "off":
            return []
        if self._index_method not in ("hnsw", "ivfflat"):
            return []
        if self._pgvector_version < ddl.MIN_VERSION_ITERATIVE_SCAN:
            logger.debug(
                "iterative scan needs pgvector %s; installed %s -- skipping",
                ".".join(str(p) for p in ddl.MIN_VERSION_ITERATIVE_SCAN),
                ".".join(str(p) for p in self._pgvector_version) or "unknown",
            )
            return []

        guc = _SCAN_GUC.get(self._index_method)
        if guc is None:
            return []
        mode = self._iterative_scan
        if self._index_method == "ivfflat" and mode == "strict_order":
            mode = "relaxed_order"
        return [
            sql.SQL("SET LOCAL {}.iterative_scan = {}").format(guc, sql.Literal(mode))
        ]

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
                except (TypeError, ValueError):
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
    ) -> tuple[list[str], list[list[Any]], list[str]]:
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
        tuple[list[str], list[list[Any]], list[str]]
            The column list, one value row per record, and the primary keys.

        Raises
        ------
        ValueError
            If a record omits the primary key.
        """
        pk = self._schema.primary_key.name
        columns: list[str] = []
        for spec in self._schema.fields:
            if spec.is_geo:
                columns.append(spec.name + GEO_LON_SUFFIX)
                columns.append(spec.name + GEO_LAT_SUFFIX)
            else:
                columns.append(spec.name)
        columns.append("extra")

        rows: list[list[Any]] = []
        ids: list[str] = []
        for record in data_list:
            names, values, extra = self._split_record(record)
            mapping: dict[str, Any] = dict(zip(names, values, strict=True))
            mapping["extra"] = json.dumps(extra) if extra else "{}"
            if mapping.get(pk) is None:
                raise ValueError(f"upsert record is missing primary key {pk!r}")
            ids.append(str(mapping[pk]))
            rows.append([mapping.get(name) for name in columns])
        return columns, rows, ids

    def _adapt_value(self, spec: FieldSpec, value: object) -> object:
        """Convert a Python value into something psycopg can bind."""
        if value is None:
            return None
        if spec.is_vector:
            return _format_vector(value)
        if spec.is_sparse:
            return json.dumps(value) if not isinstance(value, str) else value
        if spec.ov_type == "float32" and isinstance(value, float):
            # PostgreSQL sorts NaN above every number, while the reference's
            # `_in_range` returns False for every comparison against it, so a
            # stored NaN would diverge on every range filter. pgvector rejects
            # it in a vector column for the same reason; do it for scalars too.
            if math.isnan(value) or math.isinf(value):
                raise ValueError(
                    f"float32 field {spec.name!r} cannot store {value!r}: "
                    "NaN and infinity have no consistent ordering"
                )
        if spec.is_datetime:
            return parse_datetime_to_epoch_ms(value, self._tz_policy)
        if spec.is_array:
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
                return [int(p) for p in parts]
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
