# ov_postgres

Store OpenViking's vectors in PostgreSQL instead of on the local disk — without patching OpenViking.

## Why this exists

OpenViking ships five vector backends — `local`, `cuvs`, `http`, `volcengine`, and `vikingdb` — and none of them is PostgreSQL. If you already run Postgres, your agent's memory ends up somewhere else: a separate index on one machine's disk, with its own backup story, its own failure modes, and no way to query it alongside your relational data.

`ov_postgres` puts those vectors in Postgres. It installs as an ordinary package and plugs into OpenViking's documented extension point: the adapter factory imports any backend string containing a dot as a Python class path. **Nothing inside OpenViking changes**, so the integration survives upgrades that a patch or a fork would not.

## Features

- **Point one config key at it and your vectors live in Postgres.** No fork, no patch, no vendored code.
- **Every field becomes a real typed column**, so filters compile to ordinary SQL over ordinary indexes — and you can inspect your agent's memory with `psql`.
- **Exact search by default.** Switch to HNSW or IVFFlat when volume demands it, with version-gated iterative scan so a selective filter never silently returns a short page.
- **Lexical keyword search** over your text columns, via PostgreSQL full-text search — no embedding model needed.
- **Filter behaviour is pinned to OpenViking's own evaluator** by 1,089 differential cases, not to one developer's reading of the spec.
- **Runs on managed PostgreSQL** where your role cannot `CREATE EXTENSION`.

## Quick start

You need a PostgreSQL 13+ database with [pgvector](https://github.com/pgvector/pgvector) available, and OpenViking 0.4.16 or newer.

```bash
uv pip install -e . --no-deps
```

Add one block to `~/.openviking/ov.conf`, beside whatever `storage` settings you already have:

```json
{
  "storage": {
    "vectordb": {
      "backend": "ov_postgres.adapter.PgVectorCollectionAdapter",
      "name": "context",
      "custom_params": {
        "dsn": "postgresql://openviking:secret@localhost:5432/openviking"
      }
    }
  }
}
```

Start OpenViking. On first run it creates the extension, three tables, and the indexes. Confirm it took:

```bash
psql "$DSN" -c "\dt public.ov_*"
```

```
 Schema |      Name      | Type  |   Owner
--------+----------------+-------+------------
 public | ov_collections | table | openviking
 public | ov_context     | table | openviking
 public | ov_indexes     | table | openviking
```

Your vectors are now in Postgres.

## Installation

### Prerequisites

| Requirement | Notes |
|---|---|
| PostgreSQL 13+ | With the `vector` extension available |
| OpenViking 0.4.16+ | Ships the dotted-class-path extension point |
| Python 3.10+ | |

Check pgvector is present before you start:

```bash
psql "$DSN" -c "SELECT 1 FROM pg_available_extensions WHERE name='vector'"
```

An empty result means the pgvector binary is not installed on the **server**. No configuration works around that.

### Install the package

```bash
uv pip install -e . --no-deps
```

`--no-deps` matters when installing into an existing OpenViking environment: it stops the resolver from touching packages OpenViking already pinned. For a fresh environment, use `uv sync` instead.

The package must be importable by the process that runs OpenViking, not just from this directory.

### Docker for local development

```bash
docker run -d --name ov-postgres \
  -e POSTGRES_USER=openviking \
  -e POSTGRES_PASSWORD=secret \
  -e POSTGRES_DB=openviking \
  -p 5432:5432 \
  pgvector/pgvector:pg17
```

## How to use

### Keeping the password out of the config file

The DSN resolves from `custom_params.dsn`, then the top-level `url`, then the environment. Omit it from the file and export it instead:

```bash
export OPENVIKING_PGVECTOR_DSN="postgresql://openviking:secret@localhost:5432/openviking"
```

`OPENVIKING_PG_DSN` and `DATABASE_URL` also work.

### Isolating from other tables

Give OpenViking its own schema rather than sharing `public`:

```json
"custom_params": {
  "schema": "openviking",
  "table_prefix": "ov_"
}
```

### Scaling past exact search

The default is exact search: correct always, linear in collection size. Fine into the tens of thousands of rows. Beyond that, switch to an approximate index:

```json
"custom_params": {
  "dsn": "postgresql://...",
  "index_method": "hnsw",
  "index_options": { "m": 16, "ef_construction": 64 },
  "iterative_scan": "relaxed_order"
}
```

`iterative_scan` matters here. An HNSW index visits a fixed candidate pool and *then* applies the filter, so a selective filter can return fewer rows than requested — as a short result rather than an error. Iterative scan widens the search until enough rows match. It requires pgvector 0.8 or newer; on older versions the option is ignored rather than failing.

### Managed PostgreSQL

Where your role cannot create extensions, have an administrator install pgvector once and then:

```json
"custom_params": {
  "dsn": "postgresql://...",
  "create_extension": false
}
```

Startup then fails with an actionable message if the extension is genuinely missing, instead of an opaque permission error.

### Inspecting memory with SQL

Because every field is a real column, your agent's memory is queryable:

```sql
SELECT context_type, count(*)
FROM openviking.ov_context
GROUP BY context_type;

SELECT name, uri, level
FROM openviking.ov_context
WHERE uri LIKE '/user/default/notes%'
ORDER BY updated_at DESC
LIMIT 10;
```

### All configuration options

Every key goes under `custom_params`. Unknown keys are **rejected at startup**, so a typo fails loudly instead of silently leaving a default in place.

| Key | Default | Purpose |
|---|---|---|
| `dsn` | — | libpq connection string |
| `schema` | `public` | PostgreSQL schema for tables and helpers |
| `table_prefix` | `ov_` | Prefix for collection tables |
| `index_method` | `flat` | `flat`, `hnsw`, `ivfflat`, or `auto` |
| `index_options` | `{}` | Passed to `CREATE INDEX ... WITH (...)` |
| `create_extension` | `true` | Run `CREATE EXTENSION` at startup |
| `iterative_scan` | `relaxed_order` | `off`, `strict_order`, or `relaxed_order` |
| `distance` | from `distance_metric` | `cosine`, `l2`, or `ip` |
| `keyword_fields` | name, description, abstract, tags, search_tags | Columns in the full-text index |
| `text_search_config` | `simple` | Text search configuration |
| `tz_policy` | `local` | Timezone for naive timestamps |
| `min_pool_size` / `max_pool_size` | `1` / `8` | Connection pool bounds |
| `connect_timeout` | `10.0` | Seconds to wait for a connection |
| `application_name` | `openviking` | Reported to PostgreSQL |

## How it maps

One collection becomes one table, with a real typed column per declared field.

| OpenViking type | PostgreSQL |
|---|---|
| `string`, `text`, `path` | `text` |
| `int64` | `bigint` |
| `float32` | `real` |
| `bool` | `boolean` |
| `list<string>` / `list<int64>` | `text[]` / `bigint[]` |
| `date_time` | `bigint` (epoch ms, as the native engine stores it) |
| `vector` | `vector(dim)` |
| `sparse_vector` | `jsonb` |
| `geo_point` | two `real` columns (`_lon`, `_lat`) |

Fields written but not declared are kept in a `jsonb` `extra` column rather than dropped — though an explicit `output_fields` projection is honoured and will not return them.

Two registry tables, `ov_collections` and `ov_indexes`, hold collection and index metadata so `get_meta_data()` and `list_indexes()` return exactly what was declared.

### Path scoping

`viking://` URIs are stored path-style (`/user/default/notes`) and decoded on read. The `-d=N` depth parameter is an `IMMUTABLE` SQL function, `ov_path_matches`, ported from `_path_matches` in OpenViking's `cuvs_index.py`.

The prefix test uses plain equality, never `LIKE`. A scoped path is caller data, and `_` or `%` in it would otherwise act as wildcards and pull in sibling subtrees — `/user/default/my_notes` would match `/user/default/myXnotes/…`. Underscores are common in `viking://` URIs, so this is a containment property rather than a nicety.

### Vector precision

pgvector's `vector` type stores **float4**, so components are rounded on write: `1/3` reads back as `0.33333334`. That is pgvector's storage format, not a conversion this package adds. `NaN` and `Inf` are rejected rather than stored.

## What this does not do

- **`search_by_multimodal`** raises `NotImplementedError`: it requires an embedding model this layer does not have.
- **`geo_range`** filters raise `UnsupportedFilterError`. Geo points are stored and read back, but are not queryable by radius.
- **TTL** on `upsert_data` is ignored, with a warning.
- **Server-side grep** never routes here. OpenViking's `_resolve_grep_engine` hard-codes `("volcengine", "vikingdb")`, so grep falls back to the filesystem and the `content` field is not stored.

**`search_by_keywords` is implemented** using PostgreSQL full-text search, so lexical search needs no embedding model.

## Testing

The default run is fast and offline:

```bash
uv run pytest
```

The integration suite starts a `pgvector/pgvector` container through testcontainers, so it needs a running Docker daemon:

```bash
uv run pytest -m integration
```

Point it at a server you already have with `OV_PGVECTOR_TEST_DSN` instead. Each test runs in a throwaway schema that is dropped afterwards.

The suite's centrepiece is `test_filter_semantics_match_reference`: it generates random records and **1,089 filter expressions**, evaluates each one both in PostgreSQL and in Python via OpenViking's own `matches_filter`, and asserts the two agree on every row. That is what pins this backend to native behaviour rather than to an interpretation of it.

## Contributing

Run the gates before opening a pull request:

```bash
uv run ruff check ov_postgres tests
uv run ruff format --check ov_postgres tests
uv run mypy
uv run pytest
```

These are wired into `.pre-commit-config.yaml`; `prek run --all-files` runs them all once the project is a git repository.

Issues and pull requests are welcome. If you hit a filter that behaves differently from the `local` backend, that is a bug — please include the filter and both results.
