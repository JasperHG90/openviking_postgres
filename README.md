# openviking

Packages that extend [OpenViking](https://github.com/volcengine/OpenViking), collected in one repository.

## Packages

| Package | Language | Description |
| --- | --- | --- |
| [`ov-postgres`](packages/ov-postgres/) | Python | PostgreSQL + pgvector backend for OpenViking's vector store |

Each package has its own README with install and usage instructions.

## Repository layout

```
packages/
  ov-postgres/     Python package (uv workspace member)
```

Python packages are members of a single [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/): one `uv.lock` and one `.venv` at the root cover all of them. Packages in other languages will live under `packages/` beside them with their own toolchains.

## Development

```bash
uv sync --all-packages        # install every Python package with its dev group
uvx prek run --all-files      # lint, type-check, and unit-test everything
```

To work on one package, run commands scoped to it:

```bash
uv run --directory packages/ov-postgres pytest
```

## Releases

Each package releases on its own tag: `<package>-v<version>`, e.g. `ov-postgres-v0.3.0`. Pushing such a tag builds that package and publishes a GitHub release carrying its artifacts.

## License

AGPL-3.0-only. See [LICENSE](LICENSE).
