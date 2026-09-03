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

## CI

One workflow, [`ci.yaml`](.github/workflows/ci.yaml), covers the repo: repo-wide checks run once, then [`template-check.yaml`](.github/workflows/template-check.yaml) tests and builds each package whose files changed. Adding a package means adding one filter block and one name to `ci.yaml`, and its directory to the workspace members in [`pyproject.toml`](pyproject.toml).

## Releases

Each package releases on its own, from the manual [`release.yaml`](.github/workflows/release.yaml) workflow (Actions → release). Pick the package, pick the version increment — `auto` derives it from conventional commits since the package's last tag — and run with `dry_run` first to see the plan. A real run re-tests the package, pushes an annotated `<package>-v<version>` tag, and publishes a GitHub release carrying the built artifacts. Nothing releases on push.

## License

AGPL-3.0-only. See [LICENSE](LICENSE).
