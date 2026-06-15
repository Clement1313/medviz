# CI/CD and Pre-commit Setup

## Workflow

```text
feature branch → dev → main
```

* Develop features on a branch created from `dev`.
* Open a Pull Request to `dev`.
* Merge `dev` into `main` via Pull Request only.
* No direct push to `main`.

## CI (GitHub Actions)

Workflow file:

```text
.github/workflows/ci.yml
```

The CI runs on pushes and Pull Requests targeting `dev` or `main`.

### Checks performed

* Verify that services can be installed and started.
* Run Ruff linting and formatting checks.
* Ensure that only `dev` can be merged into `main`.

## Pre-commit

Pre-commit automatically runs Ruff before each commit:

```text
ruff check --fix
ruff format
```

Install it once after cloning the repository:

```bash
uvx pre-commit install
```

Run it manually on all files:

```bash
uvx pre-commit run --all-files
```