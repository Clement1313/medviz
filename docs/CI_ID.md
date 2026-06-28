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

## CD (Docker Compose)

The deployment runs only after a push on `main`, and only after the `check` and
`lint` jobs have passed.

Deployment target:

```text
main -> GitHub Actions -> SSH server -> docker compose up -d --build
```

Required GitHub repository secrets:

```text
DEPLOY_HOST      # server IP or hostname
DEPLOY_USER      # SSH user
DEPLOY_SSH_KEY   # private SSH key authorized on the server
DEPLOY_PATH      # path to the cloned repository on the server
DEPLOY_PORT      # optional, defaults to 22
```

The server must already have:

* Docker with the Compose plugin
* Git access to the repository
* A clone of the repository at `DEPLOY_PATH`

Local run:

```bash
cp .env.example .env
docker compose up -d --build
```

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

manually reformat all files (use on project root)

```bash
uvx pre-commit run --all-files
```

Run it manually on all files:

```bash
uvx pre-commit run --all-files
```
