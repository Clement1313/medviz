# CI/CD and Pre-commit Setup

## Workflow

```text
feature branch → dev → main
```

* Develop features on a branch created from `dev`.
* Open a Pull Request to `dev`.
* Merge `dev` into `main` via Pull Request only.
* No direct push to `main`. This must be enforced with a GitHub branch protection rule.

## CI (GitHub Actions)

Workflow file:

```text
.github/workflows/ci.yml
```

The CI runs on pushes to `dev` and Pull Requests targeting `dev` or `main`.

### Checks performed

* Verify that services can be installed and imported.
* Run Ruff linting and formatting checks.
* Run the Python unittest suite from `test_suite/`.
* Build and start the Docker Compose stack with `docker compose up -d --build`.
* Ensure that only `dev` can be merged into `main`.

## Main branch protection

GitHub must be configured to block direct pushes to `main`. In the repository
settings, create a branch protection rule for `main` with:

* Require a pull request before merging
* Require status checks to pass before merging
* Require branches to be up to date before merging
* Include administrators if everyone must follow the same rule
* Do not allow bypassing the above settings

The workflow also rejects Pull Requests targeting `main` when the source branch
is not `dev`.

## CD (Docker Compose)

The deployment runs only after a Pull Request from `dev` to `main` has been
merged, and only after the `check`, `lint`, `tests`, `compose`, and `main-pr-source` jobs
have passed.

Deployment target:

```text
dev -> Pull Request -> main -> GitHub Actions -> SSH server -> docker compose up -d --build
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
