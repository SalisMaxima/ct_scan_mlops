# Pre-Commit Hooks

This document describes the pre-commit hooks configured for this project.

## Setup

```bash
# Install hooks (run once after cloning)
pre-commit install

# Run all hooks manually
pre-commit run --all-files

# Update hook versions
pre-commit autoupdate
```

## Configured Hooks

### Repository Hygiene (pre-commit/pre-commit-hooks)

| Hook | Description |
|------|-------------|
| `trailing-whitespace` | Removes trailing whitespace from line endings |
| `end-of-file-fixer` | Ensures files end with a single newline |
| `check-yaml` | Validates YAML file syntax |
| `check-added-large-files` | Prevents committing files larger than 500KB |
| `check-merge-conflict` | Detects leftover merge conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`) |
| `detect-private-key` | Prevents committing private keys (RSA, DSA, EC, PGP) |

### Linting & Formatting (astral-sh/ruff-pre-commit)

| Hook | Description |
|------|-------------|
| `ruff` | Fast Python linter that replaces Flake8, isort, and others. Runs with `--fix` to auto-fix issues |
| `ruff-format` | Code formatter compatible with Black. Ensures consistent code style |

### Security (Yelp/detect-secrets)

| Hook | Description |
|------|-------------|
| `detect-secrets` | Scans for hardcoded secrets (API keys, passwords, tokens). Uses `.secrets.baseline` to track known false positives |

### Spelling (codespell-project/codespell)

| Hook | Description |
|------|-------------|
| `codespell` | Catches common spelling mistakes in code, comments, and documentation. Skips `*.lock`, `*.json`, `*.csv` files |

### Type Checking (pre-commit/mirrors-mypy)

| Hook | Description |
|------|-------------|
| `mypy` | Static type checker for Python. Catches type errors before runtime. Runs with `--ignore-missing-imports` and `--no-strict-optional` |

### Meta Hooks (meta)

| Hook | Description |
|------|-------------|
| `check-hooks-apply` | Validates that all configured hooks match at least one file in the repository |
| `check-useless-excludes` | Ensures exclude patterns actually match files (catches typos in patterns) |

## Skipping Hooks

```bash
# Skip specific hooks temporarily
SKIP=mypy git commit -m "message"

# Skip all hooks (emergency only)
git commit --no-verify -m "message"
```

## Adding False Positives to Secrets Baseline

If `detect-secrets` flags a false positive:

```bash
# Regenerate baseline with the false positive marked
detect-secrets scan --baseline .secrets.baseline

# Audit and mark false positives interactively
detect-secrets audit .secrets.baseline
```

## CI Integration

Pre-commit hooks run automatically in CI via `.github/workflows/pre-commit.yaml` on every push and pull request to `main`.

Hook versions are auto-updated daily via `.github/workflows/pre-commit-update.yaml`.
