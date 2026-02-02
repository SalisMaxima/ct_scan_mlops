"""Code quality and testing tasks."""

import os

from invoke import Context, task

WINDOWS = os.name == "nt"


@task
def ruff(ctx: Context, check_only: bool = False) -> None:
    """Run ruff linter and formatter.

    Args:
        check_only: Only check without modifying files (useful for CI)

    Examples:
        invoke quality.ruff                 # Fix and format code
        invoke quality.ruff --check-only    # Check only (CI mode)
    """
    if check_only:
        ctx.run("uv run ruff check .", echo=True, pty=not WINDOWS)
        ctx.run("uv run ruff format . --check", echo=True, pty=not WINDOWS)
    else:
        ctx.run("uv run ruff check . --fix", echo=True, pty=not WINDOWS)
        ctx.run("uv run ruff format .", echo=True, pty=not WINDOWS)


@task
def test(ctx: Context) -> None:
    """Run tests with coverage report.

    Examples:
        invoke quality.test
    """
    ctx.run("uv run coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("uv run coverage report -m -i", echo=True, pty=not WINDOWS)


@task
def test_unit(ctx: Context) -> None:
    """Run fast unit tests only (excludes slow and data-dependent tests).

    Examples:
        invoke quality.test-unit
    """
    ctx.run("uv run pytest tests/ -m 'not slow and not requires_data' -v", echo=True, pty=not WINDOWS)


@task
def test_all(ctx: Context) -> None:
    """Run all tests including slow and data-dependent ones.

    Examples:
        invoke quality.test-all
    """
    ctx.run("uv run pytest tests/ -v", echo=True, pty=not WINDOWS)


@task
def test_watch(ctx: Context) -> None:
    """Run tests in watch mode for development.

    Requires pytest-watch to be installed.

    Examples:
        invoke quality.test-watch
    """
    from loguru import logger

    try:
        ctx.run("uv run pytest-watch tests/", echo=True, pty=not WINDOWS)
    except Exception:
        logger.error("pytest-watch not found. Install with: uv add --dev pytest-watch")
        raise


@task(pre=[ruff])
def ci(ctx: Context) -> None:
    """Run full CI pipeline locally (lint + format check + tests).

    This runs the same checks as CI but locally, useful before committing.

    Examples:
        invoke quality.ci
    """
    print("\n" + "=" * 60)
    print("Running CI Pipeline")
    print("=" * 60 + "\n")

    print("Step 1/3: Linting...")
    ctx.run("uv run ruff check .", echo=True, pty=not WINDOWS)

    print("\nStep 2/3: Format check...")
    ctx.run("uv run ruff format . --check", echo=True, pty=not WINDOWS)

    print("\nStep 3/3: Running tests...")
    test_unit(ctx)

    print("\n" + "=" * 60)
    print("✓ CI Pipeline Passed!")
    print("=" * 60)


@task
def security_check(ctx: Context) -> None:
    """Run security vulnerability checks on code and dependencies.

    Examples:
        invoke quality.security-check
    """
    print("Running security checks...\n")

    print("1. Checking for known vulnerabilities in dependencies...")
    result = ctx.run("uv run pip-audit", warn=True, echo=True, pty=not WINDOWS)

    print("\n2. Running bandit security linter...")
    ctx.run("uv run bandit -r src/ -f screen", warn=True, echo=True, pty=not WINDOWS)

    if result and result.ok:
        print("\n✓ Security checks passed!")
    else:
        print("\n⚠ Security issues found - please review above")


@task
def install_hooks(ctx: Context) -> None:
    """Install git pre-commit hooks.

    Examples:
        invoke quality.install-hooks
    """
    ctx.run("uv run pre-commit install", echo=True, pty=not WINDOWS)
    ctx.run("uv run pre-commit install --hook-type commit-msg", echo=True, pty=not WINDOWS)
    print("\n✓ Git hooks installed!")


@task
def deps_outdated(ctx: Context) -> None:
    """Check for outdated dependencies.

    Examples:
        invoke quality.deps-outdated
    """
    ctx.run("uv pip list --outdated", echo=True, pty=not WINDOWS)


@task
def deps_tree(ctx: Context) -> None:
    """Show dependency tree.

    Examples:
        invoke quality.deps-tree
    """
    ctx.run("uv pip tree", echo=True, pty=not WINDOWS)
