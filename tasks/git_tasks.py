"""Git version control tasks."""

import os

from invoke import Context, task

WINDOWS = os.name == "nt"


@task
def status(ctx: Context) -> None:
    """Show git status.

    Examples:
        invoke git.status
    """
    ctx.run("git status", echo=True, pty=not WINDOWS)


@task
def commit(ctx: Context, message: str) -> None:
    """Commit and push changes to git.

    Args:
        message: Commit message

    Examples:
        invoke git.commit --message "Add new feature"
    """
    ctx.run("git add .", echo=True, pty=not WINDOWS)
    ctx.run(f'git commit -m "{message}"', echo=True, pty=not WINDOWS)
    ctx.run("git push", echo=True, pty=not WINDOWS)


@task
def branch(ctx: Context, name: str, message: str, files: str = ".") -> None:
    """Create a new branch, commit changes, and push to remote.

    Args:
        name: Branch name (e.g., "feature/new-feature" or "docs/readme-update")
        message: Commit message
        files: Files to add (default: "." for all changes)

    Examples:
        invoke git.branch --name feature/auth --message "Add authentication"
        invoke git.branch --name docs/readme --message "Update README" --files README.md
    """
    # Import here to avoid circular dependency
    from tasks.quality import ruff

    print("Running ruff to format and lint code...")
    ruff(ctx)

    print("\nRunning pre-commit hooks to fix formatting issues...")
    ctx.run("uv run pre-commit run --all-files", echo=True, pty=not WINDOWS)

    print(f"\nCreating and switching to branch: {name}")
    ctx.run(f"git checkout -b {name}", echo=True, pty=not WINDOWS)

    print(f"\nAdding files: {files}")
    ctx.run(f"git add {files}", echo=True, pty=not WINDOWS)

    print("\nCommitting changes...")
    ctx.run(f'git commit -m "{message}"', echo=True, pty=not WINDOWS)

    print("\nPushing branch to remote...")
    ctx.run(f"git push -u origin {name}", echo=True, pty=not WINDOWS)

    print(f"\n✓ Branch '{name}' created and pushed!")
    print(f"   Create PR at: https://github.com/SalisMaxima/ct_scan_mlops/compare/{name}")
