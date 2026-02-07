"""DVC data versioning tasks."""

import os

from invoke import Context, task

WINDOWS = os.name == "nt"


@task
def pull(ctx: Context) -> None:
    """Pull data from DVC remote.

    Examples:
        invoke dvc.pull
    """
    ctx.run("dvc pull", echo=True, pty=not WINDOWS)


@task
def push(ctx: Context) -> None:
    """Push data to DVC remote.

    Examples:
        invoke dvc.push
    """
    ctx.run("dvc push", echo=True, pty=not WINDOWS)


@task
def add(ctx: Context, folder: str, message: str) -> None:
    """Add data to DVC and push to remote storage.

    Args:
        folder: Path to the folder or file to add to DVC
        message: Commit message for the changes

    Example:
        invoke dvc.add --folder data/raw --message "Add new training data"
    """
    print(f"Adding {folder} to DVC...")
    ctx.run(f"dvc add {folder}", echo=True, pty=not WINDOWS)

    print("\nStaging DVC files in git...")
    ctx.run(f"git add {folder}.dvc .gitignore", echo=True, pty=not WINDOWS)

    print("\nCommitting changes...")
    ctx.run(f'git commit -m "{message}"', echo=True, pty=not WINDOWS)

    print("\nPushing to git remote...")
    ctx.run("git push", echo=True, pty=not WINDOWS)

    print("\nPushing data to DVC remote...")
    ctx.run("dvc push", echo=True, pty=not WINDOWS)

    print(f"\n✓ Successfully added {folder} to DVC and pushed to remotes!")
