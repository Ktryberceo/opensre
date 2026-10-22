"""Linked git worktree setup for branch CI fixes."""

from __future__ import annotations

import re
import shutil
import subprocess
import uuid
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from integrations.git import BRANCH_FAILED, GitCommandError, ensure_git_repo, fetch_remote_branch
from integrations.github.tools.ci_fix.context import CiFixContext
from integrations.github.tools.ci_fix.errors import GitHubCiFixError

_GIT_TIMEOUT_SEC = 60
_BRANCH_PREFIX = "opensre/ci-fix"


@dataclass(frozen=True)
class BranchWorktree:
    """Fresh linked worktree and repair branch for a branch-targeted CI fix."""

    path: str
    branch_name: str


def build_branch_name(ctx: CiFixContext) -> str:
    """Return a unique, namespaced repair branch name."""
    branch = _slug(ctx.target_branch or ctx.base_branch or "branch")
    suffix = (ctx.head_sha or uuid.uuid4().hex)[:12]
    unique = uuid.uuid4().hex[:8]
    return f"{_BRANCH_PREFIX}-{branch}-{suffix}-{unique}"


def create_branch_worktree(
    workspace: str, ctx: CiFixContext, *, token: str | None = None
) -> BranchWorktree:
    """Create a linked worktree on a fresh repair branch based on the target branch."""
    target_branch = ctx.target_branch or ctx.base_branch
    branch_name = build_branch_name(ctx)
    path = _worktree_path(workspace, target_branch)
    try:
        ensure_git_repo(workspace)
        fetch_remote_branch(workspace, target_branch, token=token)
        result = subprocess.run(
            [
                "git",
                "worktree",
                "add",
                "-b",
                branch_name,
                str(path),
                f"origin/{target_branch}",
            ],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SEC,
            check=False,
        )
    except GitCommandError as exc:
        raise GitHubCiFixError(exc.kind, exc.message, branch_name=branch_name) from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GitHubCiFixError(
            BRANCH_FAILED,
            f"Could not create CI fix worktree for branch '{target_branch}': {exc}",
            branch_name=branch_name,
        ) from exc
    if result.returncode != 0:
        _remove_path(path)
        raise GitHubCiFixError(
            BRANCH_FAILED,
            (
                f"Could not create CI fix worktree for branch '{target_branch}': "
                f"{result.stderr.strip()}"
            ),
            branch_name=branch_name,
        )
    return BranchWorktree(path=str(path), branch_name=branch_name)


def create_branch_checkout(
    workspace: str, ctx: CiFixContext, *, token: str | None = None
) -> BranchWorktree:
    """Create the repair branch in an already-isolated checkout.

    A remote repair starts in a run-owned clone, so adding a linked worktree
    would check out the repository a second time. Explicit user checkouts still
    use :func:`create_branch_worktree` to isolate their branch and local WIP.
    """
    return _checkout_from_target(workspace, ctx, build_branch_name(ctx), "-b", token=token)


def create_target_checkout(
    workspace: str, ctx: CiFixContext, *, token: str | None = None
) -> BranchWorktree:
    """Check out the target branch itself in a run-owned clone, for a fix pushed straight to it.

    The clone already has a local branch of that name (its default branch), so
    it is reset to the fetched remote head rather than created.
    """
    target = ctx.target_branch or ctx.base_branch
    return _checkout_from_target(workspace, ctx, target, "-B", token=token)


def _checkout_from_target(
    workspace: str, ctx: CiFixContext, branch_name: str, create_flag: str, *, token: str | None
) -> BranchWorktree:
    """Point *branch_name* at the fetched target branch and switch the clone to it."""
    target_branch = ctx.target_branch or ctx.base_branch
    try:
        ensure_git_repo(workspace)
        fetch_remote_branch(workspace, target_branch, token=token)
        result = subprocess.run(
            [
                "git",
                "checkout",
                create_flag,
                branch_name,
                f"origin/{target_branch}",
            ],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SEC,
            check=False,
        )
    except GitCommandError as exc:
        raise GitHubCiFixError(exc.kind, exc.message, branch_name=branch_name) from exc
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GitHubCiFixError(
            BRANCH_FAILED,
            f"Could not create CI fix branch from '{target_branch}': {exc}",
            branch_name=branch_name,
        ) from exc
    if result.returncode != 0:
        raise GitHubCiFixError(
            BRANCH_FAILED,
            (f"Could not create CI fix branch from '{target_branch}': {result.stderr.strip()}"),
            branch_name=branch_name,
        )
    return BranchWorktree(path=workspace, branch_name=branch_name)


def cleanup_branch_worktree(workspace: str, worktree: BranchWorktree) -> None:
    """Best-effort removal of a temporary branch-fix worktree and local branch."""
    with suppress(OSError, subprocess.TimeoutExpired):
        result = subprocess.run(
            ["git", "worktree", "remove", "--force", worktree.path],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SEC,
            check=False,
        )
        if result.returncode != 0:
            _remove_path(Path(worktree.path))
    # Drop the local repair ref so repeated branch fixes do not accumulate names.
    with suppress(OSError, subprocess.TimeoutExpired):
        subprocess.run(
            ["git", "branch", "-D", worktree.branch_name],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SEC,
            check=False,
        )


def _worktree_path(workspace: str, branch: str) -> Path:
    parent = Path(workspace).expanduser().resolve().parent
    while True:
        candidate = parent / f".opensre-ci-fix-{_slug(branch)}-{uuid.uuid4().hex[:8]}"
        if not candidate.exists():
            return candidate


def _slug(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value or "")).strip("-")
    return cleaned or "branch"


def _remove_path(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


__all__ = [
    "BranchWorktree",
    "build_branch_name",
    "cleanup_branch_worktree",
    "create_branch_checkout",
    "create_branch_worktree",
    "create_target_checkout",
]
