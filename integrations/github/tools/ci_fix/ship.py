"""Commit and push a CI fix to a PR head, a repair branch, or straight onto a branch."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from integrations.coding_agent import CodingResult
from integrations.git import (
    BRANCH_FAILED,
    PUSH_RULE_REJECTED,
    GitCommandError,
    assert_not_protected,
    changed_since_baseline,
    checkout_branch,
    commit_paths,
    committed_paths_since,
    create_branch,
    current_branch,
    ensure_git_repo,
    fetch_local_branch,
    head_sha,
    push_base_branch,
    push_branch,
    remote_branch_sha,
)
from integrations.github.client import resolve_github_token
from integrations.github.pull_requests import GitHubPullRequestError, open_pull_request
from integrations.github.tools.ci_fix.context import CiFixContext
from integrations.github.tools.ci_fix.errors import (
    ERR_NO_CHANGES,
    GitHubCiFixError,
)
from integrations.github.tools.ci_fix.storage.attempts import (
    PreparedPush,
    repair_key,
    save_prepared_push,
)
from integrations.github.tools.ci_fix.worktree import build_branch_name

_GIT_TIMEOUT_SEC = 60
_SUBJECT_MAX = 72
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s*")
_SUBJECT_SKIP_LABELS = frozenset(
    {"summary", "root cause", "change", "changes", "verification", "test plan"}
)
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")
#: A first sentence this short ("Fixed.", "Done.") says nothing; the next one is used.
_MIN_SUBJECT_WORDS = 3


class BranchDelivery(StrEnum):
    """How a branch target's fix reaches the branch."""

    REPAIR_BRANCH = "repair_branch"
    """Push a fresh repair branch and leave opening a pull request to the caller."""

    PULL_REQUEST = "pull_request"
    """Push a fresh repair branch and open a pull request into the branch."""

    DIRECT = "direct"
    """Push onto the branch itself; a branch rule that requires pull requests opens one."""


@dataclass(frozen=True)
class PushResult:
    """Outcome of pushing a CI fix to a branch."""

    branch_name: str
    head_sha: str
    changed_files: list[str]
    #: The fix commit landed on the target branch itself, not on a repair branch.
    onto_branch: bool = False
    #: The pull request a refused direct push was turned into; empty otherwise.
    pull_request_url: str = ""
    #: Why a requested direct push shipped as a pull request instead; empty otherwise.
    pr_reason: str = ""


def checkout_target_branch(workspace: str, ctx: CiFixContext, *, token: str | None = None) -> None:
    """Switch the workspace to the PR head branch the fix will edit and push.

    PR mode refuses protected/base branches. Branch-target repairs use a linked
    worktree instead of this path.
    """
    try:
        ensure_git_repo(workspace)
        assert_not_protected(ctx.head_branch, protected_extra=ctx.base_branch)
        if current_branch(workspace) != ctx.head_branch:
            if not _local_branch_exists(workspace, ctx.head_branch):
                fetch_local_branch(workspace, ctx.head_branch, token=token)
            checkout_branch(workspace, ctx.head_branch)
    except GitCommandError as exc:
        raise GitHubCiFixError(exc.kind, exc.message, branch_name=ctx.head_branch) from exc


def push_ci_fix(
    workspace: str,
    *,
    ctx: CiFixContext,
    result: CodingResult,
    baseline: Mapping[str, str] | None = None,
    github_token: str | None = None,
    already_committed: bool = False,
    recorded_through: str | None = None,
    delivery: BranchDelivery = BranchDelivery.REPAIR_BRANCH,
) -> PushResult:
    """Commit files changed by the fix run and push the repair or PR branch.

    ``already_committed`` lets a run whose only change is a base-branch merge
    commit push without new file changes. ``recorded_through`` is the last
    commit whose creation was already reported; a coding agent that committed
    past it is recorded here instead of being treated as an empty repair.
    ``delivery`` other than ``REPAIR_BRANCH`` expects a branch target's own
    branch checked out in a run-owned clone; see :func:`_ship_branch_fix`.
    """
    token = resolve_github_token(github_token)
    pushed_head_sha = ""
    try:
        ensure_git_repo(workspace)
        if current_branch(workspace) != ctx.head_branch:
            if ctx.is_branch_target:
                raise GitHubCiFixError(
                    BRANCH_FAILED,
                    (
                        f"CI fix worktree is not on repair branch {ctx.head_branch}; "
                        "no push was made."
                    ),
                    branch_name=ctx.head_branch,
                )
            checkout_target_branch(workspace, ctx, token=token)
        changed = changed_since_baseline(workspace, baseline=baseline)
        if changed:
            commit_paths(
                workspace,
                changed,
                _commit_message(ctx, result.summary),
                analytics_workflow="github_ci_fix",
            )
        else:
            # A coding agent may commit the repair itself. Those commits are not
            # dirty files, so commit_paths never sees them and would otherwise
            # drop the run as "no changes" without an analytics event.
            known = recorded_through or ctx.head_sha
            current = head_sha(workspace)
            if current == known and not already_committed:
                raise GitHubCiFixError(
                    ERR_NO_CHANGES,
                    f"CI fix for {ctx.target_label} produced no file changes; no push was made.",
                    branch_name=ctx.head_branch,
                )
            if current != known:
                _capture_existing_commit(workspace, known)
        pushed_head_sha = head_sha(workspace)
        source_branch = ctx.target_branch if ctx.is_branch_target else ctx.head_branch
        if remote_branch_sha(workspace, source_branch, token=token) != ctx.head_sha:
            raise GitHubCiFixError(
                "checks_superseded",
                "The remote source head changed during repair; no push was made.",
            )
        save_prepared_push(
            repair_key(
                ctx.owner,
                ctx.repo,
                str(ctx.number) if not ctx.is_branch_target else ctx.target_branch,
            ),
            PreparedPush(ctx.head_sha, pushed_head_sha, ctx.head_branch, changed),
        )
        if ctx.is_branch_target and delivery is not BranchDelivery.REPAIR_BRANCH:
            return _ship_branch_fix(
                workspace,
                ctx=ctx,
                result=result,
                head=pushed_head_sha,
                changed=changed,
                token=token,
                direct=delivery is BranchDelivery.DIRECT,
            )
        push_branch(
            workspace,
            ctx.head_branch,
            base_default="" if ctx.is_branch_target else ctx.base_branch,
            token=token or None,
        )
    except GitCommandError as exc:
        raise GitHubCiFixError(exc.kind, exc.message, branch_name=ctx.head_branch) from exc
    return PushResult(
        branch_name=ctx.head_branch,
        head_sha=pushed_head_sha,
        changed_files=changed,
    )


def _ship_branch_fix(
    workspace: str,
    *,
    ctx: CiFixContext,
    result: CodingResult,
    head: str,
    changed: list[str],
    token: str,
    direct: bool,
) -> PushResult:
    """Land the fix commit on the target branch: pushed onto it, or as a pull request into it.

    ``direct`` pushes onto the branch itself, never forced; a branch rule that only
    accepts pull requests turns that into the pull request, and ``pr_reason`` says why.
    """
    branch = ctx.head_branch
    reason = ""
    if direct:
        try:
            push_base_branch(workspace, branch, token=token or None)
        except GitCommandError as exc:
            if exc.kind != PUSH_RULE_REJECTED:
                raise
            reason = f"{branch} only accepts changes through a pull request"
        else:
            return PushResult(
                branch_name=branch, head_sha=head, changed_files=changed, onto_branch=True
            )
    repair = build_branch_name(ctx)
    create_branch(workspace, repair, base_default=branch)
    push_branch(workspace, repair, base_default=branch, token=token or None)
    try:
        pull = open_pull_request(
            workspace,
            head_branch=repair,
            base_branch=branch,
            title=_commit_message(ctx, result.summary).splitlines()[0],
            body=_pull_request_body(ctx, result, changed, reason),
            github_token=token or None,
        )
    except GitHubPullRequestError as exc:
        raise GitHubCiFixError(exc.kind, exc.message, branch_name=repair) from exc
    return PushResult(
        branch_name=repair,
        head_sha=head,
        changed_files=changed,
        pull_request_url=pull.url,
        pr_reason=reason,
    )


def _pull_request_body(
    ctx: CiFixContext, result: CodingResult, changed: list[str], reason: str
) -> str:
    failing = ", ".join(check.name for check in ctx.failing_checks) or "failing checks"
    lines = [
        f"OpenSRE repaired failing CI on `{ctx.head_branch}` ({failing}) at {ctx.head_sha[:12]}.",
        "",
        "**Fix summary**",
        result.summary.strip() or "_(no summary provided)_",
    ]
    if changed:
        lines += ["", "**Files changed**", *(f"- `{path}`" for path in changed)]
    if reason:
        lines += ["", f"Opened as a pull request because {reason}."]
    return "\n".join(lines)


def _capture_existing_commit(workspace: str, since: str) -> None:
    from infrastructure.analytics.capture import capture_opensre_commit_created

    capture_opensre_commit_created(
        workflow="github_ci_fix",
        commit_kind="content",
        changed_file_count=len(set(committed_paths_since(workspace, since))),
    )


def _commit_message(ctx: CiFixContext, summary: str) -> str:
    subject_tail = _subject_tail(summary, fallback="repair failing CI")
    if ctx.is_branch_target:
        target = ctx.target_branch or ctx.base_branch or ctx.head_branch
    else:
        target = f"PR #{ctx.number}"
    subject = _fit_subject(f"fix: repair CI for {target}", subject_tail)
    lines = [subject, ""]
    if summary.strip():
        lines += [summary.strip(), ""]
    lines.append(f"Generated by OpenSRE from {ctx.url}.")
    return "\n".join(lines)


def _fit_subject(head: str, tail: str) -> str:
    """``head - tail`` within the subject limit: the tail's first telling sentence, cut on a word."""
    sentences = [part.strip() for part in _SENTENCE_END_RE.split(tail) if part.strip()]
    sentence = next(
        (part for part in sentences if len(part.split()) >= _MIN_SUBJECT_WORDS),
        sentences[0] if sentences else tail,
    )
    subject = f"{head} - {sentence.rstrip('.')}"
    if len(subject) <= _SUBJECT_MAX:
        return subject
    cut = subject[: _SUBJECT_MAX - 1].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;:-") + "…"


def _subject_tail(summary: str, *, fallback: str) -> str:
    """Pick a commit-subject fragment from an agent summary.

    Coding agents often start with markdown headings (``## Summary``). Using the
    first raw line produced commits like ``fix: … - ## Summary``; skip those.
    """
    for raw in summary.strip().splitlines():
        line = _MARKDOWN_HEADING_RE.sub("", raw).strip().strip("*").strip()
        if not line or line.lower() in _SUBJECT_SKIP_LABELS:
            continue
        return line
    return fallback


def _local_branch_exists(workspace: str, branch: str) -> bool:
    result = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{branch}"],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT_SEC,
        check=False,
    )
    return result.returncode == 0


__all__ = ["BranchDelivery", "PushResult", "checkout_target_branch", "push_ci_fix"]
