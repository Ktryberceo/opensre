"""The hourly CI fix sweep loop: one deterministic sweep per tick, no model turn."""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from config.constants.ci_fleet import CI_FLEET_CRON, CI_FLEET_LOOP_NAME, CI_FLEET_REPORT_BUILDER
from infrastructure.scheduling.scheduler.loop_constants import (
    LOOP_PROMPT_PARAM,
    LOOP_REPORT_ARGS_PARAM,
    LOOP_REPORT_PARAM,
)
from infrastructure.scheduling.scheduler.loops import ManualLoop, create_manual_loop, loop_channels
from infrastructure.scheduling.scheduler.storage import list_tasks, update_task
from infrastructure.scheduling.scheduler.types import Provider, TaskKind, TaskReport

logger = logging.getLogger(__name__)

REPORT_NAME = CI_FLEET_REPORT_BUILDER
"""Builder name the manual-loop runner maps to :func:`build_report`."""

_FALSE = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class ScheduledSweep:
    """The persisted loop and whether it already existed."""

    loop: ManualLoop
    reused: bool


def loop_prompt(repositories: Sequence[str], *, push_to_default_branch: bool = False) -> str:
    """What each tick does, in words; the builder does the work."""
    scope = (
        ", ".join(repositories)
        if repositories
        else "the organization's primary repositories, else its recently active ones"
    )
    default_branch = (
        "pushes default-branch fixes straight onto the default branch"
        if push_to_default_branch
        else "opens a pull request for each default-branch fix"
    )
    return (
        f"Hourly CI fix sweep over {scope}: find failing CI on open pull requests and default "
        "branches and open high or critical security alerts, fix each one with the coding agent "
        f"as the OpenSRE Agent, push pull request fixes to their head branch, {default_branch}, "
        "then verify the checks. Run fix_github_ci_fleet once and report its response_text."
    )


def _report_args(
    repositories: Sequence[str],
    *,
    include_security: bool,
    push_to_default_branch: bool,
    github_connection_id: str,
) -> dict[str, str]:
    args = {
        "repositories": ",".join(repositories),
        "include_security": "true" if include_security else "false",
        "push_to_default_branch": "true" if push_to_default_branch else "false",
    }
    if github_connection_id:
        args["github_connection_id"] = github_connection_id
    return args


def schedule_ci_fleet_loop(
    *,
    repositories: Sequence[str] = (),
    include_security: bool = True,
    push_to_default_branch: bool = False,
    github_connection_id: str = "",
    channel: str = Provider.INTERACTIVE_SHELL.value,
    chat_id: str = "",
    store_path: Path | None = None,
) -> ScheduledSweep:
    """Create the hourly sweep, or update the one that exists: one sweep loop per organization.

    Each tick runs :func:`build_report`, never a model turn, so it is fast and
    its report always comes from the sweep itself.
    """
    args = json.dumps(
        _report_args(
            repositories,
            include_security=include_security,
            push_to_default_branch=push_to_default_branch,
            github_connection_id=github_connection_id,
        ),
        sort_keys=True,
    )
    existing = next(
        (
            task
            for task in list_tasks(store_path)
            if task.kind is TaskKind.MANUAL_LOOP
            and task.params.get(LOOP_REPORT_PARAM) == REPORT_NAME
        ),
        None,
    )
    if existing is not None:
        if existing.params.get(LOOP_REPORT_ARGS_PARAM) != args or not existing.enabled:
            existing.params[LOOP_REPORT_ARGS_PARAM] = args
            existing.params[LOOP_PROMPT_PARAM] = loop_prompt(
                repositories, push_to_default_branch=push_to_default_branch
            )
            existing.enabled = True
            update_task(existing, store_path)
        loop = ManualLoop(
            task=existing, channels=loop_channels(existing), next_run=existing.next_run
        )
        return ScheduledSweep(loop=loop, reused=True)
    created = create_manual_loop(
        name=CI_FLEET_LOOP_NAME,
        prompt=loop_prompt(repositories, push_to_default_branch=push_to_default_branch),
        cron=CI_FLEET_CRON,
        timezone="UTC",
        channels=(channel,),
        slack_chat_id=chat_id if channel == Provider.SLACK.value else "",
        telegram_chat_id=chat_id if channel == Provider.TELEGRAM.value else "",
        store_path=store_path,
        report=REPORT_NAME,
        report_args=json.loads(args),
    )
    return ScheduledSweep(loop=created, reused=False)


def _flag(args: Mapping[str, str], key: str, *, default: bool) -> bool:
    raw = str(args.get(key) or "").strip().lower()
    return default if not raw else raw not in _FALSE


def build_report(args: Mapping[str, str]) -> TaskReport:
    """Deterministic tick: one sweep with the loop's stored scope and GitHub connection."""
    from integrations.github.app_connection import github_setup_url, refreshed_github_token
    from integrations.github.client import GitHubApiError
    from integrations.github.tools.ci_fleet.report import sweep_task_report
    from integrations.github.tools.ci_fleet.sweep import SweepOptions, run_fleet_sweep

    token = refreshed_github_token(args.get("github_connection_id") or None)
    if not token:
        return TaskReport(
            f"CI fix sweep blocked: connect or reconnect GitHub in the OpenSRE app: {github_setup_url()}",
            work_status="blocked",
            error_kind="github_connection_required",
        )
    repositories = tuple(
        name.strip() for name in str(args.get("repositories") or "").split(",") if name.strip()
    )
    options = SweepOptions(
        repositories=repositories,
        include_security=_flag(args, "include_security", default=True),
        push_to_default_branch=_flag(args, "push_to_default_branch", default=False),
    )
    try:
        sweep = run_fleet_sweep(token, options)
    except GitHubApiError as exc:
        logger.warning("CI fix sweep could not list repositories: %s", exc)
        return TaskReport(
            f"CI fix sweep blocked: GitHub would not list the repositories to sweep ({exc}).",
            work_status="blocked",
            error_kind="github_scan_blocked",
        )
    return sweep_task_report(sweep)


__all__ = ["REPORT_NAME", "ScheduledSweep", "build_report", "loop_prompt", "schedule_ci_fleet_loop"]
