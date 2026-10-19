"""Action tools: run one fleet-wide CI and security fix sweep now, or schedule it hourly."""

from __future__ import annotations

from typing import Any

from config.constants.ci_fleet import CI_FLEET_MAX_PARALLEL
from core.agent_harness.tools import action_context_from_agent_context
from core.domain.types.tools import ToolSurface
from core.tool import SideEffectLevel
from core.tool_framework import tool
from core.tool_framework.utils import tool_unavailable
from infrastructure.scheduling.scheduler.reload_signal import request_scheduler_reload
from infrastructure.scheduling.scheduler.types import Provider
from integrations.github.agent_tools import (
    github_tool_available,
    github_tool_params,
    require_webapp_github,
)
from integrations.github.app_connection import github_setup_url
from integrations.github.client import GitHubApiError, resolve_github_token
from integrations.github.envelope import missing_token_envelope
from integrations.github.helpers import GITHUB_INJECTED_PARAMS
from integrations.github.tools.ci_fleet.loop import schedule_ci_fleet_loop
from integrations.github.tools.ci_fleet.report import render_sweep
from integrations.github.tools.ci_fleet.sweep import SweepOptions, run_fleet_sweep
from tools.system.scheduled_loops.destination import conversation_destination

SWEEP_TOOL_NAME = "fix_github_ci_fleet"
SCHEDULE_TOOL_NAME = "schedule_github_ci_fleet_fix"
_SOURCE = "github"
_CHAT_CHANNELS = frozenset({Provider.SLACK.value, Provider.TELEGRAM.value})

_REPOSITORIES_PROPERTY = {
    "type": "array",
    "items": {"type": "string"},
    "description": (
        "owner/repo names to sweep. Omit to let the sweep choose: the organization's "
        "primary repositories, else every non-fork repository pushed to in the last two "
        "weeks. Never-change repositories are always left out."
    ),
}
_SECURITY_PROPERTY = {
    "type": "boolean",
    "description": "Also fix open high or critical security alerts; default true.",
}
_DEFAULT_BRANCH_PROPERTY = {
    "type": "boolean",
    "description": (
        "True only when the user explicitly asks for default-branch and security fixes to be "
        "pushed straight onto the default branch (e.g. 'push straight to main'). Default "
        "false: each of those fixes is opened as a pull request. Pull request fixes always "
        "go to the pull request's own branch."
    ),
}


def _action_context(context: Any) -> Any:
    if context is None:
        return None
    try:
        return action_context_from_agent_context(context)
    except RuntimeError:
        return None


def _progress(context: Any) -> Any:
    console = getattr(_action_context(context), "console", None)
    if console is None or not getattr(console, "is_terminal", False):
        return None

    def show(line: str) -> None:
        console.print(f"  [dim]{line}[/dim]")

    return show


def _repositories(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = value.replace(",", " ").split()
    if not isinstance(value, list):
        return ()
    return tuple(str(name).strip() for name in value if str(name or "").strip())


@tool(
    name=SWEEP_TOOL_NAME,
    source=_SOURCE,
    display_name="Fix CI across repositories",
    compact_description=(
        "Sweep many repositories for red pull requests, red default branches and security "
        "alerts, and fix each one."
    ),
    description=(
        "One sweep over many repositories at once: finds every open pull request and default "
        "branch whose head has failing CI, plus open high or critical Dependabot and code "
        "scanning alerts, and fixes them in parallel with the coding agent. Each fix is "
        "committed as the OpenSRE Agent; a pull request's fix is pushed to its own head "
        "branch, and a default-branch or security fix is opened as a pull request (or pushed "
        "onto the default branch with push_to_default_branch). The new checks are verified, a "
        "fix that stays red gets a second attempt, and a commit an earlier sweep could not fix "
        "is retried only after a day. Forks and drafts are skipped; nothing is merged. Without "
        "repositories it sweeps the organization's primary repositories, else its recently "
        "active ones. dry_run only reports what it would fix. Show response_text as is."
    ),
    use_cases=[
        "Fix every broken CI across our repositories",
        "Scan all our repos for failing PRs, a red main and vulnerabilities, and fix them",
        "Sweep Tracer-Cloud/opensre and Tracer-Cloud/opensre-webapp for broken CI now",
        "What CI is broken across our repos right now (dry run)",
    ],
    anti_examples=[
        "Fix one named pull request's CI (use fix_github_pr_ci)",
        "Fix one named security alert (use fix_github_security_alert)",
        "Run the sweep every hour (use schedule_github_ci_fleet_fix)",
        "Review a pull request's code",
    ],
    outputs={
        "response_text": "The sweep report to show the user as is",
        "counts": "Targets per outcome: fixed, pr_opened, pushed, failed, skipped, found",
        "outcomes": "Every target with its status, commit, branch and reason",
        "repositories": "What was swept; scope_source says why (explicit, primary, active)",
    },
    surfaces=(ToolSurface.ACTION,),
    side_effect_level=SideEffectLevel.MUTATING,
    accepts_runtime_context=True,
    input_schema={
        "type": "object",
        "properties": {
            "repositories": _REPOSITORIES_PROPERTY,
            "include_security": _SECURITY_PROPERTY,
            "push_to_default_branch": _DEFAULT_BRANCH_PROPERTY,
            "include_drafts": {
                "type": "boolean",
                "description": "Also fix draft pull requests; default false.",
            },
            "dry_run": {
                "type": "boolean",
                "description": "Scan and report what would be fixed, without changing anything.",
            },
            "max_parallel": {
                "type": "integer",
                "minimum": 1,
                "maximum": 8,
                "description": f"Fixes running at once; default {CI_FLEET_MAX_PARALLEL}.",
            },
            "github_token": {"type": "string"},
        },
        "additionalProperties": False,
    },
    is_available=github_tool_available(),
    extract_params=github_tool_params(),
    injected_params=GITHUB_INJECTED_PARAMS,
)
@require_webapp_github
def fix_github_ci_fleet(
    repositories: list[str] | None = None,
    include_security: bool | None = None,
    push_to_default_branch: bool | None = None,
    include_drafts: bool | None = None,
    dry_run: bool | None = None,
    max_parallel: int | None = None,
    github_token: str | None = None,
    context: Any = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Run one fleet sweep and return its report."""
    token = resolve_github_token(github_token)
    if not token:
        return missing_token_envelope(
            "A GitHub connection is required to sweep repositories. Connect GitHub in the "
            "OpenSRE app and end the turn. After they finish, call this tool again.",
            blocked="repositories can't be swept",
        )
    options = SweepOptions(
        repositories=_repositories(repositories),
        include_security=True if include_security is None else bool(include_security),
        include_drafts=bool(include_drafts),
        push_to_default_branch=bool(push_to_default_branch),
        dry_run=bool(dry_run),
        max_parallel=max(1, min(int(max_parallel or CI_FLEET_MAX_PARALLEL), 8)),
    )
    try:
        sweep = run_fleet_sweep(token, options, progress=_progress(context))
    except GitHubApiError as exc:
        message = (
            f"GitHub would not list the repositories to sweep ({exc}). Reconnect or review the "
            f"OpenSRE app's repository access: {github_setup_url()}"
        )
        return tool_unavailable(
            _SOURCE, message, response_text=message, setup_url=github_setup_url()
        )
    return {
        "source": _SOURCE,
        "success": True,
        **sweep.to_dict(),
        "response_text": render_sweep(sweep),
    }


@tool(
    name=SCHEDULE_TOOL_NAME,
    source=_SOURCE,
    display_name="Schedule the hourly CI fix sweep",
    description=(
        "Schedule the hourly CI fix sweep: every hour fix_github_ci_fleet runs on its own, "
        "without a model turn, over the given repositories (or the organization's primary "
        "repositories, else its recently active ones) and posts its report back to the "
        "conversation this request came from, or the local inbox off chat. Hours with "
        "nothing to fix post nothing. One sweep loop per organization: asking again updates it."
    ),
    use_cases=[
        "Every hour, fix broken CI and vulnerabilities across our repositories",
        "Run a CI fix loop hourly on Tracer-Cloud/opensre and Tracer-Cloud/opensre-webapp",
    ],
    anti_examples=[
        "Fix broken CI once, right now (use fix_github_ci_fleet)",
        "Keep one pull request green for ten minutes (use schedule_ci_repair_loop)",
    ],
    outputs={
        "loop_id": "Id to pause, resume or delete the loop with stop_scheduled_loop",
        "cron": "The hourly schedule stored",
        "next_run": "Next run in UTC",
        "reused": "True when the sweep loop already existed and was updated",
    },
    surfaces=(ToolSurface.ACTION,),
    side_effect_level=SideEffectLevel.MUTATING,
    accepts_runtime_context=True,
    input_schema={
        "type": "object",
        "properties": {
            "repositories": _REPOSITORIES_PROPERTY,
            "include_security": _SECURITY_PROPERTY,
            "push_to_default_branch": _DEFAULT_BRANCH_PROPERTY,
        },
        "additionalProperties": False,
    },
    is_available=github_tool_available(),
    extract_params=github_tool_params(),
    injected_params=("github_token", "github_connection_origin", "github_connection_id"),
)
@require_webapp_github
def schedule_github_ci_fleet_fix(
    repositories: list[str] | None = None,
    include_security: bool | None = None,
    push_to_default_branch: bool | None = None,
    context: Any = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Create or update the organization's hourly sweep loop."""
    session = getattr(_action_context(context), "session", None)
    provider, chat_id = conversation_destination(session)
    if provider and (provider not in _CHAT_CHANNELS or not chat_id):
        return {"ok": False, "error": f"The sweep cannot post to {provider} yet; ask from Slack."}
    try:
        scheduled = schedule_ci_fleet_loop(
            repositories=_repositories(repositories),
            include_security=True if include_security is None else bool(include_security),
            push_to_default_branch=bool(push_to_default_branch),
            github_connection_id=str(_kwargs.get("github_connection_id") or ""),
            channel=provider or Provider.INTERACTIVE_SHELL.value,
            chat_id=chat_id,
        )
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}
    request_scheduler_reload()
    task = scheduled.loop.task
    delivers = "this conversation" if chat_id else "the local loop inbox (`/loops messages`)"
    verb = "Updated" if scheduled.reused else "Scheduled"
    return {
        "ok": True,
        "loop_id": task.id,
        "name": task.name,
        "cron": task.cron,
        "timezone": task.timezone,
        "next_run": scheduled.loop.next_run,
        "reused": scheduled.reused,
        "response_text": (
            f"{verb} “{task.name}”: hourly ({task.cron} {task.timezone}), next run "
            f"{scheduled.loop.next_run}. Reports post to {delivers}; hours with nothing to fix "
            f"post nothing. Loop id {task.id}."
        ),
    }


__all__ = [
    "SCHEDULE_TOOL_NAME",
    "SWEEP_TOOL_NAME",
    "fix_github_ci_fleet",
    "schedule_github_ci_fleet_fix",
]
