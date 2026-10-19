"""One fleet sweep: scan every repository at once, fix what is red in parallel lanes, remember.

Pull requests are fixed on their own head branch. A red default branch and its
security alerts share one lane per repository (they all land on that branch),
through pull requests unless the sweep may push onto the default branch itself.
Every lane runs alongside the others. Each fix is the existing CI or
security fixer; this module only decides what to fix, in what order, and
records what happened.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from typing import Any

from config.constants.ci_fleet import (
    CI_FLEET_MAX_FIXES,
    CI_FLEET_MAX_PARALLEL,
    CI_FLEET_MIN_ALERT_SEVERITY,
    CI_FLEET_SWEEP_SECONDS,
)
from core.tool import report_run_error
from integrations.git import opensre_authored_commits
from integrations.github.client import GitHubRestClient
from integrations.github.tools.ci_fix.errors import (
    ERR_CHECKS_FAILED,
    ERR_CHECKS_TIMEOUT,
    ERR_EXECUTION,
    ERR_NO_CHANGES,
    ERR_NO_FAILING_CHECKS,
    ERR_TIMEOUT,
)
from integrations.github.tools.ci_fix.ledger import record_ci_fix_outcome
from integrations.github.tools.ci_fix.runner import run_ci_fix
from integrations.github.tools.ci_fix.ship import BranchDelivery
from integrations.github.tools.ci_fleet.alerts import AlertScan, scan_security_alerts
from integrations.github.tools.ci_fleet.ledger import FleetLedger
from integrations.github.tools.ci_fleet.models import (
    FleetOutcome,
    FleetSweep,
    FleetTarget,
    OutcomeStatus,
    TargetKind,
)
from integrations.github.tools.ci_fleet.scope import resolve_fleet_scope
from integrations.github.tools.ci_health_scan.graphql import RateLimitTally
from integrations.github.tools.ci_health_scan.models import FailingHead, ScanReport
from integrations.github.tools.ci_health_scan.scan import scan_repositories
from integrations.github.tools.security_fix.errors import ERR_NO_ELIGIBLE_ALERT
from integrations.github.tools.security_fix.runner import run_security_fix

logger = logging.getLogger(__name__)

#: A CI fix that pushed but stayed red, or produced nothing, gets one more try in the same sweep.
_RETRYABLE = frozenset({ERR_CHECKS_FAILED, ERR_NO_CHANGES, ERR_TIMEOUT, ERR_EXECUTION})
_CI_ATTEMPTS = 2
_NOTHING_TO_FIX_KINDS = frozenset({ERR_NO_FAILING_CHECKS, ERR_NO_ELIGIBLE_ALERT})

Fixer = Callable[[FleetTarget, str], FleetOutcome]
Progress = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class SweepOptions:
    """What one sweep covers and how hard it works."""

    repositories: tuple[str, ...] = ()
    include_security: bool = True
    include_drafts: bool = False
    #: Land default-branch and security fixes on the default branch itself, not as pull requests.
    push_to_default_branch: bool = False
    dry_run: bool = False
    max_parallel: int = CI_FLEET_MAX_PARALLEL
    max_fixes: int = CI_FLEET_MAX_FIXES
    min_alert_severity: str = CI_FLEET_MIN_ALERT_SEVERITY
    budget_seconds: float = CI_FLEET_SWEEP_SECONDS


#: Skip kinds the report counts instead of listing: they are not the sweep's to fix.
SKIP_FORK = "fork"
SKIP_DRAFT = "draft"


def _skipped(target: FleetTarget, reason: str, kind: str = "") -> FleetOutcome:
    return FleetOutcome(
        target=target, status=OutcomeStatus.SKIPPED, summary=reason, error_kind=kind
    )


def _ci_target(head: FailingHead, kind: TargetKind) -> FleetTarget:
    return FleetTarget(
        repo=head.repo,
        kind=kind,
        ref=head.ref,
        sha=head.sha,
        number=head.number,
        title=head.title,
        url=head.url,
        checks=tuple(check.name for check in head.checks),
    )


def plan_targets(
    report: ScanReport,
    alerts: AlertScan,
    ledger: FleetLedger,
    *,
    include_drafts: bool = False,
) -> tuple[list[FleetTarget], list[FleetOutcome]]:
    """Targets to fix now, red default branches first, and the ones left alone with why."""
    targets: list[FleetTarget] = []
    skipped: list[FleetOutcome] = []
    candidates = [
        _ci_target(head, TargetKind.DEFAULT_BRANCH) for head in report.failing_default_branches
    ]
    for head in report.failing_prs:
        target = _ci_target(head, TargetKind.PULL_REQUEST)
        if head.is_fork:
            skipped.append(
                _skipped(
                    target,
                    f"opened from a fork; OpenSRE only pushes to branches in {head.repo}",
                    SKIP_FORK,
                )
            )
        elif head.is_draft and not include_drafts:
            skipped.append(
                _skipped(target, "draft pull request; it is fixed once marked ready", SKIP_DRAFT)
            )
        else:
            candidates.append(target)
    candidates.extend(alerts.targets)
    skipped.extend(
        _skipped(target, "no patched release exists yet; there is nothing to upgrade to")
        for target in alerts.unpatched
    )
    for target in candidates:
        reason = ledger.skip_reason(target)
        if reason:
            skipped.append(_skipped(target, reason))
        else:
            targets.append(target)
    return targets, skipped


def build_lanes(targets: Sequence[FleetTarget]) -> list[list[FleetTarget]]:
    """Group targets that push to the same branch; each lane runs its targets in order.

    A repository's red default branch is fixed before its security alerts, which
    land on that same branch. Lanes holding a red default branch run first.
    """
    default_lanes: dict[str, list[FleetTarget]] = {}
    pr_lanes: list[list[FleetTarget]] = []
    for target in targets:
        if target.kind is TargetKind.PULL_REQUEST:
            pr_lanes.append([target])
        else:
            default_lanes.setdefault(target.repo.casefold(), []).append(target)
    for lane in default_lanes.values():
        lane.sort(key=lambda target: target.kind is not TargetKind.DEFAULT_BRANCH)
    red_first = sorted(
        default_lanes.values(), key=lambda lane: lane[0].kind is not TargetKind.DEFAULT_BRANCH
    )
    branch_lanes = [lane for lane in red_first if lane[0].kind is TargetKind.DEFAULT_BRANCH]
    alert_lanes = [lane for lane in red_first if lane[0].kind is not TargetKind.DEFAULT_BRANCH]
    return [*branch_lanes, *pr_lanes, *alert_lanes]


def _cap_lanes(
    lanes: list[list[FleetTarget]], max_fixes: int
) -> tuple[list[list[FleetTarget]], list[FleetOutcome]]:
    kept: list[list[FleetTarget]] = []
    over: list[FleetOutcome] = []
    budget = max(0, max_fixes)
    for lane in lanes:
        room = lane[:budget]
        budget -= len(room)
        if room:
            kept.append(room)
        over.extend(
            _skipped(
                target,
                f"this sweep's limit of {max_fixes} fixes was reached; the next sweep continues",
            )
            for target in lane[len(room) :]
        )
    return kept, over


def _ci_outcome(target: FleetTarget, output: dict[str, Any], attempt: int) -> FleetOutcome:
    kind = str(output.get("error_kind") or "")
    commit = str(output.get("fix_head_sha") or "")
    branch = str(output.get("branch_name") or "")
    pull_request = str(output.get("pull_request_url") or "")
    state = str(output.get("checks_state") or "")
    base = FleetOutcome(
        target=target,
        status=OutcomeStatus.FAILED,
        summary="",
        commit_sha=commit,
        pushed_to=branch,
        pull_request_url=pull_request,
        changed_files=tuple(str(path) for path in output.get("changed_files") or ()),
        checks_state=state,
        error_kind=kind,
        attempts=attempt,
    )
    if kind in _NOTHING_TO_FIX_KINDS:
        return replace(
            base,
            status=OutcomeStatus.NOTHING_TO_FIX,
            summary="no failing checks on the current head",
        )
    if output.get("success") and state == "passed":
        if pull_request:
            reason = str(output.get("pr_reason") or "")
            opened = f"{reason}, so opened" if reason else "opened"
            return replace(
                base,
                status=OutcomeStatus.PR_OPENED,
                summary=f"{opened} {pull_request}; its checks passed",
            )
        return replace(
            base,
            status=OutcomeStatus.FIXED,
            summary=f"pushed {commit[:7]} to {branch}; all checks passed",
        )
    if commit and kind == ERR_CHECKS_TIMEOUT:
        return replace(
            base,
            status=OutcomeStatus.PUSHED,
            summary=f"pushed {commit[:7]} to {branch}; checks were still running when verification stopped",
        )
    message = str(output.get("error") or output.get("response_text") or "the fix did not complete")
    return replace(base, summary=" ".join(message.split()))


def _security_outcome(target: FleetTarget, output: dict[str, Any]) -> FleetOutcome:
    kind = str(output.get("error_kind") or "")
    commit = str(output.get("commit_sha") or "")
    pull_request = str(output.get("pr_url") or "")
    branch = str(output.get("branch_name") or "")
    base = FleetOutcome(
        target=target,
        status=OutcomeStatus.FAILED,
        summary="",
        commit_sha=commit,
        pushed_to=branch,
        pull_request_url=pull_request,
        changed_files=tuple(str(path) for path in output.get("changed_files") or ()),
        error_kind=kind,
        attempts=1,
    )
    if kind in _NOTHING_TO_FIX_KINDS:
        return replace(base, status=OutcomeStatus.NOTHING_TO_FIX, summary="no longer open")
    if output.get("success") and commit:
        return replace(base, status=OutcomeStatus.FIXED, summary=f"pushed {commit[:7]} to {branch}")
    if output.get("success") and pull_request:
        return replace(
            base, status=OutcomeStatus.PR_OPENED, summary=str(output.get("response_text") or "")
        )
    message = str(output.get("error") or output.get("response_text") or "the fix did not complete")
    return replace(base, summary=" ".join(message.split()))


def fix_target(
    target: FleetTarget, token: str, *, push_to_default_branch: bool = False
) -> FleetOutcome:
    """Fix one target with the matching fixer and say what happened; never raises."""
    started = time.monotonic()
    try:
        outcome = _fix(target, token, push_to_default_branch=push_to_default_branch)
    except Exception as exc:
        report_run_error(
            exc,
            tool_name="fix_github_ci_fleet",
            source="github",
            component=__name__,
            method="fix_target",
            logger=logger,
            extras={"target": target.label},
        )
        outcome = FleetOutcome(
            target=target,
            status=OutcomeStatus.FAILED,
            summary=f"OpenSRE stopped on an internal error ({type(exc).__name__})",
            error_kind="internal_error",
        )
    return replace(outcome, seconds=time.monotonic() - started)


def _fix(target: FleetTarget, token: str, *, push_to_default_branch: bool) -> FleetOutcome:
    if target.kind is TargetKind.SECURITY_ALERT:
        output = run_security_fix(
            owner=target.owner,
            repo=target.name,
            alert_type=target.alert_type,
            alert_number=target.number,
            github_token=token,
            open_pr=True,
            push_to_default_branch=push_to_default_branch,
        )
        return _security_outcome(target, output)
    delivery = BranchDelivery.DIRECT if push_to_default_branch else BranchDelivery.PULL_REQUEST
    outcome: FleetOutcome | None = None
    for attempt in range(1, _CI_ATTEMPTS + 1):
        if target.kind is TargetKind.PULL_REQUEST:
            output = run_ci_fix(
                owner=target.owner, repo=target.name, pr_number=target.number, github_token=token
            )
        else:
            output = run_ci_fix(
                owner=target.owner,
                repo=target.name,
                branch=target.ref,
                github_token=token,
                branch_delivery=delivery,
            )
        record_ci_fix_outcome(output)
        outcome = _ci_outcome(target, output, attempt)
        if outcome.status is not OutcomeStatus.FAILED or outcome.error_kind not in _RETRYABLE:
            return outcome
    assert outcome is not None
    return outcome


def _run_lane(
    lane: list[FleetTarget], token: str, deadline: float, fixer: Fixer
) -> list[FleetOutcome]:
    outcomes: list[FleetOutcome] = []
    # Thread-local: every commit this lane makes, here or by its coding agent, is the agent's.
    with opensre_authored_commits():
        for target in lane:
            if time.monotonic() >= deadline:
                outcomes.append(
                    _skipped(target, "this sweep's time budget ran out; the next sweep continues")
                )
                continue
            outcomes.append(fixer(target, token))
    return outcomes


def run_fleet_sweep(
    token: str,
    options: SweepOptions | None = None,
    *,
    ledger: FleetLedger | None = None,
    fixer: Fixer | None = None,
    progress: Progress | None = None,
) -> FleetSweep:
    """Scan the fleet, fix every eligible red target, and return what happened.

    Raises ``GitHubApiError`` only when the repositories to sweep cannot be
    listed at all; per-repository read failures are reported in ``errors``.
    """
    opts = options or SweepOptions()
    started = time.monotonic()
    deadline = started + opts.budget_seconds
    say = progress or (lambda _line: None)
    client = GitHubRestClient(token)
    tally = RateLimitTally()
    scope = resolve_fleet_scope(client, repositories=opts.repositories, tally=tally)
    repositories = tuple(ref.full_name for ref in scope.repos)
    if not scope.repos:
        return FleetSweep(
            repositories=(),
            scope_source=scope.source,
            elapsed_seconds=time.monotonic() - started,
            scan_seconds=time.monotonic() - started,
            notices=(*scope.notices, "No repositories to sweep."),
            dry_run=opts.dry_run,
        )
    say(f"Scanning {len(repositories)} repositories ({scope.source})…")
    with ThreadPoolExecutor(max_workers=2) as pool:
        ci_scan = pool.submit(
            scan_repositories,
            client,
            scope.repos,
            owners=tuple(dict.fromkeys(ref.owner for ref in scope.repos)),
            include_all_branches=False,
            concurrency=8,
            started=started,
            tally=tally,
        )
        alert_scan = (
            pool.submit(
                scan_security_alerts, client, scope.repos, min_severity=opts.min_alert_severity
            )
            if opts.include_security
            else None
        )
        report = ci_scan.result()
        alerts = alert_scan.result() if alert_scan is not None else AlertScan()
    scan_seconds = time.monotonic() - started
    active_ledger = ledger or FleetLedger()
    targets, skipped = plan_targets(
        report, alerts, active_ledger, include_drafts=opts.include_drafts
    )
    notices = [*scope.notices, *report.coverage_notices, *alerts.notices]
    if alerts.below_threshold:
        notices.append(
            f"{alerts.below_threshold} open security alerts below {opts.min_alert_severity} "
            "severity were left for a person."
        )
    say(f"Found {len(targets)} targets to fix in {scan_seconds:.1f}s.")
    if opts.dry_run:
        found = [
            FleetOutcome(target=target, status=OutcomeStatus.FOUND, summary="would be fixed")
            for target in targets
        ]
        return FleetSweep(
            repositories=repositories,
            scope_source=scope.source,
            elapsed_seconds=time.monotonic() - started,
            scan_seconds=scan_seconds,
            open_prs=report.open_prs,
            outcomes=(*found, *skipped),
            notices=tuple(notices),
            errors=tuple(report.errors),
            dry_run=True,
        )
    lanes, over = _cap_lanes(build_lanes(targets), opts.max_fixes)
    fix = fixer or partial(fix_target, push_to_default_branch=opts.push_to_default_branch)
    outcomes: list[FleetOutcome] = []
    if lanes:
        workers = max(1, min(opts.max_parallel, len(lanes)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ci-fleet") as pool:
            futures = [pool.submit(_run_lane, lane, token, deadline, fix) for lane in lanes]
            for future in futures:
                for outcome in future.result():
                    say(f"{outcome.target.label}: {outcome.status.value} — {outcome.summary}")
                    outcomes.append(outcome)
    active_ledger.record(outcomes)
    return FleetSweep(
        repositories=repositories,
        scope_source=scope.source,
        elapsed_seconds=time.monotonic() - started,
        scan_seconds=scan_seconds,
        open_prs=report.open_prs,
        outcomes=(*outcomes, *over, *skipped),
        notices=tuple(notices),
        errors=tuple(report.errors),
    )


__all__ = [
    "SKIP_DRAFT",
    "SKIP_FORK",
    "Fixer",
    "SweepOptions",
    "build_lanes",
    "fix_target",
    "plan_targets",
    "run_fleet_sweep",
]
