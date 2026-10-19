"""A fleet sweep as the short report a person reads in Slack or the shell inbox."""

from __future__ import annotations

from infrastructure.scheduling.scheduler.outcomes import WorkStatus
from infrastructure.scheduling.scheduler.types import TaskReport
from integrations.github.tools.ci_fleet.models import (
    FleetOutcome,
    FleetSweep,
    OutcomeStatus,
    TargetKind,
)
from integrations.github.tools.ci_fleet.scope import SCOPE_ACTIVE, SCOPE_EXPLICIT, SCOPE_PRIMARY
from integrations.github.tools.ci_fleet.sweep import SKIP_DRAFT, SKIP_FORK

_MAX_LINES_PER_SECTION = 15
_SCOPE_WORDS = {
    SCOPE_EXPLICIT: "the repositories named for this loop",
    SCOPE_PRIMARY: "the organization's primary repositories",
    SCOPE_ACTIVE: "repositories pushed to in the last two weeks",
}


def _duration(seconds: float) -> str:
    whole = int(round(seconds))
    if whole < 60:
        return f"{whole}s"
    minutes, rest = divmod(whole, 60)
    return f"{minutes}m {rest:02d}s"


def _what(outcome: FleetOutcome) -> str:
    target = outcome.target
    link = f"[{target.label}]({target.url})" if target.url else target.label
    if target.kind is TargetKind.SECURITY_ALERT:
        detail = f"{target.severity} · {target.title}" if target.title else target.severity
        return f"{link} ({detail})" if detail else link
    if target.checks:
        return f"{link} ({', '.join(target.checks[:3])}{'…' if len(target.checks) > 3 else ''})"
    return link


def _lines(outcomes: list[FleetOutcome]) -> list[str]:
    lines = [f"• {_what(outcome)} — {outcome.summary}" for outcome in outcomes]
    if len(lines) > _MAX_LINES_PER_SECTION:
        hidden = len(lines) - _MAX_LINES_PER_SECTION
        lines = [*lines[:_MAX_LINES_PER_SECTION], f"• …and {hidden} more"]
    return lines


def _headline(sweep: FleetSweep) -> str:
    counts = sweep.counts
    if sweep.dry_run:
        found = counts.get(OutcomeStatus.FOUND.value, 0)
        return f"CI fix sweep (dry run): {found} to fix"
    parts = []
    landed = sum(
        counts.get(status.value, 0)
        for status in (OutcomeStatus.FIXED, OutcomeStatus.PR_OPENED, OutcomeStatus.PUSHED)
    )
    if landed:
        parts.append(f"{landed} fixed")
    if counts.get(OutcomeStatus.FAILED.value):
        parts.append(f"{counts[OutcomeStatus.FAILED.value]} need a person")
    if not parts:
        parts.append("nothing to fix")
    return "CI fix sweep: " + ", ".join(parts)


def render_sweep(sweep: FleetSweep) -> str:
    """Markdown: headline, what landed, what failed and why, what was skipped, scan figures."""
    sections: list[str] = [f"**{_headline(sweep)}** · {_duration(sweep.elapsed_seconds)}"]
    groups = (
        ("Would fix", [OutcomeStatus.FOUND]),
        ("Fixed", [OutcomeStatus.FIXED, OutcomeStatus.PR_OPENED]),
        ("Pushed, checks still running", [OutcomeStatus.PUSHED]),
        ("Could not fix — needs a person", [OutcomeStatus.FAILED]),
        ("Already green", [OutcomeStatus.NOTHING_TO_FIX]),
        ("Skipped", [OutcomeStatus.SKIPPED]),
    )
    counted = {SKIP_FORK, SKIP_DRAFT}
    for title, statuses in groups:
        chosen = [o for o in sweep.with_status(*statuses) if o.error_kind not in counted]
        if chosen:
            sections.append("\n".join([f"*{title}*", *_lines(chosen)]))
    scope = _SCOPE_WORDS.get(sweep.scope_source, sweep.scope_source)
    scanned = (
        f"Scanned {len(sweep.repositories)} repositories ({scope}) in "
        f"{_duration(sweep.scan_seconds)}: {sweep.open_prs} open pull requests."
    )
    footer = [scanned]
    forks = [o for o in sweep.outcomes if o.error_kind == SKIP_FORK]
    drafts = [o for o in sweep.outcomes if o.error_kind == SKIP_DRAFT]
    if forks or drafts:
        footer.append(
            f"Left alone: {len(forks)} failing pull requests from forks (OpenSRE cannot push "
            f"to them) and {len(drafts)} drafts."
        )
    footer.extend(sweep.notices)
    footer.extend(f"{error['repo']} could not be read: {error['error']}" for error in sweep.errors)
    sections.append("\n".join(f"_{line}_" for line in footer))
    return "\n\n".join(sections)


def sweep_task_report(sweep: FleetSweep) -> TaskReport:
    """The loop tick's result: quiet when nothing was red, failed when every attempt failed."""
    acted = [
        outcome
        for outcome in sweep.outcomes
        if outcome.status not in {OutcomeStatus.SKIPPED, OutcomeStatus.NOTHING_TO_FIX}
    ]
    headline = _headline(sweep)
    if not acted:
        return TaskReport("", summary=headline, work_status=WorkStatus.NOOP)
    failed = sweep.with_status(OutcomeStatus.FAILED)
    status = WorkStatus.FAILED if failed and len(failed) == len(acted) else WorkStatus.SUCCEEDED
    return TaskReport(render_sweep(sweep), summary=headline, work_status=status)


__all__ = ["render_sweep", "sweep_task_report"]
