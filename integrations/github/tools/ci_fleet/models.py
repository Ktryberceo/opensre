"""Typed shapes for one fleet sweep: what it found, what it did, and why."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class TargetKind(StrEnum):
    """What one sweep target is."""

    PULL_REQUEST = "pull_request"
    DEFAULT_BRANCH = "default_branch"
    SECURITY_ALERT = "security_alert"


class OutcomeStatus(StrEnum):
    """What the sweep did with one target."""

    FIXED = "fixed"
    """A fix landed on the branch and its checks passed."""

    PR_OPENED = "pr_opened"
    """A branch rule refused the direct push, so the fix went up as a pull request."""

    PUSHED = "pushed"
    """A fix landed but its checks did not finish before verification stopped."""

    FAILED = "failed"
    NOTHING_TO_FIX = "nothing_to_fix"
    SKIPPED = "skipped"
    FOUND = "found"
    """Dry run: the target would be fixed."""


#: Outcomes that changed a branch or opened a pull request.
LANDED = frozenset({OutcomeStatus.FIXED, OutcomeStatus.PR_OPENED, OutcomeStatus.PUSHED})


@dataclass(frozen=True, slots=True)
class FleetTarget:
    """One red pull request, red default branch, or open security alert."""

    repo: str
    kind: TargetKind
    ref: str
    sha: str = ""
    number: int | None = None
    title: str = ""
    url: str = ""
    checks: tuple[str, ...] = ()
    alert_type: str = ""
    severity: str = ""

    @property
    def owner(self) -> str:
        return self.repo.partition("/")[0]

    @property
    def name(self) -> str:
        return self.repo.partition("/")[2]

    @property
    def key(self) -> str:
        """Stable identity across sweeps: the same PR, branch, or alert."""
        if self.kind is TargetKind.SECURITY_ALERT:
            return f"{self.repo}|{self.alert_type}|{self.number}"
        if self.kind is TargetKind.PULL_REQUEST:
            return f"{self.repo}|pr|{self.number}"
        return f"{self.repo}|branch|{self.ref}"

    @property
    def label(self) -> str:
        if self.kind is TargetKind.PULL_REQUEST:
            return f"{self.repo}#{self.number}"
        if self.kind is TargetKind.SECURITY_ALERT:
            return f"{self.repo} {self.alert_type.replace('_', ' ')} alert #{self.number}"
        return f"{self.repo}@{self.ref}"

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "repo": self.repo,
            "kind": self.kind.value,
            "ref": self.ref,
            "label": self.label,
        }
        optional = {
            "sha": self.sha,
            "number": self.number,
            "title": self.title,
            "url": self.url,
            "failing_checks": list(self.checks),
            "alert_type": self.alert_type,
            "severity": self.severity,
        }
        payload.update({key: value for key, value in optional.items() if value})
        return payload


@dataclass(frozen=True, slots=True)
class FleetOutcome:
    """What happened to one target in this sweep."""

    target: FleetTarget
    status: OutcomeStatus
    summary: str
    commit_sha: str = ""
    pushed_to: str = ""
    pull_request_url: str = ""
    changed_files: tuple[str, ...] = ()
    checks_state: str = ""
    error_kind: str = ""
    attempts: int = 0
    seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            **self.target.to_dict(),
            "status": self.status.value,
            "summary": self.summary,
        }
        optional = {
            "commit_sha": self.commit_sha,
            "pushed_to": self.pushed_to,
            "pull_request_url": self.pull_request_url,
            "changed_files": list(self.changed_files),
            "checks_state": self.checks_state,
            "error_kind": self.error_kind,
            "attempts": self.attempts,
            "seconds": round(self.seconds, 1) if self.seconds else 0,
        }
        payload.update({key: value for key, value in optional.items() if value})
        return payload


@dataclass(frozen=True, slots=True)
class FleetSweep:
    """One whole sweep: scope, scan figures, and every target's outcome."""

    repositories: tuple[str, ...]
    scope_source: str
    elapsed_seconds: float
    scan_seconds: float
    open_prs: int = 0
    outcomes: tuple[FleetOutcome, ...] = ()
    notices: tuple[str, ...] = ()
    errors: tuple[dict[str, str], ...] = field(default=())
    dry_run: bool = False

    def with_status(self, *statuses: OutcomeStatus) -> list[FleetOutcome]:
        return [outcome for outcome in self.outcomes if outcome.status in statuses]

    @property
    def counts(self) -> dict[str, int]:
        counts = {status.value: 0 for status in OutcomeStatus}
        for outcome in self.outcomes:
            counts[outcome.status.value] += 1
        return {key: value for key, value in counts.items() if value}

    def to_dict(self) -> dict[str, Any]:
        return {
            "repositories": list(self.repositories),
            "scope_source": self.scope_source,
            "dry_run": self.dry_run,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "scan_seconds": round(self.scan_seconds, 1),
            "open_prs": self.open_prs,
            "counts": self.counts,
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
            "notices": list(self.notices),
            "errors": list(self.errors),
        }


__all__ = [
    "LANDED",
    "FleetOutcome",
    "FleetSweep",
    "FleetTarget",
    "OutcomeStatus",
    "TargetKind",
]
