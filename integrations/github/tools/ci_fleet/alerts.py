"""Open security alerts per repository, read in parallel, as sweep targets."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any

from integrations.github.client import GitHubApiError, GitHubRestClient
from integrations.github.tools.ci_fleet.models import FleetTarget, TargetKind
from integrations.github.tools.ci_health_scan.models import RepoRef

_PAGE_SIZE = 100
_SEVERITY_RANK = {"low": 1, "medium": 2, "moderate": 2, "high": 3, "critical": 4}
#: Alert families the security fixer repairs, and where GitHub lists their open alerts.
_FAMILIES = (
    ("dependabot", "dependabot/alerts"),
    ("code_scanning", "code-scanning/alerts"),
)
_UNAVAILABLE = frozenset({HTTPStatus.FORBIDDEN, HTTPStatus.NOT_FOUND})


@dataclass(frozen=True, slots=True)
class AlertScan:
    """Alerts at or above the threshold, how many fell below it, and families GitHub refused."""

    targets: tuple[FleetTarget, ...] = ()
    below_threshold: int = 0
    notices: tuple[str, ...] = ()
    #: Dependabot alerts with no patched release yet: nothing to upgrade to.
    unpatched: tuple[FleetTarget, ...] = ()


def severity_rank(severity: str) -> int:
    return _SEVERITY_RANK.get(severity.strip().lower(), 0)


def _severity(alert_type: str, alert: dict[str, Any]) -> str:
    if alert_type == "dependabot":
        advisory = alert.get("security_advisory") or {}
        vulnerability = alert.get("security_vulnerability") or {}
        return str(advisory.get("severity") or vulnerability.get("severity") or "")
    rule = alert.get("rule") or {}
    # Quality findings have no security severity; the sweep fixes security ones only.
    return str(rule.get("security_severity_level") or "")


def _title(alert_type: str, alert: dict[str, Any]) -> str:
    if alert_type == "dependabot":
        package = ((alert.get("dependency") or {}).get("package") or {}).get("name") or ""
        summary = (alert.get("security_advisory") or {}).get("summary") or ""
        return f"{package}: {summary}".strip(": ")
    rule = alert.get("rule") or {}
    return str(rule.get("description") or rule.get("id") or "")


def _has_patch(alert_type: str, alert: dict[str, Any]) -> bool:
    if alert_type != "dependabot":
        return True
    vulnerability = alert.get("security_vulnerability") or {}
    return bool((vulnerability.get("first_patched_version") or {}).get("identifier"))


def _repo_alerts(client: GitHubRestClient, repo: RepoRef, min_rank: int) -> AlertScan:
    targets: list[FleetTarget] = []
    unpatched: list[FleetTarget] = []
    notices: list[str] = []
    below = 0
    for alert_type, path in _FAMILIES:
        try:
            rows = client.request(
                "GET",
                f"repos/{repo.full_name}/{path}",
                params={"state": "open", "per_page": _PAGE_SIZE},
            )
        except GitHubApiError as exc:
            if getattr(exc, "status_code", None) not in _UNAVAILABLE:
                notices.append(f"{repo.full_name} {alert_type} alerts could not be read ({exc}).")
            continue
        for alert in rows if isinstance(rows, list) else []:
            if not isinstance(alert, dict) or not alert.get("number"):
                continue
            severity = _severity(alert_type, alert)
            if severity_rank(severity) < min_rank:
                below += 1
                continue
            target = FleetTarget(
                repo=repo.full_name,
                kind=TargetKind.SECURITY_ALERT,
                ref=alert_type,
                number=int(alert["number"]),
                title=_title(alert_type, alert),
                url=str(alert.get("html_url") or ""),
                alert_type=alert_type,
                severity=severity.lower(),
            )
            (targets if _has_patch(alert_type, alert) else unpatched).append(target)
    return AlertScan(tuple(targets), below, tuple(notices), tuple(unpatched))


def scan_security_alerts(
    client: GitHubRestClient,
    repos: Sequence[RepoRef],
    *,
    min_severity: str,
    concurrency: int = 8,
) -> AlertScan:
    """Every repository's open Dependabot and code-scanning security alerts, worst first.

    A repository without the feature (alerts off, no Advanced Security) is
    silently skipped: there is nothing there to fix.
    """
    min_rank = severity_rank(min_severity) or severity_rank("high")
    if not repos:
        return AlertScan()
    targets: list[FleetTarget] = []
    unpatched: list[FleetTarget] = []
    notices: list[str] = []
    below = 0
    with ThreadPoolExecutor(max_workers=max(1, min(concurrency, len(repos)))) as pool:
        for scan in pool.map(lambda repo: _repo_alerts(client, repo, min_rank), repos):
            targets.extend(scan.targets)
            unpatched.extend(scan.unpatched)
            notices.extend(scan.notices)
            below += scan.below_threshold
    targets.sort(key=lambda target: -severity_rank(target.severity))
    return AlertScan(tuple(targets), below, tuple(notices), tuple(unpatched))


__all__ = ["AlertScan", "scan_security_alerts", "severity_rank"]
