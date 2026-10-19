"""What earlier sweeps did per target, so an hourly sweep never repeats a failed fix."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from filelock import FileLock

from config.constants.ci_fleet import CI_FLEET_DIRECTORY, CI_FLEET_RETRY_HOURS
from config.constants.paths import deployment_home
from integrations.github.tools.ci_fleet.models import (
    LANDED,
    FleetOutcome,
    FleetTarget,
    OutcomeStatus,
    TargetKind,
)

_LEDGER_FILENAME = "ledger.json"
#: Entries older than this are dropped on save, so the file stays small.
_KEEP_SECONDS = 14 * 24 * 3600


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """The last attempt on one target."""

    status: str
    sha: str
    fix_sha: str
    at: float


def ledger_path() -> Path:
    """The organization's sweep ledger; on a hosted gateway it survives rollouts."""
    return deployment_home() / CI_FLEET_DIRECTORY / _LEDGER_FILENAME


class FleetLedger:
    """Read once at the start of a sweep, written once at the end."""

    def __init__(self, path: Path | None = None, *, retry_hours: float = CI_FLEET_RETRY_HOURS):
        self.path = path or ledger_path()
        self.retry_seconds = retry_hours * 3600
        self._entries = self._load()

    def _load(self) -> dict[str, LedgerEntry]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(raw, dict):
            return {}
        entries: dict[str, LedgerEntry] = {}
        for key, value in raw.items():
            if isinstance(value, dict):
                try:
                    entries[str(key)] = LedgerEntry(
                        status=str(value["status"]),
                        sha=str(value.get("sha") or ""),
                        fix_sha=str(value.get("fix_sha") or ""),
                        at=float(value["at"]),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
        return entries

    def skip_reason(self, target: FleetTarget, *, now: float | None = None) -> str:
        """Why an earlier sweep's attempt rules this target out now, or ``""``."""
        entry = self._entries.get(target.key)
        if entry is None:
            return ""
        recent = (now or time.time()) - entry.at < self.retry_seconds
        if not recent:
            return ""
        if target.kind is TargetKind.SECURITY_ALERT:
            if entry.status in {status.value for status in LANDED}:
                return "fixed by an earlier sweep; GitHub has not closed the alert yet"
            if entry.status == OutcomeStatus.FAILED.value:
                return "an earlier sweep could not fix it; it is retried a day later"
            return ""
        same_head = bool(target.sha) and any(
            sha and (sha.startswith(target.sha) or target.sha.startswith(sha))
            for sha in (entry.sha, entry.fix_sha)
        )
        if same_head and entry.status == OutcomeStatus.FAILED.value:
            return "an earlier sweep could not fix this commit; a new push or a day retries it"
        return ""

    def record(self, outcomes: Iterable[FleetOutcome], *, now: float | None = None) -> None:
        """Keep every attempted outcome; skipped and dry-run targets leave no trace."""
        stamp = now or time.time()
        attempted = [
            outcome
            for outcome in outcomes
            if outcome.status not in {OutcomeStatus.SKIPPED, OutcomeStatus.FOUND}
        ]
        if not attempted:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with FileLock(str(self.path) + ".lock", timeout=30):
            entries = self._load()
            for outcome in attempted:
                entries[outcome.target.key] = LedgerEntry(
                    status=outcome.status.value,
                    sha=outcome.target.sha,
                    fix_sha=outcome.commit_sha,
                    at=stamp,
                )
            kept = {
                key: entry for key, entry in entries.items() if stamp - entry.at < _KEEP_SECONDS
            }
            self._write({key: asdict(entry) for key, entry in sorted(kept.items())})
            self._entries = kept

    def _write(self, payload: dict[str, dict[str, object]]) -> None:
        handle, temporary = tempfile.mkstemp(
            dir=self.path.parent, prefix=".ledger-", suffix=".json"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2)
            os.replace(temporary, self.path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise


__all__ = ["FleetLedger", "LedgerEntry", "ledger_path"]
