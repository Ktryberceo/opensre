"""Collect provider-reported model facts within one logical turn, including its workers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelRuntimeSnapshot:
    """Reported execution details, owned by the configured connection that produced them."""

    provider: str
    requested_model: str
    model: str | None = None
    reasoning_effort: str | None = None


_snapshots: ContextVar[list[ModelRuntimeSnapshot] | None] = ContextVar(
    "model_runtime_snapshots", default=None
)


@contextmanager
def collect_model_runtime() -> Iterator[list[ModelRuntimeSnapshot]]:
    """Capture this turn's facts without changing clients shared by concurrent sessions."""
    snapshots: list[ModelRuntimeSnapshot] = []
    token = _snapshots.set(snapshots)
    try:
        yield snapshots
    finally:
        _snapshots.reset(token)


def report_model_runtime(snapshot: ModelRuntimeSnapshot) -> None:
    """Record a completed request only for the currently bound collector, if any."""
    snapshots = _snapshots.get()
    if snapshots is not None:
        snapshots.append(snapshot)
