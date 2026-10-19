"""Commit statuses that report a hosting service's policy, not a broken build.

Vercel, for one, posts ``Deployment was blocked`` on a commit whose author is
not a member of its team, such as a fix the OpenSRE Agent committed. No code
change can turn that green, so CI scans and fix verification must not count it.
"""

from __future__ import annotations

_POLICY_BLOCKED_MARKERS = (
    "deployment was blocked",
    "must have access to the project",
)


def is_policy_blocked_status(description: object) -> bool:
    """Whether a failed commit status's description says a policy, not the build, stopped it."""
    text = " ".join(str(description or "").split()).lower()
    return any(marker in text for marker in _POLICY_BLOCKED_MARKERS)


__all__ = ["is_policy_blocked_status"]
