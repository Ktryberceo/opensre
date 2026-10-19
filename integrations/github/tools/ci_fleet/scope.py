"""Which repositories a fleet sweep covers, without anyone naming them each time."""

from __future__ import annotations

from dataclasses import dataclass

from config.constants.ci_fleet import CI_FLEET_ACTIVE_DAYS, CI_FLEET_MAX_REPOSITORIES
from core.domain.repositories import load_repository_preferences
from core.domain.repositories.preferences import normalize_repository
from integrations.github.client import GitHubRestClient
from integrations.github.tools.ci_health_scan.graphql import RateLimitTally
from integrations.github.tools.ci_health_scan.models import RepoRef
from integrations.github.tools.ci_health_scan.scope import resolve_scope

SCOPE_EXPLICIT = "explicit"
SCOPE_PRIMARY = "primary"
SCOPE_ACTIVE = "active"


@dataclass(frozen=True, slots=True)
class FleetScope:
    """The repositories to sweep, where that list came from, and what was left out."""

    repos: tuple[RepoRef, ...]
    source: str
    notices: tuple[str, ...] = ()


def _ref(full_name: str) -> RepoRef:
    owner, _, name = full_name.partition("/")
    return RepoRef(owner=owner, name=name)


def resolve_fleet_scope(
    client: GitHubRestClient,
    *,
    repositories: list[str] | tuple[str, ...] | None = None,
    active_days: int = CI_FLEET_ACTIVE_DAYS,
    max_repositories: int = CI_FLEET_MAX_REPOSITORIES,
    tally: RateLimitTally | None = None,
) -> FleetScope:
    """Named repositories, else the organization's primary ones, else its active ones.

    Never-change repositories are always dropped. Active repositories are those
    the connection reaches, pushed to in the last ``active_days``, that are not
    forks: a fork's CI belongs to whoever owns the original. Raises
    ``GitHubApiError`` only when the active listing cannot be read at all.
    """
    preferences = load_repository_preferences()
    never = {name.casefold() for name in preferences.excluded}
    notices: list[str] = []
    explicit = [found for raw in repositories or () if (found := normalize_repository(raw))]
    if explicit:
        source, names = SCOPE_EXPLICIT, explicit
    elif preferences.primary:
        source, names = SCOPE_PRIMARY, list(preferences.primary)
    else:
        source = SCOPE_ACTIVE
        resolution = resolve_scope(client, owners=None, since_days=active_days, tally=tally)
        notices.extend(resolution.coverage_notices)
        newest_first = sorted(resolution.repos, key=lambda ref: ref.pushed_at, reverse=True)
        names = [ref.full_name for ref in newest_first if not ref.is_fork]
    kept: list[str] = []
    for name in names:
        if name.casefold() in never:
            notices.append(f"{name} is a never-change repository; it was not swept.")
        elif name.casefold() not in {seen.casefold() for seen in kept}:
            kept.append(name)
    if len(kept) > max_repositories:
        notices.append(
            f"Swept the first {max_repositories} of {len(kept)} repositories; save the important "
            "ones as primary repositories to choose which."
        )
        kept = kept[:max_repositories]
    return FleetScope(
        repos=tuple(_ref(name) for name in kept), source=source, notices=tuple(notices)
    )


__all__ = [
    "SCOPE_ACTIVE",
    "SCOPE_EXPLICIT",
    "SCOPE_PRIMARY",
    "FleetScope",
    "resolve_fleet_scope",
]
