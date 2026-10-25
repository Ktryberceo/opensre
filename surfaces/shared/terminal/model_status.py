"""Compact model and effective-effort labels without runtime probes during redraw."""

from __future__ import annotations

from config.llm_auth.subscriptions import SUBSCRIPTION_BY_VALUE, SUBSCRIPTION_PROVIDERS
from config.llm_reasoning_effort import (
    ReasoningEffort,
    resolve_provider_reasoning_effort,
)
from infrastructure.safety.terminal_output import strip_terminal_controls

_MODEL_LABELS = {spec.value: dict(spec.status_model_labels) for spec in SUBSCRIPTION_PROVIDERS}


def model_status_label(provider: str, model: str) -> str:
    """Show a compact connection label, preserving exact provider-reported model IDs."""
    subscription = SUBSCRIPTION_BY_VALUE.get(provider)
    label = subscription.status_label if subscription and subscription.status_label else provider
    visible_model = _MODEL_LABELS.get(provider, {}).get(model, model) or "provider default"
    return strip_terminal_controls(f"{label} / {visible_model}")


def effort_status_label(
    provider: str, model: str, choice: ReasoningEffort | str | None
) -> str | None:
    """Never advertise an OpenSRE override that the active provider ignores."""
    return resolve_provider_reasoning_effort(provider, model, choice=choice)
