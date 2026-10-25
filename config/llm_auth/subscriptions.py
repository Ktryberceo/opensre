"""Subscription metadata, independent of API-key credentials."""

from dataclasses import dataclass
from typing import Literal

from config.constants.claude_code import (
    CLAUDE_SUBSCRIPTION_DEFAULT_EFFORT,
    CLAUDE_SUBSCRIPTION_MODEL_ENV,
    CLAUDE_SUBSCRIPTION_MODELS,
    CLAUDE_SUBSCRIPTION_PROVIDER,
)


@dataclass(frozen=True)
class SubscriptionProvider:
    """A subscription connection; authentication method is independent of billing."""

    value: str
    label: str
    model_env: str
    default_model: str
    auth_method: str = "oauth"
    auth_hint: str = "Sign in using your browser; provider plan limits apply."
    usage_hint: str = "Provider subscription limits apply."
    transport: Literal["litellm", "cli"] = "litellm"
    status_label: str = ""
    status_model_labels: tuple[tuple[str, str], ...] = ()
    default_reasoning_effort: str | None = None
    """When set, apply the configured effort across all CLI roles, including tool calls."""


SUBSCRIPTION_PROVIDERS = (
    SubscriptionProvider(
        CLAUDE_SUBSCRIPTION_PROVIDER,
        "Claude subscription (via Claude Code)",
        CLAUDE_SUBSCRIPTION_MODEL_ENV,
        "sonnet",
        auth_method="cli",
        auth_hint="Reuse your Claude Code subscription session, or sign in through its official CLI.",
        usage_hint="Claude Code manages plan limits and renewal; its extra-usage settings still apply.",
        transport="cli",
        status_label="Claude",
        default_reasoning_effort=CLAUDE_SUBSCRIPTION_DEFAULT_EFFORT,
        status_model_labels=tuple((model, model.title()) for model in CLAUDE_SUBSCRIPTION_MODELS),
    ),
    SubscriptionProvider(
        "chatgpt",
        "ChatGPT subscription",
        "CHATGPT_MODEL",
        "gpt-6.1-sol",
        usage_hint="Manage limits in ChatGPT Settings → Usage.",
    ),
    SubscriptionProvider(
        "grok",
        "Grok subscription",
        "GROK_MODEL",
        "grok-4.3",
        auth_hint="Requires an xAI-approved OAuth client ID; sign in on any browser.",
        usage_hint="Eligible xAI subscription and approved OAuth client required; live validation pending.",
    ),
)
SUBSCRIPTION_BY_VALUE = {provider.value: provider for provider in SUBSCRIPTION_PROVIDERS}
