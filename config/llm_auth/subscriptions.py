"""Subscription metadata, independent of API-key credentials."""

from dataclasses import dataclass
from typing import Literal

from config.constants.claude_code import CLAUDE_SUBSCRIPTION_MODEL_ENV, CLAUDE_SUBSCRIPTION_PROVIDER


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


SUBSCRIPTION_PROVIDERS = (
    SubscriptionProvider(
        CLAUDE_SUBSCRIPTION_PROVIDER,
        "Claude subscription (via Claude Code)",
        CLAUDE_SUBSCRIPTION_MODEL_ENV,
        "sonnet",
        auth_method="cli",
        auth_hint="Sign in with `claude auth login`, then connect your Claude subscription here.",
        usage_hint="Claude Code manages plan limits and renewal; its extra-usage settings still apply.",
        transport="cli",
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
