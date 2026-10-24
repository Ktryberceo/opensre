"""Supported subscription registrations; provider metadata stays in config."""

from dataclasses import dataclass

from config.llm_auth.subscriptions import SUBSCRIPTION_BY_VALUE
from core.llm.subscriptions import chatgpt, chatgpt_clients, grok, grok_clients
from core.llm.subscriptions.types import SubscriptionAdapter, SubscriptionClientFactory


@dataclass(frozen=True)
class SubscriptionRegistration:
    """Pair a credential lifecycle with its separately implemented inference route."""

    auth: SubscriptionAdapter
    clients: SubscriptionClientFactory


SUBSCRIPTION_ADAPTERS: dict[str, SubscriptionRegistration] = {
    "chatgpt": SubscriptionRegistration(chatgpt, chatgpt_clients),
    "grok": SubscriptionRegistration(grok, grok_clients),
}


def subscription_adapter(provider: str) -> SubscriptionRegistration:
    """Refuse unknown or catalog-only subscriptions instead of selecting another route."""
    if provider not in SUBSCRIPTION_BY_VALUE or provider not in SUBSCRIPTION_ADAPTERS:
        raise ValueError(f"No subscription adapter registered for '{provider}'.")
    return SUBSCRIPTION_ADAPTERS[provider]


def register_subscription_adapter(provider: str, registration: SubscriptionRegistration) -> None:
    """Install a catalogued integration-owned adapter at process composition time."""
    if provider not in SUBSCRIPTION_BY_VALUE:
        raise ValueError(f"Unknown subscription provider: {provider}")
    SUBSCRIPTION_ADAPTERS[provider] = registration
