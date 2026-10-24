"""Subscription client construction through the installed CLI runtime port."""

from collections.abc import Callable
from dataclasses import dataclass

from core.llm.transports.sdk.agent_clients import CLIBackedAgentClient
from core.llm.types import AgentLLMClient, CliLLMClient
from infrastructure.harness_providers.cli_llm import build_cli_client, cli_provider_registration


@dataclass(frozen=True)
class CLISubscriptionClients:
    """Reuse CLI runtime construction without importing integration implementations."""

    provider: str

    def _adapter(self) -> object:
        registration = cli_provider_registration(self.provider)
        if registration is None:
            raise RuntimeError(f"CLI subscription provider '{self.provider}' is not registered.")
        return registration.adapter_factory()

    def agent_client(self, model: str) -> AgentLLMClient:
        """Construct the existing CLI tool-loop bridge."""
        return CLIBackedAgentClient(self._adapter(), model=model)

    def llm_client(
        self,
        model: str,
        *,
        usage_callback: Callable[..., object] | None = None,
    ) -> CliLLMClient:
        """Construct a CLI reasoning client with per-client usage reporting."""
        return build_cli_client(self._adapter(), model=model).with_config(
            usage_callback=usage_callback
        )
