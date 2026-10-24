"""Claude subscription connection using only the unmodified vendor CLI."""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from config.constants.claude_code import (
    CLAUDE_SUBSCRIPTION_BIN_ENV,
    CLAUDE_SUBSCRIPTION_CONFLICT_ENVS,
    CLAUDE_SUBSCRIPTION_MODELS,
    CLAUDE_SUBSCRIPTION_PROVIDER,
)
from core.llm.subscriptions.types import CLISubscriptionAuth, SubscriptionLoginRequired
from integrations.llm_cli.base import CLIInvocation, CLIProbe
from integrations.llm_cli.binary_resolver import (
    candidate_binary_names,
    default_cli_fallback_paths,
    resolve_cli_binary,
)
from integrations.llm_cli.claude_code import ClaudeCodeAdapter
from integrations.llm_cli.claude_subscription_stream import ClaudeSubscriptionStream
from integrations.llm_cli.subprocess_env import build_cli_subprocess_env


def _binary() -> str:
    binary = resolve_cli_binary(
        explicit_env_key=CLAUDE_SUBSCRIPTION_BIN_ENV,
        binary_names=candidate_binary_names("claude"),
        fallback_paths=lambda: default_cli_fallback_paths("claude"),
    )
    if not binary:
        raise SubscriptionLoginRequired(
            "Install the official Claude Code CLI or set CLAUDE_CODE_SUBSCRIPTION_BIN, then run `claude auth login`."
        )
    return binary


def resolve() -> CLISubscriptionAuth:
    """Ask the official CLI for subscription auth; never inspect or copy its tokens."""
    conflicts = [
        key for key in CLAUDE_SUBSCRIPTION_CONFLICT_ENVS if os.environ.get(key, "").strip()
    ]
    if conflicts:
        raise SubscriptionLoginRequired(
            "Claude subscription billing cannot be confirmed while these overrides are set: "
            + ", ".join(conflicts)
            + ". Remove them for this connection, or select the ordinary claude-code/API connection."
        )
    try:
        proc = subprocess.run(
            [_binary(), "auth", "status"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=8,
            check=False,
            env=build_cli_subprocess_env(None),
        )
        data: Any = json.loads(proc.stdout)
    except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
        raise SubscriptionLoginRequired(
            "Claude Code subscription status could not be confirmed. Run `claude auth status` and retry."
        ) from exc
    if (
        proc.returncode != 0
        or not isinstance(data, dict)
        or data.get("loggedIn") is not True
        or data.get("authMethod") != "claude.ai"
        or data.get("apiProvider", "firstParty") != "firstParty"
        or data.get("apiKeySource")
    ):
        raise SubscriptionLoginRequired(
            "Sign in to your Claude subscription with `claude auth login`; this connection requires CLI-confirmed subscription billing."
        )
    plan = str(data.get("subscriptionType") or "subscription")
    return CLISubscriptionAuth(
        CLAUDE_SUBSCRIPTION_PROVIDER,
        f"Claude Code confirms {plan} authentication; Claude Code manages subscription limits and renewal.",
    )


def models() -> list[tuple[str, str]]:
    """Offer documented CLI aliases; Claude Code checks actual model entitlement."""
    resolve()
    return [(model, model.title()) for model in CLAUDE_SUBSCRIPTION_MODELS]


def verify() -> list[tuple[str, str]]:
    """Verify CLI subscription auth without generating a model request."""
    return models()


def login(
    present: Callable[[str], None],
    *,
    select_model: Callable[[list[tuple[str, str]]], str],
    profile_id: str | None = None,
    new_account: bool = False,
    credential: str | None = None,
) -> str:
    """Connect an existing vendor-owned login; the CLI owns its interactive sign-in."""
    del present
    if profile_id or new_account or credential is not None:
        raise SubscriptionLoginRequired(
            "Claude Code manages accounts and credentials. Use `claude auth logout` / `claude auth login` to change accounts, then reconnect."
        )
    return select_model(models())


def logout() -> str:
    """Delegate explicit logout to the official CLI without reading its credential store."""
    try:
        proc = subprocess.run(
            [_binary(), "auth", "logout"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
            env=build_cli_subprocess_env(None),
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise SubscriptionLoginRequired(
            "Claude Code logout could not finish. Run `claude auth logout` directly."
        ) from exc
    if proc.returncode != 0:
        raise SubscriptionLoginRequired(
            "Claude Code logout failed. Run `claude auth logout` directly."
        )
    return "Signed out through Claude Code. Applications sharing this CLI session will also require sign-in."


class ClaudeCodeSubscriptionAdapter(ClaudeCodeAdapter):
    """A subscription-selected invocation, revalidated before every model request."""

    name = CLAUDE_SUBSCRIPTION_PROVIDER
    binary_env_key = CLAUDE_SUBSCRIPTION_BIN_ENV
    auth_hint = "Run `claude auth login` with your own subscription, then reconnect."
    streams_plain_stdout = False
    stream_decoder_factory = ClaudeSubscriptionStream

    def detect(self) -> CLIProbe:
        try:
            auth = resolve()
            return CLIProbe(True, None, True, _binary(), auth.detail)
        except (RuntimeError, OSError) as exc:
            try:
                binary = _binary()
            except RuntimeError:
                binary = None
            return CLIProbe(bool(binary), None, False, binary, str(exc))

    def _resolve_binary(self) -> str:
        return _binary()

    def build(
        self, *, prompt: str, model: str | None, workspace: str, reasoning_effort: str | None = None
    ) -> CLIInvocation:
        resolve()
        original = super().build(
            prompt=prompt, model=model, workspace=workspace, reasoning_effort=reasoning_effort
        )
        argv = list(original.argv)
        argv[argv.index("--output-format") + 1] = "stream-json"
        argv.extend(["--verbose", "--include-partial-messages"])
        return replace(original, argv=tuple(argv))

    def parse(self, *, stdout: str, stderr: str, returncode: int) -> str:
        if returncode != 0:
            raise RuntimeError(
                self.explain_failure(stdout=stdout, stderr=stderr, returncode=returncode)
            )
        decoder = ClaudeSubscriptionStream()
        decoder.feed(stdout)
        return decoder.finish().content
