"""Claude subscription billing and native CLI stream contract."""

import os
from pathlib import Path

import pytest

from integrations.llm_cli.claude_code import ClaudeCodeAdapter


def test_existing_claude_code_connection_keeps_explicit_api_auth(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-api-key")
    monkeypatch.setattr(ClaudeCodeAdapter, "_resolve_binary", lambda _self: "/synthetic/claude")
    invocation = ClaudeCodeAdapter().build(prompt="hello", model=None, workspace=str(tmp_path))
    assert invocation.argv == ("/synthetic/claude", "-p", "--output-format", "text")
    assert invocation.env and invocation.env["ANTHROPIC_API_KEY"] == "synthetic-api-key"


def test_stream_retains_reported_model_and_effort_without_exposing_envelopes():
    import json

    from integrations.llm_cli.claude_subscription_stream import ClaudeSubscriptionStream

    decoder = ClaudeSubscriptionStream()
    assert (
        decoder.feed(
            json.dumps(
                {"type": "system", "subtype": "init", "model": "claude-opus-4-7", "effort": "high"}
            )
            + "\n"
        )
        == []
    )
    decoder.feed(
        json.dumps({"type": "result", "subtype": "success", "is_error": False, "result": "hello"})
        + "\n"
    )
    response = decoder.finish()
    assert response.content == "hello"
    assert response.model == "claude-opus-4-7"
    assert response.reasoning_effort == "high"


@pytest.mark.parametrize(
    "models, expected",
    [
        ({"claude-opus-4-7": {}}, "claude-opus-4-7"),
        ({"claude-opus-4-7": {}, "claude-haiku-4-5": {}}, None),
        ({}, None),
    ],
)
def test_stream_does_not_guess_primary_model_or_effort(models, expected):
    import json

    from integrations.llm_cli.claude_subscription_stream import ClaudeSubscriptionStream

    decoder = ClaudeSubscriptionStream()
    decoder.feed(
        json.dumps({"type": "system", "subtype": "init", "model": [], "effort": {"high": True}})
        + "\n"
    )
    decoder.feed(
        json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": "hello",
                "modelUsage": models,
            }
        )
        + "\n"
    )
    response = decoder.finish()
    assert response.model == expected
    assert response.reasoning_effort is None


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    import json
    import subprocess
    import sys
    from dataclasses import replace

    from config.constants.claude_code import CLAUDE_SUBSCRIPTION_CONFLICT_ENVS
    from integrations.llm_cli import claude_subscription

    for key in CLAUDE_SUBSCRIPTION_CONFLICT_ENVS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OPENSRE_LLM_AUTH_METADATA_PATH", str(tmp_path / "auth-metadata.json"))
    monkeypatch.setenv("OPENSRE_PROJECT_ENV_PATH", str(tmp_path / ".env"))
    monkeypatch.setenv("OPENSRE_DISABLE_KEYRING", "1")
    monkeypatch.setattr("surfaces.shared.llm_setup.env_sync.PROJECT_ENV_PATH", tmp_path / ".env")
    monkeypatch.setattr(claude_subscription, "_binary", lambda: sys.executable)
    state = {
        "auth": {
            "loggedIn": True,
            "authMethod": "claude.ai",
            "apiProvider": "firstParty",
            "subscriptionType": "max",
        },
        "commands": [],
        "invocations": [],
        "script": tmp_path / "synthetic_cli.py",
        "pid": tmp_path / "pid",
        "release": tmp_path / "release",
    }

    def auth_command(argv, **_kwargs):
        state["commands"].append(argv)
        if argv[1:] == ["auth", "login", "--claudeai"]:
            assert not _kwargs.get("capture_output")
            assert all(_kwargs.get(stream) is None for stream in ("stdin", "stdout", "stderr"))
            state["auth"].update(loggedIn=True, authMethod="claude.ai")
            return subprocess.CompletedProcess(argv, 0, "", "")
        if argv[1:] == ["auth", "logout"]:
            state["auth"]["loggedIn"] = False
            return subprocess.CompletedProcess(argv, 0, "", "")
        assert argv[1:] == ["auth", "status"]
        return subprocess.CompletedProcess(
            argv, 0 if state["auth"]["loggedIn"] else 1, json.dumps(state["auth"]), ""
        )

    monkeypatch.setattr(claude_subscription.subprocess, "run", auth_command)

    class SyntheticClaude(claude_subscription.ClaudeCodeSubscriptionAdapter):
        def build(self, **kwargs):
            invocation = super().build(**kwargs)
            state["invocations"].append(invocation)
            return replace(invocation, argv=(sys.executable, str(state["script"])))

    state["adapter"] = SyntheticClaude
    state["script"].write_text(
        "import sys, json\n"
        "sys.stdin.read()\n"
        "print(json.dumps({'type':'stream_event','event':{'delta':{'type':'text_delta','text':'hello'}}}), flush=True)\n"
        "print(json.dumps({'type':'result','subtype':'success','is_error':False,'result':'hello', 'usage':{'input_tokens':11,'output_tokens':7,'cache_read_input_tokens':3}}), flush=True)\n"
    )
    return state


@pytest.mark.parametrize("signed_in", [False, True])
def test_login_picker_authorizes_only_when_signed_out(wire, monkeypatch, tmp_path, signed_in):
    from click.testing import CliRunner

    from surfaces.cli.app import cli
    from surfaces.shared.llm_setup.auth_profiles import resolve_auth_profile

    if not signed_in:
        wire["auth"].update(loggedIn=False, authMethod="none")
    monkeypatch.setattr(
        "surfaces.cli.commands.auth.prompt_auth_provider",
        lambda: resolve_auth_profile("claude-code-subscription"),
    )
    result = CliRunner().invoke(cli, ["auth", "login", "--model", "sonnet"])

    assert result.exit_code == 0, result.output
    commands = [command[1:] for command in wire["commands"]]
    assert commands == (
        [["auth", "status"]]
        if signed_in
        else [["auth", "status"], ["auth", "login", "--claudeai"], ["auth", "status"]]
    )
    env = (tmp_path / ".env").read_text()
    assert "LLM_PROVIDER=claude-code-subscription" in env
    assert "CLAUDE_CODE_SUBSCRIPTION_MODEL=sonnet" in env


@pytest.mark.parametrize("outcome", ["failed", "timeout", "cancelled", "signed_out", "api"])
def test_login_never_saves_provider_without_confirmed_subscription(
    wire, monkeypatch, tmp_path, outcome
):
    import subprocess

    from click.testing import CliRunner

    from integrations.llm_cli import claude_subscription
    from surfaces.cli.app import cli

    wire["auth"].update(loggedIn=False, authMethod="none")
    auth_command = claude_subscription.subprocess.run
    login_commands = []

    def run(argv, **kwargs):
        if argv[1:] != ["auth", "login", "--claudeai"]:
            return auth_command(argv, **kwargs)
        login_commands.append(argv)
        if outcome == "timeout":
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        if outcome == "cancelled":
            raise KeyboardInterrupt
        if outcome == "api":
            wire["auth"].update(loggedIn=True, authMethod="api_key", apiKeySource="apiKeyHelper")
        return subprocess.CompletedProcess(argv, 1 if outcome == "failed" else 0, "", "")

    monkeypatch.setattr(claude_subscription.subprocess, "run", run)
    result = CliRunner().invoke(
        cli, ["auth", "login", "claude-code-subscription", "--model", "sonnet"]
    )

    assert len(login_commands) == 1
    assert result.exit_code != 0
    assert not (tmp_path / ".env").exists()


@pytest.mark.parametrize("status", ["override", "api", "unknown", "timeout"])
def test_login_does_not_replace_ambiguous_or_conflicting_auth(wire, monkeypatch, status):
    import subprocess

    from core.llm.subscriptions.types import SubscriptionLoginRequired
    from integrations.llm_cli import claude_subscription

    if status == "override":
        monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-conflict")
    elif status == "api":
        wire["auth"].update(authMethod="api_key", apiKeySource="apiKeyHelper")
    elif status == "unknown":
        wire["auth"].pop("authMethod")
    else:

        def timeout(argv, **_kwargs):
            raise subprocess.TimeoutExpired(argv, 8)

        monkeypatch.setattr(claude_subscription.subprocess, "run", timeout)

    with pytest.raises(SubscriptionLoginRequired):
        claude_subscription.login(lambda _url: None, select_model=lambda models: models[0][0])
    assert all(command[1:] == ["auth", "status"] for command in wire["commands"])


def test_subscription_setup_and_runtime_use_cli_without_storing_tokens(wire, monkeypatch, tmp_path):
    from click.testing import CliRunner

    from config.llm_auth.credentials import status
    from core.llm.factory import LLMRole, get_llm, reset_llm_clients, resolve_llm_route
    from core.llm.subscriptions import registry
    from core.llm.subscriptions.cli_clients import CLISubscriptionClients
    from infrastructure.harness_providers.cli_llm import CliLlmAdapters
    from integrations.llm_cli.registry import CLIProviderRegistration
    from integrations.llm_cli.runner import CLIBackedLLMClient
    from integrations.llm_cli.text import flatten_messages_to_prompt
    from surfaces.cli.app import cli
    from surfaces.shared.llm_setup.auth_service import logout_provider, verify_provider

    registration = CLIProviderRegistration(wire["adapter"], "CLAUDE_CODE_SUBSCRIPTION_MODEL")

    def registration_for(provider):
        assert provider == "claude-code-subscription"
        return registration

    def build(adapter, **kwargs):
        return CLIBackedLLMClient(
            adapter, **{key: value for key, value in kwargs.items() if value is not None}
        )

    CliLlmAdapters(registration_for, build, flatten_messages_to_prompt).install()
    # The process normally registers the real adapter. Swap only its factory for an isolated executable.
    monkeypatch.setitem(
        registry.SUBSCRIPTION_ADAPTERS,
        "claude-code-subscription",
        registry.SubscriptionRegistration(
            registry.SUBSCRIPTION_ADAPTERS["claude-code-subscription"].auth,
            CLISubscriptionClients("claude-code-subscription"),
        ),
    )
    result = CliRunner().invoke(
        cli, ["auth", "login", "claude-code-subscription", "--model", "sonnet"]
    )
    assert result.exit_code == 0, result.output
    env = (tmp_path / ".env").read_text()
    assert "LLM_PROVIDER=claude-code-subscription" in env and "OPENSRE_LLM_CONNECTION=local" in env
    assert "CLAUDE_CODE_SUBSCRIPTION_MODEL=sonnet" in env
    assert not (tmp_path / "subscriptions").exists()
    assert status("claude-code-subscription").source == "subscription"
    reset_llm_clients()
    try:
        assert resolve_llm_route().use_litellm is False
        assert list(get_llm(LLMRole.REASONING).invoke_stream("hello")) == ["hello"]
        agent = get_llm(LLMRole.AGENT)
        response = agent.invoke([{"role": "user", "content": "hello"}])
        assert response.content == "hello" and (response.input_tokens, response.output_tokens) == (
            11,
            7,
        )
        assert response.cache_read_tokens == 3
        assert all(get_llm(role) is not None for role in LLMRole)
    finally:
        reset_llm_clients()
    assert verify_provider("claude-code-subscription").verified
    assert "Signed out" in logout_provider("claude-code-subscription")
    assert not status("claude-code-subscription").configured
    assert all(
        command[1:] in (["auth", "status"], ["auth", "logout"]) for command in wire["commands"]
    )
    invocation = wire["invocations"][0]
    assert invocation.argv[1:] == (
        "-p",
        "--output-format",
        "stream-json",
        "--model",
        "sonnet",
        "--verbose",
        "--include-partial-messages",
        "--effort",
        "high",
    )
    assert "--bare" not in invocation.argv


def test_cli_runtime_reports_model_only_to_current_turn(wire):
    import json

    from core.llm.shared.model_runtime import collect_model_runtime
    from integrations.llm_cli.runner import CLIBackedLLMClient

    init = {"type": "system", "subtype": "init", "model": "claude-opus-4-7"}
    original = wire["script"].read_text()
    wire["script"].write_text(f"print({json.dumps(init)!r}, flush=True)\n" + original)
    with collect_model_runtime() as snapshots:
        client = CLIBackedLLMClient(wire["adapter"](), model="opus")
        assert client.invoke("hello").model == "claude-opus-4-7"
    assert len(snapshots) == 1
    assert (snapshots[0].provider, snapshots[0].requested_model, snapshots[0].model) == (
        "claude-code-subscription",
        "opus",
        "claude-opus-4-7",
    )
    assert snapshots[0].reasoning_effort is None
    client.invoke("outside the turn")
    assert len(snapshots) == 1


@pytest.mark.parametrize(
    "choice, env, expected",
    [
        (None, None, "high"),
        (None, "low", "low"),
        ("medium", "low", "medium"),
        ("max", None, "xhigh"),
    ],
)
def test_subscription_effort_matches_footer_and_cli_for_every_role(
    wire, monkeypatch, choice, env, expected
):
    from config.llm_reasoning_effort import apply_reasoning_effort, parse_reasoning_effort
    from core.llm.shared.model_runtime import ModelRuntimeSnapshot
    from core.llm.types import ModelType
    from integrations.llm_cli.runner import CLIBackedLLMClient
    from surfaces.interactive_shell.session import Session
    from surfaces.interactive_shell.ui.auto_status import auto_status_ansi

    monkeypatch.delenv("OPENSRE_REASONING_EFFORT", raising=False)
    if env:
        monkeypatch.setenv("OPENSRE_REASONING_EFFORT", env)
    monkeypatch.setenv("CLAUDE_CODE_EFFORT_LEVEL", "max")
    session = Session()
    session.terminal.provider_model = ("claude-code-subscription", "opus")
    session.reasoning_effort = parse_reasoning_effort(choice)
    # Last response metadata must not override the preference for the next call.
    session.terminal.model_runtime = ModelRuntimeSnapshot(
        "claude-code-subscription", "opus", "claude-opus-4-7", "low"
    )
    assert f"claude-opus-4-7 ({expected})" in auto_status_ansi(session, max_width=160)
    with apply_reasoning_effort(session.reasoning_effort):
        for role in (None, *ModelType):
            client = CLIBackedLLMClient(wire["adapter"](), model="opus", model_type=role)
            assert client.invoke("hello").content == "hello"
    for invocation in wire["invocations"]:
        index = invocation.argv.index("--effort")
        assert invocation.argv[index + 1] == expected
        assert invocation.env["CLAUDE_CODE_EFFORT_LEVEL"] == expected
    assert os.environ["CLAUDE_CODE_EFFORT_LEVEL"] == "max"


@pytest.mark.parametrize("mutation", ["key", "token", "cloud", "helper", "unknown", "logged_out"])
def test_subscription_cannot_silently_change_billing_after_cached_probe(
    wire, monkeypatch, mutation
):
    from core.llm.subscriptions.types import SubscriptionLoginRequired
    from integrations.llm_cli.runner import CLIBackedLLMClient

    client = CLIBackedLLMClient(wire["adapter"]())
    assert client.invoke("hello").content == "hello"
    wire["invocations"].clear()
    if mutation == "key":
        monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-key-must-not-use")
    elif mutation == "token":
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "synthetic-token-must-not-read")
    elif mutation == "cloud":
        monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    elif mutation == "helper":
        wire["auth"]["apiKeySource"] = "apiKeyHelper"
    elif mutation == "unknown":
        wire["auth"].pop("authMethod")
    else:
        wire["auth"]["loggedIn"] = False
    with pytest.raises(SubscriptionLoginRequired):
        client.invoke("must not execute")
    assert wire["invocations"] == []


@pytest.mark.parametrize("failure", ["missing", "error_result", "quota", "invalid"])
@pytest.mark.parametrize("delivery", ["stream", "buffered", "agent"])
def test_partial_stream_never_becomes_success_or_retries(wire, failure, delivery):
    import json

    from core.llm.types import LLMStreamInterrupted
    from integrations.llm_cli.runner import CLIBackedLLMClient

    partial = '{"tool_calls":[{"name":"lookup","input":'
    events = [{"type": "stream_event", "event": {"delta": {"type": "text_delta", "text": partial}}}]
    if failure == "error_result":
        events.append(
            {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "result": "not success",
            }
        )
    elif failure == "quota":
        events.append({"type": "rate_limit_event", "rate_limit_info": {"status": "rejected"}})
    payload = "\n".join(json.dumps(event) for event in events) + "\n"
    if failure == "invalid":
        payload += "not-json\n"
    wire["script"].write_text(
        "import sys\nsys.stdin.read()\n" + f"print({payload!r},end='',flush=True)\n"
    )
    client = CLIBackedLLMClient(wire["adapter"]())
    if delivery == "stream":
        stream = client.invoke_stream("hello")
        assert next(stream) == partial
        with pytest.raises(LLMStreamInterrupted) as error:
            list(stream)
        assert error.value.partial_text == partial
    else:
        if delivery == "agent":
            from core.llm.transports.sdk.agent_clients import CLIBackedAgentClient

            client = CLIBackedAgentClient(wire["adapter"]())
        with pytest.raises(RuntimeError) as error:
            client.invoke("hello")
        assert not isinstance(error.value, LLMStreamInterrupted)
        assert not getattr(error.value, "partial_text", None)
    assert len(wire["invocations"]) == 1


def test_live_text_and_cancellation_reap_cli_without_waiting_for_another_event(wire):
    import time
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    import psutil

    from core.llm.types import LLMRequestCancelled
    from integrations.llm_cli.runner import CLIBackedLLMClient

    wire["script"].write_text(
        "import sys, time, os\nfrom pathlib import Path\nsys.stdin.read()\n"
        f"Path({str(wire['pid'])!r}).write_text(str(os.getpid()))\n"
        'print(\'{"type":"stream_event","event":{"delta":{"type":"text_delta","text":"hello"}}}\',flush=True)\n'
        f"while not Path({str(wire['release'])!r}).exists(): time.sleep(0.02)\n"
    )
    client = CLIBackedLLMClient(wire["adapter"]())
    stream = client.invoke_stream("hello")
    assert next(stream) == "hello"
    pid = int(wire["pid"].read_text())
    assert psutil.pid_exists(pid)
    stream.close()
    assert not psutil.pid_exists(pid)
    wire["pid"].unlink()
    stopped = Event()
    with ThreadPoolExecutor(max_workers=1) as workers:
        pending = workers.submit(client.invoke_cancellable, "hello", should_stop=stopped.is_set)
        deadline = time.monotonic() + 5
        while not wire["pid"].exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert wire["pid"].exists()
        pid = int(wire["pid"].read_text())
        stopped.set()
        with pytest.raises(LLMRequestCancelled):
            pending.result(timeout=3)
    assert not psutil.pid_exists(pid)


def test_tools_use_completed_model_json_and_replay_results(wire):
    import json

    from core.llm.transports.sdk.agent_clients import CLIBackedAgentClient

    payload = json.dumps(
        {"tool_calls": [{"id": "call-1", "name": "lookup", "input": {"service": "api"}}]}
    )
    event = json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": payload,
            "usage": {"input_tokens": 12, "output_tokens": 8},
        }
    )
    wire["script"].write_text("import sys\nsys.stdin.read()\n" + f"print({event!r},flush=True)\n")
    client = CLIBackedAgentClient(wire["adapter"](), model="sonnet")
    result = client.invoke(
        [{"role": "user", "content": "check service"}],
        tools=[{"name": "lookup", "input_schema": {"type": "object"}}],
    )
    assert result.tool_calls[0].input == {"service": "api"}
    replay = client.build_tool_result_message(result.tool_calls, ["healthy"])
    client.invoke([client.build_assistant_message(result.content, result.tool_calls), replay])
    assert "healthy" in wire["invocations"][-1].stdin
    assert (result.input_tokens, result.output_tokens) == (12, 8)


def test_probe_timeout_preserves_unknown_auth_and_never_invokes(wire, monkeypatch):
    import subprocess

    from core.llm.subscriptions.types import SubscriptionLoginRequired
    from integrations.llm_cli import claude_subscription
    from integrations.llm_cli.runner import CLIBackedLLMClient

    def timeout(argv, **_kwargs):
        raise subprocess.TimeoutExpired(argv, 8)

    monkeypatch.setattr(claude_subscription.subprocess, "run", timeout)
    adapter = wire["adapter"]()
    probe = adapter.detect()
    assert probe.installed and probe.logged_in is None
    with pytest.raises(SubscriptionLoginRequired, match="could not be confirmed"):
        CLIBackedLLMClient(adapter).invoke("hello")
    assert wire["invocations"] == []


@pytest.mark.skipif(not hasattr(os, "killpg"), reason="POSIX process-group regression")
def test_stream_close_stops_sigterm_resistant_descendant_after_leader_exits(wire):
    import contextlib
    import json
    import signal
    import sys
    import time

    import psutil

    from integrations.llm_cli.runner import CLIBackedLLMClient

    child_pid_file = wire["pid"].with_name("child-pid")
    child_script = (
        "import os,signal,time\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(child_pid_file)!r}).write_text(str(os.getpid()))\n"
        "while True: time.sleep(0.02)\n"
    )
    delta = json.dumps(
        {"type": "stream_event", "event": {"delta": {"type": "text_delta", "text": "ready"}}}
    )
    wire["script"].write_text(
        "import subprocess,sys,time\nfrom pathlib import Path\nsys.stdin.read()\n"
        f"subprocess.Popen([{sys.executable!r}, '-c', {child_script!r}])\n"
        f"while not Path({str(child_pid_file)!r}).exists(): time.sleep(0.01)\n"
        f"print({delta!r},flush=True)\n"
        "while True: time.sleep(0.02)\n"
    )
    stream = CLIBackedLLMClient(wire["adapter"]()).invoke_stream("hello")
    assert next(stream) == "ready"
    child_pid = int(child_pid_file.read_text())
    try:
        stream.close()
        deadline = time.monotonic() + 2
        while True:
            try:
                state = psutil.Process(child_pid).status()
            except psutil.NoSuchProcess:
                state = None
            if state in (None, psutil.STATUS_ZOMBIE) or time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        assert state in (None, psutil.STATUS_ZOMBIE), f"descendant still executing: {state}"
    finally:
        with contextlib.suppress(ProcessLookupError):
            os.kill(child_pid, signal.SIGKILL)
