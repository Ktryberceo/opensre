"""Prompt redraws use a route snapshot refreshed outside the rendering callback."""

from __future__ import annotations

import asyncio
import threading
from io import StringIO

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import DummyInput
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from config.account import AccountLLMRoute
from surfaces.interactive_shell.runtime import shell_turn_execution, turn_host
from surfaces.interactive_shell.runtime.core.prompt_builder import PromptBuilder
from surfaces.interactive_shell.runtime.core.state import ReplState, SpinnerState
from surfaces.interactive_shell.runtime.turn_host import AgentTurnResources, run_agent_turn
from surfaces.interactive_shell.session import Session


@pytest.mark.asyncio
async def test_concurrent_turns_keep_reported_models_in_their_own_footer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.llm.shared.model_runtime import ModelRuntimeSnapshot, report_model_runtime
    from surfaces.interactive_shell.runtime.agent_presentation import AgentEvent
    from surfaces.interactive_shell.ui.auto_status import auto_status_ansi

    barrier = threading.Barrier(2)

    def invoke(text: str, *_args: object, **_kwargs: object) -> None:
        barrier.wait(timeout=3)
        report_model_runtime(
            ModelRuntimeSnapshot(
                provider="claude-code-subscription",
                requested_model="opus",
                model=text,
                reasoning_effort="high" if text.endswith("b") else None,
            )
        )

    monkeypatch.setattr(shell_turn_execution, "execute_shell_turn", invoke)
    monkeypatch.setattr(
        turn_host, "detect_provider_model", lambda: ("claude-code-subscription", "opus")
    )
    runtimes = [
        AgentTurnResources(
            session=Session(),
            state=ReplState(),
            spinner=SpinnerState(),
            invalidate_prompt=lambda: None,
            console=Console(file=StringIO()),
        )
        for _ in range(2)
    ]

    async def emit(_event: AgentEvent) -> None:
        """Keep the process-wide terminal renderer out of concurrent session workers."""

    async def run(runtime: AgentTurnResources, text: str) -> None:
        cancel = threading.Event()
        await turn_host._run_agent_turn_loop(
            runtime=runtime,
            text=text,
            output=turn_host._streaming_console(runtime, cancel),
            confirm=lambda _prompt: "n",
            emit=emit,
            dispatch_cancel=cancel,
        )

    await asyncio.wait_for(
        asyncio.gather(
            run(runtimes[0], "claude-opus-version-a"),
            run(runtimes[1], "claude-opus-version-b"),
        ),
        timeout=5,
    )
    first = auto_status_ansi(runtimes[0].session, max_width=160)
    assert "Claude / claude-opus-version-a" in first
    assert "CLI default" not in first
    assert "(high)" in first
    assert "Claude / claude-opus-version-b (high)" in auto_status_ansi(
        runtimes[1].session, max_width=160
    )
    runtimes[0].session.terminal.provider_model = ("claude-code-subscription", "sonnet")
    assert "Claude / Sonnet" in auto_status_ansi(runtimes[0].session, max_width=160)
    assert "version-a" not in auto_status_ansi(runtimes[0].session, max_width=160)


@pytest.mark.asyncio
async def test_redraws_do_not_resolve_credentials_and_completed_turn_refreshes_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current: list[AccountLLMRoute | None] = [
        AccountLLMRoute(base_url="https://example.test", model="hosted-model")
    ]
    reads: list[int] = []

    def route() -> AccountLLMRoute | None:
        reads.append(threading.get_ident())
        return current[0]

    def switch(*_args: object, **_kwargs: object) -> None:
        current[0] = None
        monkeypatch.setenv("LLM_PROVIDER", "chatgpt")
        monkeypatch.setenv("CHATGPT_MODEL", "local-model")

    monkeypatch.setattr("config.account.account_llm_route", route)
    monkeypatch.setattr(shell_turn_execution, "execute_shell_turn", switch)
    session = Session()
    state, spinner = ReplState(), SpinnerState()
    with create_app_session(input=DummyInput(), output=DummyOutput()):
        builder = PromptBuilder(session, state, spinner)
        builder.setup()
        try:
            for _ in range(5):
                assert "openai / hosted-model" in builder._status_line()
            assert len(reads) == 1
            runtime = AgentTurnResources(
                session=session,
                state=state,
                spinner=spinner,
                invalidate_prompt=lambda: None,
                console=Console(file=StringIO()),
            )
            await run_agent_turn(runtime, "/model set chatgpt local-model")
            for _ in range(5):
                assert "chatgpt / local-model" in builder._status_line()
            assert len(reads) == 2
            assert reads[-1] != threading.get_ident()
        finally:
            await builder.close()


@pytest.mark.asyncio
async def test_cancelling_blocked_display_refresh_stops_spinner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    release = threading.Event()
    loop = asyncio.get_running_loop()

    def blocked_route() -> tuple[str, str]:
        loop.call_soon_threadsafe(started.set)
        release.wait()
        return ("chatgpt", "local-model")

    monkeypatch.setattr(turn_host, "detect_provider_model", blocked_route)
    monkeypatch.setattr(shell_turn_execution, "execute_shell_turn", lambda *_args, **_kwargs: None)
    spinner = SpinnerState()
    runtime = AgentTurnResources(
        session=Session(),
        state=ReplState(),
        spinner=spinner,
        invalidate_prompt=lambda: None,
        console=Console(file=StringIO()),
    )
    task = asyncio.create_task(run_agent_turn(runtime, "check service health"))
    try:
        await asyncio.wait_for(started.wait(), timeout=2)
        assert spinner.streaming
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not spinner.streaming
    finally:
        release.set()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
