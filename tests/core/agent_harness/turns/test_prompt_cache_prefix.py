"""Each turn's request extends the previous turn's byte for byte, so the prompt cache covers it.

Anthropic caches the request prefix — tools, then system, then messages — up to
a breakpoint, and any byte that changes inside it re-prices everything after.
These tests run real turns through the harness, the default tool catalog and
the Anthropic agent client; only the HTTP call is replaced, by a scripted SDK
client that records each request body.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable
from io import StringIO
from typing import Any

import pytest
from anthropic.types import Message
from pydantic import BaseModel
from rich.console import Console

from config.constants import OPENSRE_MEMORY_DISABLED_ENV
from core.agent_harness import SessionCore
from core.agent_harness.prompts.action.assemble import SESSION_CONTEXT_HEADER
from core.agent_harness.session.lifecycle import SessionManager
from core.agent_harness.session.persistence.contracts import RestoreContextKey
from core.agent_harness.session.persistence.memory import InMemorySessionStore
from core.agent_harness.tools.tool_provider import DefaultToolProvider
from core.agent_harness.turns.headless_build import InMemoryHeadlessBuild
from core.llm.transports.sdk.agent_clients import AnthropicAgentClient
from tests.utils.anthropic_streams import FakeMessageStream

_REMEMBER = {
    "name": "payments-deploy-day",
    "type": "infrastructure",
    "description": "Payments deploys every Tuesday",
    "content": "The payments service deploys every Tuesday.",
    "source": "user",
    "evidence": "payments deploys every Tuesday",
}


def _sdk_model(value: Any) -> Any:
    """Encode an SDK object the way the SDK encodes a request body."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_unset=True, by_alias=True)
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _without_markers(node: Any) -> Any:
    """Drop what the API ignores when it matches a prefix: cache markers, string vs text block."""
    if isinstance(node, dict):
        stripped = {key: _without_markers(value) for key, value in node.items()}
        stripped.pop("cache_control", None)
        if isinstance(stripped.get("content"), str) and "role" in stripped:
            stripped["content"] = [{"type": "text", "text": stripped["content"]}]
        return stripped
    if isinstance(node, list):
        return [_without_markers(item) for item in node]
    return node


class _ScriptedAnthropic:
    """Stands in for the Anthropic SDK client: records each request, returns the next reply.

    The agent client streams; ``stream`` records and replies the same way ``create`` does.
    """

    def __init__(self, replies: Iterable[list[dict[str, Any]]]) -> None:
        self.messages = self
        self.requests: list[dict[str, Any]] = []
        self._replies = iter(replies)
        self._ids = itertools.count(1)

    def create(self, **kwargs: Any) -> Message:
        wire = json.dumps(kwargs, ensure_ascii=False, default=_sdk_model)
        self.requests.append(_without_markers(json.loads(wire)))
        content = next(self._replies)
        return Message.model_validate(
            {
                "id": f"msg_{next(self._ids)}",
                "type": "message",
                "role": "assistant",
                "model": kwargs["model"],
                "content": content,
                "stop_reason": "tool_use"
                if any(block["type"] == "tool_use" for block in content)
                else "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 1, "output_tokens": 1},
            }
        )

    def stream(self, **kwargs: Any) -> FakeMessageStream:
        return FakeMessageStream(self.create(**kwargs))


def _tool_use(call_id: str, name: str, tool_input: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": call_id, "name": name, "input": tool_input},
    ]


def _text(text: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": text}]


def _dump(part: Any) -> str:
    return json.dumps(part, ensure_ascii=False)


def _assert_extends(previous: dict[str, Any], current: dict[str, Any]) -> None:
    """``current`` repeats ``previous`` byte for byte before its first new message."""
    assert _dump(current["tools"]) == _dump(previous["tools"])
    assert _dump(current["system"]) == _dump(previous["system"])
    shared = len(previous["messages"])
    for index, message in enumerate(previous["messages"]):
        assert _dump(current["messages"][index]) == _dump(message), f"messages[{index}] changed"
    assert len(current["messages"]) > shared


def _user_text(message: dict[str, Any]) -> str:
    return "".join(block.get("text", "") for block in message["content"])


def _rebuilt_from_log(session: SessionCore) -> SessionCore:
    """The session the gateway resolves for the next message: a new object restored from its log."""
    data = {
        RestoreContextKey.CLI_AGENT_MESSAGES: [list(pair) for pair in session.cli_agent_messages],
        RestoreContextKey.TURN_EVIDENCE: [
            json.loads(json.dumps(record.to_json())) for record in session.turn_evidence
        ],
    }
    rebuilt = SessionCore(session_id=session.session_id, store=session.store)
    return SessionManager(store=session.store).restore_context(rebuilt, data)


@pytest.fixture(autouse=True)
def _memory_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(OPENSRE_MEMORY_DISABLED_ENV, raising=False)


@pytest.mark.parametrize("rebuild", [False, True], ids=["live_session", "rebuilt_from_log"])
def test_the_next_turn_repeats_the_previous_request_byte_for_byte(rebuild: bool) -> None:
    """Pins the four ways the prefix used to change between turns.

    Turn one reveals a hidden tool with tool search and writes a memory, which
    changes both the tool list and the session context. The next turn must still
    resend everything the last request of turn one sent — the user message with
    its context, the tool calls and results, the tools — byte for byte, and
    carry the changed context in its own new message instead. The gateway
    rebuilds the session from its log for every message, so the same must hold
    for a session restored between turns.
    """
    sdk = _ScriptedAnthropic(
        [
            _tool_use("toolu_01", "tool_search", {"names": ["memory_remember"]}),
            _tool_use("toolu_02", "memory_remember", _REMEMBER),
            _text("Saved: payments deploys on Tuesdays."),
            _text("Payments deploys on Tuesdays."),
            _text("Anytime."),
        ]
    )
    session = SessionCore(store=InMemorySessionStore())
    agent = InMemoryHeadlessBuild(session=session).agent(
        tools=DefaultToolProvider(session, Console(file=StringIO(), force_terminal=False)),
        llm_factory=lambda: AnthropicAgentClient(model="claude-test", client=sdk),
    )

    for message in (
        "Remember that payments deploys every Tuesday.",
        "When does payments deploy?",
        "Thanks!",
    ):
        agent.dispatch(message)
        if rebuild:
            session = _rebuilt_from_log(session)
            agent.bind_session(session)

    first, _found, last_of_turn_one, turn_two, turn_three = sdk.requests
    _assert_extends(last_of_turn_one, turn_two)
    _assert_extends(turn_two, turn_three)
    # Session context rides in the conversation, never in the cached system prompt.
    assert "LONG-TERM MEMORY" not in _dump(first["system"])
    assert SESSION_CONTEXT_HEADER in _user_text(first["messages"][0])
    # The memory written in turn one reaches turn two as a new section of its own
    # message; turn three, with nothing changed, sends none.
    assert "payments-deploy-day" not in _user_text(first["messages"][0])
    assert "payments-deploy-day" in _user_text(turn_two["messages"][-1])
    assert SESSION_CONTEXT_HEADER not in _user_text(turn_three["messages"][-1])
    # The tool tool search revealed stays visible after its turn.
    assert "memory_remember" not in {tool["name"] for tool in first["tools"]}
    assert "memory_remember" in {tool["name"] for tool in turn_two["tools"]}
