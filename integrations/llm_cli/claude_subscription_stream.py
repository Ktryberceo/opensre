"""Decode Claude Code stream-json without displaying envelopes or partial failures as success."""

from __future__ import annotations

import json
from typing import Any

from core.llm.types import LLMResponse


class ClaudeSubscriptionStream:
    """One invocation's decoder: require a successful terminal result and retain usage."""

    def __init__(self) -> None:
        self._pending = ""
        self._result: dict[str, Any] | None = None
        self._text: list[str] = []
        self._model: str | None = None
        self._effort: str | None = None

    def feed(self, chunk: str) -> list[str]:
        self._pending += chunk
        lines = self._pending.split("\n")
        self._pending = lines.pop()
        deltas: list[str] = []
        for line in lines:
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except ValueError as exc:
                raise RuntimeError("Claude Code returned an invalid stream record.") from exc
            if not isinstance(event, dict):
                raise RuntimeError("Claude Code returned an invalid stream record.")
            kind = event.get("type")
            if kind == "system" and event.get("subtype") == "init" and self._result is None:
                model = event.get("model")
                if isinstance(model, str) and model.strip():
                    self._model = model.strip()
                effort = event.get("effort")
                if isinstance(effort, str) and effort in {
                    "low",
                    "medium",
                    "high",
                    "xhigh",
                    "max",
                    "auto",
                }:
                    self._effort = effort
            elif kind == "result":
                if self._result is not None:
                    raise RuntimeError("Claude Code returned more than one terminal result.")
                self._result = event
            elif kind == "stream_event":
                if self._result is not None:
                    raise RuntimeError("Claude Code emitted stream data after its terminal result.")
                delta = (event.get("event") or {}).get("delta") or {}
                if delta.get("type") == "text_delta" and isinstance(delta.get("text"), str):
                    text = delta["text"]
                    self._text.append(text)
                    deltas.append(text)
            elif kind == "rate_limit_event":
                info = event.get("rate_limit_info") or {}
                if info.get("status") == "rejected":
                    raise RuntimeError(
                        "Claude subscription quota is exhausted. Wait for reset or check Claude Code /usage; no API-billing fallback was selected."
                    )
        return deltas

    def finish(self) -> LLMResponse:
        """Reject missing/error results even when the CLI exits zero after emitting text."""
        if self._pending.strip():
            self.feed("\n")
        result = self._result
        if result is None:
            raise RuntimeError("Claude Code stream ended before its terminal result.")
        if result.get("is_error") is not False or result.get("subtype") != "success":
            raise RuntimeError(
                "Claude Code did not complete the request. Check its authentication, subscription quota, and permissions; no API-billing fallback was selected."
            )
        content = result.get("result")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Claude Code returned an empty terminal result.")
        usage = result.get("usage") or {}
        model = self._model
        model_usage = result.get("modelUsage")
        if model is None and isinstance(model_usage, dict) and len(model_usage) == 1:
            reported_model = next(iter(model_usage))
            if isinstance(reported_model, str) and reported_model.strip():
                model = reported_model.strip()

        def tokens(key: str) -> int | None:
            value = usage.get(key)
            return (
                value
                if isinstance(value, int) and not isinstance(value, bool) and value >= 0
                else None
            )

        return LLMResponse(
            content,
            tokens("input_tokens"),
            tokens("output_tokens"),
            tokens("cache_read_input_tokens"),
            tokens("cache_creation_input_tokens"),
            model=model,
            reasoning_effort=self._effort,
        )
