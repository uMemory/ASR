"""Claude LLM adapter (Anthropic SDK, via fastCode AI gateway)."""
from __future__ import annotations

from typing import Any

from anthropic import Anthropic

from .base import LLMAdapter


class ClaudeAdapter(LLMAdapter):
    """Claude API via Anthropic SDK.

    By default routes through the fastCode AI unified gateway
    (``CLAUDE_BASE_URL`` in ``.env``) which proxies Claude to avoid
    direct internet access requirements on some cloud machines.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "claude-sonnet-4-20250514",
    ) -> None:
        self._model = model
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = Anthropic(**kwargs)

    @property
    def model_name(self) -> str:
        return self._model

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        # Anthropic does not use a separate "system" role — extract it.
        system_prompt = ""
        user_messages: list[dict[str, Any]] = []
        for m in messages:
            if m["role"] == "system":
                system_prompt += m.get("content", "") + "\n"
            else:
                user_messages.append(m)

        response = self._client.messages.create(
            model=self._model,
            system=system_prompt.strip() or None,
            messages=user_messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        # Anthropic returns a list of content blocks
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""
