"""DeepSeek LLM adapter (OpenAI-compatible protocol)."""
from __future__ import annotations

from typing import Any

from openai import OpenAI

from .base import LLMAdapter


class DeepSeekAdapter(LLMAdapter):
    """DeepSeek API via OpenAI-compatible SDK.

    Default model: deepseek-v4-flash (V4 lightweight, fast + cheap).
    Alternatives: deepseek-v4-pro (V4 flagship), deepseek-chat (legacy V3.2).
    deepseek-chat will be deprecated on 2026-07-24.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        model: str = "deepseek-v4-flash",
    ) -> None:
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url)

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
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return response.choices[0].message.content or ""
