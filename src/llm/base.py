"""LLM adapter abstract base class.

All backends (DeepSeek, Claude, MiniMax) implement the same ``chat``
interface so the post-processing functions in this package are
backend-agnostic.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class LLMAdapter(ABC):
    """Abstract LLM adapter.  Subclasses implement ``chat``."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        **kwargs: Any,
    ) -> str:
        """Send a conversation to the LLM and return its response text.

        Parameters
        ----------
        messages : list[dict]
            OpenAI-style message list.  Each dict has at minimum
            ``"role"`` (``"system"`` / ``"user"`` / ``"assistant"``)
            and ``"content"``.
        temperature : float
            Sampling temperature (default 0.1 for deterministic output).
        max_tokens : int
            Maximum tokens in the response.

        Returns
        -------
        str
            LLM response text.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier (e.g. ``"deepseek-chat"``)."""
        ...
