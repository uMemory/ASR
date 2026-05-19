"""LLM adapter factory — creates the active adapter from env / config."""
from __future__ import annotations

from typing import Any

from src.utils.config import get_env

from .base import LLMAdapter


def load_llm_adapter(cfg: dict[str, Any] | None = None,
                     model: str | None = None) -> LLMAdapter:
    """Create an LLM adapter for the currently active backend.

    The backend is selected via the ``ACTIVE_LLM`` env var (``.env``),
    which can be ``"deepseek"``, ``"claude"``, or ``"minimax"``.
    Falls back to ``"deepseek"`` if unset.

    Parameters
    ----------
    cfg : dict or None
        Optional config dict.
    model : str or None
        Override model name.  Falls back to env var ``DEEPSEEK_MODEL`` /
        ``CLAUDE_MODEL``, then to code defaults.
    """
    backend = (get_env("ACTIVE_LLM") or "deepseek").strip().lower()

    if backend == "deepseek":
        from .deepseek import DeepSeekAdapter

        api_key = get_env("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set in .env")
        base_url = get_env("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1"
        _model = model or get_env("DEEPSEEK_MODEL") or "deepseek-v4-flash"
        return DeepSeekAdapter(api_key=api_key, base_url=base_url, model=_model)

    if backend == "claude":
        from .claude import ClaudeAdapter

        api_key = get_env("CLAUDE_API_KEY")
        if not api_key:
            raise RuntimeError("CLAUDE_API_KEY is not set in .env")
        base_url = get_env("CLAUDE_BASE_URL") or None
        _model = model or get_env("CLAUDE_MODEL") or "claude-sonnet-4-20250514"
        return ClaudeAdapter(api_key=api_key, base_url=base_url, model=_model)

    if backend == "minimax":
        # MiniMax is OpenAI-compatible; reuse DeepSeek adapter with
        # MiniMax credentials.
        from .deepseek import DeepSeekAdapter as MiniMaxAdapter

        api_key = get_env("MINIMAX_API_KEY")
        if not api_key:
            raise RuntimeError("MINIMAX_API_KEY is not set in .env")
        base_url = get_env("MINIMAX_BASE_URL") or "https://api.minimaxi.com/v1"
        return MiniMaxAdapter(api_key=api_key, base_url=base_url, model="minimax-text-01")

    raise ValueError(f"Unknown LLM backend: {backend!r}.  "
                     "Set ACTIVE_LLM to claude / deepseek / minimax in .env")
