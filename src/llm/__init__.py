"""LLM post-processing module.

Provides:
- ``load_llm_adapter`` — factory for DeepSeek / Claude / MiniMax
- ``llm_correct_segments`` — context-aware ASR error correction
- ``llm_check_consistency`` — speaker identity consistency check
- ``llm_tag_intents`` — dialogue intent labelling
- ``generate_summary`` — meeting summary generation
"""
from __future__ import annotations

from .base import LLMAdapter
from .factory import load_llm_adapter
from .corrector import llm_correct_segments, clean_hallucination, zh_simplify
from .consistency import llm_check_consistency
from .intent_tagger import llm_tag_intents
from .summarizer import generate_summary

__all__ = [
    "LLMAdapter",
    "load_llm_adapter",
    "llm_correct_segments",
    "llm_check_consistency",
    "llm_tag_intents",
    "generate_summary",
    "clean_hallucination",
]
