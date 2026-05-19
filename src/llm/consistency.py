"""LLM cross-speaker consistency check.

Analyses the conversation flow and checks whether Pyannote has:
- Split the same speaker into multiple speaker IDs (under-segmentation fix)
- Merged different speakers into one ID (over-segmentation fix)
- Assigned unrealistic speaker switches given the dialogue context
"""
from __future__ import annotations

import json
import re
from typing import Any

from src.utils.logger import get_logger

from .base import LLMAdapter

logger = get_logger(__name__)

_SYSTEM_ZH = """你是一个说话人日志（Speaker Diarization）后处理助手。

你收到一段经过自动说话人分离的对话转写。由于算法限制，可能出现以下问题：
1. 同一说话人被错误分配了多个不同的 Speaker ID（应该合并）
2. 不同说话人被错误合并到同一个 Speaker ID（应该拆分）

你的任务是分析对话内容、说话风格和逻辑，判断是否存在上述问题。

输出格式（严格 JSON）：
{
  "merge_suggestions": [{"from": "SPEAKER_01", "to": "SPEAKER_03", "confidence": 0.8, "reason": "两者语调一致"}],
  "split_suggestions": [],
  "notes": "整体分析说明"
}

如果没有发现问题，merge_suggestions 和 split_suggestions 为空列表。"""

_SYSTEM_EN = """You are a speaker diarization post-processing assistant.

You receive a conversation transcript processed by automatic speaker diarization.
Due to algorithmic limitations, the following issues may occur:
1. The same speaker was incorrectly assigned multiple different Speaker IDs (should be merged)
2. Different speakers were incorrectly merged into the same Speaker ID (should be split)

Your task is to analyze the conversation content, speaking style, and logic to identify such issues.

Output format (strict JSON):
{
  "merge_suggestions": [{"from": "SPEAKER_01", "to": "SPEAKER_03", "confidence": 0.8, "reason": "similar speaking style"}],
  "split_suggestions": [],
  "notes": "overall analysis"
}

If no issues are found, merge_suggestions and split_suggestions should be empty lists."""


def _format_for_consistency(segments: list[dict[str, Any]]) -> str:
    """Format segments grouped by speaker for consistency analysis."""
    # Group by speaker
    speaker_texts: dict[str, list[str]] = {}
    for seg in segments:
        spk = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        if text:
            speaker_texts.setdefault(spk, []).append(text)

    lines: list[str] = ["=== Speaker utterances ==="]
    for spk, texts in sorted(speaker_texts.items()):
        lines.append(f"\n{spk}:")
        for t in texts[:10]:  # limit per speaker
            lines.append(f"  - {t}")
    return "\n".join(lines)


def llm_check_consistency(
    adapter: LLMAdapter,
    segments: list[dict[str, Any]],
    language: str = "zh",
) -> list[dict[str, Any]]:
    """Check and optionally fix speaker identity consistency.

    Parameters
    ----------
    adapter : LLMAdapter
    segments : list[dict]
        Merged segments.
    language : str

    Returns
    -------
    list[dict]
        Segments with speaker labels possibly adjusted.
    """
    if len(segments) < 4:
        return segments  # too few segments for meaningful consistency check

    system = _SYSTEM_EN if language == "en" else _SYSTEM_ZH
    transcript = _format_for_consistency(segments)

    prompt = "请检查以下说话人标注是否一致" if language != "en" else \
             "Please check the following speaker labels for consistency"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{prompt}:\n\n{transcript}"},
    ]

    logger.info("Consistency: checking %d segments with %s",
                len(segments), adapter.model_name)
    raw = adapter.chat(messages, temperature=0.1, max_tokens=2048)

    # Parse merge suggestions
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        logger.debug("Consistency: no JSON in response, keeping labels")
        return segments

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return segments

    merges = parsed.get("merge_suggestions", [])
    if not merges:
        return segments

    # Apply merges
    merge_map: dict[str, str] = {}
    for m in merges:
        frm = m.get("from", "")
        to = m.get("to", "")
        if frm and to and frm != to:
            merge_map[frm] = to
            logger.info("  Merge: %s → %s (reason: %s)", frm, to, m.get("reason", "-"))

    if not merge_map:
        return segments

    result: list[dict[str, Any]] = []
    for seg in segments:
        new_seg = dict(seg)
        spk = seg.get("speaker", "")
        if spk in merge_map:
            new_seg["speaker"] = merge_map[spk]
            if "consistency_note" not in new_seg:
                new_seg["consistency_note"] = f"merged {spk}→{merge_map[spk]}"
        result.append(new_seg)

    return result
