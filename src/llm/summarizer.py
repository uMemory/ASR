"""LLM long-form audio summary generator.

Takes the full transcript (all segments) and asks the LLM to produce
a structured summary: topic, key points, decisions, action items.
"""
from __future__ import annotations

from typing import Any

from src.utils.logger import get_logger

from .base import LLMAdapter

logger = get_logger(__name__)

_SUMMARY_PROMPT_ZH = """你是一个长音频内容整理助手。请根据以下转写内容，
生成一份适用于会议、访谈、新闻、节目、播客、影视片段等多场景的结构化摘要。

输出格式（严格 JSON）：
{
  "topic": "内容主题（一句话）",
  "summary": "内容概要（2-3句话）",
  "key_points": ["关键点1", "关键点2", "..."],
  "decisions": ["决策1", "决策2", "..."] 或 [],
  "action_items": ["待办1", "待办2", "..."] 或 [],
  "participants": ["说话人A", "说话人B", "..."]
}

注意：
- 如果对话中没有明确的决策或待办事项，对应字段返回空列表
- key_points 至少列出 1-3 个要点
- 使用中文输出"""

_SUMMARY_PROMPT_EN = """You are a long-form audio content assistant. Based on the
following transcript, generate a structured summary for meetings, interviews,
news, programs, podcasts, film/TV clips, or other spoken-audio scenarios.

Output format (strict JSON):
{
  "topic": "Content topic (1 sentence)",
  "summary": "Brief summary (2-3 sentences)",
  "key_points": ["point1", "point2", "..."],
  "decisions": ["decision1", "decision2", "..."] or [],
  "action_items": ["action1", "action2", "..."] or [],
  "participants": ["Speaker A", "Speaker B", "..."]
}

Notes:
- If no clear decisions or action items, return empty lists
- At least 1-3 key points required"""


def generate_summary(
    adapter: LLMAdapter,
    segments: list[dict[str, Any]],
    language: str = "zh",
) -> dict[str, Any]:
    """Generate a structured content summary from all segments.

    Returns a dict with keys: topic, summary, key_points, decisions,
    action_items, participants.
    """
    import json
    import re

    if not segments:
        return {"topic": "", "summary": "", "key_points": [], "decisions": [],
                "action_items": [], "participants": []}

    # 构建完整对话文本
    lines = []
    speakers = set()
    for i, seg in enumerate(segments):
        spk = seg.get("speaker", "?")
        text = seg.get("llm_text", "") or seg.get("text", "")
        if text.strip():
            lines.append(f"[{i}] {spk}: {text}")
            speakers.add(spk)

    transcript = "\n".join(lines)

    if language == "en":
        system = _SUMMARY_PROMPT_EN
        user_msg = f"Please summarize this conversation:\n\n{transcript}"
    else:
        system = _SUMMARY_PROMPT_ZH
        user_msg = f"请为以下对话生成摘要：\n\n{transcript}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_msg},
    ]

    logger.info("Summarizer: sending %d segments to %s", len(segments), adapter.model_name)
    raw = adapter.chat(messages, temperature=0.3, max_tokens=2048)

    # 解析 JSON
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            result = json.loads(match.group(0))
            return {
                "topic": result.get("topic", ""),
                "summary": result.get("summary", ""),
                "key_points": result.get("key_points", []),
                "decisions": result.get("decisions", []),
                "action_items": result.get("action_items", []),
                "participants": result.get("participants", list(speakers)),
            }
        except json.JSONDecodeError:
            logger.warning("Summarizer: invalid JSON in LLM response")

    return {"topic": "", "summary": "", "key_points": [], "decisions": [],
            "action_items": [], "participants": list(speakers)}
