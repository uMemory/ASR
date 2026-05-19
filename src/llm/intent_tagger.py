"""LLM intent tagging — labels each utterance with one or more dialogue acts.

Each utterance may carry **multiple** intents (e.g. "I agree with the plan,
but the budget is too tight" → agree + disagree).

Supported Chinese labels (12 categories):
    陈述 / 提问 / 同意 / 反对 / 提议 / 总结 / 命令 /
    澄清 / 确认 / 打断 / 寒暄 / 回应

English (12 categories):
    statement / question / agree / disagree / proposal / summary /
    command / clarification / confirmation / interruption /
    smalltalk / acknowledgment
"""
from __future__ import annotations

import json
import re
from typing import Any

from src.utils.logger import get_logger

from .base import LLMAdapter

logger = get_logger(__name__)

_MAX_PER_BATCH = 20

# ── Label definitions (injected into prompt) ─────────────────────────
_ZH_LABEL_DEFS = {
    "陈述": "陈述一个事实、观点、信息或观察（最常见的意图）",
    "提问": "提出一个问题，向他人询问信息或意见",
    "同意": "表示赞同、认可他人的观点或建议",
    "反对": "表示不同意、否定或质疑他人的观点",
    "提议": "主动提出一个建议、方案、计划或行动方向",
    "总结": "对之前的讨论进行归纳、概括或做出结论",
    "命令": "发出指令、要求、安排或分配任务",
    "澄清": "追问细节、要求对方进一步解释或说明",
    "确认": "确认已收到的信息，如'对吗？''明白了''收到了'",
    "打断": "打断别人说话、插话或强行转换话题",
    "寒暄": "开场白、问候、客套话、与主题无关的闲聊",
    "回应": "简短的回应词：'嗯''好''对''是''哦'等，无实质内容",
}

_EN_LABEL_DEFS = {
    "statement": "Stating a fact, opinion, information, or observation (most common intent)",
    "question": "Asking a question, requesting information or opinion from others",
    "agree": "Expressing agreement with or endorsement of another's point or suggestion",
    "disagree": "Expressing disagreement, denial, or challenge to another's point",
    "proposal": "Proactively suggesting an idea, plan, course of action",
    "summary": "Summarizing, recapping, or drawing conclusions from prior discussion",
    "command": "Issuing an instruction, demand, assignment, or task delegation",
    "clarification": "Asking for more detail, requesting further explanation",
    "confirmation": "Confirming received info, e.g. 'right?', 'got it', 'understood'",
    "interruption": "Cutting someone off, interjecting, or forcibly changing the topic",
    "smalltalk": "Greetings, pleasantries, off-topic chatter unrelated to the main subject",
    "acknowledgment": "Brief backchannel: 'mm', 'okay', 'right', 'yeah' — no substantive content",
}

_ZH_LABELS = list(_ZH_LABEL_DEFS.keys())
_EN_LABELS = list(_EN_LABEL_DEFS.keys())

_DEFAULT_ZH = ["陈述"]
_DEFAULT_EN = ["statement"]

# ── Prompt templates ─────────────────────────────────────────────────

_ZH_SYSTEM = """你是一个对话意图标注助手。请为以下对话中的每一句话标注意图。

意图标签及含义：
{label_desc}

标注原则：
- 一句话可能有多个意图，请标注所有适用的标签
  （例："我同意这个方案，但是预算不够" → 同意 + 反对）
- 根据说话内容和上下文判断意图，而非仅看表面词汇
- 简短回应词（"嗯""好""对"）标注为"回应"
- 连续的编号/报数通常是"陈述"
- 有疑问语气且期待回答 → "提问"
- 反对通常带有否定词或质疑语气
- 大多数句子至少包含"陈述"（除非是纯回应或寒暄）

输出严格 JSON（不要其他文字）：
{{"intents": [{{"index": 0, "intents": ["陈述"]}}, {{"index": 1, "intents": ["提议", "提问"]}}, ...]}}"""

_EN_SYSTEM = """You are a dialogue intent annotation assistant. Label each utterance with ALL applicable intents.

Intent labels and meanings:
{label_desc}

Annotation principles:
- An utterance may carry MULTIPLE intents — label ALL that apply
  (e.g. "I agree with the plan, but the budget is too low" → agree + disagree)
- Judge intent by content AND conversational context, not just surface words
- Brief backchannels ("mm", "okay", "right") → "acknowledgment"
- Sequential number-reading is usually "statement"
- Questioning tone expecting an answer → "question"
- Disagreement usually contains negation words or challenging tone
- Most utterances carry at least "statement" (unless pure acknowledgment or smalltalk)

Output strict JSON (no extra text):
{{"intents": [{{"index": 0, "intents": ["statement"]}}, {{"index": 1, "intents": ["proposal", "question"]}}, ...]}}"""


def _build_label_description(language: str) -> str:
    defs = _ZH_LABEL_DEFS if language == "zh" else _EN_LABEL_DEFS
    lines = [f"- {label}：{desc}" if language == "zh"
             else f"- {label}: {desc}"
             for label, desc in defs.items()]
    return "\n".join(lines)


def _format_utterances(segments: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for i, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        speaker = seg.get("speaker", "UNKNOWN")
        if text:
            lines.append(f"[{i}] {speaker}: {text}")
    return "\n".join(lines)


def _normalize_intent(raw_intent: Any) -> list[str]:
    """Normalize LLM output to a list of intent strings.

    Handles both old-style (single string) and new-style (list) formats.
    """
    if isinstance(raw_intent, str):
        return [raw_intent.strip()] if raw_intent.strip() else []
    if isinstance(raw_intent, list):
        return [s.strip() for s in raw_intent if isinstance(s, str) and s.strip()]
    return []


def _parse_intents(raw: str, count: int, default: list[str]) -> list[list[str]]:
    """Parse multi-label intents from the LLM JSON response.

    Returns a list of ``count`` entries, each a ``list[str]`` of labels.
    Falls back to ``default`` for unparseable indices.
    """
    intents: list[list[str]] = [list(default) for _ in range(count)]

    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        logger.warning("IntentTagger: no JSON in response:\n%s", raw[:500])
        return intents

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.warning("IntentTagger: invalid JSON:\n%s", raw[:500])
        return intents

    items = parsed.get("intents", [])
    if not isinstance(items, list):
        return intents

    for item in items:
        idx = item.get("index")
        raw_intent = item.get("intents") or item.get("intent")
        if not (isinstance(idx, int) and 0 <= idx < count):
            continue
        labels = _normalize_intent(raw_intent)
        if labels:
            intents[idx] = labels

    return intents


def llm_tag_intents(
    adapter: LLMAdapter,
    segments: list[dict[str, Any]],
    language: str = "zh",
    intent_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Tag each segment with one or more dialogue intent labels.

    Parameters
    ----------
    adapter : LLMAdapter
    segments : list[dict]
        Merged diarization × ASR segments.
    language : str
        ``"zh"`` or ``"en"``.
    intent_labels : list[str] or None
        Custom label set.  ``None`` uses the built-in 12-label taxonomy.

    Returns
    -------
    list[dict]
        Segments with ``"intent"`` set to a **list** of labels,
        e.g. ``["陈述"]`` or ``["同意", "反对"]``.
    """
    if not segments:
        return segments

    if intent_labels is None:
        intent_labels = _EN_LABELS if language == "en" else _ZH_LABELS
    default = _DEFAULT_EN if language == "en" else _DEFAULT_ZH

    label_desc = _build_label_description(language)
    template = _EN_SYSTEM if language == "en" else _ZH_SYSTEM
    system = template.format(label_desc=label_desc)

    result: list[dict[str, Any]] = []
    for batch_start in range(0, len(segments), _MAX_PER_BATCH):
        batch = segments[batch_start: batch_start + _MAX_PER_BATCH]
        utterances = _format_utterances(batch)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": utterances},
        ]

        logger.info("IntentTagger: tagging %d segments with %s",
                    len(batch), adapter.model_name)
        raw = adapter.chat(messages, temperature=0.0, max_tokens=2048)
        intents = _parse_intents(raw, len(batch), default)

        for i, seg in enumerate(batch):
            new_seg = dict(seg)
            new_seg["intent"] = intents[i]
            result.append(new_seg)

    return result
