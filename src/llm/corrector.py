"""LLM context-aware ASR error correction.

Reads the full conversation transcript (timestamps + speaker labels)
and uses the LLM's language understanding to fix obvious ASR mistakes
by considering the dialogue context.
"""
from __future__ import annotations

import json
import re
from typing import Any

from src.utils.logger import get_logger

from .base import LLMAdapter

logger = get_logger(__name__)

# Maximum segments per LLM call (to stay within context window limits).
_MAX_SEGMENTS_PER_BATCH = 15


# 繁→简常用字映射
_ZH_HANS = {
    "臺": "台", "鏈": "链", "曆": "历", "歷": "历", "鬱": "郁",
    "襯": "衬", "纔": "才", "嚐": "尝", "沖": "冲", "醜": "丑",
    "導": "导", "鐺": "铛", "澱": "淀", "鬥": "斗", "髮": "发",
    "範": "范", "復": "复", "幹": "干", "穀": "谷", "後": "后",
    "劃": "划", "夥": "伙", "獲": "获", "擊": "击", "飢": "饥",
    "儘": "尽", "驚": "惊", "據": "据", "啓": "启", "墾": "垦",
    "虧": "亏", "瞭": "了", "黴": "霉", "彌": "弥", "麵": "面",
    "廟": "庙", "闢": "辟", "蘋": "苹", "僕": "仆", "樸": "朴",
    "簽": "签", "確": "确", "灑": "洒", "捨": "舍", "勝": "胜",
    "適": "适", "術": "术", "蘇": "苏", "壇": "坛", "嘆": "叹",
    "塗": "涂", "圍": "围", "繫": "系", "係": "系", "鹹": "咸",
    "纖": "纤", "鬚": "须", "鏇": "旋", "藥": "药", "葉": "叶",
    "傭": "佣", "餘": "余", "禦": "御", "籲": "吁", "園": "园",
    "願": "愿", "嶽": "岳", "徵": "征", "緻": "致", "製": "制",
    "築": "筑", "鍾": "钟", "鐘": "钟", "醞": "酝", "妳": "你",
    "牠": "它", "祇": "只", "佈": "布", "佔": "占", "併": "并",
    "採": "采", "綵": "彩", "睏": "困", "託": "托", "洩": "泄",
    "淩": "凌", "菸": "烟", "蹟": "迹", "體": "体", "國": "国",
    "會": "会", "時": "时", "現": "现", "實": "实", "個": "个",
    "為": "为", "這": "这", "對": "对", "說": "说", "來": "来",
    "們": "们", "過": "过", "開": "开", "關": "关", "學": "学",
    "頭": "头", "書": "书", "長": "长", "門": "门", "見": "见",
    "貝": "贝", "車": "车", "馬": "马", "魚": "鱼", "鳥": "鸟",
    "龍": "龙", "龜": "龟", "風": "风", "電": "电", "飛": "飞",
    "無": "无", "東": "东", "萬": "万", "與": "与", "義": "义",
    "樂": "乐", "喬": "乔", "買": "买", "賣": "卖", "讀": "读",
    "覺": "觉", "黃": "黄", "鵬": "鹏", "還": "还", "應": "应",
    "價": "价", "錢": "钱", "規": "规", "歲": "岁", "資": "资",
    "針": "针", "點": "点", "產": "产", "業": "业", "務": "务",
    "話": "话", "題": "题", "問": "问", "員": "员", "選": "选",
    "讓": "让", "聽": "听", "雜": "杂", "簡": "简", "斷": "断",
    "許": "许", "區": "区", "討": "讨", "論": "论", "適": "适",
    "給": "给", "聲": "声", "華": "华", "廣": "广", "線": "线",
}


def zh_simplify(text: str) -> str:
    """繁→简转换（字符映射，无外部依赖）。"""
    result = []
    for ch in text:
        result.append(_ZH_HANS.get(ch, ch))
    return "".join(result)


def clean_hallucination(text: str) -> str:
    """清理 ASR 幻觉：大段重复字符→截断为合理形式。

    规则：
    - 同一字符连续出现 >15 次 → 保留前 2 个
    - 同一 2-3 字词组重复 >8 次 → 保留第一次出现
    """
    if not text:
        return text

    # Common Whisper outro hallucinations triggered by short/noisy clips.
    for phrase in ("谢谢大家的观看", "谢谢大家的收看", "我们下个影片见", "下个影片见", "感谢观看"):
        text = text.replace(phrase, "")

    # 1. 单字符重复：如 "嗯"×100 → "嗯嗯"
    text = re.sub(r'(.)\1{15,}', r'\1\1', text)

    # 2. 双字重复：如 "嗯嗯嗯嗯嗯嗯" → "嗯嗯"（已被规则1覆盖）
    #    但 "对对对对对..." → "对对"
    #    已被规则1处理

    # 3. 三字词组重复：如 "还不知道还不知道还不知道..." → "还不知道"
    text = re.sub(r'(.{2,3}?)\1{8,}', r'\1', text)

    # 4. 清理"嗯嗯"中可能残留的空格和标点打断
    text = re.sub(r'嗯[\s嗯]{10,}嗯', '嗯嗯', text)

    return text.strip()


_SYSTEM_ZH = """你是一个中文语音识别后处理助手。你的任务是：
1. 阅读一段多人对话的转写文本（已标注说话人和时间）
2. 结合对话上下文，识别并修正明显的 ASR（语音识别）错误
3. 输出 JSON 格式的修正结果

修正原则：
- 修正同音字混淆、语义不通、上下文矛盾
- 保留说话风格和口语化表达
- 不要添加或删除完整句子
- 如果某句话没有明显的识别错误，保持原样
- 不修改时间戳和说话人标识

重要的清理规则（必须执行）：
1. ASR 幻觉清理：如果一句话中出现大段无意义的重复（如"嗯"重复20次以上、
   "对"重复10次以上、或大段无意义的数字串），必须清理为可读形式。
   - "对对对 十二对,还得看我们这个价钱和规律嗯嗯嗯嗯...(100个嗯)..."
     → "对，十二对，还得看我们这个价钱和规律。"
   - 保留少量自然的口语重复（如"对对"、"嗯嗯"），但删除机械性的大量重复。
2. 数字编号：保留实际朗读的会议编号（如"零零六"），不要删除真实对话内容。
3. 语气词规范：少量"嗯"、"啊"等口语词可以保留，但大段机械重复必须删除。
4. 繁体转简体：文本中的繁体中文必须转为简体中文输出。
5. 标点符号恢复：为每句话添加适当的中文标点符号（。！？，、），
   根据语义和语气确定句末标点（陈述用。疑问用？感叹用！），
   句中适当使用逗号分隔意群。不要过度使用感叹号。这是最重要的要求之一。

输出格式（严格 JSON）：
{"segments": [{"index": 0, "text": "修正后的文本", "note": "修正说明（可选）"}, ...]}

其中 index 对应输入中的序号，note 只在有修正时填写。"""

_SYSTEM_EN = """You are an English speech recognition post-processing assistant. Your tasks:
1. Read a multi-speaker conversation transcript (with speaker labels and timestamps)
2. Identify and correct obvious ASR (speech recognition) errors using dialogue context
3. Output corrections in JSON format

Correction principles:
- Only fix clearly wrong words (homophone confusion, semantic errors, context contradictions)
- Preserve speaking style and colloquial expressions
- Do not add or remove complete sentences
- If a sentence has no obvious errors, keep it as-is
- Do not modify timestamps or speaker labels

Output format (strict JSON):
{"segments": [{"index": 0, "text": "corrected text", "note": "correction note (optional)"}, ...]}

The index corresponds to the input sequence number. Only include 'note' when a correction was made."""


def _format_transcript(segments: list[dict[str, Any]]) -> str:
    """Format segments as a readable transcript for the LLM."""
    lines: list[str] = []
    for i, seg in enumerate(segments):
        start = seg.get("start", 0)
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "")
        if not text.strip():
            continue
        # 预清理 ASR 幻觉（重复字符截断）
        text = clean_hallucination(text)
        if not text.strip():
            continue
        ts = f"{int(start // 60):02d}:{int(start % 60):02d}"
        lines.append(f"[{i}] {ts} {speaker}: {text}")
    return "\n".join(lines)


def _parse_corrections(raw: str, count: int) -> list[dict[str, Any]]:
    """Parse the LLM JSON response into a list of correction dicts.

    Returns a list index-aligned with ``segments``.  Each entry is
    ``{"text": str, "note": str | None}``.
    """
    # Initialise defaults (no correction)
    corrections: list[dict[str, Any]] = [
        {"text": "", "note": None} for _ in range(count)
    ]

    # Extract JSON from the response (may be wrapped in markdown fences)
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        logger.warning("Corrector: could not find JSON in LLM response:\n%s", raw[:500])
        return corrections

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        logger.warning("Corrector: invalid JSON in LLM response:\n%s", raw[:500])
        return corrections

    segs = parsed.get("segments", [])
    if not isinstance(segs, list):
        return corrections

    for item in segs:
        idx = item.get("index")
        text = item.get("text", "")
        note = item.get("note") or None
        if isinstance(idx, int) and 0 <= idx < count and text.strip():
            corrections[idx] = {"text": text.strip(), "note": note}

    return corrections


def _build_user_prompt(transcript: str, language: str) -> str:
    if language == "en":
        return f"Please review and correct the following conversation transcript:\n\n{transcript}"
    return f"请检查并修正以下对话转写：\n\n{transcript}"


def llm_correct_segments(
    adapter: LLMAdapter,
    segments: list[dict[str, Any]],
    language: str = "zh",
) -> list[dict[str, Any]]:
    """Apply LLM context-aware correction to merged segments.

    Parameters
    ----------
    adapter : LLMAdapter
        The LLM backend to use.
    segments : list[dict]
        Merged diarization × ASR segments with ``start``, ``end``,
        ``speaker``, ``text`` keys.
    language : str
        ``"zh"`` or ``"en"`` — controls prompt language.

    Returns
    -------
    list[dict]
        Segments with ``text`` optionally corrected by the LLM.
        Non-text keys are preserved unchanged.
    """
    if not segments:
        return segments

    system = _SYSTEM_ZH if language == "en" else _SYSTEM_ZH  # default zh
    # Actually: use language-appropriate system prompt
    system = _SYSTEM_EN if language == "en" else _SYSTEM_ZH

    # Batch processing for long transcripts
    result: list[dict[str, Any]] = []
    for batch_start in range(0, len(segments), _MAX_SEGMENTS_PER_BATCH):
        batch = segments[batch_start: batch_start + _MAX_SEGMENTS_PER_BATCH]
        transcript = _format_transcript(batch)

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": _build_user_prompt(transcript, language)},
        ]

        logger.info("Corrector: sending %d segments to %s", len(batch), adapter.model_name)
        raw = adapter.chat(messages, temperature=0.1, max_tokens=4096)
        corrections = _parse_corrections(raw, len(batch))

        for i, seg in enumerate(batch):
            corr = corrections[i]
            new_seg = dict(seg)
            if corr["text"]:
                if corr["text"] != seg.get("text", "").strip():
                    logger.debug(
                        "  Corrected [%d]: %r → %r  (%s)",
                        batch_start + i,
                        seg.get("text"),
                        corr["text"],
                        corr.get("note", "-"),
                    )
                new_seg["text"] = corr["text"]
                if corr.get("note"):
                    new_seg["correction_note"] = corr["note"]
            result.append(new_seg)

    return result
