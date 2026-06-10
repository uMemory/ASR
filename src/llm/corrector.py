"""LLM context-aware ASR error correction.

Reads the full conversation transcript (timestamps + speaker labels)
and uses the LLM's language understanding to fix obvious ASR mistakes
by considering the dialogue context.
"""
from __future__ import annotations

import json
import re
import ast
from typing import Any

from src.utils.logger import get_logger

from .base import LLMAdapter

logger = get_logger(__name__)

# Maximum segments per LLM call (to stay within context window limits).
_MAX_SEGMENTS_PER_BATCH = 15
_MAX_CORRECTION_TOKENS = 12000


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
    "沒": "没", "備": "备", "場": "场", "續": "续",
    "運": "运", "轉": "转", "辦": "办", "間": "间", "從": "从",
    "親": "亲", "劉": "刘", "陳": "陈", "張": "张", "鄭": "郑",
    "吳": "吴", "楊": "杨", "孫": "孙", "趙": "赵", "羅": "罗",
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
- 默认保持原文，只有在确认存在明显 ASR 错误时才修改
- 修正同音字混淆、语义不通、上下文矛盾
- 保留说话风格和口语化表达
- 不要润色、改写、概括、补充信息或把口语改成书面语
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
   - 报号、叫号、编号、验证码、房间号、工位号、名单编号等数字/字母数字序列必须原样保留。
   - 不要把"001,016,017"这类报号内容改写成"各位"、"大家"或其他语义句子。
   - 不要把阿拉伯数字改写成概括性词语；除非只是补充标点，否则保持原字符串。
3. 语气词规范：少量"嗯"、"啊"等口语词可以保留，但大段机械重复必须删除。
4. 繁体转简体：文本中的繁体中文必须转为简体中文输出。
5. 标点符号恢复：为每句话添加适当的中文标点符号（。！？，、），
   根据语义和语气确定句末标点（陈述用。疑问用？感叹用！），
   句中适当使用逗号分隔意群。不要过度使用感叹号。这是最重要的要求之一。

输出格式（严格 JSON）：
{"segments": [{"index": 0, "text": "修正后的文本", "note": "修正说明（可选）"}, ...]}

其中 index 对应输入中的序号，note 只在有修正时填写。
只输出 JSON 本身，不要输出 Markdown 代码块、解释文字或额外前后缀。"""

_SYSTEM_EN = """You are an English speech recognition post-processing assistant. Your tasks:
1. Read a multi-speaker conversation transcript (with speaker labels and timestamps)
2. Identify and correct obvious ASR (speech recognition) errors using dialogue context
3. Output corrections in JSON format

Correction principles:
- Keep the original text by default. Only modify a segment when there is a clearly identifiable ASR error.
- Only fix clearly wrong words (homophone confusion, semantic errors, context contradictions)
- Preserve speaking style and colloquial expressions
- Do not polish, paraphrase, summarize, normalize style, or make the sentence more formal.
- Do not add or remove complete sentences
- If a sentence has no obvious errors, keep it as-is
- Preserve numbers, IDs, names, titles, abbreviations, and code-like strings unless they are clearly recognized incorrectly.
- Do not modify timestamps or speaker labels

Output format (strict JSON):
{"segments": [{"index": 0, "text": "corrected text", "note": "correction note (optional)"}, ...]}

The index corresponds to the input sequence number. Only include 'note' when a correction was made.
Return raw JSON only. Do not include markdown fences, explanations, or extra text before/after JSON."""


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


def _strip_json_noise(text: str) -> str:
    """Remove common non-JSON wrappers without changing valid JSON content."""
    text = text.strip().lstrip("\ufeff")
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = re.sub(r"^\s*```(?:json|JSON)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def _balanced_json_candidates(raw: str) -> list[str]:
    """Return likely JSON snippets from an LLM response.

    LLMs sometimes wrap JSON in markdown, add explanatory text, or return a
    top-level array.  A balanced scan is safer than a greedy ``\{.*\}`` regex.
    """
    raw = raw or ""
    candidates: list[str] = []

    stripped = _strip_json_noise(raw)
    if stripped:
        candidates.append(stripped)

    for match in re.finditer(r"```(?:json|JSON)?\s*([\s\S]*?)```", raw):
        block = _strip_json_noise(match.group(1))
        if block:
            candidates.append(block)

    for opener, closer in (("{", "}"), ("[", "]")):
        stack = 0
        start: int | None = None
        in_string = False
        escape_next = False
        for i, ch in enumerate(raw):
            if in_string:
                if escape_next:
                    escape_next = False
                elif ch == "\\":
                    escape_next = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == opener:
                if stack == 0:
                    start = i
                stack += 1
            elif ch == closer and stack:
                stack -= 1
                if stack == 0 and start is not None:
                    candidates.append(raw[start:i + 1].strip())
                    start = None

    unique: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        candidate = _strip_json_noise(candidate)
        if candidate and candidate not in seen:
            seen.add(candidate)
            unique.append(candidate)
    return unique


def _loads_loose_json(candidate: str) -> Any:
    """Parse strict JSON first, then tolerate common LLM JSON mistakes."""
    candidate = _strip_json_noise(candidate)
    attempts = [candidate]

    repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
    repaired = re.sub(r"^\s*(?:json|JSON)\s*[:：]\s*", "", repaired)
    if repaired != candidate:
        attempts.append(repaired)

    last_error: Exception | None = None
    for text in attempts:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            last_error = exc

    # Python-literal style responses are common when the model uses single quotes.
    for text in attempts:
        try:
            return ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            last_error = exc

    if last_error:
        raise last_error
    raise ValueError("empty JSON candidate")


def _normalise_correction_items(parsed: Any) -> list[Any]:
    """Extract correction items from common LLM response shapes."""
    if isinstance(parsed, list):
        return parsed
    if not isinstance(parsed, dict):
        return []
    for key in ("segments", "corrections", "results", "items"):
        value = parsed.get(key)
        if isinstance(value, list):
            return value
    # Occasionally returned as {"0": "text", "1": {"text": "..."}}
    if parsed and all(str(k).isdigit() for k in parsed.keys()):
        items: list[dict[str, Any]] = []
        for k, v in parsed.items():
            if isinstance(v, dict):
                item = dict(v)
                item.setdefault("index", int(k))
            else:
                item = {"index": int(k), "text": v}
            items.append(item)
        return items
    return []


def _parse_corrections(raw: str, count: int) -> list[dict[str, Any]]:
    """Parse the LLM JSON response into a list of correction dicts.

    Returns a list index-aligned with ``segments``.  Each entry is
    ``{"text": str, "note": str | None}``.
    """
    # Initialise defaults (no correction)
    corrections: list[dict[str, Any]] = [
        {"text": "", "note": None} for _ in range(count)
    ]

    candidates = _balanced_json_candidates(raw)
    if not candidates:
        logger.warning("Corrector: could not find JSON in LLM response:\n%s", raw[:500])
        return corrections

    parsed: Any = None
    parse_errors: list[str] = []
    for candidate in candidates:
        try:
            parsed = _loads_loose_json(candidate)
            if _normalise_correction_items(parsed):
                break
        except Exception as exc:
            parse_errors.append(f"{type(exc).__name__}: {exc}")
            parsed = None

    if parsed is None:
        logger.warning(
            "Corrector: invalid JSON in LLM response (%s):\n%s",
            "; ".join(parse_errors[-2:]) if parse_errors else "unknown error",
            raw[:500],
        )
        return corrections

    segs = _normalise_correction_items(parsed)
    if not segs:
        logger.warning("Corrector: JSON parsed but no correction list found:\n%s", raw[:500])
        return corrections

    for item in segs:
        if not isinstance(item, dict):
            continue
        idx = item.get("index", item.get("id"))
        if isinstance(idx, str) and idx.isdigit():
            idx = int(idx)
        text = item.get("text", item.get("corrected_text", item.get("correction", "")))
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        note = item.get("note") or item.get("reason") or None
        if isinstance(idx, int) and 0 <= idx < count and text.strip():
            corrections[idx] = {"text": text.strip(), "note": note}

    return corrections


def _build_user_prompt(transcript: str, language: str) -> str:
    if language == "en":
        return (
            "Please review the following transcript. Only correct clear speech recognition errors. "
            "If uncertain, or if the text is merely colloquial or not perfectly fluent, keep it unchanged. "
            "Do not polish, paraphrase, summarize, or change the original meaning.\n\n"
            f"{transcript}"
        )
    return (
        "请检查以下对话转写。只修正明确的语音识别错误；"
        "如果不确定、只是表达不够通顺、或只是口语化表达，请保持原文不变。"
        "不要润色、概括或改写原意。\n\n"
        f"{transcript}"
    )


def _numeric_tokens(text: str) -> list[str]:
    """Return digit/alphanumeric tokens that should survive correction."""
    return re.findall(r"[A-Za-z]?\d+(?:[-_]\d+)?", text or "")


def _is_number_callout(text: str) -> bool:
    """Detect short call-number / ID-list utterances that LLM must not rewrite."""
    raw = (text or "").strip()
    if not raw:
        return False
    digits = re.findall(r"\d", raw)
    if len(digits) >= 3:
        non_space = re.sub(r"\s+", "", raw)
        numeric_like = re.sub(r"[0-9A-Za-z,，.。:：;；、/\-_\[\]()（）号第零一二三四五六七八九十百千万]+", "", non_space)
        # Mostly identifiers, separators, and number words: e.g. 001,016,017.
        if len(numeric_like) <= max(1, len(non_space) * 0.2):
            return True
    zh_digit_chars = len(re.findall(r"[零一二三四五六七八九十百千万]", raw))
    if zh_digit_chars >= 4 and len(raw) <= 30:
        return True
    return False


def _should_accept_correction(original: str, corrected: str, language: str) -> bool:
    """Reject LLM rewrites that destroy number/code utterances."""
    original = (original or "").strip()
    corrected = (corrected or "").strip()
    if not corrected:
        return False
    if language != "zh":
        return True

    if _is_number_callout(original):
        # For call-number utterances, allow at most punctuation/full-width changes.
        orig_core = re.sub(r"[\s,，.。;；:：、]+", "", zh_simplify(original))
        corr_core = re.sub(r"[\s,，.。;；:：、]+", "", zh_simplify(corrected))
        return orig_core == corr_core

    orig_tokens = _numeric_tokens(original)
    if orig_tokens:
        corr_tokens = _numeric_tokens(corrected)
        missing = [tok for tok in orig_tokens if tok not in corr_tokens]
        if missing:
            return False
    return True


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
        raw = adapter.chat(messages, temperature=0.1, max_tokens=_MAX_CORRECTION_TOKENS)
        corrections = _parse_corrections(raw, len(batch))

        for i, seg in enumerate(batch):
            corr = corrections[i]
            new_seg = dict(seg)
            if corr["text"]:
                original_text = str(seg.get("text", "")).strip()
                if not _should_accept_correction(original_text, corr["text"], language):
                    logger.info(
                        "Corrector: rejected unsafe numeric/code rewrite [%d]: %r -> %r",
                        batch_start + i,
                        original_text,
                        corr["text"],
                    )
                    result.append(new_seg)
                    continue
                if corr["text"] != original_text:
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
