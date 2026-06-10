"""Query rewriting for bilingual transcript retrieval.

The preferred path uses an LLM to translate and normalize user queries into
Chinese/English variants.  A rule-based fallback is kept so retrieval still
works when API credentials are missing or the LLM call fails.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RewrittenQuery:
    original: str
    semantic_queries: list[str] = field(default_factory=list)
    keyword_queries: list[str] = field(default_factory=list)
    speaker: str | None = None
    time_range: tuple[float, float] | None = None
    intent: list[str] = field(default_factory=list)

    def as_debug_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "semantic_queries": self.semantic_queries,
            "keyword_queries": self.keyword_queries,
            "speaker": self.speaker,
            "time_range": self.time_range,
            "intent": self.intent,
        }


_BILINGUAL_PHRASES: list[tuple[str, str]] = [
    ("价格", "price cost"),
    ("价钱", "price cost"),
    ("成本", "cost"),
    ("租金", "rent"),
    ("房租", "rent"),
    ("太贵", "too expensive high price costs too much"),
    ("便宜", "cheap inexpensive low price"),
    ("预算", "budget"),
    ("交通", "transportation traffic commute"),
    ("公交", "bus public transport"),
    ("地铁", "subway metro"),
    ("安全", "safety security"),
    ("环境", "environment"),
    ("舒适", "comfortable comfort"),
    ("功能", "feature function"),
    ("操作", "operation operate usability"),
    ("用户", "user customer"),
    ("客户", "customer client"),
    ("人群", "target audience users customers"),
    ("需求", "need demand requirement"),
    ("短视频", "short video app"),
    ("购物", "shopping purchase buying"),
    ("主播", "host livestream anchor"),
    ("带货", "livestream shopping selling"),
    ("同意", "agree support yes"),
    ("赞同", "agree support"),
    ("反对", "disagree object oppose"),
    ("不同意", "disagree object oppose"),
    ("质疑", "question challenge doubt"),
    ("建议", "suggest propose recommendation"),
    ("提议", "proposal suggestion"),
    ("总结", "summary conclude recap"),
    ("提问", "question ask"),
    ("询问", "ask question"),
    ("解释", "explain clarification"),
    ("确认", "confirm confirmation"),
    ("讨论", "discuss discussion"),
    ("影响", "impact influence effect"),
    ("政策", "policy policies"),
    ("楼市", "real estate housing market property market"),
    ("房地产", "real estate property housing"),
    ("房贷", "mortgage housing loan"),
    ("贷款", "loan mortgage"),
    ("首套房", "first home first house"),
    ("限购", "purchase restriction home purchase restriction"),
    ("降息", "interest rate cut rate reduction"),
    ("印度使徒", "Apostle of the Indies"),
    ("使徒", "apostle"),
    ("洗礼", "baptism baptized"),
    ("受洗", "baptism baptized"),
    ("有效", "valid effective"),
    ("工位", "workstation desk seat"),
    ("会议室", "meeting room conference room"),
    ("小会议室", "small meeting room"),
    ("大会议室", "large meeting room conference room"),
    ("吸烟室", "smoking room"),
    ("地毯", "carpet"),
    ("刷漆", "paint repaint wall painting"),
    ("仿瓷", "porcelain-like coating wall coating"),
    ("体验", "experience"),
    ("新闻", "news"),
    ("嘉宾", "guest"),
    ("主持人", "host presenter"),
]

_EN_TO_ZH: list[tuple[str, str]] = [
    ("price", "价格 价钱"),
    ("cost", "成本 价格"),
    ("rent", "租金 房租"),
    ("expensive", "太贵 价格高"),
    ("budget", "预算"),
    ("traffic", "交通"),
    ("transport", "交通 公交"),
    ("subway", "地铁"),
    ("metro", "地铁"),
    ("safety", "安全"),
    ("security", "安全"),
    ("environment", "环境"),
    ("comfortable", "舒适"),
    ("feature", "功能"),
    ("function", "功能"),
    ("operate", "操作"),
    ("operation", "操作"),
    ("user", "用户"),
    ("customer", "客户"),
    ("demand", "需求"),
    ("requirement", "需求"),
    ("short video", "短视频"),
    ("shopping", "购物"),
    ("purchase", "购买 购物"),
    ("host", "主持人 主播"),
    ("livestream", "直播 主播 带货"),
    ("agree", "同意 赞同"),
    ("disagree", "反对 不同意"),
    ("oppose", "反对"),
    ("suggest", "建议 提议"),
    ("proposal", "提议 建议"),
    ("summary", "总结"),
    ("question", "提问 问题"),
    ("confirm", "确认"),
    ("discuss", "讨论"),
    ("impact", "影响"),
    ("policy", "政策"),
    ("real estate", "房地产 楼市"),
    ("housing market", "楼市 房地产"),
    ("mortgage", "房贷 贷款"),
    ("loan", "贷款 房贷"),
    ("apostle", "使徒 印度使徒"),
    ("baptism", "洗礼 受洗"),
    ("baptized", "洗礼 受洗"),
    ("valid", "有效"),
    ("workstation", "工位"),
    ("desk", "工位"),
    ("meeting room", "会议室"),
    ("conference room", "会议室"),
    ("smoking room", "吸烟室"),
    ("carpet", "地毯"),
    ("paint", "刷漆"),
    ("experience", "体验"),
    ("news", "新闻"),
    ("guest", "嘉宾"),
]

_INTENT_ALIASES: dict[str, str] = {
    "反对": "反对", "不同意": "反对", "质疑": "反对",
    "disagree": "反对", "oppose": "反对", "object": "反对",
    "提问": "提问", "询问": "提问", "问题": "提问",
    "question": "提问", "ask": "提问",
    "同意": "同意", "赞同": "同意", "认可": "同意",
    "agree": "同意", "support": "同意",
    "提议": "提议", "建议": "提议", "提出": "提议",
    "suggest": "提议", "proposal": "提议", "recommend": "提议",
    "总结": "总结", "归纳": "总结", "summary": "总结", "recap": "总结",
    "命令": "命令", "要求": "命令", "command": "命令", "require": "命令",
    "澄清": "澄清", "解释": "澄清", "clarification": "澄清", "explain": "澄清",
    "确认": "确认", "confirm": "确认", "confirmation": "确认",
    "打断": "打断", "插话": "打断", "interrupt": "打断",
    "寒暄": "寒暄", "闲聊": "寒暄", "smalltalk": "寒暄",
    "回应": "回应", "acknowledgment": "回应",
}


_LLM_SYSTEM = """You rewrite transcript search queries for bilingual retrieval.

Return strict JSON only. Do not add explanations.

The transcript index may contain Chinese and English segments. Given a user
query, produce:
- semantic_queries: 2-4 concise queries including the original meaning in both Chinese and English when useful.
- keyword_queries: 2-4 keyword-only strings in Chinese and English.
- intent: zero or more Chinese labels from [陈述, 提问, 回答, 同意, 反对, 提议, 总结, 命令, 澄清, 确认, 打断, 寒暄, 回应, 旁白, 引述, 转场, 介绍, 评论].
- speaker: a normalized speaker id such as SPEAKER_00 if explicitly requested, otherwise null.
- time_range: [start_seconds, end_seconds] only if explicitly requested, otherwise null.

JSON schema:
{
  "semantic_queries": ["..."],
  "keyword_queries": ["..."],
  "intent": ["..."],
  "speaker": null,
  "time_range": null
}
"""


def rewrite_query(query: str, adapter: Any | None = None) -> RewrittenQuery:
    """Return bilingual semantic/keyword query variants plus constraints.

    If *adapter* is provided, LLM rewrite is attempted first. Any exception or
    malformed response falls back to deterministic rule-based rewriting.
    """
    if adapter is not None:
        try:
            return rewrite_query_with_llm(query, adapter)
        except Exception:
            pass
    return rewrite_query_rule_based(query)


def rewrite_query_with_llm(query: str, adapter: Any) -> RewrittenQuery:
    """Rewrite query through an LLM adapter, then normalize the JSON result."""
    original = (query or "").strip()
    if not original:
        return RewrittenQuery(original="")

    fallback = rewrite_query_rule_based(original)
    messages = [
        {"role": "system", "content": _LLM_SYSTEM},
        {"role": "user", "content": original},
    ]
    raw = adapter.chat(messages, temperature=0.0, max_tokens=700)
    parsed = _extract_json(raw)

    semantic = _coerce_str_list(parsed.get("semantic_queries"))
    keywords = _coerce_str_list(parsed.get("keyword_queries"))
    intent = _coerce_str_list(parsed.get("intent"))
    speaker = parsed.get("speaker")
    time_range = parsed.get("time_range")

    rewritten = RewrittenQuery(
        original=original,
        semantic_queries=_dedupe_nonempty([original] + semantic + fallback.semantic_queries),
        keyword_queries=_dedupe_nonempty([original] + keywords + fallback.keyword_queries),
        speaker=str(speaker).strip() if isinstance(speaker, str) and speaker.strip() else fallback.speaker,
        time_range=_coerce_time_range(time_range) or fallback.time_range,
        # Only use rule-parsed explicit transcript intents. LLMs often label the
        # user's own information-seeking question as "提问", which incorrectly
        # filters out declarative target segments during retrieval.
        intent=_dedupe_nonempty(fallback.intent),
    )
    return rewritten


def rewrite_query_rule_based(query: str) -> RewrittenQuery:
    """Return bilingual semantic/keyword query variants plus constraints."""
    original = (query or "").strip()
    rewritten = RewrittenQuery(original=original)
    if not original:
        return rewritten

    lowered = original.lower()
    rewritten.speaker = _parse_speaker(original)
    rewritten.time_range = _parse_time_range(original)
    rewritten.intent = _parse_intents(original, lowered)

    semantic = [original]
    keywords = [original]

    zh_terms: list[str] = []
    en_terms: list[str] = []

    for zh, en in _BILINGUAL_PHRASES:
        if zh in original:
            zh_terms.append(zh)
            en_terms.append(en)
    for en, zh in _EN_TO_ZH:
        if en in lowered:
            en_terms.append(en)
            zh_terms.append(zh)

    if zh_terms:
        semantic.append(" ".join(dict.fromkeys(zh_terms)))
        keywords.append(" ".join(dict.fromkeys(zh_terms)))
    if en_terms:
        semantic.append(" ".join(dict.fromkeys(en_terms)))
        keywords.append(" ".join(dict.fromkeys(en_terms)))

    if rewritten.intent:
        semantic.append(" ".join(rewritten.intent))
        keywords.append(" ".join(rewritten.intent))

    rewritten.semantic_queries = _dedupe_nonempty(semantic)
    rewritten.keyword_queries = _dedupe_nonempty(keywords)
    return rewritten


def _extract_json(raw: str) -> dict[str, Any]:
    match = re.search(r"\{[\s\S]*\}", raw or "")
    if not match:
        raise ValueError("no JSON object in LLM rewrite response")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("LLM rewrite response is not an object")
    return parsed



def _coerce_str_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    return []


def _coerce_time_range(value: Any) -> tuple[float, float] | None:
    if not (isinstance(value, list) and len(value) == 2):
        return None
    try:
        start = float(value[0])
        end = float(value[1])
    except (TypeError, ValueError):
        return None
    if end <= start:
        return None
    return (start, end)


def _parse_speaker(query: str) -> str | None:
    m = re.search(r"Speaker[_ ]?([A-Za-z]|\d+)", query, re.IGNORECASE)
    if not m:
        return None
    token = m.group(1).strip()
    if token.isdigit():
        return f"SPEAKER_{token.zfill(2)}"
    return token.upper()


def _parse_intents(query: str, lowered: str) -> list[str]:
    found: list[str] = []
    for kw, label in _INTENT_ALIASES.items():
        if (kw in query) or (kw in lowered):
            found.append(label)
    return _dedupe_nonempty(found)


def _parse_time_range(query: str) -> tuple[float, float] | None:
    m = re.search(r"前\s*(\d+)\s*([秒分])", query)
    if m:
        seconds = int(m.group(1)) * (60 if m.group(2) == "分" else 1)
        return (0.0, float(seconds))
    m = re.search(r"(\d+)\s*[:：]\s*(\d+)\s*[到\\-~]\s*(\d+)\s*[:：]\s*(\d+)", query)
    if m:
        start = int(m.group(1)) * 60 + int(m.group(2))
        end = int(m.group(3)) * 60 + int(m.group(4))
        return (float(start), float(end))
    return None


def _dedupe_nonempty(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = " ".join(str(value).split())
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result
