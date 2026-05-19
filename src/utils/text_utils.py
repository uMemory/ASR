"""文本后处理工具——长段切分、标点处理等。"""
from __future__ import annotations

import re
from typing import Any

# 用于切分的中文标点（含半角和全角逗号）
_SPLIT_PUNCT = re.compile(r'[。！？；，,、\n]')
# 仅句末标点
_SENTENCE_END = re.compile(r'[。！？\n]')
# 句中标点
_CLAUSE_PUNCT = re.compile(r'[，,、；]')

_MAX_CHARS = 80       # 单段最大字符数
_MAX_DURATION = 15.0  # 单段最大秒数


def split_long_segments(
    segments: list[dict[str, Any]],
    max_chars: int = _MAX_CHARS,
    max_duration: float = _MAX_DURATION,
) -> list[dict[str, Any]]:
    """将过长的段切分为较短的段。

    策略：
    1. 优先在句末标点（。！？）处切分
    2. 其次在句中标点（，、；）处切分
    3. 无标点时按字符数均匀切分
    4. 时间段按字符比例分配

    参数
    ----------
    segments : list[dict]
        含 ``start``, ``end``, ``text`` 的段列表。
    max_chars : int
        单段最大字符数（默认 80）。
    max_duration : float
        单段最大秒数（默认 15.0）。

    返回
    -------
    list[dict]
        切分后的段列表。
    """
    result: list[dict[str, Any]] = []

    for seg in segments:
        txt = seg.get("text", "")
        dur = seg["end"] - seg["start"]

        if dur <= max_duration and len(txt) <= max_chars:
            result.append(seg)
            continue

        # 查找所有可切分位置
        punct_matches = list(_SPLIT_PUNCT.finditer(txt))

        if punct_matches:
            result.extend(_split_at_punct(seg, txt, dur, punct_matches, max_chars, max_duration))
        else:
            result.extend(_split_evenly(seg, txt, dur, max_chars, max_duration))

    return result


def _split_at_punct(
    seg: dict[str, Any],
    txt: str,
    dur: float,
    punct_matches: list[re.Match],
    max_chars: int,
    max_duration: float,
) -> list[dict[str, Any]]:
    """在标点处切分，同时确保每段不超过 max_chars 和 max_duration。"""
    import math

    parts: list[str] = []
    last_end = 0

    for m in punct_matches:
        end = m.end()
        parts.append(txt[last_end:end])
        last_end = end
    if last_end < len(txt):
        parts.append(txt[last_end:])

    # 合并过短的段，控制每段不超过 max_chars 且不超 max_duration
    merged: list[str] = []
    buf = ""
    for p in parts:
        # 估算当前 buf 的时长占比
        buf_ratio = (len(buf) + len(p)) / max(len(txt), 1)
        buf_dur = buf_ratio * dur if buf else 0
        if buf and (len(buf) + len(p) > max_chars or buf_dur > max_duration):
            merged.append(buf)
            buf = p
        else:
            buf += p
    if buf:
        merged.append(buf)

    total_chars = max(sum(len(p) for p in merged), 1)
    t = seg["start"]
    result: list[dict[str, Any]] = []
    for p in merged:
        part_dur = dur * len(p) / total_chars
        result.append({**seg, "text": p.strip(), "start": t, "end": t + part_dur})
        t += part_dur
    return result


def _split_evenly(
    seg: dict[str, Any],
    txt: str,
    dur: float,
    max_chars: int,
    max_duration: float,
) -> list[dict[str, Any]]:
    """无标点时按字符数和时长双重约束均匀切分。"""
    import math

    total_chars = len(txt)
    # 同时满足字符数和时长约束，取更严格者
    n_by_chars = max(1, math.ceil(total_chars / max_chars))
    n_by_dur = max(1, math.ceil(dur / max_duration))
    n_chunks = max(n_by_chars, n_by_dur)

    chars_per_chunk = math.ceil(total_chars / n_chunks)
    dur_per_chunk = dur / n_chunks

    result: list[dict[str, Any]] = []
    pos = 0
    t = seg["start"]
    for _ in range(n_chunks):
        chunk = txt[pos: pos + chars_per_chunk]
        result.append({**seg, "text": chunk.strip(), "start": round(t, 3), "end": round(t + dur_per_chunk, 3)})
        pos += chars_per_chunk
        t += dur_per_chunk
    return result
