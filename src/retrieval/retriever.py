"""多维复合检索器——5 维检索 + RRF（倒数排名融合）。

检索维度：
    1. 语义检索  → BGE-M3 dense embedding cosine similarity
    2. 关键词检索 → BGE-M3 sparse lexical weights
    3. 说话人过滤 → 精确匹配 speaker ID
    4. 时间范围过滤 → 区间查询
    5. 意图过滤   → any-of 匹配

用法：
    from src.retrieval.retriever import Retriever
    retriever = Retriever(index_path="./outputs/index", encoder=enc)
    results = retriever.search("Speaker B 的反对意见")
    # → 自动解析出 speaker=B + intent=反对 + 语义搜索
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

# RRF 默认参数
RRF_K = 60
DEFAULT_WEIGHTS = {
    "semantic": 1.0,
    "keyword": 1.0,
    "speaker": 1.5,
    "time": 1.0,
    "intent": 1.2,
}
DEFAULT_TOP_K = 20


class Retriever:
    """多维复合检索器。

    Parameters
    ----------
    index_path : str or Path
        索引存储目录（含 dense.faiss / metadata.json / sparse_weights.pkl）。
    encoder : EmbeddingEncoder
        已加载的 BGE-M3 编码器。
    rrf_k : int
        RRF 融合常数（默认 60）。
    weights : dict
        各维度权重。
    """

    def __init__(
        self,
        index_path: str | Path,
        encoder: Any,
        rrf_k: int = RRF_K,
        weights: dict[str, float] | None = None,
    ) -> None:
        self.index_path = Path(index_path)
        self.encoder = encoder
        self.rrf_k = rrf_k
        self.weights = weights or DEFAULT_WEIGHTS

        self._faiss: Any = None
        self._meta: dict[str, Any] = {}
        self._segments: list[dict[str, Any]] = []
        self._sparse: list[dict[str, float]] = []
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        from .indexer import load_index
        self._meta, self._faiss, self._sparse = load_index(self.index_path)
        self._segments = self._meta.get("segments", [])
        self._loaded = True
        logger.info("Retriever 就绪: %d 个段", len(self._segments))

    # ── 公开 API ────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        speaker: str | None = None,
        time_range: tuple[float, float] | None = None,
        intent: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """多维检索入口。

        直接指定维度约束时，不会从自然语言中解析。
        仅提供 ``query`` 时，自动解析语义+关键词+隐式约束。

        Parameters
        ----------
        query : str
            自然语言查询（如 "Speaker B 的反对意见"）。
        top_k : int
            返回结果数。
        speaker : str or None
            精确说话人约束。
        time_range : tuple or None
            (start_seconds, end_seconds) 时间约束。
        intent : list[str] or None
            意图约束列表。

        Returns
        -------
        list[dict]
            排序后的 segment 列表，每项附加 ``_score`` 和 ``_rank_sources``。
        """
        self._ensure_loaded()
        n_total = len(self._segments)
        if n_total == 0:
            return []

        # ── 1. 解析自然语言（仅当没有显式约束时） ─────────────────
        parsed = _parse_query(query)

        speaker = speaker or parsed.get("speaker")
        intent = intent or parsed.get("intent")

        # ── 2. 各维度独立排序 ──────────────────────────────────────
        rank_lists: dict[str, list[tuple[int, float]]] = {}

        # 2a. 语义检索
        rank_lists["semantic"] = self._semantic_search(query, top_k * 2)

        # 2b. 关键词检索
        rank_lists["keyword"] = self._keyword_search(query, top_k * 2)

        # 2c. 说话人过滤（转为伪排序：匹配的 rank 0，否则 rank n_total）
        if speaker:
            rank_lists["speaker"] = self._speaker_rank(speaker)

        # 2d. 时间范围过滤
        if time_range:
            rank_lists["time"] = self._time_rank(time_range[0], time_range[1])

        # 2e. 意图过滤
        if intent:
            rank_lists["intent"] = self._intent_rank(intent)

        # ── 3. RRF 融合 ─────────────────────────────────────────────
        fused = self._rrf_fuse(rank_lists, n_total)

        # ── 4. 取 top-k 并附加元数据 ─────────────────────────────────
        results: list[dict[str, Any]] = []
        for idx, score in fused[:top_k]:
            seg = dict(self._segments[idx])
            seg["_score"] = round(score, 4)
            seg["_rank_sources"] = {
                dim: [i for i, (j, _) in enumerate(ranks) if j == idx][0] + 1
                if any(j == idx for j, _ in ranks) else None
                for dim, ranks in rank_lists.items()
                if any(j == idx for j, _ in ranks)
            }
            results.append(seg)

        return results

    # ── 维度实现 ────────────────────────────────────────────────────

    def _semantic_search(
        self, query: str, top_k: int
    ) -> list[tuple[int, float]]:
        """稠密语义检索。"""
        import faiss
        q_result = self.encoder.encode_queries([query])
        q_vec = q_result.dense_vecs[0].astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(q_vec)
        scores, indices = self._faiss.search(q_vec, min(top_k, self._faiss.ntotal))
        return [(int(indices[0][i]), float(scores[0][i]))
                for i in range(len(indices[0])) if indices[0][i] >= 0]

    def _keyword_search(
        self, query: str, top_k: int
    ) -> list[tuple[int, float]]:
        """BGE-M3 稀疏关键词检索。"""
        q_result = self.encoder.encode_queries([query])
        q_weights = q_result.lexical_weights[0]
        if not q_weights:
            return []

        scores: list[tuple[int, float]] = []
        for i, doc_weights in enumerate(self._sparse):
            score = self.encoder.compute_lexical_score(q_weights, doc_weights)
            if score > 0:
                scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _speaker_rank(self, speaker: str) -> list[tuple[int, float]]:
        """说话人精确匹配：匹配的 rank=0，否则很大的 rank。"""
        results: list[tuple[int, float]] = []
        for i, seg in enumerate(self._segments):
            if seg.get("speaker", "") == speaker:
                results.append((i, 1.0))
            else:
                results.append((i, -1.0))  # 负分 → 排在末尾
        return results

    def _time_rank(
        self, t_start: float, t_end: float
    ) -> list[tuple[int, float]]:
        """时间范围过滤：越靠近区间中心，分数越高。"""
        t_mid = (t_start + t_end) / 2
        t_span = max(t_end - t_start, 1.0)
        results: list[tuple[int, float]] = []
        for i, seg in enumerate(self._segments):
            s = seg.get("start", 0)
            e = seg.get("end", s + 0.1)
            overlap = max(0.0, min(e, t_end) - max(s, t_start))
            if overlap <= 0:
                results.append((i, -1.0))
            else:
                # 距离区间中心越近分数越高
                seg_mid = (s + e) / 2
                dist = abs(seg_mid - t_mid) / t_span
                results.append((i, 1.0 - min(dist, 0.5)))
        return results

    def _intent_rank(self, intents: list[str]) -> list[tuple[int, float]]:
        """意图 any-of 匹配。"""
        intents_set = set(intents)
        results: list[tuple[int, float]] = []
        for i, seg in enumerate(self._segments):
            seg_intent = seg.get("intent", "")
            if isinstance(seg_intent, list):
                matched = any(v in intents_set for v in seg_intent)
            else:
                matched = seg_intent in intents_set
            results.append((i, 1.0 if matched else -1.0))
        return results

    def _rrf_fuse(
        self,
        rank_lists: dict[str, list[tuple[int, float]]],
        n_total: int,
    ) -> list[tuple[int, float]]:
        """倒数排名融合（RRF）。

        每个维度的排序结果按 rank 取倒数加权，最终排序靠前的为最佳结果。
        """
        scores: dict[int, float] = {}
        for dim, ranks in rank_lists.items():
            w = self.weights.get(dim, 1.0)
            for rank_pos, (idx, _) in enumerate(ranks):
                rrf_score = w / (self.rrf_k + rank_pos + 1)
                scores[idx] = scores.get(idx, 0.0) + rrf_score

        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results


# ── 自然语言查询解析 ──────────────────────────────────────────────────

def _parse_query(query: str) -> dict[str, Any]:
    """从自然语言查询中提取结构化约束。

    支持的隐式约束：
    - "Speaker X 的..." → speaker=X
    - "反对意见" / "提问" → intent=反对/提问
    - "后半段" / "前30秒" → time 范围（相对）
    - "包含'预算'的" → 交给关键词检索处理

    返回 dict，可能含 speaker / intent / time_range。
    """
    result: dict[str, Any] = {}

    # 说话人识别："Speaker_00 的反对意见" / "Speaker B 的发言"
    m = re.search(r"Speaker[_ ](\S+)", query, re.IGNORECASE)
    if m:
        result["speaker"] = f"SPEAKER_{m.group(1).zfill(2)}"

    # 意图识别：从中文意图标签中匹配
    intent_map = {
        "反对": "反对", "不同意": "反对", "质疑": "反对",
        "提问": "提问", "问": "提问", "询问": "提问",
        "同意": "同意", "赞同": "同意", "认可": "同意",
        "提议": "提议", "建议": "提议", "提出": "提议",
        "总结": "总结", "归纳": "总结",
        "命令": "命令", "要求": "命令", "指令": "命令",
        "澄清": "澄清", "解释": "澄清",
        "确认": "确认",
        "打断": "打断", "插话": "打断",
        "寒暄": "寒暄", "闲聊": "寒暄",
        "回应": "回应",
    }
    for kw, intent_val in intent_map.items():
        if kw in query:
            result.setdefault("intent", []).append(intent_val)

    # 时间识别："后半段" / "前半段" / "前30秒" / "后1分钟"
    m = re.search(r"([前后])\s*(\d+)\s*([秒分])", query)
    if m:
        direction = m.group(1)
        num = int(m.group(2))
        unit = m.group(3)
        seconds = num if unit == "秒" else num * 60
        if direction == "前":
            result["time_range"] = (0, seconds)

    return result
