"""评估已保存转写 JSON 上的检索效果。

脚本不重新运行 ASR，只读取保存的转写片段构建索引，并执行链路自检
和模拟用户查询评估。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.retrieval.indexer import build_index
from src.retrieval.retriever import Retriever


def _load_segments(result_dirs: list[Path]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for ds_root in result_dirs:
        dataset = ds_root.name
        for fp in sorted(ds_root.glob("*.json")):
            if fp.name.startswith("manual_corrected"):
                continue
            data = json.loads(fp.read_text(encoding="utf-8"))
            for i, seg in enumerate(data.get("segments", [])):
                text = (seg.get("text") or "").strip()
                if len(text) < 4:
                    continue
                item = dict(seg)
                item["_uid"] = f"{dataset}/{fp.name}#{i}"
                item["_dataset"] = dataset
                item["_file"] = fp.name
                segments.append(item)
    return segments


class _EncodingResult:
    def __init__(self, dense_vecs: np.ndarray, lexical_weights: list[dict[str, float]]) -> None:
        self.dense_vecs = dense_vecs.astype(np.float32)
        self.lexical_weights = lexical_weights


class SimpleHashEncoder:
    """用于检索链路调试的小型确定性编码器。

    它不是 BGE-M3 的替代品，只用于在无法加载大模型时验证索引构建、
    RRF 融合、关键词匹配和 Top-K 返回流程。
    """

    def __init__(self, dim: int = 384) -> None:
        self.dim = dim

    def load(self) -> "SimpleHashEncoder":
        return self

    def unload(self) -> None:
        return None

    def encode(self, texts: list[str], batch_size: int = 32, max_length: int = 512) -> _EncodingResult:
        dense: list[np.ndarray] = []
        sparse: list[dict[str, float]] = []
        for text in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            weights: dict[str, float] = {}
            for tok in self._tokens(text):
                h = hash(tok) % self.dim
                vec[h] += 1.0
                weights[str(h)] = weights.get(str(h), 0.0) + 1.0
            norm = float(np.linalg.norm(vec))
            if norm > 0:
                vec /= norm
            dense.append(vec)
            sparse.append(weights)
        return _EncodingResult(np.vstack(dense), sparse)

    def encode_queries(self, queries: list[str], batch_size: int = 8, max_length: int = 256) -> _EncodingResult:
        return self.encode(queries, batch_size=batch_size, max_length=max_length)

    def compute_lexical_score(self, query_weights: dict[str, float], doc_weights: dict[str, float]) -> float:
        return sum(qw * doc_weights.get(tok, 0.0) for tok, qw in query_weights.items())

    @staticmethod
    def _tokens(text: str) -> list[str]:
        text = text.lower().strip()
        tokens: list[str] = []
        word = []
        for ch in text:
            if ch.isascii() and ch.isalnum():
                word.append(ch)
                continue
            if word:
                tokens.append("".join(word))
                word.clear()
            if "\u4e00" <= ch <= "\u9fff":
                tokens.append(ch)
        if word:
            tokens.append("".join(word))
        # Character bigrams help partial Chinese queries.
        cjk = [t for t in tokens if len(t) == 1 and "\u4e00" <= t <= "\u9fff"]
        tokens.extend(a + b for a, b in zip(cjk, cjk[1:]))
        return tokens


def _zh_partial(text: str) -> str:
    return text[: min(18, max(6, len(text) // 2))]


def _en_partial(text: str) -> str:
    words = text.split()
    return " ".join(words[: min(8, max(3, len(words) // 2))]) or text[:32]


def _sample_cases(pool: list[dict[str, Any]], n: int, mode: str) -> list[dict[str, Any]]:
    chosen = random.sample(pool, min(n, len(pool)))
    cases: list[dict[str, Any]] = []
    for seg in chosen:
        text = (seg.get("text") or "").strip()
        if mode == "exact":
            query = text
        elif seg.get("_dataset") == "aishell1":
            query = _zh_partial(text)
        else:
            query = _en_partial(text)
        cases.append(
            {
                "query": query,
                "target_uid": seg["_uid"],
                "target_text": text,
                "dataset": seg["_dataset"],
                "mode": mode,
            }
        )
    return cases


def _aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        return {}
    return {
        "queries": len(items),
        "hit_at_1": round(sum(bool(x["hit_at_1"]) for x in items) / len(items), 4),
        "hit_at_5": round(sum(bool(x["hit_at_5"]) for x in items) / len(items), 4),
        "mrr_at_5": round(
            sum((1 / x["rank"]) if x["rank"] else 0 for x in items) / len(items), 4
        ),
        "avg_latency_ms": round(mean(x["latency_ms"] for x in items), 2),
    }


def _contains_all(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return all(term.lower() in lowered for term in terms)


def _overlaps(seg: dict[str, Any], time_range: tuple[float, float]) -> bool:
    start = float(seg.get("start", 0.0))
    end = float(seg.get("end", start))
    return max(0.0, min(end, time_range[1]) - max(start, time_range[0])) > 0


def _intent_matches_case(seg_intent: Any, expected: list[str]) -> bool:
    if not expected:
        return True
    if isinstance(seg_intent, list):
        values = {str(v) for v in seg_intent}
    elif seg_intent:
        values = {str(seg_intent)}
    else:
        values = set()
    return bool(values.intersection(expected))


def _is_relevant(hit: dict[str, Any], case: dict[str, Any]) -> bool:
    if case.get("dataset") and hit.get("_dataset") != case["dataset"]:
        return False
    if case.get("file") and hit.get("_file") != case["file"]:
        return False
    if case.get("speaker") and hit.get("speaker") != case["speaker"]:
        return False
    if case.get("time_range") and not _overlaps(hit, tuple(case["time_range"])):
        return False
    if case.get("intent") and not _intent_matches_case(hit.get("intent"), case["intent"]):
        return False
    text = (hit.get("text") or "").strip()
    any_groups = case.get("any_term_groups") or []
    return any(_contains_all(text, group) for group in any_groups)


def _realistic_query_cases() -> list[dict[str, Any]]:
    """更接近真实用户搜索方式的小规模手工查询集。

    与 exact/partial 自检不同，这些查询不是直接复制转写文本。命中结果
    需要属于期望数据集，并包含人工指定的证据词组。
    """
    return [
        {
            "query": "哪些政策对楼市影响最大",
            "dimension": "semantic",
            "dataset": "aishell1",
            "any_term_groups": [["楼市", "影响", "政策"], ["影响最大", "政策"]],
        },
        {
            "query": "哪些城市取消或放松了限购",
            "dimension": "keyword",
            "dataset": "aishell1",
            "any_term_groups": [["限购", "城市"], ["取消", "限购"], ["放松", "限购"]],
        },
        {
            "query": "首套房贷款和房贷政策有什么变化",
            "dimension": "semantic+keyword",
            "dataset": "aishell1",
            "any_term_groups": [["首套房贷款"], ["房贷政策"], ["贷款", "普通商品住房"]],
        },
        {
            "query": "央行降息对房地产有什么影响",
            "dimension": "semantic+keyword",
            "dataset": "aishell1",
            "any_term_groups": [["央行", "降息"], ["降息", "市场"]],
        },
        {
            "query": "海外房地产投资规模超过三百亿美元",
            "dimension": "semantic+keyword",
            "dataset": "aishell1",
            "any_term_groups": [["海外房地产投资"], ["房地产投资", "300亿美元"], ["超过300亿美元"]],
        },
        {
            "query": "支持改善居住条件的政策建议",
            "dimension": "intent+semantic",
            "dataset": "aishell1",
            "intent": ["提议"],
            "any_term_groups": [["改善", "住房"], ["商品房"], ["改善型需求"]],
        },
        {
            "query": "前10秒内关于首套房贷款的内容",
            "dimension": "time+keyword",
            "dataset": "aishell1",
            "time_range": (0.0, 10.0),
            "any_term_groups": [["首套房贷款"], ["贷款", "住房"]],
        },
        {
            "query": "find the passage about stew, carrots, potatoes and mutton for dinner",
            "display_query": "查找关于晚餐炖菜、胡萝卜、土豆和羊肉的片段",
            "dimension": "semantic",
            "dataset": "librispeech",
            "any_term_groups": [["stew", "carrots", "potatoes"], ["mutton", "dinner"]],
        },
        {
            "query": "who is called the Apostle of the Indies",
            "display_query": "谁被称为印度使徒",
            "dimension": "semantic",
            "dataset": "librispeech",
            "any_term_groups": [["apostle", "indies"]],
        },
        {
            "query": "Saint Francis Xavier feast day and retreat",
            "display_query": "圣方济各沙勿略的瞻礼日和静修",
            "dimension": "keyword",
            "dataset": "librispeech",
            "any_term_groups": [["francis", "xavier"], ["retreat", "feast"]],
        },
        {
            "query": "kingdom of heaven and the meek in the beatitudes",
            "display_query": "八福中天国和温顺者的问题",
            "dimension": "semantic+keyword",
            "dataset": "librispeech",
            "any_term_groups": [["kingdom", "heaven"], ["beatitude", "meek"]],
        },
        {
            "query": "Friday afternoon confession after beads",
            "display_query": "周五下午念珠后告解",
            "dimension": "time+keyword",
            "dataset": "librispeech",
            "time_range": (0.0, 20.0),
            "any_term_groups": [["friday", "confession"], ["afternoon", "beads"]],
        },
        {
            "query": "英文里有人问洗礼是否有效的问题",
            "display_query": "英文里有人问洗礼是否有效的问题",
            "dimension": "cross_language+intent",
            "dataset": "librispeech",
            "intent": ["question"],
            "any_term_groups": [["baptism", "child"], ["water", "words"]],
        },
        {
            "query": "Speaker 1 says good night",
            "display_query": "说话人01说晚安",
            "dimension": "speaker+keyword",
            "dataset": "librispeech",
            "speaker": "SPEAKER_01",
            "any_term_groups": [["good", "night"], ["husband"]],
        },
        {
            "query": "SPEAKER_00 提出的关于 kingdom of heaven 的问题",
            "display_query": "SPEAKER_00 提出的关于天国的问题",
            "dimension": "speaker+intent+cross_language",
            "dataset": "librispeech",
            "speaker": "SPEAKER_00",
            "intent": ["question"],
            "any_term_groups": [["kingdom", "heaven"], ["beatitude", "meek"]],
        },
        {
            "query": "前15秒内关于 stolen pound 的问题",
            "display_query": "前15秒内关于偷走一英镑的问题",
            "dimension": "time+intent+keyword",
            "dataset": "librispeech",
            "time_range": (0.0, 15.0),
            "intent": ["question"],
            "any_term_groups": [["stolen", "pound"], ["fortune", "give back"]],
        },
        {
            "query": "Elizabeth Warren biggest threat New Hampshire Biden",
            "display_query": "ahnss 中关于 Elizabeth Warren 是最大威胁的讨论",
            "dimension": "speaker+keyword",
            "dataset": "test_results",
            "file": "ahnss.wav.json",
            "speaker": "SPEAKER_03",
            "any_term_groups": [["biggest", "threat"], ["Elizabeth", "Warren"], ["New", "Hampshire"]],
        },
        {
            "query": "Bernie Sanders heart attack healthy viable race",
            "display_query": "ahnss 中 SPEAKER_02 谈 Bernie Sanders 健康和参选",
            "dimension": "speaker+keyword",
            "dataset": "test_results",
            "file": "ahnss.wav.json",
            "speaker": "SPEAKER_02",
            "any_term_groups": [["Bernie", "healthy"], ["Sanders", "race"], ["heart", "attack"]],
        },
        {
            "query": "前两分钟里关于 Warren 和 Bernie 同台的评论",
            "display_query": "ahnss 前两分钟关于 Warren 和 Bernie 同台的评论",
            "dimension": "time+speaker+keyword",
            "dataset": "test_results",
            "file": "ahnss.wav.json",
            "speaker": "SPEAKER_03",
            "time_range": (0.0, 140.0),
            "any_term_groups": [["Bernie", "Liz"], ["same", "stage"], ["lockstep"]],
        },
        {
            "query": "SPEAKER_00 对 Kamala Harris 和加州记录的评价",
            "display_query": "ahnss 中 SPEAKER_00 对 Kamala Harris 和加州记录的评价",
            "dimension": "speaker+semantic",
            "dataset": "test_results",
            "file": "ahnss.wav.json",
            "speaker": "SPEAKER_00",
            "any_term_groups": [["Kamala", "Harris"], ["Californians"], ["California", "record"], ["prosecutor"]],
        },
        {
            "query": "Democratic debate private health insurance government plan",
            "display_query": "cjfer 中关于私人医保和政府医保计划的提问",
            "dimension": "intent+keyword",
            "dataset": "test_results",
            "file": "cjfer.wav.json",
            "intent": ["question"],
            "any_term_groups": [["private", "health", "insurance"], ["government-run", "plan"]],
        },
        {
            "query": "undocumented immigrants coverage raise your hand",
            "display_query": "cjfer 中关于无证移民医保覆盖的提问",
            "dimension": "speaker+intent+keyword",
            "dataset": "test_results",
            "file": "cjfer.wav.json",
            "speaker": "SPEAKER_00",
            "intent": ["question", "command"],
            "any_term_groups": [["undocumented", "immigrants"], ["Raise", "hand"], ["coverage"]],
        },
        {
            "query": "climate crisis 12 years irreparable damage",
            "display_query": "cjfer 中关于气候危机和 12 年窗口的讨论",
            "dimension": "keyword+time",
            "dataset": "test_results",
            "file": "cjfer.wav.json",
            "time_range": (60.0, 90.0),
            "any_term_groups": [["climate", "crisis"], ["12", "years"], ["irreparable", "damage"]],
        },
        {
            "query": "Speaker 07 asks Dana political suicide",
            "display_query": "cjfer 中 SPEAKER_07 询问 Democrats 是否政治自杀",
            "dimension": "speaker+intent+keyword",
            "dataset": "test_results",
            "file": "cjfer.wav.json",
            "speaker": "SPEAKER_07",
            "intent": ["question", "transition"],
            "any_term_groups": [["political", "suicide"], ["Dana", "Democrats"]],
        },
        {
            "query": "R8007 备选方案 房租 交通 公交",
            "display_query": "R8007 中关于备选方案、房租和交通的早期讨论",
            "dimension": "time+speaker+keyword",
            "dataset": "test_results",
            "file": "R8007_M8010_N_SPK8050.wav.json",
            "speaker": "SPEAKER_02",
            "time_range": (20.0, 120.0),
            "any_term_groups": [["备选方案"], ["房租"], ["交通"], ["公交"]],
        },
        {
            "query": "SPEAKER_01 大会议室 小会议室 工位",
            "display_query": "R8007 中 SPEAKER_01 讨论工位和会议室配置",
            "dimension": "speaker+keyword",
            "dataset": "test_results",
            "file": "R8007_M8010_N_SPK8050.wav.json",
            "speaker": "SPEAKER_01",
            "any_term_groups": [["工位"], ["会议室"], ["小会议室"], ["大会议室"]],
        },
        {
            "query": "吸烟室 地毯 刷漆 仿瓷",
            "display_query": "R8007 中关于吸烟室、地毯、刷漆和仿瓷的装修建议",
            "dimension": "semantic+keyword",
            "dataset": "test_results",
            "file": "R8007_M8010_N_SPK8050.wav.json",
            "any_term_groups": [["吸烟室"], ["地毯"], ["刷漆"], ["仿瓷"]],
        },
        {
            "query": "超市 生鲜 晚上促销 剩菜 引流",
            "display_query": "S_R003 中关于生鲜晚上促销和引流的建议",
            "dimension": "intent+keyword",
            "dataset": "test_results",
            "file": "S_R003S01C01.flac.json",
            "intent": ["提议"],
            "any_term_groups": [["晚上", "促销"], ["引流"], ["价格降低"], ["开门", "活动"]],
        },
        {
            "query": "超市烟草部 低价烟 高档烟 黄鹤楼",
            "display_query": "L_R004 中关于烟草部低价烟和高档烟的汇报",
            "dimension": "keyword",
            "dataset": "test_results",
            "file": "L_R004S01C01.flac.json",
            "any_term_groups": [["低价烟"], ["高价烟"], ["黄鹤楼"], ["高档烟"]],
        },
        {
            "query": "L_R004 海鲜 花蛤 鲈鱼 桂鱼 新鲜度",
            "display_query": "L_R004 中关于海鲜花蛤、鲈鱼和新鲜度的讨论",
            "dimension": "speaker+keyword",
            "dataset": "test_results",
            "file": "L_R004S01C01.flac.json",
            "speaker": "SPEAKER_05",
            "any_term_groups": [["花蛤"], ["鲈鱼"], ["桂鱼"], ["新鲜度"]],
        },
        {
            "query": "S_R003 豆芽 土豆 胡萝卜 白萝卜 卖得好",
            "display_query": "S_R003 中关于豆芽、土豆和萝卜等生鲜销售情况",
            "dimension": "time+keyword",
            "dataset": "test_results",
            "file": "S_R003S01C01.flac.json",
            "time_range": (200.0, 260.0),
            "any_term_groups": [["豆芽"], ["土豆"], ["胡萝卜"], ["白萝卜"]],
        },
        {
            "query": "手机 18到35岁 年轻化 电商直播带货 双11",
            "display_query": "R8001 中关于手机年轻化定位和直播带货促销",
            "dimension": "semantic+keyword",
            "dataset": "test_results",
            "file": "R8001_M8004_MS801.wav.json",
            "any_term_groups": [["18到35岁"], ["年轻化"], ["直播带货"], ["双11"]],
        },
        {
            "query": "R8001 明星代言 网红 直播 平台",
            "display_query": "R8001 中关于明星代言、网红和直播平台的讨论",
            "dimension": "speaker+keyword",
            "dataset": "test_results",
            "file": "R8001_M8004_MS801.wav.json",
            "speaker": "SPEAKER_03",
            "any_term_groups": [["明星代言"], ["网红"], ["直播"], ["平台"]],
        },
        {
            "query": "cjfer Marianne Williamson return to love frontrunner",
            "display_query": "cjfer 中关于 Marianne Williamson 和 Return to Love 的评论",
            "dimension": "speaker+keyword",
            "dataset": "test_results",
            "file": "cjfer.wav.json",
            "speaker": "SPEAKER_09",
            "any_term_groups": [["Marianne", "Williamson"], ["return", "love"], ["frontrunner"]],
        },
    ]


def _evaluate_realistic_queries(
    retriever: Retriever,
    segments: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    evaluated: list[dict[str, Any]] = []
    for case in _realistic_query_cases():
        if not any(_is_relevant(seg, case) for seg in segments):
            continue
        t0 = time.time()
        hits = retriever.search(
            case["query"],
            top_k=top_k,
            speaker=case.get("speaker"),
            time_range=case.get("time_range"),
            intent=case.get("intent"),
        )
        latency_ms = (time.time() - t0) * 1000
        rank = None
        for idx, hit in enumerate(hits, 1):
            if _is_relevant(hit, case):
                rank = idx
                break
        evaluated.append(
            {
                "query": case["query"],
                "display_query": case.get("display_query", case["query"]),
                "dimension": case["dimension"],
                "dataset": case["dataset"],
                "file": case.get("file"),
                "speaker": case.get("speaker"),
                "time_range": case.get("time_range"),
                "intent": case.get("intent"),
                "latency_ms": round(latency_ms, 2),
                "rank": rank,
                "hit_at_1": rank == 1,
                "hit_at_5": rank is not None and rank <= top_k,
                "evidence_terms": case["any_term_groups"],
                "hits": [
                    {
                        "dataset": h.get("_dataset"),
                        "file": h.get("_file"),
                        "speaker": h.get("speaker"),
                        "text": h.get("text"),
                        "score": h.get("_score"),
                        "relevant": _is_relevant(h, case),
                    }
                    for h in hits
                ],
            }
        )
    return evaluated


_DIMENSION_LABELS = {
    "semantic": "语义",
    "keyword": "关键词",
    "semantic+keyword": "语义+关键词",
    "intent+semantic": "意图+语义",
    "time+keyword": "时间范围+关键词",
    "cross_language+intent": "跨语言+意图",
    "speaker+keyword": "说话人+关键词",
    "speaker+intent+cross_language": "说话人+意图+跨语言",
    "time+intent+keyword": "时间范围+意图+关键词",
    "time+speaker+keyword": "时间范围+说话人+关键词",
    "speaker+semantic": "说话人+语义",
    "intent+keyword": "意图+关键词",
    "speaker+intent+keyword": "说话人+意图+关键词",
    "keyword+time": "关键词+时间范围",
}


def _dimension_label(value: str) -> str:
    return _DIMENSION_LABELS.get(value, value)


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# 检索评估结果\n\n")
    lines.append(f"结果来源：`{summary['source']}`\n\n")
    lines.append(f"编码器模式：`{summary['encoder']}`\n\n")
    lines.append(f"索引片段数：{summary['index_segments']}\n\n")
    lines.append(f"索引构建耗时：{summary['index_build_s']}s\n\n")
    lines.append("## 链路自检指标\n\n")
    lines.append("| 分组 | 查询数 | Hit@1 | Hit@5 | MRR@5 | 平均延迟 |\n")
    lines.append("|---|---:|---:|---:|---:|---:|\n")
    rows = [
        ("整体", summary["overall"]),
        ("AISHELL-1", summary["by_dataset"]["aishell1"]),
        ("LibriSpeech", summary["by_dataset"]["librispeech"]),
        ("完整原文查询", summary["by_mode"]["exact"]),
        ("片段原文查询", summary["by_mode"]["partial"]),
    ]
    for name, stats in rows:
        lines.append(
            f"| {name} | {stats.get('queries', 0)} | "
            f"{stats.get('hit_at_1', 0) * 100:.1f}% | "
            f"{stats.get('hit_at_5', 0) * 100:.1f}% | "
            f"{stats.get('mrr_at_5', 0):.3f} | "
            f"{stats.get('avg_latency_ms', 0):.1f} ms |\n"
        )
    lines.append(
        "\n这部分是基于已保存转写文本的检索链路自检。完整原文查询和片段原文查询"
        "使用原文或原文片段作为查询，因此目标片段是已知的。该指标能验证索引、"
        "编码、RRF 融合和 Top-K 返回流程，但不等同于任意真实用户问题的人工相关性基准。\n"
    )
    realistic = summary.get("realistic")
    if realistic:
        stats = realistic["metrics"]
        lines.append("\n## 模拟用户查询指标\n\n")
        lines.append("| 查询数 | Hit@1 | Hit@5 | MRR@5 | 平均延迟 |\n")
        lines.append("|---:|---:|---:|---:|---:|\n")
        lines.append(
            f"| {stats.get('queries', 0)} | "
            f"{stats.get('hit_at_1', 0) * 100:.1f}% | "
            f"{stats.get('hit_at_5', 0) * 100:.1f}% | "
            f"{stats.get('mrr_at_5', 0):.3f} | "
            f"{stats.get('avg_latency_ms', 0):.1f} ms |\n"
        )
        lines.append(
            "\n这些查询是根据转写内容人工设计的模拟用户问题。相关性判断依据数据集范围"
            "和预设证据词组，而不是直接匹配复制出来的转写片段。\n"
        )
        by_dimension = realistic.get("by_dimension", {})
        if by_dimension:
            lines.append("\n### 按检索维度分组\n\n")
            lines.append("| 检索维度 | 查询数 | Hit@1 | Hit@5 | MRR@5 | 平均延迟 |\n")
            lines.append("|---|---:|---:|---:|---:|---:|\n")
            for dim, dim_stats in sorted(by_dimension.items()):
                lines.append(
                    f"| {_dimension_label(dim)} | {dim_stats.get('queries', 0)} | "
                    f"{dim_stats.get('hit_at_1', 0) * 100:.1f}% | "
                    f"{dim_stats.get('hit_at_5', 0) * 100:.1f}% | "
                    f"{dim_stats.get('mrr_at_5', 0):.3f} | "
                    f"{dim_stats.get('avg_latency_ms', 0):.1f} ms |\n"
                )
        for item in realistic["cases"]:
            lines.append(f"\n### {item.get('display_query') or item['query']}\n\n")
            constraints = [f"数据集={item['dataset']}", f"检索维度={_dimension_label(item['dimension'])}"]
            if item.get("file"):
                constraints.append(f"文件={item['file']}")
            if item.get("speaker"):
                constraints.append(f"说话人={item['speaker']}")
            if item.get("time_range"):
                constraints.append(f"时间范围={tuple(item['time_range'])}")
            if item.get("intent"):
                constraints.append(f"意图={item['intent']}")
            lines.append(
                f"约束条件：`{'; '.join(constraints)}`；命中排名：`{item['rank']}`；"
                f"延迟：{item['latency_ms']} ms\n\n"
            )
            for i, hit in enumerate(item["hits"], 1):
                text = (hit.get("text") or "").replace("\n", " ")
                if len(text) > 120:
                    text = text[:117] + "..."
                mark = "✓" if hit.get("relevant") else "-"
                lines.append(
                    f"{i}. {mark} `{hit.get('dataset')}` `{hit.get('speaker')}` {text}\n"
                )
    lines.append("\n## 手工查询示例\n")
    for item in summary["manual_queries"]:
        lines.append(f"\n### {item.get('display_query') or item['query']}\n\n")
        lines.append(f"延迟：{item['latency_ms']} ms\n\n")
        for i, hit in enumerate(item["hits"], 1):
            text = (hit.get("text") or "").replace("\n", " ")
            if len(text) > 120:
                text = text[:117] + "..."
            lines.append(
                f"{i}. `{hit.get('dataset')}` `{hit.get('speaker')}` {text}\n"
            )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        default="outputs/batch_eval_llm/results",
        help="Root containing dataset subdirectories with saved JSON files.",
    )
    parser.add_argument("--datasets", nargs="+", default=["aishell1", "librispeech"])
    parser.add_argument(
        "--extra-result-dirs",
        nargs="*",
        default=[],
        help="Additional directories containing saved JSON files directly, e.g. tests/test_results.",
    )
    parser.add_argument("--samples-per-group", type=int, default=20)
    parser.add_argument("--output-dir", default="outputs/retrieval_eval")
    parser.add_argument(
        "--encoder",
        choices=["simple", "bge"],
        default="simple",
        help="simple verifies retrieval pipeline; bge uses configured BGE-M3 model.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    root = Path(__file__).resolve().parents[1]
    result_root = (root / args.result_root).resolve()
    output_dir = (root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index_dir = output_dir / "index"

    result_dirs = [result_root / ds for ds in args.datasets]
    result_dirs += [(root / item).resolve() for item in args.extra_result_dirs]
    segments = _load_segments(result_dirs)
    if not segments:
        raise SystemExit(f"No saved segments found under {result_root}")

    if args.encoder == "bge":
        from src.retrieval.embedding import EmbeddingEncoder
        from src.utils.config import get_model_config

        model_cfg = get_model_config()
        encoder = EmbeddingEncoder(
            model_path=model_cfg["embedding"]["model"],
            device=model_cfg["device"],
        )
    else:
        encoder = SimpleHashEncoder()
    encoder.load()

    print(f"[retrieval-eval] loaded segments: {len(segments)}", flush=True)
    start = time.time()
    meta, _, _ = build_index(segments, encoder, store_path=index_dir)
    index_build_s = time.time() - start
    retriever = Retriever(index_path=index_dir, encoder=encoder)

    zh = [s for s in segments if s.get("_dataset") == "aishell1"]
    en = [s for s in segments if s.get("_dataset") == "librispeech"]

    cases: list[dict[str, Any]] = []
    cases += _sample_cases(zh, args.samples_per_group, "exact")
    cases += _sample_cases(en, args.samples_per_group, "exact")
    cases += _sample_cases(zh, args.samples_per_group, "partial")
    cases += _sample_cases(en, args.samples_per_group, "partial")

    results: list[dict[str, Any]] = []
    latencies: list[float] = []
    for case in cases:
        t0 = time.time()
        hits = retriever.search(case["query"], top_k=5)
        latency_ms = (time.time() - t0) * 1000
        latencies.append(latency_ms)
        rank = None
        for idx, hit in enumerate(hits, 1):
            same_uid = hit.get("_uid") == case["target_uid"]
            same_text = (hit.get("text") or "").strip() == case["target_text"]
            if same_uid or same_text:
                rank = idx
                break
        results.append(
            {
                **case,
                "latency_ms": round(latency_ms, 2),
                "rank": rank,
                "hit_at_1": rank == 1,
                "hit_at_5": rank is not None and rank <= 5,
                "top1_text": (hits[0].get("text") if hits else None),
            }
        )

    manual_queries = [
        {"query": "房地产市场政策", "display_query": "房地产市场政策", "dataset_hint": "aishell1"},
        {"query": "限购 普通住宅", "display_query": "限购与普通住宅", "dataset_hint": "aishell1"},
        {"query": "kingdom of heaven", "display_query": "天国相关片段", "dataset_hint": "librispeech"},
        {"query": "Saint Francis Xavier", "display_query": "圣方济各沙勿略相关片段", "dataset_hint": "librispeech"},
    ]
    manual: list[dict[str, Any]] = []
    for query in manual_queries:
        t0 = time.time()
        hits = retriever.search(query["query"], top_k=3)
        latency_ms = (time.time() - t0) * 1000
        manual.append(
            {
                **query,
                "latency_ms": round(latency_ms, 2),
                "hits": [
                    {
                        "dataset": h.get("_dataset"),
                        "file": h.get("_file"),
                        "speaker": h.get("speaker"),
                        "text": h.get("text"),
                        "score": h.get("_score"),
                    }
                    for h in hits
                ],
            }
        )

    realistic_cases = _evaluate_realistic_queries(retriever, segments, top_k=5)
    dimensions = sorted({case["dimension"] for case in realistic_cases})

    summary = {
        "source": str(result_root.relative_to(root)),
        "encoder": args.encoder,
        "note": (
            "simple encoder verifies the retrieval pipeline only; use --encoder bge "
            "for configured BGE-M3 metrics when the local transformers stack is stable."
        ),
        "index_segments": meta["num_segments"],
        "index_build_s": round(index_build_s, 2),
        "avg_latency_ms_all": round(mean(latencies), 2) if latencies else 0,
        "overall": _aggregate(results),
        "by_dataset": {
            "aishell1": _aggregate([x for x in results if x["dataset"] == "aishell1"]),
            "librispeech": _aggregate([x for x in results if x["dataset"] == "librispeech"]),
        },
        "by_mode": {
            "exact": _aggregate([x for x in results if x["mode"] == "exact"]),
            "partial": _aggregate([x for x in results if x["mode"] == "partial"]),
        },
        "realistic": {
            "metrics": _aggregate(realistic_cases),
            "by_dimension": {
                dim: _aggregate([x for x in realistic_cases if x["dimension"] == dim])
                for dim in dimensions
            },
            "cases": realistic_cases,
        },
        "manual_queries": manual,
        "cases": results,
    }

    json_path = output_dir / "retrieval_eval_summary.json"
    md_path = output_dir / "retrieval_eval_summary.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(summary, md_path)
    encoder.unload()

    printable = {
        "index_segments": summary["index_segments"],
        "index_build_s": summary["index_build_s"],
        "overall": summary["overall"],
        "by_dataset": summary["by_dataset"],
        "by_mode": summary["by_mode"],
        "realistic": summary["realistic"]["metrics"],
    }
    print(json.dumps(printable, ensure_ascii=False, indent=2), flush=True)
    print(f"[saved] {json_path}", flush=True)
    print(f"[saved] {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
