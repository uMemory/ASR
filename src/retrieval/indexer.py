"""检索索引构建器。

索引结构：outputs/index/
    dense.faiss        ← FAISS 稠密向量
    sparse_weights.pkl ← BGE-M3 稀疏词权重
    metadata.json      ← 元数据
    config.json        ← 索引信息
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_MIN_CHARS = 4


def _filter(segments: list[dict], min_chars: int) -> tuple[list[dict], list[str]]:
    valid, texts = [], []
    for seg in segments:
        t = seg.get("text", "").strip()
        if len(t) >= min_chars:
            valid.append(seg); texts.append(t)
    return valid, texts


def build_index(
    segments: list[dict[str, Any]], encoder: Any,
    store_path: str | Path = "./outputs/index",
    min_chars: int = DEFAULT_MIN_CHARS,
) -> tuple[dict, Any, list[dict[str, float]]]:
    import faiss
    store_path = Path(store_path)
    store_path.mkdir(parents=True, exist_ok=True)

    valid, texts = _filter(segments, min_chars)
    if not texts:
        raise ValueError("没有足够长的段来构建索引")

    logger.info("编码 %d 个文本段 ...", len(texts))
    result = encoder.encode(texts)

    dense = result.dense_vecs.astype(np.float32)
    dim = dense.shape[1]
    idx = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(dense)
    idx.add(dense)
    logger.info("FAISS: %d 向量, dim=%d", idx.ntotal, dim)

    meta = {"segments": valid, "num_segments": len(valid), "dim": dim, "texts": texts}
    sparse = result.lexical_weights

    faiss.write_index(idx, str(store_path / "dense.faiss"))
    with open(store_path / "sparse_weights.pkl", "wb") as f:
        pickle.dump(sparse, f)
    with open(store_path / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(store_path / "config.json", "w", encoding="utf-8") as f:
        json.dump({"num_segments": len(valid), "dim": dim,
                    "store_path": str(store_path.resolve())}, f, indent=2)
    logger.info("索引已保存: %s", store_path)
    return meta, idx, sparse


def load_index(
    store_path: str | Path,
) -> tuple[dict[str, Any], Any, list[dict[str, float]]]:
    import faiss
    store_path = Path(store_path)
    fp = store_path / "dense.faiss"
    if not fp.exists():
        raise FileNotFoundError(f"FAISS 索引不存在: {fp}")
    idx = faiss.read_index(str(fp))
    with open(store_path / "sparse_weights.pkl", "rb") as f:
        sparse = pickle.load(f)
    with open(store_path / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    logger.info("索引已加载: %d 段, dim=%d", meta["num_segments"], meta["dim"])
    return meta, idx, sparse
