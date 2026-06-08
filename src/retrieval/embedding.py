"""BGE-M3 编码器——稠密语义 + 稀疏关键词（原生 BGE-M3 sparse weights）。

纯 transformers 实现，不使用 FlagEmbedding 或 sentence-transformers。
稠密向量：XLMRobertaModel CLS token + L2 归一化
稀疏权重：sparse_linear.pt（nn.Linear(1024,1) + ReLU）
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Transformers 4.51 imports generation helpers that optionally import sklearn.
# In this Windows environment, sklearn -> pandas -> pyarrow can crash in native
# extension loading, while BGE-M3 embedding does not need sklearn at all.
import transformers.utils.import_utils as _tf_import_utils
_tf_import_utils._sklearn_available = False

from transformers import AutoModel, AutoTokenizer

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 特殊 token ID（XLMRoberta tokenizer）
_UNUSED_IDS = {0, 1, 2}  # <s>, <pad>, </s>


# ---------------------------------------------------------------------------
# 内部工具函数
# ---------------------------------------------------------------------------

def _resolve_path(model_path: str) -> str:
    """若 ``model_path`` 是本地存在的目录，返回绝对路径；否则原样返回（HF repo ID）。"""
    p = Path(model_path)
    if p.exists() and p.is_dir():
        return str(p.resolve())
    return model_path


def _token_weights_to_dict(
    token_weights: Tensor, input_ids: Tensor, vocab_size: int,
) -> dict[str, float]:
    """将每个 token 位置的权重转换为 {token_string: weight} 字典。

    规则：
    - 过滤特殊 token（<s>, <pad>, </s>）
    - 过滤非正权重
    - 同一 token 多次出现时取最大权重
    """
    result: dict[str, float] = {}
    for w, tid in zip(token_weights.tolist(), input_ids.tolist()):
        if tid in _UNUSED_IDS or w <= 0:
            continue
        result[str(tid)] = max(result.get(str(tid), 0.0), w)
    return result


# ---------------------------------------------------------------------------
# 编码结果
# ---------------------------------------------------------------------------

@dataclass
class EncodingResult:
    dense_vecs: np.ndarray                # (N, 1024)  float32
    lexical_weights: list[dict[str, float]]  # N 个 token→weight 映射

    def __post_init__(self) -> None:
        if self.dense_vecs.dtype != np.float32:
            self.dense_vecs = self.dense_vecs.astype(np.float32)


# ---------------------------------------------------------------------------
# 编码器
# ---------------------------------------------------------------------------

class EmbeddingEncoder:
    """BGE-M3 编码器（纯 transformers 后端）。

    Parameters
    ----------
    model_path : str or Path
        本地模型路径或 HF repo ID（如 ``BAAI/bge-m3``）。
    use_fp16 : bool
        启用 fp16 推理。
    device : str
        设备字符串 ``"cuda"`` 或 ``"cpu"``。
    """

    def __init__(
        self,
        model_path: str | Path,
        use_fp16: bool = True,
        device: str = "cuda",
    ) -> None:
        self.model_path = _resolve_path(str(model_path))
        self.use_fp16 = use_fp16 and device != "cpu" and torch.cuda.is_available()
        self.device = device if torch.cuda.is_available() else "cpu"

        # 内部状态
        self._model: nn.Module | None = None
        self._tokenizer: Any = None
        self._sparse_linear: nn.Module | None = None

    # ── 加载 / 卸载 ─────────────────────────────────────────────────

    def load(self) -> EmbeddingEncoder:
        """加载模型、分词器和稀疏线性层。"""
        if self._model is not None:
            return self

        logger.info("加载 BGE-M3: %s (device=%s, fp16=%s)",
                     self.model_path, self.device, self.use_fp16)

        dtype = torch.float16 if self.use_fp16 else torch.float32

        self._model = AutoModel.from_pretrained(
            self.model_path, torch_dtype=dtype,
        ).to(self.device).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # 加载 sparse_linear.pt（模型目录下）
        sparse_path = os.path.join(self.model_path, "sparse_linear.pt")
        if os.path.exists(sparse_path):
            hidden = self._model.config.hidden_size  # 1024
            self._sparse_linear = nn.Linear(hidden, 1, dtype=dtype).to(self.device)
            state = torch.load(sparse_path, map_location=self.device, weights_only=True)
            self._sparse_linear.load_state_dict(state)
            self._sparse_linear.eval()
            logger.debug("  稀疏线性层已加载")
        else:
            logger.warning("  未找到 sparse_linear.pt，关键词检索将不可用")

        return self

    def unload(self) -> None:
        """释放 GPU 资源。"""
        if self._model is not None:
            del self._model; self._model = None
        if self._tokenizer is not None:
            self._tokenizer = None
        if self._sparse_linear is not None:
            del self._sparse_linear; self._sparse_linear = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── 编码 ────────────────────────────────────────────────────────

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        max_length: int = 512,
    ) -> EncodingResult:
        """对文本列表编码，返回稠密向量 + 稀疏权重。"""
        self.load()
        return self._encode_impl(texts, batch_size, max_length)

    def encode_queries(
        self,
        queries: list[str],
        batch_size: int = 8,
        max_length: int = 256,
    ) -> EncodingResult:
        """查询编码（与 encode 相同，仅默认参数不同）。"""
        return self.encode(queries, batch_size=batch_size, max_length=max_length)

    def _encode_impl(
        self, texts: list[str], batch_size: int, max_length: int,
    ) -> EncodingResult:
        all_dense: list[np.ndarray] = []
        all_sparse: list[dict[str, float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            dense_i, sparse_i = self._encode_batch(batch, max_length)
            all_dense.append(dense_i)
            all_sparse.extend(sparse_i)

        dense_vecs = (
            np.concatenate(all_dense, axis=0)
            if len(all_dense) > 1
            else all_dense[0]
        )
        return EncodingResult(dense_vecs=dense_vecs, lexical_weights=all_sparse)

    @torch.no_grad()
    def _encode_batch(
        self, texts: list[str], max_length: int,
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        tok = self._tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        input_ids: Tensor = tok["input_ids"].to(self.device)
        attention_mask: Tensor = tok["attention_mask"].to(self.device)

        outputs = self._model(input_ids=input_ids, attention_mask=attention_mask)
        hidden: Tensor = outputs.last_hidden_state  # (B, S, 1024)

        # ── 稠密向量：CLS token + L2 归一化 ──
        cls_vec = hidden[:, 0, :]  # (B, 1024)
        cls_vec = F.normalize(cls_vec, p=2, dim=-1)
        dense = cls_vec.cpu().to(torch.float32).numpy()

        # ── 稀疏权重：ReLU(sparse_linear(hidden)) ──
        sparse: list[dict[str, float]] = []
        if self._sparse_linear is not None:
            token_w = torch.relu(self._sparse_linear(hidden)).squeeze(-1)  # (B, S)
            token_w = token_w.cpu().to(torch.float32)
            for b in range(token_w.size(0)):
                sparse.append(
                    _token_weights_to_dict(
                        token_w[b], input_ids[b], self._model.config.vocab_size,
                    )
                )
        else:
            sparse = [{} for _ in texts]

        return dense, sparse

    # ── 关键词匹配得分 ─────────────────────────────────────────────

    def compute_lexical_score(
        self,
        query_weights: dict[str, float],
        doc_weights: dict[str, float],
    ) -> float:
        """计算 BGE-M3 词汇匹配得分：Σ(w_q[t] × w_d[t]) for t in q ∩ d。"""
        if not query_weights or not doc_weights:
            return 0.0
        score = 0.0
        for tid, wq in query_weights.items():
            wd = doc_weights.get(tid)
            if wd is not None:
                score += wq * wd
        return score


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------

def load_encoder(cfg: dict[str, Any]) -> EmbeddingEncoder:
    """从配置字典创建编码器。"""
    return EmbeddingEncoder(
        model_path=cfg["model"],
        use_fp16=cfg.get("use_fp16", True),
        device=cfg.get("device", "cuda"),
    )
