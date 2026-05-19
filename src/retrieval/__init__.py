"""多维复合检索模块——5 维检索 + RRF 融合。

提供：
- ``EmbeddingEncoder`` — 纯 transformers BGE-M3 稠密+稀疏编码
- ``build_index`` / ``load_index`` — 索引构建/加载
- ``Retriever`` — 多维检索器
"""
from __future__ import annotations

from .embedding import EmbeddingEncoder, load_encoder
from .indexer import build_index, load_index
from .retriever import Retriever

__all__ = [
    "EmbeddingEncoder",
    "load_encoder",
    "build_index",
    "load_index",
    "Retriever",
]
