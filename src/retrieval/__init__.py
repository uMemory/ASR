"""多维复合检索模块——5 维检索 + RRF 融合。

提供：
- ``EmbeddingEncoder`` — 纯 transformers BGE-M3 稠密+稀疏编码
- ``build_index`` / ``load_index`` — 索引构建/加载
- ``Retriever`` — 多维检索器
"""
from __future__ import annotations

__all__ = [
    "EmbeddingEncoder",
    "load_encoder",
    "build_index",
    "load_index",
    "RewrittenQuery",
    "rewrite_query",
    "rewrite_query_rule_based",
    "rewrite_query_with_llm",
    "Retriever",
]


def __getattr__(name: str):
    """Lazy imports keep query rewriting lightweight and avoid native import
    side effects before retrieval is actually used."""
    if name in {"EmbeddingEncoder", "load_encoder"}:
        from .embedding import EmbeddingEncoder, load_encoder
        return {"EmbeddingEncoder": EmbeddingEncoder, "load_encoder": load_encoder}[name]
    if name in {"build_index", "load_index"}:
        from .indexer import build_index, load_index
        return {"build_index": build_index, "load_index": load_index}[name]
    if name in {"RewrittenQuery", "rewrite_query", "rewrite_query_rule_based", "rewrite_query_with_llm"}:
        from .query_rewriter import (
            RewrittenQuery,
            rewrite_query,
            rewrite_query_rule_based,
            rewrite_query_with_llm,
        )
        return {
            "RewrittenQuery": RewrittenQuery,
            "rewrite_query": rewrite_query,
            "rewrite_query_rule_based": rewrite_query_rule_based,
            "rewrite_query_with_llm": rewrite_query_with_llm,
        }[name]
    if name == "Retriever":
        from .retriever import Retriever
        return Retriever
    raise AttributeError(name)
