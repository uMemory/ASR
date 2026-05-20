"""多维检索端到端测试。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import run
import torch


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--audio", default="dataset/AISHELL-4/test/wav/L_R003S01C02.flac")
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--language", default="zh")
    p.add_argument("--query", default=None)
    p.add_argument("--index-dir", default="outputs/index")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    audio_path = str((root / args.audio).resolve())
    index_path = root / args.index_dir

    # ── Step 1: 管线（复刻 test_import_after.py 已验证模式） ─────
    print("── Step 1: 管线 ──")
    result = run(audio_path, language=args.language,
                 max_duration_s=args.seconds, llm_overrides={"enabled": False})
    segments = result["segments"]
    print(f"片段: {len(segments)}  说话人: {result['num_speakers']}")

    # ── 清理 GPU（必须在导入检索前） ──────────────────────────────
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print("GPU 已清理")

    # ── Step 2: 索引 ──────────────────────────────────────────────
    print("── Step 2: 构建索引 ──")
    from src.retrieval import EmbeddingEncoder, build_index
    from src.utils.config import get_model_config

    mcfg = get_model_config()
    enc = EmbeddingEncoder(
        model_path=mcfg["embedding"]["model"],
        device=mcfg["device"],
    )
    enc.load()
    print(f"编码器: {enc.device}")

    meta, _, _ = build_index(segments, enc, store_path=index_path)
    print(f"索引: {meta['num_segments']} 段, dim={meta['dim']}")

    # ── Step 3: 检索 ──────────────────────────────────────────────
    print("── Step 3: 检索 ──")
    from src.retrieval import Retriever
    ret = Retriever(index_path=index_path, encoder=enc)

    queries = [args.query] if args.query else [
        "反对意见", "Speaker_00 的发言", "包含小区",
    ]
    for q in queries:
        results = ret.search(q, top_k=5)
        print(f"\n查询: {q}  →  {len(results)} 条")
        for i, seg in enumerate(results):
            print(f"  {i+1}. [{seg['start']:.0f}s] {seg['speaker']}: {seg['text']}")

    # ── Step 4: 说话人 ────────────────────────────────────────────
    print("\n── Step 4: 说话人过滤 ──")
    for spk in sorted({s.get("speaker", "") for s in segments}):
        if spk in ("SPEAKER_UNKNOWN", "UNKNOWN"):
            continue
        results = ret.search("", speaker=spk, top_k=3)
        print(f"  {spk}: {len(results)} 条")

    enc.unload()
    print("✅ 检索测试完成")
    return 0


if __name__ == "__main__":
    sys.exit(main())
