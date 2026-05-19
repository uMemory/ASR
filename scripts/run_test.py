"""一键测试：指定目录或文件，自动跑完整管线并输出结构化结果。

用法:
    python scripts/run_test.py                          # 测试默认音频
    python scripts/run_test.py --path ./my_audio.wav    # 单文件
    python scripts/run_test.py --path ./audio_dir/      # 目录（递归扫描）
    python scripts/run_test.py --path ./dir/ --no-llm   # 跳过 LLM
    python scripts/run_test.py --path ./dir/ --retrieval # 含检索
    python scripts/run_test.py --path ./dir/ --output results.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def find_audio_files(path: Path) -> list[Path]:
    """递归扫描目录或单文件，返回音频文件列表。"""
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"路径不存在: {path}")

    if path.is_file():
        if path.suffix.lower() in (".wav", ".flac", ".mp3", ".m4a", ".ogg"):
            return [path]
        raise ValueError(f"不支持的文件格式: {path.suffix}")

    # 目录：递归扫描
    extensions = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
    files: list[Path] = []
    for ext in extensions:
        files.extend(path.rglob(f"*{ext}"))
    files.sort()
    return files


def format_segment(seg: dict, idx: int | None = None) -> str:
    """格式化单个段为可读字符串。"""
    prefix = f"  {idx}. " if idx is not None else "  "
    start_s = seg.get("start", 0)
    end_s = seg.get("end", 0)
    speaker = seg.get("speaker", "?")
    text = seg.get("text", "")
    intent = seg.get("intent", [])

    ts = f"[{int(start_s // 60):02d}:{start_s % 60:04.1f} - {int(end_s // 60):02d}:{end_s % 60:04.1f}]"
    line = f"{ts} {speaker}: {text}"
    if intent:
        line += f"  [{', '.join(intent)}]"
    return line


def print_separator(title: str) -> None:
    width = 72
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def main() -> int:
    p = argparse.ArgumentParser(description="一键测试：多说话人语音转写管线")
    p.add_argument("--path", default=None,
                   help="音频文件或目录路径（默认: dataset/AISHELL-4/test/wav/L_R003S01C02.flac）")
    p.add_argument("--language", default="zh", help="语言代码 (zh/en)")
    p.add_argument("--seconds", type=float, default=60.0,
                   help="每文件最大处理时长（秒），0=完整文件")
    p.add_argument("--no-llm", action="store_true", help="跳过 LLM 后处理")
    p.add_argument("--retrieval", action="store_true", help="构建检索索引")
    p.add_argument("--output", default=None, help="结果保存到 JSON 文件")
    p.add_argument("--profile", default=None, help="配置 profile (local/cloud)")
    args = p.parse_args()

    # 解析路径
    root = Path(__file__).resolve().parents[1]
    if args.path:
        audio_path = Path(args.path)
        if not audio_path.is_absolute():
            audio_path = root / audio_path
    else:
        audio_path = root / "dataset/AISHELL-4/test/wav/L_R003S01C02.flac"

    # 找到所有音频文件
    try:
        files = find_audio_files(audio_path)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}")
        return 1

    print_separator("多说话人语音转写系统 — 一键测试")
    print(f"  音频文件: {len(files)} 个")
    print(f"  每文件最长: {args.seconds}s" if args.seconds > 0 else "  完整文件（不截断）")
    print(f"  LLM 后处理: {'关闭' if args.no_llm else '开启'}")
    print(f"  检索: {'开启' if args.retrieval else '关闭'}")
    print(f"  语言: {args.language}")

    # 导入管线
    import torch
    from src.pipeline import run as run_pipeline

    all_results: list[dict] = []
    total_start = time.time()

    # 如果需要检索，提前加载编码器（复用）
    encoder = None
    retrieval_results: list[dict] = []
    if args.retrieval:
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        from src.retrieval.embedding import EmbeddingEncoder
        from src.retrieval.indexer import build_index
        from src.retrieval.retriever import Retriever
        from src.utils.config import get_model_config
        mcfg = get_model_config(args.profile)
        encoder = EmbeddingEncoder(
            model_path=mcfg["embedding"]["model"],
            device=mcfg["device"],
        )
        encoder.load()
        print(f"  检索编码器已加载: {encoder.device}")

    # 逐个处理
    for fi, fpath in enumerate(files):
        print_separator(f"文件 {fi+1}/{len(files)}: {fpath.name}")

        t0 = time.time()

        try:
            result = run_pipeline(
                str(fpath),
                language=args.language,
                profile=args.profile,
                max_duration_s=args.seconds if args.seconds > 0 else None,
                llm_overrides={"enabled": not args.no_llm},
            )
        except Exception as e:
            print(f"  [ERROR] 管线失败: {e}")
            import traceback
            traceback.print_exc()
            continue

        elapsed = time.time() - t0
        segments = result["segments"]

        # 打印结果
        print(f"\n  说话人数: {result.get('num_speakers', '?')}")
        print(f"  段数: {len(segments)}")
        print(f"  耗时: {elapsed:.1f}s")
        print(f"  语言: {result.get('language', '?')}\n")

        for i, seg in enumerate(segments):
            print(format_segment(seg, i + 1))

        # 检索
        if args.retrieval and encoder is not None and segments:
            import gc as _gc
            _gc.collect()
            torch.cuda.empty_cache()

            index_path = root / "outputs" / f"test_index_{fpath.stem}"
            meta, _, _ = build_index(segments, encoder, store_path=index_path)
            ret = Retriever(index_path=index_path, encoder=encoder)

            queries = [
                ("语义", segments[0].get("text", "")[:30] if segments else "测试"),
                ("反对", "反对"),
                ("提问", "提问"),
            ]

            print(f"\n  ── 检索测试 ──")
            for label, q in queries:
                hits = ret.search(q, top_k=3)
                print(f"  [{label}] '{q}' → {len(hits)} 条结果")
                for h in hits:
                    print(f"    [{h['start']:.0f}s] {h.get('speaker','?')}: {h.get('text','')[:50]}")

            print(f"  索引已保存: {index_path}")

        # 保存结果
        all_results.append({
            "file": str(fpath),
            "name": fpath.name,
            "num_speakers": result.get("num_speakers"),
            "num_segments": len(segments),
            "language": result.get("language"),
            "elapsed_s": round(elapsed, 1),
            "timing": result.get("timing", {}),
            "segments": segments,
        })

    # 汇总
    total_elapsed = time.time() - total_start
    print_separator("汇总")
    print(f"  处理文件: {len(all_results)}/{len(files)}")
    print(f"  总耗时: {total_elapsed:.1f}s")
    if all_results:
        avg_s = sum(r["num_segments"] for r in all_results) / len(all_results)
        print(f"  平均段数: {avg_s:.1f}")

    # 保存 JSON
    if args.output:
        out_path = root / args.output if not Path(args.output).is_absolute() else Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "config": {
                    "path": str(audio_path),
                    "language": args.language,
                    "seconds": args.seconds,
                    "llm_enabled": not args.no_llm,
                    "retrieval": args.retrieval,
                },
                "results": all_results,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n  结果已保存: {out_path}")

    # 清理
    if encoder is not None:
        encoder.unload()

    return 0


if __name__ == "__main__":
    sys.exit(main())
