"""30-second chunk streaming pipeline demo.

Usage:
    python scripts/demo_streaming.py                         # default 60s, zh
    python scripts/demo_streaming.py --seconds 120           # 2 minutes
    python scripts/demo_streaming.py --no-llm                # skip LLM for speed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.table import Table

from src.streaming import ChunkProcessor
from src.utils.config import project_root

console = Console()


def fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--audio", default="dataset/AISHELL-4/test/wav/L_R003S01C02.flac")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--language", default="zh")
    p.add_argument("--out", default="outputs/demo_streaming.json")
    args = p.parse_args()

    root = project_root()
    audio_path = (root / args.audio).resolve()
    out_path = (root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    console.rule("[bold cyan]Streaming Pipeline Demo (30s chunks)")
    console.print(f"Audio:    {audio_path.name}")
    console.print(f"Language: {args.language}")
    console.print()

    cp = ChunkProcessor(
        chunk_duration_s=30,
        overlap_s=5,
        language=args.language,
    )

    # Process in streaming mode
    chunk_count = 0
    for result in cp.process_file(audio_path):
        chunk_count += 1
        segs = result["segments"]
        console.print(
            f"  Chunk {result['chunk_index']:2d}  "
            f"t={result['chunk_start_s']:6.1f}s  "
            f"segments={len(segs):2d}  "
            f"diar={result['timing'].get('diarization',0):.1f}s  "
            f"asr={result['timing'].get('asr',0):.1f}s"
        )

        if chunk_count * 30 >= args.seconds:
            # Truncate: close the processor manually
            # We stop iteration here; the processor would continue otherwise
            break

    console.rule("[bold green]Merged Result")
    merged = cp.process_file_merged(audio_path)
    segments = merged["segments"]

    tbl = Table(title=f"Streaming Result ({len(segments)} segments)")
    tbl.add_column("Time", style="dim")
    tbl.add_column("Speaker")
    tbl.add_column("Text")
    for seg in segments[:20]:
        ts = f"{fmt_time(seg['start'])} - {fmt_time(seg['end'])}"
        tbl.add_row(ts, seg.get("speaker", "?"), seg.get("text", ""))
    console.print(tbl)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    console.print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
