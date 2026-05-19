"""End-to-end pipeline demo with LLM post-processing.

Usage:
    conda activate TTS
    python scripts/demo_llm.py                          # default: 30s, zh
    python scripts/demo_llm.py --seconds 60 --language en
    python scripts/demo_llm.py --no-correction          # skip LLM correction
    python scripts/demo_llm.py --no-intent              # skip intent tagging

Compares output with and without LLM to show the effect.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.table import Table

from src.pipeline import run
from src.utils.config import project_root

console = Console()


def fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--audio",
        default="dataset/AISHELL-4/test/wav/L_R003S01C02.flac",
    )
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--language", default="zh")
    p.add_argument("--out", default="outputs/demo_llm.json")
    p.add_argument("--no-correction", action="store_true",
                   help="Skip LLM ASR correction")
    p.add_argument("--no-consistency", action="store_true",
                   help="Skip speaker consistency check")
    p.add_argument("--no-intent", action="store_true",
                   help="Skip intent tagging")
    p.add_argument("--no-llm", action="store_true",
                   help="Skip ALL LLM post-processing (baseline)")
    args = p.parse_args()

    root = project_root()
    audio_path = (root / args.audio).resolve()
    out_path = (root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    console.rule("[bold cyan]Pipeline Demo — LLM Post-Processing")
    console.print(f"Audio:    {audio_path.name}")
    console.print(f"Duration: {args.seconds:.0f}s")
    console.print(f"Language: {args.language}")
    console.print(f"LLM:      {'OFF (baseline)' if args.no_llm else 'ON'}")
    if not args.no_llm:
        parts = []
        if not args.no_correction: parts.append("correction")
        if not args.no_consistency: parts.append("consistency")
        if not args.no_intent: parts.append("intent")
        console.print(f"  Stages:  {', '.join(parts) if parts else 'none'}")

    # ── Run pipeline ───────────────────────────────────────────────────
    llm_overrides = {}
    if args.no_llm:
        llm_overrides["enabled"] = False
    if args.no_correction:
        llm_overrides["correction"] = False
    if args.no_consistency:
        llm_overrides["consistency"] = False
    if args.no_intent:
        llm_overrides["intent_tagging"] = False

    t0 = time.time()
    result = run(str(audio_path), language=args.language,
                 max_duration_s=args.seconds,
                 llm_overrides=llm_overrides or None)
    wall = time.time() - t0

    # ── Display results ────────────────────────────────────────────────
    console.rule("[bold green]Results")

    segments = result["segments"]
    # Detect whether intents are present
    has_intent = all(
        seg.get("intent") for seg in segments
    ) if segments else False

    tbl = Table(title="Merged Segments", show_lines=False)
    tbl.add_column("Time", style="dim")
    tbl.add_column("Speaker")
    tbl.add_column("Text")
    if has_intent:
        tbl.add_column("Intent")

    for seg in segments:
        ts = f"{fmt_time(seg['start'])} - {fmt_time(seg['end'])}"
        row = [ts, seg.get("speaker", "?"), seg.get("text", "")]
        if has_intent:
            intent_val = seg.get("intent", "-")
            if isinstance(intent_val, list):
                intent_val = " + ".join(intent_val)
            row.append(intent_val)
        tbl.add_row(*row)

    console.print(tbl)
    console.print()

    # Timing summary
    tt = result["timing"]
    console.print(f"[dim]Timing: diarization={tt.get('diarization',0):.1f}s  "
                  f"asr={tt.get('asr',0):.1f}s  "
                  f"alignment={tt.get('alignment',0):.1f}s  "
                  f"llm={tt.get('llm',0):.1f}s  "
                  f"total={tt.get('total',0):.1f}s (wall={wall:.1f}s)[/dim]")

    # Save
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    console.print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
