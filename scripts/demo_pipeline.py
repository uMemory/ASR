"""End-to-end pipeline demo.

Runs the full ASR + diarization pipeline on one audio file and displays
the "who spoke what when" result.

Usage:
    # Default: first 30 s of the first AISHELL-4 file
    python scripts/demo_pipeline.py

    # Specify audio and duration
    python scripts/demo_pipeline.py --audio path/to/file.wav --seconds 60
    python scripts/demo_pipeline.py --audio path/to/file.wav --seconds 30 --language en
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.table import Table
from rich.text import Text

console = Console()


def _format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _color_for_speaker(speaker: str) -> str:
    """Deterministic color based on speaker id."""
    colors = ["cyan", "green", "yellow", "magenta", "blue", "red", "bright_cyan",
              "bright_green", "bright_yellow", "bright_magenta"]
    if speaker == "SPEAKER_UNKNOWN":
        return "white"
    # Extract number from "SPEAKER_XX"
    try:
        idx = int(speaker.split("_")[-1]) % len(colors)
    except (ValueError, IndexError):
        idx = 0
    return colors[idx]


def main() -> int:
    p = argparse.ArgumentParser(description="Multi-speaker ASR pipeline demo")
    p.add_argument("--audio",
                   default="dataset/AISHELL-4/test/wav/L_R003S01C02.flac",
                   help="Audio file path (relative to project root)")
    p.add_argument("--seconds", type=float, default=30.0,
                   help="Process only first N seconds of audio")
    p.add_argument("--language", default="zh",
                   help="ASR language code (zh/en/...)")
    p.add_argument("--out",
                   default="outputs/demo_pipeline.json",
                   help="Output JSON path (relative to project root)")
    p.add_argument("--profile", default=None,
                   help="Config profile: local (default) or cloud")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    audio_path = (root / args.audio).resolve()
    out_path = (root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not audio_path.exists():
        console.print(f"[red]Audio file not found:[/red] {audio_path}")
        return 1

    console.rule("[bold cyan]Multi-Speaker ASR Pipeline Demo")
    console.print(f"Audio:       {audio_path.name}")
    console.print(f"Duration:    {args.seconds}s")
    console.print(f"Language:    {args.language}")
    console.print(f"Profile:     {args.profile or '(default)'}")
    console.print()

    # ── Run pipeline ──────────────────────────────────────────────────
    try:
        from src.pipeline import run

        result = run(
            audio_path=str(audio_path),
            language=args.language,
            profile=args.profile,
            max_duration_s=args.seconds,
        )
    except Exception as e:
        console.print(f"[red]Pipeline failed:[/red] {type(e).__name__}: {e}")
        import traceback
        console.print(traceback.format_exc())
        return 1

    # ── Display timing ────────────────────────────────────────────────
    timing = result.get("timing", {})
    t_table = Table(title="Timing", show_lines=False)
    t_table.add_column("Stage")
    t_table.add_column("Time")
    for stage, seconds in timing.items():
        t_table.add_row(stage, f"{seconds}s")
    console.print(t_table)
    console.print()

    # ── Display segments ──────────────────────────────────────────────
    segments = result.get("segments", [])
    console.print(f"Results: [bold]{len(segments)}[/bold] segments, "
                  f"[bold]{result['num_speakers']}[/bold] speakers, "
                  f"language=[bold]{result['language']}[/bold]")

    seg_table = Table(title="Segments", show_lines=True)
    seg_table.add_column("Time", style="dim")
    seg_table.add_column("Speaker", no_wrap=True)
    seg_table.add_column("Text")

    for seg in segments[:30]:  # show first 30 segments
        t = f"{_format_time(seg['start'])} - {_format_time(seg['end'])}"
        spk = Text(seg["speaker"], style=_color_for_speaker(seg["speaker"]))
        seg_table.add_row(t, spk, seg["text"])

    if len(segments) > 30:
        more = len(segments) - 30
        seg_table.add_row("…", f"[dim]{more} more[/dim]", "")

    console.print(seg_table)

    # ── Save output ───────────────────────────────────────────────────
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    console.print(f"\nSaved: [green]{out_path}[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
