"""Evaluate diarization (DER) on AISHELL-4 test audio.

Compares pipeline output against ground-truth RTTM using
``pyannote.metrics.diarization.DiarizationErrorRate``.

Usage:
    python scripts/eval_diarization.py --audio dataset/AISHELL-4/test/wav/L_R003S01C02.flac --seconds 60
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soundfile as sf
import torch
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from rich.console import Console
from rich.table import Table

from src.utils.config import project_root

console = Console()


def read_rttm(rttm_path: Path) -> Annotation:
    """Parse an RTTM file into a pyannote ``Annotation``.

    RTTM format: SPEAKER <file> <channel> <start> <duration> <NA> <NA> <speaker> <NA> <NA>
    """
    annotation = Annotation()
    with open(rttm_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            annotation[Segment(start, start + duration)] = speaker
    return annotation


def segments_to_annotation(segments: list[dict]) -> Annotation:
    """Convert pipeline-style segment dicts to a pyannote ``Annotation``."""
    annotation = Annotation()
    for seg in segments:
        speaker = seg["speaker"]
        if speaker == "SPEAKER_UNKNOWN":
            continue  # skip unknown labels
        annotation[Segment(seg["start"], seg["end"])] = speaker
    return annotation


def main() -> int:
    p = argparse.ArgumentParser(description="Diarization error rate on AISHELL-4")
    p.add_argument("--audio", default="dataset/AISHELL-4/test/wav/L_R003S01C02.flac")
    p.add_argument("--rttm", default=None,
                   help="RTTM path (auto-resolved from audio path if omitted)")
    p.add_argument("--seconds", type=float, default=60.0)
    p.add_argument("--profile", default=None)
    args = p.parse_args()

    root = project_root()
    audio_path = (root / args.audio).resolve()

    # Resolve RTTM: same basename in TextGrid dir
    if args.rttm is None:
        rttm_path = (root / "dataset/AISHELL-4/test/TextGrid" / audio_path.stem).with_suffix(".rttm")
    else:
        rttm_path = Path(args.rttm)

    if not audio_path.exists():
        console.print(f"[red]Audio not found:[/red] {audio_path}")
        return 1
    if not rttm_path.exists():
        console.print(f"[red]RTTM not found:[/red] {rttm_path}")
        return 1

    console.rule("[bold cyan]DER Evaluation")
    console.print(f"Audio:     {audio_path.name}")
    console.print(f"RTTM:      {rttm_path.name}")

    # ── Load ground truth ──────────────────────────────────────────────
    reference = read_rttm(rttm_path)
    # Filter to the evaluation window
    duration = args.seconds
    reference = reference.crop(Segment(0, duration))

    console.print(f"Reference: {len(reference)} segments, "
                  f"{len(reference.labels())} speakers")
    console.print()

    # ── Run pipeline ───────────────────────────────────────────────────
    from src.pipeline import run

    console.print("Running pipeline …")
    result = run(
        audio_path=str(audio_path),
        language="zh",
        profile=args.profile,
        max_duration_s=duration,
    )
    console.print()

    hypothesis = segments_to_annotation(result["segments"])
    hypothesis = hypothesis.crop(Segment(0, duration))

    console.print(f"Hypothesis: {len(hypothesis)} segments, "
                  f"{len(hypothesis.labels())} speakers")
    console.print()

    # ── Compute DER ────────────────────────────────────────────────────
    metric = DiarizationErrorRate()
    der = metric(reference, hypothesis)

    # Detailed breakdown
    table = Table(title="Diarization Metrics")
    table.add_column("Metric")
    table.add_column("Value")

    detail = metric(reference, hypothesis, detailed=True)
    table.add_row("DER", f"{der:.2%}")
    table.add_row("False alarm", f"{detail['false alarm']:.2%}")
    table.add_row("Missed detection", f"{detail['missed detection']:.2%}")
    table.add_row("Confusion", f"{detail['confusion']:.2%}")

    console.print(table)

    # Collar analysis
    collar_metric = DiarizationErrorRate(collar=0.250, skip_overlap=False)
    der_collar = collar_metric(reference, hypothesis)
    console.print(f"\nDER (collar=0.25s): [bold]{der_collar:.2%}[/bold]")

    console.print(f"\n[bold green]DER = {der:.2%}[/bold green]")
    return 0 if der < 1.0 else 1


if __name__ == "__main__":
    sys.exit(main())
