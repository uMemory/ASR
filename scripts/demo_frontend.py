"""Demucs BGM vocal separation demo.

Usage:
    python scripts/demo_frontend.py                          # default 10s AISHELL-4
    python scripts/demo_frontend.py --audio path/to/movie.wav --seconds 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soundfile as sf
import numpy as np
from rich.console import Console

from src.frontend import load_separator
from src.utils.config import get_model_config

console = Console()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--audio", default="dataset/AISHELL-4/test/wav/L_R003S01C02.flac")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--out", default="outputs/demo_vocals.wav")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    audio_path = (root / args.audio).resolve()
    out_path = (root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    console.rule("[bold cyan]Demucs Vocal Separation Demo")

    # Load a short clip
    waveform, sr = sf.read(str(audio_path), dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    n = int(args.seconds * sr)
    waveform = waveform[:n]

    console.print(f"Audio:  {audio_path.name}  ({args.seconds:.0f}s, {sr}Hz)")
    console.print(f"Input RMS: {np.sqrt(np.mean(waveform**2)):.4f}")

    # Separate
    mcfg = get_model_config()
    sep = load_separator({**mcfg["separation"], "device": mcfg["device"]})
    sep.load()

    import time
    t0 = time.time()
    result = sep.separate(waveform)
    dt = time.time() - t0

    vocals = result.to_mono_16k()
    console.print(f"Separation: {dt:.1f}s  (RTF={dt/args.seconds:.2f}x)")
    console.print(f"Output RMS: {np.sqrt(np.mean(vocals**2)):.4f}")

    # For meeting audio (no BGM), vocals should be very similar to input
    # For film audio (with BGM), vocals RMS will differ from input RMS

    sf.write(str(out_path), vocals, 16000)
    console.print(f"Saved: {out_path}")

    sep.unload()
    console.print("[green]Done.[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
