"""Smoke test: transcribe the first 60s of one AISHELL-4 file with Whisper-medium.

Usage:
    conda activate TTS
    python scripts/demo_asr.py
    python scripts/demo_asr.py --audio path/to/file.wav --seconds 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import librosa
import numpy as np
import torch
from rich.console import Console

from src.asr import load_asr_backend
from src.utils.config import get_model_config, load_config

console = Console()


def load_clip(path: Path, seconds: float, target_sr: int = 16000) -> np.ndarray:
    """Load mono PCM float32 in [-1, 1], clipped to `seconds`."""
    audio, sr = librosa.load(str(path), sr=target_sr, mono=True, duration=seconds)
    return audio.astype(np.float32)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--audio",
        default="dataset/AISHELL-4/test/wav/L_R003S01C02.flac",
        help="Audio file (relative to project root or absolute)",
    )
    p.add_argument("--seconds", type=float, default=60.0, help="Seconds to transcribe")
    p.add_argument("--language", default="zh", help="ASR language (zh/en/...)")
    p.add_argument("--out", default="outputs/demo_asr.json")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    audio_path = (root / args.audio).resolve() if not Path(args.audio).is_absolute() else Path(args.audio)
    out_path = (root / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    console.rule("[bold cyan]Whisper-medium smoke test")
    console.print(f"Audio:    {audio_path}")
    console.print(f"Clip:     first {args.seconds}s, language={args.language}")

    t0 = time.time()
    audio = load_clip(audio_path, args.seconds)
    console.print(f"Loaded:   {len(audio)/16000:.1f}s of audio in {time.time()-t0:.2f}s")

    mcfg = get_model_config()
    console.print(f"Profile:  {mcfg['profile']}  device={mcfg['device']}")
    console.print(f"Backend:  {mcfg['asr'].get('backend')}  model={mcfg['asr']['model']}")

    backend = load_asr_backend({**mcfg["asr"], "device": mcfg["device"]})

    # NOTE: HF pipeline + Whisper has a known issue where initial_prompt tokens
    # leak into the transcription output when long-form chunking is enabled.
    # `language=zh` alone is enough to force Simplified Chinese output, so we
    # skip the prompt for now. Revisit if punctuation quality suffers.
    initial_prompt = None

    t0 = time.time()
    backend.load()
    console.print(f"Model loaded in {time.time()-t0:.1f}s "
                  f"(CUDA mem: {torch.cuda.memory_allocated()/1e9:.2f}GB)")

    t0 = time.time()
    result = backend.transcribe(
        audio,
        language=args.language,
        initial_prompt=initial_prompt,
    )
    dt = time.time() - t0
    rtf = dt / (len(audio) / 16000)
    console.print(f"Transcribed in {dt:.1f}s  (RTF={rtf:.2f}x)")

    console.rule("[bold green]Segments")
    for seg in result["segments"]:
        console.print(
            f"[{seg['start']:6.2f}-{seg['end']:6.2f}] {seg['text']}"
        )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    console.print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
