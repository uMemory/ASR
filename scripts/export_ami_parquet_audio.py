"""Export AMI parquet utterance clips to WAV files for pipeline testing.

AMI parquet rows contain short utterance audio in an ``audio`` dict, not a
standalone meeting WAV path. This script reconstructs small meeting slices by
placing those clips on their original timeline and filling gaps with silence.

Examples:
    python scripts/export_ami_parquet_audio.py --input dataset/AMI/SDM --max-meetings 2 --duration 180
    python scripts/export_ami_parquet_audio.py --input dataset/AMI/IHM/test-00000-of-00004.parquet --duration 120
"""
from __future__ import annotations

import argparse
import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf


def find_parquets(path: Path) -> list[Path]:
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input does not exist: {path}")
    if path.is_file():
        if path.suffix.lower() != ".parquet":
            raise ValueError(f"Input file is not parquet: {path}")
        return [path]
    files = sorted(path.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under: {path}")
    return files


def read_audio_bytes(value: object) -> tuple[np.ndarray, int]:
    if not isinstance(value, dict) or not value.get("bytes"):
        raise ValueError("AMI row audio field does not contain audio bytes")
    audio, sr = sf.read(BytesIO(value["bytes"]), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def safe_name(text: str) -> str:
    keep = []
    for ch in text:
        keep.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    return "".join(keep).strip("_") or "ami"


def collect_groups(parquets: list[Path]) -> dict[tuple[str, str], pd.DataFrame]:
    frames = []
    for p in parquets:
        df = pd.read_parquet(p)
        required = {"meeting_id", "audio", "begin_time", "end_time", "microphone_id", "speaker_id", "text"}
        missing = sorted(required - set(df.columns))
        if missing:
            raise ValueError(f"{p} missing columns: {missing}")
        frames.append(df)

    all_rows = pd.concat(frames, ignore_index=True)
    all_rows = all_rows.sort_values(["meeting_id", "microphone_id", "begin_time", "end_time"])
    return {
        (str(meeting_id), str(mic_id)): group.reset_index(drop=True)
        for (meeting_id, mic_id), group in all_rows.groupby(["meeting_id", "microphone_id"], sort=True)
    }


def export_group(
    meeting_id: str,
    mic_id: str,
    rows: pd.DataFrame,
    out_dir: Path,
    duration_s: float,
    min_utterances: int,
) -> Path | None:
    if len(rows) < min_utterances:
        return None

    rows = rows.sort_values(["begin_time", "end_time"]).reset_index(drop=True)
    start_abs = float(rows.iloc[0]["begin_time"])
    end_limit_abs = start_abs + duration_s if duration_s > 0 else float(rows["end_time"].max())
    rows = rows[rows["begin_time"].astype(float) < end_limit_abs].reset_index(drop=True)
    if len(rows) < min_utterances:
        return None

    first_audio, sr = read_audio_bytes(rows.iloc[0]["audio"])
    total_len = max(1, int(round((min(float(rows["end_time"].max()), end_limit_abs) - start_abs) * sr)))
    timeline = np.zeros(total_len, dtype=np.float32)
    refs: list[dict] = []

    for _, row in rows.iterrows():
        begin_abs = float(row["begin_time"])
        end_abs = float(row["end_time"])
        if begin_abs >= end_limit_abs:
            continue

        audio, row_sr = read_audio_bytes(row["audio"])
        if row_sr != sr:
            raise ValueError(f"Mixed sample rates in {meeting_id}/{mic_id}: {sr} and {row_sr}")

        start_i = max(0, int(round((begin_abs - start_abs) * sr)))
        end_i = min(total_len, start_i + len(audio))
        if end_i <= start_i:
            continue

        clip = audio[: end_i - start_i]
        timeline[start_i:end_i] += clip
        refs.append(
            {
                "start": round(begin_abs - start_abs, 3),
                "end": round(min(end_abs, end_limit_abs) - start_abs, 3),
                "speaker": str(row["speaker_id"]),
                "text": str(row["text"]),
                "meeting_id": meeting_id,
                "microphone_id": mic_id,
            }
        )

    if not refs:
        return None

    peak = float(np.max(np.abs(timeline)))
    if peak > 0.98:
        timeline = timeline / peak * 0.98

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = safe_name(f"{meeting_id}_{mic_id}_{int(duration_s) if duration_s > 0 else 'full'}s")
    wav_path = out_dir / f"{stem}.wav"
    json_path = out_dir / f"{stem}.json"
    sf.write(wav_path, timeline, sr)
    json_path.write_text(json.dumps(refs, ensure_ascii=False, indent=2), encoding="utf-8")
    return wav_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Export AMI parquet clips to WAV files.")
    parser.add_argument("--input", required=True, help="AMI parquet file or directory")
    parser.add_argument("--output", default="dataset/AMI/exported_audio", help="Output directory")
    parser.add_argument("--duration", type=float, default=180.0, help="Seconds per exported meeting slice; 0=full available span")
    parser.add_argument("--max-meetings", type=int, default=5, help="Maximum meeting/microphone groups to export")
    parser.add_argument("--min-utterances", type=int, default=8, help="Skip groups with fewer utterances")
    args = parser.parse_args()

    parquets = find_parquets(Path(args.input))
    groups = collect_groups(parquets)
    out_dir = Path(args.output)

    exported = []
    for (meeting_id, mic_id), rows in groups.items():
        wav = export_group(meeting_id, mic_id, rows, out_dir, args.duration, args.min_utterances)
        if wav is not None:
            exported.append(wav)
            print(f"[OK] {wav}")
        if len(exported) >= args.max_meetings:
            break

    if not exported:
        print("[WARN] No WAV files exported. Try lowering --min-utterances or checking the parquet contents.")
        return 1

    print(f"\nExported {len(exported)} WAV file(s) to {out_dir.resolve()}")
    print("Use them with:")
    print(f"  python scripts/run_test.py --path {out_dir} --language en --seconds 180 --no-llm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
