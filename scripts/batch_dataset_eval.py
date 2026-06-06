"""Incremental batch transcription and evaluation for AISHELL-1/LibriSpeech.

Each run processes the next small batch and stores all artifacts under
``outputs/batch_eval``.  It is designed for scheduled execution.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import run as run_pipeline


AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def strip_result_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".json"):
        name = name[:-5]
    for suffix in AUDIO_SUFFIXES:
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def load_aishell_ids(root: Path) -> set[str]:
    transcript = root / "dataset" / "AISHELL" / "transcript" / "aishell_transcript_v0.8.txt"
    ids: set[str] = set()
    if transcript.exists():
        with open(transcript, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if parts:
                    ids.add(parts[0])
    return ids


def collect_librispeech(root: Path, max_files: int) -> list[Path]:
    base = root / "dataset" / "LibriSpeech" / "test-clean"
    return sorted(base.rglob("*.flac"))[:max_files]


def collect_aishell(root: Path, out_root: Path, max_files: int) -> list[Path]:
    """Extract a small deterministic AISHELL-1 wav subset from speaker tarballs."""
    aishell_root = root / "dataset" / "AISHELL"
    wav_root = aishell_root / "wav"
    extracted_root = out_root / "audio" / "aishell1"
    extracted_root.mkdir(parents=True, exist_ok=True)
    transcript_ids = load_aishell_ids(root)

    existing = sorted(extracted_root.glob("*.wav"))
    if len(existing) >= max_files:
        return existing[:max_files]

    for tar_path in sorted(wav_root.glob("*.tar.gz")):
        if len(sorted(extracted_root.glob("*.wav"))) >= max_files:
            break
        try:
            with tarfile.open(tar_path, "r:gz") as tf:
                members = [m for m in tf.getmembers() if m.isfile() and m.name.lower().endswith(".wav")]
                for member in sorted(members, key=lambda m: m.name):
                    if len(sorted(extracted_root.glob("*.wav"))) >= max_files:
                        break
                    utt_id = Path(member.name).stem
                    if transcript_ids and utt_id not in transcript_ids:
                        continue
                    dst = extracted_root / f"{utt_id}.wav"
                    if dst.exists():
                        continue
                    src = tf.extractfile(member)
                    if src is None:
                        continue
                    with open(dst, "wb") as f:
                        shutil.copyfileobj(src, f)
        except Exception as exc:
            print(f"[WARN] Failed to scan {tar_path}: {exc}", flush=True)

    return sorted(extracted_root.glob("*.wav"))[:max_files]


def result_path_for(audio: Path, dataset: str, out_root: Path) -> Path:
    return out_root / "results" / dataset / f"{audio.name}.json"


def collect_dataset_files(root: Path, out_root: Path, dataset: str, max_files: int) -> list[Path]:
    if dataset == "aishell1":
        return collect_aishell(root, out_root, max_files)
    if dataset == "librispeech":
        return collect_librispeech(root, max_files)
    raise ValueError(f"Unsupported dataset: {dataset}")


def pending_files(files: list[Path], dataset: str, out_root: Path, state: dict[str, Any]) -> list[Path]:
    done = set(state.get(dataset, {}).get("done", []))
    pending: list[Path] = []
    for audio in files:
        key = audio.name
        if key in done:
            continue
        if result_path_for(audio, dataset, out_root).exists():
            continue
        pending.append(audio)
    return pending


def transcribe_one(audio: Path, dataset: str, out_root: Path, enable_llm: bool, seconds: float) -> Path:
    language = "zh" if dataset == "aishell1" else "en"
    result = run_pipeline(
        str(audio),
        language=language,
        max_duration_s=seconds if seconds > 0 else None,
        llm_overrides={"enabled": enable_llm},
    )
    result["file"] = str(audio)
    result["name"] = audio.name
    out = result_path_for(audio, dataset, out_root)
    save_json(out, result)
    return out


def run_evaluation(root: Path, dataset: str, out_root: Path, compare_llm: bool) -> int:
    pred_dir = out_root / "results" / dataset
    if not any(pred_dir.glob("*.json")):
        return 0
    language = "zh" if dataset == "aishell1" else "en"
    output = out_root / f"eval_{dataset}.json"
    cmd = [
        sys.executable,
        "-B",
        str(root / "experiments" / "evaluate_system.py"),
        "--dataset",
        dataset,
        "--pred-dir",
        str(pred_dir),
        "--language",
        language,
        "--output",
        str(output),
    ]
    if compare_llm:
        cmd.append("--compare-llm")
    print("[EVAL]", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(root))


def main() -> int:
    parser = argparse.ArgumentParser(description="Incrementally process AISHELL-1/LibriSpeech batches.")
    parser.add_argument("--dataset", choices=["aishell1", "librispeech", "both"], default="both")
    parser.add_argument("--batch-size", type=int, default=5, help="Files per dataset per run.")
    parser.add_argument("--max-files", type=int, default=50, help="Maximum sample files per dataset.")
    parser.add_argument("--seconds", type=float, default=0.0, help="Max seconds per file, 0=full clip.")
    parser.add_argument("--output-root", default="outputs/batch_eval")
    parser.add_argument("--llm", action="store_true", help="Enable LLM post-processing.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    out_root = (root / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    state_path = out_root / "state.json"
    state = load_json(state_path, {})

    datasets = ["aishell1", "librispeech"] if args.dataset == "both" else [args.dataset]
    any_processed = False

    for dataset in datasets:
        state.setdefault(dataset, {"done": [], "failed": []})
        files = collect_dataset_files(root, out_root, dataset, args.max_files)
        pending = pending_files(files, dataset, out_root, state)
        batch = pending[: max(0, args.batch_size)]
        print(f"[{dataset}] total={len(files)} pending={len(pending)} batch={len(batch)}", flush=True)
        for audio in batch:
            print(f"[RUN] {dataset}: {audio}", flush=True)
            if args.dry_run:
                continue
            try:
                transcribe_one(audio, dataset, out_root, enable_llm=args.llm, seconds=args.seconds)
                state[dataset]["done"].append(audio.name)
                any_processed = True
                save_json(state_path, state)
            except Exception as exc:
                print(f"[ERROR] {audio.name}: {exc}", flush=True)
                state[dataset]["failed"].append({"file": audio.name, "error": str(exc)})
                save_json(state_path, state)

        if not args.dry_run:
            run_evaluation(root, dataset, out_root, compare_llm=args.llm)

    if args.dry_run:
        print("[DRY-RUN] No files were processed.", flush=True)
    elif not any_processed:
        print("[DONE] No pending files in this run.", flush=True)
    else:
        print(f"[DONE] State saved to {state_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
