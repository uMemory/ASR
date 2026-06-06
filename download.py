"""Download local model snapshots used by the ASR project.

The script downloads model files into the directory layout expected by
``configs/models.yaml``.  Large files are ignored by Git; only the empty
directory structure and this downloader are committed.

Usage:
    python download.py
    python download.py --only whisper-medium bge-m3
    python download.py --hf-token YOUR_TOKEN
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo_id: str
    local_dir: Path
    gated: bool = False
    note: str = ""


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="whisper-medium",
        repo_id="openai/whisper-medium",
        local_dir=MODELS_DIR / "whisper-medium",
    ),
    ModelSpec(
        name="bge-m3",
        repo_id="BAAI/bge-m3",
        local_dir=MODELS_DIR / "bge-m3",
    ),
    ModelSpec(
        name="wav2vec2-zh",
        repo_id="jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
        local_dir=MODELS_DIR / "wav2vec2-large-xlsr-53-chinese-zh-cn",
    ),
    ModelSpec(
        name="pyannote-diarization",
        repo_id="pyannote/speaker-diarization-3.1",
        local_dir=MODELS_DIR / "speaker-diarization-3.1",
        gated=True,
        note="Requires accepting the Hugging Face model terms.",
    ),
    ModelSpec(
        name="pyannote-segmentation",
        repo_id="pyannote/segmentation-3.0",
        local_dir=MODELS_DIR / "segmentation-3.0",
        gated=True,
        note="Requires accepting the Hugging Face model terms.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download model snapshots for this project.")
    parser.add_argument(
        "--only",
        nargs="+",
        choices=[spec.name for spec in MODEL_SPECS],
        help="Download only selected model names.",
    )
    parser.add_argument(
        "--hf-token",
        default=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN or HUGGINGFACE_HUB_TOKEN.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force snapshot_download even when the local directory is non-empty.",
    )
    return parser.parse_args()


def download_model(spec: ModelSpec, token: str | None, force: bool) -> None:
    spec.local_dir.mkdir(parents=True, exist_ok=True)
    has_files = any(p.name != ".gitkeep" for p in spec.local_dir.iterdir())
    if has_files and not force:
        print(f"[SKIP] {spec.name}: {spec.local_dir} already contains files")
        return

    if spec.gated and not token:
        print(f"[WARN] {spec.name}: gated model; set HF_TOKEN or pass --hf-token")

    print(f"[GET ] {spec.name}: {spec.repo_id} -> {spec.local_dir}")
    if spec.note:
        print(f"       {spec.note}")
    snapshot_download(
        repo_id=spec.repo_id,
        local_dir=str(spec.local_dir),
        local_dir_use_symlinks=False,
        token=token,
        resume_download=True,
    )
    print(f"[DONE] {spec.name}")


def main() -> None:
    args = parse_args()
    selected = set(args.only or [spec.name for spec in MODEL_SPECS])
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for spec in MODEL_SPECS:
        if spec.name in selected:
            download_model(spec, token=args.hf_token, force=args.force)


if __name__ == "__main__":
    main()
