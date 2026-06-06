"""Pyannote speaker diarization backend.

Follows the same factory + class pattern as src/asr/ backends.

Supports two loading modes:
  **Local**: ``pipeline_config`` is an absolute or relative path to a
    ``config.yaml`` that references local ``.bin`` files.  CWD is
    temporarily switched to the project root so that relative paths
    inside the YAML resolve correctly (official pyannote convention).

  **Cloud**: ``pipeline_config`` is a HuggingFace repo id such as
    ``pyannote/speaker-diarization-3.1``.  The pipeline and sub-models
    are downloaded from HF Hub (requires HF_TOKEN in .env for gated models).

Usage:
    from src.diarization import load_diarization_backend
    backend = load_diarization_backend(cfg["diarization"] | {"device": cfg["device"]})
    backend.load()
    result = backend.diarize("/path/to/audio.wav")
    backend.unload()
"""
from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _drop_optional_speechbrain_lazy_modules() -> None:
    """Avoid optional SpeechBrain k2 lazy imports during PyTorch inspection.

    PyTorch Lightning calls ``inspect.stack()`` while loading checkpoints.
    Python's inspect module walks ``sys.modules`` and probes ``__file__``;
    SpeechBrain's optional k2 integration is a LazyModule, so that probe can
    try to import k2 even though diarization does not use it.
    """
    for name in list(sys.modules):
        if name.startswith("speechbrain.integrations.k2_fsa"):
            sys.modules.pop(name, None)
    sys.modules.setdefault("k2", types.ModuleType("k2"))
    try:
        from speechbrain.utils.importutils import LazyModule

        if not getattr(LazyModule, "_asr_file_probe_patch", False):
            original_getattr = LazyModule.__getattr__

            def _safe_getattr(self, attr):
                if attr == "__file__":
                    raise AttributeError(attr)
                return original_getattr(self, attr)

            LazyModule.__getattr__ = _safe_getattr
            LazyModule._asr_file_probe_patch = True
    except Exception:
        pass


@dataclass
class DiarizationSegment:
    start: float
    end: float
    speaker: str
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "speaker": self.speaker,
        }
        if self.confidence is not None:
            d["confidence"] = round(self.confidence, 4)
        return d


class PyannoteDiarizationBackend:
    """Pyannote speaker diarization wrapper.

    Automatically detects loading mode:
    - **Local**  (``pipeline_config`` exists as a file on disk)
    - **Cloud**  (``pipeline_config`` is a HuggingFace repo id)

    Paths inside local config YAML are relative to the **project root**
    directory (pyannote resolves them against CWD, which is set to the
    project root during loading).
    """

    def __init__(
        self,
        pipeline_config: str | Path,
        device: str = "cuda",
    ) -> None:
        self.pipeline_config = str(pipeline_config)
        self.device = device if torch.cuda.is_available() else "cpu"
        self._pipeline: Any = None

    @property
    def _is_local(self) -> bool:
        """True when ``pipeline_config`` is a file path that exists on disk."""
        try:
            return Path(self.pipeline_config).exists()
        except OSError:
            return False

    def load(self) -> Any:
        """Load the pyannote Pipeline (idempotent).

        For **local** configs the working directory is temporarily switched
        to the project root so that relative model paths inside
        ``config.yaml`` are resolved correctly.

        For **cloud** repos the ``HF_TOKEN`` env var is used for
        authentication (required for gated models like
        ``pyannote/segmentation-3.0`` and
        ``pyannote/wespeaker-voxceleb-resnet34-LM``).
        """
        if self._pipeline is not None:
            return self._pipeline

        from pyannote.audio import Pipeline
        _drop_optional_speechbrain_lazy_modules()

        from src.utils.config import project_root, get_env

        if self._is_local:
            # ── Local mode ───────────────────────────────────────────
            # Switch to project root so relative paths in config.yaml
            # resolve correctly (official pyannote convention).
            cwd = str(Path.cwd().resolve())
            root = str(project_root())

            try:
                os.chdir(root)
                self._pipeline = Pipeline.from_pretrained(self.pipeline_config)
            finally:
                os.chdir(cwd)

        else:
            # ── Cloud / HF Hub mode ──────────────────────────────────
            hf_token = get_env("HF_TOKEN")
            self._pipeline = Pipeline.from_pretrained(
                self.pipeline_config,
                use_auth_token=hf_token,
            )

        self._pipeline.to(torch.device(self.device))
        return self._pipeline

    def diarize(
        self,
        audio: str | Path | np.ndarray,
        sample_rate: int = 16000,
        **_: Any,
    ) -> dict[str, Any]:
        """Run speaker diarization on an audio file or waveform.

        Parameters
        ----------
        audio : str, Path, or np.ndarray
            Path to an audio file, or a mono float32 waveform in [-1, 1].
        sample_rate : int
            Sample rate of the waveform.  When ``audio`` is a file path
            the actual file sample rate is detected and used.

        Returns
        -------
        dict with keys ``segments`` and ``num_speakers``.
        """
        pipeline = self.load()

        # ── Prepare input ────────────────────────────────────────────
        if isinstance(audio, (str, Path)):
            import soundfile as sf

            waveform, file_sr = sf.read(str(audio))
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            waveform = torch.from_numpy(waveform.astype(np.float32))
            sample_rate = file_sr
        elif isinstance(audio, np.ndarray):
            waveform = torch.from_numpy(audio)
        else:
            raise TypeError(f"Unsupported audio type: {type(audio)}")

        diarization = pipeline({
            "waveform": waveform[None, :],
            "sample_rate": sample_rate,
        })

        # ── Extract segments ─────────────────────────────────────────
        segments: list[dict[str, Any]] = []
        for speech_turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append(
                DiarizationSegment(
                    start=float(speech_turn.start),
                    end=float(speech_turn.end),
                    speaker=speaker,
                ).to_dict()
            )

        segments.sort(key=lambda s: s["start"])
        return {
            "segments": segments,
            "num_speakers": len({s["speaker"] for s in segments}),
        }

    def unload(self) -> None:
        """Release GPU memory."""
        if self._pipeline is not None:
            self._pipeline = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def load_diarization_backend(cfg: dict[str, Any]) -> PyannoteDiarizationBackend:
    """Factory: create a diarization backend from the resolved config block.

    ``cfg`` is the ``diarization`` block from ``configs/models.yaml`` after
    resolution, plus an injected ``device`` key.

    The block provides at least ``pipeline_config``.
    ``segmentation_model`` is available but not directly consumed by the
    backend — pyannote reads it from the pipeline's own ``config.yaml``.
    """
    return PyannoteDiarizationBackend(
        pipeline_config=cfg["pipeline_config"],
        device=cfg.get("device", "cuda"),
    )
