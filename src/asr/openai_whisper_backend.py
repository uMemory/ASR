"""openai-whisper backend.

Used for local debugging on Windows where the faster-whisper / ctranslate2 path
hits cuDNN DLL conflicts with pyannote's onnxruntime. Slower than CTranslate2
(~3x) but rock-solid: pure PyTorch, no native deps beyond what torch already
ships.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import whisper


@dataclass
class ASRSegment:
    start: float
    end: float
    text: str
    avg_logprob: float | None = None
    no_speech_prob: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text.strip(),
            "avg_logprob": self.avg_logprob,
            "no_speech_prob": self.no_speech_prob,
        }


class OpenAIWhisperBackend:
    """Thin wrapper around openai-whisper for transcription with segments."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda",
        compute_type: str = "float16",
    ) -> None:
        self.model_path = str(model_path)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.fp16 = (compute_type == "float16") and (self.device == "cuda")
        self._model: whisper.Whisper | None = None

    def load(self) -> whisper.Whisper:
        if self._model is None:
            # whisper.load_model accepts either a known name ("medium") or a
            # path to a .pt checkpoint. The HF snapshot directory at
            # ./models/whisper-medium is NOT a .pt file — fall back to the
            # canonical name and let whisper download / cache via its own
            # mechanism (already cached if the user ran it before).
            # If a local .pt exists under model_path, prefer it.
            local_pt = Path(self.model_path) / "model.pt"
            name_or_path = str(local_pt) if local_pt.exists() else self._canonical_name()
            self._model = whisper.load_model(name_or_path, device=self.device)
        return self._model

    def _canonical_name(self) -> str:
        p = Path(self.model_path).name.lower()
        for size in ("tiny", "base", "small", "medium", "large-v3", "large-v2", "large"):
            if size in p:
                return size
        return "medium"

    def transcribe(
        self,
        audio: str | Path | np.ndarray,
        language: str | None = None,
        initial_prompt: str | None = None,
        beam_size: int = 5,
        verbose: bool = False,
    ) -> dict[str, Any]:
        model = self.load()
        result = model.transcribe(
            str(audio) if not isinstance(audio, np.ndarray) else audio,
            language=language,
            initial_prompt=initial_prompt,
            beam_size=beam_size,
            fp16=self.fp16,
            verbose=verbose,
        )
        segments = [
            ASRSegment(
                start=s["start"],
                end=s["end"],
                text=s["text"],
                avg_logprob=s.get("avg_logprob"),
                no_speech_prob=s.get("no_speech_prob"),
            ).to_dict()
            for s in result.get("segments", [])
        ]
        return {
            "text": result.get("text", ""),
            "language": result.get("language"),
            "segments": segments,
        }

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
