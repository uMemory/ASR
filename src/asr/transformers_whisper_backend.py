"""Whisper ASR via HuggingFace `transformers`.

Loads Whisper from a HuggingFace snapshot (either a local directory like
`./models/whisper-medium` or a HF repo id like `openai/whisper-large-v3`).

Why this backend instead of openai-whisper:
- Same code path local (medium) and cloud (large-v3).
- Reads HF snapshot natively — no .pt conversion.
- No ctranslate2 / cuDNN DLL hassle on Windows.
- Built-in long-form chunking via the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)


@dataclass
class ASRSegment:
    start: float
    end: float
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text.strip(),
        }


class TransformersWhisperBackend:
    """HuggingFace `transformers` Whisper wrapper with chunked long-form ASR."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cuda",
        compute_type: str = "float16",
        chunk_length_s: float = 30.0,
        batch_size: int = 8,
    ) -> None:
        self.model_path = str(model_path)
        self.device = device if torch.cuda.is_available() else "cpu"
        self.torch_dtype = (
            torch.float16 if (compute_type == "float16" and self.device == "cuda")
            else torch.float32
        )
        self.chunk_length_s = chunk_length_s
        self.batch_size = batch_size
        self._pipe = None

    def load(self):
        if self._pipe is not None:
            return self._pipe
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(self.device)
        processor = AutoProcessor.from_pretrained(self.model_path)
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            chunk_length_s=self.chunk_length_s,
            batch_size=self.batch_size,
            torch_dtype=self.torch_dtype,
            device=self.device,
            return_timestamps=True,
        )
        return self._pipe

    def transcribe(
        self,
        audio: str | Path | np.ndarray,
        language: str | None = None,
        initial_prompt: str | None = None,
        sample_rate: int = 16000,
        **_: Any,
    ) -> dict[str, Any]:
        pipe = self.load()

        generate_kwargs: dict[str, Any] = {}
        # Long-form Whisper decoding guardrails recommended by the
        # Transformers Whisper docs. They reduce silence hallucinations and
        # repeated loops without relying on dataset-specific text rules.
        generate_kwargs.update({
            "condition_on_prev_tokens": False,
            "compression_ratio_threshold": 1.35,
            "logprob_threshold": -1.0,
            "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            "return_timestamps": True,
        })
        if language:
            generate_kwargs["language"] = language
            generate_kwargs["task"] = "transcribe"
        if initial_prompt:
            # Whisper's `prompt_ids` mechanism — biases the decoder.
            prompt_ids = pipe.tokenizer.get_prompt_ids(
                initial_prompt, return_tensors="pt"
            ).to(self.device)
            generate_kwargs["prompt_ids"] = prompt_ids

        if isinstance(audio, np.ndarray):
            audio_input: Any = {"raw": audio, "sampling_rate": sample_rate}
        else:
            audio_input = str(audio)

        result = pipe(audio_input, generate_kwargs=generate_kwargs)
        chunks = result.get("chunks", [])
        segments = [
            ASRSegment(
                start=(c["timestamp"][0] or 0.0),
                end=(c["timestamp"][1] or 0.0),
                text=c["text"],
            ).to_dict()
            for c in chunks
        ]
        return {
            "text": result.get("text", ""),
            "language": language,
            "segments": segments,
        }

    def unload(self) -> None:
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
