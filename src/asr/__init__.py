"""ASR backends. Factory dispatches by config `backend` field."""
from __future__ import annotations

from typing import Any


def load_asr_backend(cfg: dict[str, Any]):
    """cfg is the resolved `asr` block from configs/models.yaml + injected `device`."""
    backend = cfg.get("backend", "transformers")
    device = cfg.get("device", "cuda")
    compute_type = cfg.get("compute_type", "float16")

    if backend == "transformers":
        from .transformers_whisper_backend import TransformersWhisperBackend
        return TransformersWhisperBackend(
            model_path=cfg["model"],
            device=device,
            compute_type=compute_type,
        )
    if backend == "openai-whisper":
        # Legacy fallback. Kept around in case transformers path misbehaves.
        from .openai_whisper_backend import OpenAIWhisperBackend
        return OpenAIWhisperBackend(
            model_path=cfg["model"],
            device=device,
            compute_type=compute_type,
        )
    if backend == "faster-whisper":
        raise NotImplementedError("faster-whisper backend deferred to cloud profile")
    raise ValueError(f"Unknown ASR backend: {backend}")
