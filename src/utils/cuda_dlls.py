"""Windows-only DLL search path fix for ctranslate2 / faster-whisper.

On Windows, ctranslate2 cannot locate the cuDNN / cuBLAS / NVRTC DLLs that
the `nvidia-*-cu12` pip wheels install under `site-packages/nvidia/*/bin`.
We register those directories via `os.add_dll_directory` at process start.

Call `add_cuda_dll_dirs()` BEFORE `import faster_whisper` (and before any
WhisperX import, since whisperx wraps faster-whisper).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ADDED = False


def add_cuda_dll_dirs() -> list[str]:
    """Register NVIDIA wheel DLL dirs.

    IMPORT ORDER RULE (Windows only):
        Always import `faster_whisper` / `whisperx` BEFORE any module that
        pulls in `pyannote.audio` (which transitively imports onnxruntime).
        Once onnxruntime is loaded it pins the process to a cuDNN that
        ctranslate2 cannot then initialise, producing WinError 1114.

        In pipeline code:
            import torch
            from src.utils import add_cuda_dll_dirs; add_cuda_dll_dirs()
            from faster_whisper import WhisperModel       # FIRST
            from pyannote.audio import Pipeline           # SECOND

    Idempotent. Returns the list of directories actually added.
    """
    global _ADDED
    if _ADDED or sys.platform != "win32":
        return []
    site = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    added: list[str] = []
    for sub in ("cudnn/bin", "cublas/bin", "cuda_runtime/bin", "cuda_nvrtc/bin"):
        p = site / sub
        if p.exists():
            os.add_dll_directory(str(p))
            added.append(str(p))
    _ADDED = True
    return added
