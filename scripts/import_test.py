"""Quick import smoke test for all core deps (no traceback to avoid Windows TxF lock)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.cuda_dlls import add_cuda_dll_dirs
add_cuda_dll_dirs()


def t(name, fn):
    try:
        fn()
        print(f"[OK]   {name}")
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")

import torch
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
import numpy as np
print(f"numpy={np.__version__}")

t("librosa", lambda: __import__("librosa"))
t("whisper (openai)", lambda: __import__("whisper"))
t("pyannote.audio", lambda: __import__("pyannote.audio"))
# faster_whisper is retained for WhisperX import compatibility, but is not the
# current local ASR backend.
t("faster_whisper.WhisperModel", lambda: __import__("faster_whisper", fromlist=["WhisperModel"]))
t("whisperx", lambda: __import__("whisperx"))
t("transformers.AutoModel", lambda: __import__("transformers", fromlist=["AutoModel"]))
t("gradio", lambda: __import__("gradio"))
t("websockets", lambda: __import__("websockets"))
t("sounddevice", lambda: __import__("sounddevice"))
t("anthropic", lambda: __import__("anthropic"))
t("openai", lambda: __import__("openai"))
t("torchaudio", lambda: __import__("torchaudio"))
t("soundfile", lambda: __import__("soundfile"))
t("jiwer", lambda: __import__("jiwer"))
t("textgrid", lambda: __import__("textgrid"))
t("rank_bm25", lambda: __import__("rank_bm25"))
t("faiss", lambda: __import__("faiss"))
t("onnxruntime", lambda: __import__("onnxruntime"))
t("pandas", lambda: __import__("pandas"))
t("pyarrow", lambda: __import__("pyarrow"))
t("omegaconf", lambda: __import__("omegaconf"))
t("ffmpeg", lambda: __import__("ffmpeg"))
