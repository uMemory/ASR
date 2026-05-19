"""Demucs BGM / vocal separation frontend.

Wraps the ``htdemucs_ft`` model to extract vocals from mixed audio
(e.g. film/TV scenes with background music).  For pure meeting speech
this stage can be safely skipped.

Usage:
    from src.frontend import load_separator
    sep = load_separator(cfg["separation"] | {"device": "cuda"})
    sep.load()
    vocals = sep.separate("audio_with_bgm.wav")   # → numpy float32 mono
    sep.unload()
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class SeparationResult:
    waveform: np.ndarray   # float32, mono, in [-1, 1]
    sample_rate: int       # model native sample rate (44100 for htdemucs_ft)

    def to_mono_16k(self) -> np.ndarray:
        """Downmix to 16 kHz mono for downstream pipeline stages."""
        import librosa
        if self.sample_rate == 16000:
            return self.waveform.astype(np.float32)
        return librosa.resample(
            self.waveform, orig_sr=self.sample_rate, target_sr=16000
        ).astype(np.float32)


class DemucsSeparator:
    """Demucs htdemucs_ft vocal separator.

    Parameters
    ----------
    model : str
        Demucs model name (``"htdemucs_ft"`` by default).
    two_stems : str
        Target stem (``"vocals"`` or ``"all"``).  When ``"vocals"`` only
        the vocal track is returned.
    device : str
        ``"cuda"`` or ``"cpu"``.
    """

    def __init__(
        self,
        model: str = "htdemucs_ft",
        two_stems: str = "vocals",
        device: str = "cuda",
    ) -> None:
        self._model_name = model
        self._two_stems = two_stems
        self.device = device if torch.cuda.is_available() else "cpu"
        self._model: Any = None

    def load(self) -> Any:
        """Load the Demucs model (idempotent).  Downloads on first run."""
        if self._model is not None:
            return self._model

        from demucs.pretrained import get_model

        self._model = get_model(name=self._model_name)
        self._model.to(self.device)
        self._model.eval()
        return self._model

    def separate(self, audio: str | Path | np.ndarray) -> SeparationResult:
        """Extract vocals from an audio file or waveform.

        Parameters
        ----------
        audio : str, Path, or np.ndarray
            Path to an audio file, or a mono float32 waveform in [-1, 1].
            If a file path, the file is loaded and resampled automatically.

        Returns
        -------
        SeparationResult
            Contains ``.waveform`` (mono float32) and ``.sample_rate``.
            Call ``.to_mono_16k()`` to get a 16 kHz mono waveform for
            downstream processing.
        """
        model = self.load()

        from demucs.separate import load_track, apply_model

        # ── Load audio ────────────────────────────────────────────────
        if isinstance(audio, np.ndarray):
            wav = audio
            file_sr = 16000  # assume 16k for numpy input
        else:
            wav, file_sr = load_track(
                str(audio), model.audio_channels, model.samplerate
            )

        # ── Convert to stereo if needed ───────────────────────────────
        wav_tensor = torch.from_numpy(wav).float()
        if wav_tensor.ndim == 1:
            wav_tensor = wav_tensor.unsqueeze(0).repeat(2, 1)   # mono → stereo
        elif wav_tensor.shape[0] == 1:
            wav_tensor = wav_tensor.repeat(2, 1)
        # shape: [2, num_samples]

        # ── Normalize (Demucs expects zero-mean unit-variance) ────────
        ref = wav_tensor.mean(0)
        wav_tensor = (wav_tensor - ref.mean()) / (ref.std() + 1e-8)

        # ── Run separation ────────────────────────────────────────────
        sources = apply_model(
            model,
            wav_tensor[None],     # add batch dim → [1, 2, samples]
            device=self.device,
            progress=True,
            shifts=1,
            split=True,
            overlap=0.25,
        )
        # sources shape: [1, num_stems, 2, samples]

        # ── Extract target stem ───────────────────────────────────────
        if self._two_stems == "vocals":
            # vocals is the LAST stem in htdemucs_ft output
            stem_idx = -1
        else:
            stem_idx = 0  # return first stem (unused for now)

        # Extract mono from stereo output
        stem_wave = sources[0, stem_idx]          # [2, samples]
        vocals = stem_wave.mean(dim=0).cpu().numpy().astype(np.float32)

        return SeparationResult(
            waveform=vocals,
            sample_rate=model.samplerate,          # 44100 for htdemucs_ft
        )

    def unload(self) -> None:
        """Release GPU memory."""
        if self._model is not None:
            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def load_separator(cfg: dict[str, Any]) -> DemucsSeparator:
    """Factory: create a Demucs separator from a resolved config block.

    ``cfg`` should contain at least:
        model     : str   (model name, e.g. ``"htdemucs_ft"``)
        two_stems : str   (target stem)
        device    : str   (``"cuda"`` or ``"cpu"``)
    """
    return DemucsSeparator(
        model=cfg.get("model", "htdemucs_ft"),
        two_stems=cfg.get("two_stems", "vocals"),
        device=cfg.get("device", "cuda"),
    )
