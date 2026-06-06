"""Lightweight realtime microphone ASR backend used by the Gradio app.

Realtime mode intentionally avoids diarization and LLM correction. Speaker
diarization needs longer context and is too slow/unstable for low-latency
microphone feedback; the saved recording can still be refined offline through
the full pipeline.
"""
from __future__ import annotations

import torch
import numpy as np


class RealtimePipeline:
    def __init__(self, language: str = "zh", enable_llm: bool = False) -> None:
        from src.utils.config import get_model_config, load_config

        mcfg = get_model_config()
        lcfg = load_config("languages")
        self.language = language
        self.enable_llm = False
        self.device = mcfg["device"]
        self.lang_cfg = lcfg.get(language, {})

        from src.asr import load_asr_backend
        self.asr_backend = load_asr_backend({**mcfg["asr"], "device": self.device})
        self.asr_backend.load()

    def process_fast(self, waveform: np.ndarray) -> list[dict]:
        if waveform.size == 0:
            return []

        sr = 16000
        prompt = self.lang_cfg.get("asr_initial_prompt")

        asr = self.asr_backend.transcribe(
            waveform,
            language=self.language,
            initial_prompt=prompt,
            sample_rate=sr,
        )

        from src.pipeline import (
            _adjust_neighbor_playback_boundaries,
            _filter_asr_artifacts,
            _filter_asr_by_speech_activity,
            _filter_unreliable_turns,
            _presplit_segments,
            _turn_limits_for_language,
        )
        asr_segs = _filter_asr_artifacts(asr["segments"])
        asr_segs = _filter_asr_by_speech_activity(asr_segs, waveform, sr)

        from src.llm.corrector import clean_hallucination, zh_simplify
        from src.pipeline import _dedupe_turn_boundaries, _merge_adjacent_turns
        turn_limits = _turn_limits_for_language(self.language)
        merged = _presplit_segments(
            asr_segs,
            max_dur=float(turn_limits["max_dur"]),
            max_chars=int(turn_limits["max_chars"]),
        )
        merged = _filter_unreliable_turns(merged)
        for seg in merged:
            txt = clean_hallucination(seg.get("text", ""))
            seg["text"] = zh_simplify(txt)
            seg["speaker"] = "LIVE"
            seg["speaker_original"] = "LIVE"
        merged = _merge_adjacent_turns(
            merged,
            max_gap=float(turn_limits["merge_gap"]),
            max_dur=float(turn_limits["merge_max_dur"]),
            max_chars=int(turn_limits["merge_max_chars"]),
        )
        merged = _dedupe_turn_boundaries(merged)
        merged = _adjust_neighbor_playback_boundaries(merged)
        return merged

    def process_llm(self, merged: list[dict]) -> list[dict]:
        return merged

    def unload(self) -> None:
        self.asr_backend.unload()
        del self.asr_backend
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
