"""Full pipeline orchestrator."""
from __future__ import annotations

import time
import warnings
# 屏蔽已知的库警告（torch/pyannote/speechbrain/lightning 无力修改的噪音）
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.models.whisper")
warnings.filterwarnings("ignore", category=FutureWarning, module="lightning_fabric")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote.audio")
warnings.filterwarnings("ignore", category=UserWarning, module="speechbrain")
warnings.filterwarnings("ignore", message=".*weights_only.*")
from pathlib import Path
from typing import Any

from src.utils.config import get_model_config, load_config
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SPEAKER_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _aggregate_words_to_turns(
    words: list[dict[str, Any]],
    max_gap: float = 0.5,
    max_dur: float = 15.0,
    max_chars: int = 80,
) -> list[dict[str, Any]]:
    """Aggregate word-level segments into speaker turns.

    Rules:
    - Same speaker + gap < *max_gap* → merge
    - Speaker change or gap > *max_gap* → split
    - Turn exceeds *max_dur* or *max_chars* → force-split
    """
    turns: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for w in words:
        spk = w.get("speaker", "SPEAKER_UNKNOWN")
        w_start = float(w["start"])
        w_end = float(w["end"])
        w_text = str(w.get("word", ""))

        need_new = (
            current is None
            or spk != current["speaker"]
            or w_start - current["end"] > max_gap
            or (current["end"] - current["start"]) >= max_dur
            or len(current["text"]) >= max_chars
        )

        if need_new:
            if current:
                turns.append(current)
            current = {"start": w_start, "end": w_end, "speaker": spk, "text": w_text}
        else:
            current["end"] = w_end
            current["text"] += w_text

    if current:
        turns.append(current)
    return turns


def _build_speaker_map(segments: list[dict]) -> dict[str, str]:
    """构建 SPEAKER_XX → A,B,C 映射。

    按首次出现时间排序，SPEAKER_UNKNOWN 排到最后。
    """
    seen: dict[str, float] = {}
    for seg in segments:
        spk = seg.get("speaker", "")
        if spk and spk not in seen:
            seen[spk] = seg.get("start", float("inf"))

    # 按首次出现排序，UNKNOWN 排最后
    sorted_spks = sorted(seen.keys(),
                         key=lambda s: (1e9 if "UNKNOWN" in s.upper() else 0, seen[s]))

    return {spk: _SPEAKER_LABELS[i] if i < 26 else f"S{i-25}"
            for i, spk in enumerate(sorted_spks)}


def run(
    audio_path: str | Path,
    language: str | None = None,
    profile: str | None = None,
    max_duration_s: float | None = None,
    llm_overrides: dict[str, bool] | None = None,
    llm_model: str | None = None,   # override model name for LLM
    waveform: Any = None,           # np.ndarray, float32, mono — bypasses file I/O
    sample_rate: int | None = None, # required with waveform
) -> dict[str, Any]:
    """Run the full ASR + diarization pipeline on a single audio file.

    Parameters
    ----------
    audio_path : str or Path
        Path to an audio file (WAV, FLAC, …).  Used as metadata label
        when ``waveform`` is provided.
    language : str or None
        Force ASR language (e.g. ``"zh"``, ``"en"``).  ``None`` = auto-detect.
    profile : str or None
        Config profile (``"local"``, ``"cloud"``).  Defaults to the YAML
        ``profile`` field.
    max_duration_s : float or None
        Truncate audio to the first *N* seconds for quick testing.
    llm_overrides : dict or None
        Optional LLM config overrides, e.g.
        ``{"enabled": True, "correction": False, "intent_tagging": True}``.
        Overrides the values from ``configs/models.yaml`` ``llm`` section.
    waveform : np.ndarray or None
        Pre-loaded mono float32 waveform in [-1, 1].  When given, file
        I/O is skipped and ``sample_rate`` must also be provided.
    sample_rate : int or None
        Required when ``waveform`` is not ``None``.

    Returns
    -------
    dict with keys:

        segments : list[dict]
            Merged diarization × ASR segments with ``start``, ``end``,
            ``speaker``, ``text``.
        num_speakers : int
        language : str or None
        timing : dict
            Per-stage wall-clock times.
    """
    timing: dict[str, float] = {}
    t_start = time.time()

    # ---- 0. Load config ---------------------------------------------------
    mcfg = get_model_config(profile)
    lcfg = load_config("languages")
    device = mcfg["device"]

    if language is None:
        language = lcfg.get("default", "zh")

    logger.info(
        "Pipeline start  profile=%s  device=%s  language=%s  audio=%s",
        mcfg["profile"], device, language, audio_path,
    )

    # ---- Load audio once (before any pyannote import) ----------------------
    import soundfile as sf  # soundfile BEFORE pyannote to avoid speechbrain conflict
    import torch

    if waveform is not None:
        if sample_rate is None:
            raise ValueError("sample_rate is required when waveform is provided")
        waveform = waveform.astype("float32")
        sr = sample_rate
        if max_duration_s is not None:
            n = int(max_duration_s * sr)
            waveform = waveform[:n]
    else:
        waveform, sr = sf.read(str(audio_path))
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        waveform = waveform.astype("float32")
        if max_duration_s is not None:
            n = int(max_duration_s * sr)
            waveform = waveform[:n]

    logger.info("Audio loaded  duration=%.1fs  sr=%d", len(waveform) / sr, sr)

    # ---- 1. (optional) BGM / vocal separation -------------------------
    sep_cfg = mcfg.get("separation", {})
    if sep_cfg.get("enabled", False):
        logger.info("Stage 1: BGM vocal separation …")
        t1 = time.time()

        from src.frontend import load_separator

        sep = load_separator({**sep_cfg, "device": device})
        sep.load()
        sr_result = sep.separate(waveform)
        waveform = sr_result.to_mono_16k()
        sr = 16000
        sep.unload()
        del sep

        timing["separation"] = round(time.time() - t1, 2)
        logger.info("  → vocals extracted (%.1fs)", timing["separation"])

    # ---- 2. Speaker diarization -----------------------------------------
    logger.info("Stage 2: diarization …")
    t2 = time.time()

    from src.diarization import load_diarization_backend

    dia_cfg = {**mcfg["diarization"], "device": device}
    dia_backend = load_diarization_backend(dia_cfg)
    dia_backend.load()
    dia_result = dia_backend.diarize(waveform, sample_rate=sr)

    timing["diarization"] = round(time.time() - t2, 2)
    logger.info("  → %d segments, %d speakers (%.1fs)",
                len(dia_result["segments"]),
                dia_result["num_speakers"],
                timing["diarization"])

    # ---- 3. ASR ----------------------------------------------------------
    logger.info("Stage 3: ASR …")
    t3 = time.time()

    from src.asr import load_asr_backend

    asr_cfg = {**mcfg["asr"], "device": device}

    asr_backend = load_asr_backend(asr_cfg)
    asr_backend.load()

    lang_cfg = lcfg.get(language, {})
    initial_prompt = lang_cfg.get("asr_initial_prompt")

    asr_result = asr_backend.transcribe(
        waveform,
        language=language,
        initial_prompt=initial_prompt,
    )

    asr_backend.unload()
    timing["asr"] = round(time.time() - t3, 2)
    logger.info("  → %d segments (%.1fs)", len(asr_result["segments"]), timing["asr"])

    # ---- 4. Word-level speaker assignment + turn aggregation -------------
    logger.info("Stage 4: word-level speaker assignment …")
    t4 = time.time()

    import pandas as pd

    # 4a. Convert ASR word chunks to whisperx-compatible format
    asr_words = asr_result["segments"]  # word-level [{start, end, text}, …]
    word_segments: list[dict[str, Any]] = []
    for w in asr_words:
        txt = w.get("text", "").strip()
        if not txt:
            continue
        word_segments.append({
            "word": txt,
            "start": w["start"],
            "end": w["end"],
            "score": 0.5,
        })

    # Build a synthetic segment wrapping all words — assign_word_speakers
    # does per-word overlap matching, so the segment grouping is cosmetic.
    aligned_for_ws: dict[str, Any] = {
        "segments": [{
            "start": word_segments[0]["start"] if word_segments else 0.0,
            "end": word_segments[-1]["end"] if word_segments else 0.0,
            "text": " ".join(w["word"] for w in word_segments),
            "words": word_segments,
        }],
        "word_segments": word_segments,
    }

    # 4b. Diarization → pandas DataFrame (whisperx format)
    dia_segs = dia_result["segments"]
    diarize_df = pd.DataFrame([
        {"start": s["start"], "end": s["end"], "speaker": s["speaker"]}
        for s in dia_segs
    ])

    # 4c. Per-word speaker assignment (no faster-whisper dependency)
    import whisperx
    result_with_spk = whisperx.assign_word_speakers(diarize_df, aligned_for_ws)
    speaker_words = result_with_spk.get("word_segments", [])

    # 4d. Aggregate words → speaker turns
    merged = _aggregate_words_to_turns(speaker_words)
    if not merged:
        logger.warning("  No speaker turns produced; falling back to raw ASR")
        merged = asr_words

    # 4e. Hallucination cleanup + zh_simplify + speaker map
    from src.llm.corrector import clean_hallucination, zh_simplify
    for seg in merged:
        txt = clean_hallucination(seg.get("text", ""))
        seg["text"] = zh_simplify(txt)

    speaker_map = _build_speaker_map(merged)
    for seg in merged:
        old_spk = seg.get("speaker", "")
        seg["speaker"] = speaker_map.get(old_spk, old_spk)
        seg["speaker_original"] = old_spk

    timing["alignment"] = round(time.time() - t4, 2)
    logger.info("  → %d speaker turns from %d words (%.1fs)",
                len(merged), len(speaker_words), timing["alignment"])

    # ---- Cleanup (free GPU before LLM API calls) ------------------------
    dia_backend.unload()
    del dia_backend, asr_backend, waveform
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.debug("GPU models unloaded")

    # ---- 5. LLM post-processing -----------------------------------------
    llm_cfg = dict(mcfg.get("llm", {}))
    if llm_overrides:
        llm_cfg.update(llm_overrides)
    meeting_summary: dict[str, Any] = {}
    if llm_cfg.get("enabled", True):
        logger.info("Stage 5: LLM post-processing …")
        t5 = time.time()

        from src.llm import (
            load_llm_adapter,
            llm_correct_segments,
            llm_check_consistency,
            llm_tag_intents,
        )

        adapter = load_llm_adapter(model=llm_model)
        logger.info("  Backend: %s (model=%s)", adapter.model_name,
                     getattr(adapter, '_model', '?'))

        # 5a. Context-aware correction
        if llm_cfg.get("correction", True):
            merged = llm_correct_segments(adapter, merged, language)

        # 5b. Speaker consistency
        if llm_cfg.get("consistency", True):
            merged = llm_check_consistency(adapter, merged, language)

        # 5c. Intent tagging
        if llm_cfg.get("intent_tagging", True):
            intent_labels = lang_cfg.get("intent_labels")
            merged = llm_tag_intents(adapter, merged, language, intent_labels)

        # 5d. Meeting summary (optional, non-blocking)
        meeting_summary: dict[str, Any] = {}
        try:
            from src.llm.summarizer import generate_summary
            meeting_summary = generate_summary(adapter, merged, language)
            logger.info("  Summary: %s", meeting_summary.get("topic", "")[:60])
        except Exception as e:
            logger.warning("  Summary skipped: %s", e)

        timing["llm"] = round(time.time() - t5, 2)
        logger.info("  → LLM done (%.1fs)", timing["llm"])

    # ---- 6. Build retrieval index -------------------------------------
    ret_cfg = mcfg.get("retrieval", {})
    if ret_cfg.get("build_index", False) and merged:
        logger.info("Stage 6: 构建检索索引 …")
        t6 = time.time()

        from src.retrieval import EmbeddingEncoder, build_index

        emb_cfg = {**mcfg["embedding"], "device": device}
        encoder = EmbeddingEncoder(**emb_cfg)
        encoder.load()

        store_path = ret_cfg.get("store_path", "./outputs/index")
        from src.utils.config import project_root as _root
        store_path = str(_root() / store_path)

        try:
            _, _, _ = build_index(merged, encoder, store_path=store_path)
        finally:
            encoder.unload()
            del encoder

        timing["retrieval_index"] = round(time.time() - t6, 2)
        logger.info("  → 索引已保存到 %s (%.1fs)", store_path,
                     timing["retrieval_index"])

    timing["total"] = round(time.time() - t_start, 2)
    logger.info("Pipeline done  total=%.1fs", timing["total"])

    return {
        "segments": merged,
        "num_speakers": dia_result["num_speakers"],
        "meeting_summary": meeting_summary,
        "language": asr_result.get("language", language),
        "timing": timing,
    }
