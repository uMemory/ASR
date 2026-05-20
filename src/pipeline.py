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


def _filter_asr_artifacts(
    segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove ASR hallucination segments before alignment.

    Known Whisper artifacts for Chinese: 【翻譯】, brackets-only text,
    extremely short text spanning very long durations.
    """
    import re as _re
    from src.llm.corrector import clean_hallucination

    cleaned: list[dict[str, Any]] = []
    for seg in segments:
        txt = clean_hallucination(seg.get("text", "").strip())
        if not txt:
            continue
        # Remove known hallucination pattern: 【翻譯】 or similar bracket artifacts
        txt = _re.sub(r'[【\[].*?[】\]]', '', txt).strip()
        if not txt:
            continue
        # Filter punctuation-only artifacts, but keep numeric-only content:
        # meeting recordings can contain real spoken IDs or agenda numbers.
        if _re.fullmatch(r'[\s\W_]+', txt):
            continue
        # Filter: single char spanning >10s (alignment artifact risk)
        dur = seg["end"] - seg["start"]
        if dur <= 0:
            continue
        if len(txt) <= 2 and dur > 10:
            continue
        seg["text"] = txt
        cleaned.append(seg)
    return cleaned


def _presplit_segments(
    segments: list[dict[str, Any]],
    max_dur: float = 15.0,
    max_chars: int = 80,
) -> list[dict[str, Any]]:
    """Split long ASR segments into shorter chunks for forced alignment.

    Does NOT require punctuation — uses hard character/duration limits.
    The forced alignment step will refine word timestamps afterwards.
    """
    import math
    result: list[dict[str, Any]] = []
    for seg in segments:
        txt = seg.get("text", "")
        dur = float(seg["end"] - seg["start"])
        if dur <= max_dur and len(txt) <= max_chars:
            result.append(seg)
            continue
        n_chunks = max(1, max(math.ceil(len(txt) / max_chars), math.ceil(dur / max_dur)))
        chars_per = math.ceil(len(txt) / n_chunks)
        dur_per = dur / n_chunks
        t = float(seg["start"])
        for i in range(n_chunks):
            chunk = txt[i * chars_per: (i + 1) * chars_per]
            if not chunk.strip():
                continue
            result.append({**seg, "text": chunk.strip(), "start": round(t, 3),
                           "end": round(t + dur_per, 3)})
            t += dur_per
    return result


def _resample_to_16k(waveform: Any, sr: int) -> tuple[Any, int]:
    """Return mono float32 audio at 16 kHz.

    Whisper, WhisperX alignment, realtime browser audio, and most local
    diarization configs are expected to share the same 16 kHz timeline.
    Keeping one sample rate prevents ASR/diarization/alignment drift.
    """
    import numpy as np

    waveform = waveform.astype("float32")
    if sr == 16000:
        return waveform, sr

    import librosa

    waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    return waveform.astype(np.float32), 16000


def _energy_speech_regions(waveform: Any, sr: int) -> list[dict[str, float]]:
    """Detect speech-like regions from the waveform, independent of diarization."""
    import numpy as np

    duration_s = len(waveform) / sr
    frame_len = max(1, int(0.10 * sr))
    hop_len = max(1, int(0.05 * sr))
    if len(waveform) < frame_len:
        return []

    rms_vals: list[float] = []
    starts: list[int] = []
    for lo in range(0, len(waveform) - frame_len + 1, hop_len):
        frame = waveform[lo:lo + frame_len]
        rms_vals.append(float(np.sqrt(np.mean(frame * frame))))
        starts.append(lo)

    if not rms_vals:
        return []

    rms = np.asarray(rms_vals, dtype=np.float32)
    max_rms = float(np.max(rms))
    if max_rms <= 1e-6:
        return []

    threshold = max(max_rms * (10 ** (-35 / 20)), 0.0015)
    active = rms >= threshold

    regions: list[dict[str, float]] = []
    region_start: int | None = None
    region_end = 0
    for is_active, lo in zip(active.tolist(), starts):
        if is_active:
            if region_start is None:
                region_start = lo
            region_end = lo + frame_len
        elif region_start is not None:
            start = max(0.0, region_start / sr)
            end = min(duration_s, region_end / sr)
            if end - start >= 0.12:
                regions.append({"start": start, "end": end})
            region_start = None
    if region_start is not None:
        start = max(0.0, region_start / sr)
        end = min(duration_s, region_end / sr)
        if end - start >= 0.12:
            regions.append({"start": start, "end": end})
    return regions


def _trim_asr_segments_to_speech(
    segments: list[dict[str, Any]],
    waveform: Any,
    sr: int,
    speech_segments: list[dict[str, Any]] | None = None,
    pad_start_s: float = 0.35,
    pad_end_s: float = 0.45,
    min_shift_s: float = 0.50,
    min_keep_s: float = 0.50,
) -> list[dict[str, Any]]:
    """Tighten ASR segment boundaries to nearby acoustic activity.

    Long-form Whisper can emit a correct text segment whose start timestamp is
    anchored before leading silence. Click-to-play should use the acoustic
    speech boundary, not the coarse decoder boundary.
    """
    if not segments:
        return segments

    diarization_regions = [
        {
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", s.get("start", 0.0))),
        }
        for s in (speech_segments or [])
        if float(s.get("end", s.get("start", 0.0))) > float(s.get("start", 0.0))
    ]
    energy_regions = _energy_speech_regions(waveform, sr)
    if not diarization_regions and not energy_regions:
        return segments

    duration_s = len(waveform) / sr
    trimmed: list[dict[str, Any]] = []
    for seg in segments:
        start = max(0.0, float(seg.get("start", 0.0)))
        end = min(duration_s, float(seg.get("end", start)))
        if end <= start:
            continue

        overlaps = [
            r for r in diarization_regions
            if min(end, float(r["end"])) - max(start, float(r["start"])) > 0.05
        ]
        if not overlaps:
            overlaps = [
                r for r in energy_regions
                if min(end, float(r["end"])) - max(start, float(r["start"])) > 0.05
            ]
        if not overlaps:
            trimmed.append(seg)
            continue

        new_start = max(start, float(overlaps[0]["start"]) - pad_start_s)
        new_end = min(end, float(overlaps[-1]["end"]) + pad_end_s)
        if new_end <= new_start:
            trimmed.append(seg)
            continue

        # Avoid cutting real speech on coarse diarization boundaries. Very
        # short segments and large trims are especially risky, so keep the
        # decoder boundary unless the retained duration is clearly sufficient.
        if new_end - new_start < min_keep_s:
            trimmed.append(seg)
            continue

        item = dict(seg)
        if abs(new_start - start) >= min_shift_s:
            item["start_asr_raw"] = round(start, 3)
            item["start"] = round(new_start, 3)
        if abs(new_end - end) >= min_shift_s:
            item["end_asr_raw"] = round(end, 3)
            item["end"] = round(new_end, 3)
        trimmed.append(item)
    return trimmed


def _transcribe_speech_chunks(
    asr_backend: Any,
    waveform: Any,
    sr: int,
    speech_segments: list[dict[str, Any]],
    language: str | None,
    initial_prompt: str | None,
    pad_s: float = 2.0,
    max_chunk_s: float = 20.0,
    merge_gap_s: float = 1.5,
) -> dict[str, Any]:
    """Run low-latency ASR on diarized speech spans and restore timestamps.

    This is kept for realtime/transformers mode. Offline file transcription
    uses a sequential long-form backend as the ASR master clock.
    """
    if not speech_segments:
        return asr_backend.transcribe(
            waveform,
            language=language,
            initial_prompt=initial_prompt,
            sample_rate=sr,
        )

    duration_s = len(waveform) / sr
    all_segments: list[dict[str, Any]] = []
    texts: list[str] = []

    energy_regions = _energy_speech_regions(waveform, sr)
    speech_regions: list[dict[str, float]] = []
    for speech in sorted(speech_segments, key=lambda s: s.get("start", 0.0)):
        d_start = max(0.0, float(speech.get("start", 0.0)))
        d_end = min(duration_s, float(speech.get("end", d_start)))
        if d_end <= d_start:
            continue
        if energy_regions:
            active_s = 0.0
            for energy in energy_regions:
                active_s += max(0.0, min(d_end, energy["end"]) - max(d_start, energy["start"]))
            if active_s < 0.25 and active_s / max(d_end - d_start, 0.001) < 0.20:
                continue

        if speech_regions and d_start - speech_regions[-1]["end"] <= merge_gap_s:
            speech_regions[-1]["end"] = max(speech_regions[-1]["end"], d_end)
        else:
            speech_regions.append({"start": d_start, "end": d_end})

    for speech in speech_regions:
        valid_start = max(0.0, speech["start"])
        valid_end = min(duration_s, speech["end"])
        asr_start = max(0.0, valid_start - pad_s)
        asr_end = min(duration_s, valid_end + pad_s)
        if asr_end - asr_start < 0.25:
            continue

        chunk_start = asr_start
        while chunk_start < asr_end:
            chunk_end = min(asr_end, chunk_start + max_chunk_s)
            lo = int(chunk_start * sr)
            hi = int(chunk_end * sr)
            chunk = waveform[lo:hi]
            if len(chunk) < int(0.25 * sr):
                break

            chunk_result = asr_backend.transcribe(
                chunk,
                language=language,
                initial_prompt=initial_prompt,
                sample_rate=sr,
            )
            for seg in chunk_result.get("segments", []):
                text = seg.get("text", "").strip()
                if not text:
                    continue
                s = float(seg.get("start", 0.0)) + chunk_start
                e = float(seg.get("end", 0.0)) + chunk_start
                if e <= s:
                    e = min(chunk_end, s + 0.1)
                if e < valid_start or s > valid_end:
                    continue
                out_start = max(valid_start, s)
                out_end = min(valid_end, e)
                if out_end <= out_start:
                    continue
                all_segments.append({
                    "start": round(out_start, 3),
                    "end": round(out_end, 3),
                    "text": text,
                })
                texts.append(text)

            if chunk_end >= asr_end:
                break
            chunk_start = chunk_end

    all_segments.sort(key=lambda s: (s["start"], s["end"]))
    return {
        "text": "".join(texts),
        "language": language,
        "segments": all_segments,
    }


def _aggregate_words_to_turns(
    words: list[dict[str, Any]],
    max_gap: float = 0.5,
    max_dur: float = 15.0,
    max_chars: int = 80,
    max_word_dur: float = 2.5,
) -> list[dict[str, Any]]:
    """Aggregate word-level segments into speaker turns.

    Rules:
    - Same speaker + gap < *max_gap* → merge
    - Speaker change or gap > *max_gap* → split
    - Turn exceeds *max_dur* or *max_chars* → force-split
    """
    turns: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def _word_items(w: dict[str, Any]) -> list[dict[str, Any]]:
        if "start" not in w or "end" not in w:
            return []
        spk = w.get("speaker", "SPEAKER_UNKNOWN")
        w_start = float(w["start"])
        w_end = float(w["end"])
        w_text = str(w.get("word", "")).strip()
        if not w_text or w_end <= w_start:
            return []

        dur = w_end - w_start
        # Alignment occasionally pins a short Chinese token to a long
        # silence/noise span. Keep the token, but cap the span so a single
        # character cannot dominate the turn timeline.
        if len(w_text) <= 2 and dur > max_word_dur:
            return [{
                "start": w_start,
                "end": min(w_end, w_start + max_word_dur),
                "speaker": spk,
                "word": w_text,
            }]

        # WhisperX Chinese "words" can be multi-character chunks. If a
        # chunk spans too long, split it proportionally so turn limits stay
        # usable without constraining forced alignment beforehand.
        if len(w_text) > 2 and dur > max_word_dur:
            step = dur / len(w_text)
            return [
                {
                    "start": w_start + i * step,
                    "end": w_start + (i + 1) * step,
                    "speaker": spk,
                    "word": ch,
                }
                for i, ch in enumerate(w_text)
                if ch.strip()
            ]

        return [{"start": w_start, "end": w_end, "speaker": spk, "word": w_text}]

    for raw_word in words:
        for w in _word_items(raw_word):
            spk = w.get("speaker", "SPEAKER_UNKNOWN")
            w_start = float(w["start"])
            w_end = float(w["end"])
            w_text = str(w.get("word", ""))
            if not w_text.strip() or w_end <= w_start:
                continue

            need_new = (
                current is None
                or spk != current["speaker"]
                or w_start - current["end"] > max_gap
                or (w_end - current["start"]) > max_dur
                or len(current["text"]) + len(w_text) > max_chars
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
    return [
        t for t in turns
        if t.get("text", "").strip() and float(t["end"]) - float(t["start"]) >= 0.05
    ]


def _filter_unreliable_turns(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove turns that are physically implausible after ASR/alignment."""
    result: list[dict[str, Any]] = []
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        dur = max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))
        speaker = str(seg.get("speaker", ""))

        if "UNKNOWN" in speaker.upper():
            if dur < 0.5:
                continue
            if dur > 0 and len(text) / dur > 18:
                continue
        result.append(seg)
    return result


def _turns_need_segment_fallback(segments: list[dict[str, Any]]) -> bool:
    """Detect failed word alignment before exposing bad playback timestamps.

    WhisperX alignment sometimes collapses a long Chinese sentence into a very
    short span or stretches a single character over seconds. ASR segment
    timestamps are less fine-grained, but they are safer than unusable word
    timestamps in those cases.
    """
    checked = 0
    bad = 0
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        dur = max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))
        if dur <= 0:
            bad += 1
            continue
        checked += 1
        chars_per_s = len(text) / dur
        if len(text) >= 12 and chars_per_s > 12.0:
            bad += 1
            continue
        if len(text) <= 1 and dur > 1.2:
            bad += 1
            continue
        if len(text) <= 2 and dur > 2.0:
            bad += 1

    if checked == 0:
        return True
    return bad >= 2 or bad / checked >= 0.12


def _merge_adjacent_turns(
    segments: list[dict[str, Any]],
    max_gap: float = 1.6,
    max_dur: float = 15.0,
    max_chars: int = 80,
) -> list[dict[str, Any]]:
    """Merge nearby turns from the same speaker after text cleanup."""
    merged: list[dict[str, Any]] = []
    for seg in sorted(segments, key=lambda s: (s.get("start", 0.0), s.get("end", 0.0))):
        if not merged:
            merged.append(dict(seg))
            continue
        prev = merged[-1]
        same_speaker = seg.get("speaker") == prev.get("speaker")
        gap = float(seg.get("start", 0.0)) - float(prev.get("end", 0.0))
        combined_dur = float(seg.get("end", 0.0)) - float(prev.get("start", 0.0))
        combined_text = str(prev.get("text", "")) + str(seg.get("text", ""))
        if same_speaker and gap <= max_gap and combined_dur <= max_dur and len(combined_text) <= max_chars:
            prev_text = str(prev.get("text", "")).strip()
            prev_dur = float(prev.get("end", 0.0)) - float(prev.get("start", 0.0))
            # WhisperX may place a one-character lead word too early when ASR
            # used padded context. Keep the word in text, but do not let that
            # anomalous timestamp define the playback start of the merged turn.
            if len(prev_text) <= 1 and prev_dur > 1.0 and gap > 0.5:
                prev["start"] = round(max(float(prev.get("start", 0.0)), float(seg.get("start", 0.0)) - 0.35), 3)
            prev["end"] = seg.get("end", prev.get("end"))
            prev["text"] = combined_text
        else:
            merged.append(dict(seg))
    return merged


def _dedupe_turn_boundaries(
    segments: list[dict[str, Any]],
    min_gap: float = 0.04,
) -> list[dict[str, Any]]:
    """Prevent adjacent turns from overlapping in playback.

    ASR padding helps Whisper keep tail words, but forced alignment may assign
    neighboring words slightly overlapping timestamps. For clickable playback,
    the later turn must not start before the previous turn ends.
    """
    ordered = [dict(s) for s in sorted(segments, key=lambda x: (x.get("start", 0.0), x.get("end", 0.0)))]
    result: list[dict[str, Any]] = []
    for seg in ordered:
        if result:
            prev = result[-1]
            if float(seg.get("start", 0.0)) < float(prev.get("end", 0.0)):
                seg["start"] = round(float(prev["end"]) + min_gap, 3)
        if float(seg.get("end", 0.0)) <= float(seg.get("start", 0.0)):
            continue
        result.append(seg)
    return result


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

    mapped: dict[str, str] = {}
    label_i = 0
    for spk in sorted_spks:
        if "UNKNOWN" in spk.upper():
            mapped[spk] = "UNKNOWN"
            continue
        mapped[spk] = _SPEAKER_LABELS[label_i] if label_i < 26 else f"S{label_i-25}"
        label_i += 1
    return mapped


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

    waveform, sr = _resample_to_16k(waveform, sr)

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
    offline_backend = asr_cfg.pop("offline_backend", None)
    if offline_backend:
        asr_cfg["backend"] = offline_backend

    asr_backend = load_asr_backend(asr_cfg)
    asr_backend.load()

    lang_cfg = lcfg.get(language, {})
    initial_prompt = lang_cfg.get("asr_initial_prompt")

    if asr_cfg.get("backend") == "openai-whisper":
        asr_result = asr_backend.transcribe(
            waveform,
            language=language,
            initial_prompt=initial_prompt,
            sample_rate=sr,
        )
    else:
        asr_result = _transcribe_speech_chunks(
            asr_backend,
            waveform,
            sr,
            dia_result["segments"],
            language,
            initial_prompt,
        )

    asr_result["segments"] = _trim_asr_segments_to_speech(
        asr_result.get("segments", []), waveform, sr, dia_result["segments"],
    )

    asr_backend.unload()
    timing["asr"] = round(time.time() - t3, 2)
    logger.info("  → %d segments (%.1fs)", len(asr_result["segments"]), timing["asr"])

    # ---- 4. Word-level speaker assignment + turn aggregation -------------
    logger.info("Stage 4: word-level speaker assignment …")
    t4 = time.time()

    import pandas as pd

    # 4a. Pre-filter ASR hallucination artifacts.
    # Do not pre-split before forced alignment: fake time windows force text
    # into the wrong audio span and cause click-to-play mismatch.
    from src.llm.corrector import clean_hallucination, zh_simplify
    asr_segs = _filter_asr_artifacts(asr_result["segments"])

    # 4b. Forced alignment → word timestamps (wav2vec2, via whisperx)
    import whisperx
    segments_for_align = [
        {"start": s["start"], "end": s["end"], "text": s["text"]}
        for s in asr_segs
    ]

    align_model_name_raw = mcfg["alignment"].get(language or "zh", "")
    # Resolve relative local paths to absolute
    from src.utils.config import project_root as _proot
    align_model_name: str | Path = align_model_name_raw
    if align_model_name_raw.startswith("./") or align_model_name_raw.startswith(".\\"):
        align_model_name = str(_proot() / align_model_name_raw)
    align_model, align_metadata = whisperx.load_align_model(
        language_code=(language or "zh"),
        device=device,
        model_name=align_model_name,
    )
    speaker_words: list[dict[str, Any]] = []
    used_segment_fallback = False
    try:
        aligned = whisperx.align(
            segments_for_align, align_model, align_metadata,
            waveform, device=device, return_char_alignments=False,
        )

        # 4c. Diarization → pandas DataFrame; per-word speaker assignment
        dia_segs = dia_result["segments"]
        diarize_df = pd.DataFrame([
            {"start": s["start"], "end": s["end"], "speaker": s["speaker"]}
            for s in dia_segs
        ])
        result_with_spk = whisperx.assign_word_speakers(diarize_df, aligned)
        speaker_words = result_with_spk.get("word_segments", [])

        # 4d. Aggregate words → speaker turns
        merged = _aggregate_words_to_turns(speaker_words)
        if _turns_need_segment_fallback(merged):
            logger.warning("  Word-level alignment looks unreliable; falling back to ASR segment timestamps")
            from src.alignment import align_segments
            merged = align_segments(
                dia_result["segments"], _presplit_segments(asr_segs), min_speaker_ratio=0.15,
            )
            used_segment_fallback = True
    except Exception as e:
        logger.warning("  Word-level alignment failed; falling back to segment alignment: %s", e)
        from src.alignment import align_segments
        merged = align_segments(
            dia_result["segments"], _presplit_segments(asr_segs), min_speaker_ratio=0.15,
        )
        used_segment_fallback = True
    finally:
        del align_model

    if not merged:
        logger.warning("  No speaker turns; falling back to segment alignment")
        from src.alignment import align_segments
        merged = align_segments(
            dia_result["segments"], _presplit_segments(asr_segs), min_speaker_ratio=0.15,
        )
        used_segment_fallback = True

    # 4e. zh_simplify + speaker map (hallucination already filtered in 4a)
    merged = _filter_unreliable_turns(merged)
    for seg in merged:
        txt = clean_hallucination(seg.get("text", ""))
        seg["text"] = zh_simplify(txt)
    if not used_segment_fallback:
        merged = _merge_adjacent_turns(merged)
    merged = _dedupe_turn_boundaries(merged)

    for seg in merged:
        old_spk = seg.get("speaker", "")
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
