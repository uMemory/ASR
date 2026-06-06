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


def _warn_segment_fallback(reason: str) -> None:
    """Emit an obvious console message when word timestamps are not used."""
    msg = f"[ASR] Word-level alignment fallback -> segment timestamps ({reason})"
    print(msg, flush=True)
    logger.warning("  %s", msg)


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


def _filter_asr_by_speech_activity(
    segments: list[dict[str, Any]],
    waveform: Any,
    sr: int,
    pad_s: float = 0.35,
    min_overlap_s: float = 0.16,
    min_overlap_ratio: float = 0.08,
) -> list[dict[str, Any]]:
    """Drop ASR text that has no acoustic speech support.

    Whisper can hallucinate plausible text over near-field silence or
    background bleed. A neural VAD gate keeps segments with enough real speech
    support and removes long text spans validated only by a short noise click.
    """
    if not segments:
        return segments

    speech_regions = _speech_activity_regions(waveform, sr)
    if not speech_regions:
        return segments

    kept: list[dict[str, Any]] = []
    dropped = 0
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        dur = max(0.0, end - start)
        if dur <= 0:
            dropped += 1
            continue

        overlap = 0.0
        for region in speech_regions:
            r_start = max(0.0, float(region["start"]) - pad_s)
            r_end = float(region["end"]) + pad_s
            overlap += max(0.0, min(end, r_end) - max(start, r_start))

        text_len = len(str(seg.get("text", "")).strip())
        required_overlap = min_overlap_s
        if dur >= 10.0 or text_len >= 28:
            required_overlap = max(0.90, dur * 0.12)
        elif dur >= 4.0 or text_len >= 14:
            required_overlap = max(0.35, dur * min_overlap_ratio)

        if overlap >= required_overlap:
            seg["speech_overlap_s"] = round(overlap, 3)
            seg["speech_overlap_ratio"] = round(overlap / dur, 3)
            kept.append(seg)
        else:
            dropped += 1
            logger.debug(
                "Drop ASR hallucination without enough speech support: %.2f-%.2f "
                "overlap=%.2f required=%.2f %s",
                start, end, overlap, required_overlap, str(seg.get("text", "")).strip()[:60],
            )

    if dropped:
        logger.info("  ASR speech-activity gate dropped %d hallucinated segment(s)", dropped)
    return kept


def _has_suspicious_repeated_phrase(text: str) -> bool:
    """Detect adjacent repeated non-filler phrases in short ASR text."""
    import re as _re

    normalized = _re.sub(r"[，,。！？!?、；;：:\s]+", " ", text.strip())
    allowed = {"对", "嗯", "啊", "好", "是", "对对", "嗯嗯", "哈哈"}
    for match in _re.finditer(r"(?<!\S)([\u4e00-\u9fffA-Za-z0-9]{2,6})(?:\s+\1){1,}(?!\S)", normalized):
        phrase = match.group(1)
        if phrase not in allowed:
            return True
    compact = _re.sub(r"[，,。！？!?、；;：:\s]+", "", text.strip())
    for n in range(2, 7):
        for i in range(0, max(0, len(compact) - 2 * n + 1)):
            phrase = compact[i:i + n]
            if phrase in allowed:
                continue
            if compact[i + n:i + 2 * n] == phrase:
                return True
    return False


def _low_confidence_reasons(seg: dict[str, Any]) -> list[str]:
    """Return conservative low-confidence reasons without mutating text."""
    reasons: list[str] = []
    text = str(seg.get("text", "")).strip()
    dur = max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))
    avg_logprob = seg.get("avg_logprob")
    no_speech_prob = seg.get("no_speech_prob")
    speech_overlap = seg.get("speech_overlap_s")
    speech_ratio = seg.get("speech_overlap_ratio")

    try:
        avg_logprob_f = float(avg_logprob) if avg_logprob is not None else None
    except (TypeError, ValueError):
        avg_logprob_f = None
    try:
        no_speech_prob_f = float(no_speech_prob) if no_speech_prob is not None else None
    except (TypeError, ValueError):
        no_speech_prob_f = None
    try:
        speech_ratio_f = float(speech_ratio) if speech_ratio is not None else None
    except (TypeError, ValueError):
        speech_ratio_f = None
    try:
        speech_overlap_f = float(speech_overlap) if speech_overlap is not None else None
    except (TypeError, ValueError):
        speech_overlap_f = None

    suspicious_repeat = _has_suspicious_repeated_phrase(text)
    low_avg = avg_logprob_f is not None and avg_logprob_f <= -0.65
    low_overlap = speech_ratio_f is not None and dur >= 4.0 and speech_ratio_f < 0.22
    low_absolute_speech = (
        speech_overlap_f is not None
        and dur >= 4.0
        and speech_overlap_f < min(1.2, dur * 0.20)
    )
    high_no_speech = no_speech_prob_f is not None and no_speech_prob_f >= 0.65

    if low_avg:
        reasons.append(f"low avg_logprob={avg_logprob_f:.2f}")
    if low_overlap:
        reasons.append(f"low speech overlap={speech_ratio_f:.0%}")
    if low_absolute_speech:
        reasons.append(f"short speech support={speech_overlap_f:.1f}s/{dur:.1f}s")
    if suspicious_repeat:
        reasons.append("suspicious repeated phrase")
    if high_no_speech and (low_avg or low_overlap or low_absolute_speech or suspicious_repeat):
        reasons.append(f"high no_speech_prob={no_speech_prob_f:.2f}")
    if (
        dur > 0
        and len(text) / dur > 12
        and no_speech_prob_f is not None
        and no_speech_prob_f >= 0.50
        and (low_overlap or low_absolute_speech or suspicious_repeat)
    ):
        reasons.append("dense text with high no_speech_prob")

    return reasons


def _annotate_low_confidence_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for seg in segments:
        item = dict(seg)
        reasons = _low_confidence_reasons(item)
        if reasons:
            item["low_confidence"] = True
            item["low_confidence_reasons"] = reasons
        annotated.append(item)
    return annotated


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


def _turn_limits_for_language(language: str | None) -> dict[str, float | int]:
    """Return turn-size limits tuned for each language."""
    if language == "en":
        return {
            "max_gap": 0.8,
            "max_dur": 36.0,
            "max_chars": 420,
            "merge_gap": 1.2,
            "merge_max_dur": 40.0,
            "merge_max_chars": 560,
        }
    return {
        "max_gap": 0.5,
        "max_dur": 15.0,
        "max_chars": 80,
        "merge_gap": 1.6,
        "merge_max_dur": 15.0,
        "merge_max_chars": 70,
    }


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


def _light_denoise_waveform(
    waveform: Any,
    sr: int,
    reduction: float = 1.6,
    gate_floor: float = 0.18,
) -> Any:
    """Conservative spectral denoise + soft noise gate.

    Near-field meeting channels often contain low-level bleed from other
    speakers. Whisper can hallucinate that bleed as foreground speech, causing
    early text/audio mismatches. This preprocessing is deliberately light:
    spectral subtraction reduces stationary noise, and the gate attenuates
    low-energy frames instead of hard-zeroing them.
    """
    import numpy as np

    x = np.asarray(waveform, dtype=np.float32)
    if x.size < int(0.5 * sr):
        return x

    try:
        import librosa

        n_fft = 1024
        hop = 256
        stft = librosa.stft(x, n_fft=n_fft, hop_length=hop, center=True)
        mag = np.abs(stft)
        phase = np.exp(1j * np.angle(stft))
        noise = np.percentile(mag, 20, axis=1, keepdims=True)
        clean = np.maximum(mag - reduction * noise, 0.0)
        mask = clean / (mag + 1e-8)
        denoised = librosa.istft(mag * mask * phase, hop_length=hop, length=len(x))
    except Exception:
        denoised = x

    frame_len = max(1, int(0.08 * sr))
    hop_len = max(1, int(0.04 * sr))
    if len(denoised) < frame_len:
        return denoised.astype(np.float32)

    centers: list[int] = []
    rms_vals: list[float] = []
    for lo in range(0, len(denoised) - frame_len + 1, hop_len):
        frame = denoised[lo:lo + frame_len]
        centers.append(lo + frame_len // 2)
        rms_vals.append(float(np.sqrt(np.mean(frame * frame))))
    if not rms_vals:
        return denoised.astype(np.float32)

    rms = np.asarray(rms_vals, dtype=np.float32)
    threshold = max(
        float(np.percentile(rms, 25)) * 2.5,
        float(np.percentile(rms, 85)) * 0.10,
        0.0008,
    )
    gains = np.clip(rms / max(threshold, 1e-8), gate_floor, 1.0)
    sample_idx = np.arange(len(denoised), dtype=np.float32)
    gain_curve = np.interp(
        sample_idx,
        np.asarray([0, *centers, len(denoised) - 1], dtype=np.float32),
        np.asarray([gains[0], *gains.tolist(), gains[-1]], dtype=np.float32),
    )
    return np.clip(denoised * gain_curve, -1.0, 1.0).astype(np.float32)


def _energy_speech_regions(waveform: Any, sr: int) -> list[dict[str, float]]:
    """Detect speech-like regions from a lightly noise-gated waveform.

    This is intentionally only used as an activity gate for timestamp
    refinement/chunk filtering. The ASR model still receives the original
    waveform, so a conservative gate cannot damage recognition quality.
    """
    import numpy as np

    duration_s = len(waveform) / sr
    frame_len = max(1, int(0.08 * sr))
    hop_len = max(1, int(0.04 * sr))
    if len(waveform) < frame_len:
        return []

    # Light spectral subtraction for activity detection.  Stationary room
    # noise often fools raw RMS thresholds and pyannote boundaries, especially
    # near the beginning of a file.  Percentile noise floors avoid assuming
    # that the first seconds are silence.
    gated = np.asarray(waveform, dtype=np.float32)
    try:
        import librosa

        n_fft = min(1024, max(256, 2 ** int(np.floor(np.log2(max(frame_len, 2))))))
        hop = max(1, n_fft // 4)
        stft = librosa.stft(gated, n_fft=n_fft, hop_length=hop, center=False)
        mag = np.abs(stft)
        phase = np.exp(1j * np.angle(stft))
        floor = np.percentile(mag, 20, axis=1, keepdims=True)
        cleaned = np.maximum(mag - 1.4 * floor, 0.0)
        gated = librosa.istft(cleaned * phase, hop_length=hop, length=len(gated))
        gated = gated.astype(np.float32, copy=False)
    except Exception:
        pass

    rms_vals: list[float] = []
    starts: list[int] = []
    for lo in range(0, len(gated) - frame_len + 1, hop_len):
        frame = gated[lo:lo + frame_len]
        rms_vals.append(float(np.sqrt(np.mean(frame * frame))))
        starts.append(lo)

    if not rms_vals:
        return []

    rms = np.asarray(rms_vals, dtype=np.float32)
    max_rms = float(np.max(rms))
    if max_rms <= 1e-6:
        return []

    floor = float(np.percentile(rms, 25))
    speech_ref = float(np.percentile(rms, 88))
    threshold = max(
        max_rms * (10 ** (-26 / 20)),
        floor * 3.0,
        speech_ref * 0.18,
        0.0012,
    )
    active = rms >= threshold

    raw_regions: list[dict[str, float]] = []
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
            if end - start >= 0.16:
                raw_regions.append({"start": start, "end": end})
            region_start = None
    if region_start is not None:
        start = max(0.0, region_start / sr)
        end = min(duration_s, region_end / sr)
        if end - start >= 0.16:
            raw_regions.append({"start": start, "end": end})

    regions: list[dict[str, float]] = []
    for region in raw_regions:
        if regions and region["start"] - regions[-1]["end"] <= 0.24:
            regions[-1]["end"] = region["end"]
        else:
            regions.append(region)
    return regions


_SILERO_VAD_MODEL: Any | None = None


def _silero_speech_regions(
    waveform: Any,
    sr: int,
    threshold: float = 0.35,
    min_speech_duration_ms: int = 120,
    min_silence_duration_ms: int = 180,
    speech_pad_ms: int = 80,
) -> list[dict[str, float]]:
    """Detect speech timestamps with Silero VAD.

    This follows the common production pattern: use a neural VAD to find
    acoustic speech boundaries, then keep a small pad so consonant onsets and
    tail phones are not clipped.
    """
    global _SILERO_VAD_MODEL
    try:
        import numpy as np
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad

        if sr != 16000:
            return []
        if _SILERO_VAD_MODEL is None:
            _SILERO_VAD_MODEL = load_silero_vad()
        wav = torch.from_numpy(np.asarray(waveform, dtype=np.float32))
        stamps = get_speech_timestamps(
            wav,
            _SILERO_VAD_MODEL,
            sampling_rate=sr,
            threshold=threshold,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            return_seconds=True,
        )
        return [
            {"start": float(s["start"]), "end": float(s["end"])}
            for s in stamps
            if float(s["end"]) > float(s["start"])
        ]
    except Exception as exc:
        logger.debug("Silero VAD unavailable, falling back to energy VAD: %s", exc)
        return []


def _speech_activity_regions(waveform: Any, sr: int) -> list[dict[str, float]]:
    """Return preferred speech activity regions for endpointing."""
    regions = _silero_speech_regions(waveform, sr)
    return regions if regions else _energy_speech_regions(waveform, sr)


def _strip_unsupported_filler_prefix(text: str) -> str:
    """Remove leading filler repeats when VAD proves the prefix is unsupported."""
    import re as _re

    value = text.strip()
    if not value:
        return value

    filler = r"(?:对|嗯|啊|呃|额|好|是)"
    spaced = _re.match(rf"^(?:(?:{filler})[\s,，、]+){{2,}}(.+)$", value)
    if spaced:
        return spaced.group(1).strip()

    compact = _re.match(r"^([对嗯啊呃额好是]{2,})([\s,，、]+)(.+)$", value)
    if compact:
        return compact.group(3).strip()

    return value


def _refine_segment_playback_boundaries(
    segments: list[dict[str, Any]],
    waveform: Any,
    sr: int,
    speech_segments: list[dict[str, Any]] | None = None,
    pad_start_s: float = 0.35,
    playback_tail_pad_s: float = 0.65,
    min_shift_s: float = 0.50,
    min_keep_s: float = 0.50,
    max_long_text_shift_s: float = 1.5,
    playback_rewind_s: float = 0.35,
    first_segment_rewind_s: float = 1.20,
) -> list[dict[str, Any]]:
    """Refine final turn boundaries for click-to-play.

    We only move the displayed start forward when there is clear leading
    non-speech. The displayed end is never shortened here because that risks
    cutting tail phones; instead ``playback_end`` gets a small hidden pad used
    by the GUI player.
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
    energy_regions = _speech_activity_regions(waveform, sr)
    if not diarization_regions and not energy_regions:
        return segments

    duration_s = len(waveform) / sr
    trimmed: list[dict[str, Any]] = []
    for idx, seg in enumerate(segments):
        start = max(0.0, float(seg.get("start", 0.0)))
        end = min(duration_s, float(seg.get("end", start)))
        if end <= start:
            continue

        overlaps = [
            r for r in energy_regions
            if min(end, float(r["end"])) - max(start, float(r["start"])) > 0.05
        ]
        if not overlaps:
            overlaps = [
                r for r in diarization_regions
                if min(end, float(r["end"])) - max(start, float(r["start"])) > 0.05
            ]
        if not overlaps:
            trimmed.append(seg)
            continue

        item = dict(seg)
        overlaps.sort(key=lambda r: float(r["start"]))
        text_len = len(str(seg.get("text", "")).strip())
        if text_len <= 8:
            selected = max(
                overlaps,
                key=lambda r: min(end, float(r["end"])) - max(start, float(r["start"])),
            )
        else:
            selected = overlaps[0]

        # At the very beginning of some near-field channels, a background
        # burst appears before the real first utterance. If that first active
        # island is followed by a clear gap, skip it. For later segments we
        # keep the earliest activity to avoid cutting low-energy syllables.
        if start <= 0.5 and len(overlaps) >= 2:
            first_end = float(overlaps[0]["end"])
            for candidate in overlaps[1:]:
                if float(candidate["start"]) - first_end >= 1.2:
                    selected = candidate
                    break

        new_start = max(start, float(selected["start"]) - pad_start_s)
        # Avoid cutting real speech on coarse diarization boundaries. Very
        # short segments are especially risky, so keep the decoder boundary
        # unless the retained duration is clearly sufficient.
        shift = new_start - start
        endpoint_start = start
        max_shift = 20.0 if text_len <= 8 or start <= 0.5 else max_long_text_shift_s
        if shift >= min_shift_s and end - new_start >= min_keep_s:
            endpoint_start = new_start
            if shift <= max_shift:
                item["start_asr_raw"] = round(start, 3)
                item["start"] = round(new_start, 3)
            else:
                stripped_text = _strip_unsupported_filler_prefix(str(item.get("text", "")))
                if stripped_text and stripped_text != str(item.get("text", "")).strip():
                    item["text"] = stripped_text
                    item["start_asr_raw"] = round(start, 3)
                    item["start"] = round(new_start, 3)

        # Playback may use a tighter endpoint than the displayed timestamp.
        # This removes audible leading silence without rewriting the transcript
        # clock when the required correction is large.
        # Playback must be more conservative than display timestamps.  If VAD
        # or diarization trims the first active syllable too aggressively, a
        # clickable segment can miss the beginning of its text.  Keep a hidden
        # rewind bounded by the original ASR start; use a larger rewind for the
        # first transcript line where endpointing is most error-prone.
        display_start = float(item.get("start", start))
        rewind = first_segment_rewind_s if idx == 0 else playback_rewind_s
        playback_start = min(display_start, max(start, endpoint_start - rewind))
        playback_end = min(duration_s, end + playback_tail_pad_s)
        if playback_end > end:
            item["playback_start"] = round(max(0.0, playback_start - 0.05), 3)
            item["playback_end"] = round(playback_end, 3)
        trimmed.append(item)
    return trimmed


def _adjust_neighbor_playback_boundaries(
    segments: list[dict[str, Any]],
    continuous_gap_s: float = 0.30,
    adjacent_lead_pad_s: float = 0.06,
    boundary_gap_s: float = 0.02,
    max_adjacent_rewind_s: float = 0.50,
) -> list[dict[str, Any]]:
    """Keep hidden playback windows from crossing adjacent transcript turns.

    Isolated turns keep the normal tail pad so final phones are not cut.  For
    back-to-back turns, the previous turn must stop before the next text starts,
    while the next turn gets a slightly larger lead pad to avoid losing its
    first syllable.
    """
    if not segments:
        return segments

    ordered = [dict(s) for s in sorted(segments, key=lambda x: (x.get("start", 0.0), x.get("end", 0.0)))]
    continuous_after = [False] * len(ordered)
    continuous_before = [False] * len(ordered)
    for i in range(len(ordered) - 1):
        end = float(ordered[i].get("end", ordered[i].get("start", 0.0)))
        next_start = float(ordered[i + 1].get("start", end))
        if next_start - end <= continuous_gap_s:
            continuous_after[i] = True
            continuous_before[i + 1] = True

    for i, seg in enumerate(ordered):
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))
        play_start = float(seg.get("playback_start", max(0.0, start - 0.05)))
        play_end = float(seg.get("playback_end", end))

        if continuous_before[i] and play_start - start <= max_adjacent_rewind_s:
            play_start = min(play_start, max(0.0, start - adjacent_lead_pad_s))

        if continuous_after[i]:
            # In continuous speech, any tail padding is likely to play the next
            # transcript line. Cap at the displayed boundary, then a second
            # pass below removes overlap with the next line's lead pad.
            play_end = min(play_end, end)

        if play_end > play_start:
            seg["playback_start"] = round(play_start, 3)
            seg["playback_end"] = round(play_end, 3)

    for i in range(len(ordered) - 1):
        if not continuous_after[i]:
            continue
        next_play_start = float(ordered[i + 1].get("playback_start", ordered[i + 1].get("start", 0.0)))
        prev_start = float(ordered[i].get("playback_start", ordered[i].get("start", 0.0)))
        capped = next_play_start - boundary_gap_s
        if capped > prev_start:
            ordered[i]["playback_end"] = round(min(float(ordered[i].get("playback_end", capped)), capped), 3)
    return ordered


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


def _concat_turn_text(left: str, right: str) -> str:
    """Concatenate turn text, adding a space for adjacent ASCII words."""
    if not left:
        return right
    if not right:
        return left
    if left[-1].isspace() or right[0].isspace():
        return left + right
    if left[-1] in ".?!,;:" and right[0].isascii() and right[0].isalnum():
        return left + " " + right
    if left[-1].isascii() and right[0].isascii() and left[-1].isalnum() and right[0].isalnum():
        return left + " " + right
    return left + right


def _english_word_tokens(text: str) -> list[str]:
    import re as _re

    return [
        t.lower()
        for t in _re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+", text)
    ]


def _subsequence_index(tokens: list[str], needle: list[str], start: int = 0) -> int:
    if not needle or len(needle) > len(tokens):
        return -1
    last = len(tokens) - len(needle)
    for idx in range(max(0, start), last + 1):
        if tokens[idx:idx + len(needle)] == needle:
            return idx
    return -1


def _format_english_prefix(tokens: list[str]) -> str:
    text = " ".join(tokens).strip()
    if not text:
        return text
    text = text[0].upper() + text[1:]
    if text[-1] not in ".?!,;:":
        text += "."
    return text


def _repair_english_boundary_prefixes(
    asr_backend: Any,
    waveform: Any,
    sr: int,
    segments: list[dict[str, Any]],
    language: str | None,
    max_repairs: int = 8,
) -> list[dict[str, Any]]:
    """Recover short English phrases lost at ASR chunk/segment boundaries.

    Whisper occasionally drops a brief phrase at a boundary in continuous
    English speech. A local re-decode around the boundary often contains
    ``previous tail + missing prefix + next head``. When that pattern is
    unambiguous, prepend the missing phrase to the following segment.
    """
    if language != "en" or len(segments) < 2:
        return segments

    repaired = [dict(s) for s in segments]
    repairs = 0
    duration_s = len(waveform) / sr
    for idx in range(len(repaired) - 1):
        if repairs >= max_repairs:
            break

        prev = repaired[idx]
        nxt = repaired[idx + 1]
        prev_end = float(prev.get("end", 0.0))
        next_start = float(nxt.get("start", prev_end))
        if abs(next_start - prev_end) > 0.35:
            continue

        prev_tokens = _english_word_tokens(str(prev.get("text", "")))
        next_tokens = _english_word_tokens(str(nxt.get("text", "")))
        if len(prev_tokens) < 4 or len(next_tokens) < 4:
            continue

        win_start = max(0.0, prev_end - 2.5)
        win_end = min(duration_s, next_start + 8.0)
        if win_end - win_start < 4.0:
            continue

        chunk = waveform[int(win_start * sr): int(win_end * sr)]
        if len(chunk) < int(1.0 * sr):
            continue

        try:
            probe = asr_backend.transcribe(chunk, language="en", sample_rate=sr)
        except Exception as exc:
            logger.debug("English boundary probe failed at %.2fs: %s", prev_end, exc)
            continue

        probe_tokens = _english_word_tokens(str(probe.get("text", "")))
        if len(probe_tokens) < 6:
            continue

        found: list[str] = []
        for tail_len in range(min(8, len(prev_tokens)), 2, -1):
            tail = prev_tokens[-tail_len:]
            tail_idx = _subsequence_index(probe_tokens, tail)
            if tail_idx < 0:
                continue
            after_tail = tail_idx + tail_len
            for head_len in range(min(7, len(next_tokens)), 2, -1):
                head = next_tokens[:head_len]
                head_idx = _subsequence_index(probe_tokens, head, after_tail)
                if head_idx <= after_tail:
                    continue
                candidate = probe_tokens[after_tail:head_idx]
                if 2 <= len(candidate) <= 10:
                    found = candidate
                    break
            if found:
                break

        if not found:
            continue

        existing_tokens = prev_tokens + next_tokens
        if _subsequence_index(existing_tokens, found) >= 0:
            continue
        prefix = _format_english_prefix(found)
        if not prefix:
            continue
        nxt["text"] = f"{prefix} {str(nxt.get('text', '')).lstrip()}"
        nxt["boundary_prefix_repaired"] = True
        nxt["boundary_prefix_source"] = {
            "window_start": round(win_start, 3),
            "window_end": round(win_end, 3),
            "text": prefix,
        }
        repairs += 1

    if repairs:
        logger.info("  English boundary repair recovered %d prefix phrase(s)", repairs)
    return repaired


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
                current["text"] = _concat_turn_text(str(current["text"]), w_text)

    if current:
        turns.append(current)
    return [
        t for t in turns
        if t.get("text", "").strip() and float(t["end"]) - float(t["start"]) >= 0.05
    ]


def _asr_segments_with_word_timing(
    asr_segments: list[dict[str, Any]],
    words: list[dict[str, Any]],
    diarization_segments: list[dict[str, Any]] | None = None,
    min_coverage: float = 0.35,
) -> list[dict[str, Any]]:
    """Use word timestamps to time ASR segments while preserving ASR text.

    WhisperX's Chinese "word" chunks are often character-like and can produce
    awkward turn text when directly concatenated.  The ASR segment text is
    usually more coherent, so use aligned words as timing/speaker evidence and
    keep the original ASR sentence.
    """
    timed: list[dict[str, Any]] = []
    valid_words = [
        w for w in words
        if "start" in w and "end" in w and float(w["end"]) > float(w["start"])
    ]
    if not valid_words:
        return []

    ordered_asr = sorted(asr_segments, key=lambda s: (s.get("start", 0.0), s.get("end", 0.0)))
    for idx, seg in enumerate(ordered_asr):
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        s0 = float(seg.get("start", 0.0))
        e0 = float(seg.get("end", s0))
        if e0 <= s0:
            continue
        prev_end = float(ordered_asr[idx - 1].get("end", s0)) if idx > 0 else None
        next_start = float(ordered_asr[idx + 1].get("start", e0)) if idx + 1 < len(ordered_asr) else None
        has_close_prev = prev_end is not None and s0 - prev_end <= 0.5
        has_close_next = next_start is not None and next_start - e0 <= 0.5

        local = []
        for w in valid_words:
            ws = float(w["start"])
            we = float(w["end"])
            mid = (ws + we) / 2
            if has_close_prev and we <= s0 + 0.05:
                continue
            if has_close_next and ws >= e0 - 0.02:
                continue
            if s0 - 0.3 <= mid <= e0 + 0.3:
                local.append(w)
        if not local:
            continue

        aligned_chars = sum(len(str(w.get("word", "")).strip()) for w in local)
        if len(text) >= 8 and aligned_chars / max(len(text), 1) < min_coverage:
            continue

        speaker_dur: dict[str, float] = {}
        for w in local:
            spk = str(w.get("speaker", "SPEAKER_UNKNOWN"))
            dur = max(0.0, float(w["end"]) - float(w["start"]))
            speaker_dur[spk] = speaker_dur.get(spk, 0.0) + dur
        non_unknown = {
            spk: dur for spk, dur in speaker_dur.items()
            if "UNKNOWN" not in spk.upper()
        }
        if non_unknown:
            speaker = max(non_unknown, key=non_unknown.get)
        elif diarization_segments:
            seg_start = float(local[0]["start"])
            seg_end = float(local[-1]["end"])
            overlap_by_speaker: dict[str, float] = {}
            for dia in diarization_segments:
                d_start = float(dia.get("start", 0.0))
                d_end = float(dia.get("end", d_start))
                overlap = max(0.0, min(seg_end, d_end) - max(seg_start, d_start))
                if overlap > 0:
                    spk = str(dia.get("speaker", "SPEAKER_UNKNOWN"))
                    overlap_by_speaker[spk] = overlap_by_speaker.get(spk, 0.0) + overlap
            speaker = max(overlap_by_speaker, key=overlap_by_speaker.get) if overlap_by_speaker else "SPEAKER_UNKNOWN"
        else:
            speaker = max(speaker_dur, key=speaker_dur.get) if speaker_dur else "SPEAKER_UNKNOWN"

        timed.append({
            "start": round(float(local[0]["start"]), 3),
            "end": round(float(local[-1]["end"]), 3),
            "speaker": speaker,
            "text": text,
            "avg_logprob": seg.get("avg_logprob"),
            "no_speech_prob": seg.get("no_speech_prob"),
            "speech_overlap_s": seg.get("speech_overlap_s"),
            "speech_overlap_ratio": seg.get("speech_overlap_ratio"),
        })
    return timed


def _filter_unreliable_turns(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove turns that are physically implausible after ASR/alignment."""
    result: list[dict[str, Any]] = []
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        dur = max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))
        speaker = str(seg.get("speaker", ""))

        if dur > 8.0 and len(text) <= 4:
            continue
        if dur > 4.0 and len(text) <= 8 and any(not ch.isalnum() and not ch.isspace() for ch in text):
            continue

        if "UNKNOWN" in speaker.upper():
            import re as _re

            if len(text) <= 2:
                continue
            if dur < 1.0 and len(text) <= 8:
                continue
            # Unknown-speaker turns are often alignment bleed. Keep them only
            # when the content has enough duration and textual mass to be a
            # plausible independent utterance.
            if dur < 2.0 and len(text) <= 14:
                continue
            if len(text) < 6 and dur > 2.0:
                continue
            repeated_non_filler = _re.search(r"([^对嗯啊哈\s])\1", text)
            repeated_phrase = any(
                len(phrase.strip()) >= 4 and text.count(phrase) >= 2
                for n in range(4, min(9, len(text) // 2 + 1))
                for phrase in (text[i:i + n] for i in range(0, len(text) - n + 1))
            )
            if repeated_non_filler or repeated_phrase:
                continue
            if dur < 0.5:
                continue
            if dur > 0 and len(text) / dur > 18:
                continue
            # Low-level bleed/noise may survive denoising and decode as a few
            # characters stretched over many seconds. If diarization cannot
            # assign a real speaker, treat this as background artifact.
            if dur > 4.0 and len(text) <= 8:
                continue
        result.append(seg)
    return result


def _turns_need_segment_fallback(
    segments: list[dict[str, Any]],
    asr_segments: list[dict[str, Any]] | None = None,
) -> tuple[bool, str]:
    """Detect failed word alignment before exposing bad playback timestamps.

    WhisperX alignment sometimes collapses a long Chinese sentence into a very
    short span or stretches a single character over seconds. ASR segment
    timestamps are less fine-grained, but they should only replace word-level
    timestamps when the overall alignment is bad. Local Chinese token outliers
    are repaired downstream and should not trigger full fallback.
    """
    checked = 0
    severe_bad = 0
    total_chars = 0
    total_dur = 0.0
    aligned_chars = 0
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        dur = max(0.0, float(seg.get("end", 0.0)) - float(seg.get("start", 0.0)))
        if dur <= 0:
            severe_bad += 1
            continue
        checked += 1
        total_chars += len(text)
        total_dur += dur
        aligned_chars += len(text)
        chars_per_s = len(text) / dur
        if len(text) >= 12 and chars_per_s > 18.0:
            severe_bad += 1
            continue
        if len(text) <= 1 and dur > 3.0:
            severe_bad += 1
            continue
        if len(text) <= 2 and dur > 4.0:
            severe_bad += 1

    if checked == 0:
        return True, "no valid word-level turns"

    if asr_segments:
        source_chars = sum(len(str(s.get("text", "")).strip()) for s in asr_segments)
        if source_chars >= 20:
            coverage = aligned_chars / max(source_chars, 1)
            if coverage < 0.45:
                return True, f"low aligned text coverage {coverage:.0%}"

    if severe_bad >= 5 and severe_bad / checked >= 0.30:
        return True, f"many implausible word turns {severe_bad}/{checked}"

    if total_chars >= 80 and total_dur > 0 and total_chars / total_dur > 22.0:
        return True, "global word timing is too compressed"

    return False, "word-level alignment accepted"


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
        combined_text = _concat_turn_text(str(prev.get("text", "")), str(seg.get("text", "")))
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


def _reattach_drifted_prefix_fragments(
    segments: list[dict[str, Any]],
    max_gap: float = 8.0,
    max_prefix_chars: int = 2,
    lead_pad_s: float = 0.35,
) -> list[dict[str, Any]]:
    """Attach tiny prefix fragments whose word timestamp drifted too early.

    Forced alignment can occasionally pin the first Chinese character of a
    sentence to earlier noise/silence, yielding a segment such as "今" several
    seconds before "天...". Keep the text, but let the following turn's acoustic
    boundary define playback.
    """
    ordered = [dict(s) for s in sorted(segments, key=lambda x: (x.get("start", 0.0), x.get("end", 0.0)))]
    repaired: list[dict[str, Any]] = []
    i = 0
    while i < len(ordered):
        current = ordered[i]
        if i + 1 < len(ordered):
            nxt = ordered[i + 1]
            text = str(current.get("text", "")).strip()
            gap = float(nxt.get("start", 0.0)) - float(current.get("end", 0.0))
            same_speaker = current.get("speaker") == nxt.get("speaker")
            if (
                same_speaker
                and 0.0 <= gap <= max_gap
                and 0 < len(text) <= max_prefix_chars
                and float(current.get("end", 0.0)) - float(current.get("start", 0.0)) > 0.8
            ):
                merged = dict(nxt)
                merged["text"] = text + str(nxt.get("text", ""))
                merged["start_alignment_raw"] = round(float(current.get("start", 0.0)), 3)
                merged["start"] = round(max(0.0, float(nxt.get("start", 0.0)) - lead_pad_s), 3)
                repaired.append(merged)
                i += 2
                continue
        repaired.append(current)
        i += 1
    return repaired


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
    boundary_waveform = waveform.copy()
    asr_waveform = waveform

    prep_cfg = dict(mcfg.get("preprocessing", {}))
    if prep_cfg.get("light_denoise", False):
        asr_waveform = _light_denoise_waveform(
            waveform,
            sr,
            reduction=float(prep_cfg.get("denoise_reduction", 1.6)),
            gate_floor=float(prep_cfg.get("denoise_gate_floor", 0.18)),
        )
        logger.info("Light denoise enabled for ASR")

    logger.info("Audio loaded  duration=%.1fs  sr=%d", len(waveform) / sr, sr)

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
            asr_waveform,
            language=language,
            initial_prompt=initial_prompt,
            sample_rate=sr,
        )
    else:
        asr_result = _transcribe_speech_chunks(
            asr_backend,
            asr_waveform,
            sr,
            dia_result["segments"],
            language,
            initial_prompt,
        )

    asr_result["segments"] = _repair_english_boundary_prefixes(
        asr_backend,
        asr_waveform,
        sr,
        asr_result.get("segments", []),
        language,
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
    asr_segs = _filter_asr_by_speech_activity(asr_segs, boundary_waveform, sr)

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
    turn_limits = _turn_limits_for_language(language)
    try:
        aligned = whisperx.align(
            segments_for_align, align_model, align_metadata,
            asr_waveform, device=device, return_char_alignments=False,
        )

        # 4c. Diarization → pandas DataFrame; per-word speaker assignment
        dia_segs = dia_result["segments"]
        diarize_df = pd.DataFrame([
            {"start": s["start"], "end": s["end"], "speaker": s["speaker"]}
            for s in dia_segs
        ])
        result_with_spk = whisperx.assign_word_speakers(
            diarize_df, aligned, fill_nearest=True,
        )
        speaker_words = result_with_spk.get("word_segments", [])

        # 4d. Prefer word timestamps for timing, but keep ASR segment text.
        # Directly concatenating Chinese word chunks often fragments sentences.
        merged = _asr_segments_with_word_timing(asr_segs, speaker_words, dia_result["segments"])
        if not merged:
            merged = _aggregate_words_to_turns(
                speaker_words,
                max_gap=float(turn_limits["max_gap"]),
                max_dur=float(turn_limits["max_dur"]),
                max_chars=int(turn_limits["max_chars"]),
            )
        need_fallback, fallback_reason = _turns_need_segment_fallback(merged, asr_segs)
        if need_fallback:
            _warn_segment_fallback(fallback_reason)
            from src.alignment import align_segments
            merged = align_segments(
                dia_result["segments"],
                _presplit_segments(
                    asr_segs,
                    max_dur=float(turn_limits["max_dur"]),
                    max_chars=int(turn_limits["max_chars"]),
                ),
                min_speaker_ratio=0.15,
            )
            used_segment_fallback = True
        else:
            logger.info("  Word-level alignment accepted: %s", fallback_reason)
    except Exception as e:
        _warn_segment_fallback(f"alignment failed: {e}")
        from src.alignment import align_segments
        merged = align_segments(
            dia_result["segments"],
            _presplit_segments(
                asr_segs,
                max_dur=float(turn_limits["max_dur"]),
                max_chars=int(turn_limits["max_chars"]),
            ),
            min_speaker_ratio=0.15,
        )
        used_segment_fallback = True
    finally:
        del align_model

    if not merged:
        _warn_segment_fallback("no speaker turns")
        from src.alignment import align_segments
        merged = align_segments(
            dia_result["segments"],
            _presplit_segments(
                asr_segs,
                max_dur=float(turn_limits["max_dur"]),
                max_chars=int(turn_limits["max_chars"]),
            ),
            min_speaker_ratio=0.15,
        )
        used_segment_fallback = True

    # 4e. zh_simplify + speaker map (hallucination already filtered in 4a)
    merged = _filter_unreliable_turns(merged)
    for seg in merged:
        txt = clean_hallucination(seg.get("text", ""))
        seg["text"] = zh_simplify(txt)
    if language == "en" or not used_segment_fallback:
        merged = _merge_adjacent_turns(
            merged,
            max_gap=float(turn_limits["merge_gap"]),
            max_dur=float(turn_limits["merge_max_dur"]),
            max_chars=int(turn_limits["merge_max_chars"]),
        )
        if language != "en" and not used_segment_fallback:
            merged = _reattach_drifted_prefix_fragments(merged)
    merged = _filter_unreliable_turns(merged)
    merged = _dedupe_turn_boundaries(merged)
    merged = _refine_segment_playback_boundaries(
        merged, boundary_waveform, sr, dia_result["segments"],
    )
    merged = _adjust_neighbor_playback_boundaries(merged)
    merged = _annotate_low_confidence_segments(merged)

    for seg in merged:
        old_spk = seg.get("speaker", "")
        seg["speaker_original"] = old_spk

    timing["alignment"] = round(time.time() - t4, 2)
    logger.info("  → %d speaker turns from %d words (%.1fs)",
                len(merged), len(speaker_words), timing["alignment"])

    # ---- Cleanup (free GPU before LLM API calls) ------------------------
    dia_backend.unload()
    del dia_backend, asr_backend, waveform, asr_waveform, boundary_waveform
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

        for seg in merged:
            seg.setdefault("text_before_llm", seg.get("text", ""))

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

        for seg in merged:
            seg["phase"] = "corrected"

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
        "metadata": {
            "alignment_level": "segment" if used_segment_fallback else "word",
            "used_segment_fallback": used_segment_fallback,
            "word_segments": len(speaker_words),
        },
    }
