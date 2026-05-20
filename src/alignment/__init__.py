"""VAD AND timestamp alignment strategy.

Aligns Pyannote speaker labels to ASR (Whisper) segment boundaries.

Core idea:
    Whisper's Silero VAD and Pyannote's internal VAD produce slightly
    different timestamps.  Instead of choosing one, we use the **ASR
    segment boundaries as the "master clock"** and ask: for each spoken
    chunk, which speaker has the most acoustic overlap?

Algorithm:
    1. For each ASR segment ``[asr_start, asr_end]``, find every
       diarization segment that overlaps it.
    2. Sum the overlap duration per speaker.
    3. Assign the speaker whose overlap fraction exceeds ``min_speaker_ratio``
       (default 0.5).  If no speaker reaches the threshold, label the segment
       ``SPEAKER_UNKNOWN``.
"""
from __future__ import annotations

from typing import Any

SPEAKER_UNKNOWN = "SPEAKER_UNKNOWN"


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Duration of overlap between two intervals [a_start, a_end) and [b_start, b_end)."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def align_segments(
    diarization_segments: list[dict[str, Any]],
    asr_segments: list[dict[str, Any]],
    min_speaker_ratio: float = 0.5,
    min_speaker_overlap_s: float = 0.25,
) -> list[dict[str, Any]]:
    """Align ASR text segments with speaker labels from diarization.

    Parameters
    ----------
    diarization_segments : list[dict]
        Each dict has ``start``, ``end``, and ``speaker`` keys.
        Output of ``PyannoteDiarizationBackend.diarize()["segments"]``.
    asr_segments : list[dict]
        Each dict has ``start``, ``end``, and ``text`` keys.
        Output of ASR backend ``transcribe()["segments"]``.
    min_speaker_ratio : float
        Minimum fraction of the ASR segment that a speaker must occupy
        to be assigned (default 0.5 = 50 %).
    min_speaker_overlap_s : float
        Absolute overlap floor. This handles ASR segments that include long
        pauses: a speaker can be confidently present even if the overlap ratio
        is low because the segment duration includes silence.

    Returns
    -------
    list[dict]
        Merged segments with keys ``start``, ``end``, ``speaker``, ``text``.
    """
    merged: list[dict[str, Any]] = []

    for asr_seg in asr_segments:
        a_start = asr_seg["start"]
        a_end = asr_seg["end"]
        a_dur = a_end - a_start

        if a_dur <= 0:
            continue

        # Collect overlap durations per speaker
        speaker_overlap: dict[str, float] = {}

        for dia_seg in diarization_segments:
            d_start = dia_seg["start"]
            d_end = dia_seg["end"]
            overlap = _overlap(a_start, a_end, d_start, d_end)
            if overlap > 0:
                spk = dia_seg["speaker"]
                speaker_overlap[spk] = speaker_overlap.get(spk, 0.0) + overlap

        if not speaker_overlap:
            # No diarization overlap at all → unknown
            assigned_speaker = SPEAKER_UNKNOWN
        else:
            # Find the dominant speaker
            best_speaker = max(speaker_overlap, key=speaker_overlap.get)  # type: ignore[arg-type]
            best_overlap = speaker_overlap[best_speaker]
            best_ratio = best_overlap / a_dur
            assigned_speaker = (
                best_speaker
                if best_ratio >= min_speaker_ratio or best_overlap >= min_speaker_overlap_s
                else SPEAKER_UNKNOWN
            )

        merged.append({
            "start": round(a_start, 3),
            "end": round(a_end, 3),
            "speaker": assigned_speaker,
            "text": asr_seg.get("text", "").strip(),
        })

    return merged
