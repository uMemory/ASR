"""30-second chunk streaming processor.

Splits long audio into overlapping chunks, runs the pipeline on each,
and merges the results with deduplication.  This simulates quasi-real-time
processing for meeting scenarios.

Usage:
    from src.streaming import ChunkProcessor
    cp = ChunkProcessor(chunk_duration_s=30, overlap_s=5)
    for result in cp.process_file("meeting.wav"):
        print(f"Chunk {result['chunk_index']}: {len(result['segments'])} segments")
    all_merged = cp.process_file_merged("meeting.wav")
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from src.pipeline import run
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ChunkConfig:
    chunk_duration_s: float = 30.0
    overlap_s: float = 5.0           # cross-chunk overlap to avoid boundary cuts
    language: str = "zh"
    profile: str | None = None


class ChunkProcessor:
    """30-second chunk streaming processor.

    Parameters
    ----------
    chunk_duration_s : float
        Duration of each audio chunk in seconds (default 30).
    overlap_s : float
        Overlap between consecutive chunks (default 5).  Segments that
        fall entirely within the overlap zone are duplicated across
        chunks, and duplicates are removed during merge.
    language : str
        ASR language code.
    profile : str or None
        Config profile override.
    """

    def __init__(
        self,
        chunk_duration_s: float = 30.0,
        overlap_s: float = 5.0,
        language: str = "zh",
        profile: str | None = None,
    ) -> None:
        if overlap_s >= chunk_duration_s:
            raise ValueError(
                f"overlap ({overlap_s}s) must be less than chunk ({chunk_duration_s}s)"
            )
        self._cfg = ChunkConfig(
            chunk_duration_s=chunk_duration_s,
            overlap_s=overlap_s,
            language=language,
            profile=profile,
        )

    # ── Public API ────────────────────────────────────────────────────

    def process_file(
        self,
        audio_path: str | Path,
    ) -> Iterator[dict[str, Any]]:
        """Process audio chunk-by-chunk, yielding results as they complete.

        Each yielded dict contains:
            chunk_index : int
            chunk_start_s : float
            segments : list[dict]
            timing : dict
        """
        audio, sr = self._load_audio(str(audio_path))

        total = len(audio) / sr
        n_chunks = max(1, self._chunk_count(total))
        logger.info(
            "Streaming: %s  total=%.1fs  sr=%d  chunks=%d  chunk=%.0fs  overlap=%.0fs",
            Path(audio_path).name,
            total,
            sr,
            n_chunks,
            self._cfg.chunk_duration_s,
            self._cfg.overlap_s,
        )

        for ci in range(n_chunks):
            t_start = ci * (self._cfg.chunk_duration_s - self._cfg.overlap_s)
            t_end = t_start + self._cfg.chunk_duration_s
            chunk = self._slice(audio, sr, t_start, t_end)

            result = run(
                str(audio_path),
                language=self._cfg.language,
                profile=self._cfg.profile,
                waveform=chunk,
                sample_rate=sr,
                llm_overrides={"enabled": False},
            )

            # Adjust timestamps to global time
            for seg in result["segments"]:
                seg["start"] = round(seg["start"] + t_start, 3)
                seg["end"] = round(seg["end"] + t_start, 3)

            yield {
                "chunk_index": ci,
                "chunk_start_s": round(t_start, 1),
                "segments": result["segments"],
                "timing": result.get("timing", {}),
            }

    def process_file_merged(
        self,
        audio_path: str | Path,
    ) -> dict[str, Any]:
        """Process all chunks and return merged, deduplicated segments.

        Returns the same dict shape as ``pipeline.run()`` with
        ``segments``, ``num_speakers``, ``language``, ``timing``.
        """
        all_segments: list[dict[str, Any]] = []
        total_timing: dict[str, float] = {}

        for chunk_result in self.process_file(audio_path):
            all_segments.extend(chunk_result["segments"])
            for k, v in chunk_result.get("timing", {}).items():
                total_timing[k] = total_timing.get(k, 0) + v

        merged = _deduplicate_segments(all_segments, self._cfg.overlap_s)
        return {
            "segments": merged,
            "num_speakers": len({s.get("speaker", "") for s in merged}),
            "language": self._cfg.language,
            "timing": total_timing,
        }

    # ── Internals ─────────────────────────────────────────────────────

    def _chunk_count(self, total_s: float) -> int:
        step = self._cfg.chunk_duration_s - self._cfg.overlap_s
        if step <= 0 or total_s <= self._cfg.chunk_duration_s:
            return 1
        return max(1, int(np.ceil((total_s - self._cfg.overlap_s) / step)))

    @staticmethod
    def _load_audio(path: str) -> tuple[np.ndarray, int]:
        import soundfile as sf

        data, sr = sf.read(str(path))
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data.astype(np.float32), int(sr)

    @staticmethod
    def _slice(
        audio: np.ndarray,
        sr: int,
        start_s: float,
        end_s: float,
    ) -> np.ndarray:
        lo = int(start_s * sr)
        hi = int(end_s * sr)
        chunk = audio[lo:hi]
        # Pad short final chunk to full length
        target_len = int((end_s - start_s) * sr)
        if len(chunk) < target_len:
            padded = np.zeros(target_len, dtype=audio.dtype)
            padded[: len(chunk)] = chunk
            return padded
        return chunk


# ── Deduplication ──────────────────────────────────────────────────────

def _deduplicate_segments(
    segments: list[dict[str, Any]],
    overlap_s: float,
) -> list[dict[str, Any]]:
    """Remove duplicate segments that appear in chunk overlap zones.

    Two segments are considered duplicates if they have overlapping time
    spans and identical text.  The earlier segment is kept.
    """
    if not segments:
        return []

    segments.sort(key=lambda s: (s.get("start", 0), s.get("end", 0)))
    merged: list[dict[str, Any]] = []

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        # Check if this segment duplicates a previously-kept one
        dup = False
        for prev in merged:
            # Same text AND overlapping time window
            if prev.get("text", "").strip() == text:
                p_start = prev.get("start", 0)
                p_end = prev.get("end", 0)
                s_start = seg.get("start", 0)
                s_end = seg.get("end", 0)
                overlap = max(0.0, min(p_end, s_end) - max(p_start, s_start))
                if overlap > 0:
                    dup = True
                    break
        if not dup:
            merged.append(seg)

    return merged
