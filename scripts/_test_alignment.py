"""Test: forced alignment with wav2vec2 + word-level speaker assignment.

IMPORTANT: whisperx must be imported BEFORE pyannote/torch to avoid CUDA conflicts.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Critical import order for Windows CUDA ──
from src.utils.cuda_dlls import add_cuda_dll_dirs
add_cuda_dll_dirs()

# 1. whisperx FIRST (triggers faster-whisper/ctranslate2, must be before pyannote)
import whisperx

# 2. torch (via transformers)
import torch
import soundfile as sf
import numpy as np

# 3. librosa BEFORE pyannote (avoids speechbrain conflict)
import librosa

print("=== Step 1: transformers Whisper ASR ===")
from src.asr.transformers_whisper_backend import TransformersWhisperBackend
asr = TransformersWhisperBackend(model_path="models/whisper-medium", device="cuda", compute_type="float16")
asr.load()

audio_path = "dataset/AISHELL-4/test/wav/S_R003S01C01.flac"
audio, sr = sf.read(audio_path)
if audio.ndim > 1:
    audio = audio.mean(axis=1)
audio = audio.astype(np.float32)[:16000 * 30]

t0 = time.time()
asr_result = asr.transcribe(audio, language="zh")
print(f"  ASR OK ({time.time()-t0:.1f}s)  segments={len(asr_result['segments'])}")
for s in asr_result['segments'][:3]:
    print(f"    [{s['start']:.1f}-{s['end']:.1f}] {len(s['text'])}c: {s['text'][:60]}...")
asr.unload()

print("\n=== Step 2: Forced alignment (wav2vec2 via whisperx.align) ===")
t0 = time.time()
align_model, align_metadata = whisperx.load_align_model(
    language_code="zh", device="cuda",
    model_name="models/wav2vec2-large-xlsr-53-chinese-zh-cn",
)
print(f"  loaded ({time.time()-t0:.1f}s)")

segments_for_align = [
    {"start": s["start"], "end": s["end"], "text": s["text"]}
    for s in asr_result["segments"]
]

t0 = time.time()
aligned = whisperx.align(
    segments_for_align, align_model, align_metadata,
    audio, device="cuda", return_char_alignments=False,
)
n_words = len(aligned.get("word_segments", []))
print(f"  aligned ({time.time()-t0:.1f}s)  words={n_words}")

words = aligned.get("word_segments", [])
for w in words[:10]:
    print(f"    [{w['start']:.2f}-{w['end']:.2f}] s={w.get('score',0):.2f} {w['word']}")

print("\n=== Step 3: Word-level speaker assignment ===")
from src.diarization import load_diarization_backend
dia_backend = load_diarization_backend({
    "pipeline_config": "models/speaker-diarization-3.1/config.yaml",
    "device": "cuda",
})
dia_backend.load()
dia_result = dia_backend.diarize(audio, sample_rate=16000)
print(f"  diarization: {len(dia_result['segments'])} segs, {dia_result['num_speakers']} speakers")

import pandas as pd
dia_segs = dia_result["segments"]
diarize_df = pd.DataFrame([
    {"start": s["start"], "end": s["end"], "speaker": s["speaker"]}
    for s in dia_segs
])

t0 = time.time()
result_with_spk = whisperx.assign_word_speakers(diarize_df, aligned)
print(f"  assign_word_speakers OK ({time.time()-t0:.1f}s)")

words = result_with_spk.get("word_segments", [])
words_with_spk = [w for w in words if 'speaker' in w]
print(f"  words with speaker: {len(words_with_spk)}/{len(words)}")
for w in words_with_spk[:15]:
    print(f"    [{w['start']:.2f}-{w['end']:.2f}] {w.get('speaker','?'):16s} {w['word']}")

print("\n=== Step 4: Turn segmentation ===")
def words_to_turns(words, max_gap=0.5, max_dur=15.0, max_chars=80):
    turns = []
    current = None
    for w in words:
        spk = w.get('speaker', 'SPEAKER_UNKNOWN')
        w_start = w['start']; w_end = w['end']; w_text = w['word']
        need_new = (
            current is None
            or spk != current['speaker']
            or w_start - current['end'] > max_gap
            or (current['end'] - current['start']) >= max_dur
            or len(current['text']) >= max_chars
        )
        if need_new:
            if current:
                turns.append(current)
            current = {'start': w_start, 'end': w_end, 'speaker': spk, 'text': w_text}
        else:
            current['end'] = w_end
            current['text'] += w_text
    if current:
        turns.append(current)
    return turns

turns = words_to_turns(words_with_spk)
print(f"  turns: {len(turns)}")
for t in turns[:10]:
    dur = t['end'] - t['start']
    print(f"    [{t['start']:.1f}-{t['end']:.1f}] {dur:.1f}s {len(t['text'])}c {t['speaker']}: {t['text'][:60]}")

max_d = max(t['end'] - t['start'] for t in turns) if turns else 0
max_c = max(len(t['text']) for t in turns) if turns else 0
print(f"  max turn: {max_d:.1f}s, {max_c} chars")

dia_backend.unload()
del align_model
if torch.cuda.is_available():
    torch.cuda.empty_cache()
print("\n=== ALL TESTS PASSED ===")
