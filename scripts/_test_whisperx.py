"""Quick test: can WhisperX load and run end-to-end on this system?"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Patch cuDNN DLLs (same as pipeline startup)
from src.utils.cuda_dlls import ensure_cuda_dlls
ensure_cuda_dlls()

import whisperx

# Test 1: load model
print("=== Test 1: load_model ===")
t0 = time.time()
model = whisperx.load_model("models/whisper-medium", device="cuda", compute_type="float16")
print(f"  OK ({time.time()-t0:.1f}s)  type={type(model).__name__}")

# Test 2: load alignment model
print("=== Test 2: load_align_model ===")
t0 = time.time()
align_model, align_metadata = whisperx.load_align_model(
    language_code="zh",
    device="cuda",
    model_name="models/wav2vec2-large-xlsr-53-chinese-zh-cn",
)
print(f"  OK ({time.time()-t0:.1f}s)")

# Test 3: transcribe + align short audio
print("=== Test 3: transcribe + align ===")
audio_path = "dataset/AISHELL-4/test/wav/S_R003S01C01.flac"
audio = whisperx.load_audio(audio_path)
audio = audio[:16000 * 30]  # first 30s
print(f"  audio: {len(audio)/16000:.1f}s")

t0 = time.time()
result = model.transcribe(audio, batch_size=8, language="zh")
print(f"  transcribe OK ({time.time()-t0:.1f}s)  segments={len(result['segments'])}")

t0 = time.time()
result = whisperx.align(
    result["segments"], align_model, align_metadata, audio,
    device="cuda", return_char_alignments=False,
)
print(f"  align OK ({time.time()-t0:.1f}s)  words={len(result.get('word_segments',[]))}")

# Show word-level timestamps
words = result.get("word_segments", [])
if words:
    print(f"  First 5 words:")
    for w in words[:5]:
        print(f"    [{w['start']:.2f}-{w['end']:.2f}] {w['word']}")

# Test 4: diarization + assign_word_speakers
print("=== Test 4: diarization + assign_word_speakers ===")
from src.utils.config import get_env
hf_token = get_env("HF_TOKEN")
from pyannote.audio import Pipeline

# Need to import librosa first to avoid speechbrain conflict
import librosa

t0 = time.time()
dia_pipe = Pipeline.from_pretrained("models/speaker-diarization-3.1/config.yaml")
dia_pipe.to(whisperx.torch.device("cuda"))
print(f"  diarization model loaded ({time.time()-t0:.1f}s)")

t0 = time.time()
diarize_result = dia_pipe({
    "waveform": whisperx.torch.from_numpy(audio[None, :]),
    "sample_rate": 16000,
})
import pandas as pd
diarize_df = pd.DataFrame(
    diarize_result.itertracks(yield_label=True),
    columns=['segment', 'label', 'speaker']
)
diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)
print(f"  diarization OK ({time.time()-t0:.1f}s)  speakers={diarize_df['speaker'].nunique()}")

t0 = time.time()
result_with_speakers = whisperx.assign_word_speakers(diarize_df, result)
print(f"  assign_word_speakers OK ({time.time()-t0:.1f}s)")

# Check word-level speakers
words = result_with_speakers.get("word_segments", [])
words_with_spk = [w for w in words if 'speaker' in w]
print(f"  words with speaker: {len(words_with_spk)}/{len(words)}")

# Show some
for w in words_with_spk[:10]:
    print(f"    [{w['start']:.2f}-{w['end']:.2f}] {w.get('speaker','?'):12s} {w['word']}")

print("\n=== ALL TESTS PASSED ===")
