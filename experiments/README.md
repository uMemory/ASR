# Evaluation README

This directory contains evaluation tools for saved transcription JSON files.
The evaluator reads existing outputs from `tests/test_results` and does not
rerun ASR by default.

## Current Evaluation Positioning

The system outputs user-facing speaker turns: each segment is intended to be
readable, clickable, editable, and searchable. Public datasets such as
AISHELL-4 and AliMeeting often provide much finer TextGrid intervals. Because
of this granularity mismatch, segment-level boundary metrics and DER/JER can be
misleadingly low even when the transcript is usable.

For the current project stage, use these as primary indicators:

- Chinese text recognition: `CER`, insertion, deletion, substitution.
- LLM impact: compare `text_before_llm` with final text when available.
- Reliability: low-confidence segment rate.
- Search usability: qualitative Top-K query checks over saved transcripts.
- English/multi-speaker samples without transcripts: qualitative transcript,
  speaker-switching, playback, and cross-language retrieval checks.

Use these only as auxiliary observations:

- `Start MAE`, `End MAE`, `Hit@0.5s`, `Hit@1.0s`.
- `DER`, `JER`, speaker turn accuracy.

These auxiliary metrics are still computed by the script, but they should not
be used as the main conclusion when the reference and prediction segmentation
granularity differs.

## Main Script

```powershell
conda run -n TTS python -B experiments/evaluate_system.py
```

## Saved Outputs Used in This Project

The current evaluation is based on the files already processed into:

```text
tests/test_results
```

Only the result JSON files that match local TextGrid references can be scored
automatically. English samples without official transcript references are kept
as qualitative demonstrations.

## Chinese Text Metrics

For AISHELL-1 Chinese ASR samples:

```powershell
conda run -n TTS python -B experiments/evaluate_system.py `
  --dataset aishell1 `
  --pred-dir tests/test_results `
  --language zh `
  --compare-llm `
  --output outputs/eval_aishell1.json
```

For LibriSpeech English ASR samples:

```powershell
conda run -n TTS python -B experiments/evaluate_system.py `
  --dataset librispeech `
  --pred-dir tests/test_results `
  --language en `
  --compare-llm `
  --output outputs/eval_librispeech.json
```

For legacy TextGrid datasets:

```powershell
conda run -n TTS python -B experiments/evaluate_system.py `
  --dataset alimeeting-near `
  --pred-dir tests/test_results `
  --compare-llm `
  --output outputs/eval_alimeeting_near.json
```

For AISHELL-4 samples:

```powershell
conda run -n TTS python -B experiments/evaluate_system.py `
  --dataset aishell4 `
  --pred tests/test_results/L_R004S01C01.flac.json `
         tests/test_results/M_R003S01C01.flac.json `
         tests/test_results/S_R003S01C01.flac.json `
  --duration 240 `
  --compare-llm `
  --output outputs/eval_aishell4_240.json
```

For far-field samples, prefer a fixed duration because full-file speaker
metrics can be slow:

```powershell
conda run -n TTS python -B experiments/evaluate_system.py `
  --dataset alimeeting-far `
  --pred tests/test_results/R8007_M8010_MS803.wav.json `
  --duration 240 `
  --compare-llm `
  --output outputs/eval_alimeeting_far_240.json
```

## LLM Before/After Evaluation

If the JSON contains `text_before_llm`, `--compare-llm` reports:

- final `CER/WER`
- pre-LLM `CER/WER`
- delta in error rate
- insertion/deletion/substitution changes

A negative delta means the LLM reduced text error rate. A positive delta means
it made the strict reference metric worse. This does not necessarily mean the
text is less readable, because LLM punctuation and normalization may differ
from reference transcription conventions.

## English Dataset Notes

The current English samples in `tests/test_results` do not have official
reference transcripts in this workspace. They are useful for qualitative
checks:

- multiple speakers are present;
- the audio is continuous and easier to listen to than AMI Parquet-reconstructed
  clips;
- playback alignment and cross-language retrieval can be demonstrated.

If a quantitative English ASR score is required later, use a dataset with
official transcripts:

- LibriSpeech `test-clean` or `test-other` for English WER sanity checks;
- AMI original continuous audio plus original annotations if meeting-style
  English evaluation is required.

VoxConverse-style files are better for speaker diarization demonstrations than
WER, because they commonly provide RTTM speaker activity labels rather than
complete manual transcripts.

## Manual Reference Templates

For a short English or custom clip, export a template:

```powershell
conda run -n TTS python -B experiments/evaluate_system.py `
  --dataset alimeeting-near `
  --pred tests/test_results/ahnss.wav.json `
  --export-reference-template references/ahnss_reference_template.json
```

Then manually edit `segments[start,end,speaker,text]` and evaluate with:

```powershell
conda run -n TTS python -B experiments/evaluate_system.py `
  --dataset alimeeting-near `
  --pred tests/test_results/ahnss.wav.json `
  --reference-json references/ahnss_reference_template.json `
  --language en
```
