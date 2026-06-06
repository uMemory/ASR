"""Automatic three-layer evaluation for saved ASR result JSON files.

This evaluator intentionally avoids metrics that require manual per-segment
annotation. It supports TextGrid references for AISHELL-4/AliMeeting and
plain transcript references for AISHELL-1/LibriSpeech.

Examples:
    python experiments/evaluate_system.py --dataset alimeeting-near --pred-dir tests/test_results
    python experiments/evaluate_system.py --dataset aishell4 --pred-dir tests/test_results --duration 240
    python experiments/evaluate_system.py --pred tests/test_results/R8007_M8011_N_SPK8066.wav.json --dataset alimeeting-near
    python experiments/evaluate_system.py --dataset aishell1 --pred-dir tests/test_results --language zh
    python experiments/evaluate_system.py --dataset librispeech --pred-dir tests/test_results --language en
"""
from __future__ import annotations

import argparse
import json
import math
import re
import string
import sys
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.table import Table

from src.utils.config import project_root

console = Console()


PUNCT_RE = re.compile(r"[\s，。！？、；：,.!?;:\"'“”‘’（）()《》<>【】\[\]{}\-—_…·/\\|]+")
TAG_RE = re.compile(r"<[^>]+>")


@dataclass
class EvalConfig:
    language: str = "zh"
    duration: float | None = None
    boundary_tolerance: float = 0.2
    min_overlap: float = 0.05


TEXT_ONLY_DATASETS = {"aishell1", "librispeech"}


def dataset_paths(dataset: str, root: Path) -> tuple[Path, Path | None]:
    if dataset == "aishell4":
        tg_dir = root / "dataset/AISHELL-4/test/TextGrid"
        rttm_dir = tg_dir
    elif dataset == "alimeeting-near":
        tg_dir = root / "dataset/AIMeeting/Eval_Ali/Eval_Ali_near/textgrid_dir"
        rttm_dir = None
    elif dataset == "alimeeting-far":
        tg_dir = root / "dataset/AIMeeting/Eval_Ali/Eval_Ali_far/textgrid_dir"
        rttm_dir = None
    elif dataset == "aishell1":
        tg_dir = root / "dataset/AISHELL/transcript/aishell_transcript_v0.8.txt"
        rttm_dir = None
    elif dataset == "librispeech":
        tg_dir = root / "dataset/LibriSpeech"
        rttm_dir = None
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return tg_dir, rttm_dir


def strip_result_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".json"):
        name = name[:-5]
    for suffix in (".wav", ".flac", ".mp3", ".m4a"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def find_textgrid(pred_json: Path, dataset: str, tg_dir: Path) -> Path | None:
    base = strip_result_suffix(pred_json)
    candidates = [tg_dir / f"{base}.TextGrid"]

    if dataset == "alimeeting-far":
        # Far audio names look like R8007_M8011_MS806.wav, while TextGrid is
        # R8007_M8011.TextGrid.
        parts = base.split("_")
        if len(parts) >= 2:
            candidates.append(tg_dir / f"{parts[0]}_{parts[1]}.TextGrid")

    for cand in candidates:
        if cand.exists():
            return cand
    return None


def load_reference_json(path: Path, cfg: EvalConfig) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    segments = data.get("segments", data if isinstance(data, list) else [])
    rows: list[dict[str, Any]] = []
    for item in segments:
        text = str(item.get("text", "")).strip()
        if not clean_text(text, cfg.language):
            continue
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        if cfg.duration is not None and start >= cfg.duration:
            continue
        if cfg.duration is not None:
            end = min(end, cfg.duration)
        if end <= start:
            continue
        rows.append({
            "start": start,
            "end": end,
            "speaker": str(item.get("speaker", "REF")),
            "text": text,
        })
    rows.sort(key=lambda x: (x["start"], x["end"]))
    return rows


_AISHELL1_CACHE: dict[Path, dict[str, str]] = {}
_LIBRISPEECH_CACHE: dict[Path, dict[str, str]] = {}


def _reference_duration(hyp: list[dict[str, Any]], cfg: EvalConfig) -> float:
    if cfg.duration is not None:
        return cfg.duration
    if hyp:
        return max(float(s.get("end", s.get("start", 0.0))) for s in hyp)
    return 1.0


def _single_reference_segment(text: str, hyp: list[dict[str, Any]], cfg: EvalConfig) -> list[dict[str, Any]]:
    if not clean_text(text, cfg.language):
        return []
    return [{
        "start": 0.0,
        "end": max(0.01, _reference_duration(hyp, cfg)),
        "speaker": "REF",
        "text": text,
    }]


def _load_aishell1_transcripts(path: Path) -> dict[str, str]:
    path = path.resolve()
    if path in _AISHELL1_CACHE:
        return _AISHELL1_CACHE[path]
    refs: dict[str, str] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) == 2:
                    refs[parts[0]] = parts[1]
    _AISHELL1_CACHE[path] = refs
    return refs


def _load_librispeech_transcripts(root: Path) -> dict[str, str]:
    root = root.resolve()
    if root in _LIBRISPEECH_CACHE:
        return _LIBRISPEECH_CACHE[root]
    refs: dict[str, str] = {}
    if root.exists():
        for trans_path in root.rglob("*.trans.txt"):
            with open(trans_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        refs[parts[0]] = parts[1]
    _LIBRISPEECH_CACHE[root] = refs
    return refs


def load_plain_transcript_reference(
    pred_json: Path,
    dataset: str,
    ref_path: Path,
    hyp: list[dict[str, Any]],
    cfg: EvalConfig,
) -> list[dict[str, Any]] | None:
    base = strip_result_suffix(pred_json)
    if dataset == "aishell1":
        text = _load_aishell1_transcripts(ref_path).get(base)
    elif dataset == "librispeech":
        text = _load_librispeech_transcripts(ref_path).get(base)
    else:
        text = None
    if not text:
        return None
    return _single_reference_segment(text, hyp, cfg)


def export_reference_template(pred_json: Path, output: Path) -> None:
    data = json.loads(pred_json.read_text(encoding="utf-8"))
    segments = data.get("segments", data if isinstance(data, list) else [])
    template = {
        "source_prediction": str(pred_json),
        "note": "Edit text/speaker/start/end to create a manual reference. Keep only reviewed segments.",
        "segments": [
            {
                "start": float(s.get("start", 0.0)),
                "end": float(s.get("end", s.get("start", 0.0))),
                "speaker": s.get("speaker", "REF"),
                "text": s.get("text_before_llm") or s.get("text", ""),
            }
            for s in segments
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(template, ensure_ascii=False, indent=2), encoding="utf-8")


def find_rttm(pred_json: Path, rttm_dir: Path | None) -> Path | None:
    if rttm_dir is None:
        return None
    base = strip_result_suffix(pred_json)
    cand = rttm_dir / f"{base}.rttm"
    return cand if cand.exists() else None


def infer_speaker_from_filename(pred_json: Path) -> str | None:
    base = strip_result_suffix(pred_json)
    m = re.search(r"_N_(SPK\d+)$", base)
    return m.group(1) if m else None


def clean_text(text: str, language: str) -> str:
    text = TAG_RE.sub("", text or "")
    text = text.replace("<sil>", "")
    if language == "zh":
        return PUNCT_RE.sub("", text).strip().lower()
    text = text.lower()
    table = str.maketrans("", "", string.punctuation)
    return " ".join(text.translate(table).split())


def tokenize(text: str, language: str) -> list[str]:
    cleaned = clean_text(text, language)
    if not cleaned:
        return []
    if language == "zh":
        return list(cleaned)
    return cleaned.split()


def edit_counts(ref: list[str], hyp: list[str]) -> dict[str, int | float]:
    """Levenshtein counts with substitution/insertion/deletion breakdown."""
    n, m = len(ref), len(hyp)
    prev = [(i, 0, 0, i) for i in range(n + 1)]
    for j in range(1, m + 1):
        curr = [(j, 0, j, 0)]
        for i in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                best = prev[i - 1]
            else:
                d, s, ins, dele = prev[i - 1]
                best = (d + 1, s + 1, ins, dele)

            d, s, ins, dele = curr[i - 1]
            ins_cand = (d + 1, s, ins + 1, dele)
            d, s, ins, dele = prev[i]
            del_cand = (d + 1, s, ins, dele + 1)
            curr.append(min(best, ins_cand, del_cand, key=lambda x: (x[0], x[1] + x[2] + x[3])))
        prev = curr
    dist, subs, inserts, deletes = prev[n]
    denom = max(1, n)
    return {
        "distance": dist,
        "substitutions": subs,
        "insertions": inserts,
        "deletions": deletes,
        "error_rate": dist / denom,
        "ref_units": n,
        "hyp_units": m,
    }


def parse_textgrid(tg_path: Path, cfg: EvalConfig, default_speaker: str | None = None) -> list[dict[str, Any]]:
    import textgrid as tg_lib

    tg = tg_lib.TextGrid.fromFile(str(tg_path))
    rows: list[dict[str, Any]] = []
    for tier in tg.tiers:
        tier_speaker = default_speaker or str(tier.name)
        for iv in tier.intervals:
            text = (iv.mark or "").strip()
            cleaned = clean_text(text, cfg.language)
            if not cleaned:
                continue
            start = float(iv.minTime)
            end = float(iv.maxTime)
            if cfg.duration is not None and start >= cfg.duration:
                continue
            if cfg.duration is not None:
                end = min(end, cfg.duration)
            if end <= start:
                continue
            rows.append({"start": start, "end": end, "speaker": tier_speaker, "text": text})
    rows.sort(key=lambda x: (x["start"], x["end"]))
    return rows


def load_prediction(pred_json: Path, cfg: EvalConfig) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = json.loads(pred_json.read_text(encoding="utf-8"))
    segments = data.get("segments", data if isinstance(data, list) else [])
    kept: list[dict[str, Any]] = []
    for item in segments:
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        if cfg.duration is not None and start >= cfg.duration:
            continue
        if cfg.duration is not None:
            end = min(end, cfg.duration)
        if end <= start:
            continue
        row = dict(item)
        row["start"] = start
        row["end"] = end
        kept.append(row)
    kept.sort(key=lambda x: (x["start"], x["end"]))
    return kept, data.get("metadata", {})


def segment_text(seg: dict[str, Any], mode: str = "final") -> str:
    if mode == "pre_llm":
        return str(
            seg.get("text_before_llm")
            or seg.get("text_original_asr")
            or seg.get("text")
            or ""
        )
    return str(seg.get("text") or "")


def concat_text(segments: list[dict[str, Any]], language: str, mode: str = "final") -> str:
    return "".join(clean_text(segment_text(s, mode), language) for s in segments)


def overlap(a: dict[str, Any], b: dict[str, Any]) -> float:
    return max(0.0, min(float(a["end"]), float(b["end"])) - max(float(a["start"]), float(b["start"])))


def match_by_overlap(ref: list[dict[str, Any]], hyp: list[dict[str, Any]], min_overlap: float) -> list[tuple[dict, dict, float]]:
    matches: list[tuple[dict, dict, float]] = []
    for r in ref:
        best_h = None
        best_ov = 0.0
        for h in hyp:
            ov = overlap(r, h)
            if ov > best_ov:
                best_ov = ov
                best_h = h
        if best_h is not None and best_ov >= min_overlap:
            matches.append((r, best_h, best_ov))
    return matches


def time_metrics(ref: list[dict[str, Any]], hyp: list[dict[str, Any]], cfg: EvalConfig) -> dict[str, Any]:
    matches = match_by_overlap(ref, hyp, cfg.min_overlap)
    if not matches:
        return {
            "matched_ref_segments": 0,
            "match_rate": 0.0,
            "start_mae": None,
            "end_mae": None,
            "hit_0_5": 0.0,
            "hit_1_0": 0.0,
            "coverage_rate": 0.0,
            "silent_prefix_avg": None,
            "suffix_extra_avg": None,
        }

    start_errs: list[float] = []
    end_errs: list[float] = []
    hit05 = 0
    hit10 = 0
    covered = 0
    prefix_extra: list[float] = []
    suffix_extra: list[float] = []

    for r, h, _ in matches:
        hs = float(h.get("playback_start", h["start"]))
        he = float(h.get("playback_end", h["end"]))
        rs = float(r["start"])
        re = float(r["end"])
        se = abs(hs - rs)
        ee = abs(he - re)
        start_errs.append(se)
        end_errs.append(ee)
        if se <= 0.5 and ee <= 0.5:
            hit05 += 1
        if se <= 1.0 and ee <= 1.0:
            hit10 += 1
        if hs <= rs + cfg.boundary_tolerance and he >= re - cfg.boundary_tolerance:
            covered += 1
        prefix_extra.append(max(0.0, rs - hs))
        suffix_extra.append(max(0.0, he - re))

    n = len(matches)
    return {
        "matched_ref_segments": n,
        "match_rate": n / max(1, len(ref)),
        "start_mae": sum(start_errs) / n,
        "end_mae": sum(end_errs) / n,
        "hit_0_5": hit05 / n,
        "hit_1_0": hit10 / n,
        "coverage_rate": covered / n,
        "silent_prefix_avg": sum(prefix_extra) / n,
        "suffix_extra_avg": sum(suffix_extra) / n,
    }


def confidence_metrics(segments: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    total = len(segments)
    low = sum(1 for s in segments if s.get("low_confidence"))
    fallback = metadata.get("used_segment_fallback")
    level = metadata.get("alignment_level")
    return {
        "segments": total,
        "low_confidence_segments": low,
        "low_confidence_rate": low / total if total else 0.0,
        "alignment_level": level,
        "used_segment_fallback": fallback,
        "fallback_rate": float(fallback) if isinstance(fallback, bool) else None,
    }


def annotation_from_segments(segments: list[dict[str, Any]]):
    from pyannote.core import Annotation, Segment

    ann = Annotation()
    for s in segments:
        speaker = str(s.get("speaker") or "UNKNOWN")
        ann[Segment(float(s["start"]), float(s["end"]))] = speaker
    return ann


def diarization_metrics(ref: list[dict[str, Any]], hyp: list[dict[str, Any]], cfg: EvalConfig) -> dict[str, Any]:
    try:
        from pyannote.core import Segment
        from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
    except Exception as exc:
        return {"der": None, "jer": None, "speaker_eval_error": str(exc)}

    ref_ann = annotation_from_segments(ref)
    hyp_ann = annotation_from_segments(hyp)
    if cfg.duration is not None:
        window = Segment(0, cfg.duration)
        ref_ann = ref_ann.crop(window)
        hyp_ann = hyp_ann.crop(window)
    try:
        der = float(DiarizationErrorRate(collar=0.25, skip_overlap=False)(ref_ann, hyp_ann))
    except Exception:
        der = None
    try:
        jer = float(JaccardErrorRate(collar=0.25, skip_overlap=False)(ref_ann, hyp_ann))
    except Exception:
        jer = None
    return {"der": der, "jer": jer}


def speaker_turn_accuracy(ref: list[dict[str, Any]], hyp: list[dict[str, Any]], cfg: EvalConfig) -> dict[str, Any]:
    matches = match_by_overlap(ref, hyp, cfg.min_overlap)
    if not matches:
        return {"speaker_turn_accuracy": None, "speaker_confusion_rate": None}

    overlap_by_pair: dict[tuple[str, str], float] = {}
    for r, h, ov in matches:
        key = (str(h.get("speaker") or "UNKNOWN"), str(r.get("speaker") or "UNKNOWN"))
        overlap_by_pair[key] = overlap_by_pair.get(key, 0.0) + ov

    # Greedy one-to-one hyp->ref speaker mapping by overlap. Good enough for the
    # small number of speakers in AISHELL-4/AliMeeting and avoids scipy.
    mapping: dict[str, str] = {}
    used_ref: set[str] = set()
    for (hyp_spk, ref_spk), _ in sorted(overlap_by_pair.items(), key=lambda x: x[1], reverse=True):
        if hyp_spk not in mapping and ref_spk not in used_ref:
            mapping[hyp_spk] = ref_spk
            used_ref.add(ref_spk)

    correct = 0
    for r, h, _ in matches:
        hyp_spk = str(h.get("speaker") or "UNKNOWN")
        if mapping.get(hyp_spk, hyp_spk) == str(r.get("speaker") or "UNKNOWN"):
            correct += 1
    acc = correct / len(matches)
    return {"speaker_turn_accuracy": acc, "speaker_confusion_rate": 1.0 - acc}


def compute_text_metrics(ref: list[dict[str, Any]], hyp: list[dict[str, Any]], cfg: EvalConfig, mode: str) -> dict[str, Any]:
    ref_text = tokenize(concat_text(ref, cfg.language), cfg.language)
    hyp_text = tokenize(concat_text(hyp, cfg.language, mode=mode), cfg.language)
    text = edit_counts(ref_text, hyp_text)
    metric_name = "cer" if cfg.language == "zh" else "wer"
    return {
        metric_name: text["error_rate"],
        "substitutions": text["substitutions"],
        "insertions": text["insertions"],
        "deletions": text["deletions"],
        "ref_units": text["ref_units"],
        "hyp_units": text["hyp_units"],
    }


def evaluate_one(
    pred_json: Path,
    dataset: str,
    tg_dir: Path,
    cfg: EvalConfig,
    reference_json: Path | None = None,
    compare_llm: bool = False,
) -> dict[str, Any] | None:
    hyp, metadata = load_prediction(pred_json, cfg)
    effective_cfg = cfg
    if cfg.duration is None and hyp:
        effective_cfg = replace(cfg, duration=max(float(s["end"]) for s in hyp))

    text_only = dataset in TEXT_ONLY_DATASETS
    if reference_json:
        ref = load_reference_json(reference_json, effective_cfg)
        ref_label = str(reference_json)
        text_only = False
    elif text_only:
        ref = load_plain_transcript_reference(pred_json, dataset, tg_dir, hyp, effective_cfg)
        if ref is None:
            return None
        ref_label = str(tg_dir)
    else:
        tg_path = find_textgrid(pred_json, dataset, tg_dir)
        if tg_path is None:
            return None
        default_speaker = infer_speaker_from_filename(pred_json) if dataset == "alimeeting-near" else None
        ref = parse_textgrid(tg_path, effective_cfg, default_speaker=default_speaker)
        ref_label = str(tg_path)

    metric_name = "cer" if effective_cfg.language == "zh" else "wer"
    text_final = compute_text_metrics(ref, hyp, effective_cfg, mode="final")

    out: dict[str, Any] = {
        "file": pred_json.name,
        "reference": ref_label,
        "text": text_final,
        "alignment": {
            "matched_ref_segments": None,
            "match_rate": None,
            "start_mae": None,
            "end_mae": None,
            "hit_0_5": None,
            "hit_1_0": None,
            "coverage_rate": None,
            "silent_prefix_avg": None,
            "suffix_extra_avg": None,
        } if text_only else time_metrics(ref, hyp, effective_cfg),
        "confidence": confidence_metrics(hyp, metadata),
        "speaker": {"der": None, "jer": None, "speaker_turn_accuracy": None, "speaker_confusion_rate": None} if text_only else {},
        "reference_type": "plain_transcript" if text_only else "timed_segments",
    }
    if compare_llm:
        pre = compute_text_metrics(ref, hyp, effective_cfg, mode="pre_llm")
        out["text_pre_llm"] = pre
        out["llm_delta"] = {
            metric_name: text_final.get(metric_name, 0.0) - pre.get(metric_name, 0.0),
            "insertions": int(text_final.get("insertions", 0)) - int(pre.get("insertions", 0)),
            "deletions": int(text_final.get("deletions", 0)) - int(pre.get("deletions", 0)),
            "substitutions": int(text_final.get("substitutions", 0)) - int(pre.get("substitutions", 0)),
        }
    if not text_only:
        out["speaker"].update(diarization_metrics(ref, hyp, effective_cfg))
        out["speaker"].update(speaker_turn_accuracy(ref, hyp, effective_cfg))
    return out


def mean(values: list[float | None]) -> float | None:
    vals = [v for v in values if v is not None and not math.isnan(float(v))]
    return sum(vals) / len(vals) if vals else None


def aggregate(items: list[dict[str, Any]], language: str) -> dict[str, Any]:
    text_key = "cer" if language == "zh" else "wer"
    result = {
        "files": len(items),
        "text": {
            text_key: mean([i["text"].get(text_key) for i in items]),
            "insertions": sum(int(i["text"].get("insertions", 0)) for i in items),
            "deletions": sum(int(i["text"].get("deletions", 0)) for i in items),
            "substitutions": sum(int(i["text"].get("substitutions", 0)) for i in items),
        },
        "alignment": {
            "start_mae": mean([i["alignment"].get("start_mae") for i in items]),
            "end_mae": mean([i["alignment"].get("end_mae") for i in items]),
            "hit_0_5": mean([i["alignment"].get("hit_0_5") for i in items]),
            "hit_1_0": mean([i["alignment"].get("hit_1_0") for i in items]),
            "coverage_rate": mean([i["alignment"].get("coverage_rate") for i in items]),
            "silent_prefix_avg": mean([i["alignment"].get("silent_prefix_avg") for i in items]),
            "suffix_extra_avg": mean([i["alignment"].get("suffix_extra_avg") for i in items]),
        },
        "confidence": {
            "low_confidence_rate": mean([i["confidence"].get("low_confidence_rate") for i in items]),
            "fallback_rate": mean([i["confidence"].get("fallback_rate") for i in items]),
        },
        "speaker": {
            "der": mean([i["speaker"].get("der") for i in items]),
            "jer": mean([i["speaker"].get("jer") for i in items]),
            "speaker_turn_accuracy": mean([i["speaker"].get("speaker_turn_accuracy") for i in items]),
            "speaker_confusion_rate": mean([i["speaker"].get("speaker_confusion_rate") for i in items]),
        },
    }
    if any("text_pre_llm" in i for i in items):
        result["text_pre_llm"] = {
            text_key: mean([i.get("text_pre_llm", {}).get(text_key) for i in items]),
            "insertions": sum(int(i.get("text_pre_llm", {}).get("insertions", 0)) for i in items),
            "deletions": sum(int(i.get("text_pre_llm", {}).get("deletions", 0)) for i in items),
            "substitutions": sum(int(i.get("text_pre_llm", {}).get("substitutions", 0)) for i in items),
        }
        final_v = result["text"].get(text_key)
        pre_v = result["text_pre_llm"].get(text_key)
        result["llm_delta"] = {
            text_key: None if final_v is None or pre_v is None else final_v - pre_v,
            "insertions": result["text"]["insertions"] - result["text_pre_llm"]["insertions"],
            "deletions": result["text"]["deletions"] - result["text_pre_llm"]["deletions"],
            "substitutions": result["text"]["substitutions"] - result["text_pre_llm"]["substitutions"],
        }
    return result


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.2%}"


def sec(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}s"


def print_report(report: dict[str, Any], language: str) -> None:
    agg = report["aggregate"]
    text_key = "cer" if language == "zh" else "wer"

    console.rule("[bold green]Automatic Evaluation")
    t = Table(title="Three-layer Metrics")
    t.add_column("Layer")
    t.add_column("Metric")
    t.add_column("Value")
    t.add_row("Text", text_key.upper(), pct(agg["text"][text_key]))
    if "text_pre_llm" in agg:
        t.add_row("Text", f"{text_key.upper()} pre-LLM", pct(agg["text_pre_llm"][text_key]))
        t.add_row("Text", "LLM delta", pct(agg["llm_delta"][text_key]))
    t.add_row("Text", "Insert/Delete/Substitute", f'{agg["text"]["insertions"]}/{agg["text"]["deletions"]}/{agg["text"]["substitutions"]}')
    t.add_row("Timeline", "Start MAE", sec(agg["alignment"]["start_mae"]))
    t.add_row("Timeline", "End MAE", sec(agg["alignment"]["end_mae"]))
    t.add_row("Timeline", "Hit@0.5s / Hit@1.0s", f'{pct(agg["alignment"]["hit_0_5"])} / {pct(agg["alignment"]["hit_1_0"])}')
    t.add_row("Timeline", "Coverage", pct(agg["alignment"]["coverage_rate"]))
    t.add_row("Timeline", "Prefix/Suffix extra", f'{sec(agg["alignment"]["silent_prefix_avg"])} / {sec(agg["alignment"]["suffix_extra_avg"])}')
    t.add_row("Reliability", "Low-confidence rate", pct(agg["confidence"]["low_confidence_rate"]))
    t.add_row("Reliability", "Fallback rate", pct(agg["confidence"]["fallback_rate"]))
    t.add_row("Speaker", "DER / JER", f'{pct(agg["speaker"]["der"])} / {pct(agg["speaker"]["jer"])}')
    t.add_row("Speaker", "Turn accuracy", pct(agg["speaker"]["speaker_turn_accuracy"]))
    console.print(t)

    per = Table(title="Per-file Summary")
    per.add_column("File")
    per.add_column(text_key.upper())
    if "text_pre_llm" in agg:
        per.add_column(f"Pre-{text_key.upper()}")
    per.add_column("Start/End MAE")
    per.add_column("Hit@1s")
    per.add_column("LowConf")
    per.add_column("DER")
    for item in report["files"]:
        row = [
            item["file"][:36],
            pct(item["text"].get(text_key)),
        ]
        if "text_pre_llm" in agg:
            row.append(pct(item.get("text_pre_llm", {}).get(text_key)))
        row.extend([
            f'{sec(item["alignment"].get("start_mae"))}/{sec(item["alignment"].get("end_mae"))}',
            pct(item["alignment"].get("hit_1_0")),
            pct(item["confidence"].get("low_confidence_rate")),
            pct(item["speaker"].get("der")),
        ])
        per.add_row(*row)
    console.print(per)


def collect_predictions(args: argparse.Namespace) -> list[Path]:
    if args.pred:
        return [Path(p) for p in args.pred]
    pred_dir = Path(args.pred_dir)
    return sorted(pred_dir.glob("*.json"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved ASR JSON outputs with dataset references.")
    parser.add_argument("--dataset", choices=["aishell1", "librispeech", "aishell4", "alimeeting-near", "alimeeting-far"], required=True)
    parser.add_argument("--pred", nargs="*", help="Specific result JSON files to evaluate.")
    parser.add_argument("--pred-dir", default="tests/test_results", help="Directory containing saved result JSON files.")
    parser.add_argument("--textgrid-dir", default=None, help="Override reference TextGrid/transcript path.")
    parser.add_argument("--language", choices=["zh", "en"], default="zh")
    parser.add_argument("--duration", type=float, default=None, help="Evaluate only the first N seconds.")
    parser.add_argument("--output", default="outputs/eval_system.json")
    parser.add_argument("--compare-llm", action="store_true", help="Compare final text against pre-LLM text_before_llm when present.")
    parser.add_argument("--reference-json", default=None, help="Manual reference JSON with segments[start,end,speaker,text]. Only valid with one --pred file.")
    parser.add_argument("--export-reference-template", default=None, help="Create a manual reference template from one prediction JSON and exit.")
    args = parser.parse_args()

    root = project_root()
    default_tg_dir, _ = dataset_paths(args.dataset, root)
    tg_dir = Path(args.textgrid_dir) if args.textgrid_dir else default_tg_dir
    cfg = EvalConfig(language=args.language, duration=args.duration)

    pred_files = collect_predictions(args)
    if not pred_files:
        console.print("[red]No prediction JSON files found.[/red]")
        return 1

    if args.export_reference_template:
        if len(pred_files) != 1:
            console.print("[red]--export-reference-template requires exactly one --pred file.[/red]")
            return 1
        export_reference_template(pred_files[0], Path(args.export_reference_template))
        console.print(f"[green]Saved reference template:[/green] {args.export_reference_template}")
        return 0

    ref_json = Path(args.reference_json) if args.reference_json else None
    if ref_json and len(pred_files) != 1:
        console.print("[red]--reference-json currently supports exactly one --pred file.[/red]")
        return 1

    results: list[dict[str, Any]] = []
    skipped: list[str] = []
    for pred in pred_files:
        item = evaluate_one(pred, args.dataset, tg_dir, cfg, reference_json=ref_json, compare_llm=args.compare_llm)
        if item is None:
            skipped.append(pred.name)
            continue
        results.append(item)

    if not results:
        console.print("[red]No files matched available TextGrid references.[/red]")
        if skipped:
            console.print("Skipped:", ", ".join(skipped[:10]))
        return 1

    report = {
        "config": {
            "dataset": args.dataset,
            "language": args.language,
            "duration": args.duration,
            "textgrid_dir": str(tg_dir),
            "compare_llm": args.compare_llm,
            "reference_json": str(ref_json) if ref_json else None,
        },
        "aggregate": aggregate(results, args.language),
        "files": results,
        "skipped": skipped,
    }
    print_report(report, args.language)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[green]Saved report:[/green] {out}")
    if skipped:
        console.print(f"[yellow]Skipped {len(skipped)} file(s) without matching TextGrid.[/yellow]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
