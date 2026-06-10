"""多说话人语音转写与智能检索系统——Gradio 统一界面。

模式：📁文件转写 | 🎤实时麦克风 | 🔍智能检索
"""
from __future__ import annotations

import argparse, base64, io, json, re, shutil
import sys, time, wave
from html import escape
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr
import numpy as np
import soundfile as sf

from src.pipeline import run
from src.utils.config import project_root
from gui.realtime_controller import RealtimeController
from gui.static_js import MIC_JS, PLAY_JS


# ══════════════════════════════════════════════════════════════
# 工具
# ══════════════════════════════════════════════════════════════

def fmt_time(seconds: float) -> str:
    m = int(seconds // 60); s = seconds % 60
    return f"{m:02d}:{s:04.1f}"


def audio_to_base64(audio: np.ndarray, sr: int) -> str:
    audio_i16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(audio_i16.tobytes())
    return base64.b64encode(buf.getvalue()).decode()


def _audio_src_for_path(path: str | Path) -> str:
    """Return a lightweight Gradio-served URL for an audio file path."""
    p = Path(path)
    return "/file=" + quote(str(p.resolve()).replace("\\", "/"), safe="/:")


# ══════════════════════════════════════════════════════════════
# 全局状态
# ══════════════════════════════════════════════════════════════

_last_result: dict | None = None
_last_file_results: list[dict] = []
_last_audio_files: list[tuple[str, str]] = []  # (label, filepath)
_last_segments_raw: list[dict] = []   # 未映射的原始段
_index_path: Path = project_root() / "outputs" / "index"
_output_dir: Path = project_root() / "tests" / "test_results"
_search_result_dirs: list[Path] = [
    project_root() / "outputs" / "batch_eval_llm" / "results",
    _output_dir,
]
_retrieval_encoder = None
_retrieval_query_adapter = None
_search_dir_signatures: dict[str, tuple[int, float]] = {}
_current_audio_b64: str = ""
_current_audio_sr: int = 16000

_rt_controller: RealtimeController | None = None
_AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


# ══════════════════════════════════════════════════════════════
# 文件转写
# ══════════════════════════════════════════════════════════════

def _project_path(path_str: str | Path) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else project_root() / p


def _canonical_path_key(path_value: object) -> str:
    path = str(path_value or "").strip()
    if not path:
        return ""
    try:
        return str(Path(path).expanduser().resolve()).lower()
    except Exception:
        return str(Path(path).expanduser()).replace("\\", "/").lower()


def _result_index_for_file(path: object) -> int | None:
    target = _canonical_path_key(path)
    if not target:
        return None
    for idx, existing in enumerate(_last_file_results):
        if _canonical_path_key((existing or {}).get("file", "")) == target:
            return idx
    return None


def _normalise_result_item(result: dict, fallback_name: str = "") -> dict:
    item = dict(result)
    file_path = str(item.get("file", "") or "")
    item.setdefault("name", Path(file_path).name or fallback_name or "audio")
    item.setdefault("segments", [])
    if not isinstance(item.get("segments"), list):
        item["segments"] = []
    return item


def _merge_result_into_state(
    result: dict,
    out_dir: Path,
    *,
    replace_existing: bool,
) -> int:
    """Merge one file result into UI memory and keep audio choices index-aligned."""
    global _last_result
    item = _normalise_result_item(result)
    existing_idx = _result_index_for_file(item.get("file", ""))
    if existing_idx is None:
        idx = len(_last_file_results)
        _last_file_results.append(item)
    else:
        idx = existing_idx
        if replace_existing:
            _last_file_results[idx] = item
        else:
            item = _last_file_results[idx]

    _last_result = _last_file_results[-1] if _last_file_results else item

    file_path = item.get("file", "")
    while len(_last_audio_files) <= idx:
        _last_audio_files.append(("", ""))
    copied = _copy_audio_for_player(file_path, out_dir, idx) if file_path else None
    if copied:
        _last_audio_files[idx] = copied
    return idx


def _hydrate_missing_results_from_output_dir(out_dir: Path, file_paths: list[str] | None = None) -> int:
    """Recover file results saved on disk but missing from current UI memory."""
    if not out_dir.exists():
        return 0
    allowed_keys = {_canonical_path_key(path) for path in (file_paths or []) if path}
    allowed_names = {Path(path).name.lower() for path in (file_paths or []) if path}
    recovered = 0
    for json_path in sorted(out_dir.glob("*.json"), key=lambda p: p.stat().st_mtime):
        if json_path.name.startswith("manual_corrected"):
            continue
        if allowed_names and json_path.name[:-5].lower() not in allowed_names:
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        raw_results = data.get("files", []) if isinstance(data, dict) and "files" in data else [data]
        for raw in raw_results:
            if not isinstance(raw, dict) or not raw.get("segments"):
                continue
            file_path = raw.get("file", "")
            if allowed_keys and _canonical_path_key(file_path) not in allowed_keys:
                continue
            if file_path and _result_index_for_file(file_path) is None:
                _merge_result_into_state(raw, out_dir, replace_existing=False)
                recovered += 1
    return recovered


def _load_search_results_from_dir(out_dir: Path) -> int:
    """Load saved JSON results for retrieval without requiring manual history load."""
    if not out_dir.exists():
        return 0
    loaded = 0
    json_files = sorted(out_dir.rglob("*.json"), key=lambda p: str(p.relative_to(out_dir)).lower())
    for json_path in json_files:
        if json_path.name.startswith("manual_corrected"):
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        raw_results = data.get("files", []) if isinstance(data, dict) and "files" in data else [data]
        for raw in raw_results:
            if not isinstance(raw, dict) or not raw.get("segments"):
                continue
            _merge_result_into_state(raw, out_dir, replace_existing=True)
            loaded += 1
    return loaded


def _load_search_results_from_default_dirs() -> tuple[int, list[str]]:
    loaded = 0
    sources: list[str] = []
    seen: set[str] = set()
    for out_dir in _search_result_dirs:
        key = _canonical_path_key(out_dir)
        if not key or key in seen:
            continue
        seen.add(key)
        signature = _dir_json_signature(out_dir)
        if _search_dir_signatures.get(key) == signature:
            if out_dir.exists():
                try:
                    sources.append(str(out_dir.relative_to(project_root())))
                except Exception:
                    sources.append(str(out_dir))
            continue
        before = len(_last_file_results)
        n = _load_search_results_from_dir(out_dir)
        _search_dir_signatures[key] = signature
        loaded += n
        if n or len(_last_file_results) > before or out_dir.exists():
            try:
                sources.append(str(out_dir.relative_to(project_root())))
            except Exception:
                sources.append(str(out_dir))
    return loaded, sources


def _dir_json_signature(out_dir: Path) -> tuple[int, float]:
    if not out_dir.exists():
        return (0, 0.0)
    count = 0
    latest = 0.0
    for json_path in out_dir.rglob("*.json"):
        if json_path.name.startswith("manual_corrected"):
            continue
        try:
            stat = json_path.stat()
        except OSError:
            continue
        count += 1
        latest = max(latest, stat.st_mtime)
    return (count, latest)


def _speaker_label(speaker: object) -> str:
    return str(speaker or "?").strip()


def _render_seg_html(
    seg: dict,
    show_intent: bool = True,
    audio_id: str | None = None,
    onclick: str | None = None,
    file_index: int | None = None,
) -> str:
    """渲染单个段为 HTML 可点击行。"""
    ts = f"[{fmt_time(seg['start'])} - {fmt_time(seg['end'])}]"
    spk = escape(_speaker_label(seg.get("speaker", "?")))
    txt = seg.get("llm_text", "") or seg.get("text", "")
    txt = escape(txt)
    orig_source = seg.get("text_before_llm") or seg.get("text_original_asr") or seg.get("text", "")
    orig = escape(orig_source)
    phase = seg.get("phase", "")
    intent = seg.get("llm_intent") or seg.get("intent", "")
    if isinstance(intent, list): intent = "+".join(intent)

    if txt != orig:
        prefix = "✅"
        note = (f" <span style='color:#999;font-size:0.8em'>"
                f"(原: {orig[:40]}{'…' if len(orig)>40 else ''})</span>")
    elif phase == "corrected":
        prefix = "✅"; note = ""
    else:
        prefix = "⚡"; note = ""

    intent_html = f" <span style='color:#e67e22;font-size:0.85em'>[{intent}]</span>" if show_intent and intent else ""
    confidence_html = ""
    if seg.get("low_confidence"):
        reasons = seg.get("low_confidence_reasons") or []
        reason_text = escape("; ".join(str(r) for r in reasons) or "需人工复核")
        confidence_html = (
            f" <span title='{reason_text}' "
            f"style='color:#b9770e;font-size:0.85em'>[低置信度]</span>"
        )
    source_name = str(seg.get("_file_name") or seg.get("name") or "").strip()
    source_html = ""
    if source_name:
        source_html = (
            f"<span title='音频来源' "
            f"style='color:#5d6d7e;background:#eef3f8;border:1px solid #d7e1ec;"
            f"border-radius:4px;padding:1px 5px;margin-right:4px;font-size:0.82em'>"
            f"{escape(source_name)}</span> "
        )
    play_start = float(seg.get("playback_start", seg["start"]))
    play_end = float(seg.get("playback_end", seg["end"]))
    audio_arg = f",'{audio_id}',this" if audio_id else ",null,this"
    click_js = onclick or f"playSeg({play_start},{play_end}{audio_arg})"
    play_data_attr = ""
    if not onclick:
        play_data_attr = (
            f"data-play-start='{play_start}' data-play-end='{play_end}' "
            f"data-audio-id='{escape(audio_id or '')}'"
        )
    file_attr = f"data-file-index='{file_index}'" if file_index is not None else ""
    b_file_attr = str(file_index) if file_index is not None else ""
    onclick_attr = f"onclick=\"{click_js}\"" if click_js else ""
    return (
        f"<div class='seg-line' {onclick_attr} "
        f"{file_attr} "
        f"{play_data_attr} "
        f"title='点击播放 [{ts}]' "
        f"style='cursor:pointer;padding:3px 6px;margin:1px 0;border-radius:4px;"
        f"transition:background 0.15s' "
        f"onmouseover='this.style.background=\"#fdebd0\"' "
        f"onmouseout='this.style.background=\"transparent\"'>"
        f"<span style='color:#e67e22'>▸</span> {prefix} "
        f"{source_html}"
        f"<span style='color:#888;font-family:monospace;font-size:0.9em'>{ts}</span> "
        f"<b data-speaker='{spk}' data-file-index='{b_file_attr}' "
        f"style='color:#c0392b'>{spk}</b>: {txt}{note}{intent_html}{confidence_html}</div>"
    )


def _set_refined_realtime_result(result: dict, rec_path: Path) -> None:
    global _last_result, _last_file_results, _last_audio_files, _last_segments_raw
    _last_result = result
    _last_file_results = [result]
    _last_segments_raw = result.get("segments", [])
    _last_audio_files = [(f"1. {rec_path.name}", str(rec_path))]


def _get_rt_controller() -> RealtimeController:
    global _rt_controller
    if _rt_controller is None:
        _rt_controller = RealtimeController(
            output_dir=_output_dir,
            render_seg_html=_render_seg_html,
            audio_to_base64=audio_to_base64,
            run_pipeline=run,
            on_refined=_set_refined_realtime_result,
        )
    return _rt_controller


def process_files(
    file_paths,
    language,
    enable_llm,
    max_duration,
    output_dir_str,
    llm_model,
    progress=gr.Progress(),
):
    global _last_result, _last_file_results, _last_audio_files
    global _last_segments_raw, _current_audio_b64, _current_audio_sr

    # 收集文件
    files: list[str] = []
    if file_paths:
        for item in file_paths:
            f = item.get("path", "") if isinstance(item, dict) else item
            if f and Path(f).suffix.lower() in _AUDIO_SUFFIXES:
                files.append(f)

    if not files:
        return (
            "<p style='color:#888'>请先手动添加音频文件</p>",
            "", "", "", "", "",
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
            gr.update(choices=[], value=None),
        )

    out_dir = _project_path(output_dir_str.strip()) if output_dir_str.strip() else _output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    recovered_count = _hydrate_missing_results_from_output_dir(out_dir, files)

    all_stats: list[str] = []
    saved_files: list[str] = []
    primary_audio_b64 = ""; primary_audio_sr = 16000

    for fi, fpath in enumerate(files):
        fname = Path(fpath).name
        progress(fi / len(files), desc=f"[{fi+1}/{len(files)}] {fname}")

        try:
            audio_data, audio_sr = sf.read(fpath)
            if audio_data.ndim > 1: audio_data = audio_data.mean(axis=1)
            audio_data = audio_data.astype(np.float32)
            if max_duration > 0 and len(audio_data) > max_duration * audio_sr:
                audio_data = audio_data[:int(max_duration * audio_sr)]
            if fi == 0: primary_audio_b64 = audio_to_base64(audio_data, audio_sr); primary_audio_sr = audio_sr
        except Exception:
            audio_data, audio_sr = np.zeros(100, dtype=np.float32), 16000

        llm_ov = {"enabled": enable_llm} if enable_llm else {
            "enabled": False, "correction": False, "consistency": False, "intent_tagging": False}

        try:
            result = run(str(fpath), language=language,
                         max_duration_s=max_duration if max_duration > 0 else None,
                         llm_overrides=llm_ov, llm_model=llm_model if llm_model else None)
        except Exception as e:
            all_stats.append(f"{fname}: 失败 - {e}")
            continue

        result["file"] = fpath
        result["name"] = fname
        segs = result["segments"]
        timing = result.get("timing", {})
        n_spk = len({s.get("speaker", "?") for s in segs})
        _merge_result_into_state(result, out_dir, replace_existing=True)

        out_path = out_dir / f"{fname}.json"
        meeting_summary = result.get("meeting_summary", {})
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"file": fpath, "segments": segs, "timing": timing,
                        "num_speakers": result.get("num_speakers"),
                        "meeting_summary": meeting_summary},
                       f, ensure_ascii=False, indent=2)
        saved_files.append(str(out_path))
        all_stats.append(f"{fname}: {n_spk}人 · {len(segs)}段 · {timing.get('total',0):.1f}s")

    progress(1.0, desc="完成")
    _current_audio_b64 = primary_audio_b64; _current_audio_sr = primary_audio_sr
    _last_segments_raw = [
        seg
        for result in _last_file_results
        for seg in result.get("segments", [])
    ]

    full_html = _render_current_result_html()
    # 添加刷新警告
    full_html = ("<p style='color:#e67e22;font-size:0.85em'>"
                 "⚠ 新转写会追加到当前页面；同一音频重新转写会替换旧结果。结果已自动保存到输出目录</p>") + full_html

    valid_audio_files = [(label, path) for label, path in _last_audio_files if label and path]
    audio_choices = [label for label, _ in valid_audio_files]
    manual_file_choices = _file_choices()
    manual_file_value = manual_file_choices[-1] if manual_file_choices else None
    edit_text = _segments_to_edit_text(_segments_for_file_choice(manual_file_value))
    speaker_choices = _speaker_choices_for_file(_parse_file_choice(manual_file_value))
    audio_file_path = ""
    if valid_audio_files:
        selected_file_idx = _parse_file_choice(manual_file_value)
        if selected_file_idx is not None and selected_file_idx < len(_last_audio_files):
            audio_file_path = _last_audio_files[selected_file_idx][1]
        if not audio_file_path:
            audio_file_path = valid_audio_files[-1][1]
    return (
        full_html,
        ("\n".join(all_stats) if all_stats else f"当前累计 {_last_file_results.__len__()} 个结果")
        + (f"\n已从本地恢复 {recovered_count} 条历史结果" if recovered_count else ""),
        f"输出: {out_dir}" + (f"\n保存: " + "\n".join(saved_files) if saved_files else ""),
        audio_file_path,
        str(primary_audio_sr),
        edit_text,
        gr.update(choices=audio_choices, value=audio_choices[-1] if audio_choices else None),
        gr.update(choices=speaker_choices, value=speaker_choices[0] if speaker_choices else None),
        gr.update(choices=manual_file_choices, value=manual_file_value),
    )


def select_audio_for_player(label):
    """Return the copied audio path selected for the bottom player."""
    if not label:
        return _last_audio_files[0][1] if _last_audio_files else ""
    for item_label, item_path in _last_audio_files:
        if item_label == label:
            return item_path
    return _last_audio_files[0][1] if _last_audio_files else ""


def _copy_audio_for_player(src_path: str, out_dir: Path, index: int = 0) -> tuple[str, str] | None:
    src = Path(src_path)
    if not src.exists():
        return None
    dst = out_dir / f"_current_audio_{index}_{src.stem}.wav"
    try:
        needs_write = True
        if dst.exists():
            try:
                needs_write = dst.stat().st_mtime < src.stat().st_mtime or dst.stat().st_size == 0
            except OSError:
                needs_write = True
        if needs_write:
            ad, asr = sf.read(str(src), dtype="float32", always_2d=False)
            if ad.ndim > 1:
                ad = ad.mean(axis=1)
            sf.write(str(dst), ad.astype(np.float32), asr, subtype="PCM_16")
        return (f"{index + 1}. {src.name}", str(dst))
    except Exception:
        fallback = out_dir / f"_current_audio_{index}_{src.name}"
        try:
            if not fallback.exists() or fallback.stat().st_size != src.stat().st_size:
                shutil.copy2(src, fallback)
            return (f"{index + 1}. {src.name}", str(fallback))
        except Exception:
            return None


def refresh_history(output_dir_str):
    out_dir = _project_path((output_dir_str or "").strip()) if (output_dir_str or "").strip() else _output_dir
    if not out_dir.exists():
        return gr.update(choices=[], value=None), f"历史目录不存在: {out_dir}"
    files = sorted(
        [p for p in out_dir.glob("*.json") if not p.name.startswith("manual_corrected")],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    choices = [str(p) for p in files]
    return gr.update(choices=choices, value=choices[0] if choices else None), f"找到 {len(files)} 条历史记录"


def load_history_result(history_json_path, output_dir_str):
    global _last_result, _last_file_results, _last_audio_files, _last_segments_raw
    if not history_json_path:
        return "<p style='color:#888'>请选择历史记录</p>", "", "", "", gr.update(choices=[], value=None)

    path = Path(history_json_path)
    if not path.exists():
        return f"<p style='color:red'>历史记录不存在: {escape(str(path))}</p>", "", "", "", gr.update(choices=[], value=None)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "files" in data:
        results = data.get("files", [])
    else:
        results = [data]

    _last_file_results = []
    _last_audio_files = []
    _last_segments_raw = []
    out_dir = _project_path((output_dir_str or "").strip()) if (output_dir_str or "").strip() else path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for result in results:
        if isinstance(result, dict):
            _merge_result_into_state(result, out_dir, replace_existing=True)

    _last_segments_raw = [
        seg
        for result in _last_file_results
        for seg in result.get("segments", [])
    ]

    _last_result = _last_file_results[-1] if _last_file_results else None
    html = _render_current_result_html()
    audio_choices = [label for label, _ in _last_audio_files]
    audio_value = _last_audio_files[0][1] if _last_audio_files else ""
    stats = "\n".join(
        f"{r.get('name', 'history')}: {len(r.get('segments', []))}段"
        for r in _last_file_results
    )
    return (
        html,
        stats,
        f"已加载历史: {path}",
        audio_value,
        gr.update(choices=audio_choices, value=audio_choices[0] if audio_choices else None),
    )


def _all_segments_with_file_index() -> list[dict]:
    all_segments: list[dict] = []
    for fi, result in enumerate(_last_file_results):
        for si, seg in enumerate(result.get("segments", [])):
            item = dict(seg)
            item["_file_index"] = fi
            item["_segment_index"] = si
            item["_file_name"] = result.get("name", f"file{fi}")
            all_segments.append(item)
    if not all_segments and _last_result:
        for si, seg in enumerate(_last_result.get("segments", [])):
            item = dict(seg)
            item["_file_index"] = 0
            item["_segment_index"] = si
            item["_file_name"] = _last_result.get("name", "latest")
            all_segments.append(item)
    return all_segments


def _segments_to_edit_text(segments: list[dict]) -> str:
    lines: list[str] = []
    for i, seg in enumerate(segments):
        ts = f"{fmt_time(float(seg.get('start', 0)))}-{fmt_time(float(seg.get('end', 0)))}"
        spk = seg.get("speaker", "?")
        fid = seg.get("_file_index", 0)
        sid = seg.get("_segment_index", i)
        fname = seg.get("_file_name", "")
        text = seg.get("text", "")
        lines.append(f"[{fid}:{sid}] {fname} {ts} {spk}: {text}")
    return "\n".join(lines)


def _speaker_choices() -> list[str]:
    choices: list[str] = []
    seen: set[str] = set()
    for seg in _all_segments_with_file_index():
        spk = str(seg.get("speaker", "")).strip()
        if spk and spk not in seen:
            seen.add(spk)
            choices.append(spk)
    return choices


def _speaker_choices_for_file(file_idx: int | None = None) -> list[str]:
    choices: list[str] = []
    seen: set[str] = set()
    results = _last_file_results or ([_last_result] if _last_result else [])
    for fi, result in enumerate(results):
        if file_idx is not None and fi != file_idx:
            continue
        for seg in (result or {}).get("segments", []):
            spk = str(seg.get("speaker", "")).strip()
            if spk and spk not in seen:
                seen.add(spk)
                choices.append(spk)
    return choices


def _file_choices() -> list[str]:
    results = _last_file_results or ([_last_result] if _last_result else [])
    return [f"{fi}. {result.get('name', f'file{fi}')}" for fi, result in enumerate(results) if result]


def _parse_file_choice(label: str | None) -> int | None:
    if not label:
        return None
    m = re.match(r"^\s*(\d+)\.", str(label))
    return int(m.group(1)) if m else None


def _segments_for_file_choice(label: str | None) -> list[dict]:
    file_idx = _parse_file_choice(label)
    if file_idx is None:
        return _all_segments_with_file_index()
    return [seg for seg in _all_segments_with_file_index() if seg.get("_file_index") == file_idx]


def select_manual_file(label):
    file_idx = _parse_file_choice(label)
    choices = _speaker_choices_for_file(file_idx)
    return (
        _segments_to_edit_text(_segments_for_file_choice(label)),
        gr.update(choices=choices, value=choices[0] if choices else None),
    )


def _render_summary_panel(summary: dict | None) -> str:
    """Render optional LLM content summary saved by the pipeline."""
    if not isinstance(summary, dict):
        return ""

    topic = str(summary.get("topic") or "").strip()
    brief = str(summary.get("summary") or "").strip()
    key_points = summary.get("key_points") or []
    decisions = summary.get("decisions") or []
    action_items = summary.get("action_items") or []
    participants = summary.get("participants") or []

    if not any([topic, brief, key_points, decisions, action_items, participants]):
        return ""

    def _items(values) -> str:
        if isinstance(values, str):
            values = [values] if values.strip() else []
        if not isinstance(values, (list, tuple)):
            return ""
        lis = "".join(f"<li>{escape(str(v).strip())}</li>" for v in values if str(v).strip())
        return f"<ul style='margin:4px 0 0 20px;padding:0'>{lis}</ul>" if lis else ""

    rows: list[str] = []
    if topic:
        rows.append(f"<div><b>主题：</b>{escape(topic)}</div>")
    if brief:
        rows.append(f"<div><b>概要：</b>{escape(brief)}</div>")
    for title, values in (
        ("要点", key_points),
        ("结论", decisions),
        ("待办", action_items),
        ("参与者", participants),
    ):
        rendered = _items(values)
        if rendered:
            rows.append(f"<div><b>{title}：</b>{rendered}</div>")

    return (
        "<div class='content-summary' "
        "style='margin:10px 0;padding:10px;border-left:3px solid #f6a23a;background:#fff8ef;"
        "line-height:1.65;color:#263238'>"
        "<div style='font-weight:bold;color:#e67e22;margin-bottom:4px'>内容摘要</div>"
        + "".join(rows)
        + "</div>"
    )


def _render_current_result_html(player_audio_id: str = "main-audio-player") -> str:
    if not _last_file_results and not _last_result:
        return "<p style='color:#888'>请先处理音频</p>"
    parts: list[str] = []
    for fi, result in enumerate(_last_file_results or [_last_result]):
        if not result:
            continue
        segs = result.get("segments", [])
        audio_id = f"file-audio-{fi}"
        file_audio_html = ""
        if fi < len(_last_audio_files) and _last_audio_files[fi][1]:
            audio_path = Path(_last_audio_files[fi][1])
            if audio_path.exists():
                try:
                    audio_data, audio_sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
                    if audio_data.ndim > 1:
                        audio_data = audio_data.mean(axis=1)
                    audio_src = "data:audio/wav;base64," + audio_to_base64(
                        audio_data.astype(np.float32), audio_sr
                    )
                    file_audio_html = (
                        f"<audio id='{audio_id}' src='{audio_src}' "
                        f"preload='metadata' playsinline style='display:none'></audio>"
                    )
                except Exception:
                    audio_id = player_audio_id
            else:
                audio_id = player_audio_id
        else:
            audio_id = player_audio_id
        summary_html = _render_summary_panel(result.get("meeting_summary"))
        rename_panel = _build_rename_panel(segs, fi)
        seg_html = "\n".join(
            _render_seg_html(s, audio_id=audio_id, file_index=fi)
            for s in segs
        )
        name = escape(result.get("name", f"file{fi}"))
        parts.append(
            f"<details open class='transcript-file' data-file-index='{fi}' "
            f"style='margin:8px 0;border:1px solid #f0d0a0;border-radius:6px;padding:8px'>"
            f"<summary style='cursor:pointer;font-weight:bold;color:#e67e22'>{name}</summary>"
            f"{file_audio_html}{summary_html}{rename_panel}{seg_html}</details>"
        )
    return "\n".join(parts)


def apply_manual_edits(edit_text, output_dir_str, manual_file_label):
    """Apply manual transcript text edits to the latest file transcription."""
    global _last_result, _last_file_results
    if not _last_file_results and not _last_result:
        return (
            "<p style='color:#888'>请先处理音频</p>",
            edit_text or "",
            "无可修改结果",
            gr.update(choices=[], value=None),
        )

    if not _last_file_results and _last_result:
        _last_file_results = [_last_result]

    changed = 0
    line_re = re.compile(
        r"^\[(?:(\d+):)?(\d+)\]\s+.+?\s+\d{2,}:\d{2}\.\d-\d{2,}:\d{2}\.\d\s+(\S+):\s*(.*)$"
    )
    for line in (edit_text or "").splitlines():
        m = line_re.match(line.strip())
        if not m:
            continue
        file_idx = int(m.group(1) or 0)
        seg_idx = int(m.group(2))
        new_speaker = m.group(3).strip()
        new_text = m.group(4).strip()
        if 0 <= file_idx < len(_last_file_results):
            segments = _last_file_results[file_idx].get("segments", [])
        else:
            continue
        if 0 <= seg_idx < len(segments) and new_text:
            if segments[seg_idx].get("speaker", "") != new_speaker:
                segments[seg_idx]["speaker_original_manual"] = segments[seg_idx].get("speaker", "")
                segments[seg_idx]["speaker"] = new_speaker
                segments[seg_idx]["manual_speaker_edited"] = True
                changed += 1
            if segments[seg_idx].get("text", "") != new_text:
                segments[seg_idx]["text_original_asr"] = segments[seg_idx].get("text", "")
                segments[seg_idx]["text"] = new_text
                segments[seg_idx]["manual_edited"] = True
                changed += 1

    _last_result = _last_file_results[-1] if _last_file_results else _last_result

    out_dir = _project_path(output_dir_str.strip()) if output_dir_str.strip() else _output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "manual_corrected_latest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"files": _last_file_results}, f, ensure_ascii=False, indent=2)

    rendered = _render_current_result_html()
    refreshed_edit_text = _segments_to_edit_text(_segments_for_file_choice(manual_file_label))
    file_idx = _parse_file_choice(manual_file_label)
    speaker_choices = _speaker_choices_for_file(file_idx)
    status = f"已应用 {changed} 处人工修改，保存: {out_path}"
    return (
        rendered,
        refreshed_edit_text,
        status,
        gr.update(choices=speaker_choices, value=speaker_choices[0] if speaker_choices else None),
    )


def apply_speaker_mapping(source_speaker, target_speaker, output_dir_str, manual_file_label):
    """Batch replace one speaker label in the selected file."""
    global _last_result, _last_file_results
    source = (source_speaker or "").strip()
    target = (target_speaker or "").strip()
    file_idx = _parse_file_choice(manual_file_label)
    if not source or not target:
        choices = _speaker_choices_for_file(file_idx)
        return (
            _render_current_result_html(),
            _segments_to_edit_text(_segments_for_file_choice(manual_file_label)),
            "请选择原说话人并输入目标名称",
            gr.update(choices=choices, value=source or (choices[0] if choices else None)),
        )
    if not _last_file_results and _last_result:
        _last_file_results = [_last_result]

    changed = 0
    for fi, result in enumerate(_last_file_results):
        if file_idx is not None and fi != file_idx:
            continue
        for seg in result.get("segments", []):
            if str(seg.get("speaker", "")).strip() == source:
                seg["speaker_original_manual"] = seg.get("speaker", "")
                seg["speaker"] = target
                seg["manual_speaker_edited"] = True
                changed += 1

    _last_result = _last_file_results[-1] if _last_file_results else _last_result
    out_dir = _project_path(output_dir_str.strip()) if output_dir_str.strip() else _output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "manual_corrected_latest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"files": _last_file_results}, f, ensure_ascii=False, indent=2)

    choices = _speaker_choices_for_file(file_idx)
    next_value = target if target in choices else (choices[0] if choices else None)
    scope = manual_file_label or "全部文件"
    return (
        _render_current_result_html(),
        _segments_to_edit_text(_segments_for_file_choice(manual_file_label)),
        f"已在 {scope} 将 {source} 批量修改为 {target}，影响 {changed} 段，保存: {out_path}",
        gr.update(choices=choices, value=next_value),
    )


def _build_rename_panel(segments: list[dict], file_index: int | None = None) -> str:
    """生成说话人重命名 HTML 面板。"""
    spks = list(dict.fromkeys(s.get("speaker", "?") for s in segments))
    if not spks: return ""
    rows = []
    for spk in spks:
        safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", str(spk))
        spk_html = escape(str(spk))
        spk_attr = escape(str(spk), quote=True)
        file_attr = "" if file_index is None else f" data-file-index='{file_index}'"
        id_prefix = f"rename-{file_index}-" if file_index is not None else "rename-"
        rows.append(
            f"<div style='display:flex;align-items:center;gap:8px;margin:3px 0'>"
            f"<span style='font-weight:bold;min-width:80px;color:#c0392b'>{spk_html}</span>"
            f"→ <input class='speaker-rename' data-speaker='{spk_attr}'{file_attr} "
            f"id='{id_prefix}{safe_id}' placeholder='仅临时显示姓名...' "
            f"style='padding:2px 6px;border:1px solid #e67e22;border-radius:4px;width:120px' "
            f"oninput='applyRename()'>"
            f"</div>"
        )
    return (
        f"<details style='margin-top:12px;padding:8px;border:1px dashed #e67e22;border-radius:6px'>"
        f"<summary style='cursor:pointer;font-weight:bold;color:#e67e22'>🔤 说话人重命名（展开后临时显示）</summary>"
        f"<div style='margin-top:6px'>"
        + "".join(rows) +
        f"</div></details>"
    )


# ══════════════════════════════════════════════════════════════
# 检索
# ══════════════════════════════════════════════════════════════

def _ensure_bge_index_for_segments(segs: list[dict]) -> tuple[bool, str]:
    """Ensure outputs/index matches current searchable segments."""
    if not segs:
        return False, "无可索引片段"
    searchable = _searchable_segments_for_index(segs)
    if not searchable:
        return False, "无足够长的可索引片段"
    if (_index_path / "dense.faiss").exists():
        try:
            from src.retrieval.indexer import load_index
            meta, _, _ = load_index(_index_path)
            if _segments_match_current(meta.get("segments", []), searchable):
                return True, "已使用现有 BGE-M3 索引"
        except Exception:
            pass

    try:
        from src.retrieval import EmbeddingEncoder
        from src.retrieval.indexer import build_index
        from src.utils.config import get_model_config

        cfg = get_model_config()
        enc = EmbeddingEncoder(
            model_path=cfg["embedding"]["model"],
            use_fp16=cfg["embedding"].get("use_fp16", True),
            device=cfg.get("device", "cuda"),
        )
        enc.load()
        try:
            meta, _, _ = build_index(segs, enc, store_path=_index_path)
        finally:
            enc.unload()
        return True, f"已重建 BGE-M3 索引：{meta.get('num_segments', len(searchable))} 个片段"
    except Exception as exc:
        return False, f"BGE-M3 索引构建失败，已降级关键词检索：{exc}"


def _get_retrieval_encoder():
    global _retrieval_encoder
    if _retrieval_encoder is not None:
        return _retrieval_encoder
    from src.retrieval import EmbeddingEncoder
    from src.utils.config import get_model_config

    cfg = get_model_config()
    enc = EmbeddingEncoder(
        model_path=cfg["embedding"]["model"],
        use_fp16=cfg["embedding"].get("use_fp16", True),
        device=cfg.get("device", "cuda"),
    )
    enc.load()
    _retrieval_encoder = enc
    return enc


def _get_query_rewriter_adapter():
    global _retrieval_query_adapter
    if _retrieval_query_adapter is not None:
        return _retrieval_query_adapter
    try:
        from src.llm import load_llm_adapter
        _retrieval_query_adapter = load_llm_adapter()
    except Exception:
        _retrieval_query_adapter = False
    return None if _retrieval_query_adapter is False else _retrieval_query_adapter


def search_transcript(query, top_k):
    t0 = time.time()
    _, sources = _load_search_results_from_default_dirs()
    segs = _all_segments_with_file_index()
    if not segs: return "<p style='color:#888'>无数据</p>"
    indexed_ok, index_status = _ensure_bge_index_for_segments(segs)
    if indexed_ok:
        try:
            from src.retrieval import Retriever
            from src.retrieval.indexer import load_index
            meta, _, _ = load_index(_index_path)
            indexed = meta.get("segments", [])
            if _segments_match_current(indexed, _searchable_segments_for_index(segs)):
                enc = _get_retrieval_encoder()
                query_adapter = _get_query_rewriter_adapter()
                ret = Retriever(index_path=_index_path, encoder=enc, query_rewriter=query_adapter)
                candidate_k = max(int(top_k) * 4, 10)
                results = _select_display_search_results(ret.search(query, top_k=candidate_k), query, int(top_k))
                results = _attach_file_indices_to_search_results(results, segs)
            else:
                results = _simple_search(segs, query, top_k)
        except Exception:
            results = _simple_search(segs, query, top_k)
    else: results = _simple_search(segs, query, top_k)
    elapsed_ms = (time.time() - t0) * 1000
    source_text = "、".join(sources) if sources else "当前会话"
    if not results:
        return (
            f"<p style='color:#777;font-size:0.9em;margin:0 0 8px 0'>"
            f"{escape(index_status)}；当前检索语料 {len(segs)} 段，来源：{escape(source_text)}；"
            f"本次耗时 {elapsed_ms:.1f} ms</p>"
            "<p style='color:#888'>未找到</p>"
        )
    used_files = sorted({int(s.get("_file_index", 0)) for s in results if "_file_index" in s})
    audio_tags = _audio_tags_for_current_files(used_files)
    status_html = (
        f"<p style='color:#777;font-size:0.9em;margin:0 0 8px 0'>"
        f"{escape(index_status)}；当前检索语料 {len(segs)} 段，来源：{escape(source_text)}；"
        f"本次耗时 {elapsed_ms:.1f} ms"
        f"</p>"
    )
    return status_html + audio_tags + "\n".join(
        _render_seg_html(
            s,
            audio_id=f"file-audio-{int(s.get('_file_index', 0))}" if "_file_index" in s else None,
            file_index=s.get("_file_index"),
        )
        for s in results
    )


def _segments_match_current(indexed: list[dict], current: list[dict]) -> bool:
    if len(indexed) != len(current):
        return False
    for a, b in zip(indexed, current):
        if str(a.get("text", "")).strip() != str(b.get("text", "")).strip():
            return False
        if str(a.get("speaker", "")).strip() != str(b.get("speaker", "")).strip():
            return False
        if abs(float(a.get("start", 0.0)) - float(b.get("start", 0.0))) > 0.05:
            return False
    return True


def _searchable_segments_for_index(segments: list[dict]) -> list[dict]:
    return [seg for seg in segments if len(str(seg.get("text", "")).strip()) >= 4]


def _select_display_search_results(results: list[dict], query: str, top_k: int) -> list[dict]:
    if top_k <= 0:
        return []
    selected = [seg for seg in results if _is_informative_search_result(seg, query)]
    if len(selected) < top_k:
        seen = {
            (
                str(s.get("file", "")),
                str(s.get("text", "")),
                round(float(s.get("start", 0.0)), 2),
            )
            for s in selected
        }
        for seg in results:
            key = (
                str(seg.get("file", "")),
                str(seg.get("text", "")),
                round(float(seg.get("start", 0.0)), 2),
            )
            if key not in seen:
                selected.append(seg)
                seen.add(key)
            if len(selected) >= top_k:
                break
    return selected[:top_k]


def _is_informative_search_result(seg: dict, query: str) -> bool:
    text = str(seg.get("text", "") or "").strip()
    query_text = str(query or "").strip()
    if len(query_text) < 8:
        return True
    cjk_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    latin_words = re.findall(r"[A-Za-z]+", text)
    # Avoid showing one-word/one-phrase hits such as “会议室。” as the first
    # result for multi-keyword demonstration queries.
    return cjk_count >= 8 or len(latin_words) >= 4


def _attach_file_indices_to_search_results(results: list[dict], current: list[dict]) -> list[dict]:
    by_key: dict[tuple[str, str, float], dict] = {}
    for seg in current:
        key = (
            str(seg.get("text", "")).strip(),
            str(seg.get("speaker", "")).strip(),
            round(float(seg.get("start", 0.0)), 1),
        )
        by_key[key] = seg
    attached: list[dict] = []
    for seg in results:
        item = dict(seg)
        key = (
            str(seg.get("text", "")).strip(),
            str(seg.get("speaker", "")).strip(),
            round(float(seg.get("start", 0.0)), 1),
        )
        if key in by_key:
            item["_file_index"] = by_key[key].get("_file_index")
            item["_segment_index"] = by_key[key].get("_segment_index")
            item["_file_name"] = by_key[key].get("_file_name")
        attached.append(item)
    return attached


def _audio_tags_for_current_files(file_indices: list[int]) -> str:
    parts: list[str] = []
    for fi in file_indices:
        if fi < 0 or fi >= len(_last_audio_files):
            continue
        try:
            audio_path = Path(_last_audio_files[fi][1])
            if not audio_path.exists():
                continue
            audio_data, audio_sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            audio_src = "data:audio/wav;base64," + audio_to_base64(
                audio_data.astype(np.float32), audio_sr
            )
            parts.append(
                f"<audio id='file-audio-{fi}' src='{audio_src}' "
                f"preload='metadata' style='display:none'></audio>"
            )
        except Exception:
            continue
    return "".join(parts)

def _simple_search(segments, query, top_k):
    ql=query.lower(); scored=[]
    for seg in segments:
        text=seg.get("text",""); score=0
        if ql in text.lower(): score+=10
        for w in ql.split():
            if w in text.lower(): score+=2
        if score>0: sc=dict(seg); sc["_score"]=score; scored.append(sc)
    scored.sort(key=lambda s:s["_score"], reverse=True)
    return scored[:top_k]


# ══════════════════════════════════════════════════════════════
# UI 构建
# ══════════════════════════════════════════════════════════════

def create_ui(host: str = "127.0.0.1") -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="orange",
        neutral_hue="gray",
    )

    with gr.Blocks(title="多说话人语音转写与智能检索", head=MIC_JS + PLAY_JS, theme=theme,
                   css="body { background: rgb(248,248,246) !important; } "
                        ".gradio-container { background: rgb(248,248,246); } "
                        ".directory-field { gap: 8px !important; } "
                        ".directory-card { background: #fff !important; border-radius: 8px !important; padding: 14px 16px !important; "
                        "gap: 10px !important; box-shadow: none !important; border: 0 !important; } "
                        ".directory-card .upload-label-wrap { margin-bottom: 0 !important; } "
                        ".directory-card .gradio-textbox { margin: 0 !important; } "
                        ".directory-card textarea, .directory-card input { background: #fff !important; } "
                        ".audio-file-card { position: relative !important; background: #fff !important; border-radius: 8px !important; padding: 14px 16px !important; "
                        "gap: 10px !important; box-shadow: none !important; border: 0 !important; } "
                        ".audio-file-card .upload-label-wrap { margin-bottom: 0 !important; } "
                        "#clear-audio-files { position: absolute !important; top: 18px !important; right: 18px !important; "
                        "width: 32px !important; min-width: 32px !important; height: 32px !important; padding: 0 !important; "
                        "border: 0 !important; border-radius: 8px !important; background: #fff !important; color: #f97316 !important; "
                        "box-shadow: 0 1px 6px rgba(0,0,0,.12) !important; font-size: 22px !important; line-height: 1 !important; z-index: 10 !important; } "
                        "#clear-audio-files:hover { background: #fff0df !important; } "
                        ".upload-label-wrap { position: relative !important; display: inline-block !important; width: fit-content !important; "
                        "min-width: 0 !important; margin-bottom: 8px !important; } "
                        ".clickable-field-label { display: inline-flex; align-items: center; width: fit-content; padding: 6px 10px; "
                        "border-radius: 8px; background: #fff0df; color: #f97316; font-size: 16px; font-weight: 700; "
                        "line-height: 1.25; cursor: pointer; user-select: none; } "
                        ".upload-label-wrap:hover .clickable-field-label { background: #ffe4c4; } "
                        ".upload-label-click-layer { position: absolute !important; inset: 0 auto auto 0 !important; width: 100% !important; "
                        "height: 100% !important; min-width: 0 !important; opacity: 0 !important; z-index: 3 !important; margin: 0 !important; } "
                        ".upload-label-click-layer button, .upload-label-click-layer label, .upload-label-click-layer [role='button'] { "
                        "width: 100% !important; height: 100% !important; min-width: 0 !important; min-height: 0 !important; padding: 0 !important; "
                        "margin: 0 !important; cursor: pointer !important; } "
                        ".directory-field textarea, .directory-field input { cursor: text; } "
                        ".native-file-list { margin-top: 8px !important; } "
                        ".native-file-list > label { display: none !important; } "
                        ".native-file-list button[aria-label='Clear'], "
                        ".native-file-list button[aria-label='Clear files'], "
                        ".native-file-list button[aria-label='Clear All'], "
                        ".native-file-list button[title='Clear'], "
                        ".native-file-list button[title='Clear files'], "
                        ".native-file-list button[title='Clear All'], "
                        ".native-file-list button[aria-label='清除'], "
                        ".native-file-list button[title='清除'] { "
                        "position: fixed !important; left: -10000px !important; top: -10000px !important; "
                        "width: 1px !important; height: 1px !important; opacity: 0 !important; pointer-events: none !important; }") as app:
        gr.Markdown("# 🎙️ 多说话人语音转写与智能检索系统")

        mode = gr.Radio(
            choices=["📁 文件转写", "🕘 历史转写", "🎤 实时麦克风", "🔍 智能检索"],
            value="📁 文件转写", label="选择模式", interactive=True,
        )

        # ── 文件转写 ──────────────────────────────────────────
        with gr.Column(visible=True) as file_panel:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙ 实时记录控制")
                    accumulated_files = gr.State([])
                    with gr.Column(elem_classes=["audio-file-card"]):
                        with gr.Column(elem_classes=["upload-label-wrap"]):
                            gr.HTML("<div class='clickable-field-label'>📄 选择音频文件（可多次追加）</div>")
                            file_upload = gr.UploadButton(
                                "", file_types=["audio"], file_count="multiple",
                                size="sm", min_width=0, elem_classes=["upload-label-click-layer"],
                            )
                        clear_audio_files = gr.Button("×", elem_id="clear-audio-files")
                        file_list = gr.File(
                            file_types=["audio"], file_count="multiple",
                            show_label=False, elem_classes=["native-file-list"],
                        )

                    def _upload_path(f):
                        if isinstance(f, str):
                            return f
                        if isinstance(f, dict):
                            return f.get("path") or f.get("name") or str(f)
                        return getattr(f, "path", None) or getattr(f, "name", None) or str(f)

                    # 每次新选择合并到累积列表
                    def _accumulate_files(new_files, existing):
                        existing = existing or []
                        if not new_files:
                            return existing, existing
                        seen_names = {Path(_upload_path(f)).name.lower() for f in existing}
                        for f in (new_files if isinstance(new_files, list) else [new_files]):
                            path = _upload_path(f)
                            if not path:
                                continue
                            name = Path(path).name.lower()
                            if name not in seen_names:
                                existing.append(path)
                                seen_names.add(name)
                        return existing, existing

                    def _sync_file_list(current_files):
                        current_files = current_files or []
                        return [_upload_path(f) for f in current_files if _upload_path(f)]

                    file_upload.upload(
                        fn=_accumulate_files,
                        inputs=[file_upload, accumulated_files],
                        outputs=[accumulated_files, file_list],
                        show_api=False,
                    )
                    file_list.change(fn=_sync_file_list, inputs=[file_list], outputs=[accumulated_files], show_api=False)
                    clear_audio_files.click(fn=lambda: ([], []), inputs=[], outputs=[accumulated_files, file_list], show_api=False)
                    with gr.Row():
                        file_lang = gr.Dropdown(choices=["zh","en"], value="zh", label="语言", scale=1)
                        file_dur = gr.Number(value=120, label="最长秒数 (0=不限)", precision=0, scale=1)
                    file_llm = gr.Checkbox(value=False, label="LLM 后处理（纠错+意图标注）")
                    llm_model_choice = gr.Dropdown(
                        choices=["deepseek-v4-flash", "deepseek-v4-pro",
                                 "claude-sonnet-4-20250514"],
                        value="deepseek-v4-flash", label="LLM 模型",
                        info="flash=快速便宜 pro=旗舰质量")
                    with gr.Column(elem_classes=["directory-field", "directory-card"]):
                        with gr.Column(elem_classes=["upload-label-wrap"]):
                            gr.HTML("<div class='clickable-field-label'>输出目录</div>")
                            out_browse = gr.Button(
                                "", size="sm", min_width=0,
                                elem_classes=["upload-label-click-layer"])
                        output_dir = gr.Textbox(
                            value="tests/test_results", show_label=False,
                            elem_id="output-dir-input")
                    file_btn = gr.Button("▶ 开始处理", variant="primary", size="lg")

                    def _choose_directory(current_value):
                        try:
                            import tkinter as tk
                            from tkinter import filedialog

                            current = (current_value or "").strip()
                            initial = Path(current)
                            if current and not initial.is_absolute():
                                initial = project_root() / initial
                            if not initial.exists():
                                initial = initial.parent if initial.parent.exists() else project_root()

                            root = tk.Tk()
                            root.withdraw()
                            root.attributes("-topmost", True)
                            selected = filedialog.askdirectory(initialdir=str(initial))
                            root.destroy()
                            return selected or current_value
                        except Exception:
                            return current_value

                    out_browse.click(fn=_choose_directory, inputs=[output_dir], outputs=[output_dir], show_api=False)

                with gr.Column(scale=2):
                    gr.Markdown("### 📋 转写结果（▸点击播放 · 点击文件名折叠）")
                    file_html = gr.HTML(
                        value="<p style='color:#888'>选择文件 → 开始处理 → 点击文字 ▸ 播放对应音频段</p>",
                    )
                    gr.Markdown("### ✍ 人工修正")
                    manual_file_select = gr.Dropdown(
                        choices=[], label="修正音频文件", interactive=True,
                    )
                    manual_edit_text = gr.Textbox(
                        label="可编辑转写文本",
                        lines=8,
                        placeholder="[0:0] audio.wav 00:00.0-00:03.0 SPEAKER_00: 在这里修改文本，也可修改 SPEAKER_00",
                    )
                    apply_edit_btn = gr.Button("应用人工修改并保存 JSON", variant="secondary")
                    with gr.Row():
                        speaker_source = gr.Dropdown(
                            choices=[], label="批量修改说话人", interactive=True, scale=1,
                        )
                        speaker_target = gr.Textbox(
                            label="改为", placeholder="SPEAKER_00 或姓名", scale=1,
                        )
                        apply_speaker_btn = gr.Button("批量应用", variant="secondary", scale=0)
                    with gr.Row():
                        file_stats = gr.Textbox(label="统计", lines=2, interactive=False, scale=2)
                        file_saved = gr.Textbox(label="保存位置", lines=2, interactive=False, scale=2)

            with gr.Row():
                file_audio_select = gr.Dropdown(
                    choices=[], label="播放器音频", interactive=True, scale=1,
                )
                file_audio_player = gr.Audio(
                    label="音频播放器（点击文字 ▸ 可跳转播放）", type="filepath",
                    interactive=False, visible=True, elem_id="main-audio-player", scale=2,
                )
            file_audio_sr = gr.Textbox(visible=False)

            file_btn.click(
                fn=process_files,
                inputs=[accumulated_files, file_lang, file_llm, file_dur, output_dir, llm_model_choice],
                outputs=[
                    file_html, file_stats, file_saved, file_audio_player, file_audio_sr,
                    manual_edit_text, file_audio_select, speaker_source, manual_file_select,
                ],
                show_api=False,
            )
            file_audio_select.change(
                fn=select_audio_for_player,
                inputs=[file_audio_select],
                outputs=[file_audio_player],
                show_api=False,
            )
            apply_edit_btn.click(
                fn=apply_manual_edits,
                inputs=[manual_edit_text, output_dir, manual_file_select],
                outputs=[file_html, manual_edit_text, file_saved, speaker_source],
                show_api=False,
            )
            apply_speaker_btn.click(
                fn=apply_speaker_mapping,
                inputs=[speaker_source, speaker_target, output_dir, manual_file_select],
                outputs=[file_html, manual_edit_text, file_saved, speaker_source],
                show_api=False,
            )
            manual_file_select.change(
                fn=select_manual_file,
                inputs=[manual_file_select],
                outputs=[manual_edit_text, speaker_source],
                show_api=False,
            )

        # ── 历史转写 ──────────────────────────────────────────
        with gr.Column(visible=False) as history_panel:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🕘 历史记录")
                    history_dir = gr.Textbox(
                        value="tests/test_results", label="历史目录",
                    )
                    history_refresh = gr.Button("刷新历史", variant="secondary")
                    history_choice = gr.Dropdown(
                        choices=[], label="历史 JSON", interactive=True,
                    )
                    history_load = gr.Button("加载历史转写", variant="primary")
                    history_stats = gr.Textbox(label="统计", lines=3, interactive=False)
                    history_saved = gr.Textbox(label="状态", lines=2, interactive=False)
                with gr.Column(scale=2):
                    history_html = gr.HTML(
                        value="<p style='color:#888'>选择历史 JSON 后加载，避免重复转写</p>",
                    )
            with gr.Row():
                history_audio_select = gr.Dropdown(
                    choices=[], label="播放器音频", interactive=True, scale=1,
                )
                history_audio_player = gr.Audio(
                    label="历史音频播放器（点击文字 ▸ 可跳转播放）", type="filepath",
                    interactive=False, visible=True, elem_id="history-audio-player", scale=2,
                )

            history_refresh.click(
                fn=refresh_history,
                inputs=[history_dir],
                outputs=[history_choice, history_saved],
                show_api=False,
            )
            history_load.click(
                fn=load_history_result,
                inputs=[history_choice, history_dir],
                outputs=[history_html, history_stats, history_saved, history_audio_player, history_audio_select],
                show_api=False,
            )
            history_audio_select.change(
                fn=select_audio_for_player,
                inputs=[history_audio_select],
                outputs=[history_audio_player],
                show_api=False,
            )

        # ── 实时麦克风 ────────────────────────────────────────
        with gr.Column(visible=False) as rt_panel:
            rt = _get_rt_controller()
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙ 控制")
                    rt_chunk = gr.Slider(5, 30, value=10, step=1, label="块时长（秒）")
                    gr.Markdown(
                        "实时阶段只跑 ASR，不做说话人分离。块越短延迟越低但断句更粗；"
                        "推荐 8-10 秒。5-6 秒偏实时，12-15 秒更稳，停止后可离线精修。"
                    )
                    rt_lang = gr.Dropdown(choices=["zh","en"], value="zh", label="语言")
                    rt_llm = gr.Checkbox(value=True, label="离线精修时启用 LLM")
                    rt_llm_model = gr.Dropdown(
                        choices=["deepseek-v4-flash", "deepseek-v4-pro"],
                        value="deepseek-v4-flash", label="离线精修 LLM 模型")
                    rt_save_dir = gr.Textbox(
                        value="tests/test_results",
                        label="保存目录",
                        elem_id="rt-output-dir-input",
                    )
                    rt_test_audio = gr.File(
                        label="实时测试音频文件",
                        type="filepath",
                        file_types=[".wav", ".flac", ".mp3", ".m4a", ".ogg"],
                    )
                    with gr.Row():
                        rt_load_btn = gr.Button("🔌 加载实时 ASR", variant="secondary")
                        rt_start_btn = gr.Button("▶ 开始录音", variant="primary")
                        rt_file_test_btn = gr.Button("▶ 文件流式测试", variant="secondary")
                        rt_system_btn = gr.Button("▶ 浏览器音频共享", variant="secondary")
                        rt_pause_btn = gr.Button("Ⅱ 暂停", variant="secondary")
                        rt_stop_btn = gr.Button("⏹ 停止并保存", variant="stop")
                        rt_discard_btn = gr.Button("丢弃", variant="secondary")
                    rt_refine_btn = gr.Button("🧭 离线精修本次录音", variant="primary")
                    rt_status_text = gr.Textbox(label="状态", value="未加载", interactive=False)
                    rt_session_info = gr.Textbox(label="会话", value="就绪", interactive=False)

                with gr.Column(scale=2):
                    gr.Markdown("### ⚡ ASR 直接实时结果")
                    gr.HTML(
                        "<pre id='rt-raw-stream-box' "
                        "style='min-height:180px;max-height:260px;overflow:auto;"
                        "white-space:pre-wrap;background:#fffaf2;border:1px solid #f5d6a5;"
                        "border-radius:6px;padding:10px 12px;margin:0;"
                        "font-family:Consolas,monospace;font-size:0.92em;line-height:1.45'>"
                        "等待 ASR 直接结果...</pre>"
                    )
                    rt_raw_html = gr.Textbox(
                        value="等待 ASR 直接结果...",
                        visible=False,
                        elem_id="rt-raw-source",
                    )
                    gr.Markdown("### 📋 实时转写临时稿（无说话人 · 停止后 ▸ 点击播放）")
                    rt_html = gr.HTML(value="<p style='color:#888'>等待录音...</p>")

            rt_timer = gr.Timer(1.5)
            rt_dummy = gr.Textbox(visible=False)
            rt_timer.tick(fn=rt.poll, inputs=[rt_dummy], outputs=[rt_html, rt_raw_html, rt_session_info, rt_status_text], show_api=False)
            rt_load_btn.click(fn=rt.init_model, inputs=[rt_lang, rt_llm, rt_llm_model], outputs=[rt_status_text, rt_html], show_api=False)
            rt_start_btn.click(fn=rt.mark_recording, inputs=[rt_lang, rt_llm, rt_llm_model], outputs=[rt_status_text, rt_html], show_api=False)
            rt_start_btn.click(fn=None, inputs=[rt_chunk], outputs=[rt_start_btn], js="async (c)=>{micChunkDuration=c||10;const s=await startMic();return {__type__:'update',value:(s==='error'?'▶ 开始录音':'● 录音中'),variant:(s==='error'?'primary':'secondary')}}", show_api=False)
            rt_file_test_btn.click(
                fn=rt.start_file_stream_test,
                inputs=[rt_test_audio, rt_chunk, rt_lang, rt_llm, rt_llm_model],
                outputs=[rt_status_text, rt_html, rt_raw_html, rt_session_info],
                show_api=False,
            )
            rt_system_btn.click(fn=rt.mark_recording, inputs=[rt_lang, rt_llm, rt_llm_model], outputs=[rt_status_text, rt_html], show_api=False)
            rt_system_btn.click(fn=None, inputs=[rt_chunk], outputs=[rt_system_btn], js="async (c)=>{micChunkDuration=c||10;const s=await startSystemAudio();let label='▶ 浏览器音频共享';if(s==='recording')label='● 浏览器音频中';if(s==='noaudio')label='未共享音频';return {__type__:'update',value:label,variant:(s==='recording'?'secondary':'secondary')}}", show_api=False)
            rt_pause_btn.click(fn=None, inputs=[], outputs=[rt_pause_btn], js="()=>{const s=pauseMic();return {__type__:'update',value:(s==='paused'?'▶ 继续':'Ⅱ 暂停'),variant:'secondary'}}", show_api=False)
            rt_pause_btn.click(fn=rt.toggle_pause, inputs=[], outputs=[rt_status_text, rt_session_info], show_api=False)
            rt_stop_btn.click(fn=None, inputs=[], outputs=[], js="()=>{stopMic();return[]}", show_api=False)
            rt_stop_btn.click(fn=rt.stop, inputs=[rt_save_dir], outputs=[rt_status_text, rt_html, rt_session_info], show_api=False)
            rt_stop_btn.click(fn=None, inputs=[], outputs=[rt_start_btn, rt_system_btn, rt_pause_btn], js="()=>[{__type__:'update',value:'▶ 开始录音',variant:'primary'},{__type__:'update',value:'▶ 浏览器音频共享',variant:'secondary'},{__type__:'update',value:'Ⅱ 暂停',variant:'secondary'}]", show_api=False)
            rt_refine_btn.click(
                fn=rt.refine_last_recording,
                inputs=[rt_lang, rt_llm, rt_llm_model, rt_save_dir],
                outputs=[rt_status_text, rt_html, rt_session_info],
                show_api=False,
            )
            rt_discard_btn.click(fn=None, inputs=[], outputs=[], js="()=>{stopMic();return[]}", show_api=False)
            rt_discard_btn.click(fn=rt.discard, inputs=[], outputs=[rt_status_text, rt_html, rt_session_info], show_api=False)
            rt_discard_btn.click(fn=None, inputs=[], outputs=[rt_start_btn, rt_system_btn, rt_pause_btn], js="()=>[{__type__:'update',value:'▶ 开始录音',variant:'primary'},{__type__:'update',value:'▶ 浏览器音频共享',variant:'secondary'},{__type__:'update',value:'Ⅱ 暂停',variant:'secondary'}]", show_api=False)

        # ── 检索 ──────────────────────────────────────────────
        with gr.Column(visible=False) as search_panel:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🔍 条件")
                    search_query = gr.Textbox(label="查询", placeholder="Speaker B 的反对意见", lines=2)
                    search_topk = gr.Slider(1, 50, value=3, step=1, label="返回条数")
                    search_btn = gr.Button("🔍 搜索", variant="primary")
                with gr.Column(scale=2):
                    search_output = gr.HTML(value="<p style='color:#888'>请先在文件转写模式处理音频</p>")
            search_btn.click(fn=search_transcript, inputs=[search_query, search_topk], outputs=[search_output], show_api=False)

        # ── 模式切换 ──
        def on_mode_change(m):
            return (gr.update(visible=m=="📁 文件转写"),
                    gr.update(visible=m=="🕘 历史转写"),
                    gr.update(visible=m=="🎤 实时麦克风"),
                    gr.update(visible=m=="🔍 智能检索"))
        mode.change(
            fn=on_mode_change,
            inputs=[mode],
            outputs=[file_panel, history_panel, rt_panel, search_panel],
            show_api=False,
        )

    return app


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860); p.add_argument("--ws-port", type=int, default=7861)
    p.add_argument("--no-ws", action="store_true"); args = p.parse_args()
    if not args.no_ws:
        import threading
        threading.Thread(target=_get_rt_controller().start_ws_server, args=(args.host, args.ws_port), daemon=True).start()
        time.sleep(0.5)
    app = create_ui(host=args.host)
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
        allowed_paths=[str(project_root())],
    )

if __name__ == "__main__":
    main()
