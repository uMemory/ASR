"""多说话人语音转写与智能检索系统——Gradio 统一界面。

模式：📁文件转写 | 🎤实时麦克风 | 🔍智能检索
"""
from __future__ import annotations

import argparse, asyncio, base64, io, json, queue, re, shutil
import struct, sys, threading, time, wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr
import numpy as np
import soundfile as sf
import torch

from src.pipeline import run, _build_speaker_map
from src.utils.config import project_root


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


# ══════════════════════════════════════════════════════════════
# 全局状态
# ══════════════════════════════════════════════════════════════

_last_result: dict | None = None
_last_segments_raw: list[dict] = []   # 未映射的原始段
_index_path: Path = project_root() / "outputs" / "index"
_output_dir: Path = project_root() / "tests" / "test_results"
_current_audio_b64: str = ""
_current_audio_sr: int = 16000

_rt_pipeline: object | None = None
_rt_queue: queue.Queue = queue.Queue()
_rt_segments: dict[int, dict] = {}
_rt_seg_counter: int = 0
_rt_recording: list[np.ndarray] = []
_rt_chunk_b64: dict[int, str] = {}


# ══════════════════════════════════════════════════════════════
# 实时管线后端
# ══════════════════════════════════════════════════════════════

class RealtimePipeline:
    def __init__(self, language: str = "zh", enable_llm: bool = True) -> None:
        from src.utils.config import get_model_config, load_config
        mcfg = get_model_config(); lcfg = load_config("languages")
        self.language = language; self.enable_llm = enable_llm
        self.device = mcfg["device"]; self.lang_cfg = lcfg.get(language, {})
        from src.diarization import load_diarization_backend
        self.dia_backend = load_diarization_backend({**mcfg["diarization"], "device": self.device})
        self.dia_backend.load()
        from src.asr import load_asr_backend
        self.asr_backend = load_asr_backend({**mcfg["asr"], "device": self.device})
        self.asr_backend.load()
        self.llm_adapter = None
        if enable_llm:
            from src.llm import load_llm_adapter, llm_correct_segments, llm_tag_intents
            self.llm_adapter = load_llm_adapter()
            self._llm_correct = llm_correct_segments; self._llm_tag = llm_tag_intents
        self._history: list[dict] = []

    def process_fast(self, waveform: np.ndarray) -> list[dict]:
        if waveform.size == 0: return []
        sr = 16000
        dia = self.dia_backend.diarize(waveform, sample_rate=sr)
        prompt = self.lang_cfg.get("asr_initial_prompt")
        asr = self.asr_backend.transcribe(waveform, language=self.language, initial_prompt=prompt)

        # Word-level speaker assignment + turn aggregation
        import pandas as pd
        from src.pipeline import _aggregate_words_to_turns

        asr_words = asr["segments"]  # word-level now
        word_segs = []
        for w in asr_words:
            txt = w.get("text", "").strip()
            if not txt:
                continue
            word_segs.append({"word": txt, "start": w["start"], "end": w["end"], "score": 0.5})

        aligned = {
            "segments": [{
                "start": word_segs[0]["start"] if word_segs else 0.0,
                "end": word_segs[-1]["end"] if word_segs else 0.0,
                "text": " ".join(w["word"] for w in word_segs),
                "words": word_segs,
            }],
            "word_segments": word_segs,
        }
        dia_segs = dia["segments"]
        diarize_df = pd.DataFrame([
            {"start": s["start"], "end": s["end"], "speaker": s["speaker"]}
            for s in dia_segs
        ])
        import whisperx
        result = whisperx.assign_word_speakers(diarize_df, aligned)
        merged = _aggregate_words_to_turns(result.get("word_segments", []))

        # Hallucination cleanup + speaker map
        from src.llm.corrector import clean_hallucination, zh_simplify
        for seg in merged:
            txt = clean_hallucination(seg.get("text", ""))
            seg["text"] = zh_simplify(txt)
        spk_map = _build_speaker_map(merged)
        for seg in merged:
            seg["speaker"] = spk_map.get(seg.get("speaker", ""), seg.get("speaker", ""))
        return merged

    def process_llm(self, merged: list[dict]) -> list[dict]:
        if not merged or not self.enable_llm or self.llm_adapter is None: return merged
        _MAX = 15; max_ctx = max(0, min(10, _MAX - len(merged)))
        ctx = self._history[-max_ctx:] if self._history and max_ctx > 0 else []
        try:
            corrected = self._llm_correct(self.llm_adapter, ctx + merged, self.language)
            new_segs = corrected[-len(merged):]
            labels = self.lang_cfg.get("intent_labels")
            new_segs = self._llm_tag(self.llm_adapter, new_segs, self.language, labels)
        except Exception:
            new_segs = merged
        self._history.extend(merged)
        if len(self._history) > 100: self._history = self._history[-50:]
        return new_segs

    def unload(self) -> None:
        self.dia_backend.unload(); self.asr_backend.unload()
        del self.dia_backend; del self.asr_backend; self.llm_adapter = None
        if torch.cuda.is_available(): torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# 文件转写
# ══════════════════════════════════════════════════════════════

def _project_path(path_str: str | Path) -> Path:
    p = Path(path_str).expanduser()
    return p if p.is_absolute() else project_root() / p


def find_audio_files(path_str: str) -> list[str]:
    p = _project_path(path_str).resolve()
    if not p.exists(): return []
    if p.is_file() and p.suffix.lower() in (".wav", ".flac", ".mp3", ".m4a", ".ogg"):
        return [str(p)]
    if p.is_dir():
        exts = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}
        return sorted([str(f) for ext in exts for f in p.rglob(f"*{ext}")])
    return []


def _speaker_label(speaker: object) -> str:
    label = str(speaker or "?").strip()
    if re.fullmatch(r"[A-Z]", label):
        return f"Speaker {label}"
    return label


def _render_seg_html(
    seg: dict,
    show_intent: bool = True,
    audio_id: str | None = None,
    onclick: str | None = None,
) -> str:
    """渲染单个段为 HTML 可点击行。"""
    ts = f"[{fmt_time(seg['start'])} - {fmt_time(seg['end'])}]"
    spk = _speaker_label(seg.get("speaker", "?")).replace("<", "&lt;")
    txt = seg.get("llm_text", "") or seg.get("text", "")
    txt = txt.replace("<", "&lt;").replace(">", "&gt;")
    orig = seg.get("text", "").replace("<", "&lt;").replace(">", "&gt;")
    phase = seg.get("phase", "")
    intent = seg.get("llm_intent") or seg.get("intent", "")
    if isinstance(intent, list): intent = "+".join(intent)

    if phase == "corrected" and txt != orig:
        prefix = "✅"
        note = (f" <span style='color:#999;font-size:0.8em'>"
                f"(原: {orig[:40]}{'…' if len(orig)>40 else ''})</span>")
    elif phase == "corrected":
        prefix = "✅"; note = ""
    else:
        prefix = "⚡"; note = ""

    intent_html = f" <span style='color:#e67e22;font-size:0.85em'>[{intent}]</span>" if show_intent and intent else ""
    audio_arg = f",'{audio_id}'" if audio_id else ""
    click_js = onclick or f"playSeg({seg['start']},{seg['end']}{audio_arg})"
    return (
        f"<div class='seg-line' onclick=\"{click_js}\" "
        f"title='点击播放 [{ts}]' "
        f"style='cursor:pointer;padding:3px 6px;margin:1px 0;border-radius:4px;"
        f"transition:background 0.15s' "
        f"onmouseover='this.style.background=\"#fdebd0\"' "
        f"onmouseout='this.style.background=\"transparent\"'>"
        f"<span style='color:#e67e22'>▸</span> {prefix} "
        f"<span style='color:#888;font-family:monospace;font-size:0.9em'>{ts}</span> "
        f"<b style='color:#c0392b'>{spk}</b>: {txt}{note}{intent_html}</div>"
    )


def process_files(
    file_paths: list[str] | None,
    dir_path: str,
    language: str,
    enable_llm: bool,
    max_duration: float,
    output_dir_str: str,
    llm_model: str,
    progress=gr.Progress(),
) -> tuple[str, str, str, str, str]:
    global _last_result, _last_segments_raw, _current_audio_b64, _current_audio_sr

    # 收集文件
    files: list[str] = []
    if file_paths:
        for item in file_paths:
            f = item.get("path", "") if isinstance(item, dict) else item
            if f and Path(f).suffix.lower() in (".wav", ".flac", ".mp3", ".m4a", ".ogg"):
                files.append(f)
    if dir_path and dir_path.strip():
        dir_files = find_audio_files(dir_path)
        for df in dir_files:
            if df not in files:
                files.append(df)

    if not files:
        return ("<p style='color:#888'>请选择音频文件或输入目录路径</p>", "", "", "", "")

    out_dir = _project_path(output_dir_str.strip()) if output_dir_str.strip() else _output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    html_parts: list[str] = []
    all_stats: list[str] = []
    saved_files: list[str] = []
    primary_audio_b64 = ""; primary_audio_sr = 16000
    _last_segments_raw = []

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
            html_parts.append(f"<p style='color:red'>✗ {fname}: {e}</p>")
            continue

        _last_result = result; segs = result["segments"]
        _last_segments_raw.extend(segs)
        timing = result.get("timing", {})

        # 会议摘要
        meeting_summary = result.get("meeting_summary", {})
        summary_html = ""
        if meeting_summary.get("topic"):
            kps = "".join(f"<li>{p}</li>" for p in meeting_summary.get("key_points", [])[:5])
            decs = "".join(f"<li>{d}</li>" for d in meeting_summary.get("decisions", [])[:3])
            acts = "".join(f"<li>{a}</li>" for a in meeting_summary.get("action_items", [])[:3])
            ul_open = '<ul style="margin:4px 0">'
            ul_close = '</ul>'
            kps_block = f"<br><b>要点:</b>{ul_open}{kps}{ul_close}" if kps else ""
            decs_block = f"<b>决策:</b>{ul_open}{decs}{ul_close}" if decs else ""
            acts_block = f"<b>待办:</b>{ul_open}{acts}{ul_close}" if acts else ""
            summary_html = (
                f"<div style='background:#fef5e7;border-left:3px solid #e67e22;"
                f"padding:8px 12px;margin-bottom:10px;border-radius:0 6px 6px 0;font-size:0.95em'>"
                f"<b>📋 {meeting_summary['topic']}</b><br>"
                f"{meeting_summary.get('summary','')}"
                f"{kps_block}{decs_block}{acts_block}"
                f"</div>"
            )

        # 折叠区：每个文件一个 <details>
        audio_id = f"file-audio-{fi}"
        file_audio_html = ""
        try:
            file_audio_b64 = audio_to_base64(audio_data, audio_sr)
            file_audio_html = (
                f"<audio id='{audio_id}' src='data:audio/wav;base64,{file_audio_b64}' "
                f"preload='metadata' style='display:none'></audio>"
            )
        except Exception:
            audio_id = None
        seg_html = "\n".join(_render_seg_html(s, audio_id=audio_id) for s in segs)
        n_spk = len({s.get("speaker", "?") for s in segs})
        summary_label = (f"▸ {fname} — "
                   f"{n_spk}人 · {len(segs)}段 · "
                   f"总耗时 {timing.get('total',0):.1f}s"
                   f"{' +LLM' if timing.get('llm') else ''}")
        html_parts.append(
            f"<details open style='margin:8px 0;border:1px solid #f0d0a0;border-radius:6px;padding:8px'>"
            f"<summary style='cursor:pointer;font-weight:bold;color:#e67e22'>{summary_label}</summary>"
            f"{file_audio_html}<div style='margin-top:6px'>{summary_html}{seg_html}</div>"
            f"</details>"
        )

        out_path = out_dir / f"{fname}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"file": fpath, "segments": segs, "timing": timing,
                        "num_speakers": result.get("num_speakers"),
                        "meeting_summary": meeting_summary},
                       f, ensure_ascii=False, indent=2)
        saved_files.append(str(out_path))
        all_stats.append(f"{fname}: {n_spk}人 · {len(segs)}段 · {timing.get('total',0):.1f}s")

    progress(1.0, desc="完成")
    _current_audio_b64 = primary_audio_b64; _current_audio_sr = primary_audio_sr

    # 保存第一个音频文件供播放器使用
    audio_file_path = ""
    if files:
        first_audio = files[0]
        audio_copy = out_dir / "_current_audio.wav"
        try:
            ad, asr = sf.read(first_audio)
            if ad.ndim > 1: ad = ad.mean(axis=1)
            if max_duration > 0 and len(ad) > max_duration * asr:
                ad = ad[:int(max_duration * asr)]
            sf.write(str(audio_copy), ad.astype(np.float32), asr)
            audio_file_path = str(audio_copy)
        except Exception:
            pass

    full_html = "\n".join(html_parts) if html_parts else "<p>无结果</p>"
    # 添加刷新警告
    full_html = ("<p style='color:#e67e22;font-size:0.85em'>"
                 "⚠ 处理中请勿刷新页面，结果已自动保存到输出目录</p>") + full_html

    return full_html, "\n".join(all_stats), f"输出: {out_dir}", audio_file_path, str(primary_audio_sr)


def _build_rename_panel(segments: list[dict]) -> str:
    """生成说话人重命名 HTML 面板。"""
    spks = list(dict.fromkeys(s.get("speaker", "?") for s in segments))
    if not spks: return ""
    rows = []
    for spk in spks:
        rows.append(
            f"<div style='display:flex;align-items:center;gap:8px;margin:3px 0'>"
            f"<span style='font-weight:bold;min-width:30px;color:#c0392b'>{spk}</span>"
            f"→ <input id='rename-{spk}' placeholder='输入姓名...' "
            f"style='padding:2px 6px;border:1px solid #e67e22;border-radius:4px;width:120px' "
            f"oninput='applyRename()'>"
            f"</div>"
        )
    return (
        f"<div style='margin-top:12px;padding:8px;border:1px dashed #e67e22;border-radius:6px'>"
        f"<div style='font-weight:bold;color:#e67e22;margin-bottom:4px'>🔤 说话人重命名</div>"
        + "".join(rows) +
        f"</div>"
        f"<script>function applyRename(){{"
        f"document.querySelectorAll('.seg-line b').forEach(el=>{{"
        f"var spk=el.textContent; var inp=document.getElementById('rename-'+spk);"
        f"if(inp&&inp.value) el.textContent=inp.value;"
        f"}});}}</script>"
    )


# ══════════════════════════════════════════════════════════════
# 实时麦克风
# ══════════════════════════════════════════════════════════════

def rt_init_model(language: str, enable_llm: bool, llm_model: str) -> tuple[str, str]:
    global _rt_pipeline, _rt_recording, _rt_chunk_b64
    if _rt_pipeline is not None:
        return "模型已加载", _render_rt_html()
    import os
    if llm_model:
        os.environ["DEEPSEEK_MODEL"] = llm_model
    _rt_pipeline = RealtimePipeline(language=language, enable_llm=enable_llm)
    _rt_recording.clear(); _rt_chunk_b64.clear()
    return "模型就绪 — 点击「开始录音」", ""


def rt_poll(_dummy: str) -> tuple[str, str, str]:
    global _rt_segments, _rt_seg_counter, _rt_chunk_b64
    while not _rt_queue.empty():
        try: item = _rt_queue.get_nowait()
        except queue.Empty: break
        typ = item.get("type")
        if typ == "asr":
            sid = _rt_seg_counter; _rt_seg_counter += 1
            _rt_segments[sid] = {"phase":"asr","speaker":item.get("speaker","?"),
                "start":item.get("start",0),"end":item.get("end",0),
                "text":item.get("text",""),"intent":item.get("intent",[]),
                "llm_text":"","llm_intent":[],"chunk_id":item.get("chunk_id",0),
                "chunk_start":item.get("chunk_start",0.0)}
        elif typ == "llm":
            for lseg in item.get("segments",[]):
                lstart=lseg.get("start",0); best_sid=None; best_dist=float("inf")
                for sid,seg in _rt_segments.items():
                    if seg["phase"]=="asr":
                        d=abs(seg["start"]-lstart)
                        if d<best_dist and d<2.0: best_dist=d; best_sid=sid
                if best_sid is not None:
                    _rt_segments[best_sid]["llm_text"]=lseg.get("text","")
                    _rt_segments[best_sid]["llm_intent"]=lseg.get("intent",[])
                    if _rt_segments[best_sid]["llm_text"]!=_rt_segments[best_sid]["text"]:
                        _rt_segments[best_sid]["phase"]="corrected"
    html=_render_rt_html()
    cids=sorted(_rt_chunk_b64.keys())
    return html, f"块:{len(cids)} 段:{len(_rt_segments)} 录音:{sum(len(c) for c in _rt_recording)/16000:.0f}s", ""


def _render_rt_html() -> str:
    global _rt_segments, _rt_chunk_b64
    if not _rt_segments and not _rt_chunk_b64:
        return "<p style='color:#888'>等待录音...</p>"
    parts=[]
    for cid in sorted(_rt_chunk_b64.keys()):
        b64=_rt_chunk_b64.get(cid,"")
        if b64: parts.append(f"<audio id='rt-audio-{cid}' src='data:audio/wav;base64,{b64}' style='display:none'></audio>")
    for sid in sorted(_rt_segments.keys()):
        seg=_rt_segments[sid]; cid=seg.get("chunk_id",0)
        offset=seg["start"]-seg.get("chunk_start",0)
        onclick=f"playRtSeg({cid},{offset:.3f},{seg['end']-seg['start']:.3f})"
        parts.append(_render_seg_html({**seg,"start":seg["start"],"end":seg["end"]}, onclick=onclick))
    return "\n".join(parts)


def rt_stop() -> tuple[str, str, str]:
    global _rt_pipeline, _rt_segments, _rt_seg_counter, _rt_recording, _rt_chunk_b64
    if _rt_recording:
        full=np.concatenate(_rt_recording)
        _output_dir.mkdir(parents=True, exist_ok=True)
        sp=_output_dir/f"recording_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        sf.write(str(sp), full.astype(np.float32), 16000)
        saved=f"录音已保存: {sp}"
    else: saved=""
    if _rt_pipeline: _rt_pipeline.unload(); _rt_pipeline=None
    final=_render_rt_html()
    if saved: final=f"<p style='color:green'>{saved}</p>"+final
    _rt_segments.clear(); _rt_seg_counter=0; _rt_recording.clear(); _rt_chunk_b64.clear()
    return saved or "已停止", final, ""


# ══════════════════════════════════════════════════════════════
# JS: 播放 + 重命名
# ══════════════════════════════════════════════════════════════

_PLAY_JS = """<script>
let activeTimeout=null;
function _getAudio(audioId){
  if(audioId){
    var direct=document.getElementById(audioId);
    if(direct)return direct;
  }
  // 优先从 Gradio Audio 组件 (elem_id=main-audio-player) 中找 <audio>
  var wrap=document.getElementById('main-audio-player');
  if(wrap){var a=wrap.querySelector('audio');if(a)return a;}
  // 回退：查找页面上任意 audio
  var all=document.querySelectorAll('audio');
  for(var i=0;i<all.length;i++){if(all[i].src)return all[i];}
  return all.length>0?all[0]:null;
}
function _seekAndPlay(a,s,e){
  if(activeTimeout)clearTimeout(activeTimeout);
  const duration=Math.max(0.1,(e-s))*1000+300;
  const start=function(){
    try{a.currentTime=s;}catch(err){}
    const p=a.play();
    if(p&&p.catch)p.catch(err=>console.log('audio play blocked or failed',err));
    activeTimeout=setTimeout(()=>{a.pause()},duration);
  };
  if(a.readyState>=1){start();return;}
  a.addEventListener('loadedmetadata',start,{once:true});
  a.addEventListener('canplay',start,{once:true});
  try{a.load();}catch(err){}
}
function playSeg(s,e,audioId){let a=_getAudio(audioId);if(!a){console.log('no audio found');return;}
_seekAndPlay(a,s,e);}
function playRtSeg(c,o,d){let a=document.getElementById('rt-audio-'+c);if(!a){a=_getAudio();if(!a)return;}
_seekAndPlay(a,o,o+d);}
function applyRename(){
document.querySelectorAll('.seg-line b').forEach(el=>{
var spk=el.textContent.trim();
var inp=document.getElementById('rename-'+spk);
if(inp&&inp.value.trim()) el.textContent=inp.value.trim();
});
}
</script>"""


# ══════════════════════════════════════════════════════════════
# WebSocket 服务器
# ══════════════════════════════════════════════════════════════

_WS_PORT = 7861

async def _ws_handler(websocket) -> None:
    global _rt_queue, _rt_recording, _rt_chunk_b64
    try:
        async for message in websocket:
            try: msg = json.loads(message)
            except: continue
            if msg.get("type") == "audio":
                b64 = msg.get("data", "")
                if not b64: continue
                wav_bytes = base64.b64decode(b64)
                buf = io.BytesIO(wav_bytes)
                with wave.open(buf, "rb") as wf:
                    sr = wf.getframerate(); nch = wf.getnchannels()
                    raw = wf.readframes(wf.getnframes())
                audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                if nch > 1: audio = audio.reshape(-1, nch).mean(axis=1)
                if audio.size < sr * 0.3: continue
                cid = msg.get("chunk_id", 0)
                _rt_chunk_b64[cid] = b64
                _rt_recording.append(audio.astype(np.float32))
                session_offset = sum(len(c) for c in _rt_recording[:-1]) / sr
                segs = _rt_pipeline.process_fast(audio) if _rt_pipeline else []
                for seg in segs:
                    _rt_queue.put({"type":"asr","chunk_id":cid,"chunk_start":session_offset,**seg})
                await websocket.send(json.dumps({"type":"segments","phase":"asr","segments":segs,"chunk_id":cid}, ensure_ascii=False))
                if segs and _rt_pipeline and _rt_pipeline.enable_llm:
                    corrected = _rt_pipeline.process_llm(segs)
                    _rt_queue.put({"type":"llm","segments":corrected})
                    await websocket.send(json.dumps({"type":"segments","phase":"llm","segments":corrected,"chunk_id":cid}, ensure_ascii=False))
            elif msg.get("type") == "init":
                await websocket.send(json.dumps({"type":"ready","sample_rate":16000,"chunk_duration":msg.get("chunk_duration",10)}))
    except: pass

def _start_ws_server(host: str, port: int) -> None:
    try: import websockets as ws_lib
    except ImportError: print("[WARN] websockets 未安装"); return

    async def _serve():
        async with ws_lib.serve(_ws_handler, host, port, max_size=4*1024*1024):
            print(f"[WS] ws://{host}:{port}"); await asyncio.Future()

    loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
    try: loop.run_until_complete(_serve())
    except Exception as e: print(f"[WS] 错误: {e}")


# ══════════════════════════════════════════════════════════════
# 检索
# ══════════════════════════════════════════════════════════════

def search_transcript(query: str, top_k: int) -> str:
    global _last_result
    if _last_result is None: return "<p style='color:#888'>请先在文件转写模式处理音频</p>"
    segs = _last_result.get("segments", [])
    if not segs: return "<p style='color:#888'>无数据</p>"
    if (_index_path / "dense.faiss").exists():
        try:
            from src.retrieval import EmbeddingEncoder, Retriever
            from src.utils.config import get_model_config
            cfg = get_model_config()
            enc = EmbeddingEncoder(model_path=cfg["embedding"]["model"],
                                   use_fp16=cfg["embedding"].get("use_fp16",True),
                                   device=cfg.get("device","cuda"))
            enc.load(); ret = Retriever(index_path=_index_path, encoder=enc)
            results = ret.search(query, top_k=top_k); enc.unload()
        except: results = _simple_search(segs, query, top_k)
    else: results = _simple_search(segs, query, top_k)
    if not results: return "<p style='color:#888'>未找到</p>"
    return "\n".join(_render_seg_html(s) for s in results)

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
# 前端 JS（浏览器麦克风）
# ══════════════════════════════════════════════════════════════

_MIC_JS = """<script>
let micWs=null,micStream=null,micCtx=null,micRunning=false,micBuffer=[],micChunkId=0,micSampleRate=16000,micChunkDuration=10;
function encodeWAV(samples,sr){const buf=new ArrayBuffer(44+samples.length*2);const v=new DataView(buf);function ws(o,s){for(let i=0;i<s.length;i++)v.setUint8(o+i,s.charCodeAt(i))}ws(0,'RIFF');v.setUint32(4,36+samples.length*2,true);ws(8,'WAVE');ws(12,'fmt ');v.setUint32(16,16,true);v.setUint16(20,1,true);v.setUint16(22,1,true);v.setUint32(24,sr,true);v.setUint32(28,sr*2,true);v.setUint16(32,2,true);v.setUint16(34,16,true);ws(36,'data');v.setUint32(40,samples.length*2,true);for(let i=0;i<samples.length;i++){const s=Math.max(-1,Math.min(1,samples[i]));v.setInt16(44+i*2,s<0?s*0x8000:s*0x7FFF,true)}return new Uint8Array(buf)}
function ab2b64(buf){let b='';const u=new Uint8Array(buf);for(let i=0;i<u.length;i++)b+=String.fromCharCode(u[i]);return btoa(b)}
function connectWs(){const proto=location.protocol==='https:'?'wss:':'ws:';micWs=new WebSocket(proto+'//'+location.hostname+':7861');micWs.onopen=()=>{micWs.send(JSON.stringify({type:'init',chunk_duration:micChunkDuration}))};micWs.onmessage=()=>{};micWs.onclose=()=>{}}
async function startMic(){if(micRunning)return;try{micStream=await navigator.mediaDevices.getUserMedia({audio:{sampleRate:micSampleRate,channelCount:1,echoCancellation:true}})}catch(e){return}connectWs();micCtx=new AudioContext({sampleRate:micSampleRate});const src=micCtx.createMediaStreamSource(micStream);const proc=micCtx.createScriptProcessor(4096,1,1);micBuffer=[];micChunkId=0;micRunning=true;proc.onaudioprocess=function(e){if(!micRunning)return;const inp=e.inputBuffer.getChannelData(0);for(let i=0;i<inp.length;i++)micBuffer.push(inp[i]);if(micBuffer.length>=micSampleRate*micChunkDuration){const chunk=new Float32Array(micBuffer.splice(0,micSampleRate*micChunkDuration));micChunkId++;if(micWs&&micWs.readyState===WebSocket.OPEN)micWs.send(JSON.stringify({type:'audio',data:ab2b64(encodeWAV(chunk,micSampleRate)),chunk_id:micChunkId}))}};src.connect(proc);proc.connect(micCtx.destination)}
function stopMic(){micRunning=false;if(micStream){micStream.getTracks().forEach(t=>t.stop());micStream=null}if(micCtx){micCtx.close();micCtx=null}if(micWs){micWs.close();micWs=null}}
document.addEventListener('focusin', function(e){
  const target=e.target;
  if(!target || !target.closest || !target.closest('#output-dir-input')) return;
  if(target.tagName==='TEXTAREA' || target.tagName==='INPUT'){
    setTimeout(()=>target.select(), 0);
  }
});
</script>"""


# ══════════════════════════════════════════════════════════════
# UI 构建
# ══════════════════════════════════════════════════════════════

def create_ui(host: str = "127.0.0.1") -> gr.Blocks:
    theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="orange",
        neutral_hue="gray",
    )

    with gr.Blocks(title="多说话人语音转写与智能检索", head=_MIC_JS + _PLAY_JS, theme=theme,
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
            choices=["📁 文件转写", "🎤 实时麦克风", "🔍 智能检索"],
            value="📁 文件转写", label="选择模式", interactive=True,
        )

        # ── 文件转写 ──────────────────────────────────────────
        with gr.Column(visible=True) as file_panel:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙ 控制")
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
                    )
                    file_list.change(fn=_sync_file_list, inputs=[file_list], outputs=[accumulated_files])
                    clear_audio_files.click(fn=lambda: ([], []), inputs=[], outputs=[accumulated_files, file_list])
                    with gr.Column(elem_classes=["directory-field", "directory-card"]):
                        with gr.Column(elem_classes=["upload-label-wrap"]):
                            gr.HTML("<div class='clickable-field-label'>输入目录</div>")
                            dir_browse = gr.Button(
                                "", size="sm", min_width=0,
                                elem_classes=["upload-label-click-layer"])
                        dir_input = gr.Textbox(
                            show_label=False, placeholder="dataset/AISHELL-4/test/wav/")
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

                    dir_browse.click(fn=_choose_directory, inputs=[dir_input], outputs=[dir_input])
                    out_browse.click(fn=_choose_directory, inputs=[output_dir], outputs=[output_dir])

                with gr.Column(scale=2):
                    gr.Markdown("### 📋 转写结果（▸点击播放 · 点击文件名折叠）")
                    file_html = gr.HTML(
                        value="<p style='color:#888'>选择文件 → 开始处理 → 点击文字 ▸ 播放对应音频段</p>",
                    )
                    with gr.Row():
                        file_stats = gr.Textbox(label="统计", lines=2, interactive=False, scale=2)
                        file_saved = gr.Textbox(label="保存位置", lines=2, interactive=False, scale=2)

            file_audio_player = gr.Audio(label="音频播放器（点击文字 ▸ 可跳转播放）", type="filepath",
                                         interactive=False, visible=True, elem_id="main-audio-player")
            file_audio_sr = gr.Textbox(visible=False)

            file_btn.click(
                fn=process_files,
                inputs=[accumulated_files, dir_input, file_lang, file_llm, file_dur, output_dir, llm_model_choice],
                outputs=[file_html, file_stats, file_saved, file_audio_player, file_audio_sr],
            )

        # ── 实时麦克风 ────────────────────────────────────────
        with gr.Column(visible=False) as rt_panel:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### ⚙ 控制")
                    rt_chunk = gr.Slider(5, 30, value=10, step=1, label="块时长（秒）")
                    rt_lang = gr.Dropdown(choices=["zh","en"], value="zh", label="语言")
                    rt_llm = gr.Checkbox(value=True, label="LLM 上下文纠错")
                    rt_llm_model = gr.Dropdown(
                        choices=["deepseek-v4-flash", "deepseek-v4-pro"],
                        value="deepseek-v4-flash", label="LLM 模型")
                    with gr.Row():
                        rt_load_btn = gr.Button("🔌 加载模型", variant="secondary")
                        rt_start_btn = gr.Button("▶ 开始录音", variant="primary")
                        rt_stop_btn = gr.Button("⏹ 停止并保存", variant="stop")
                    rt_status_text = gr.Textbox(label="状态", value="未加载", interactive=False)
                    rt_session_info = gr.Textbox(label="会话", value="就绪", interactive=False)

                with gr.Column(scale=2):
                    gr.Markdown("### 📋 实时转写（▸ 点击播放）")
                    rt_html = gr.HTML(value="<p style='color:#888'>等待录音...</p>")

            rt_timer = gr.Timer(1.5)
            rt_dummy = gr.Textbox(visible=False)
            rt_timer.tick(fn=rt_poll, inputs=[rt_dummy], outputs=[rt_html, rt_session_info, rt_status_text])
            rt_load_btn.click(fn=rt_init_model, inputs=[rt_lang, rt_llm, rt_llm_model], outputs=[rt_status_text, rt_html])
            rt_start_btn.click(fn=None, inputs=[rt_chunk], outputs=[], js="(c)=>{micChunkDuration=c||10;startMic();return[]}")
            rt_stop_btn.click(fn=None, inputs=[], outputs=[], js="()=>{stopMic();return[]}")
            rt_stop_btn.click(fn=rt_stop, inputs=[], outputs=[rt_status_text, rt_html, rt_session_info])

        # ── 检索 ──────────────────────────────────────────────
        with gr.Column(visible=False) as search_panel:
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🔍 条件")
                    search_query = gr.Textbox(label="查询", placeholder="Speaker B 的反对意见", lines=2)
                    search_topk = gr.Slider(1, 50, value=10, step=1, label="返回条数")
                    search_btn = gr.Button("🔍 搜索", variant="primary")
                with gr.Column(scale=2):
                    search_output = gr.HTML(value="<p style='color:#888'>请先在文件转写模式处理音频</p>")
            search_btn.click(fn=search_transcript, inputs=[search_query, search_topk], outputs=[search_output])

        # ── 模式切换 ──
        def on_mode_change(m):
            return (gr.update(visible=m=="📁 文件转写"),
                    gr.update(visible=m=="🎤 实时麦克风"),
                    gr.update(visible=m=="🔍 智能检索"))
        mode.change(fn=on_mode_change, inputs=[mode], outputs=[file_panel, rt_panel, search_panel])

    return app


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7860); p.add_argument("--ws-port", type=int, default=7861)
    p.add_argument("--no-ws", action="store_true"); args = p.parse_args()
    global _WS_PORT; _WS_PORT = args.ws_port
    if not args.no_ws:
        threading.Thread(target=_start_ws_server, args=(args.host, args.ws_port), daemon=True).start()
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
