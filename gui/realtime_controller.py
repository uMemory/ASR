"""Realtime microphone controller for the Gradio UI."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import queue
import threading
import time
import wave
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf
import gradio as gr

from gui.realtime_backend import RealtimePipeline


class RealtimeController:
    def __init__(
        self,
        output_dir: Path,
        render_seg_html: Callable[..., str],
        audio_to_base64: Callable[[np.ndarray, int], str],
        run_pipeline: Callable[..., dict],
        on_refined: Callable[[dict, Path], None] | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.render_seg_html = render_seg_html
        self.audio_to_base64 = audio_to_base64
        self.run_pipeline = run_pipeline
        self.on_refined = on_refined

        self.pipeline: RealtimePipeline | None = None
        self.queue: queue.Queue = queue.Queue()
        self.work_queue: queue.Queue = queue.Queue()
        self.worker_started = False
        self.segments: dict[int, dict] = {}
        self.raw_asr_items: dict[int, dict] = {}
        self.raw_asr_seq = 0
        self._last_render_html = ""
        self._last_raw_text = ""
        self._last_session_info = ""
        self._last_status_for_poll = ""
        self.seg_counter = 0
        self.recording: list[np.ndarray] = []
        self.status_text = "未加载"
        self.recording_active = False
        self.recording_paused = False
        self.session_id = 0
        self.last_recording_path = ""
        self.received_chunks: set[tuple[int, int]] = set()
        self.committed_until = 0.0
        self.raw_committed_until = 0.0
        self.raw_window_seq = 0
        self.committed_text_tail = ""

        self.window_s = 22.0
        self.stable_lag_s = 3.0
        self.min_short_seg_s = 0.8
        self.min_commit_overlap_s = 0.6
        self.temp_playback_lead_s = 0.45
        self.temp_playback_tail_s = 0.55

    def init_model(self, language, enable_llm, llm_model):
        if self.pipeline is not None:
            if self.pipeline.language == language:
                self.status_text = "实时 ASR 模型已加载，等待录音"
                return self.status_text, self.render_html()
            self.pipeline.unload()
            self.pipeline = None
        self.pipeline = RealtimePipeline(language=language, enable_llm=False)
        self.status_text = "实时 ASR 模型加载成功，等待录音"
        return self.status_text, self.render_html()

    def _ensure_settings(self, language, enable_llm, llm_model) -> bool:
        if self.pipeline is None:
            self.status_text = "请先加载模型"
            return False
        if self.pipeline.language == language:
            return True
        self.status_text = "正在按当前语言重新加载实时 ASR 模型"
        self.pipeline.unload()
        self.pipeline = RealtimePipeline(language=language, enable_llm=False)
        return True

    def mark_recording(self, language, enable_llm, llm_model):
        if self._ensure_settings(language, enable_llm, llm_model):
            if self.recording_active:
                self.status_text = "正在录音"
                return self.status_text, self.render_html()
            self._reset_session()
            self.recording_active = True
            self.status_text = "正在录音"
        return self.status_text, self.render_html()

    def _reset_session(self) -> None:
        self.session_id += 1
        self._drain(self.queue)
        self._drain(self.work_queue)
        self.segments.clear()
        self.raw_asr_items.clear()
        self.raw_asr_seq = 0
        self.raw_window_seq = 0
        self._last_render_html = ""
        self._last_raw_text = ""
        self._last_session_info = ""
        self._last_status_for_poll = ""
        self.seg_counter = 0
        self.recording.clear()
        self.received_chunks.clear()
        self.committed_until = 0.0
        self.raw_committed_until = 0.0
        self.committed_text_tail = ""
        self.recording_paused = False

    def start_file_stream_test(self, audio_path, chunk_duration, language, enable_llm, llm_model):
        """Feed an audio file into the realtime worker as if it was live input."""
        if not audio_path:
            self.status_text = "请选择用于实时测试的音频文件"
            return self.status_text, self.render_html(), self.render_raw_text(), self.session_info()
        if not self._ensure_settings(language, enable_llm, llm_model):
            return self.status_text, self.render_html(), self.render_raw_text(), self.session_info()

        try:
            if isinstance(audio_path, dict):
                raw_path = audio_path.get("path") or audio_path.get("name")
            else:
                raw_path = getattr(audio_path, "name", audio_path)
            path = Path(str(raw_path))
            audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
                sr = 16000
            audio = np.asarray(audio, dtype=np.float32)
        except Exception as exc:
            self.status_text = f"读取测试音频失败: {exc}"
            return self.status_text, self.render_html(), self.render_raw_text(), self.session_info()

        if audio.size < int(0.3 * 16000):
            self.status_text = "测试音频太短"
            return self.status_text, self.render_html(), self.render_raw_text(), self.session_info()

        self._reset_session()
        self.recording_active = True
        chunk_s = max(1.0, float(chunk_duration or 10.0))
        self.ensure_worker()
        session_id = self.session_id
        threading.Thread(
            target=self._file_stream_loop,
            args=(session_id, path.name, audio, chunk_s),
            daemon=True,
        ).start()

        total_chunks = max(1, int(np.ceil(len(audio) / (chunk_s * 16000))))
        self.status_text = f"文件流式测试已开始: {path.name}，约 {total_chunks} 个块"
        return self.status_text, self.render_html(), self.render_raw_text(), self.session_info()

    def _file_stream_loop(self, session_id: int, name: str, audio: np.ndarray, chunk_s: float) -> None:
        chunk_n = int(chunk_s * 16000)
        cid = 0
        offset = 0.0
        for lo in range(0, len(audio), chunk_n):
            if session_id != self.session_id or not self.recording_active:
                break
            while self.recording_paused and session_id == self.session_id and self.recording_active:
                time.sleep(0.1)
            if session_id != self.session_id or not self.recording_active:
                break
            chunk = audio[lo:lo + chunk_n]
            if len(chunk) < int(0.3 * 16000):
                break
            cid += 1
            self._enqueue_audio_chunk(session_id, cid, chunk.astype(np.float32), offset)
            offset += len(chunk) / 16000
            self.status_text = f"文件流式测试中: {name}，已送入第 {cid} 块"
            slept = 0.0
            while slept < chunk_s and session_id == self.session_id and self.recording_active:
                if not self.recording_paused:
                    slept += 0.1
                time.sleep(0.1)
        if session_id == self.session_id and self.recording_active:
            self.status_text = f"文件流式测试完成: {name}，可停止并保存或离线精修"

    def _enqueue_audio_chunk(self, session_id: int, cid: int, audio: np.ndarray, offset: float) -> None:
        self.recording.append(audio.astype(np.float32))
        self.work_queue.put({
            "session_id": session_id,
            "chunk_id": cid,
            "chunk_start": offset,
            "chunk_end": offset + len(audio) / 16000,
        })

    def toggle_pause(self):
        if not self.recording_active:
            self.status_text = "当前没有正在进行的实时任务"
        else:
            self.recording_paused = not self.recording_paused
            self.status_text = "已暂停" if self.recording_paused else "正在录音"
        return self.status_text, self.session_info()

    @staticmethod
    def _drain(q: queue.Queue) -> None:
        while not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                break

    def poll(self, _dummy):
        while not self.queue.empty():
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break
            if item.get("session_id", self.session_id) != self.session_id:
                continue
            if item.get("type") == "asr":
                self._store_asr_segment(item)
            elif item.get("type") == "raw_asr":
                self._store_raw_asr(item)
            elif item.get("type") == "llm":
                self._apply_llm_segments(item)
        html = self.render_html()
        raw_text = self.render_raw_text()
        info = self.session_info()
        status = self.status_text
        html_out = gr.update() if html == self._last_render_html else html
        raw_out = gr.update() if raw_text == self._last_raw_text else raw_text
        info_out = gr.update() if info == self._last_session_info else info
        status_out = gr.update() if status == self._last_status_for_poll else status
        self._last_render_html = html
        self._last_raw_text = raw_text
        self._last_session_info = info
        self._last_status_for_poll = status
        return html_out, raw_out, info_out, status_out

    def _store_asr_segment(self, item: dict) -> None:
        sid = self.seg_counter
        self.seg_counter += 1
        start = float(item.get("start", 0.0))
        end = float(item.get("end", 0.0))
        self.segments[sid] = {
            "phase": "asr",
            "speaker": item.get("speaker", "LIVE"),
            "start": start,
            "end": end,
            "playback_start": float(item.get("playback_start", start)),
            "playback_end": float(item.get("playback_end", end)),
            "text": item.get("text", ""),
            "intent": item.get("intent", []),
            "llm_text": "",
            "llm_intent": [],
            "chunk_id": item.get("chunk_id", 0),
            "chunk_start": 0.0,
        }

    def _store_raw_asr(self, item: dict) -> None:
        text = item.get("text", "")
        if not str(text).strip():
            return
        start = float(item.get("start", 0.0))
        end = float(item.get("end", 0.0))
        self.raw_asr_seq += 1
        self.raw_asr_items[self.raw_asr_seq] = {
            "chunk_id": self.raw_asr_seq,
            "start": start,
            "end": end,
            "text": text,
        }

    def _apply_llm_segments(self, item: dict) -> None:
        for lseg in item.get("segments", []):
            lstart = float(lseg.get("start", 0))
            best_sid = None
            best_dist = float("inf")
            for sid, seg in self.segments.items():
                if seg["phase"] == "asr":
                    dist = abs(seg["start"] - lstart)
                    if dist < best_dist and dist < 2.0:
                        best_dist = dist
                        best_sid = sid
            if best_sid is not None:
                seg = self.segments[best_sid]
                seg["llm_text"] = lseg.get("text", "")
                seg["llm_intent"] = lseg.get("intent", [])
                if seg["llm_text"] != seg["text"]:
                    seg["phase"] = "corrected"

    def render_html(self) -> str:
        if not self.segments and not self.recording:
            return "<p style='color:#888'>等待录音...</p>"
        parts = [
            "<p style='color:#b9770e;font-size:0.9em'>"
            "实时转写为轻量 ASR 临时稿：不做说话人分离和 LLM 纠错；停止后可用离线精修生成最终多说话人结果。"
            "</p>"
        ]
        can_play = bool(self.recording) and not self.recording_active
        if self.recording_active:
            parts.append(
                "<p style='color:#999;font-size:0.85em'>"
                "录音/文件流进行中仅预览文本；停止并保存后可点击播放对应音频。"
                "</p>"
            )
        include_audio = can_play
        if include_audio:
            try:
                full_audio = np.concatenate(self.recording)
                b64 = self.audio_to_base64(full_audio.astype(np.float32), 16000)
                parts.append(f"<audio id='rt-audio-full' src='data:audio/wav;base64,{b64}' style='display:none'></audio>")
            except Exception:
                pass
        ordered = sorted(
            self.segments.items(),
            key=lambda kv: (float(kv[1].get("start", 0.0)), float(kv[1].get("end", 0.0)), kv[0]),
        )
        for _, seg in ordered:
            play_start = float(seg.get("playback_start", seg["start"]))
            play_end = float(seg.get("playback_end", seg["end"]))
            onclick = (
                f"playSeg({play_start:.3f},{play_end:.3f},'rt-audio-full',this)"
                if can_play else
                "void(0)"
            )
            parts.append(self.render_seg_html(
                {**seg, "start": seg["start"], "end": seg["end"]},
                audio_id="rt-audio-full" if can_play else None,
                onclick=onclick,
            ))
        return "\n".join(parts)

    def render_raw_text(self) -> str:
        if not self.raw_asr_items:
            return "等待 ASR 直接结果..."
        parts = ["直接结果来自当前滑动窗口，延迟更低，但可能重复或随后被稳定结果修正。"]
        items = sorted(self.raw_asr_items.values(), key=lambda x: x.get("chunk_id", 0))
        for item in items:
            text = str(item.get("text", "")).strip()
            if not text:
                continue
            start = float(item.get("start", 0.0))
            end = float(item.get("end", start))
            parts.append(f"[{self._fmt_time(start)} - {self._fmt_time(end)}] {text}")
        return "\n\n".join(parts)

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        m = int(seconds // 60)
        s = seconds - m * 60
        return f"{m:02d}:{s:04.1f}"

    def session_info(self) -> str:
        pending = self.work_queue.qsize()
        suffix = f" 队列:{pending}" if pending else ""
        rec_s = sum(len(c) for c in self.recording) / 16000
        return f"段:{len(self.segments)} 录音:{rec_s:.0f}s{suffix}"

    def set_output_dir(self, output_dir: str | Path | None) -> None:
        if output_dir:
            self.output_dir = Path(output_dir).expanduser()

    def stop(self, output_dir: str | None = None):
        self.set_output_dir(output_dir)
        self.recording_active = False
        self.recording_paused = False
        self.status_text = "已停止，正在保存录音"
        time.sleep(0.8)
        saved = ""
        if self.recording:
            full = np.concatenate(self.recording)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / f"recording_{time.strftime('%Y%m%d_%H%M%S')}.wav"
            sf.write(str(path), full.astype(np.float32), 16000)
            self.last_recording_path = str(path)
            saved = f"录音已保存: {path}"
        html = self.render_html()
        if saved:
            html = f"<p style='color:green'>{saved}</p>" + html
        self.status_text = f"已停止。{saved}。实时 ASR 模型仍已加载，可直接再次开始录音" if saved else "已停止。实时 ASR 模型仍已加载"
        return self.status_text, html, self.session_info()

    def discard(self):
        self.session_id += 1
        self.recording_active = False
        self.recording_paused = False
        self._drain(self.queue)
        self._drain(self.work_queue)
        self.segments.clear()
        self.raw_asr_items.clear()
        self.raw_asr_seq = 0
        self.raw_window_seq = 0
        self.seg_counter = 0
        self.recording.clear()
        self.received_chunks.clear()
        self.committed_until = 0.0
        self.raw_committed_until = 0.0
        self.committed_text_tail = ""
        self.last_recording_path = ""
        self.status_text = "已丢弃本次录音和实时结果，实时 ASR 模型仍已加载"
        return self.status_text, "<p style='color:#888'>等待录音...</p>", "已丢弃"

    def refine_last_recording(self, language, enable_llm, llm_model, output_dir: str | None = None):
        self.set_output_dir(output_dir)
        if not self.last_recording_path:
            self.status_text = "请先停止并保存一段录音"
            return self.status_text, self.render_html(), "无可精修录音"
        rec_path = Path(self.last_recording_path)
        if not rec_path.exists():
            self.status_text = f"录音文件不存在: {rec_path}"
            return self.status_text, self.render_html(), "录音文件不存在"
        self._ensure_settings(language, enable_llm, llm_model)
        self.status_text = "正在离线精修本次录音"
        llm_ov = {"enabled": enable_llm} if enable_llm else {
            "enabled": False, "correction": False, "consistency": False, "intent_tagging": False,
        }
        try:
            result = self.run_pipeline(
                str(rec_path), language=language, max_duration_s=None,
                llm_overrides=llm_ov, llm_model=llm_model if llm_model else None,
            )
        except Exception as exc:
            self.status_text = f"离线精修失败: {exc}"
            return self.status_text, self.render_html(), "离线精修失败"
        result["file"] = str(rec_path)
        result["name"] = rec_path.name
        if self.on_refined:
            self.on_refined(result, rec_path)
        out_path = self.output_dir / f"refined_{rec_path.stem}.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "file": str(rec_path),
                "segments": result.get("segments", []),
                "timing": result.get("timing", {}),
                "num_speakers": result.get("num_speakers"),
                "meeting_summary": result.get("meeting_summary", {}),
            }, f, ensure_ascii=False, indent=2)
        html = self._render_refined_html(result, rec_path, out_path)
        self.status_text = "离线精修完成。实时 ASR 模型仍已加载，可继续实时录音"
        return self.status_text, html, f"离线精修: {len(result.get('segments', []))} 段 · 保存: {out_path}"

    def _render_refined_html(self, result: dict, rec_path: Path, out_path: Path) -> str:
        try:
            audio_data, audio_sr = sf.read(str(rec_path))
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            audio_tag = (
                f"<audio id='rt-refined-audio' src='data:audio/wav;base64,{self.audio_to_base64(audio_data.astype(np.float32), audio_sr)}' "
                f"preload='metadata' style='display:none'></audio>"
            )
        except Exception:
            audio_tag = ""
        seg_html = "\n".join(
            self.render_seg_html(seg, audio_id="rt-refined-audio", file_index=0)
            for seg in result.get("segments", [])
        )
        return (
            f"<p style='color:green'>已生成离线精修稿，保存: {out_path}</p>"
            f"{audio_tag}"
            f"<details open style='margin:8px 0;border:1px solid #f0d0a0;border-radius:6px;padding:8px'>"
            f"<summary style='cursor:pointer;font-weight:bold;color:#e67e22'>离线精修结果 — {rec_path.name}</summary>"
            f"{seg_html}</details>"
        )

    async def ws_handler(self, websocket) -> None:
        try:
            async for message in websocket:
                try:
                    msg = json.loads(message)
                except Exception:
                    continue
                if msg.get("type") == "init":
                    await websocket.send(json.dumps({"type": "ready", "sample_rate": 16000, "chunk_duration": msg.get("chunk_duration", 10)}))
                elif msg.get("type") == "audio":
                    await self._handle_audio_message(websocket, msg)
        except Exception:
            pass

    async def _handle_audio_message(self, websocket, msg: dict) -> None:
        b64 = msg.get("data", "")
        if not b64:
            return
        wav_bytes = base64.b64decode(b64)
        with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
            sr = wf.getframerate()
            nch = wf.getnchannels()
            raw = wf.readframes(wf.getnframes())
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if nch > 1:
            audio = audio.reshape(-1, nch).mean(axis=1)
        if audio.size < sr * 0.3:
            return
        cid = int(msg.get("chunk_id", 0))
        key = (self.session_id, cid)
        if key in self.received_chunks:
            await websocket.send(json.dumps({"type": "duplicate", "chunk_id": cid}, ensure_ascii=False))
            return
        self.received_chunks.add(key)
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=16000)
        session_offset = sum(len(c) for c in self.recording) / 16000
        self.ensure_worker()
        self._enqueue_audio_chunk(self.session_id, cid, audio.astype(np.float32), session_offset)
        await websocket.send(json.dumps({"type": "queued", "chunk_id": cid, "queue": self.work_queue.qsize()}, ensure_ascii=False))

    def ensure_worker(self) -> None:
        if self.worker_started:
            return
        self.worker_started = True
        threading.Thread(target=self.worker_loop, daemon=True).start()

    def worker_loop(self) -> None:
        while True:
            item = self.work_queue.get()
            try:
                self._process_window(item)
            except Exception as exc:
                self.status_text = f"实时处理失败: {exc}"
                print(f"[RT] worker error: {exc}")
            finally:
                self.work_queue.task_done()

    def _process_window(self, item: dict) -> None:
        pipeline = self.pipeline
        if pipeline is None:
            return
        session_id = item.get("session_id", self.session_id)
        if session_id != self.session_id or not self.recording:
            return
        cid = item["chunk_id"]
        full_audio = np.concatenate(self.recording).astype(np.float32)
        available_end = min(float(item.get("chunk_end", 0.0)), len(full_audio) / 16000.0)
        stable_until = max(0.0, available_end - self.stable_lag_s)
        window_start = max(0.0, available_end - self.window_s)
        window_audio = full_audio[int(window_start * 16000): int(available_end * 16000)]
        if window_audio.size < int(0.5 * 16000):
            return
        self.status_text = f"正在处理滑动窗口：块 {cid}"
        segs = pipeline.process_fast(window_audio)
        if session_id != self.session_id:
            return
        raw_item = self._raw_increment(segs, window_start, available_end)
        if raw_item:
            self.queue.put({
                "type": "raw_asr",
                "session_id": session_id,
                "chunk_id": cid,
                **raw_item,
            })
        if stable_until > self.committed_until + 0.05:
            commit, new_until = self._stable_segments(segs, window_start, stable_until)
            for seg in commit:
                self.queue.put({"type": "asr", "session_id": session_id, "chunk_id": cid, **seg})
            if commit:
                self.committed_until = max(self.committed_until, new_until)
        self.status_text = "正在录音" if self.recording_active else "已停止，实时 ASR 模型仍已加载"

    def _raw_increment(self, segs: list[dict], window_start: float, available_end: float) -> dict | None:
        """Return the current low-latency window text.

        Raw ASR is intentionally shown as a window history, so users can see
        every decoding pass. Stable transcript de-duplicates separately.
        """
        texts: list[str] = []
        item_start: float | None = None
        item_end = window_start

        for seg in segs:
            local_start = float(seg.get("start", 0.0))
            local_end = float(seg.get("end", local_start))
            g_start = local_start + window_start
            g_end = local_end + window_start
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            if item_start is None:
                item_start = g_start
            item_end = max(item_end, g_end)
            texts.append(text)

        if not texts or item_start is None:
            return None
        if item_end <= self.raw_committed_until + 0.05:
            return None
        text = " ".join(texts)
        start = max(item_start, self.raw_committed_until)
        if item_start < self.raw_committed_until - 0.05:
            text = self._drop_repeated_prefix(text, self._recent_raw_text())
            if not text:
                return None
        self.raw_window_seq += 1
        self.raw_committed_until = max(self.raw_committed_until, item_end)
        return {
            "start": round(start, 3),
            "end": round(item_end, 3),
            "text": text,
        }

    def _stable_segments(self, segs: list[dict], window_start: float, stable_until: float) -> tuple[list[dict], float]:
        commit: list[dict] = []
        new_until = self.committed_until
        max_boundary_overrun_s = max(1.0, self.stable_lag_s)
        for seg in segs:
            local_start = float(seg.get("start", 0.0))
            local_end = float(seg.get("end", local_start))
            g_start = local_start + window_start
            g_end = local_end + window_start
            text = str(seg.get("text", "")).strip()
            dur = max(0.0, g_end - g_start)
            was_clamped = False
            if not text:
                continue
            if g_start < self.committed_until - 0.05:
                # This segment mostly belongs to an already committed window.
                # Do not emit it again with a shifted overlapping timestamp.
                # If it is a boundary-crossing continuation, commit only when
                # there is enough new tail and clamp the display/playback start
                # to the committed boundary. The text is kept as-is because this
                # is a temporary transcript; the final offline pass handles exact
                # word-level trimming.
                new_tail = g_end - self.committed_until
                if new_tail < self.min_commit_overlap_s:
                    continue
                g_start = self.committed_until
                was_clamped = True
            if g_end <= self.committed_until + 0.05:
                continue
            if g_start >= stable_until - 0.05:
                continue
            if g_end > stable_until:
                # Do not publish a partial ASR segment. If we clamp the end
                # while keeping the full text, the segment's tail audio moves
                # into the next clickable range and all following playback is
                # shifted by that leftover prefix. However, realtime ASR often
                # ends a valid segment at the current window tail, so allow a
                # small stable-lag overrun and publish the full segment instead
                # of cutting it.
                if g_end - stable_until > max_boundary_overrun_s:
                    continue
            if dur < self.min_short_seg_s and len(text) <= 2:
                continue
            item = dict(seg)
            if was_clamped:
                text = self._drop_recent_repeated_prefix(text)
                if not text:
                    continue
                item["text"] = text
            item["start"] = round(g_start, 3)
            item["end"] = round(g_end, 3)
            play_start = float(seg.get("playback_start", local_start)) + window_start
            # Temporary realtime transcript must prefer coverage over tight
            # boundaries.  Segment-level timestamps are not word-aligned here,
            # so a small hidden lead/tail pad avoids dropping boundary audio.
            play_start = min(play_start, g_start) - self.temp_playback_lead_s
            item["playback_start"] = round(max(0.0, play_start), 3)
            play_end = max(float(seg.get("playback_end", local_end)) + window_start, g_end)
            play_end += self.temp_playback_tail_s
            item["playback_end"] = round(play_end, 3)
            commit.append(item)
            self.committed_text_tail = (self.committed_text_tail + text)[-500:]
            new_until = max(new_until, g_end)
        return commit, new_until

    def _recent_raw_text(self) -> str:
        return "".join(
            str(item.get("text", ""))
            for item in sorted(
                self.raw_asr_items.values(),
                key=lambda x: int(x.get("chunk_id", 0)),
            )[-3:]
        )

    def _drop_recent_repeated_prefix(self, text: str) -> str:
        """Trim text already visible in recent stable segments.

        Sliding-window ASR can return a long segment that starts before the
        committed boundary. We cannot word-align in realtime, but removing a
        repeated prefix keeps the temporary transcript readable.
        """
        text = text.strip()
        if not text:
            return text
        recent = "".join(
            str(seg.get("text", ""))
            for _, seg in sorted(
                self.segments.items(),
                key=lambda kv: float(kv[1].get("end", 0.0)),
            )[-3:]
        )
        recent = (recent + self.committed_text_tail)[-800:]
        if not recent:
            return text

        return self._drop_repeated_prefix(text, recent)

    @staticmethod
    def _drop_repeated_prefix(text: str, recent: str) -> str:
        text = text.strip()
        if not text or not recent:
            return text

        def norm(s: str) -> str:
            import re
            return re.sub(r"[\s，。！？、；：,.!?;:\"'“”‘’（）()\[\]{}\-—_…]+", "", s).lower()

        n_recent = norm(recent)
        n_text = norm(text)
        if not n_text:
            return ""
        best = 0
        max_len = min(len(n_recent), len(n_text))
        for size in range(max_len, 3, -1):
            if n_recent.endswith(n_text[:size]):
                best = size
                break
        if best <= 0:
            return text

        consumed = 0
        raw_idx = 0
        import re
        while raw_idx < len(text) and consumed < best:
            if not re.match(r"[\s，。！？、；：,.!?;:\"'“”‘’（）()\[\]{}\-—_…]", text[raw_idx]):
                consumed += 1
            raw_idx += 1
        return text[raw_idx:].strip()

    def start_ws_server(self, host: str, port: int) -> None:
        try:
            import websockets as ws_lib
        except ImportError:
            print("[WARN] websockets 未安装")
            return

        async def serve():
            async with ws_lib.serve(self.ws_handler, host, port, max_size=4 * 1024 * 1024):
                print(f"[WS] ws://{host}:{port}")
                await asyncio.Future()

        def exception_handler(loop, context):
            exc = context.get("exception")
            if isinstance(exc, ConnectionResetError) and getattr(exc, "winerror", None) == 10054:
                return
            loop.default_exception_handler(context)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.set_exception_handler(exception_handler)
        try:
            loop.run_until_complete(serve())
        except Exception as exc:
            print(f"[WS] 错误: {exc}")
