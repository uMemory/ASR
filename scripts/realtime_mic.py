"""实时麦克风语音转写 —— 流式识别说话人、内容，LLM 上下文实时纠错。

用法:
    python scripts/realtime_mic.py                        # 默认 10s 块
    python scripts/realtime_mic.py --chunk 15             # 15 秒块
    python scripts/realtime_mic.py --device 1             # 指定麦克风设备
    python scripts/realtime_mic.py --list-devices         # 列出音频设备
    python scripts/realtime_mic.py --context 20           # 传递前 20 段作为 LLM 上下文

架构:
    录音线程 ──→ 音频缓冲队列 ──→ 主线程定时取块
                                    │
                                    ├─ Pyannote 说话人分离
                                    ├─ Whisper ASR 识别
                                    ├─ VAD AND 对齐
                                    └─ LLM 上下文纠错 + 意图标注
                                       │
                                       └─ 实时打印到屏幕
"""
from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import sounddevice as sd
import torch


# ═══════════════════════════════════════════════════════════════════════════
# 音频捕获
# ═══════════════════════════════════════════════════════════════════════════

class AudioCapture:
    """后台录音线程，持续将音频块放入队列。"""

    def __init__(self, sample_rate: int = 16000, device: int | None = None):
        self.sample_rate = sample_rate
        self.device = device
        self._queue: queue.Queue = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def _callback(self, indata: np.ndarray, frames: int, _time, _status) -> None:
        """sounddevice 回调：每 blocksize 帧调用一次。"""
        if self._running:
            # 转单声道 + float32
            if indata.ndim > 1:
                audio = indata.mean(axis=1).astype(np.float32)
            else:
                audio = indata.astype(np.float32)
            self._queue.put(audio.copy())

    def start(self) -> None:
        """启动录音线程。"""
        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            device=self.device,
            channels=1,
            callback=self._callback,
            blocksize=int(self.sample_rate * 0.2),  # 200ms blocks
        )
        self._stream.start()

    def stop(self) -> None:
        """停止录音。"""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def get_chunk(self, duration_s: float) -> np.ndarray:
        """从队列中取出 duration_s 秒的音频。

        如果队列为空，等待新数据。返回 float32 mono numpy 数组。
        """
        needed = int(self.sample_rate * duration_s)
        chunks: list[np.ndarray] = []
        collected = 0

        while collected < needed:
            try:
                chunk = self._queue.get(timeout=0.5)
                chunks.append(chunk)
                collected += len(chunk)
            except queue.Empty:
                if not self._running:
                    break

        if not chunks:
            return np.array([], dtype=np.float32)

        audio = np.concatenate(chunks)
        return audio[:needed]


# ═══════════════════════════════════════════════════════════════════════════
# 实时管线
# ═══════════════════════════════════════════════════════════════════════════

class RealtimePipeline:
    """复用模型后端，对音频块执行 Diarization → ASR → Alignment → LLM。"""

    def __init__(self, language: str = "zh", profile: str | None = None,
                 context_segments: int = 15):
        from src.utils.config import get_model_config, load_config

        self.language = language
        self.context_segments = context_segments
        mcfg = get_model_config(profile)
        lcfg = load_config("languages")
        self.device = mcfg["device"]
        self.lang_cfg = lcfg.get(language, {})

        # ── 加载后端（只加载一次） ──────────────────────────────
        print("[初始化] 加载说话人分离模型...", flush=True, end=" ")
        from src.diarization import load_diarization_backend
        dia_cfg = {**mcfg["diarization"], "device": self.device}
        self.dia_backend = load_diarization_backend(dia_cfg)
        self.dia_backend.load()
        print("OK", flush=True)

        print("[初始化] 加载 ASR 模型...", flush=True, end=" ")
        from src.asr import load_asr_backend
        asr_cfg = {**mcfg["asr"], "device": self.device}
        self.asr_backend = load_asr_backend(asr_cfg)
        self.asr_backend.load()
        print("OK", flush=True)

        print("[初始化] 加载 LLM 适配器...", flush=True, end=" ")
        from src.llm import load_llm_adapter, llm_correct_segments, llm_tag_intents
        self.llm_adapter = load_llm_adapter()
        self._llm_correct = llm_correct_segments
        self._llm_tag = llm_tag_intents
        print(f"OK ({self.llm_adapter.model_name})", flush=True)

        # 上下文累积
        self._history: list[dict] = []

        print(f"[就绪] 开始实时转写 (语言={language}, 上下文={context_segments}段)\n",
              flush=True)

    def process_chunk_fast(self, waveform: np.ndarray) -> list[dict]:
        """Phase 1: 快速 ASR + 对齐（即刻返回，不等 LLM）。"""
        if waveform.size == 0:
            return []

        sr = 16000

        # ── Stage 2: Diarization ──────────────────────────────
        dia_result = self.dia_backend.diarize(waveform, sample_rate=sr)

        # ── Stage 3: ASR ─────────────────────────────────────
        initial_prompt = self.lang_cfg.get("asr_initial_prompt")
        asr_result = self.asr_backend.transcribe(
            waveform, language=self.language, initial_prompt=initial_prompt,
        )

        # ── Stage 4: Alignment ────────────────────────────────
        from src.alignment import align_segments
        merged = align_segments(dia_result["segments"], asr_result["segments"])

        return merged

    def process_chunk_llm(self, merged: list[dict]) -> list[dict]:
        """Phase 2: LLM 上下文纠错 + 意图标注（允许延迟）。

        使用累积的历史段作为上下文传给纠错器。
        """
        if not merged:
            return []

        # 纠错器最多处理 15 段/批次，确保上下文+当前段在同一个批次内
        _MAX_BATCH = 15
        max_ctx = max(0, min(self.context_segments, _MAX_BATCH - len(merged)))
        ctx = self._history[-max_ctx:] if self._history and max_ctx > 0 else []
        segments_with_context = ctx + merged

        try:
            corrected_all = self._llm_correct(
                self.llm_adapter, segments_with_context, self.language,
            )
            # 仅取最后 len(merged) 个段（当前块），忽略对历史段的修正
            new_segments = corrected_all[-len(merged):]
        except Exception as e:
            print(f"  [WARN] LLM 纠错失败: {e}", flush=True)
            new_segments = merged

        # 意图标注（只标当前块）
        try:
            intent_labels = self.lang_cfg.get("intent_labels")
            new_segments = self._llm_tag(
                self.llm_adapter, new_segments, self.language, intent_labels,
            )
        except Exception as e:
            print(f"  [WARN] 意图标注失败: {e}", flush=True)

        # 累积历史
        self._history.extend(new_segments)
        # 限制历史长度
        if len(self._history) > 200:
            self._history = self._history[-100:]

        return new_segments

    def unload(self) -> None:
        """释放 GPU 资源。"""
        self.dia_backend.unload()
        self.asr_backend.unload()
        del self.dia_backend, self.asr_backend, self.llm_adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════════════
# 显示
# ═══════════════════════════════════════════════════════════════════════════

def print_segment(seg: dict, global_offset: float) -> None:
    """格式化打印 ASR 段（即刻输出）。"""
    start = seg.get("start", 0) + global_offset
    end = seg.get("end", 0) + global_offset
    speaker = seg.get("speaker", "?")
    text = seg.get("text", "")
    intent = seg.get("intent", [])

    ts = f"[{int(start // 60):02d}:{start % 60:04.1f} → {int(end // 60):02d}:{end % 60:04.1f}]"
    line = f"  ⚡ {ts}  {speaker}: {text}"
    if intent:
        line += f"  [{', '.join(intent)}]"
    print(line, flush=True)


def print_segment_corrected(seg: dict, global_offset: float) -> None:
    """格式化打印 LLM 纠错后的段。"""
    start = seg.get("start", 0) + global_offset
    end = seg.get("end", 0) + global_offset
    speaker = seg.get("speaker", "?")
    text = seg.get("text", "")
    intent = seg.get("intent", [])

    ts = f"[{int(start // 60):02d}:{start % 60:04.1f} → {int(end // 60):02d}:{end % 60:04.1f}]"
    line = f"  ✅ {ts}  {speaker}: {text}"
    if intent:
        line += f"  [{', '.join(intent)}]"
    print(line, flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════

def main() -> int:
    p = argparse.ArgumentParser(description="实时麦克风语音转写")
    p.add_argument("--chunk", type=float, default=10.0,
                   help="每个处理块的时长（秒），默认 10")
    p.add_argument("--device", type=int, default=None, help="音频输入设备 ID")
    p.add_argument("--list-devices", action="store_true", help="列出可用音频设备")
    p.add_argument("--language", default="zh", help="语言代码 (zh/en)")
    p.add_argument("--profile", default=None, help="配置 profile")
    p.add_argument("--context", type=int, default=15,
                   help="传递给 LLM 的历史段数（默认15）")
    p.add_argument("--no-llm", action="store_true", help="跳过 LLM 后处理")
    p.add_argument("--save-audio", default=None,
                   help="保存完整录音到指定路径 (如 outputs/recording.wav)")
    args = p.parse_args()

    # 列出设备
    if args.list_devices:
        print("可用音频输入设备:", flush=True)
        devices = sd.query_devices()
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                print(f"  [{i}] {d['name']} (in:{d['max_input_channels']}, "
                      f"sr:{int(d['default_samplerate'])})", flush=True)
        return 0

    print("═" * 66)
    print("  实时多说话人语音转写系统")
    print(f"  块长度: {args.chunk}s  |  语言: {args.language}  |  "
          f"LLM: {'关闭' if args.no_llm else '开启（含上下文纠错）'}")
    print("  按 Ctrl+C 停止")
    print("═" * 66)

    # ── 初始化管线（一次性加载所有模型） ────────────────────
    pipeline = RealtimePipeline(
        language=args.language,
        profile=args.profile,
        context_segments=args.context,
    )

    # ── 启动录音 ────────────────────────────────────────────
    capture = AudioCapture(sample_rate=16000, device=args.device)
    capture.start()
    print("[录音] 已开始\n", flush=True)

    # ── 主循环 ──────────────────────────────────────────────
    chunk_idx = 0
    global_time = 0.0
    all_audio: list[np.ndarray] = []  # 累积全部录音（供保存）

    try:
        while True:
            chunk_start = time.time()

            # 取音频块
            audio = capture.get_chunk(args.chunk)
            if audio.size == 0:
                time.sleep(0.1)
                continue

            # 保存音频
            if args.save_audio:
                all_audio.append(audio.copy())

            chunk_idx += 1
            actual_dur = len(audio) / 16000
            print(f"\n── 块 #{chunk_idx} ({actual_dur:.1f}s) "
                  f"[累计 {global_time + actual_dur:.0f}s] ──", flush=True)

            # ── Phase 1: 快速 ASR（即刻输出）──
            t0 = time.time()
            asr_segments = pipeline.process_chunk_fast(audio)
            asr_time = time.time() - t0

            if asr_segments:
                for seg in asr_segments:
                    print_segment(seg, global_time)
            else:
                print("  (未检测到语音)", flush=True)

            print(f"  ⚡ ASR 耗时: {asr_time:.1f}s", flush=True)

            # ── Phase 2: LLM 纠错（允许延迟）──
            if asr_segments and not args.no_llm:
                t_llm = time.time()
                corrected = pipeline.process_chunk_llm(asr_segments)
                llm_time = time.time() - t_llm

                if corrected:
                    for seg in corrected:
                        print_segment_corrected(seg, global_time)

                # 累积历史（用原始 ASR 段，LLM 内部已累积 corrected）
                # history 由 process_chunk_llm 内部管理
                print(f"  ✅ LLM 纠错耗时: {llm_time:.1f}s", flush=True)

            global_time += actual_dur

            # 如果处理时间超过块时长，提示延迟
            overhead = time.time() - chunk_start
            if overhead > args.chunk:
                print(f"  ⚠ 处理延迟 {overhead - args.chunk:.1f}s (块={args.chunk}s)",
                      flush=True)

    except KeyboardInterrupt:
        print("\n\n[停止] 收到中断信号，正在清理...", flush=True)

    finally:
        capture.stop()

        # 保存录音
        if args.save_audio and all_audio:
            full = np.concatenate(all_audio)
            import soundfile as sf
            save_path = Path(args.save_audio)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(save_path), full.astype(np.float32), 16000)
            print(f"[保存] 录音已保存到: {save_path} ({len(full)/16000:.0f}s)", flush=True)

        pipeline.unload()
        print("[清理] 资源已释放", flush=True)

    # 输出完整会话摘要
    print("\n" + "═" * 66)
    print(f"  会话结束。共 {chunk_idx} 个块，{len(pipeline._history)} 段发言。")
    print("═" * 66)

    return 0


if __name__ == "__main__":
    sys.exit(main())
