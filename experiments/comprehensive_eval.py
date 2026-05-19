"""Comprehensive evaluation: DER + WER/CER + retrieval + ablation.

评测策略：
  DER: pyannote.metrics，参考与假设裁剪到同一时间窗口后对比
  WER/CER: 段时间重叠匹配（IoU），配对后逐段计算 → 加权平均
  时序: pipeline.run() 返回的 timing 字典

用法:
    python experiments/comprehensive_eval.py                      # 全量 AISHELL-4
    python experiments/comprehensive_eval.py --subset 5           # 前 5 文件快测
    python experiments/comprehensive_eval.py --no-llm             # 消融：无 LLM
    python experiments/comprehensive_eval.py --retrieval          # 含检索评测
    python experiments/comprehensive_eval.py --dataset alimeeting # AliMeeting
"""
from __future__ import annotations

import argparse, json, sys, time, warnings
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from pyannote.core import Annotation, Segment
from pyannote.metrics.diarization import DiarizationErrorRate
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.utils.config import project_root

warnings.filterwarnings("ignore")
console = Console()


# ══════════════════════════════════════════════════════════════
# TextGrid → 参考标注
# ══════════════════════════════════════════════════════════════

def parse_textgrid(tg_path: Path, max_dur: float | None = None
                   ) -> tuple[Annotation, list[dict]]:
    """解析 AISHELL-4 TextGrid → (pyannote Annotation, [{start,end,speaker,text}])。

    - 过滤空白和占位符 ``<%>`` / ``<$>``
    - 多个说话人 tier 各自独立解析
    - 按 start 排序后返回
    """
    import textgrid as tg_lib
    tg = tg_lib.TextGrid.fromFile(str(tg_path))
    annotation = Annotation()
    ref_segs: list[dict] = []

    for tier in tg.tiers:
        spk = tier.name
        for iv in tier.intervals:
            text = iv.mark.strip()
            # 过滤空文本和占位符
            if not text or text in ("", "<%>", "<$>", "%", "$"):
                continue
            s = float(iv.minTime); e = float(iv.maxTime)
            if max_dur is not None and s >= max_dur:
                continue
            if max_dur is not None:
                e = min(e, max_dur)
            if e <= s:
                continue
            annotation[Segment(s, e)] = spk
            ref_segs.append({"start": s, "end": e, "speaker": spk, "text": text})

    ref_segs.sort(key=lambda x: x["start"])
    return annotation, ref_segs


# ══════════════════════════════════════════════════════════════
# 段时间重叠匹配（用于 WER 配对）
# ══════════════════════════════════════════════════════════════

def compute_wer(ref_segs: list[dict], hyp_segs: list[dict]) -> dict[str, float]:
    """窗口内文本拼接后计算 WER/CER。

    将参考段和假设段分别按时间排序、拼接文本，用 jiwer 计算。
    同时返回匹配信息供参考。
    """
    import jiwer

    # 参考文本：所有参考段按时间拼接
    ref_sorted = sorted(ref_segs, key=lambda x: x["start"])
    ref_text = " ".join(r["text"].strip() for r in ref_sorted if r["text"].strip())

    # 假设文本：所有假设段按时间拼接
    hyp_sorted = sorted(hyp_segs, key=lambda x: x["start"])
    hyp_text = " ".join(
        (h.get("llm_text") or h.get("text", "")).strip()
        for h in hyp_sorted
    ).strip()

    if not ref_text and not hyp_text:
        return {"wer": 0.0, "cer": 0.0, "ref_chars": 0, "hyp_chars": 0}

    if not ref_text:
        return {"wer": 1.0, "cer": 1.0, "ref_chars": 0, "hyp_chars": len(hyp_text)}

    if not hyp_text:
        return {"wer": 1.0, "cer": 1.0, "ref_chars": len(ref_text), "hyp_chars": 0}

    try:
        wer = float(jiwer.wer(ref_text, hyp_text))
    except Exception:
        wer = 0.0 if ref_text == hyp_text else 1.0
    try:
        cer = float(jiwer.cer(ref_text, hyp_text))
    except Exception:
        cer = 0.0 if ref_text == hyp_text else 1.0

    return {
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "ref_chars": len(ref_text),
        "hyp_chars": len(hyp_text),
        "ref_first_60": ref_text[:60],
        "hyp_first_60": hyp_text[:60],
    }


# ══════════════════════════════════════════════════════════════
# 指标收集
# ══════════════════════════════════════════════════════════════

class Collector:
    def __init__(self): self.items: list[dict] = []

    def add(self, file: str, m: dict): m["file"] = file; self.items.append(m)

    def agg(self) -> dict:
        if not self.items: return {}
        keys = ["der", "der_collar", "wer", "cer",
                "total_s", "asr_s", "dia_s", "llm_s"]
        r: dict = {"n": len(self.items)}
        for k in keys:
            vals = [it[k] for it in self.items if k in it and it[k] is not None]
            if vals:
                r[f"{k}_avg"] = round(float(np.mean(vals)), 4)
                r[f"{k}_std"] = round(float(np.std(vals)), 4) if len(vals) > 1 else 0.0
        return r


# ══════════════════════════════════════════════════════════════
# 检索评测
# ══════════════════════════════════════════════════════════════

def eval_retrieval(segments: list[dict], encoder, index_path: Path) -> dict:
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    from src.retrieval import build_index, Retriever
    t0 = time.time()
    try:
        meta, _, _ = build_index(segments, encoder, store_path=index_path)
    except Exception as e:
        return {"error": str(e)}
    idx_t = time.time() - t0
    ret = Retriever(index_path=index_path, encoder=encoder)
    spks = sorted({s.get("speaker","") for s in segments if s.get("speaker","")})
    intents = set()
    for s in segments:
        for i in (s.get("intent") or []): intents.add(i)
    queries = []
    if segments:
        queries.append({"q": segments[0].get("text","")[:20], "type": "semantic"})
    for spk in spks[:3]:
        queries.append({"q": "发言", "speaker": spk, "type": "speaker"})
    for intent in list(intents)[:3]:
        queries.append({"q": intent, "intent": [intent], "type": "intent"})
    qr = []
    for q in queries:
        t1 = time.time()
        try:
            hits = ret.search(q["q"], top_k=5, speaker=q.get("speaker"),
                              intent=q.get("intent"))
        except Exception as e:
            qr.append({"type": q["type"], "error": str(e)}); continue
        lat = round((time.time()-t1)*1000, 1)
        qr.append({"type": q["type"], "query": q["q"], "latency_ms": lat,
                    "hits": len(hits)})
    return {"index_build_s": round(idx_t, 2),
            "index_segments": meta["num_segments"],
            "avg_latency_ms": round(np.mean([x.get("latency_ms",0) for x in qr]),1) if qr else 0,
            "queries": qr}


# ══════════════════════════════════════════════════════════════
# 评测驱动
# ══════════════════════════════════════════════════════════════

def run_eval(dataset: str = "aishell4", subset: int = 0,
             max_duration: float = 60.0, llm_enabled: bool = True,
             do_retrieval: bool = False, profile: str | None = None,
             ) -> tuple[Collector, dict | None]:
    root = project_root()
    if dataset == "aishell4":
        wav_dir = root / "dataset/AISHELL-4/test/wav"
        tg_dir = root / "dataset/AISHELL-4/test/TextGrid"
        wavs = sorted(wav_dir.glob("*.flac"))
    elif dataset == "alimeeting":
        wav_dir = root / "dataset/AIMeeting/Eval_Ali/Eval_Ali_far/audio_dir"
        tg_dir = root / "dataset/AIMeeting/Eval_Ali/Eval_Ali_far/textgrid_dir"
        wavs = sorted(wav_dir.glob("*.wav"))
    else:
        raise ValueError(f"未知数据集: {dataset}")
    if subset > 0: wavs = wavs[:subset]
    if not wavs:
        console.print("[red]未找到音频文件[/red]"); return Collector(), None

    console.rule(f"[bold cyan]{dataset.upper()} 评测")
    console.print(f"文件数: {len(wavs)}  窗口: {max_duration}s  LLM: {llm_enabled}  检索: {do_retrieval}")

    col = Collector()
    retrieval: dict | None = {} if do_retrieval else None
    from src.pipeline import run as run_pipe

    encoder = None
    if do_retrieval:
        from src.retrieval.embedding import EmbeddingEncoder
        from src.utils.config import get_model_config
        mcfg = get_model_config(profile)
        encoder = EmbeddingEncoder(model_path=mcfg["embedding"]["model"],
                                   device=mcfg["device"])
        encoder.load()

    der_met = DiarizationErrorRate()
    der_met_c = DiarizationErrorRate(collar=0.250, skip_overlap=False)

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
                  console=console) as prog:
        task = prog.add_task("评测中...", total=len(wavs))
        for wav_path in wavs:
            fname = wav_path.stem
            prog.update(task, description=f"[dim]{fname}[/dim]")

            tg_path = tg_dir / f"{fname}.TextGrid"
            if not tg_path.exists():
                prog.advance(task); continue

            ref_ann, ref_segs = parse_textgrid(tg_path, max_duration)

            try:
                result = run_pipe(str(wav_path), language="zh", profile=profile,
                                  max_duration_s=max_duration,
                                  llm_overrides={"enabled": llm_enabled})
            except Exception as e:
                console.print(f"[red]{fname}: 管线失败 — {e}[/red]")
                prog.advance(task); continue

            segs = result["segments"]
            timing = result.get("timing", {})

            # ── DER ─────────────────────────────────────
            hyp_ann = Annotation()
            for seg in segs:
                spk = seg.get("speaker", "")
                if not spk:
                    continue
                hyp_ann[Segment(seg["start"], seg["end"])] = spk

            ref_crop = ref_ann.crop(Segment(0, max_duration))
            hyp_crop = hyp_ann.crop(Segment(0, max_duration))
            try:
                der = der_met(ref_crop, hyp_crop)
                der_c = der_met_c(ref_crop, hyp_crop)
            except Exception:
                der, der_c = 1.0, 1.0

            # ── WER/CER ─────────────────────────────
            ref_window = [s for s in ref_segs if s["start"] < max_duration]
            wc = compute_wer(ref_window, segs)

            # ── 记录 ─────────────────────────────────
            col.add(fname, {
                "der": round(float(der), 4),
                "der_collar": round(float(der_c), 4),
                "wer": wc["wer"],
                "cer": wc["cer"],
                "ref_chars": wc.get("ref_chars", 0),
                "hyp_chars": wc.get("hyp_chars", 0),
                "total_s": round(timing.get("total", 0), 2),
                "asr_s": round(timing.get("asr", 0), 2),
                "dia_s": round(timing.get("diarization", 0), 2),
                "llm_s": round(timing.get("llm", 0), 2),
                "num_seg": len(segs),
                "num_spk": result.get("num_speakers", 0),
                "ref_first": wc.get("ref_first_60", ""),
                "hyp_first": wc.get("hyp_first_60", ""),
            })

            # 检索（仅第一个文件）
            if do_retrieval and retrieval is not None and not retrieval:
                idx_p = root / "outputs/eval_index"
                retrieval = eval_retrieval(segs, encoder, idx_p)
                retrieval["file"] = fname

            prog.advance(task)

    if encoder:
        encoder.unload(); del encoder
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return col, retrieval


# ══════════════════════════════════════════════════════════════
# 报告
# ══════════════════════════════════════════════════════════════

def print_report(col: Collector, retrieval: dict | None,
                 dataset: str, llm: bool) -> None:
    a = col.agg()
    console.rule("[bold green]结果")
    console.print(f"数据集: {dataset}  文件: {a.get('n',0)}  LLM: {llm}")

    # DER
    t = Table(title="DER (说话人分离错误率)")
    t.add_column("指标"); t.add_column("均值"); t.add_column("标准差")
    for label, key in [("DER", "der_avg"), ("DER collar=0.25s", "der_collar_avg")]:
        m = a.get(key); s = a.get(key.replace("_avg","_std"))
        t.add_row(label, f"{m:.2%}" if m is not None else "—",
                  f"±{s:.2%}" if s else "—")
    console.print(t)

    # WER/CER
    t2 = Table(title="ASR 准确率 (窗口内文本拼接)")
    t2.add_column("指标"); t2.add_column("均值"); t2.add_column("标准差")
    for label, key in [("WER", "wer_avg"), ("CER", "cer_avg")]:
        m = a.get(key); s = a.get(key.replace("_avg","_std"))
        t2.add_row(label, f"{m:.2%}" if m is not None else "—",
                   f"±{s:.2%}" if s else "—")
    console.print(t2)

    # 时序
    t3 = Table(title="管线耗时 (秒/文件)")
    t3.add_column("阶段"); t3.add_column("均值")
    for label, key in [("总计", "total_s_avg"), ("ASR", "asr_s_avg"),
                       ("Diarization", "dia_s_avg"), ("LLM", "llm_s_avg")]:
        m = a.get(key)
        if m and m > 0: t3.add_row(label, f"{m:.1f}s")
    console.print(t3)

    # 逐文件
    t4 = Table(title="逐文件")
    t4.add_column("文件"); t4.add_column("DER"); t4.add_column("WER")
    t4.add_column("CER"); t4.add_column("耗时")
    for it in col.items:
        t4.add_row(it["file"][:28],
                   f"{it.get('der',1):.1%}", f"{it.get('wer',1):.1%}",
                   f"{it.get('cer',1):.1%}", f"{it.get('total_s',0):.0f}s")
    console.print(t4)

    # 检索
    if retrieval and not retrieval.get("error"):
        console.print()
        rt = Table(title="检索")
        rt.add_column("指标"); rt.add_column("值")
        rt.add_row("索引构建", f"{retrieval.get('index_build_s','?')}s")
        rt.add_row("段数", str(retrieval.get("index_segments","?")))
        rt.add_row("平均查询延迟", f"{retrieval.get('avg_latency_ms','?')}ms")
        console.print(rt)


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════

def main() -> int:
    p = argparse.ArgumentParser(description="ASR 综合评测")
    p.add_argument("--dataset", default="aishell4", choices=["aishell4","alimeeting"])
    p.add_argument("--subset", type=int, default=0)
    p.add_argument("--duration", type=float, default=60.0)
    p.add_argument("--no-llm", action="store_true")
    p.add_argument("--retrieval", action="store_true")
    p.add_argument("--profile", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--verbose", action="store_true", help="打印参考/假设文本片段")
    args = p.parse_args()

    console.print(Panel.fit(
        f"[bold]ASR Pipeline — 综合评测[/bold]\n"
        f"数据集: {args.dataset} | 窗口: {args.duration}s | LLM: {not args.no_llm}"
    ))

    col, ret = run_eval(dataset=args.dataset, subset=args.subset,
                        max_duration=args.duration, llm_enabled=not args.no_llm,
                        do_retrieval=args.retrieval, profile=args.profile)
    print_report(col, ret, args.dataset, not args.no_llm)

    if args.verbose and col.items:
        console.rule("[bold]参考 vs 假设文本片段")
        for it in col.items:
            console.print(f"\n[cyan]{it['file']}[/cyan]")
            console.print(f"  [dim]参考: {it.get('ref_first','?')}[/dim]")
            console.print(f"  [dim]假设: {it.get('hyp_first','?')}[/dim]")

    if args.output:
        report = {"config": {"dataset": args.dataset, "duration_s": args.duration,
                             "llm_enabled": not args.no_llm},
                  "aggregate": col.agg(), "per_file": col.items, "retrieval": ret}
        out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        console.print(f"\n[green]已保存: {out}[/green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
