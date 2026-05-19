"""Generate the final report by filling evaluation results into the template.

Reads outputs/eval_*.json and updates docs/final_report.md with actual numbers.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import project_root


def load_eval(path: str) -> dict:
    with open(project_root() / path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_pct(val: float | None) -> str:
    if val is None:
        return "—"
    return f"{val * 100:.1f}%"


def format_time(val: float | None) -> str:
    if val is None:
        return "—"
    return f"{val:.1f}s"


def generate_results_table(baseline: dict, full: dict | None) -> str:
    """Generate markdown tables for the experiments section."""
    rows = []

    # DER table
    rows.append("### 6.1 说话人分离（DER）结果\n")
    rows.append("| 配置 | DER | DER (collar=0.25s) | False Alarm | Missed | Confusion |")
    rows.append("|------|-----|--------------------|--------------|--------|-----------|")

    for label, data in [("基线（无 LLM）", baseline), ("完整管线（含 LLM）", full)]:
        if data is None:
            continue
        agg = data.get("aggregate", {})
        rows.append(
            f"| {label} | {format_pct(agg.get('avg_der'))} | "
            f"{format_pct(agg.get('avg_der_collar'))} | "
            f"{format_pct(agg.get('avg_false_alarm'))} | "
            f"{format_pct(agg.get('avg_missed_detection'))} | "
            f"{format_pct(agg.get('avg_confusion'))} |"
        )

    rows.append("")

    # WER/CER table
    rows.append("### 6.2 语音识别（WER/CER）结果\n")
    rows.append("| 配置 | WER | CER |")
    rows.append("|------|-----|-----|")

    for label, data in [("基线（无 LLM）", baseline), ("完整管线（含 LLM）", full)]:
        if data is None:
            continue
        agg = data.get("aggregate", {})
        rows.append(
            f"| {label} | {format_pct(agg.get('avg_wer'))} | "
            f"{format_pct(agg.get('avg_cer'))} |"
        )

    rows.append("")

    # Timing table
    rows.append("### 6.3 管线时序分析\n")
    rows.append("| Stage | 基线 (s) | 完整管线 (s) |")
    rows.append("|-------|---------|-------------|")

    stages = [
        ("Diarization", "avg_diarization_time"),
        ("ASR", "avg_asr_time"),
        ("LLM 后处理", "avg_llm_time"),
        ("Total", "avg_total_time"),
    ]

    b_agg = baseline.get("aggregate", {})
    f_agg = full.get("aggregate", {}) if full else {}
    for label, key in stages:
        rows.append(
            f"| {label} | {format_time(b_agg.get(key))} | "
            f"{format_time(f_agg.get(key))} |"
        )

    rows.append("")

    # Retrieval table
    if full and full.get("retrieval"):
        ret = full["retrieval"]
        rows.append("### 6.4 检索系统性能\n")
        rows.append("| 指标 | 值 |")
        rows.append("|------|------|")
        rows.append(f"| 索引构建时间 | {ret.get('index_build_time', '—')}s |")
        rows.append(f"| 索引段数 | {ret.get('index_segments', '—')} |")
        rows.append(f"| 平均查询延迟 | {ret.get('avg_latency_ms', '—')}ms |")
        rows.append("")

    return "\n".join(rows)


def main() -> int:
    root = project_root()

    # Load all eval results
    baseline_path = root / "outputs/eval_baseline_aishell4.json"
    full_path = root / "outputs/eval_full_aishell4.json"

    baseline = load_eval(str(baseline_path)) if baseline_path.exists() else None
    full = load_eval(str(full_path)) if full_path.exists() else None

    if baseline is None:
        print("[red]Baseline eval not found. Run E1 first.[/red]")
        return 1

    # Generate results section
    tables = generate_results_table(baseline, full)

    # Read report template and replace placeholder
    report_path = root / "docs/final_report.md"
    with open(report_path, "r", encoding="utf-8") as f:
        report = f.read()

    # Replace the experiments placeholder section (lines between "## 6. 实验结果" and "## 7. Demo")
    pattern = r"(## 6\. 实验结果\n).*?(\n## 7\. Demo)"
    replacement = r"\1\n" + tables + r"\2"
    report = re.sub(pattern, replacement, report, flags=re.DOTALL)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[green]Report updated: {report_path}[/green]")

    # Word count
    # Strip markdown formatting for rough word count
    text_only = re.sub(r'[#*\[\]()`\|]', '', report)
    text_only = re.sub(r'\s+', ' ', text_only)
    # Count Chinese characters + English words
    chinese_chars = len(re.findall(r'[一-鿿]', text_only))
    english_words = len(re.findall(r'[a-zA-Z]+', text_only))
    total = chinese_chars + english_words
    print(f"Word count: {total} (Chinese chars: {chinese_chars}, English words: {english_words})")
    if total >= 3000:
        print("[green]Word count requirement met (>= 3000)[/green]")
    else:
        print(f"[yellow]Need {3000 - total} more words[/yellow]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
