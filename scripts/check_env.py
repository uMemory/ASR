"""Environment smoke test.

Verifies:
    - Python / PyTorch / CUDA availability
    - All model paths in configs/models.yaml exist on disk
    - .env keys are loaded
    - Datasets are reachable

Run:
    conda activate TTS
    python scripts/check_env.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rich.console import Console
from rich.table import Table

from src.utils.config import get_env, get_model_config, project_root
from src.utils.cuda_dlls import add_cuda_dll_dirs

add_cuda_dll_dirs()

console = Console()


def check_torch() -> tuple[bool, str]:
    try:
        import torch

        info = f"torch={torch.__version__}, cuda={torch.cuda.is_available()}"
        if torch.cuda.is_available():
            info += f", device={torch.cuda.get_device_name(0)}, "
            info += f"vram={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
        return True, info
    except Exception as e:
        return False, f"import torch failed: {e}"


def check_path(label: str, path: str) -> tuple[bool, str]:
    if path is None:
        return False, "(null)"
    if not path.startswith(("/", ".")) and ":" not in path and "\\" not in path:
        return True, f"(HF repo / external id) {path}"
    exists = Path(path).exists()
    return exists, path


def check_env_keys(keys: list[str]) -> dict[str, bool]:
    return {k: bool(get_env(k)) for k in keys}


def main() -> int:
    console.rule("[bold cyan]ASR project — environment check")

    root = project_root()
    console.print(f"[dim]Project root:[/dim] {root}")

    ok_torch, torch_info = check_torch()
    console.print(f"\n[bold]PyTorch:[/bold] {'OK' if ok_torch else 'FAIL'} — {torch_info}")

    mcfg = get_model_config()
    console.print(f"\n[bold]Active profile:[/bold] {mcfg['profile']}  device={mcfg['device']}")

    tbl = Table(title="Model paths", show_lines=False)
    tbl.add_column("module")
    tbl.add_column("key")
    tbl.add_column("path / id")
    tbl.add_column("status")

    failed = 0
    for module in ("asr", "diarization", "alignment", "embedding"):
        for k, v in mcfg.get(module, {}).items():
            if not isinstance(v, str):
                continue
            ok, shown = check_path(f"{module}.{k}", v)
            tbl.add_row(module, k, shown, "[green]OK" if ok else "[red]MISSING")
            if not ok:
                failed += 1
    console.print(tbl)

    keys = ["DEEPSEEK_API_KEY", "CLAUDE_API_KEY", "CLAUDE_BASE_URL",
            "MINIMAX_API_KEY", "ACTIVE_LLM"]
    env_status = check_env_keys(keys)
    env_tbl = Table(title=".env keys")
    env_tbl.add_column("key")
    env_tbl.add_column("loaded")
    for k, ok in env_status.items():
        env_tbl.add_row(k, "[green]yes" if ok else "[red]no")
    console.print(env_tbl)

    ds_tbl = Table(title="Datasets")
    ds_tbl.add_column("dataset")
    ds_tbl.add_column("path")
    ds_tbl.add_column("status")
    for label, rel in [
        ("AliMeeting Eval far", "dataset/AIMeeting/Eval_Ali/Eval_Ali_far"),
        ("AliMeeting Eval near", "dataset/AIMeeting/Eval_Ali/Eval_Ali_near"),
        ("AISHELL-4 test wav", "dataset/AISHELL-4/test/wav"),
        ("AISHELL-4 test TextGrid", "dataset/AISHELL-4/test/TextGrid"),
    ]:
        p = root / rel
        ok = p.exists()
        ds_tbl.add_row(label, str(p), "[green]OK" if ok else "[red]MISSING")
        if not ok:
            failed += 1
    console.print(ds_tbl)

    console.rule()
    if failed:
        console.print(f"[red]{failed} issue(s) found.[/red]")
        return 1
    console.print("[bold green]All checks passed.[/bold green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
