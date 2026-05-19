"""Configuration loader. Reads YAML configs and resolves model paths."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_env() -> None:
    load_dotenv(project_root() / ".env")


@lru_cache(maxsize=8)
def load_config(name: str) -> dict[str, Any]:
    """Load configs/<name>.yaml as a dict."""
    path = project_root() / "configs" / f"{name}.yaml"
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_path(p: str) -> str:
    """Local model paths starting with './' get resolved to absolute paths;
    HuggingFace repo ids and torchaudio constants pass through untouched."""
    if p.startswith("./") or p.startswith(".\\"):
        return str((project_root() / p).resolve())
    return p


def get_model_config(profile: str | None = None) -> dict[str, Any]:
    """Return the active profile (local/cloud) with paths resolved."""
    cfg = load_config("models")
    profile = profile or cfg.get("profile", "local")
    block = cfg[profile]

    resolved: dict[str, Any] = {}
    for module, params in block.items():
        resolved[module] = {
            k: _resolve_path(v) if isinstance(v, str) else v for k, v in params.items()
        }
    resolved["device"] = cfg.get("device", "cuda")
    resolved["profile"] = profile
    resolved["hf_cache"] = _resolve_path(cfg.get("hf_cache", "./cache/huggingface"))
    resolved["torch_cache"] = _resolve_path(cfg.get("torch_cache", "./cache/torch"))
    # Top-level non-path configs (profile-independent)
    resolved["llm"] = cfg.get("llm", {})
    return resolved


def get_env(key: str, default: str | None = None) -> str | None:
    load_env()
    return os.getenv(key, default)
