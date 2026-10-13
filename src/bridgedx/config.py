"""Load config/option_a.yaml and resolve paths."""
from __future__ import annotations

import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "option_a.yaml"


def load(path: Path | str | None = None) -> dict:
    with open(path or DEFAULT_CONFIG) as f:
        return yaml.safe_load(f)


def data_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load()
    env = os.environ.get(cfg["paths"]["data_env"])
    p = Path(env) if env else (REPO_ROOT / cfg["paths"]["data_default"])
    return p.resolve()


def cache_path(cfg: dict | None = None) -> Path:
    cfg = cfg or load()
    p = REPO_ROOT / cfg["paths"]["cache"]
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def outputs_dir(cfg: dict | None = None) -> Path:
    cfg = cfg or load()
    p = REPO_ROOT / cfg["paths"]["outputs"]
    p.mkdir(parents=True, exist_ok=True)
    return p
