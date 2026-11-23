"""Utilitarios comuns: configuracao, caminhos, sementes e E/S."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "configs" / "experiment.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def data_dir(cfg: dict) -> Path:
    d = ROOT / cfg["data"]["root"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def out_dir(*parts: str) -> Path:
    d = ROOT.joinpath("outputs", *parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True
    except ImportError:
        pass


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
