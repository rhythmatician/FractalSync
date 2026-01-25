"""Shared show control configuration loader."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class ShowConfig:
    contour_d_star: float
    contour_max_step: float
    continuity_idle: float
    continuity_hit: float
    variety_bins: int
    variety_min: float
    eval_thresholds: Dict[str, Any]


def _default_config() -> ShowConfig:
    return ShowConfig(
        contour_d_star=0.3,
        contour_max_step=0.03,
        continuity_idle=0.02,
        continuity_hit=0.08,
        variety_bins=16,
        variety_min=0.35,
        eval_thresholds={},
    )


def load_show_config(path: str | None = None) -> ShowConfig:
    if path is None:
        repo_root = Path(__file__).resolve().parents[2]
        path = str(repo_root / "frontend" / "public" / "show_control.json")
    cfg = _default_config()
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return cfg

    contour = raw.get("contour", {})
    continuity = raw.get("continuity_budget", {})
    variety = raw.get("variety", {})
    thresholds = raw.get("eval_thresholds", {})
    return ShowConfig(
        contour_d_star=float(contour.get("d_star", cfg.contour_d_star)),
        contour_max_step=float(contour.get("max_step", cfg.contour_max_step)),
        continuity_idle=float(continuity.get("idle", cfg.continuity_idle)),
        continuity_hit=float(continuity.get("hit", cfg.continuity_hit)),
        variety_bins=int(variety.get("coverage_bins", cfg.variety_bins)),
        variety_min=float(variety.get("min_coverage", cfg.variety_min)),
        eval_thresholds=thresholds,
    )
