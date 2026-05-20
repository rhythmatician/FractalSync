"""Quick evaluator for training history with hit-transition alignment summary.

Usage:
    python backend/scripts/evaluate_training_history.py --history checkpoints/training_history.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List


def _coerce_series(history: Dict[str, object], key: str) -> List[float]:
    raw = history.get(key, [])
    if not isinstance(raw, list):
        return []
    out: List[float] = []
    for item in raw:
        try:
            out.append(float(item))
        except Exception:
            continue
    return out


def _last(series: List[float]) -> float | None:
    if not series:
        return None
    return series[-1]


def _trend_slope(series: List[float]) -> float:
    """Return least-squares slope per epoch for a series."""
    n = len(series)
    if n < 2:
        return 0.0
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(series) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, series))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0.0:
        return 0.0
    return num / den


def _alignment_score(
    sequence_perceptual_last: float | None,
    hit_alignment_last: float | None,
) -> Dict[str, float]:
    """Compute a compact 0-100 alignment score from epoch-level losses.

    sequence_perceptual_loss is negative correlation loss, so lower is better.
    hit_alignment_loss is MSE between normalized hit intensity and transition speed,
    so lower is better.
    """
    if sequence_perceptual_last is None or hit_alignment_last is None:
        return {
            "score": 0.0,
            "corr_component": 0.0,
            "hit_component": 0.0,
        }

    corr_estimate = max(-1.0, min(1.0, -sequence_perceptual_last))
    corr_component = (corr_estimate + 1.0) * 0.5
    hit_component = math.exp(-max(0.0, hit_alignment_last))
    score = 100.0 * (0.6 * corr_component + 0.4 * hit_component)

    return {
        "score": score,
        "corr_component": corr_component,
        "hit_component": hit_component,
    }


def _save_plots(
    out_dir: Path,
    history: Dict[str, List[float]],
) -> List[Path]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return []

    out_paths: List[Path] = []
    max_len = max((len(v) for v in history.values() if v), default=0)
    if max_len == 0:
        return out_paths

    epochs = list(range(1, max_len + 1))

    def plot_series(path: Path, title: str, keys: List[str]) -> None:
        plt.figure(figsize=(10, 5))
        plotted = False
        for key in keys:
            series = history.get(key, [])
            if not series:
                continue
            x = epochs[: len(series)]
            plt.plot(x, series, label=key)
            plotted = True
        if not plotted:
            plt.close()
            return
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(title)
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        out_paths.append(path)

    plot_series(
        out_dir / "loss_core.png",
        "Core Training Losses",
        [
            "loss",
            "control_loss",
            "timbre_color_loss",
            "transient_impact_loss",
            "loudness_distance_loss",
        ],
    )

    plot_series(
        out_dir / "loss_sequence.png",
        "Sequence And Hit-Aware Losses",
        [
            "temporal_smoothness_loss",
            "sequence_perceptual_loss",
            "hit_alignment_loss",
            "rollout_loss",
        ],
    )

    return out_paths


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate training history and compute hit-transition alignment score",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=Path("checkpoints") / "training_history.json",
        help="Path to training_history.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("checkpoints") / "analysis",
        help="Directory for plots and summary JSON",
    )
    args = parser.parse_args()

    if not args.history.exists():
        print(f"History file not found: {args.history}")
        return 1

    with args.history.open("r", encoding="utf-8") as f:
        raw_history = json.load(f)

    if not isinstance(raw_history, dict):
        print("History file is not a JSON object")
        return 1

    history: Dict[str, List[float]] = {
        key: _coerce_series(raw_history, key) for key in raw_history.keys()
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)

    seq_last = _last(history.get("sequence_perceptual_loss", []))
    hit_last = _last(history.get("hit_alignment_loss", []))
    smooth_last = _last(history.get("temporal_smoothness_loss", []))
    rollout_last = _last(history.get("rollout_loss", []))

    alignment = _alignment_score(seq_last, hit_last)

    summary = {
        "history_path": str(args.history),
        "epochs": max((len(v) for v in history.values() if v), default=0),
        "last": {
            "loss": _last(history.get("loss", [])),
            "sequence_perceptual_loss": seq_last,
            "hit_alignment_loss": hit_last,
            "temporal_smoothness_loss": smooth_last,
            "rollout_loss": rollout_last,
        },
        "trend_slope_per_epoch": {
            "loss": _trend_slope(history.get("loss", [])),
            "sequence_perceptual_loss": _trend_slope(
                history.get("sequence_perceptual_loss", [])
            ),
            "hit_alignment_loss": _trend_slope(history.get("hit_alignment_loss", [])),
            "temporal_smoothness_loss": _trend_slope(
                history.get("temporal_smoothness_loss", [])
            ),
            "rollout_loss": _trend_slope(history.get("rollout_loss", [])),
        },
        "alignment": alignment,
    }

    summary_path = args.out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_paths = _save_plots(args.out_dir, history)

    print("Training history evaluation")
    print(f"  history: {args.history}")
    print(f"  epochs: {summary['epochs']}")
    print(
        "  alignment score (0-100): "
        f"{alignment['score']:.2f} "
        f"(corr={alignment['corr_component']:.3f}, hit={alignment['hit_component']:.3f})"
    )
    print(f"  summary: {summary_path}")

    if plot_paths:
        print("  plots:")
        for path in plot_paths:
            print(f"    - {path}")
    else:
        print("  plots: skipped (matplotlib not available)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
