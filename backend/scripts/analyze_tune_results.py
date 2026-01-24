"""Analyze tune_contour results and recommend parameter combinations.

Outputs:
 - CSV file alongside input JSON
 - Prints top-N recommended combos
 - Optionally writes heatmaps (requires matplotlib)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import List

import numpy as np


def score_row(row):
    # Heuristic scoring: prioritize low mean_deltaV and low jump_rate,
    # and prefer mean_distance close to d_star
    mean_deltaV = row["mean_deltaV"]
    jump_rate = row["jump_rate"]
    mean_distance = row["mean_distance"]
    d_star = row["d_star"]

    score = mean_deltaV + jump_rate * 10.0 + abs(mean_distance - d_star) * 5.0
    return score


def analyze(
    results: List[dict],
    out_csv: Path,
    heatmap_prefix: Path | None = None,
    top_n: int = 5,
):
    # Write CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "d_star",
            "max_step",
            "trials",
            "steps",
            "mean_deltaV",
            "mean_distance",
            "jump_rate",
            "score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            r2 = {k: r.get(k) for k in fieldnames}
            r2["score"] = score_row(r)
            writer.writerow(r2)

    # Sort and print top results
    ranked = sorted(results, key=lambda r: score_row(r))
    print(f"Top {top_n} parameter combos by heuristic score:")
    for r in ranked[:top_n]:
        print(
            {
                "d_star": r["d_star"],
                "max_step": r["max_step"],
                "mean_deltaV": r["mean_deltaV"],
                "jump_rate": r["jump_rate"],
                "mean_distance": r["mean_distance"],
                "score": score_row(r),
            }
        )

    # Optional heatmaps
    if heatmap_prefix is not None:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("matplotlib not available; skipping heatmaps")
            return

        # Unique axes
        d_stars = sorted(set(r["d_star"] for r in results))
        max_steps = sorted(set(r["max_step"] for r in results))

        grid = np.zeros((len(d_stars), len(max_steps)))
        grid_jump = np.zeros_like(grid)

        for r in results:
            i = d_stars.index(r["d_star"])
            j = max_steps.index(r["max_step"])
            grid[i, j] = r["mean_deltaV"]
            grid_jump[i, j] = r["jump_rate"]

        # mean_deltaV heatmap
        plt.figure(figsize=(6, 4))
        plt.imshow(grid, origin="lower", aspect="auto", cmap="viridis")
        plt.colorbar(label="mean_deltaV")
        plt.xticks(range(len(max_steps)), [str(x) for x in max_steps], rotation=45)
        plt.yticks(range(len(d_stars)), [str(x) for x in d_stars])
        plt.xlabel("max_step")
        plt.ylabel("d_star")
        plt.title("mean_deltaV heatmap")
        plt.tight_layout()
        plt.savefig(str(heatmap_prefix) + "_deltaV.png")

        # jump_rate heatmap
        plt.figure(figsize=(6, 4))
        plt.imshow(grid_jump, origin="lower", aspect="auto", cmap="magma")
        plt.colorbar(label="jump_rate")
        plt.xticks(range(len(max_steps)), [str(x) for x in max_steps], rotation=45)
        plt.yticks(range(len(d_stars)), [str(x) for x in d_stars])
        plt.xlabel("max_step")
        plt.ylabel("d_star")
        plt.title("jump_rate heatmap")
        plt.tight_layout()
        plt.savefig(str(heatmap_prefix) + "_jump.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("input", type=str, help="Path to tune_contour results JSON")
    p.add_argument("--out-csv", type=str, default=None, help="Output CSV file path")
    p.add_argument(
        "--heatmap-prefix",
        type=str,
        default=None,
        help="Prefix for heatmap output files (PNG)",
    )
    p.add_argument("--top-n", type=int, default=5)

    args = p.parse_args()
    inp = Path(args.input)
    if not inp.exists():
        raise SystemExit(f"Input file not found: {inp}")

    results = json.loads(inp.read_text())
    out_csv = Path(args.out_csv) if args.out_csv else inp.with_suffix(".csv")
    heatmap_prefix = Path(args.heatmap_prefix) if args.heatmap_prefix else None

    analyze(results, out_csv, heatmap_prefix, args.top_n)
    print(f"Wrote CSV to {out_csv}")
