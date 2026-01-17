#!/usr/bin/env python3
from __future__ import annotations

import math
import re
from pathlib import Path
from statistics import mean, median, pstdev, stdev

import matplotlib.pyplot as plt

# Matches floats like -0.7, 1, 2.3e-4, -5E+6
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

RE_JULIA_REAL = re.compile(rf"\bjuliaReal\s*:\s*({FLOAT})\b")
RE_JULIA_IMAG = re.compile(rf"\bjuliaImag\s*:\s*({FLOAT})\b")


def describe(xs: list[float]) -> dict[str, float]:
    return {
        "count": float(len(xs)),
        "min": min(xs),
        "max": max(xs),
        "mean": mean(xs),
        "median": median(xs),
        "stdev_sample": stdev(xs) if len(xs) >= 2 else float("nan"),
        "stdev_pop": pstdev(xs) if len(xs) >= 1 else float("nan"),
    }


def main() -> None:
    path = Path("console.log")
    if not path.exists():
        raise SystemExit("console.log not found in current directory.")

    reals: list[float] = []
    imags: list[float] = []

    # time series (by occurrence order)
    t: list[int] = []
    real_ts: list[float] = []
    imag_ts: list[float] = []

    matched_lines = 0
    lines_with_both = 0
    missing_real = 0
    missing_imag = 0
    nonfinite_dropped = 0

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m_r = RE_JULIA_REAL.search(line)
            m_i = RE_JULIA_IMAG.search(line)

            if not (m_r or m_i):
                continue

            matched_lines += 1

            if not m_r and m_i:
                missing_real += 1
                continue
            if not m_i and m_r:
                missing_imag += 1
                continue

            # Both present
            r = float(m_r.group(1))
            i = float(m_i.group(1))
            if not (math.isfinite(r) and math.isfinite(i)):
                nonfinite_dropped += 1
                continue

            lines_with_both += 1
            reals.append(r)
            imags.append(i)

            # time index is “occurrence order among lines with both”
            t.append(lines_with_both - 1)
            real_ts.append(r)
            imag_ts.append(i)

    if not reals or not imags:
        raise SystemExit(
            f"Found {len(reals)} juliaReal values and {len(imags)} juliaImag values. "
            "Not enough data to summarize."
        )

    real_stats = describe(reals)
    imag_stats = describe(imags)

    def print_stats(name: str, stats: dict[str, float]) -> None:
        print(f"\n{name}")
        print("-" * len(name))
        print(f"count         : {int(stats['count'])}")
        print(f"min           : {stats['min']:.12g}")
        print(f"max           : {stats['max']:.12g}")
        print(f"mean          : {stats['mean']:.12g}")
        print(f"median        : {stats['median']:.12g}")
        print(f"stdev (sample): {stats['stdev_sample']:.12g}")
        print(f"stdev (pop)   : {stats['stdev_pop']:.12g}")

    print(f"File: {path} ({path.stat().st_size} bytes)")
    print(f"Lines with at least one match: {matched_lines}")
    print(f"Lines with both values used  : {lines_with_both}")
    print(f"Lines missing juliaReal (but had juliaImag): {missing_real}")
    print(f"Lines missing juliaImag (but had juliaReal): {missing_imag}")
    print(f"Lines dropped for non-finite values        : {nonfinite_dropped}")

    print_stats("juliaReal", real_stats)
    print_stats("juliaImag", imag_stats)

    # Plot over occurrence order
    plt.figure()
    plt.plot(t, real_ts, label="juliaReal")
    plt.plot(t, imag_ts, label="juliaImag")
    plt.xlabel("Occurrence index (lines with both values)")
    plt.ylabel("Value")
    plt.title("juliaReal / juliaImag over time (file order)")
    plt.legend()
    out = Path("julia_over_time.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"\nWrote plot: {out.resolve()}")


if __name__ == "__main__":
    main()
