#!/usr/bin/env python3
from __future__ import annotations

import math
import re
from pathlib import Path
from statistics import mean, median, pstdev, stdev

# Matches floats like -0.7, 1, 2.3e-4, -5E+6
FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

RE_JULIA_REAL = re.compile(rf"\bjuliaReal\s*:\s*({FLOAT})\b")
RE_JULIA_IMAG = re.compile(rf"\bjuliaImag\s*:\s*({FLOAT})\b")


def describe(xs: list[float]) -> dict[str, float]:
    # Sample stdev is usually what people mean; also include population stdev for completeness.
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
    matched_lines = 0
    missing_real = 0
    missing_imag = 0

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m_r = RE_JULIA_REAL.search(line)
            m_i = RE_JULIA_IMAG.search(line)

            if m_r or m_i:
                matched_lines += 1

            if m_r:
                x = float(m_r.group(1))
                if math.isfinite(x):
                    reals.append(x)
            else:
                if m_i:  # line had imag but not real
                    missing_real += 1

            if m_i:
                x = float(m_i.group(1))
                if math.isfinite(x):
                    imags.append(x)
            else:
                if m_r:  # line had real but not imag
                    missing_imag += 1

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
        print(f"count        : {int(stats['count'])}")
        print(f"min          : {stats['min']:.12g}")
        print(f"max          : {stats['max']:.12g}")
        print(f"mean         : {stats['mean']:.12g}")
        print(f"median       : {stats['median']:.12g}")
        print(f"stdev (sample): {stats['stdev_sample']:.12g}")
        print(f"stdev (pop)   : {stats['stdev_pop']:.12g}")

    print(f"File: {path} ({path.stat().st_size} bytes)")
    print(f"Lines with at least one match: {matched_lines}")
    print(f"Lines missing juliaReal (but had juliaImag): {missing_real}")
    print(f"Lines missing juliaImag (but had juliaReal): {missing_imag}")

    print_stats("juliaReal", real_stats)
    print_stats("juliaImag", imag_stats)


if __name__ == "__main__":
    main()
