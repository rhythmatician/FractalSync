"""Run parameter sweep for contour-biased integrator and report metrics.

Usage:
    python scripts/tune_contour.py --d-star 0.2,0.4 --max-step 0.01,0.05 --trials 5 --steps 200
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np

import sys

# Ensure repository root is in path so `src` imports work when script is invoked directly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import runtime_core as rc  # noqa: E402
from src.visual_metrics import VisualMetrics  # noqa: E402


class PyDistanceField:
    def __init__(
        self,
        field: List[float],
        res: int,
        real_range: Tuple[float, float],
        imag_range: Tuple[float, float],
        max_distance: float,
        slowdown_threshold: float,
    ):
        self.field = field
        self.res = res
        self.real_min, self.real_max = real_range
        self.imag_min, self.imag_max = imag_range
        self.max_distance = max_distance
        self.slowdown_threshold = slowdown_threshold

    def _coords(self, real: float, imag: float):
        real_scale = (self.real_max - self.real_min) / (self.res)
        imag_scale = (self.imag_max - self.imag_min) / (self.res)
        col_f = (real - self.real_min) / real_scale
        row_f = (imag - self.imag_min) / imag_scale
        return col_f, row_f

    def sample_bilinear(self, real: float, imag: float) -> float:
        col_f, row_f = self._coords(real, imag)
        if col_f < 0 or col_f > (self.res - 1) or row_f < 0 or row_f > (self.res - 1):
            return 1.0
        col0 = int(math.floor(col_f))
        row0 = int(math.floor(row_f))
        col1 = min(col0 + 1, self.res - 1)
        row1 = min(row0 + 1, self.res - 1)
        dx = float(col_f - col0)
        dy = float(row_f - row0)
        v00 = self.field[row0 * self.res + col0]
        v10 = self.field[row0 * self.res + col1]
        v01 = self.field[row1 * self.res + col0]
        v11 = self.field[row1 * self.res + col1]
        top = v00 * (1.0 - dx) + v10 * dx
        bottom = v01 * (1.0 - dx) + v11 * dx
        return float(top * (1.0 - dy) + bottom * dy)

    def gradient(self, real: float, imag: float) -> Tuple[float, float]:
        real_scale = (self.real_max - self.real_min) / (self.res)
        imag_scale = (self.imag_max - self.imag_min) / (self.res)
        left = self.sample_bilinear(real - real_scale, imag)
        right = self.sample_bilinear(real + real_scale, imag)
        down = self.sample_bilinear(real, imag - imag_scale)
        up = self.sample_bilinear(real, imag + imag_scale)
        gx = (right - left) / (2.0 * real_scale)
        gy = (up - down) / (2.0 * imag_scale)
        return gx, gy

    def get_velocity_scale(self, real: float, imag: float) -> float:
        dist = self.sample_bilinear(real, imag)
        th = self.slowdown_threshold
        if dist >= th:
            return 1.0
        t = dist / th
        s = t * t * (3.0 - 2.0 * t)
        return float(s)


def load_distance_field() -> PyDistanceField:
    df_base = Path("data") / "mandelbrot_distance_field"
    npy_path = df_base.with_suffix(".npy")
    json_path = df_base.with_suffix(".json")
    if not npy_path.exists() or not json_path.exists():
        # Fallback: generate a synthetic radial distance field so tests can run
        res = 64
        real_range = (-2.0, 2.0)
        imag_range = (-2.0, 2.0)
        xs = np.linspace(real_range[0], real_range[1], res)
        ys = np.linspace(imag_range[0], imag_range[1], res)
        field = []
        for y in ys:
            for x in xs:
                r = math.sqrt(x * x + y * y)
                # map radius to [0,1] (near center => small distance), clamp
                v = float(min(1.0, r / 2.0))
                field.append(v)
        max_distance = 2.0
        slowdown_threshold = 0.1
        return PyDistanceField(
            field, res, real_range, imag_range, max_distance, slowdown_threshold
        )

    # load json for ranges
    import json as _json

    with open(json_path, "r", encoding="utf-8") as f:
        meta = _json.load(f)

    field = np.load(str(npy_path)).astype(np.float32).flatten().tolist()
    res = meta.get("resolution") or int(math.sqrt(len(field)))
    real_range = (meta.get("real_min", -2.0), meta.get("real_max", 2.0))
    imag_range = (meta.get("imag_min", -2.0), meta.get("imag_max", 2.0))
    max_distance = float(meta.get("max_distance", 2.0))
    slowdown_threshold = float(meta.get("slowdown_threshold", 0.05))

    # Wrap in Python-side DistanceField for portability (avoids requiring runtime_core DistanceField)
    return PyDistanceField(
        field, res, real_range, imag_range, max_distance, slowdown_threshold
    )


def contour_biased_step_py(c_cur, u_real, u_imag, h, d_star, max_step, df):
    # Mirror the Rust `contour_biased_step` behavior in Python for environments
    # where the Rust binding isn't available.
    gx, gy = df.gradient(float(c_cur.real), float(c_cur.imag))
    grad_norm = math.hypot(gx, gy)

    if grad_norm <= 1e-12:
        u_mag = math.hypot(u_real, u_imag)
        scale = (max_step / u_mag) if (u_mag > max_step and u_mag > 0) else 1.0
        return type(
            "C",
            (),
            {"real": c_cur.real + u_real * scale, "imag": c_cur.imag + u_imag * scale},
        )()

    nx = gx / grad_norm
    ny = gy / grad_norm
    tx = -gy / grad_norm
    ty = gx / grad_norm

    proj_t = u_real * tx + u_imag * ty
    proj_n = u_real * nx + u_imag * ny

    normal_scale_no_hit = 0.05
    normal_scale_hit = 1.0
    tangential_scale = 1.0
    normal_scale = normal_scale_no_hit + (normal_scale_hit - normal_scale_no_hit) * max(
        0.0, min(1.0, h)
    )

    servo_gain = 0.2
    d = df.sample_bilinear(float(c_cur.real), float(c_cur.imag))
    servo = servo_gain * (d_star - d)

    dx = tx * (proj_t * tangential_scale) + nx * (proj_n * normal_scale + servo)
    dy = ty * (proj_t * tangential_scale) + ny * (proj_n * normal_scale + servo)

    mag = math.hypot(dx, dy)
    if mag > max_step and mag > 0.0:
        s = max_step / mag
        dx *= s
        dy *= s

    return type("C", (), {"real": c_cur.real + dx, "imag": c_cur.imag + dy})()


def run_trial(
    df: PyDistanceField,
    d_star: float,
    max_step: float,
    steps: int,
    seed: int,
    hit_prob: float,
) -> dict:
    rng = random.Random(seed)

    # initialize orbit state deterministically
    state = rc.OrbitState.new_with_seed(
        1,
        0,
        0.0,
        float(rc.DEFAULT_BASE_OMEGA),
        1.02,
        0.3,
        int(rc.DEFAULT_K_RESIDUALS),
        float(rc.DEFAULT_RESIDUAL_OMEGA_SCALE),
        seed,
    )
    rp = rc.ResidualParams(
        int(rc.DEFAULT_K_RESIDUALS), float(rc.DEFAULT_RESIDUAL_CAP), 1.0
    )

    vm = VisualMetrics()

    prev_frame = None
    delta_vs = []
    distances = []
    jumps = 0

    for t in range(steps):
        # current point
        c_cur = state.synthesize(rp)
        # advance phases by small dt and propose new c
        state.advance(0.1)
        c_prop = state.synthesize(rp)

        u_real = float(c_prop.real - c_cur.real)
        u_imag = float(c_prop.imag - c_cur.imag)

        # transient hit
        h = 1.0 if rng.random() < hit_prob else 0.0

        # apply contour integrator: prefer runtime_core binding if available
        if hasattr(rc, "contour_biased_step"):
            # runtime_core binding may not be available in this environment; fall back to Python version
            try:
                c_next = rc.contour_biased_step(
                    float(c_cur.real),
                    float(c_cur.imag),
                    u_real,
                    u_imag,
                    h,
                    d_star,
                    max_step,
                    None,
                )
            except Exception:
                c_next = contour_biased_step_py(
                    c_cur, u_real, u_imag, h, d_star, max_step, df
                )
        else:
            c_next = contour_biased_step_py(
                c_cur, u_real, u_imag, h, d_star, max_step, df
            )

        # render proxy frame small
        img = vm.render_julia_set(
            seed_real=float(c_next.real),
            seed_imag=float(c_next.imag),
            width=64,
            height=64,
            max_iter=40,
        )
        gray = np.mean(img, axis=2) if img.ndim == 3 else img
        if prev_frame is None:
            delta_v = 0.0
        else:
            delta_v = float(
                np.mean(np.abs(gray.astype(np.float32) - prev_frame.astype(np.float32)))
            )
        delta_vs.append(delta_v)
        prev_frame = gray

        # distance to boundary
        d = float(df.sample_bilinear(float(c_next.real), float(c_next.imag)))
        distances.append(d)

        # jump magnitude
        jump_mag = math.hypot(u_real, u_imag)
        if jump_mag > max_step * 1.5:
            jumps += 1

    return {
        "mean_deltaV": float(np.mean(delta_vs)),
        "std_deltaV": float(np.std(delta_vs)),
        "mean_distance": float(np.mean(distances)),
        "std_distance": float(np.std(distances)),
        "jump_rate": jumps / float(steps),
    }


def sweep(
    d_stars: List[float],
    max_steps: List[float],
    trials: int,
    steps: int,
    hit_prob: float,
    out_path: Path,
):
    df = load_distance_field()
    results = []

    for d_star in d_stars:
        for max_step in max_steps:
            agg = {"mean_deltaV": [], "mean_distance": [], "jump_rate": []}
            for trial in range(trials):
                seed = 1000 + trial
                r = run_trial(df, d_star, max_step, steps, seed, hit_prob)
                agg["mean_deltaV"].append(r["mean_deltaV"])
                agg["mean_distance"].append(r["mean_distance"])
                agg["jump_rate"].append(r["jump_rate"])

            results.append(
                {
                    "d_star": d_star,
                    "max_step": max_step,
                    "trials": trials,
                    "steps": steps,
                    "mean_deltaV": float(np.mean(agg["mean_deltaV"])),
                    "mean_distance": float(np.mean(agg["mean_distance"])),
                    "jump_rate": float(np.mean(agg["jump_rate"])),
                }
            )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


def parse_list(s: str) -> List[float]:
    return [float(x) for x in s.split(",") if x.strip()]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--d-star", type=str, default="0.2,0.4,0.6")
    p.add_argument("--max-step", type=str, default="0.01,0.05,0.1")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--hit-prob", type=float, default=0.05)
    p.add_argument("--out", type=str, default="logs/tune_contour.json")

    args = p.parse_args()
    d_stars = parse_list(args.d_star)
    max_steps = parse_list(args.max_step)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Running sweep d*={d_stars} max_step={max_steps} trials={args.trials} steps={args.steps}"
    )
    results = sweep(
        d_stars, max_steps, args.trials, args.steps, args.hit_prob, out_path
    )
    print(f"Wrote results to {out_path}")

    for r in results:
        print(r)


if __name__ == "__main__":
    main()
