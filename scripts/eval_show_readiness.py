#!/usr/bin/env python3
"""Evaluate show readiness metrics for FractalSync control pipeline."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import sys
import os

# Ensure backend root is in path so `src` imports work
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND_ROOT = os.path.join(REPO_ROOT, "backend")
if BACKEND_ROOT not in sys.path:
    sys.path.insert(0, BACKEND_ROOT)

try:
    import onnxruntime as ort
except Exception as exc:  # pragma: no cover
    raise RuntimeError("onnxruntime is required to run eval_show_readiness") from exc

from src.distance_field_loader import load_distance_field_for_runtime  # type: ignore
from src.model_contract import MODEL_INPUT_NAME, MODEL_OUTPUT_NAME  # type: ignore
from src.show_config import load_show_config  # type: ignore
from src.runtime_core_bridge import make_orbit_state, step_orbit  # type: ignore
from src.visual_metrics import VisualMetrics, proxy_delta_v  # type: ignore


def read_recording(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("_meta"):
                continue
            records.append(rec)
    return records


def compute_alignment(h: np.ndarray, delta_v: np.ndarray, max_lag: int = 5) -> Dict[str, Any]:
    best = {"lag": 0, "corr": -1.0}
    for lag in range(max_lag + 1):
        if lag == 0:
            h_slice = h
            d_slice = delta_v
        else:
            h_slice = h[:-lag]
            d_slice = delta_v[lag:]
        if len(h_slice) < 2:
            continue
        corr = np.corrcoef(h_slice, d_slice)[0, 1]
        if math.isnan(corr):
            corr = -1.0
        if corr > best["corr"]:
            best = {"lag": lag, "corr": float(corr)}
    return best


def percentile(arr: np.ndarray, q: float) -> float:
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, q))


def variety_metric(c_series: np.ndarray, bins: int = 16) -> float:
    angles = np.arctan2(c_series[:, 1], c_series[:, 0])
    hist, _ = np.histogram(angles, bins=bins, range=(-math.pi, math.pi))
    coverage = np.count_nonzero(hist) / float(bins)
    return float(coverage)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording", type=str, default="docs/sample_recordings/sample.ndjson")
    parser.add_argument("--model", type=str, default="frontend/public/models/model.onnx")
    parser.add_argument("--metadata", type=str, default=None)
    parser.add_argument("--output", type=str, default="reports/show_readiness.json")
    args = parser.parse_args()

    random.seed(1337)
    np.random.seed(1337)

    config = load_show_config()
    thresholds = config.eval_thresholds

    recording_path = Path(args.recording)
    records = read_recording(recording_path)
    if not records:
        raise RuntimeError(f"No records found in {recording_path}")

    model_path = Path(args.model)
    metadata_path = Path(args.metadata) if args.metadata else model_path.with_suffix(".onnx_metadata.json")
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    feature_mean = np.asarray(metadata.get("feature_mean", []), dtype=np.float32)
    feature_std = np.asarray(metadata.get("feature_std", []), dtype=np.float32)

    sess = ort.InferenceSession(str(model_path))
    input_name = MODEL_INPUT_NAME
    output_name = MODEL_OUTPUT_NAME

    df_path = Path("backend/data/mandelbrot_distance_field")
    distance_field = load_distance_field_for_runtime(str(df_path))

    visual_metrics = VisualMetrics()

    # initialize orbit state
    orbit = make_orbit_state(
        lobe=1,
        sub_lobe=0,
        theta=0.0,
        omega=1.0,
        s=1.02,
        alpha=0.3,
        seed=1337,
    )

    dt = 1.0 / 60.0
    c_series = []
    delta_v_series = []
    h_series = []
    d_series = []

    prev_proxy = None

    for rec in records:
        feats = np.asarray(rec["audio_features"], dtype=np.float32)
        if feature_mean.size > 0:
            feats = (feats - feature_mean) / (feature_std + 1e-8)

        h = float(rec.get("h", 0.0))
        h_series.append(h)

        inp = feats.reshape(1, -1).astype(np.float32)
        outputs = sess.run(None, {input_name: inp})[0][0]
        s_target, alpha, omega_scale = outputs[0:3]
        band_gates = outputs[3:]

        orbit.s = float(s_target)
        orbit.alpha = float(alpha)
        orbit.omega = float(omega_scale)

        c = step_orbit(
            orbit,
            dt,
            residual_params=None,
            band_gates=band_gates.tolist(),
            distance_field=distance_field,
            h=h,
            d_star=config.contour_d_star,
            max_step=config.contour_max_step,
        )
        c_series.append([float(c.real), float(c.imag)])
        d_series.append(distance_field.lookup(float(c.real), float(c.imag)))

        proxy = visual_metrics.render_julia_set(
            seed_real=float(c.real),
            seed_imag=float(c.imag),
            width=64,
            height=64,
            max_iter=40,
        )
        proxy_gray = np.mean(proxy, axis=2) if proxy.ndim == 3 else proxy
        delta_v = proxy_delta_v(prev_proxy, proxy_gray) if prev_proxy is not None else 0.0
        delta_v_series.append(delta_v)
        prev_proxy = proxy_gray

    c_series = np.asarray(c_series, dtype=np.float32)
    delta_v_series = np.asarray(delta_v_series, dtype=np.float32)
    h_series = np.asarray(h_series, dtype=np.float32)
    d_series = np.asarray(d_series, dtype=np.float32)

    alignment = compute_alignment(h_series, delta_v_series, max_lag=5)

    idle_mask = h_series < thresholds.get("idle_h_threshold", 0.2)
    hit_mask = h_series > thresholds.get("hit_h_threshold", 0.7)

    continuity = {
        "median_idle": float(np.median(delta_v_series[idle_mask])) if idle_mask.any() else 0.0,
        "p95_idle": percentile(delta_v_series[idle_mask], 95),
    }
    impact = {
        "p95_hit": percentile(delta_v_series[hit_mask], 95),
    }

    variety = variety_metric(c_series, bins=config.variety_bins)

    report = {
        "alignment": alignment,
        "continuity": continuity,
        "impact": impact,
        "boundary": {
            "d_mean": float(np.mean(d_series)),
            "d_p10": percentile(d_series, 10),
            "d_p90": percentile(d_series, 90),
        },
        "variety": variety,
        "samples": len(records),
    }

    # Evaluate pass/fail
    pass_fail = {
        "alignment": alignment["lag"] <= thresholds.get("max_best_lag", 1),
        "continuity_median": continuity["median_idle"] <= thresholds.get("idle_median_max", 0.03),
        "continuity_p95": continuity["p95_idle"] <= thresholds.get("idle_p95_max", 0.06),
        "impact": impact["p95_hit"] >= thresholds.get("hit_p95_min", 0.08),
        "variety": variety >= thresholds.get("min_variety", 0.35),
        "boundary": thresholds.get("d_low", 0.05)
        <= report["boundary"]["d_mean"]
        <= thresholds.get("d_high", 0.6),
    }
    report["pass_fail"] = pass_fail
    report["overall_pass"] = all(pass_fail.values())

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))

    summary_path = output_path.with_suffix(".md")
    summary_lines = [
        "# Show Readiness Report",
        f"- Samples: {report['samples']}",
        f"- Overall: {'PASS' if report['overall_pass'] else 'FAIL'}",
        f"- Alignment: lag={alignment['lag']} corr={alignment['corr']:.3f}",
        f"- Continuity (idle): median={continuity['median_idle']:.4f} p95={continuity['p95_idle']:.4f}",
        f"- Impact (hit): p95={impact['p95_hit']:.4f}",
        f"- Variety coverage: {variety:.3f}",
        f"- Boundary d_mean: {report['boundary']['d_mean']:.3f}",
    ]
    summary_path.write_text("\n".join(summary_lines))

    print(f"Wrote report: {output_path}")
    print(f"Wrote summary: {summary_path}")
    return 0 if report["overall_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
