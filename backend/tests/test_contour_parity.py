import random

import numpy as np
import torch

import runtime_core as rc
from src.differentiable_integrator import TorchDistanceField, contour_biased_step_torch


def make_df_grid(res=32):
    # simple radial distance field for parity test
    xs = np.linspace(-1.5, 1.5, res)
    ys = np.linspace(-1.5, 1.5, res)
    X, Y = np.meshgrid(xs, ys)
    R = np.sqrt(X**2 + Y**2)
    field = np.clip(1.0 - R / R.max(), 0.0, 1.0).astype(np.float32)
    return field


def test_contour_parity_with_distance_field():
    res = 32
    field = make_df_grid(res)
    flat = list(field.ravel())
    df = rc.DistanceField(flat, res, (-1.5, 1.5), (-1.5, 1.5), 1.0, 0.05)

    td_field = torch.from_numpy(field)
    tdf = TorchDistanceField(
        td_field,
        real_min=-1.5,
        real_max=1.5,
        imag_min=-1.5,
        imag_max=1.5,
        max_distance=1.0,
        slowdown_threshold=0.05,
        use_runtime_sampler=True,
    )

    rng = random.Random(1234)
    diffs = []
    for _ in range(20):
        cr = rng.uniform(-1.0, 1.0)
        ci = rng.uniform(-1.0, 1.0)
        ur = rng.uniform(-0.1, 0.1)
        ui = rng.uniform(-0.1, 0.1)
        h = rng.uniform(0.0, 1.0)
        d_star = 0.5
        max_step = 0.2

        # Rust step
        out = rc.contour_biased_step(cr, ci, ur, ui, h, d_star, max_step, df)
        rust_r = float(out.real)
        rust_i = float(out.imag)

        # Torch step (single sample tensors)
        cr_t = torch.tensor([cr], dtype=torch.float32)
        ci_t = torch.tensor([ci], dtype=torch.float32)
        ur_t = torch.tensor([ur], dtype=torch.float32)
        ui_t = torch.tensor([ui], dtype=torch.float32)
        h_t = torch.tensor([h], dtype=torch.float32)

        tr, ti = contour_biased_step_torch(
            cr_t, ci_t, ur_t, ui_t, h_t, d_star, max_step, tdf
        )
        tr = float(tr.item())
        ti = float(ti.item())

        diffs.append(abs(rust_r - tr))
        diffs.append(abs(rust_i - ti))

    # Check distribution of differences: expect median small and maxima bounded
    med = float(np.median(diffs))
    mx = float(max(diffs))
    assert med < 5e-3, f"Median diff too large: {med}"
    assert mx < 5e-2, f"Max diff too large: {mx}"
