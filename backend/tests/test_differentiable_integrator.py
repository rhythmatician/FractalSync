from __future__ import annotations

import random

import numpy as np
import torch

from src import differentiable_integrator as di
from scripts.tune_contour import load_distance_field
import runtime_core as rc


def _make_torch_df_from_py(py_df):
    # build field tensor from py_df.field (list)
    res = py_df.res
    field = np.array(py_df.field, dtype=np.float32).reshape((res, res))
    t_field = torch.from_numpy(field)
    return di.TorchDistanceField(
        t_field,
        real_min=py_df.real_min,
        real_max=py_df.real_max,
        imag_min=py_df.imag_min,
        imag_max=py_df.imag_max,
        max_distance=py_df.max_distance,
        slowdown_threshold=py_df.slowdown_threshold,
    )


def test_single_step_parity():
    df = load_distance_field()
    tdf = _make_torch_df_from_py(df)

    rng = random.Random(123)
    for _ in range(10):
        real = rng.uniform(-1.5, 1.5)
        imag = rng.uniform(-1.5, 1.5)
        # propose a small u
        u_real = rng.uniform(-0.05, 0.05)
        u_imag = rng.uniform(-0.05, 0.05)
        h = rng.choice([0.0, 1.0])

        # Use runtime-core DistanceField to compute the reference step
        flat = list(np.array(df.field, dtype=np.float32).ravel())
        rc_df = rc.DistanceField(flat, df.res, (df.real_min, df.real_max), (df.imag_min, df.imag_max), df.max_distance, df.slowdown_threshold)
        out_py = rc.contour_biased_step(real, imag, u_real, u_imag, h, 0.5, 0.05, rc_df)

        # torch inputs
        c_r = torch.tensor([real], dtype=torch.float32)
        c_i = torch.tensor([imag], dtype=torch.float32)
        u_r = torch.tensor([u_real], dtype=torch.float32)
        u_i = torch.tensor([u_imag], dtype=torch.float32)
        ht = torch.tensor([h], dtype=torch.float32)

        nr, ni = di.contour_biased_step_torch(c_r, c_i, u_r, u_i, ht, 0.5, 0.05, tdf)

        # Allow small numerical differences; this is a PoC parity check
        # Allow modest numerical differences; this is a PoC parity check
        err_real = abs(float(nr.item()) - float(out_py.real))
        err_imag = abs(float(ni.item()) - float(out_py.imag))
        # assert errors are small in absolute terms
        assert (
            err_real < 2e-2 and err_imag < 2e-2
        ), f"parity error too large: real={err_real}, imag={err_imag}"


def test_autograd_flow():
    df = load_distance_field()
    tdf = _make_torch_df_from_py(df)

    # small batch of 4
    u_r = torch.randn(4, requires_grad=True)
    u_i = torch.randn(4, requires_grad=True)
    c_r = torch.randn(4, requires_grad=False)
    c_i = torch.randn(4, requires_grad=False)
    h = torch.rand(4)

    nr, ni = di.contour_biased_step_torch(c_r, c_i, u_r, u_i, h, 0.5, 0.05, tdf)
    loss = (nr**2 + ni**2).sum()
    loss.backward()

    assert u_r.grad is not None and u_i.grad is not None
    assert torch.any(torch.abs(u_r.grad) > 0.0) or torch.any(torch.abs(u_i.grad) > 0.0)
