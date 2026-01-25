import random

import numpy as np
import torch

import runtime_core as rc
from src import differentiable_integrator as di


def test_sequence_rollout_parity():
    res = 32
    xs = np.linspace(-1.5, 1.5, res)
    ys = np.linspace(-1.5, 1.5, res)
    X, Y = np.meshgrid(xs, ys)
    R = np.sqrt(X**2 + Y**2)
    field = np.clip(1.0 - R / R.max(), 0.0, 1.0).astype(np.float32)

    rc_df = rc.DistanceField(
        list(field.ravel()),
        res,
        (-1.5, 1.5),
        (-1.5, 1.5),
        1.0,
        0.05,
    )

    tdf = di.TorchDistanceField(
        torch.from_numpy(field),
        real_min=-1.5,
        real_max=1.5,
        imag_min=-1.5,
        imag_max=1.5,
        max_distance=1.0,
        slowdown_threshold=0.05,
        use_runtime_sampler=True,
    )

    rng = random.Random(42)
    c_real = 0.1
    c_imag = -0.2
    steps = []
    for _ in range(12):
        u_real = rng.uniform(-0.05, 0.05)
        u_imag = rng.uniform(-0.05, 0.05)
        h = rng.uniform(0.0, 1.0)
        steps.append((u_real, u_imag, h))

    # runtime-core rollout
    rc_traj = []
    for u_real, u_imag, h in steps:
        c = rc.contour_biased_step(
            c_real, c_imag, u_real, u_imag, h, 0.5, 0.05, rc_df
        )
        c_real, c_imag = c.real, c.imag
        rc_traj.append((c_real, c_imag))

    # torch rollout
    t_real = torch.tensor([0.1], dtype=torch.float32)
    t_imag = torch.tensor([-0.2], dtype=torch.float32)
    torch_traj = []
    for u_real, u_imag, h in steps:
        u_r = torch.tensor([u_real], dtype=torch.float32)
        u_i = torch.tensor([u_imag], dtype=torch.float32)
        h_t = torch.tensor([h], dtype=torch.float32)
        t_real, t_imag = di.contour_biased_step_torch(
            t_real, t_imag, u_r, u_i, h_t, 0.5, 0.05, tdf
        )
        torch_traj.append((float(t_real.item()), float(t_imag.item())))

    for (r0, i0), (r1, i1) in zip(rc_traj, torch_traj):
        assert abs(r0 - r1) < 2e-2
        assert abs(i0 - i1) < 2e-2
