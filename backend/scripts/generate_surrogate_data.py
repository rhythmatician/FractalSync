"""Generate dataset for SurrogateDeltaV model.

Produces a torch .pt file containing tensors:
 - c_prev: (N,2)
 - c_next: (N,2)
 - d_prev: (N,)
 - grad_prev: (N,2)
 - probes: (N,P) optional (here P=8 samples)
 - delta_v: (N,)

Usage (example):
  python scripts/generate_surrogate_data.py --out data/surrogate/samples_small.pt --n 5000
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import random

import torch

from src.visual_proxy import ProxyRenderer, frame_diff
from src.differentiable_integrator import TorchDistanceField


def sample_points_near_boundary(n, df: TorchDistanceField, max_radius=2.0):
    # Simple strategy: sample real,imag uniformly and keep those where df.sample small
    rs = []
    attempts = 0
    while len(rs) < n and attempts < n * 20:
        r = (random.random() * 4.0) - 2.0
        i = (random.random() * 4.0) - 2.0
        d = float(df.sample_bilinear(torch.tensor([r]), torch.tensor([i]))[0])
        # bias towards low distance (near boundary)
        if d < 0.5 or random.random() < 0.05:
            rs.append((r, i))
        attempts += 1
    if len(rs) < n:
        # fallback: fill randomly
        while len(rs) < n:
            rs.append(((random.random() * 4.0) - 2.0, (random.random() * 4.0) - 2.0))
    return rs[:n]


def main():
    parser = argparse.ArgumentParser(description="Generate surrogate ΔV dataset")
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--max-step", type=float, default=0.03)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # Use proxy renderer
    renderer = ProxyRenderer(resolution=64, max_iter=20, device=str(device))

    # For DF, try to use loaded numpy then wrap TorchDistanceField if available
    # Attempt to load precomputed DF from data/mandelbrot_distance_field.npy
    df = None
    try:
        import numpy as np
        from pathlib import Path as _P

        p = _P("data/mandelbrot_distance_field.npy")
        if p.exists():
            arr = np.load(str(p))
            # create TorchDistanceField with default meta
            df = TorchDistanceField(torch.tensor(arr, dtype=torch.float32))
    except Exception:
        df = None

    # If no DF, we still sample uniformly
    pts = sample_points_near_boundary(args.n, df) if df is not None else [((random.random() * 4.0) - 2.0, (random.random() * 4.0) - 2.0) for _ in range(args.n)]

    c_prev = []
    c_next = []
    d_prev = []
    grad_prev = []
    delta_v = []

    for (r, i) in pts:
        # sample small delta direction and magnitude
        theta = random.random() * 2 * math.pi
        mag = random.random() * args.max_step
        dr = math.cos(theta) * mag
        di = math.sin(theta) * mag

        c1r = r
        c1i = i
        c2r = r + dr
        c2i = i + di

        # compute frames and ΔV
        f1 = renderer.render(torch.tensor([c1r], device=device), torch.tensor([c1i], device=device))[0]
        f2 = renderer.render(torch.tensor([c2r], device=device), torch.tensor([c2i], device=device))[0]
        dv = float(frame_diff(f1.unsqueeze(0), f2.unsqueeze(0))[0].item())

        c_prev.append([c1r, c1i])
        c_next.append([c2r, c2i])
        delta_v.append(dv)

        if df is not None:
            dval = float(df.sample_bilinear(torch.tensor([c1r], device=device), torch.tensor([c1i], device=device))[0].item())
            gx, gy = df.gradient(torch.tensor([c1r], device=device), torch.tensor([c1i], device=device))
            d_prev.append(dval)
            grad_prev.append([float(gx[0].item()), float(gy[0].item())])
        else:
            d_prev.append(1.0)
            grad_prev.append([0.0, 0.0])

    # Save to torch file
    torch.save(
        {
            "c_prev": torch.tensor(c_prev, dtype=torch.float32),
            "c_next": torch.tensor(c_next, dtype=torch.float32),
            "d_prev": torch.tensor(d_prev, dtype=torch.float32),
            "grad_prev": torch.tensor(grad_prev, dtype=torch.float32),
            "delta_v": torch.tensor(delta_v, dtype=torch.float32),
        },
        str(out_path),
    )
    print(f"Wrote surrogate dataset: {out_path} (N={len(c_prev)})")


if __name__ == "__main__":
    main()
