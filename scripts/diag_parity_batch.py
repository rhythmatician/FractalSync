import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import backend.src.differentiable_integrator as di
import runtime_core as rc

res = 32
xs = np.linspace(-1.5, 1.5, res)
ys = np.linspace(-1.5, 1.5, res)
X, Y = np.meshgrid(xs, ys)
R = np.sqrt(X**2 + Y**2)
field = np.clip(1.0 - R / R.max(), 0.0, 1.0).astype(np.float32)

# torch distance field using runtime sampler (non-diff)
td_field = torch.from_numpy(field)
tdf = di.TorchDistanceField(
    td_field,
    real_min=-1.5,
    real_max=1.5,
    imag_min=-1.5,
    imag_max=1.5,
    max_distance=1.0,
    slowdown_threshold=0.05,
    use_runtime_sampler=True,
)

rng = random.Random(123)
for idx in range(10):
    real = rng.uniform(-1.5, 1.5)
    imag = rng.uniform(-1.5, 1.5)
    u_real = rng.uniform(-0.05, 0.05)
    u_imag = rng.uniform(-0.05, 0.05)
    h = rng.choice([0.0, 1.0])

    flat = list(field.ravel())
    rc_df = rc.DistanceField(flat, res, (-1.5, 1.5), (-1.5, 1.5), 1.0, 0.05)
    out_py = rc.contour_biased_step(real, imag, u_real, u_imag, h, 0.5, 0.05, rc_df)

    c_r = torch.tensor([real], dtype=torch.float32)
    c_i = torch.tensor([imag], dtype=torch.float32)
    u_r = torch.tensor([u_real], dtype=torch.float32)
    u_i = torch.tensor([u_imag], dtype=torch.float32)
    ht = torch.tensor([h], dtype=torch.float32)

    nr, ni = di.contour_biased_step_torch(c_r, c_i, u_r, u_i, ht, 0.5, 0.05, tdf)

    err_real = abs(float(nr.item()) - float(out_py.real))
    err_imag = abs(float(ni.item()) - float(out_py.imag))

    # Use vectorized runtime sampler to get reference d
    d_rc = rc.sample_bilinear_batch(flat, res, -1.5, 1.5, -1.5, 1.5, [real], [imag])[0]
    grad_rc = rc_df.gradient(real, imag)
    d_t = float(tdf.sample_bilinear(c_r, c_i).item())
    gx, gy = tdf.gradient(c_r, c_i)
    gx = float(gx.item())
    gy = float(gy.item())

    print(f"idx={idx} real={real:.6f} imag={imag:.6f} u=({u_real:.6f},{u_imag:.6f}) h={h}")
    print(f"  rc out: {out_py.real:.6f}, {out_py.imag:.6f}")
    print(f"  torch out: {float(nr.item()):.6f}, {float(ni.item()):.6f}")
    print(f"  err: real={err_real:.6f}, imag={err_imag:.6f}")
    print(f"  d: rc={d_rc:.6f}, torch={d_t:.6f}")
    print(f"  grad: rc=({grad_rc[0]:.6f},{grad_rc[1]:.6f}), torch=({gx:.6f},{gy:.6f})")
    if idx == 7:
        real_scale = (1.5 - (-1.5)) / res
        imag_scale = (1.5 - (-1.5)) / res
        step_x = real_scale * 0.5
        step_y = imag_scale * 0.5
        coords = [
            (real - step_x, imag),
            (real + step_x, imag),
            (real, imag - step_y),
            (real, imag + step_y),
        ]
        print('  debug samples (runtime sampler):')
        for (rx, ry) in coords:
            v = rc.sample_bilinear_batch(flat, res, -1.5, 1.5, -1.5, 1.5, [rx], [ry])[0]
            print(f"    ({rx:.6f},{ry:.6f}) -> {v:.6f}")
        print('  debug samples (torch sampler):')
        for (rx, ry) in coords:
            val = tdf.sample_bilinear(torch.tensor([rx],dtype=torch.float32), torch.tensor([ry],dtype=torch.float32))
            print(f"    ({rx:.6f},{ry:.6f}) -> {float(val.item()):.6f}")
    print('')
