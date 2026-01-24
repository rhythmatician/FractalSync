"""Differentiable proxy renderer for quick visual losses.

Provides a small, vectorized PyTorch renderer that computes an escape-time
potential/DE-like map for a set of complex seeds `c = x + i y`. It is not a
full Julia renderer (no color shading) and is designed to be cheap and
fully differentiable for training losses.

API:
- ProxyRenderer(resolution=64, max_iter=20)
- render(c_real: Tensor[N], c_imag: Tensor[N]) -> Tensor[N, H, W]
- frame_diff(prev: Tensor[N, H, W], curr: Tensor[N, H, W]) -> Tensor[N]

The renderer uses a fixed grid centered on each seed c to compute a local
potential map using a soft-escape heuristic (smooth potential accumulation)
so gradients are non-zero near boundaries.
"""

from __future__ import annotations

import torch


class ProxyRenderer:
    """Fast differentiable proxy renderer.

    Args:
        resolution: int resolution (H = W = resolution)
        max_iter: int number of iterations in the escape-time loop (small, e.g. 20)
        device: optional PyTorch device
    """

    def __init__(
        self, resolution: int = 64, max_iter: int = 20, device: str | None = None
    ):
        self.resolution = int(resolution)
        self.max_iter = int(max_iter)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Precompute grid coordinates in normalized [-1,1] square
        h = self.resolution
        x = torch.linspace(-1.0, 1.0, steps=h, device=self.device)
        y = torch.linspace(-1.0, 1.0, steps=h, device=self.device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        self.registered_grid = torch.stack([xx, yy], dim=-1)  # (H, W, 2)

    def _make_grid_for_seeds(
        self, c_real: torch.Tensor, c_imag: torch.Tensor
    ) -> torch.Tensor:
        # Broadcast grid to (N, H, W, 2) and offset by c
        N = c_real.shape[0]
        grid = self.registered_grid.unsqueeze(0).expand(N, -1, -1, -1).to(c_real.device)
        # scale & translate: small window around c (scale factor chosen heuristically)
        # allow a scale to be tuned later; for now use 2.0 window
        return grid

    def render(self, c_real: torch.Tensor, c_imag: torch.Tensor) -> torch.Tensor:
        """Render proxy potential map for batch of seeds.

        Args:
            c_real, c_imag: tensors shape (N,)
        Returns:
            frames: tensor shape (N, H, W) float32 in [0, 1]
        """
        device = c_real.device
        h = self.resolution
        N = c_real.shape[0]

        # Construct complex grid z0 for each seed: z = grid * scale + c
        # Here grid is in [-1,1]^2; we scale by 1.5 for a reasonable field of view
        grid = self.registered_grid.to(device)  # (H,W,2)
        grid = grid.unsqueeze(0).expand(N, -1, -1, -1)  # (N,H,W,2)

        # offset by c in the real/imag plane; we use small scaling so pixel coords map to complex plane
        scale = 1.5
        c_xy = torch.stack([c_real, c_imag], dim=-1).view(N, 1, 1, 2)
        z = grid * scale + c_xy
        z_real = z[..., 0].clone()
        z_imag = z[..., 1].clone()

        # Accumulate a smooth potential map using log(1+|z|) per iteration
        potential = torch.zeros((N, h, h), dtype=torch.float32, device=device)
        escape_mask = torch.zeros_like(potential, dtype=torch.bool)

        for i in range(self.max_iter):
            # z = z^2 + c
            zr = z_real * z_real - z_imag * z_imag + c_xy[..., 0]
            zi = 2.0 * z_real * z_imag + c_xy[..., 1]
            z_real = zr
            z_imag = zi
            mag = torch.hypot(z_real, z_imag)
            # guard against Inf/NaN
            mag = torch.nan_to_num(mag, nan=0.0, posinf=1e6, neginf=0.0)
            # soft potential contribution; use log1p for smoother gradients
            potential = potential + torch.log1p(mag)
            # mark escaped (for optional early-stop logging)
            escaped = mag > 4.0
            escape_mask = escape_mask | escaped
        # Normalize potential to roughly [0,1] per-sample
        flat = potential.view(N, -1)
        minv = flat.min(dim=1, keepdim=True)[0].view(N, 1, 1)
        maxv = flat.max(dim=1, keepdim=True)[0].view(N, 1, 1)
        potential = torch.nan_to_num(potential, nan=0.0, posinf=1e6, neginf=0.0)
        potential = (potential - minv) / (maxv - minv + 1e-8)
        return potential


def frame_diff(prev: torch.Tensor, curr: torch.Tensor) -> torch.Tensor:
    """Compute mean absolute difference per sample.

    Args:
        prev, curr: Tensors shape (N, H, W)
    Returns:
        diff: Tensor shape (N,) with mean(|curr - prev|)
    """
    assert prev.shape == curr.shape
    return torch.mean(torch.abs(curr - prev), dim=[1, 2])
