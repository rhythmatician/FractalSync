"""Torch-based differentiable approximation of contour-biased integrator (PoC).

This module provides a vectorized `contour_biased_step_torch` function and a
simple `TorchDistanceField` wrapper that supports bilinear sampling and a
finite-difference gradient. It mirrors the logic in
`backend/scripts/tune_contour.py::contour_biased_step_py` so we can verify
parity and enable autograd for unrolled training.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import runtime_core as rc


class TorchDistanceField:
    """Distance field represented as a torch tensor with sampling helpers.

    field: shape (1,1,H,W) float32 tensor (normalized distances in [0,1])
    ranges: real_min, real_max, imag_min, imag_max
    """

    def __init__(
        self,
        field: torch.Tensor,
        real_min: float = -2.0,
        real_max: float = 2.0,
        imag_min: float = -2.0,
        imag_max: float = 2.0,
        max_distance: float = 2.0,
        slowdown_threshold: float = 0.05,
        use_runtime_sampler: bool = False,
    ):
        assert field.ndim == 2 or field.ndim == 4
        if field.ndim == 2:
            field = field.unsqueeze(0).unsqueeze(0)
        elif field.ndim == 4:
            # assume already (1,1,H,W)
            pass
        self.field = field.to(dtype=torch.float32)
        self.real_min = float(real_min)
        self.real_max = float(real_max)
        self.imag_min = float(imag_min)
        self.imag_max = float(imag_max)
        self.max_distance = float(max_distance)
        self.slowdown_threshold = float(slowdown_threshold)
        _, _, self.H, self.W = self.field.shape
        self.use_runtime_sampler = bool(use_runtime_sampler)
        self._runtime_field_flat = None
        if self.use_runtime_sampler:
            # prepare flattened python list for runtime_core sampler (CPU only)
            self._runtime_field_flat = (
                self.field.detach()
                .cpu()
                .squeeze()
                .numpy()
                .astype("float32")
                .ravel()
                .tolist()
            )

    def _normalize_to_grid(
        self, real: torch.Tensor, imag: torch.Tensor
    ) -> torch.Tensor:
        """Convert real/imag coords (N,) -> grid coords (N,1,1,2) in [-1,1] for grid_sample."""
        # x corresponds to real (width), y to imag (height)
        sx = (real - self.real_min) / (self.real_max - self.real_min)
        sy = (imag - self.imag_min) / (self.imag_max - self.imag_min)
        # grid_sample expects normalized coords in [-1,1]
        gx = sx * 2.0 - 1.0
        gy = sy * 2.0 - 1.0
        # stack as (N,1,1,2) with last dim = (gx, gy)
        g = torch.stack([gx, gy], dim=-1).view(-1, 1, 1, 2)
        return g

    def sample_bilinear(self, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
        """Sample field at given (real, imag), returning (N,) tensor."""
        g = self._normalize_to_grid(real, imag)
        # grid_sample expects input batch size == grid batch size. Repeat field across batch.
        N = g.shape[0]
        input_field = self.field.expand(N, -1, -1, -1)
        if not self.use_runtime_sampler:
            out = F.grid_sample(
                input_field,
                g,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            # out shape (N,1,1,1) -> Squeeze to (N,)
            out = out.view(-1)
            return out
        # else: use runtime_core exact sampling semantics (non-differentiable)

        N = g.shape[0]
        # extract real/imags as python lists
        # runtime_core's batch sampler expects a prepared flattened field
        assert self._runtime_field_flat is not None, "runtime field not prepared"
        reals = real.detach().cpu().numpy().tolist()
        imags = imag.detach().cpu().numpy().tolist()
        vals = rc.sample_bilinear_batch(
            self._runtime_field_flat,
            self.W,
            self.real_min,
            self.real_max,
            self.imag_min,
            self.imag_max,
            reals,
            imags,
        )  # runtime_core provides a fast batch bilinear sampler
        # convert back to tensor on same device/dtype
        out_t = torch.tensor(vals, dtype=self.field.dtype, device=self.field.device)
        return out_t

    def gradient(
        self, real: torch.Tensor, imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # finite-difference with one grid cell offsets
        # Use half-grid offsets to match runtime_core implementation and
        # reduce aliasing when sampling near grid cell boundaries.
        real_scale = (self.real_max - self.real_min) / float(self.W)
        imag_scale = (self.imag_max - self.imag_min) / float(self.H)
        step_x = real_scale * 0.5
        step_y = imag_scale * 0.5
        # When using the runtime sampler, clamp sample coordinates to the
        # valid region to match the runtime_core gradient behaviour.
        if self.use_runtime_sampler:
            # real and imag may be batched tensors (N,), so clamp elementwise
            left_x = torch.clamp(real - step_x, min=self.real_min, max=self.real_max)
            right_x = torch.clamp(real + step_x, min=self.real_min, max=self.real_max)
            down_y = torch.clamp(imag - step_y, min=self.imag_min, max=self.imag_max)
            up_y = torch.clamp(imag + step_y, min=self.imag_min, max=self.imag_max)

            left = self.sample_bilinear(left_x, imag)
            right = self.sample_bilinear(right_x, imag)
            down = self.sample_bilinear(real, down_y)
            up = self.sample_bilinear(real, up_y)
        else:
            left = self.sample_bilinear(real - step_x, imag)
            right = self.sample_bilinear(real + step_x, imag)
            down = self.sample_bilinear(real, imag - step_y)
            up = self.sample_bilinear(real, imag + step_y)

        gx = (right - left) / (2.0 * step_x)
        gy = (up - down) / (2.0 * step_y)
        return gx, gy

    def get_velocity_scale(
        self, real: torch.Tensor, imag: torch.Tensor
    ) -> torch.Tensor:
        d = self.sample_bilinear(real, imag)
        th = self.slowdown_threshold
        # s = t*t*(3-2t) with t = d/th for d < th
        t = torch.clamp(d / th, min=0.0, max=1.0)
        s = t * t * (3.0 - 2.0 * t)
        return s


def contour_biased_step_torch(
    c_real: torch.Tensor,
    c_imag: torch.Tensor,
    u_real: torch.Tensor,
    u_imag: torch.Tensor,
    h: torch.Tensor,
    d_star: float,
    max_step: float,
    df: Optional[TorchDistanceField] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Vectorized torch implementation mirroring contour_biased_step_py.

    All inputs are (N,) tensors. Returns next c_real, c_imag as (N,) tensors.
    """
    # gradient
    if df is None:
        # if no DF, fall back to simple scaled u (respect max_step)
        u_mag = torch.hypot(u_real, u_imag)
        scale = torch.where(
            (u_mag > max_step) & (u_mag > 0.0), max_step / u_mag, torch.ones_like(u_mag)
        )
        return c_real + u_real * scale, c_imag + u_imag * scale

    gx, gy = df.gradient(c_real, c_imag)
    grad_norm = torch.hypot(gx, gy)

    # where grad_norm <= eps, treat as near-zero gradient and use scaled u as a safe path
    eps = 1e-12
    near_zero_mask = grad_norm <= eps

    # Compute normal and tangent
    nx = gx / (grad_norm + eps)
    ny = gy / (grad_norm + eps)
    tx = -gy / (grad_norm + eps)
    ty = gx / (grad_norm + eps)

    proj_t = u_real * tx + u_imag * ty
    proj_n = u_real * nx + u_imag * ny

    normal_scale_no_hit = 0.05
    normal_scale_hit = 1.0
    normal_scale = normal_scale_no_hit + (
        normal_scale_hit - normal_scale_no_hit
    ) * torch.clamp(h, 0.0, 1.0)
    tangential_scale = 1.0
    servo_gain = 0.2

    d = df.sample_bilinear(c_real, c_imag)
    servo = servo_gain * (d_star - d)

    dx = tx * (proj_t * tangential_scale) + nx * (proj_n * normal_scale + servo)
    dy = ty * (proj_t * tangential_scale) + ny * (proj_n * normal_scale + servo)

    mag = torch.hypot(dx, dy)
    s = torch.where(
        (mag > max_step) & (mag > 0.0), max_step / mag, torch.ones_like(mag)
    )
    dx = dx * s
    dy = dy * s

    next_real = c_real + dx
    next_imag = c_imag + dy

    # Handle near-zero gradient cases by using scaled u as a safe path
    if near_zero_mask.any():
        u_mag = torch.hypot(u_real, u_imag)
        scale2 = torch.where(
            (u_mag > max_step) & (u_mag > 0.0), max_step / u_mag, torch.ones_like(u_mag)
        )
        next_real = torch.where(near_zero_mask, c_real + u_real * scale2, next_real)
        next_imag = torch.where(near_zero_mask, c_imag + u_imag * scale2, next_imag)

    # Optional debug guard to catch non-finite intermediate tensors in CI/local debugging.
    # Enable by setting environment variable FRACTALSYNC_DEBUG_NAN=1 when running tests.
    try:
        import os

        if os.getenv("FRACTALSYNC_DEBUG_NAN") == "1":
            checks = [
                ("gx", gx),
                ("gy", gy),
                ("grad_norm", grad_norm),
                ("dx", dx),
                ("dy", dy),
                ("mag", mag),
                ("next_real", next_real),
                ("next_imag", next_imag),
            ]
            for name, tensor in checks:
                if not torch.all(torch.isfinite(tensor)):
                    # Raise with some diagnostics so we can capture shapes and small slices.
                    raise RuntimeError(
                        f"Non-finite detected in differentiable_integrator: {name}; "
                        f"shape={tuple(tensor.shape)}; values={tensor.detach().cpu()[:8]}"
                    )
    except Exception:
        # Important: don't let the debug check crash normal runs; re-raise only for debug mode
        raise

    return next_real, next_imag
