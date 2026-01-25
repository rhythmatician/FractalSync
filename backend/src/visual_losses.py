"""Trainable visual loss functions for FractalSync.

Provides:
- MultiscaleDeltaVLoss: mean absolute difference between multiscale proxy frames
- SpeedBoundLoss: penalize steps larger than df-scaled allowed speed
- HitAlignmentLoss: correlate band energies with gate predictions
- CoverageLoss: encourage angular coverage across a sliding window
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from .differentiable_integrator import TorchDistanceField


class MultiscaleDeltaVLoss(torch.nn.Module):
    def __init__(self, weight: float = 1.0, scales: list[int] = [1, 2, 4]):
        super().__init__()
        self.weight = float(weight)
        self.scales = list(scales)

    def forward(
        self,
        prev_frames: torch.Tensor,
        curr_frames: torch.Tensor,
        h_t: Optional[torch.Tensor] = None,
        budget_idle: float = 0.02,
        budget_hit: float = 0.08,
    ) -> torch.Tensor:
        """Compute a budgeted multiscale ΔV loss.

        prev_frames, curr_frames: shape (N, H, W)
        h_t: optional shape (N,) normalized transient strength (0..1)
        Returns scalar loss
        """
        assert prev_frames.shape == curr_frames.shape
        N, H, W = prev_frames.shape
        losses = []
        for s in self.scales:
            if s == 1:
                p = prev_frames
                c = curr_frames
            else:
                p = F.avg_pool2d(
                    prev_frames.unsqueeze(1), kernel_size=s, stride=s
                ).squeeze(1)
                c = F.avg_pool2d(
                    curr_frames.unsqueeze(1), kernel_size=s, stride=s
                ).squeeze(1)
            delta = torch.mean(torch.abs(c - p), dim=[1, 2])  # (N,)
            losses.append(delta)
        multi = torch.stack(losses, dim=0)  # (len(scales), N)
        # per-sample mean across scales
        per_sample = torch.mean(multi, dim=0)
        if h_t is None:
            h_t = torch.zeros_like(per_sample)
        h_t = torch.clamp(h_t, 0.0, 1.0)
        budget = budget_idle * (1.0 - h_t) + budget_hit * h_t
        loss = torch.mean(torch.relu(per_sample - budget) ** 2)
        return self.weight * loss

    @staticmethod
    def compute_delta_v(prev_frames: torch.Tensor, curr_frames: torch.Tensor) -> torch.Tensor:
        """Return per-sample multiscale ΔV without applying any budget."""
        assert prev_frames.shape == curr_frames.shape
        losses = []
        for s in [1, 2, 4]:
            if s == 1:
                p = prev_frames
                c = curr_frames
            else:
                p = F.avg_pool2d(prev_frames.unsqueeze(1), kernel_size=s, stride=s).squeeze(1)
                c = F.avg_pool2d(curr_frames.unsqueeze(1), kernel_size=s, stride=s).squeeze(1)
            delta = torch.mean(torch.abs(c - p), dim=[1, 2])
            losses.append(delta)
        multi = torch.stack(losses, dim=0)
        return torch.mean(multi, dim=0)


class SpeedBoundLoss(torch.nn.Module):
    def __init__(self, weight: float = 1.0, base_step: float = 0.03):
        super().__init__()
        self.weight = float(weight)
        self.base_step = float(base_step)

    def forward(
        self,
        c_prev: torch.Tensor,
        c_curr: torch.Tensor,
        df: Optional[TorchDistanceField] = None,
    ) -> torch.Tensor:
        """Penalize squared excess over allowed step per-sample.

        c_prev, c_curr: (N,2) tensors
        df: optional TorchDistanceField for local slowdown scaling
        """
        dxy = c_curr - c_prev
        step_mag = torch.hypot(dxy[:, 0], dxy[:, 1])
        if df is not None:
            # sample velocity scale at c_prev
            scale = df.get_velocity_scale(c_prev[:, 0], c_prev[:, 1])
            allowed = self.base_step * (0.5 + 0.5 * scale)
        else:
            allowed = torch.full_like(step_mag, self.base_step)
        excess = F.relu(step_mag - allowed)
        return self.weight * torch.mean(excess * excess)


class HitAlignmentLoss(torch.nn.Module):
    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = float(weight)

    def forward(
        self, band_energy: torch.Tensor, gate_pred: torch.Tensor
    ) -> torch.Tensor:
        """Encourage gate_pred (N,k) to align with band_energy (N,k).

        Use normalized MSE/correlation-style loss.
        """
        # Normalize across bands
        be = band_energy - band_energy.mean(dim=1, keepdim=True)
        gp = gate_pred - gate_pred.mean(dim=1, keepdim=True)
        # per-sample negative Pearson correlation (we want positive correlation)
        num = torch.sum(be * gp, dim=1)
        den = torch.sqrt(torch.sum(be * be, dim=1) * torch.sum(gp * gp, dim=1) + 1e-12)
        corr = num / (den + 1e-12)
        # loss = 1 - mean(corr) (so higher correlation -> lower loss)
        return self.weight * torch.mean(1.0 - corr)


class CoverageLoss(torch.nn.Module):
    def __init__(self, weight: float = 1.0, bins: int = 16):
        super().__init__()
        self.weight = float(weight)
        self.bins = int(bins)

    def forward(self, c_seq: torch.Tensor) -> torch.Tensor:
        """Encourage angular coverage across a sequence.

        c_seq: shape (N, T, 2)
        Returns scalar loss = mean(1 - occupancy)
        """
        N, T, _ = c_seq.shape
        angles = torch.atan2(c_seq[:, :, 1], c_seq[:, :, 0])  # (N, T)
        # Map to [0, 2pi)
        bins = torch.linspace(
            -3.14159265, 3.14159265, steps=self.bins + 1, device=c_seq.device
        )
        occupancy = []
        for i in range(N):
            hist = torch.histc(
                angles[i], bins=self.bins, min=bins[0].item(), max=bins[-1].item()
            )
            occ = torch.sum(hist > 0).to(torch.float32) / float(self.bins)
            occupancy.append(occ)
        occ = torch.stack(occupancy)
        return self.weight * torch.mean(1.0 - occ)
