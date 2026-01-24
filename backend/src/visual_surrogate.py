"""Surrogate model to predict ΔV given local geometry and step.

Provides a small MLP-based model and dataset wrapper for training/prediction.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset


class SurrogateDataset(Dataset):
    """Loads a surrogate .pt dataset produced by scripts/generate_surrogate_data.py"""

    def __init__(self, path: str):
        data = torch.load(path)
        self.c_prev = data["c_prev"]
        self.c_next = data["c_next"]
        self.d_prev = data.get("d_prev", torch.ones(len(self.c_prev)))
        self.grad_prev = data.get("grad_prev", torch.zeros((len(self.c_prev), 2)))
        self.delta_v = data["delta_v"]

    def __len__(self):
        return len(self.c_prev)

    def __getitem__(self, idx):
        return {
            "c_prev": self.c_prev[idx],
            "c_next": self.c_next[idx],
            "d_prev": self.d_prev[idx],
            "grad_prev": self.grad_prev[idx],
            "delta_v": self.delta_v[idx],
        }


class SurrogateDeltaV(nn.Module):
    """Small MLP surrogate predicting ΔV scalar from local inputs.

    Input features: [c_prev.x, c_prev.y, c_next.x, c_next.y, d_prev, gx, gy]
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7, hidden_dim),
            nn.ELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # ΔV normalized to [0,1]
        )

    def forward(self, c_prev: torch.Tensor, c_next: torch.Tensor, d_prev: Optional[torch.Tensor] = None, grad_prev: Optional[torch.Tensor] = None) -> torch.Tensor:
        # c_prev, c_next: (N,2)
        N = c_prev.shape[0]
        if d_prev is None:
            d_prev = torch.ones((N,), dtype=c_prev.dtype, device=c_prev.device)
        if grad_prev is None:
            grad_prev = torch.zeros((N, 2), dtype=c_prev.dtype, device=c_prev.device)

        x = torch.cat([c_prev, c_next, d_prev.view(-1, 1), grad_prev], dim=1).to(dtype=c_prev.dtype)
        out = self.net(x).view(-1)
        return out

    def predict(self, *args, **kwargs):
        self.eval()
        with torch.no_grad():
            return self.forward(*args, **kwargs)

    @staticmethod
    def save_checkpoint(model: "SurrogateDeltaV", path: str):
        torch.save({"state_dict": model.state_dict()}, path)

    @staticmethod
    def load_checkpoint(path: str, device: Optional[str] = None) -> "SurrogateDeltaV":
        ck = torch.load(path, map_location=device or "cpu")
        model = SurrogateDeltaV()
        model.load_state_dict(ck["state_dict"])
        return model
