"""Tests for training using minimap context."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset
from src.control_trainer import ControlTrainer
from src.control_model import AudioToControlModel
from src.visual_metrics import LossVisualMetrics


def test_train_epoch_with_curriculum_context_runs():
    # Small model with context_dim 265
    model = AudioToControlModel(
        window_frames=10, n_features_per_frame=6, hidden_dims=[16, 16], context_dim=265
    )
    trainer = ControlTrainer(
        model=model,
        visual_metrics=LossVisualMetrics(),
        device="cpu",
        use_curriculum=True,
    )

    # Create dummy features [batch, window_frames * n_features_per_frame]
    batch_size = 4
    input_dim_base = 10 * 6
    dummy = torch.randn(batch_size, input_dim_base, dtype=torch.float32)
    loader = DataLoader(TensorDataset(dummy), batch_size=batch_size)

    # Should run without errors and return loss dict
    res = trainer.train_epoch(loader, epoch=0)
    assert isinstance(res, dict)
    assert "loss" in res
