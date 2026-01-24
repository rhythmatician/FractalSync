"""Fast NaN debugging - run just a few batches to catch the issue."""

import torch
import numpy as np
from src.control_trainer import ControlTrainer
from src.control_model import AudioToControlModel
from src.visual_metrics import VisualMetrics
from src.runtime_core_bridge import make_feature_extractor


def test_epoch2_nan():
    """Reproduce epoch 2 NaN with minimal batches."""

    # Minimal setup
    device = "cpu"
    model = AudioToControlModel(window_frames=10, n_features_per_frame=6, k_bands=6)
    visual_metrics = VisualMetrics()
    feature_extractor = make_feature_extractor()

    # Ensure required precomputed distance field exists for this test
    import json as _json
    from pathlib import Path as _Path

    df_base = _Path("data") / "mandelbrot_distance_field"
    df_base.parent.mkdir(parents=True, exist_ok=True)
    res = 4
    field = np.arange(res * res, dtype=np.float32).reshape((res, res))
    np.save(str(df_base.with_suffix(".npy")), field)
    meta = {
        "resolution": res,
        "real_range": (-1.5, 1.5),
        "imag_range": (-1.5, 1.5),
        "max_distance": 1.0,
        "slowdown_threshold": 0.05,
    }
    with open(str(df_base.with_suffix(".json")), "w") as f:
        _json.dump(meta, f)

    trainer = ControlTrainer(
        model=model,
        visual_metrics=visual_metrics,
        feature_extractor=feature_extractor,
        device=device,
        use_amp=False,
        julia_renderer=None,  # CPU rendering only for speed
        julia_resolution=32,  # Tiny for speed
        julia_max_iter=20,  # Fast
        trajectory_steps=5,  # Few steps
        render_fraction=0.1,  # Minimal rendering
    )

    # Create fake data - just 2 batches
    batch_size = 8
    features = torch.randn(batch_size, 60)
    dataset = torch.utils.data.TensorDataset(features)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    # Normalize stats
    feature_extractor.feature_mean = np.zeros(60)
    feature_extractor.feature_std = np.ones(60)

    print("Testing epoch 0 (static mode)...")
    losses_epoch0 = trainer.train_epoch(dataloader, epoch=0, curriculum_decay=0.95)
    print(f"Epoch 0 loss: {losses_epoch0['loss']:.4f}")
    assert np.isfinite(losses_epoch0["loss"]), "Epoch 0 has NaN!"

    print("\nTesting epoch 1 (trajectory mode)...")
    losses_epoch1 = trainer.train_epoch(dataloader, epoch=1, curriculum_decay=0.95)
    print(f"Epoch 1 loss: {losses_epoch1['loss']:.4f}")

    if not np.isfinite(losses_epoch1["loss"]):
        print("\n❌ FOUND NaN IN EPOCH 1!")
        print("Loss breakdown:")
        for k, v in losses_epoch1.items():
            print(f"  {k}: {v}")
        return False
    else:
        print("\n✓ No NaN detected")
        return True


if __name__ == "__main__":
    success = test_epoch2_nan()
    exit(0 if success else 1)
