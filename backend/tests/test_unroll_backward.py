import torch
import numpy as np

from src.control_model import PolicyModel
from src.control_trainer import ControlTrainer
from src.visual_metrics import VisualMetrics
from src.runtime_core_bridge import make_feature_extractor


def test_unroll_training_grad_flow():
    # model config matches default input dim
    model = PolicyModel(window_frames=10, n_features_per_frame=6)
    # Ensure required precomputed distance field exists for this test
    import numpy as np
    import json as _json
    from pathlib import Path as _Path

    df_base = _Path("data") / "mandelbrot_distance_field"
    df_base.parent.mkdir(parents=True, exist_ok=True)
    res = 4
    field = np.arange(res * res, dtype=np.float32).reshape((res, res))
    np.save(str(df_base.with_suffix(".npy")), field)
    meta = {"resolution": res, "real_range": (-1.5, 1.5), "imag_range": (-1.5, 1.5), "max_distance": 1.0, "slowdown_threshold": 0.05}
    with open(str(df_base.with_suffix(".json")), "w") as f:
        _json.dump(meta, f)

    trainer = ControlTrainer(
        model=model,
        visual_metrics=VisualMetrics(),
        feature_extractor=make_feature_extractor(),
        device="cpu",
        policy_mode=True,
        sequence_training=True,
        sequence_unroll_steps=5,
        use_amp=False,
        use_curriculum=False,
    )

    # Create synthetic sequence dataset: 4 sequences, seq_len=5
    N = 4
    seq_len = 5
    input_dim = model.input_dim
    seqs = np.random.randn(N, seq_len, input_dim).astype("float32")
    td = torch.utils.data.TensorDataset(torch.from_numpy(seqs))
    loader = torch.utils.data.DataLoader(td, batch_size=2)

    # Run one epoch
    avg_losses = trainer.train_epoch(loader, epoch=1)

    # Ensure some gradient was applied (params have grad after step)
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)
    # Loss is finite
    assert np.isfinite(avg_losses["loss"])
