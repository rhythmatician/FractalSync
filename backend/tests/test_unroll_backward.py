import torch
import numpy as np

from src.control_model import PolicyModel
from src.control_trainer import ControlTrainer
from src.visual_metrics import VisualMetrics
from src.runtime_core_bridge import make_feature_extractor


def test_unroll_training_grad_flow():
    # model config matches default input dim
    model = PolicyModel(window_frames=10, n_features_per_frame=6)
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
