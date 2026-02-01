import pytest
import torch

from backend.src.control_model import AudioToControlModel
from backend.src.control_trainer import ControlTrainer
from backend.src.visual_metrics import LossVisualMetrics
from torch.utils.data import DataLoader, TensorDataset


class FakeFeatureExtractor:
    def __init__(self, n_features_per_frame=6):
        self._n = n_features_per_frame
        self.feature_mean = None
        self.feature_std = None

    def num_features_per_frame(self):
        return self._n

    def compute_normalization_stats(self, windows):
        self.feature_mean = 0.0
        self.feature_std = 1.0

    def normalize_features(self, row):
        return [float(v) for v in row]


def test_trainer_aborts_on_nan_model_params():
    model = AudioToControlModel(
        window_frames=2, n_features_per_frame=6, hidden_dims=[16], context_dim=0
    )
    # Introduce NaN in parameters
    for n, p in model.named_parameters():
        p.data.fill_(float("nan"))
        break

    trainer = ControlTrainer(
        model=model,
        visual_metrics=LossVisualMetrics(),
        feature_extractor=FakeFeatureExtractor(n_features_per_frame=6),
        device="cpu",
        use_curriculum=False,
    )

    # Create a single-window dataset matching the input dim
    sample = torch.zeros((1, model.input_dim), dtype=torch.float32)
    ds = TensorDataset(sample)
    dl = DataLoader(ds, batch_size=1)

    with pytest.raises(
        RuntimeError, match="Model parameters contain NaNs/InFs before forward"
    ):
        trainer.train_epoch(dl, epoch=0)
