import torch
from src.control_model import PolicyModel


def test_policy_model_forward_shape():
    model = PolicyModel(window_frames=10, n_features_per_frame=6, k_bands=6)
    # batch_size 2 dummy
    x = torch.randn(2, model.input_dim)
    out = model(x)
    assert out.shape == (2, model.output_dim)
    parsed = model.parse_output(out)
    assert parsed["u"].shape == (2, 2)
    assert parsed["gate_logits"].shape == (2, 6)
