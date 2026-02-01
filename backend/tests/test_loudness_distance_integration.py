import torch
from backend.src.control_model import AudioToControlModel


def test_loudness_distance_loss_gradients_reach_model():
    # Small model
    model = AudioToControlModel(
        window_frames=1, n_features_per_frame=6, hidden_dims=[16], context_dim=0
    )
    model.train()

    # Create a batch where features are constant (so RMS is constant)
    batch = torch.zeros((4, model.input_dim), dtype=torch.float32, requires_grad=False)

    out = model(batch)
    parsed = model.parse_output(out)
    delta_real = parsed["delta_real"]
    delta_imag = parsed["delta_imag"]

    c_tensor = torch.stack([delta_real, delta_imag], dim=1)
    julia_real = c_tensor[:, 0]
    julia_imag = c_tensor[:, 1]

    spectral_rms = 0.5 + 0.0 * delta_real

    # Differentiable radial distance (same as trainer uses)
    DOMAIN_R = 2.0
    c_abs = torch.sqrt(julia_real**2 + julia_imag**2)
    distance_tensor = torch.clamp(c_abs / DOMAIN_R, 0.0, 1.0)

    # correlation loss
    from backend.src.control_trainer import CorrelationLoss

    loss_fn = CorrelationLoss()

    loud_dist = loss_fn(-spectral_rms, distance_tensor)
    assert torch.isfinite(loud_dist).item()

    # Backprop -- gradients should flow into the model parameters via delta -> c_abs
    loud_dist.backward()
    found_grad = False
    for n, p in model.named_parameters():
        if p.grad is not None and torch.any(torch.isfinite(p.grad)).item():
            found_grad = True
            break
    assert (
        found_grad
    ), "Expected finite gradients on model parameters from loudness-distance loss"
