import torch
from backend.src.control_trainer import CorrelationLoss
from backend.src.control_model import AudioToControlModel


def test_loudness_distance_loss_backward_no_nan_grads():
    # create a tiny model and fake features
    model = AudioToControlModel(
        window_frames=1, n_features_per_frame=6, hidden_dims=[8], context_dim=0
    )
    model.eval()

    # constant features -> constant RMS across batch
    batch = torch.zeros((4, model.input_dim), dtype=torch.float32, requires_grad=False)

    # predicted controls (use forward pass to build computational graph)
    out = model(batch)
    parsed = model.parse_output(out)
    delta_real = parsed["delta_real"]
    delta_imag = parsed["delta_imag"]

    c_tensor = torch.stack([delta_real, delta_imag], dim=1)
    julia_real = c_tensor[:, 0]
    julia_imag = c_tensor[:, 1]

    # spectral rms and distance should depend on model outputs so backward flows to model
    spectral_rms = 0.5 + 0.0 * delta_real
    distance_tensor = 0.5 + 0.0 * delta_real

    loss_fn = CorrelationLoss()
    # negative spectral_rms vs distance
    loud_dist = loss_fn(-spectral_rms, distance_tensor)

    assert torch.isfinite(loud_dist).item()

    # Backprop through the scalar should not produce NaN in model grads
    loud_dist.backward()
    for n, p in model.named_parameters():
        if p.grad is not None:
            assert torch.all(torch.isfinite(p.grad)).item()
