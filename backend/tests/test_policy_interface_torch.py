import torch
import numpy as np

from src.policy_interface import policy_output_decoder_torch, apply_policy_deltas_torch


def test_policy_output_decoder_torch_and_apply():
    N = 4
    k = 3
    # Create random outputs
    out = torch.randn(N, 5 + k)
    out[:, 0:2] = torch.clamp(out[:, 0:2], -2.0, 2.0)  # u
    decoded = policy_output_decoder_torch(out, k)

    assert decoded["u"].shape == (N, 2)
    assert decoded["delta_s"].shape == (N,)
    assert decoded["gate_logits"].shape == (N, k)

    # Apply deltas
    s = torch.ones(N) * 1.0
    alpha = torch.zeros(N)
    omega = torch.ones(N) * 0.5
    theta = torch.zeros(N)

    new = apply_policy_deltas_torch(
        s, alpha, omega, theta, decoded, h_t=torch.ones(N) * 0.5
    )
    assert set(new.keys()) == {"s", "alpha", "omega"}
    assert new["s"].shape == (N,)
    assert new["alpha"].shape == (N,)
    assert new["omega"].shape == (N,)
