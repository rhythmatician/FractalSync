import numpy as np
import torch
from src.policy_interface import policy_output_decoder_torch
from src.runtime_core_bridge import (
    make_orbit_state,
    synthesize,
    DEFAULT_ORBIT_SEED,
    DEFAULT_RESIDUAL_OMEGA_SCALE,
    make_residual_params,
)


def test_soft_lobe_interpolation_favors_biased_lobe():
    # Setup fake model output: batch=1, k_bands=6, trailing 2 lobe logits
    B = 1
    k = 6
    out_dim = 5 + k + 2
    # Zero outputs except trailing lobe logits bias towards lobe 1
    out = torch.zeros((B, out_dim), dtype=torch.float32)
    out[0, -2] = -1.0
    out[0, -1] = 10.0

    decoded = policy_output_decoder_torch(out, k_bands=k)
    assert "lobe_logits" in decoded
    l_logits = decoded["lobe_logits"][0]
    weights = torch.softmax(l_logits, dim=-1).numpy()

    # Synthesize per-lobe and mix
    s = 1.02
    alpha = 0.3
    omega = 1.0
    theta = 0.0
    gates = [0.0] * k

    residuals = make_residual_params(k_residuals=k)
    c_list = []
    for l_idx in range(len(weights)):
        orb = make_orbit_state(
            lobe=int(l_idx),
            sub_lobe=0,
            theta=theta,
            omega=float(omega),
            s=float(s),
            alpha=float(alpha),
            k_residuals=k,
            residual_omega_scale=DEFAULT_RESIDUAL_OMEGA_SCALE,
            seed=int(DEFAULT_ORBIT_SEED + 0),
        )
        c = synthesize(orb, residuals, gates)
        c_list.append(c)

    # Weighted mix
    # Ensure softmax weights favor the biased lobe (index 1)
    assert int(np.argmax(weights)) == 1
