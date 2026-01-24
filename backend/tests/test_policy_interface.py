import numpy as np
from src.policy_interface import (
    policy_state_encoder,
    policy_output_decoder,
    apply_policy_deltas,
)
import runtime_core as rc


def test_policy_encode_shape_and_roundtrip():
    k = 6
    arr = policy_state_encoder(
        s=1.02,
        alpha=0.3,
        omega=1.0,
        theta=0.0,
        h_t=0.5,
        loudness=0.1,
        tonalness=0.2,
        noisiness=0.05,
        band_energies=[0.1] * k,
        band_deltas=[0.0] * k,
        d_c=0.2,
        grad=(0.01, -0.02),
        directional_probes=[0.0] * 16,
    )

    assert arr.dtype == np.float32
    assert arr.shape == (27 + 2 * k,)

    # Fake policy output
    out = np.zeros(5 + k, dtype=np.float32)
    out[0:2] = [0.01, -0.02]
    out[2] = 0.05
    out[3] = -0.1
    out[4] = 0.5
    out[5 : 5 + k] = np.linspace(-1, 1, k)

    decoded = policy_output_decoder(out, k)
    assert "u" in decoded and decoded["u"].shape == (2,)
    assert abs(decoded["delta_s"] - 0.05) < 1e-6

    # Apply to orbit state
    s = 1.02
    state = rc.OrbitState(
        lobe=1,
        sub_lobe=0,
        theta=0.0,
        omega=1.0,
        s=s,
        alpha=0.3,
        k_residuals=k,
        residual_omega_scale=1.0,
        seed=42,
    )
    # Apply deltas via primitives since OrbitState doesn't expose direct attributes
    new_state, stats = apply_policy_deltas(
        s=1.02,  # placeholder - use known s
        alpha=0.3,
        omega=1.0,
        theta=0.0,
        deltas=decoded,
        h_t=0.5,
        k_residuals=k,
        residual_omega_scale=1.0,
        lobe=1,
        sub_lobe=0,
        seed=42,
    )
    # Basic sanity on returned stats
    assert "s" in stats and "alpha" in stats and "omega" in stats
    # New state's synthesized point should differ from the original
    rp = rc.ResidualParams(k_residuals=k, residual_cap=0.5, radius_scale=1.0)
    c_before = state.synthesize(rp, None)
    c_after = new_state.synthesize(rp, None)
    assert (abs(c_before.real - c_after.real) > 1e-6) or (
        abs(c_before.imag - c_after.imag) > 1e-6
    )
