"""Runtime inference helpers: probe computation and policy step wrapper.

This module provides small, testable helpers that implement the steps described
in Phase 6: compute DF directional probes, run a policy (ONNX or PyTorch),
apply deltas, and synthesize next c using runtime-core helpers.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union
import math

import numpy as np

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    import torch
except Exception:
    torch = None

from .policy_interface import policy_state_encoder, policy_output_decoder
from .runtime_core_bridge import make_orbit_state, synthesize


def compute_directional_probes(
    c_real: float,
    c_imag: float,
    radii: Sequence[float] = (0.005, 0.02),
    directions: int = 8,
) -> List[float]:
    """Compute directional probe samples around c.

    Returns a flat list in order (dir0_r1, dir0_r2, dir1_r1, dir1_r2, ...)
    matching the POLICY_ONNX_SPEC expectation (8 directions × 2 radii = 16 floats).
    """
    probes: List[float] = []
    for d in range(directions):
        ang = 2.0 * math.pi * d / float(directions)
        dx = math.cos(ang)
        dy = math.sin(ang)
        for r in radii:
            x = float(c_real + r * dx)
            y = float(c_imag + r * dy)
            # Convert probe coordinate into a single scalar probe value. The
            # policy interface expects 1 float per probe (8 directions × 2
            # radii = 16 floats). Here we provide the Euclidean distance of
            # the probe from the origin which is a stable, deterministic
            # scalar placeholder until callers perform DF lookups.
            probes.append(float(math.hypot(x, y)))
    return probes


def run_policy_step(
    model: Union[str, object],
    s: float,
    alpha: float,
    omega: float,
    theta: float,
    h_t: float,
    loudness: float,
    tonalness: float,
    noisiness: float,
    band_energies: Sequence[float],
    band_deltas: Optional[Sequence[float]] = None,
    d_c: float = 1.0,
    grad: Tuple[float, float] = (0.0, 0.0),
    directional_probes: Optional[Sequence[float]] = None,
) -> dict:
    """Run a single policy inference and apply deltas to return new orbit/c.

    model may be either a path to an ONNX model (string) or a PyTorch nn.Module
    (callable that accepts a numpy or torch tensor). This function returns a
    dict with decoded outputs and the new seed complex coordinate `c`.
    """
    inp = policy_state_encoder(
        s=s,
        alpha=alpha,
        omega=omega,
        theta=theta,
        h_t=h_t,
        loudness=loudness,
        tonalness=tonalness,
        noisiness=noisiness,
        band_energies=list(band_energies),
        band_deltas=list(band_deltas) if band_deltas is not None else None,
        d_c=d_c,
        grad=grad,
        directional_probes=(
            list(directional_probes) if directional_probes is not None else None
        ),
    )

    if isinstance(model, str):
        if ort is None:
            raise RuntimeError("onnxruntime is not installed")
        sess = ort.InferenceSession(model)
        inp_name = sess.get_inputs()[0].name
        arr = inp.astype(np.float32)
        # Ensure batch dimension
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        out = sess.run(None, {inp_name: arr})[0]
        # out shape (batch, out_dim) -> take first batch
        decoded = policy_output_decoder(out[0].tolist(), k_bands=len(band_energies))
    else:
        # Expect a PyTorch model or callable
        if torch is not None and hasattr(model, "to"):
            model.eval()
            with torch.no_grad():
                t = torch.tensor(inp, dtype=torch.float32).unsqueeze(0)
                out_t = model(t).cpu().numpy()
            decoded = policy_output_decoder(
                out_t[0].tolist(), k_bands=len(band_energies)
            )
        else:
            # model is a simple callable that accepts numpy
            out = model(inp.astype(np.float32))
            out_arr = np.asarray(out, dtype=np.float32)
            decoded = policy_output_decoder(
                out_arr.tolist(), k_bands=len(band_energies)
            )

    # Apply deltas to produce new orbit state and synthesize c
    # Use make_orbit_state and synthesize (non-differentiable deterministic)
    deltas = decoded
    # s, alpha, omega, theta -> apply simple updates (mirror apply_policy_deltas)
    s_new = float(max(0.1, min(10.0, s + float(deltas.get("delta_s", 0.0)))))
    omega_new = float(
        max(-10.0, min(10.0, omega + float(deltas.get("delta_omega", 0.0))))
    )
    alpha_hit = float(deltas.get("alpha_hit", 0.0))
    alpha_new = float(max(0.0, min(5.0, alpha + float(h_t) * alpha_hit)))

    # Build orbit state and synthesize c
    orbit = make_orbit_state(
        lobe=1, sub_lobe=0, theta=theta, omega=omega_new, s=s_new, alpha=alpha_new
    )
    c_new = synthesize(orbit, None, None)

    return {
        "decoded": decoded,
        "orbit": orbit,
        "c_new": (float(c_new.real), float(c_new.imag)),
    }
