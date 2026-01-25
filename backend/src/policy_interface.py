"""Policy encoder/decoder and application helpers for orbit_policy models.

Provides deterministic encoding of orbit state + audio features into a flat input
vector that matches the ONNX contract, and decodes/validates policy outputs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np

import runtime_core as rc

# Optional torch helpers for batched tensor operations
try:
    import torch
except (
    Exception
):  # pragma: no cover - torch may not be available in some test environments
    torch = None


def policy_state_encoder(
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
) -> np.ndarray:
    """Encode inputs into the canonical policy input vector.

    Order follows POLICY_ONNX_SPEC.md. Returns float32 1D numpy array.
    """
    k = len(band_energies)
    band_deltas = band_deltas if band_deltas is not None else [0.0] * k
    if len(band_deltas) != k:
        raise ValueError("band_deltas must match band_energies length")

    if directional_probes is None:
        # 8 directions × 2 radii = 16 values
        directional_probes = [0.0] * 16
    if len(directional_probes) != 16:
        raise ValueError(
            "directional_probes must be length 16 (8 directions × 2 radii)"
        )

    parts: List[float] = []
    # Orbit state
    parts.extend([float(s), float(alpha), float(omega), float(theta)])
    # Slow features
    parts.extend([float(h_t), float(loudness), float(tonalness), float(noisiness)])
    # Per-band energies and deltas
    parts.extend([float(x) for x in band_energies])
    parts.extend([float(x) for x in band_deltas])
    # Geometry minimap
    parts.append(float(d_c))
    parts.extend([float(grad[0]), float(grad[1])])
    parts.extend([float(x) for x in directional_probes])

    arr = np.asarray(parts, dtype=np.float32)
    return arr


def policy_output_decoder(
    output: Sequence[float],
    k_bands: int,
    clamp: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, np.ndarray]:
    """Decode policy model output into named deltas.

    `output` is a flat sequence (u_x, u_y, delta_s, delta_omega, alpha_hit, gate_logits[0..k-1])
    """
    # Allow optional trailing lobe logits (soft-lobe outputs). If present, they
    # will be returned under the "lobe_logits" key as a numpy array.
    if len(output) < 5 + k_bands:
        raise ValueError(
            f"Expected output length at least {5 + k_bands}, got {len(output)}"
        )

    arr = np.asarray(output, dtype=np.float32)
    u = arr[0:2].copy()
    delta_s = float(arr[2])
    delta_omega = float(arr[3])
    alpha_hit = float(arr[4])
    gate_logits = arr[5 : 5 + k_bands].copy()

    lobe_logits = None
    if len(arr) > 5 + k_bands:
        lobe_logits = arr[5 + k_bands :].copy()

    # Clamps
    defaults = {
        "delta_s": (-0.5, 0.5),
        "delta_omega": (-1.0, 1.0),
        "u": (-1.0, 1.0),
        "alpha_hit": (0.0, 2.0),
    }
    clamp = clamp or {}

    def apply_clamp(name: str, val: float) -> float:
        lo, hi = clamp.get(name, defaults.get(name, (-np.inf, np.inf)))
        return float(max(lo, min(hi, val)))

    u[0] = apply_clamp("u", float(u[0]))
    u[1] = apply_clamp("u", float(u[1]))
    delta_s = apply_clamp("delta_s", delta_s)
    delta_omega = apply_clamp("delta_omega", delta_omega)
    alpha_hit = apply_clamp("alpha_hit", alpha_hit)

    result = {
        "u": u,
        "delta_s": delta_s,
        "delta_omega": delta_omega,
        "alpha_hit": alpha_hit,
        "gate_logits": gate_logits,
    }

    if lobe_logits is not None:
        result["lobe_logits"] = lobe_logits

    return result


# -------------------- Torch helpers (batched) --------------------
def policy_output_decoder_torch(
    output, k_bands: int, clamp: Optional[Dict[str, Tuple[float, float]]] = None
) -> Dict[str, torch.Tensor]:
    """Decode batched policy outputs (N, 6 + k_bands) into named tensors.

    Returns dict of tensors: u (N,2), delta_s (N,), delta_omega (N,), alpha_hit (N,), gate_logits (N,k)
    """
    if torch is None:
        raise RuntimeError("torch is required for policy_output_decoder_torch")
    # Satisfy type checkers
    assert torch is not None
    # Allow trailing lobe logits in the batched output. If provided, the
    # per-batch trailing values will be returned under "lobe_logits".
    if output.dim() != 2 or output.size(1) < 5 + k_bands:
        raise ValueError(
            f"Expected output shape (N, >={5 + k_bands}), got {tuple(output.shape)}"
        )

    arr = output.to(dtype=torch.float32)
    u = arr[:, 0:2].clone()
    delta_s = arr[:, 2].clone()
    delta_omega = arr[:, 3].clone()
    alpha_hit = arr[:, 4].clone()
    gate_logits = arr[:, 5 : 5 + k_bands].clone()

    lobe_logits = None
    if output.size(1) > 5 + k_bands:
        lobe_logits = arr[:, 5 + k_bands :].clone()

    # Clamps
    defaults = {
        "delta_s": (-0.5, 0.5),
        "delta_omega": (-1.0, 1.0),
        "u": (-1.0, 1.0),
        "alpha_hit": (0.0, 2.0),
    }
    clamp = clamp or {}

    def apply_clamp_t(name: str, t):
        lo, hi = clamp.get(name, defaults.get(name, (-float("inf"), float("inf"))))
        return torch.clamp(t, min=lo, max=hi)

    u = apply_clamp_t("u", u)
    delta_s = apply_clamp_t("delta_s", delta_s)
    delta_omega = apply_clamp_t("delta_omega", delta_omega)
    alpha_hit = apply_clamp_t("alpha_hit", alpha_hit)

    result = {
        "u": u,
        "delta_s": delta_s,
        "delta_omega": delta_omega,
        "alpha_hit": alpha_hit,
        "gate_logits": gate_logits,
    }

    if lobe_logits is not None:
        result["lobe_logits"] = lobe_logits

    return result


def apply_policy_deltas_torch(
    s,
    alpha,
    omega,
    theta,
    deltas: Dict[str, torch.Tensor],
    h_t=None,
    max_delta_s: float = 0.5,
    max_delta_omega: float = 1.0,
):
    """Apply deltas (batched) to primitive state tensors and return updated tensors.

    s, alpha, omega, theta are (N,) tensors. deltas contains tensors from decoder.
    Returns dict with keys 's', 'alpha', 'omega' (all (N,) tensors).
    """
    if torch is None:
        raise RuntimeError("torch is required for apply_policy_deltas_torch")
    assert torch is not None
    """Apply deltas (batched) to primitive state tensors and return updated tensors.

    s, alpha, omega, theta are (N,) tensors. deltas contains tensors from decoder.
    Returns dict with keys 's', 'alpha', 'omega' (all (N,) tensors).
    """
    if torch is None:
        raise RuntimeError("torch is required for apply_policy_deltas_torch")

    # Ensure floats
    s = s.to(dtype=torch.float32)
    alpha = alpha.to(dtype=torch.float32)
    omega = omega.to(dtype=torch.float32)
    theta = theta.to(dtype=torch.float32)

    ds = deltas.get("delta_s", None)
    domega = deltas.get("delta_omega", None)

    if ds is None:
        ds = torch.zeros_like(s)
    else:
        ds = torch.as_tensor(ds, dtype=torch.float32, device=s.device)

    if domega is None:
        domega = torch.zeros_like(s)
    else:
        domega = torch.as_tensor(domega, dtype=torch.float32, device=s.device)

    s_new = torch.clamp(
        s + torch.clamp(ds, -max_delta_s, max_delta_s), min=0.1, max=10.0
    )
    omega_new = torch.clamp(
        omega + torch.clamp(domega, -max_delta_omega, max_delta_omega),
        min=-10.0,
        max=10.0,
    )

    alpha_hit = deltas.get("alpha_hit", None)
    if alpha_hit is None:
        alpha_hit = torch.zeros_like(s)
    else:
        alpha_hit = torch.as_tensor(alpha_hit, dtype=torch.float32, device=s.device)

    if h_t is None:
        h_t = torch.zeros_like(s)
    else:
        h_t = torch.as_tensor(h_t, dtype=torch.float32, device=s.device)

    alpha_new = torch.clamp(alpha + h_t * alpha_hit, min=0.0, max=5.0)

    return {"s": s_new, "alpha": alpha_new, "omega": omega_new}


def apply_policy_deltas(
    s: float,
    alpha: float,
    omega: float,
    theta: float,
    deltas: Dict[str, np.ndarray],
    h_t: float = 0.0,
    k_residuals: int = rc.DEFAULT_K_RESIDUALS,
    residual_omega_scale: float = rc.DEFAULT_RESIDUAL_OMEGA_SCALE,
    lobe: int = 1,
    sub_lobe: int = 0,
    seed: Optional[int] = None,
    max_delta_s: float = 0.5,
    max_delta_omega: float = 1.0,
) -> tuple[rc.OrbitState, Dict[str, float]]:
    """Apply decoded deltas to given orbit state primitive values and return a new OrbitState.

    This function accepts primitive state values (s, alpha, omega, theta) and returns
    a new `rc.OrbitState` with deltas applied. This avoids relying on internal attribute
    accessors on the `OrbitState` wrapper.
    """
    # Defensive numeric conversion
    s = float(s)
    alpha = float(alpha)
    omega = float(omega)
    theta = float(theta)

    # Apply s/omega deltas
    ds = float(deltas.get("delta_s", 0.0))
    domega = float(deltas.get("delta_omega", 0.0))
    s_new = float(max(0.1, min(10.0, s + max(-max_delta_s, min(max_delta_s, ds)))))
    omega_new = float(
        max(
            -10.0,
            min(10.0, omega + max(-max_delta_omega, min(max_delta_omega, domega))),
        )
    )

    # alpha: treated as additive impulse scaled by h_t
    alpha_hit = float(deltas.get("alpha_hit", 0.0))
    alpha_new = float(max(0.0, min(5.0, alpha + h_t * alpha_hit)))

    new_state = rc.OrbitState(
        lobe=lobe,
        sub_lobe=sub_lobe,
        theta=theta,
        omega=omega_new,
        s=s_new,
        alpha=alpha_new,
        k_residuals=k_residuals,
        residual_omega_scale=residual_omega_scale,
        seed=seed,
    )

    return new_state, {"s": s_new, "alpha": alpha_new, "omega": omega_new}
