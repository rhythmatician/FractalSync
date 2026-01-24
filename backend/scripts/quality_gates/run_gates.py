"""Quality gate runner: run the defined gates on a short scenario and exit non-zero on failure.

This script runs a small deterministic scenario (synthetic audio + fixed seed)
and evaluates Continuity, Impact, Variety, and Parity gates producing a JSON
report and returning non-zero if any gate fails.

Run locally with: python -m backend.scripts.quality_gates.run_gates
"""

from __future__ import annotations

import json
import sys
from typing import List

import numpy as np

from src.quality_gates.metrics import (
    proxy_delta_v,
    window_coverage_entropy,
    detect_transients,
)
from src.runtime_core_bridge import (
    make_orbit_state,
    make_residual_params,
    synthesize,
    step_orbit,
)

# Gate thresholds (tunable)
CONTINUITY_MEDIAN_THRESH = 0.02
CONTINUITY_MAX_THRESH = 0.1
IMPACT_LAG_THRESH = 1
IMPACT_CORR_THRESH = 0.6
VARIETY_COVERAGE_THRESH = 0.05
VARIETY_ENTROPY_THRESH = 1.0
PARITY_EPS = 1e-6


def run_synthetic_scenario(duration_s: float = 10.0, dt: float = 0.1):
    """Generate a simple synthetic audio transient sequence and produce c(t) trajectory."""
    steps = int(duration_s / dt)
    spectral_flux = np.zeros(steps, dtype=np.float32)
    for i in range(0, steps, max(1, int(1.0 / dt * 2))):
        spectral_flux[i] = 1.0

    orbit = make_orbit_state(lobe=1, theta=0.0, s=1.02, alpha=0.3)
    rp = make_residual_params()

    cs = []
    proxies = []
    alphas = []
    for t_idx in range(steps):
        c = synthesize(orbit, rp, None)
        cs.append((float(c.real), float(c.imag)))
        r = np.sqrt(c.real**2 + c.imag**2)
        proxies.append(np.full((64, 64), r, dtype=np.float32))
        alphas.append(float(orbit.alpha))
        orbit.theta = orbit.theta + orbit.omega * dt

    return spectral_flux, cs, proxies, alphas


def continuity_gate(proxies: List[np.ndarray]) -> bool:
    dvs = [proxy_delta_v(proxies[i], proxies[i + 1]) for i in range(len(proxies) - 1)]
    med = float(np.median(dvs))
    mx = float(np.max(dvs))
    print(f"Continuity: median={med:.6f}, max={mx:.6f}")
    return med < CONTINUITY_MEDIAN_THRESH and mx < CONTINUITY_MAX_THRESH


def impact_gate(spectral_flux, proxies, alphas) -> bool:
    hits = detect_transients(spectral_flux, thresh=0.5)
    dvs = [proxy_delta_v(proxies[i], proxies[i + 1]) for i in range(len(proxies) - 1)]
    lags = []
    for h in hits:
        window = range(max(0, h), min(len(dvs), h + IMPACT_LAG_THRESH + 1))
        if not window:
            lags.append(999)
            continue
        local_idx = int(np.argmax([dvs[i] for i in window]))
        lag = list(window)[local_idx] - h
        lags.append(lag)
    median_lag = int(np.median(lags)) if lags else 0
    corr = np.corrcoef(spectral_flux, np.pad(dvs, (0, 1))[: len(spectral_flux)])[0, 1]
    print(f"Impact: median_lag={median_lag}, corr={corr:.3f}")
    return median_lag <= IMPACT_LAG_THRESH and (
        np.isfinite(corr) and corr >= IMPACT_CORR_THRESH
    )


def variety_gate(cs) -> bool:
    cov, ent = window_coverage_entropy(cs, bins=(32, 32))
    print(f"Variety: coverage={cov:.3f}, entropy={ent:.3f}")
    return cov >= VARIETY_COVERAGE_THRESH and ent >= VARIETY_ENTROPY_THRESH


def parity_gate(cs) -> bool:
    # Minimal parity sanity check: ensure runtime_core synthesize and step are callable
    try:
        s1 = make_orbit_state(lobe=1, theta=0.0, s=1.02, alpha=0.3)
        rp = make_residual_params()
        c0 = synthesize(s1, rp, None)
        # advance deterministically
        step_orbit(s1, 0.1, rp, None, h=0.0, d_star=0.3, max_step=0.03)
        c1 = synthesize(s1, rp, None)
        return all(
            [
                float(np.isfinite(c0.real)),
                float(np.isfinite(c0.imag)),
                float(np.isfinite(c1.real)),
                float(np.isfinite(c1.imag)),
            ]
        )
    except Exception:
        return False


def main():
    spectral_flux, cs, proxies, alphas = run_synthetic_scenario(duration_s=10.0, dt=0.1)

    results = {
        "continuity": continuity_gate(proxies),
        "impact": impact_gate(spectral_flux, proxies, alphas),
        "variety": variety_gate(cs),
        "parity": parity_gate(cs),
    }

    print(json.dumps(results, indent=2))
    if not all(results.values()):
        print("Quality gates failed", file=sys.stderr)
        sys.exit(2)

    print("All quality gates passed")


if __name__ == "__main__":
    main()
