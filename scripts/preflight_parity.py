"""Preflight parity checks: guard against training-mirror / runtime divergence.

Runs BEFORE training and hard-fails (non-zero exit) if any mandatory parity
check fails, naming the mismatching component and the max abs error:

  a) Carrier parity   - runtime_core OrbitState.synthesize vs closed form
                        c = mu/2 - mu^2/4 with mu = s*e^{i theta}.
  b) Mirror parity    - backend/src/cspace_proxies.synthesize_c vs runtime_core
                        synthesize WITH residuals (shared phases passed explicitly).
  c) Shared phase source - residual_phases_for_seed_py == OrbitState.residual_phases().

Non-fatal warning-only check:
  d) Minimap availability - baked mip pyramid artifacts + load_mip_pyramid_py binding.

Exit code 0 only if a, b, c all pass.

Usage:
    python scripts/preflight_parity.py

This module is importable so pytest can exercise the same checks
(see backend/tests/test_preflight_parity.py); all logic lives in functions
and the CLI is guarded by ``if __name__ == "__main__":``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parent.parent

# Make `src.cspace_proxies` importable (it lives under backend/src) and allow
# running this script from any CWD.
_BACKEND = REPO_ROOT / "backend"
for _p in (str(REPO_ROOT), str(_BACKEND)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEFAULT_ORBIT_SEED = 1337
K_RESIDUALS = 6
RESIDUAL_CAP = 0.5
CARRIER_TOL = 1e-9
MIRROR_TOL = 1e-5


def _import_runtime_core():
    """Import the installed runtime_core extension or raise a clear error."""
    try:
        import runtime_core  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "runtime_core is not importable. The Rust extension must be built "
            "and installed first, e.g.:\n"
            "  Push-Location runtime-core; try { maturin develop --release } "
            "finally { Pop-Location }\n"
            f"Original import error: {exc}"
        ) from exc
    return runtime_core


def _carrier_reference(theta: float, s: float) -> complex:
    """Independent closed form: c = mu/2 - mu^2/4 with mu = s * e^{i theta}."""
    import cmath

    mu = s * cmath.exp(1j * theta)
    return mu / 2 - mu**2 / 4


def check_carrier_parity(rc) -> tuple[bool, float]:
    """(a) Rust carrier synthesis vs closed form over an 8x8 grid."""
    rp = rc.ResidualParams(
        k_residuals=K_RESIDUALS, residual_cap=RESIDUAL_CAP, radius_scale=1.0
    )
    max_err = 0.0
    n_theta, n_s = 8, 8
    for i in range(n_theta):
        theta = 2.0 * math.pi * i / n_theta
        for j in range(n_s):
            s = 0.5 + (2.0 - 0.5) * j / (n_s - 1)
            state = rc.OrbitState.new_with_seed(
                1, 0, theta, 1.0, s, 0.0, K_RESIDUALS, 2.0, DEFAULT_ORBIT_SEED
            )
            c = state.synthesize(rp, None)
            expected = _carrier_reference(theta, s)
            err = max(abs(c.real - expected.real), abs(c.imag - expected.imag))
            max_err = max(max_err, err)
    return max_err <= CARRIER_TOL, max_err


def _import_synthesize_c():
    """Import the PyTorch mirror, ensuring backend/ is on sys.path.

    Kept as a function so static analyzers see an explicit import site and
    so the path setup happens exactly where the dependency is needed.
    """
    if str(_BACKEND) not in sys.path:
        sys.path.insert(0, str(_BACKEND))
    from src.cspace_proxies import synthesize_c  # noqa: E402

    return synthesize_c


def check_mirror_parity(rc) -> tuple[bool, float]:
    """(b) PyTorch mirror synthesize_c vs Rust synthesize with residuals."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for mirror parity check (b). Install backend "
            "requirements: pip install -r backend/requirements.txt"
        ) from exc

    synthesize_c = _import_synthesize_c()

    rng = torch.Generator().manual_seed(0)
    n = 64
    s_vals = 0.5 + 1.5 * torch.rand(n, generator=rng)          # [0.5, 2.0]
    alpha_vals = 0.2 + 0.8 * torch.rand(n, generator=rng)      # [0.2, 1.0]
    theta_vals = 2.0 * math.pi * torch.rand(n, generator=rng)

    rp = rc.ResidualParams(
        k_residuals=K_RESIDUALS, residual_cap=RESIDUAL_CAP, radius_scale=1.0
    )
    phases = list(rc.residual_phases_for_seed_py(DEFAULT_ORBIT_SEED, K_RESIDUALS))
    gates = [1.0] * K_RESIDUALS

    max_err = 0.0
    for idx in range(n):
        s_val = float(s_vals[idx])
        alpha_val = float(alpha_vals[idx])
        theta_val = float(theta_vals[idx])

        state = rc.OrbitState.new_with_seed(
            1, 0, theta_val, 1.0, s_val, alpha_val,
            K_RESIDUALS, 2.0, DEFAULT_ORBIT_SEED,
        )
        rust_c = state.synthesize(rp, band_gates=gates)

        pt_c = synthesize_c(
            s_target=torch.tensor([s_val]),
            alpha=torch.tensor([alpha_val]),
            band_gates=torch.ones(1, K_RESIDUALS),
            thetas=torch.tensor([theta_val]),
            k_residuals=K_RESIDUALS,
            residual_cap=RESIDUAL_CAP,
            phases=phases,
        )
        err = max(
            abs(rust_c.real - pt_c[0].real.item()),
            abs(rust_c.imag - pt_c[0].imag.item()),
        )
        max_err = max(max_err, err)
    return max_err <= MIRROR_TOL, max_err


def check_player_mirror_parity(rc) -> tuple[bool, float]:
    """(e) PlayerState mirror vs Rust PlayerState trajectories.

    THE critical check: the trainer must supervise through the same momentum
    integrator the browser executes. A divergence here means training optimizes
    physics the runtime does not run — the exact failure that wasted a 90-minute
    session and produced saturated, frozen-c controls.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for player mirror parity check (e). Install "
            "backend requirements: pip install -r backend/requirements.txt"
        ) from exc

    from src.cspace_proxies import player_step_sequence

    rng = torch.Generator().manual_seed(0)
    n_steps = 60
    max_err = 0.0
    for trial in range(4):
        s0 = float(0.5 + 2.2 * torch.rand(1, generator=rng))
        alpha0 = float(torch.rand(1, generator=rng))
        s_vals = (s0 + 0.05 * torch.randn(n_steps, generator=rng)).clamp(0.2, 3.0)
        a_vals = (alpha0 + 0.01 * torch.randn(n_steps, generator=rng)).clamp(0.0, 1.0)
        w_vals = (1.0 + 3.0 * torch.rand(n_steps, generator=rng)).clamp(0.1, 10.0)
        gates = torch.rand(n_steps, K_RESIDUALS, generator=rng)
        seg = torch.zeros(n_steps, dtype=torch.int64)

        # Rust reference: replay through the actual binding.
        p = rc.PlayerState(1, 0, s0, alpha0)
        rust_re = 0.0
        rust_im = 0.0
        for i in range(n_steps):
            p.apply_controls(float(s_vals[i]), float(a_vals[i]), float(w_vals[i]))
            rust_re, rust_im = p.step(
                1.0 / 60.0, 0.0, [float(g) for g in gates[i]]
            )

        # Mirror: same controls through the differentiable sequence. The
        # mirror's c0 must match the Rust constructor state (boundary at
        # s0/alpha0), which differs from the first frame's controls.
        theta0 = alpha0 * 2.0 * math.pi
        mu0 = complex(s0 * math.cos(theta0), s0 * math.sin(theta0))
        c0 = mu0 / 2 - mu0**2 / 4
        pt_c = player_step_sequence(
            s_target=s_vals,
            alpha=a_vals,
            omega_scale=w_vals,
            band_gates=gates,
            segment_ids=seg,
            c0=(c0.real, c0.imag),
        )
        err = max(
            abs(rust_re - pt_c[-1].real.item()),
            abs(rust_im - pt_c[-1].imag.item()),
        )
        max_err = max(max_err, err)
    return max_err <= MIRROR_TOL, max_err


def check_shared_phase_source(rc) -> tuple[bool, float]:
    """(c) residual_phases_for_seed_py == OrbitState.residual_phases()."""
    max_err = 0.0
    ok = True
    for seed in (1337, 42, 7):
        for k in (3, 6):
            shared = list(rc.residual_phases_for_seed_py(seed, k))
            state = rc.OrbitState.new_with_seed(
                1, 0, 0.0, 1.0, 1.02, 0.3, k, 2.0, seed
            )
            from_state = list(state.residual_phases())
            if len(shared) != len(from_state):
                ok = False
                max_err = float("inf")
                continue
            for a, b in zip(shared, from_state):
                err = abs(a - b)
                max_err = max(max_err, err)
                if err > 1e-12:
                    ok = False
    return ok, max_err


def check_minimap_availability(rc) -> tuple[bool, float]:
    """(d) Warning-only: report minimap artifact / binding availability.

    Always returns (True, 0.0); failures here are advisory, not fatal.
    """
    names = ("mandel_F_mips_f32.bin", "mandel_S_mips_f32.bin", "mandel_mips_meta.json")
    found: dict[str, Path | None] = {}
    for name in names:
        hit = next(
            (
                candidate
                for base in (REPO_ROOT, _BACKEND)
                if (candidate := base / name).exists()
            ),
            None,
        )
        found[name] = hit
    has_binding = hasattr(rc, "load_mip_pyramid_py")

    missing = [n for n, p in found.items() if p is None]
    if missing:
        print(f"WARNING [minimap]: missing artifacts: {', '.join(missing)}")
    else:
        print("Minimap artifacts present:")
        for name, p in found.items():
            print(f"  {name}: {p}")
    if not has_binding:
        print(
            "WARNING [minimap]: runtime_core does not expose "
            "load_mip_pyramid_py"
        )
    else:
        print("Minimap binding available: runtime_core.load_mip_pyramid_py")
    return True, 0.0


CHECKS: list[tuple[str, bool, Callable]] = [
    ("a) Carrier parity (Rust vs closed form)", True, check_carrier_parity),
    ("b) Mirror parity (synthesize_c vs Rust)", True, check_mirror_parity),
    ("c) Shared phase source", True, check_shared_phase_source),
    (
        "e) Player mirror parity (trainer vs runtime)",
        True,
        check_player_mirror_parity,
    ),
    ("d) Minimap availability (warning only)", False, check_minimap_availability),
]


def run_preflight(verbose: bool = True) -> tuple[bool, list[tuple[str, str, str]]]:
    """Run all checks. Returns (all_mandatory_passed, [(name, status, detail)]).

    Raises RuntimeError with a clear message when a prerequisite is missing
    (runtime_core not importable, torch not installed).
    """
    rc = _import_runtime_core()

    results: list[tuple[str, str, str]] = []
    all_ok = True
    for name, mandatory, fn in CHECKS:
        try:
            ok, max_err = fn(rc)
        except RuntimeError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface any check failure loudly
            results.append((name, "FAIL", f"exception: {exc!r}"))
            if mandatory:
                all_ok = False
            continue
        status = "PASS" if ok else "FAIL"
        detail = f"max abs err = {max_err:.3e}" if math.isfinite(max_err) else "n/a"
        results.append((name, status, detail))
        if mandatory and not ok:
            all_ok = False

    if verbose:
        print("=" * 64)
        print("Preflight parity checks")
        print("=" * 64)
        for name, status, detail in results:
            print(f"{status:>4}  {name:<45} {detail}")
        print("=" * 64)

    if not all_ok:
        failing = [r for r in results if r[1] == "FAIL"]
        lines = ["PREFLIGHT PARITY FAILURE — training must not start:"]
        for name, _, detail in failing:
            lines.append(f"  - {name}: {detail}")
        raise RuntimeError("\n".join(lines))

    return True, results


def main() -> int:
    try:
        run_preflight()
    except RuntimeError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    print("All mandatory parity checks PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
