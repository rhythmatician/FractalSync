"""Preflight parity checks: guard against training-mirror / runtime divergence.

Runs BEFORE training and hard-fails (non-zero exit) if any mandatory parity
check fails, naming the mismatching component and the max abs error:

  a) Carrier parity   - runtime_core OrbitState.synthesize vs closed form
                        c = mu/2 - mu^2/4 with mu = s*e^{i theta}.
  b) Mirror parity    - backend/src/cspace_proxies.synthesize_c vs runtime_core
                        synthesize WITH residuals (shared phases passed explicitly).
  c) Shared phase source - residual_phases_for_seed_py == OrbitState.residual_phases().

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
# Manifold mirror tolerance: the mirror runs its state and target/force math
# in float64 (matching the float64 Rust kernel) and calls the same Rust
# integrate_step binding, so forward parity is essentially exact (~1e-8 over
# a 60-step chaotic trajectory). The tolerance is set well above the observed
# error to absorb residual float64 libm rounding while still catching real
# divergence (sign flips, wrong constants are O(1) errors).
MANIFOLD_TOL = 1e-4
# Feature-window tolerance: librosa's FFT vs rustfft differ slightly in
# floating-point rounding; 5e-3 relative to [0,1]-scaled features is tight
# enough to catch semantic drift while tolerating library rounding.
FEATURE_TOL = 5e-3


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


def _canonical_dt() -> float:
    """The canonical physics timestep, derived from the deployed contract.

    PARITY RULE: parity tests must derive constants from the deployed
    contract, never restate them. The browser supplies
    AnalysisTick.dtSeconds = HOP_LENGTH / SAMPLE_RATE from the Rust
    timebase; advancing parity paths at any other value (e.g. 1/60) proves
    agreement at a timestep production never uses (#93 incident).
    """
    rc = _import_runtime_core()
    return rc.HOP_LENGTH / rc.SAMPLE_RATE


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
    s_vals = 0.5 + 1.5 * torch.rand(n, generator=rng)  # [0.5, 2.0]
    alpha_vals = 0.2 + 0.8 * torch.rand(n, generator=rng)  # [0.2, 1.0]
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
            1,
            0,
            theta_val,
            1.0,
            s_val,
            alpha_val,
            K_RESIDUALS,
            2.0,
            DEFAULT_ORBIT_SEED,
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
    """(e) OrbitController mirror vs Rust OrbitController trajectories.

    THE critical check: the trainer must supervise through the same controller
    semantics the browser executes. Verifies BOTH runtime paths:
      e1) flags-off (May baseline) — orbit_controller_sequence
      e2) momentum ON — orbit_controller_momentum_sequence
    A divergence in either means training optimizes physics the runtime does
    not run. (e2 exists because momentum was enabled in the browser without
    retraining/mirror support — the exact gap this check now closes.)
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for orbit mirror parity check (e). Install "
            "backend requirements: pip install -r backend/requirements.txt"
        ) from exc

    from src.cspace_proxies import (
        canonical_hop_dt,
        orbit_controller_sequence,
        orbit_controller_momentum_sequence,
    )

    # PARITY RULE: the timestep must come from the deployed contract
    # (HOP_LENGTH / SAMPLE_RATE via the installed runtime_core), never be
    # restated as a literal. The browser supplies AnalysisTick.dtSeconds
    # from the Rust timebase; testing 1/60 here would prove Rust and the
    # mirror agree at an obsolete timestep while the real paths diverge
    # (the #93 incident).
    dt = canonical_hop_dt()

    rng = torch.Generator().manual_seed(0)
    n_steps = 60
    max_err = 0.0
    for trial in range(4):
        s_vals = (1.0 + 0.4 * torch.randn(n_steps, generator=rng)).clamp(0.2, 3.0)
        a_vals = torch.rand(n_steps, generator=rng).clamp(0.0, 1.0)
        gates = torch.rand(n_steps, K_RESIDUALS, generator=rng)
        seg = torch.zeros(n_steps, dtype=torch.int64)

        # ---- e1: flags-off path ----
        c = rc.OrbitController(float(s_vals[0]), float(a_vals[0]), 1.0)
        rust_re = rust_im = 0.0
        for i in range(n_steps):
            c.apply_controls(float(s_vals[i]), float(a_vals[i]))
            rust_re, rust_im = c.step(dt, [float(g) for g in gates[i]])

        pt_c = orbit_controller_sequence(
            s_target=s_vals,
            alpha=a_vals,
            omega=1.0,
            band_gates=gates,
            segment_ids=seg,
            dt=dt,
        )
        err = max(
            abs(rust_re - pt_c[-1].real.item()),
            abs(rust_im - pt_c[-1].imag.item()),
        )
        max_err = max(max_err, err)

        # Exercise the same momentum path and audio inputs used in training.
        cm = rc.OrbitController(float(s_vals[0]), float(a_vals[0]), 1.0)
        cm.set_momentum(True)
        cm.set_drag(0.90)
        initial = complex(-0.4, 0.2)
        cm.set_c(initial.real, initial.imag)
        thrust = torch.linspace(0.001, 0.02, n_steps)
        energy = torch.linspace(0.1, 0.9, n_steps)
        rust_trajectory = []
        for i in range(n_steps):
            cm.apply_controls(float(s_vals[i]), float(a_vals[i]))
            cm.set_thrust(float(thrust[i]))
            cm.set_energy(float(energy[i]))
            re, im = cm.step(dt, [float(g) for g in gates[i]])
            rust_trajectory.append(complex(re, im))

        pt_cm = orbit_controller_momentum_sequence(
            s_target=s_vals, alpha=a_vals, omega=1.0, band_gates=gates,
            segment_ids=seg, dt=dt, drag=0.90, thrust=thrust,
            initial_c=torch.full((n_steps,), initial, dtype=torch.complex64),
            energy=energy,
        )
        for actual, expected in zip(pt_cm, rust_trajectory):
            max_err = max(max_err, abs(complex(actual.item()) - expected))

    return max_err <= MIRROR_TOL, max_err


def check_manifold_mirror_parity(rc) -> tuple[bool, float]:
    """(e5) Manifold-physics mirror vs Rust OrbitController (issue #106).

    The trainer's ``orbit_controller_manifold_sequence`` must reproduce the
    Rust ``OrbitController`` with ``manifold_physics`` enabled: same target
    synthesis, same generalized-force construction, same integrator. A
    divergence means training optimizes physics the browser does not run.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for manifold mirror parity check (e5). Install "
            "backend requirements: pip install -r backend/requirements.txt"
        ) from exc

    if not hasattr(rc, "manifold_integrate_step"):
        raise RuntimeError(
            "runtime_core missing manifold bindings; rebuild the wheel "
            "(maturin develop --release in runtime-core/)."
        )

    from src.cspace_proxies import (
        ManifoldConfig,
        canonical_hop_dt,
        orbit_controller_manifold_sequence,
    )

    # PARITY RULE: contract-derived timestep, never a literal (#93).
    dt = canonical_hop_dt()

    n_steps = 60
    max_err = 0.0
    for trial in range(3):
        rng = torch.Generator().manual_seed(trial)
        s_vals = (1.0 + 0.4 * torch.randn(n_steps, generator=rng)).clamp(0.2, 3.0)
        a_vals = torch.rand(n_steps, generator=rng).clamp(0.0, 1.0)
        gates = torch.rand(n_steps, K_RESIDUALS, generator=rng)
        seg = torch.zeros(n_steps, dtype=torch.int64)
        energy = torch.linspace(0.2, 0.8, n_steps)

        # Rust controller with manifold physics on.
        ctrl = rc.OrbitController(float(s_vals[0]), float(a_vals[0]), 1.0)
        ctrl.set_manifold_physics(True)
        ctrl.set_manifold_drag(0.1)
        ctrl.set_manifold_config(rc.ManifoldConfig(0.1, 1e-4, 1.0, 1.0))
        rust_traj: list[tuple[float, float]] = []
        for i in range(n_steps):
            ctrl.apply_controls(float(s_vals[i]), float(a_vals[i]))
            ctrl.set_energy(float(energy[i]))
            rre, rim = ctrl.step(dt, [float(g) for g in gates[i]])
            rust_traj.append((rre, rim))

        # Python mirror of the same path.
        traj, _infos = orbit_controller_manifold_sequence(
            s_target=s_vals,
            alpha=a_vals,
            omega=1.0,
            band_gates=gates,
            segment_ids=seg,
            dt=dt,
            energy=energy,
            manifold_drag=0.1,
            config=ManifoldConfig(),
        )
        for i in range(n_steps):
            err = max(
                abs(traj[i].real.item() - rust_traj[i][0]),
                abs(traj[i].imag.item() - rust_traj[i][1]),
            )
            max_err = max(max_err, err)
    return max_err <= MANIFOLD_TOL, max_err


def check_shared_phase_source(rc) -> tuple[bool, float]:
    """(c) residual_phases_for_seed_py == OrbitState.residual_phases()."""
    max_err = 0.0
    ok = True
    for seed in (1337, 42, 7):
        for k in (3, 6):
            shared = list(rc.residual_phases_for_seed_py(seed, k))
            state = rc.OrbitState.new_with_seed(1, 0, 0.0, 1.0, 1.02, 0.3, k, 2.0, seed)
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


def _load_golden() -> dict:
    import json

    golden_path = REPO_ROOT / "shared" / "golden_vectors.json"
    if not golden_path.exists():
        raise RuntimeError(
            f"Golden vectors missing at {golden_path}. Regenerate: "
            "cargo run --release -p runtime_core --bin generate_golden_vectors"
        )
    with open(golden_path) as f:
        return json.load(f)


def check_golden_version(rc) -> tuple[bool, float]:
    """(f) Golden vectors were generated by the current controller version.

    Stale goldens would silently verify the wrong contract. The generator
    stamps shared/golden_vectors.json with CONTROLLER_VERSION; this check
    requires it to match the installed runtime_core's version.
    """
    runtime_version = getattr(rc, "CONTROLLER_VERSION", None)
    if not runtime_version:
        raise RuntimeError(
            "runtime_core does not expose CONTROLLER_VERSION; rebuild and "
            "reinstall the wheel (maturin develop --release)."
        )

    golden = _load_golden()

    golden_version = golden.get("controller_version")
    if golden_version != runtime_version:
        raise RuntimeError(
            f"Golden vectors are stale: generated by '{golden_version}' but "
            f"runtime is '{runtime_version}'. Regenerate goldens in the same "
            "commit as the controller change."
        )
    return True, 0.0


def check_feature_golden_parity(rc) -> tuple[bool, float]:
    """(g) Python feature mirror vs Rust golden feature windows.

    The diagnostic NumPy oracle must reproduce the canonical Rust
    extractor's output on the deterministic synthetic audio recorded in
    shared/golden_vectors.json feature_cases. Catches any drift in feature
    definitions, causal transforms, or window layout.
    """
    import numpy as np

    from src.python_feature_extractor import DiagnosticFeatureParityOracle

    golden = _load_golden()
    cases = golden.get("feature_cases") or []
    if not cases:
        raise RuntimeError(
            "Golden vectors contain no feature_cases; regenerate goldens: "
            "cargo run --release -p runtime_core --bin generate_golden_vectors"
        )

    max_err = 0.0
    for case in cases:
        seed = int(case["seed"])
        window_frames = int(case["window_frames"])
        expected = np.array(case["features"], dtype=np.float64)

        # Reconstruct the same deterministic audio as the Rust generator:
        # harmonic stack + seeded LCG noise, 1 s at 48 kHz.
        n_samples = 48_000
        t = np.arange(n_samples, dtype=np.float64) / 48_000.0
        audio = (
            0.3 * np.sin(2 * np.pi * 220.0 * t)
            + 0.2 * np.sin(2 * np.pi * 440.0 * t)
            + 0.1 * np.sin(2 * np.pi * 880.0 * t)
        )
        lcg = np.uint64(seed)
        MUL = np.uint64(6364136223846793005)
        ADD = np.uint64(1442695040888963407)
        noise = np.empty(n_samples, dtype=np.float64)
        for i in range(n_samples):
            lcg = lcg * MUL + ADD
            noise[i] = (
                np.int64(lcg >> np.uint64(33)).astype(np.float64)
                / float(np.int64(1) << np.int64(30))
                - 1.0
            ) * 0.05
        audio = np.clip(audio + noise, -1.0, 1.0).astype(np.float32)

        fe = DiagnosticFeatureParityOracle()
        windows = fe.extract_windowed_features(audio.tolist(), window_frames)  # type: ignore[arg-type]
        if len(windows) == 0:
            raise RuntimeError(f"Diagnostic feature oracle produced no window for seed {seed}")
        got = np.asarray(windows[0], dtype=np.float64)
        if got.shape != expected.shape:
            raise RuntimeError(
                f"Feature window shape mismatch for seed {seed}: python "
                f"{got.shape} vs rust {expected.shape} — layout diverged"
            )
        err = float(np.max(np.abs(got - expected)))
        max_err = max(max_err, err)
    return max_err <= FEATURE_TOL, max_err


def check_feature_version(rc) -> tuple[bool, float]:
    """(h) Golden feature vectors were generated by the current feature
    contract version. Same staleness guard as (f), for features."""
    runtime_version = getattr(rc, "FEATURE_VERSION", None)
    if not runtime_version:
        raise RuntimeError(
            "runtime_core does not expose FEATURE_VERSION; rebuild and "
            "reinstall the wheel (maturin develop --release)."
        )

    golden = _load_golden()
    golden_version = golden.get("feature_version")
    if golden_version != runtime_version:
        raise RuntimeError(
            f"Golden feature vectors are stale: generated by "
            f"'{golden_version}' but runtime is '{runtime_version}'. "
            "Regenerate goldens in the same commit as the feature change."
        )
    return True, 0.0


CHECKS: list[tuple[str, bool, Callable]] = [
    ("a) Carrier parity (Rust vs closed form)", True, check_carrier_parity),
    ("b) Mirror parity (synthesize_c vs Rust)", True, check_mirror_parity),
    ("c) Shared phase source", True, check_shared_phase_source),
    (
        "e) Orbit mirror parity (trainer vs runtime)",
        True,
        check_player_mirror_parity,
    ),
    (
        "e5) Manifold physics parity (mirror vs Rust, issue #106)",
        True,
        check_manifold_mirror_parity,
    ),
    (
        "f) Golden vector version (stale-golden guard)",
        True,
        check_golden_version,
    ),
    (
        "g) Feature golden parity (Python mirror vs Rust)",
        True,
        check_feature_golden_parity,
    ),
    (
        "h) Feature golden version (stale-golden guard)",
        True,
        check_feature_version,
    ),
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
