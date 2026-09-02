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

        # ---- e2: momentum path (browser runs this when setMomentum(true)) ----
        cm = rc.OrbitController(float(s_vals[0]), float(a_vals[0]), 1.0)
        cm.set_momentum(True)
        cm.set_drag(0.90)
        rust_mre = rust_mim = 0.0
        for i in range(n_steps):
            cm.apply_controls(float(s_vals[i]), float(a_vals[i]))
            rust_mre, rust_mim = cm.step(dt, [float(g) for g in gates[i]])

        pt_cm = orbit_controller_momentum_sequence(
            s_target=s_vals,
            alpha=a_vals,
            omega=1.0,
            band_gates=gates,
            segment_ids=seg,
            dt=dt,
            drag=0.90,
        )
        err_m = max(
            abs(rust_mre - pt_cm[-1].real.item()),
            abs(rust_mim - pt_cm[-1].imag.item()),
        )
        max_err = max(max_err, err_m)
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
        print("WARNING [minimap]: runtime_core does not expose load_mip_pyramid_py")
    else:
        print("Minimap binding available: runtime_core.load_mip_pyramid_py")
    return True, 0.0


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

    The Python fallback extractor must reproduce the canonical Rust
    extractor's output on the deterministic synthetic audio recorded in
    shared/golden_vectors.json feature_cases. Catches any drift in feature
    definitions, causal transforms, or window layout.
    """
    import numpy as np

    from src.python_feature_extractor import PythonFeatureExtractor

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

        fe = PythonFeatureExtractor()
        windows = fe.extract_windowed_features(audio.tolist(), window_frames)  # type: ignore[arg-type]
        if len(windows) == 0:
            raise RuntimeError(f"Python extractor produced no window for seed {seed}")
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


# ---------------------------------------------------------------------------
# Shore-biased dynamics parity (e3): explicit gate around the
# `contour_biased_step` + `OrbitController(shore_bias=true, momentum=true)`
# path the BROWSER runs after orbit-controller/3. The previous preflight
# `e2` check verified the flags-off and momentum-only paths; it did NOT
# exercise the shore-bias step, so the Python trainer's "energy push" could
# (and did) drift from the runtime while the parity suite stayed green.
#
# The check installs a small in-memory mip pyramid via
# `install_pyramid_py` (no filesystem ceremony), drives the Rust
# `OrbitController` for N frames, and compares the resulting c to a Python
# oracle that calls `runtime_core.contour_biased_step_py` (the same Rust
# function the browser runs). If the trainer and runtime agree, the
# oracle and the Rust `OrbitController` must agree to within float
# rounding.
# ---------------------------------------------------------------------------

# Per-frame tolerance for c-re/im. Rust and Python both go through the
# same `contour_biased_step_py` call here, so any disagreement is one
# frame of float-rounding compounding.
SHORE_TOL = 1e-9
# Trainer-forward trajectory tolerance: 60 frames of float rounding
# (Rust vs Python going through the same contour step) compounds to
# roughly 1e-7 to 1e-6. A bigger gap means a sign / constant error,
# not just IEEE-754 noise.
TRAINER_TOL = 1e-6


def _build_flat_pyramid() -> int:
    """Install a uniform S field (all zeros): forces the analytic cardioid
    fallback everywhere. `install_pyramid_py` returns the level count.
    """
    levels_data: list[list[float]] = []
    widths: list[int] = []
    heights: list[int] = []
    for size in (64, 32, 16, 8, 4, 2, 1):
        levels_data.append([0.0] * (size * size))
        widths.append(size)
        heights.append(size)
    return _import_runtime_core().install_pyramid_py(
        levels_data, widths, heights, -2.0, 1.0, -1.5, 1.5
    )


def _build_linear_pyramid() -> int:
    """S = col / (size-1): S is flat in Im (contours run horizontally) and
    increases with Re. Matches the Rust `linear_pyramid` test fixture so a
    passing Rust test and a passing Python parity check exercise the same
    field topology.
    """
    levels_data: list[list[float]] = []
    widths: list[int] = []
    heights: list[int] = []
    for size in (64, 32, 16, 8, 4, 2, 1):
        plane: list[float] = []
        denom = max(1, size - 1)
        for i in range(size * size):
            col = i % size
            plane.append(col / denom)
        levels_data.append(plane)
        widths.append(size)
        heights.append(size)
    return _import_runtime_core().install_pyramid_py(
        levels_data, widths, heights, -2.0, 1.0, -1.5, 1.5
    )


def _rust_orbit_run(
    rc,
    pyramid_kind: str,
    n_frames: int,
    *,
    energy_seq: list[float],
    h_seq: list[float],
    max_step: float = 0.05,
) -> tuple[list[float], list[float]]:
    """Drive the Rust OrbitController (momentum + shore_bias) for n_frames
    under the given per-frame (energy, h) schedule. Returns the (re, im)
    trajectory of c. The starting c is set to the cardioid boundary at
    (s, alpha)=(0.5, 0.5) so c is on the shore from frame 0.
    """
    ctrl = rc.OrbitController(0.5, 0.5, 1.0)
    ctrl.set_momentum(True)
    ctrl.set_shore_bias(True)
    ctrl.set_drag(0.90)
    ctrl.set_max_step(max_step)
    ctrl.set_d_star(0.5)
    ctrl.set_level(0)
    # Reset c to the starting boundary point so the first step has a
    # well-defined c.
    start = _carrier_reference(
        math.pi * 0.5, 0.5
    )  # α=0.5 -> θ=π, mid-bottom of cardioid
    ctrl.set_c(start.real, start.imag)

    traj_re: list[float] = []
    traj_im: list[float] = []
    for i in range(n_frames):
        # Keep controls boring but move them so the pull has somewhere to
        # go — this is what catches the case where the proposed-delta path
        # in Rust differs from the trainer's reconstruction.
        s = 0.5 + 0.1 * math.cos(i * 0.1)
        alpha = 0.5 + 0.1 * math.sin(i * 0.1)
        ctrl.apply_controls(s, alpha)
        ctrl.set_energy(energy_seq[i] if i < len(energy_seq) else 0.0)
        h_val = h_seq[i] if i < len(h_seq) else 0.0
        re, im = ctrl.step(_canonical_dt(), [0.5, 0.5, 0.5, 0.5, 0.5, 0.5], h_val)
        traj_re.append(re)
        traj_im.append(im)
    return traj_re, traj_im


def _oracle_run(
    rc,
    pyramid_kind: str,
    n_frames: int,
    *,
    energy_seq: list[float],
    h_seq: list[float],
    max_step: float = 0.05,
) -> tuple[list[float], list[float]]:
    """Replay the SAME sequence against a Python oracle that mirrors the
    Rust `OrbitController::step` math but routes the per-frame dynamics
    through `runtime_core.contour_biased_step_py` (the same Rust function
    the browser executes). If trainer and runtime agree, this oracle and
    the Rust `_rust_orbit_run` must agree to within float rounding.
    """
    # Same construction as _rust_orbit_run: the oracle re-implements the
    # momentum integrator in Python but DELIBERATELY defers the contour
    # step to the Rust binding so the two trajectories are physically
    # identical even when the trainer's `orbit_controller_momentum_sequence`
    # uses a different (and possibly diverged) surrogate.
    omega = 1.0
    drag = 0.90
    # Initialize c at the same starting point.
    start = _carrier_reference(math.pi * 0.5, 0.5)
    c_re = start.real
    c_im = start.imag
    v_re = 0.0
    v_im = 0.0
    theta = 0.0  # mimic the wobble phase the Rust controller advances
    # Contract-derived timestep — never restate a literal here.
    dt = _canonical_dt()
    k_residuals = 6
    orr = 0.05
    two_pi = 2.0 * math.pi

    traj_re: list[float] = []
    traj_im: list[float] = []
    for i in range(n_frames):
        s = 0.5 + 0.1 * math.cos(i * 0.1)
        alpha = 0.5 + 0.1 * math.sin(i * 0.1)
        theta = (theta + omega * dt) % two_pi
        # Carrier + residuals — exact mirror of the May controller.
        theta_b = alpha * two_pi
        r = 0.25 * (1.0 - math.cos(theta_b))
        scale = min(s, 1.5)
        base_re = r * math.cos(theta_b / 2.0) * scale
        base_im = r * math.sin(theta_b / 2.0) * scale
        res_re = 0.0
        res_im = 0.0
        for k in range(k_residuals):
            freq = k + 2
            phase = freq * theta
            res_re += 0.5 * orr * math.cos(phase)
            res_im += 0.5 * orr * math.sin(phase)
        target_re = base_re + res_re
        target_im = base_im + res_im

        # Pull as acceleration; gravity (orbit-controller/3).
        accel_gain = 2.0 * dt
        a_re = (target_re - c_re) * accel_gain - 0.01 * c_re
        a_im = (target_im - c_im) * accel_gain - 0.01 * c_im
        v_re = v_re * drag + a_re
        v_im = v_im * drag + a_im
        proposed_re = v_re * dt
        proposed_im = v_im * dt

        energy = energy_seq[i] if i < len(energy_seq) else 0.0
        h_val = h_seq[i] if i < len(h_seq) else 0.0
        # DEFER to the Rust binding — this is the whole point of the oracle.
        c_re, c_im = rc.contour_biased_step_py(
            c_re,
            c_im,
            proposed_re,
            proposed_im,
            h_val,
            0.5,
            max_step,
            0,
            energy,
        )
        traj_re.append(c_re)
        traj_im.append(c_im)
    return traj_re, traj_im


def check_shore_biased_dynamics_parity(rc) -> tuple[bool, float]:
    """(e3) Rust OrbitController (momentum+shore_bias) vs Python oracle.

    The oracle mirrors the May-controller math and the momentum integrator
    exactly, but DELEGATES the per-frame contour step to
    `runtime_core.contour_biased_step_py` (the Rust function the browser
    actually calls). This means the oracle and the runtime run the SAME
    physics; the only thing that can differ is the per-frame float
    rounding from running the integrator in Rust vs Python.

    If the trainer's `orbit_controller_momentum_sequence` ever disagrees
    with this oracle, the e3 check will (intentionally) still pass — the
    check is between RUST and ORACLE-RUST, not between TRAINER and
    RUNTIME. The dedicated trainer parity check (e) is what catches
    trainer divergence from Rust. Together they pin the entire chain.
    """
    if not hasattr(rc, "contour_biased_step_py"):
        raise RuntimeError(
            "runtime_core does not expose contour_biased_step_py; rebuild "
            "and reinstall the wheel (maturin develop --release)."
        )
    if not hasattr(rc, "install_pyramid_py"):
        raise RuntimeError(
            "runtime_core does not expose install_pyramid_py; rebuild and "
            "reinstall the wheel (maturin develop --release)."
        )

    n_frames = 60
    cases = [
        # (name, pyramid_kind, energy_seq, h_seq, max_step)
        # 1. Flat-S triggers analytic cardioid fallback; varying energy.
        (
            "flat-S analytic fallback (energy ramp)",
            "flat",
            [min(1.0, i / (n_frames - 1)) for i in range(n_frames)],
            [0.0] * n_frames,
            0.05,
        ),
        # 2. Linear S gradient (contours along Im); constant moderate energy.
        (
            "linear-S gradient (constant energy)",
            "linear",
            [0.5] * n_frames,
            [0.0] * n_frames,
            0.05,
        ),
        # 3. Energy 0 vs 1 contrast on flat-S — strongest sign-of-push signal.
        (
            "flat-S energy contrast (0 vs 1)",
            "flat",
            [0.0] * (n_frames // 2) + [1.0] * (n_frames // 2),
            [0.0] * n_frames,
            0.05,
        ),
        # 4. h=0 vs h=1 contrast on linear-S — wall vs open.
        (
            "linear-S h contrast (0 wall, 1 open)",
            "linear",
            [0.0] * n_frames,
            [0.0] * (n_frames // 2) + [1.0] * (n_frames // 2),
            0.05,
        ),
        # 5. max_step clamp: huge proposed motion must clamp.
        (
            "max_step clamp (huge proposed motion)",
            "flat",
            [0.0] * n_frames,
            [1.0] * n_frames,  # open wall so we exercise the clamp, not the wall
            0.005,  # very small max_step to force the clamp
        ),
    ]

    max_err = 0.0
    for name, pyramid_kind, energy_seq, h_seq, max_step in cases:
        try:
            if pyramid_kind == "flat":
                _build_flat_pyramid()
            else:
                _build_linear_pyramid()
            rust_re, rust_im = _rust_orbit_run(
                rc,
                pyramid_kind,
                n_frames,
                energy_seq=energy_seq,
                h_seq=h_seq,
                max_step=max_step,
            )
            oracle_re, oracle_im = _oracle_run(
                rc,
                pyramid_kind,
                n_frames,
                energy_seq=energy_seq,
                h_seq=h_seq,
                max_step=max_step,
            )
        finally:
            rc.clear_pyramid_py()

        for i, (rr, ri, orr_r, orr_i) in enumerate(
            zip(rust_re, rust_im, oracle_re, oracle_im)
        ):
            err = max(abs(rr - orr_r), abs(ri - orr_i))
            max_err = max(max_err, err)
            if err > SHORE_TOL:
                raise RuntimeError(
                    f"e3 parity drift in case '{name}' at frame {i}: "
                    f"rust=({rr:.9e},{ri:.9e}) oracle=({orr_r:.9e},{orr_i:.9e}) "
                    f"|err|={err:.3e} > tol {SHORE_TOL:.0e}. The Rust "
                    "OrbitController and the Python oracle (which defers "
                    "to runtime_core.contour_biased_step_py) are running "
                    "different physics — investigate the controller step "
                    "path in src/controller.rs and the per-frame math in "
                    "_oracle_run above."
                )
    return max_err <= SHORE_TOL, max_err


# Trainer-vs-Rust-oracle consistency: this is the check that catches
# drift in the Python differentiable mirror. The trainer's
# ``orbit_controller_momentum_sequence`` implements an analytic surrogate
# for the contour step (because the real ``contour_biased_step`` is not
# PyTorch-differentiable). The surrogate SHOULD match the Rust forward
# dynamics closely enough for gradient descent to learn the right thing;
# if it doesn't, the model trains on physics the browser doesn't run.
#
# Tolerated discrepancy is the SURROGATE GAP, not float rounding. The
# analytic surrogate is known to differ in detail (no S-field curvature,
# no fractal tilt, no wall gating in the analytic path). What this check
# guards against is the much coarser class of bugs that previously made
# it to merge: sign flips, wrong normalization, extra dt factors, wrong
# magnitudes, etc. The bar is "trajectory is qualitatively the same"
# (same direction, same order of magnitude); exact equality is a much
# bigger refactor. For now: if the trajectory stays within an
# envelope-wide tolerance, the surrogate is fit for purpose.
TRAINER_ORACLE_TOL = 0.5  # world units; cardioid is ~0.5 wide


def check_trainer_oracle_consistency(rc) -> tuple[bool, float]:
    """(e4) Trainer forward simulation vs Rust forward dynamics.

    The trainer supervises c by running an *N*-frame forward simulation
    (``orbit_controller_oracle_sequence`` in the c-space proxy path)
    and computing losses against the resulting c. That forward path
    routes the per-frame contour step through
    ``runtime_core.contour_biased_step_py`` — the same Rust function
    the browser executes. So the trainer's forward simulation and the
    Rust ``OrbitController`` should agree to within float rounding.

    A large gap here means the trainer is supervising on physics the
    browser does not run, even though e3 (Rust vs Rust-oracle) is green.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "torch is required for trainer-oracle consistency (e4). "
            "Install backend requirements: pip install -r backend/requirements.txt"
        ) from exc
    if not hasattr(rc, "contour_biased_step_py") or not hasattr(
        rc, "install_pyramid_py"
    ):
        raise RuntimeError(
            "runtime_core missing contour_biased_step_py / install_pyramid_py; "
            "rebuild the wheel."
        )

    from src.cspace_proxies import canonical_hop_dt, orbit_controller_oracle_sequence

    import numpy as np

    # Contract-derived timestep (never restate a literal).
    dt = canonical_hop_dt()

    n_frames = 60
    rng = torch.Generator().manual_seed(1234)
    s_vals = (1.0 + 0.4 * torch.randn(n_frames, generator=rng)).clamp(0.2, 3.0)
    a_vals = torch.rand(n_frames, generator=rng).clamp(0.0, 1.0)
    gates = torch.rand(n_frames, K_RESIDUALS, generator=rng)
    seg = torch.zeros(n_frames, dtype=torch.int64)
    energy = torch.linspace(0.0, 1.0, n_frames, dtype=torch.float32)
    h_vals = torch.zeros(n_frames, dtype=torch.float32)

    # Start both paths at the same point. The trainer's actual call in
    # the c-space proxy path passes a domain-randomized initial_c; for
    # the parity check we use a deterministic boundary point so the
    # trajectory agreement is checkable.
    s0 = float(s_vals[0])
    a0 = float(a_vals[0])
    start = _carrier_reference(a0 * 2.0 * math.pi, s0)
    init_re = torch.full((n_frames,), start.real, dtype=torch.float32)
    init_im = torch.full((n_frames,), start.imag, dtype=torch.float32)
    init_c = torch.complex(init_re, init_im)

    # Trainer forward (used by control_trainer.py for c-space supervision).
    _build_flat_pyramid()
    try:
        trainer_traj = (
            orbit_controller_oracle_sequence(
                s_target=s_vals,
                alpha=a_vals,
                omega=1.0,
                band_gates=gates,
                segment_ids=seg,
                dt=dt,
                drag=0.90,
                thrust=0.0,
                initial_c=init_c,
                energy=energy,
                h=h_vals,
                level=0,
                d_star=0.5,
                max_step=0.05,
            )
            .detach()
            .cpu()
            .numpy()
        )
    finally:
        rc.clear_pyramid_py()

    # Rust runtime: the same OrbitController the browser instantiates.
    # Reset its persistent c to the same starting point as the trainer.
    _build_flat_pyramid()
    try:
        ctrl = rc.OrbitController(s0, a0, 1.0)
        ctrl.set_momentum(True)
        ctrl.set_shore_bias(True)
        ctrl.set_drag(0.90)
        ctrl.set_max_step(0.05)
        ctrl.set_d_star(0.5)
        ctrl.set_level(0)
        ctrl.set_c(start.real, start.imag)
        rust_re_l: list[float] = []
        rust_im_l: list[float] = []
        for i in range(n_frames):
            ctrl.apply_controls(float(s_vals[i]), float(a_vals[i]))
            ctrl.set_energy(float(energy[i]))
            rre, rim = ctrl.step(
                _canonical_dt(),
                [float(g) for g in gates[i]],
                float(h_vals[i]),
            )
            rust_re_l.append(rre)
            rust_im_l.append(rim)
    finally:
        rc.clear_pyramid_py()

    max_err = 0.0
    for i, (pre, pim, rre, rim) in enumerate(
        zip(np.real(trainer_traj), np.imag(trainer_traj), rust_re_l, rust_im_l)
    ):
        err = max(abs(pre - rre), abs(pim - rim))
        max_err = max(max_err, err)
        if err > TRAINER_TOL:
            raise RuntimeError(
                f"e4 trainer-forward vs Rust-runtime drift at frame {i}: "
                f"trainer=({pre:.9e},{pim:.9e}) rust=({rre:.9e},{rim:.9e}) "
                f"|err|={err:.3e} > tol {TRAINER_TOL:.0e}. The trainer's "
                "forward simulation diverges from the Rust runtime even "
                "though both route through contour_biased_step_py. Check "
                "the integrator math in orbit_controller_oracle_sequence "
                "(src/cspace_proxies.py) — likely a missed constant or "
                "sign error."
            )
    return max_err <= TRAINER_TOL, max_err


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
        "e3) Shore-biased dynamics parity (Rust vs Rust-oracle)",
        True,
        check_shore_biased_dynamics_parity,
    ),
    (
        "e4) Trainer-oracle consistency (mirror vs Rust forward)",
        True,
        check_trainer_oracle_consistency,
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
