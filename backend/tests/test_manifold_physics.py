"""Tests for the Mandelbrot-native manifold physics (issue #106).

Covers the acceptance criteria from the issue:

1. Connection validity — the induced metric G(c) is symmetric positive
   definite and its inverse is bounded near the Map resolution floor.
2. Friction non-energy-injecting — metric-consistent drag
   Q_drag = -beta*G*v dissipates: P = v^T Q_drag <= 0 within tolerance.
3. Bounded total-energy drift — with Controls and drag disabled, the
   semi-implicit integrator's total energy E = K + U drifts by a small
   bounded amount per step.
4. Shore crossings — the native potential ridge U = kappa*sigma(c)
   produces shoreward force without the transient-gated wall (h signal
   plays no role in the manifold path).
5. MIP boundaries — Map-derived mechanics (scale, metric) vary
   continuously across the discrete mip rungs of the distance field.

The Rust binding is the canonical source of truth (ADR 0001); these
tests exercise the Python mirror in ``src.cspace_proxies`` against the
Rust bindings directly.

Run: python -m pytest backend/tests/test_manifold_physics.py -q
"""

from __future__ import annotations

import math

import pytest
import torch

from src.cspace_proxies import (
    ManifoldConfig,
    ManifoldEnergyInfo,
    manifold_integrate_step,
    orbit_controller_manifold_sequence,
)

# Tolerances. The Rust integrator uses finite differences (h = 1e-4,
# sized to clear the f32 noise floor of the distance field) for
# gradients and Christoffel symbols, so mirror-vs-Rust agreement is
# limited by that discretization, not float rounding.
PARITY_TOL = 1e-6
ENERGY_DRIFT_TOL = 0.05
FRICTION_TOL = 1e-9


@pytest.fixture(scope="module")
def rc(runtime_core_module):  # noqa: F811 - provided by backend/conftest.py
    """The real compiled runtime_core extension."""
    mod = runtime_core_module
    if not hasattr(mod, "manifold_integrate_step"):
        pytest.skip("runtime_core wheel lacks manifold bindings; rebuild")
    return mod


DEFAULT_CONFIG = ManifoldConfig()


def _rust_step(rc, c, v, q, beta, dt, config):
    """Call the Rust scalar binding with a Python config mirror."""
    rc_config = rc.ManifoldConfig(
        config.d_ref, config.epsilon, config.lambda_sq, config.kappa
    )
    return rc.manifold_integrate_step(
        c[0], c[1], v[0], v[1], q[0], q[1], beta, dt, rc_config
    )


class TestConnectionValidity:
    """Acceptance: connection validity bounded near Map resolution floor."""

    def test_metric_symmetric_positive_definite(self, rc):
        """G(c) must be symmetric with det > 0 and g11 > 0 everywhere
        sampled, including points very close to the Shore."""
        points = [
            (0.0, 0.0),  # deep inside
            (0.25, 0.0),  # near cardioid cusp
            (-0.75, 0.0),  # near period-2 bulb boundary
            (0.2501, 0.0),  # just outside the cusp
            (-1.75, -0.05),  # near antenna
            (0.3, 0.5),  # open water
        ]
        for x, y in points:
            g = rc.manifold_induced_metric(
                complex(x, y),
                rc.ManifoldConfig(
                    DEFAULT_CONFIG.d_ref,
                    DEFAULT_CONFIG.epsilon,
                    DEFAULT_CONFIG.lambda_sq,
                    DEFAULT_CONFIG.kappa,
                ),
            )
            g11, g12, g22 = g[0][0], g[0][1], g[1][1]
            assert g12 == pytest.approx(g[1][0], abs=1e-12), f"asymmetric at ({x},{y})"
            det = g11 * g22 - g12 * g12
            assert det > 0.0, f"metric not PD at ({x},{y}): det={det}"
            assert g11 > 0.0, f"metric g11 <= 0 at ({x},{y})"

    def test_metric_inverse_bounded(self, rc):
        """G^{-1} must stay bounded near the resolution floor: the
        regularized distance rho = sqrt(D^2 + eps^2) keeps the metric
        gradient finite even exactly on the boundary."""
        config = rc.ManifoldConfig(0.1, 1e-4, 1.0, 1.0)
        # Points straddling the boundary of the main cardioid.
        for x in (-0.76, -0.75, -0.749, 0.249, 0.25, 0.251):
            g = rc.manifold_induced_metric(complex(x, 0.0), config)
            det = g[0][0] * g[1][1] - g[0][1] * g[0][1]
            inv_det = 1.0 / det
            # Bounded: no blow-up at the resolution floor.
            assert math.isfinite(inv_det)
            assert inv_det < 1e12, f"metric near-singular at x={x}: 1/det={inv_det}"

    def test_christoffel_finite_everywhere_sampled(self, rc):
        """The connection must be finite across Shore crossings."""
        config = rc.ManifoldConfig(0.1, 1e-4, 1.0, 1.0)
        for x in (-0.8, -0.75, -0.7, 0.2, 0.25, 0.3):
            gamma = rc.manifold_christoffel_symbols(complex(x, 0.0), config)
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        assert math.isfinite(gamma[i][j][k]), (
                            f"non-finite Gamma at x={x}, ({i},{j},{k})"
                        )


class TestFrictionNonEnergyInjecting:
    """Acceptance: friction cannot inject energy (P = v^T Q_drag <= 0)."""

    def test_drag_power_nonpositive(self, rc):
        """v^T Q_drag = -beta * v^T G v <= 0 for all sampled states."""
        config = rc.ManifoldConfig(0.1, 1e-4, 1.0, 1.0)
        beta = 0.1
        velocities = [(0.1, 0.0), (0.0, -0.2), (0.05, 0.05), (-0.3, 0.15)]
        points = [(0.0, 0.0), (0.25, 0.0), (-0.75, 0.0), (0.4, 0.4)]
        for x, y in points:
            for vx, vy in velocities:
                q = rc.manifold_drag_force(vx, vy, complex(x, y), beta, config)
                power = vx * q[0] + vy * q[1]
                assert power <= FRICTION_TOL, (
                    f"drag injected energy at ({x},{y}) v=({vx},{vy}): P={power}"
                )

    def test_drag_reduces_kinetic_energy(self, rc):
        """A step with drag but no other forces must not increase K."""
        config = ManifoldConfig()
        c = torch.tensor(0.0)
        v = torch.tensor(0.2)
        zero = torch.tensor(0.0)
        _, _, _, _, info = manifold_integrate_step(
            c, c, v, v, zero, zero, beta=0.5, dt=0.01, config=config
        )
        assert info.delta_kinetic <= 1e-6, (
            f"kinetic energy increased under pure drag: {info.delta_kinetic}"
        )


class TestEnergyDrift:
    """Acceptance: bounded total-energy drift with Controls/friction off."""

    def test_energy_drift_bounded_no_forces(self, rc):
        """Free geodesic motion (Q=0, beta=0) must conserve E to within
        the semi-implicit integrator's O(dt) error."""
        config = ManifoldConfig()
        dt = 0.01
        c = (0.0, 0.0)
        v = (0.01, 0.01)
        max_drift = 0.0
        for _ in range(50):
            new_re, new_im, new_vx, new_vy, info = _rust_step(
                rc, c, v, (0.0, 0.0), 0.0, dt, config
            )
            max_drift = max(max_drift, abs(info.delta_total))
            c = (new_re, new_im)
            v = (new_vx, new_vy)
        assert max_drift < ENERGY_DRIFT_TOL, (
            f"unbounded energy drift: {max_drift} > {ENERGY_DRIFT_TOL}"
        )

    def test_energy_drift_bounded_through_shore_crossing(self, rc):
        """Drift stays bounded even when the trajectory crosses the
        potential ridge (the regularized distance keeps U smooth)."""
        config = ManifoldConfig()
        dt = 0.005
        # Start outside the set moving inward toward the cardioid.
        c = (0.5, 0.0)
        v = (-0.05, 0.0)
        max_drift = 0.0
        for _ in range(100):
            new_re, new_im, new_vx, new_vy, info = _rust_step(
                rc, c, v, (0.0, 0.0), 0.0, dt, config
            )
            max_drift = max(max_drift, abs(info.delta_total))
            c = (new_re, new_im)
            v = (new_vx, new_vy)
        assert max_drift < ENERGY_DRIFT_TOL, (
            f"unbounded drift across Shore: {max_drift}"
        )


class TestShoreCrossings:
    """Acceptance: native potential-ridge Shore crossings without the
    transient-gated wall."""

    def test_potential_force_points_downhill_from_ridge(self, rc):
        """U = kappa*sigma makes the Shore a HIGH-potential ridge, so the
        potential force F_U = -G^-1 grad U must point AWAY from the Shore
        (downhill, toward lower scale). At (0.5, 0) the Shore lies in the
        -x direction; the force must point +x."""
        config = rc.ManifoldConfig(0.1, 1e-4, 1.0, 1.0)
        f = rc.manifold_potential_force(complex(0.5, 0.0), config)
        assert f[0] > 0.0, (
            f"potential force does not point downhill from the ridge: F=({f[0]},{f[1]})"
        )

    def test_h_signal_irrelevant_to_manifold_path(self, rc):
        """The manifold integrator takes no h argument: crossing the
        potential ridge is governed by U = kappa*sigma, not by the
        transient gate. Verify the binding signature has no h parameter
        and that repeated calls are deterministic."""
        rc_config = rc.ManifoldConfig(0.1, 1e-4, 1.0, 1.0)
        # The Rust step runs identically regardless of any h-like state:
        # call it twice with identical inputs, results are identical.
        a = rc.manifold_integrate_step(
            0.3, 0.1, 0.02, -0.01, 0.001, 0.001, 0.1, 0.01, rc_config
        )
        b = rc.manifold_integrate_step(
            0.3, 0.1, 0.02, -0.01, 0.001, 0.001, 0.1, 0.01, rc_config
        )
        assert a[:4] == b[:4]
        # The binding's parameter list must not contain an h-like gate:
        # the only f64 params are c(2), v(2), q(2), beta, dt.
        import inspect

        params = list(inspect.signature(rc.manifold_integrate_step).parameters)
        assert len(params) == 9, f"unexpected binding signature: {params}"
        assert not any(p.lower() == "h" for p in params), (
            f"manifold binding exposes an h gate: {params}"
        )

    def test_ridge_is_barrier_without_h_gate(self, rc):
        """The manifold path has NO transient-gated wall: the potential
        ridge U = kappa*sigma is itself the barrier. A particle launched
        at the Shore with modest KE must stall (D stays positive) — the
        ridge repels it — and the binding exposes no h argument."""
        config = ManifoldConfig()
        dt = 0.005
        c = (0.6, 0.0)
        v = (-0.05, 0.0)
        min_d = math.inf
        for _ in range(200):
            new_re, new_im, new_vx, new_vy, _ = _rust_step(
                rc, c, v, (0.0, 0.0), 0.0, dt, config
            )
            c = (new_re, new_im)
            v = (new_vx, new_vy)
            min_d = min(min_d, rc.manifold_signed_distance(complex(new_re, new_im)))
        # The particle never reaches the boundary: the ridge repels it.
        assert min_d > 0.0, (
            f"particle crossed the ridge without a gate: min D = {min_d}"
        )
        # And it did not run away to infinity either (bounded stall).
        assert abs(new_re) < 10.0, f"unbounded escape: c=({new_re},{new_im})"


class TestMipBoundaries:
    """Acceptance: Map-derived mechanics vary continuously across MIP
    boundaries."""

    def test_scale_continuous_across_distance_field_resolution(self, rc):
        """sigma(c) computed from the builtin 1024 distance field must
        be finite and vary smoothly along a sampled line — no
        discontinuous jumps at rung boundaries."""
        # Load the canonical builtin field into the in-memory slot
        # (get_builtin_distance_field_py installs it and returns metadata).
        meta = rc.get_builtin_distance_field_py("default")
        assert len(meta) == 6, f"unexpected builtin metadata: {meta}"
        config = rc.ManifoldConfig(0.1, 1e-4, 1.0, 1.0)
        # Sample a line crossing several resolution bands.
        sigmas = [
            rc.manifold_mandelbrot_scale(complex(x, 0.1), config)
            for x in (-1.5, -1.0, -0.5, 0.0, 0.5)
        ]
        for s in sigmas:
            assert math.isfinite(s)
        # Continuity: adjacent samples differ by a bounded amount
        # (no cliff). Scale changes smoothly with position.
        for a, b in zip(sigmas, sigmas[1:]):
            assert abs(a - b) < 5.0, f"scale cliff: {a} -> {b}"

    def test_metric_continuous_across_shore(self, rc):
        """G(c) varies continuously through the boundary: sampling
        symmetric points around the Shore gives nearby metrics. The
        crest gradient is steep (sigma_x ~ 1/D), so the tolerance is
        loose — the assertion is that the metric is finite and of the
        same order on both sides, not that it is flat."""
        config = rc.ManifoldConfig(0.1, 1e-4, 1.0, 1.0)
        eps = 1e-3
        g_out = rc.manifold_induced_metric(complex(0.25 + eps, 0.0), config)
        g_in = rc.manifold_induced_metric(complex(0.25 - eps, 0.0), config)
        for i in range(2):
            for j in range(2):
                assert math.isfinite(g_out[i][j]) and math.isfinite(g_in[i][j])
                # Same order of magnitude (within 4x) on both sides.
                ratio = max(g_out[i][j], g_in[i][j]) / max(
                    min(g_out[i][j], g_in[i][j]), 1e-12
                )
                assert ratio < 4.0, (
                    f"metric discontinuous across Shore at ({i},{j}): "
                    f"{g_out[i][j]} vs {g_in[i][j]}"
                )


class TestMirrorParity:
    """The Python mirror must reproduce the Rust integrator."""

    def test_mirror_matches_rust_single_step(self, rc):
        """manifold_integrate_step (mirror) vs the Rust binding."""
        config = ManifoldConfig()
        c_re = torch.tensor(0.3, requires_grad=True)
        c_im = torch.tensor(0.1, requires_grad=True)
        v_re = torch.tensor(0.02)
        v_im = torch.tensor(-0.01)
        q_re = torch.tensor(0.001)
        q_im = torch.tensor(0.001)

        new_re, new_im, _, _, _ = manifold_integrate_step(
            c_re, c_im, v_re, v_im, q_re, q_im, beta=0.1, dt=0.01, config=config
        )
        r_re, r_im, r_vx, r_vy, _ = _rust_step(
            rc, (0.3, 0.1), (0.02, -0.01), (0.001, 0.001), 0.1, 0.01, config
        )
        assert new_re.item() == pytest.approx(r_re, abs=PARITY_TOL)
        assert new_im.item() == pytest.approx(r_im, abs=PARITY_TOL)

    def test_mirror_gradients_flow(self, rc):
        """Gradients must flow through the identity surrogate."""
        config = ManifoldConfig()
        c_re = torch.tensor(0.3, requires_grad=True)
        c_im = torch.tensor(0.1, requires_grad=True)
        new_re, new_im, _, _, _ = manifold_integrate_step(
            c_re,
            c_im,
            torch.tensor(0.02),
            torch.tensor(-0.01),
            torch.tensor(0.001),
            torch.tensor(0.001),
            beta=0.1,
            dt=0.01,
            config=config,
        )
        loss = new_re + new_im
        loss.backward()
        assert c_re.grad is not None and c_re.grad.item() != 0.0
        assert c_im.grad is not None and c_im.grad.item() != 0.0


class TestManifoldSequence:
    """The per-frame sequence mirror of OrbitController::step_manifold."""

    def test_sequence_matches_rust_controller(self, rc):
        """orbit_controller_manifold_sequence vs the Rust OrbitController
        with manifold_physics enabled — same controls, same trajectory."""
        n_frames = 40
        dt = 1.0 / 60.0  # sequence test; parity preflight uses canonical dt
        omega = 1.0
        drag = 0.1
        config = ManifoldConfig()

        s_vals = [1.0 + 0.1 * math.cos(i * 0.2) for i in range(n_frames)]
        a_vals = [0.5 + 0.1 * math.sin(i * 0.2) for i in range(n_frames)]
        gates = [[0.5] * 6 for _ in range(n_frames)]
        energy = [0.5] * n_frames

        # Rust controller.
        ctrl = rc.OrbitController(s_vals[0], a_vals[0], omega)
        ctrl.set_manifold_physics(True)
        ctrl.set_manifold_drag(drag)
        ctrl.set_manifold_config(
            rc.ManifoldConfig(
                config.d_ref, config.epsilon, config.lambda_sq, config.kappa
            )
        )
        ctrl.set_energy(energy[0])
        rust_traj = [ctrl.step(dt, gates[0], 0.0)]
        for i in range(1, n_frames):
            ctrl.apply_controls(s_vals[i], a_vals[i])
            ctrl.set_energy(energy[i])
            rust_traj.append(ctrl.step(dt, gates[i], 0.0))

        # Python mirror.
        s_t = torch.tensor(s_vals, dtype=torch.float32)
        a_t = torch.tensor(a_vals, dtype=torch.float32)
        g_t = torch.tensor(gates, dtype=torch.float32)
        e_t = torch.tensor(energy, dtype=torch.float32)
        seg = torch.zeros(n_frames, dtype=torch.int64)
        traj, _infos = orbit_controller_manifold_sequence(
            s_target=s_t,
            alpha=a_t,
            omega=omega,
            band_gates=g_t,
            segment_ids=seg,
            dt=dt,
            energy=e_t,
            manifold_drag=drag,
            config=config,
        )

        # The mirror starts at c=(0,0) like the Rust default. The mirror
        # accumulates state in float32 while Rust is float64, so the gap
        # grows ~2e-6/frame; over 40 frames that is ~1e-4. The tolerance
        # must absorb f32 accumulation but still catch real divergence
        # (sign flips, wrong constants are O(1) errors).
        max_err = 0.0
        for i in range(n_frames):
            err = max(
                abs(traj[i].real - rust_traj[i][0]),
                abs(traj[i].imag - rust_traj[i][1]),
            )
            max_err = max(max_err, err)
        assert max_err < 1e-4, (
            f"manifold sequence diverged from Rust controller: {max_err:.3e}"
        )

    def test_sequence_energy_infos_populated(self, rc):
        """Every frame returns an energy diagnostic."""
        n = 10
        traj, infos = orbit_controller_manifold_sequence(
            s_target=torch.ones(n),
            alpha=torch.linspace(0, 1, n),
            omega=1.0,
            band_gates=torch.ones(n, 6) * 0.5,
            segment_ids=torch.zeros(n, dtype=torch.int64),
            dt=1.0 / 60.0,
            energy=torch.linspace(0, 1, n),
        )
        assert traj.shape == (n,)
        assert len(infos) == n
        for info in infos:
            assert isinstance(info, ManifoldEnergyInfo)
            assert math.isfinite(info.total)

    def test_sequence_gradients_flow_to_targets(self, rc):
        """Loss on the final position backprops to s_target/alpha."""
        n = 5
        s_t = torch.tensor([1.0] * n, requires_grad=True)
        a_t = torch.linspace(0, 1, n)
        traj, _ = orbit_controller_manifold_sequence(
            s_target=s_t,
            alpha=a_t,
            omega=1.0,
            band_gates=torch.ones(n, 6) * 0.5,
            segment_ids=torch.zeros(n, dtype=torch.int64),
            dt=1.0 / 60.0,
            energy=torch.ones(n) * 0.5,
        )
        loss = traj[-1].real + traj[-1].imag
        loss.backward()
        assert s_t.grad is not None
        assert s_t.grad.abs().sum() > 0.0, "gradient is zero"
