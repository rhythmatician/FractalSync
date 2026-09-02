"""Tests for the Mandelbrot-native manifold physics (issue #106).

Covers the acceptance criteria from the issue:

1. Connection validity — the induced metric G(c) is symmetric positive
   definite and its inverse is bounded near the Map resolution floor.
2. Friction non-energy-injecting — metric-consistent drag
   Q_drag = -beta*G*v dissipates: P = v^T Q_drag <= 0 within tolerance.
3. Bounded total-energy drift — with Controls and drag disabled, the
   semi-implicit integrator's total energy E = K + U drifts by a small
   bounded amount per step (rollout-level conservative tests).
4. Shore-ridge mechanics — the native potential U = kappa*sigma(c)
   creates a finite mechanical barrier without a transient-gated wall.
   Underpowered trajectories reflect; higher-energy launches reach the
   regularized crest neighborhood. Exact native crossing remains a
   follow-up under #106/#82 while near-crest derivative quality is improved.
5. Signed-SDF continuity — mechanics derive from the SINGLE signed
   distance field authority; sigma/gradient/metric vary continuously
   through the regularized Shore crest with no dependence on a discrete
   mip level.

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

# Tolerances. The Rust integrator derives its finite-difference step from
# the distance-field provider's pixel spacing (currently px/24, ~1e-4), so
# mirror-vs-Rust agreement is limited by that sampled geometry rather than
# float rounding.
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


def _locate_shore_x(rc, y=0.0, x_lo=0.2, x_hi=0.5):
    """Bisect along the line y=const to find the Shore crossing where the
    signed distance changes sign. Returns the x where D ~ 0."""
    lo, hi = x_lo, x_hi
    assert rc.manifold_signed_distance(complex(lo, y)) < 0.0, (
        f"x_lo={lo} not inside the set"
    )
    assert rc.manifold_signed_distance(complex(hi, y)) > 0.0, (
        f"x_hi={hi} not outside the set"
    )
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if rc.manifold_signed_distance(complex(mid, y)) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


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
        for x in (-0.76, -0.75, -0.749, 0.249, 0.25, 0.251):
            g = rc.manifold_induced_metric(complex(x, 0.0), config)
            det = g[0][0] * g[1][1] - g[0][1] * g[0][1]
            inv_det = 1.0 / det
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

    def test_energy_drift_bounded_near_shore(self, rc):
        """Drift stays bounded in a high-curvature region just outside the
        Shore (the regularized distance keeps U smooth). The trajectory
        approaches but does not necessarily cross the Shore — native
        crossability remains a separate #106/#82 acceptance question."""
        config = ManifoldConfig()
        dt = 0.005
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
        assert max_drift < ENERGY_DRIFT_TOL, f"unbounded drift near Shore: {max_drift}"

    def _rollout_energy_drift(self, rc, c0, v0, dt, n_steps, config):
        """Run a conservative rollout and return final/max/relative drift."""
        e0 = rc.manifold_kinetic_energy(v0[0], v0[1], complex(*c0), config) + (
            rc.manifold_potential_energy(complex(*c0), config)
        )
        c = c0
        v = v0
        max_excursion = 0.0
        e = e0
        for _ in range(n_steps):
            new_re, new_im, new_vx, new_vy, _ = _rust_step(
                rc, c, v, (0.0, 0.0), 0.0, dt, config
            )
            c = (new_re, new_im)
            v = (new_vx, new_vy)
            e = rc.manifold_kinetic_energy(v[0], v[1], complex(*c), config) + (
                rc.manifold_potential_energy(complex(*c), config)
            )
            max_excursion = max(max_excursion, abs(e - e0))
        rel = abs(e - e0) / max(abs(e0), 1e-12)
        return abs(e - e0), max_excursion, rel

    def test_energy_conserved_low_curvature_rollout(self, rc):
        config = rc.ManifoldConfig(0.1, 1e-4, 1.0, 1.0)
        dt = 0.002
        c0 = (0.0, 0.9)
        v0 = (0.01, 0.0)
        drift, excursion, _ = self._rollout_energy_drift(
            rc, c0, v0, dt, n_steps=150, config=config
        )
        assert drift < ENERGY_DRIFT_TOL
        assert excursion < ENERGY_DRIFT_TOL

    def test_energy_conserved_high_curvature_deep_scale_rollout(self, rc):
        """FD ∇σ/Hσ noise propagating into analytic Γ must remain bounded."""
        config = rc.ManifoldConfig(0.1, 1e-4, 1.0, 1.0)
        dt = 0.002
        c0 = (0.35, 0.0)
        v0 = (0.0, 0.02)
        drift, excursion, _ = self._rollout_energy_drift(
            rc, c0, v0, dt, n_steps=150, config=config
        )
        assert drift < ENERGY_DRIFT_TOL
        assert excursion < ENERGY_DRIFT_TOL

    def test_energy_conserved_near_shore_rollout(self, rc):
        config = rc.ManifoldConfig(0.1, 1e-4, 1.0, 1.0)
        dt = 0.002
        c0 = (0.30, 0.0)
        v0 = (0.0, 0.02)
        drift, excursion, _ = self._rollout_energy_drift(
            rc, c0, v0, dt, n_steps=100, config=config
        )
        assert drift < ENERGY_DRIFT_TOL
        assert excursion < ENERGY_DRIFT_TOL


class TestShoreCrossings:
    """Native Shore-ridge mechanics without a music-aware gate."""

    def test_potential_force_points_downhill_from_ridge(self, rc):
        config = rc.ManifoldConfig(0.1, 1e-4, 1.0, 1.0)
        f = rc.manifold_potential_force(complex(0.5, 0.0), config)
        assert f[0] > 0.0

    def test_h_signal_irrelevant_to_manifold_path(self, rc):
        rc_config = rc.ManifoldConfig(0.1, 1e-4, 1.0, 1.0)
        a = rc.manifold_integrate_step(
            0.3, 0.1, 0.02, -0.01, 0.001, 0.001, 0.1, 0.01, rc_config
        )
        b = rc.manifold_integrate_step(
            0.3, 0.1, 0.02, -0.01, 0.001, 0.001, 0.1, 0.01, rc_config
        )
        assert a[:4] == b[:4]
        import inspect

        params = list(inspect.signature(rc.manifold_integrate_step).parameters)
        assert len(params) == 9
        assert not any(p.lower() == "h" for p in params)

    def test_ridge_is_barrier_without_h_gate(self, rc):
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
        assert min_d > 0.0
        assert abs(new_re) < 10.0

    def test_shore_ridge_energy_ordering_and_current_crest_limit(self, rc):
        """Compare launches by actual manifold KE and preserve the current
        near-crest numerical limitation as explicit evidence.

        The foundation must show that a low-energy launch reflects and that a
        higher-energy launch penetrates substantially farther into the ridge.
        With the current raster-derived ∇σ/Hσ, exact crossing is not yet a
        stable quantitative acceptance criterion: a 2.5x-barrier launch can
        stall around D≈epsilon. That remains open under #106/#82 rather than
        being hidden by an arbitrary larger launch factor.
        """
        config = ManifoldConfig(0.1, 1e-4, 1.0, 0.1)
        rc_config = rc.ManifoldConfig(0.1, 1e-4, 1.0, 0.1)
        dt = 0.001

        c0 = (0.35, 0.0)
        x_shore = _locate_shore_x(rc, y=0.0, x_lo=0.2, x_hi=0.5)
        assert abs(x_shore - 0.25) < 0.05

        u0 = rc.manifold_potential_energy(complex(*c0), rc_config)
        u_crest = rc.manifold_potential_energy(complex(x_shore, 0.0), rc_config)
        barrier = u_crest - u0
        assert barrier > 0.0

        g0 = rc.manifold_induced_metric(complex(*c0), rc_config)
        g_xx = g0[0][0]
        assert math.isfinite(g_xx) and g_xx > 0.0

        def _vx_for_ke(target_ke):
            # For v=(-vx,0), K = 1/2 * G_xx(c0) * vx^2.
            vx = math.sqrt(2.0 * target_ke / g_xx)
            measured = rc.manifold_kinetic_energy(vx, 0.0, complex(*c0), rc_config)
            assert measured == pytest.approx(target_ke, rel=1e-9, abs=1e-12)
            return vx

        def _crest_attempt(vx):
            c = c0
            v = (-vx, 0.0)
            crossed = False
            min_d = math.inf
            energy_log = []
            for _ in range(5000):
                new_re, new_im, new_vx, new_vy, _ = _rust_step(
                    rc, c, v, (0.0, 0.0), 0.0, dt, config
                )
                c = (new_re, new_im)
                v = (new_vx, new_vy)
                d = rc.manifold_signed_distance(complex(*c))
                min_d = min(min_d, d)
                if d < 0.0:
                    crossed = True
                    k = rc.manifold_kinetic_energy(v[0], v[1], complex(*c), rc_config)
                    u = rc.manifold_potential_energy(complex(*c), rc_config)
                    energy_log.append((k, u, k + u))
                    break
            return crossed, min_d, c, energy_log

        ke_under = 0.5 * barrier
        vx_under = _vx_for_ke(ke_under)
        crossed_under, min_d_under, _, _ = _crest_attempt(vx_under)
        assert not crossed_under, (
            f"underpowered trajectory crested the ridge: min D={min_d_under}"
        )
        assert min_d_under > 0.0

        ke_high = 2.5 * barrier
        vx_high = _vx_for_ke(ke_high)
        crossed_high, min_d_high, _, energy_log = _crest_attempt(vx_high)

        # More mechanical energy must buy real progress up the same ridge.
        assert min_d_high < min_d_under, (
            f"higher-energy launch did not penetrate farther: "
            f"low={min_d_under}, high={min_d_high}"
        )

        if crossed_high:
            # Future improvements are allowed to make this case genuinely
            # cross without changing the test's semantic contract.
            assert energy_log
            k, u, e = energy_log[0]
            assert math.isfinite(k) and math.isfinite(u) and math.isfinite(e)
        else:
            # Current implementation reaches the regularization neighborhood
            # but may stick just outside D=0 because FD ∇σ/Hσ error propagates
            # into analytic Γ. Keep that limitation visible and bounded.
            assert min_d_high <= 2.0 * config.epsilon, (
                f"higher-energy launch did not reach the crest neighborhood: "
                f"min D={min_d_high}, epsilon={config.epsilon}"
            )


class TestSignedSdfContinuity:
    """Mechanics derive from one signed distance-field authority."""

    def test_signed_distance_authority_sign_and_continuity(self, rc):
        meta = rc.get_builtin_distance_field_py("default")
        assert len(meta) == 6
        assert rc.manifold_signed_distance(complex(0.0, 0.0)) < 0.0
        assert rc.manifold_signed_distance(complex(0.5, 0.0)) > 0.0
        x_shore = _locate_shore_x(rc, y=0.0, x_lo=0.2, x_hi=0.5)
        prev = None
        for x in [
            x_shore - 5e-3,
            x_shore - 2e-3,
            x_shore - 1e-3,
            x_shore,
            x_shore + 1e-3,
            x_shore + 2e-3,
            x_shore + 5e-3,
        ]:
            d = rc.manifold_signed_distance(complex(x, 0.0))
            assert math.isfinite(d)
            if prev is not None:
                assert abs(d - prev) < 0.05
            prev = d

    def test_signed_distance_unsigned_consistency(self, rc):
        meta = rc.get_builtin_distance_field_py("default")
        assert len(meta) == 6
        pts = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (-0.75, 0.0), (0.3, 0.5)]
        unsigned = rc.sample_distance_field_py([complex(x, y) for x, y in pts])
        for (x, y), u in zip(pts, unsigned):
            signed = rc.manifold_signed_distance(complex(x, y))
            assert u == pytest.approx(abs(signed), abs=1e-6)

    def test_scale_continuous_through_shore_crest(self, rc):
        meta = rc.get_builtin_distance_field_py("default")
        assert len(meta) == 6
        config = rc.ManifoldConfig(0.1, 1e-4, 1.0, 1.0)
        xs = [0.24 + 0.001 * i for i in range(21)]
        sigmas = [rc.manifold_mandelbrot_scale(complex(x, 0.0), config) for x in xs]
        for s in sigmas:
            assert math.isfinite(s)
        for a, b in zip(sigmas, sigmas[1:]):
            assert abs(a - b) < 20.0

    def test_metric_continuous_across_shore(self, rc):
        config = rc.ManifoldConfig(0.1, 1e-4, 1.0, 1.0)
        eps = 1e-3
        g_out = rc.manifold_induced_metric(complex(0.25 + eps, 0.0), config)
        g_in = rc.manifold_induced_metric(complex(0.25 - eps, 0.0), config)
        for i in range(2):
            for j in range(2):
                assert math.isfinite(g_out[i][j]) and math.isfinite(g_in[i][j])
                ratio = max(g_out[i][j], g_in[i][j]) / max(
                    min(g_out[i][j], g_in[i][j]), 1e-12
                )
                assert ratio < 4.0


class TestMirrorParity:
    """The Python mirror must reproduce the Rust integrator."""

    def test_mirror_matches_rust_single_step(self, rc):
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
        r_re, r_im, _, _, _ = _rust_step(
            rc, (0.3, 0.1), (0.02, -0.01), (0.001, 0.001), 0.1, 0.01, config
        )
        assert new_re.item() == pytest.approx(r_re, abs=PARITY_TOL)
        assert new_im.item() == pytest.approx(r_im, abs=PARITY_TOL)

    def test_mirror_ste_smoke_gradients_flow(self, rc):
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
        n_frames = 40
        dt = 1.0 / 60.0
        omega = 1.0
        drag = 0.1
        config = ManifoldConfig()

        s_vals = [1.0 + 0.1 * math.cos(i * 0.2) for i in range(n_frames)]
        a_vals = [0.5 + 0.1 * math.sin(i * 0.2) for i in range(n_frames)]
        gates = [[0.5] * 6 for _ in range(n_frames)]
        energy = [0.5] * n_frames

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

        max_err = 0.0
        for i in range(n_frames):
            err = max(
                abs(traj[i].real - rust_traj[i][0]),
                abs(traj[i].imag - rust_traj[i][1]),
            )
            max_err = max(max_err, err)
        assert max_err < 1e-4

    def test_sequence_energy_infos_populated(self, rc):
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

    def test_sequence_ste_smoke_gradients_flow_to_targets(self, rc):
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
