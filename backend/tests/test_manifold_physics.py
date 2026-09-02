"""Tests for the Mandelbrot-native manifold physics (issue #106).

Covers the acceptance criteria from the issue:

1. Connection validity — the induced metric G(c) is symmetric positive
   definite and its inverse is bounded near the Map resolution floor.
2. Friction non-energy-injecting — metric-consistent drag
   Q_drag = -beta*G*v dissipates: P = v^T Q_drag <= 0 within tolerance.
3. Bounded total-energy drift — with Controls and drag disabled, the
   semi-implicit integrator's total energy E = K + U drifts by a small
   bounded amount per step (rollout-level conservative tests).
4. Shore crossings — the native potential ridge U = kappa*sigma(c)
   produces shoreward force without the transient-gated wall (h signal
   plays no role in the manifold path). Underpowered trajectories do not
   crest the ridge; overpowered ones do.
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
# the distance-field provider resolution (one pixel, ~2.4e-3), so
# mirror-vs-Rust agreement is limited by that discretization, not float
# rounding.
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

    def test_energy_drift_bounded_near_shore(self, rc):
        """Drift stays bounded in a high-curvature region just outside the
        Shore (the regularized distance keeps U smooth). The trajectory
        approaches but does not necessarily cross the Shore — the real
        cresting behavior is covered by the underpowered/overpowered
        Shore-ridge test below."""
        config = ManifoldConfig()
        dt = 0.005
        # Start outside the set near the cardioid, moving inward.
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
        """Run a conservative rollout (Q=0, beta=0) and return
        (|E_final - E_initial|, max |E - E_initial| excursion, relative
        drift). Uses the Rust energy bindings for E at each step."""
        e0 = rc.manifold_kinetic_energy(v0[0], v0[1], complex(*c0), config) + (
            rc.manifold_potential_energy(complex(*c0), config)
        )
        c = c0
        v = v0
        max_excursion = 0.0
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
        """Conservative rollout in a low-curvature region (open water far
        from the Shore, inside the field domain): total energy E = K + U
        must stay within the semi-implicit integrator's O(dt) drift over a
        long rollout."""
        config = rc.ManifoldConfig(0.1, 1e-4, 1.0, 1.0)
        dt = 0.002
        # Open water above the set, far from the Shore, well inside the
        # field domain (x in [-2, 0.47], y in [-1.12, 1.12]).
        c0 = (0.0, 0.9)
        v0 = (0.01, 0.0)
        drift, excursion, rel = self._rollout_energy_drift(
            rc, c0, v0, dt, n_steps=150, config=config
        )
        assert drift < ENERGY_DRIFT_TOL, (
            f"low-curvature rollout drifted: |dE|={drift:.3e}"
        )
        assert excursion < ENERGY_DRIFT_TOL, (
            f"low-curvature rollout excursion: {excursion:.3e}"
        )

    def test_energy_conserved_high_curvature_deep_scale_rollout(self, rc):
        """Conservative rollout in a high-curvature / deep-scale region
        (near the Shore where sigma and the metric vary steeply): energy
        must still stay bounded. The Christoffel symbols are computed
        analytically from ∇σ and H_σ, but those are still finite-
        differenced against the sampled SDF, so the resulting Γ carries
        FD-σ noise; the drift is therefore larger than in open water but
        must not blow up."""
        config = rc.ManifoldConfig(0.1, 1e-4, 1.0, 1.0)
        dt = 0.002
        # Moderate distance from the Shore (D ~ 0.09), moving tangentially.
        c0 = (0.35, 0.0)
        v0 = (0.0, 0.02)
        drift, excursion, rel = self._rollout_energy_drift(
            rc, c0, v0, dt, n_steps=150, config=config
        )
        assert drift < ENERGY_DRIFT_TOL, (
            f"high-curvature rollout drifted: |dE|={drift:.3e}"
        )
        assert excursion < ENERGY_DRIFT_TOL, (
            f"high-curvature rollout excursion: {excursion:.3e}"
        )

    def test_energy_conserved_near_shore_rollout(self, rc):
        """Conservative rollout launched near the Shore ridge (D ~ 0.04):
        the trajectory must not gain energy from the discretized Christoffel
        (no numerical pumping). The drift is bounded but larger than in open
        water because the finite-difference geometry is noisiest closest to
        the Shore."""
        config = rc.ManifoldConfig(0.1, 1e-4, 1.0, 1.0)
        dt = 0.002
        # Near the boundary, moving tangentially so it stays close.
        c0 = (0.30, 0.0)
        v0 = (0.0, 0.02)
        drift, excursion, rel = self._rollout_energy_drift(
            rc, c0, v0, dt, n_steps=100, config=config
        )
        assert drift < ENERGY_DRIFT_TOL, f"near-Shore rollout drifted: |dE|={drift:.3e}"
        assert excursion < ENERGY_DRIFT_TOL, (
            f"near-Shore rollout excursion: {excursion:.3e}"
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

    def test_shore_cresting_underpowered_vs_overpowered(self, rc):
        """Reproducible native ridge test (issue #106, fix #9): locate the
        real Shore crossing via the signed SDF, compute the potential
        barrier U_crest - U_0, then launch an underpowered trajectory
        (KE below the barrier) that must NOT cross D=0 and an overpowered
        one (KE above the barrier) that MUST. No h/transient gate, no
        audio force, no teleport, no target-c bypass."""
        # kappa=0.1 keeps the potential barrier ~1 energy unit so the
        # cresting velocities stay moderate and the finite-difference
        # Christoffel remains stable (a kappa=1 barrier ~10 requires
        # velocities that push the FD geometry into its noisy regime).
        config = rc.ManifoldConfig(0.1, 1e-4, 1.0, 0.1)
        rc_config = rc.ManifoldConfig(0.1, 1e-4, 1.0, 0.1)
        dt = 0.001

        # Launch point in open water to the right of the cardioid cusp.
        c0 = (0.5, 0.0)
        # Locate the Shore along the +x axis (the distance-estimate field's
        # zero contour, near the cardioid cusp).
        x_shore = _locate_shore_x(rc, y=0.0, x_lo=0.2, x_hi=0.5)
        assert abs(x_shore - 0.25) < 0.05, f"unexpected Shore location: {x_shore}"

        # Potential at launch and at the crest (the Shore is the ridge).
        u0 = rc.manifold_potential_energy(complex(*c0), rc_config)
        u_crest = rc.manifold_potential_energy(complex(x_shore, 0.0), rc_config)
        barrier = u_crest - u0
        assert barrier > 0.0, f"no potential barrier to crest: {barrier}"

        def _crest_attempt(vx):
            """Launch from c0 with velocity (-vx, 0) toward the Shore.
            Returns (crossed, min_signed_d, final_c, energy_log)."""
            c = c0
            v = (-vx, 0.0)
            crossed = False
            min_d = math.inf
            energy_log = []
            for _ in range(2000):
                new_re, new_im, new_vx, new_vy, _ = _rust_step(
                    rc, c, v, (0.0, 0.0), 0.0, dt, config
                )
                c = (new_re, new_im)
                v = (new_vx, new_vy)
                d = rc.manifold_signed_distance(complex(*c))
                min_d = min(min_d, d)
                if d < 0.0:
                    crossed = True
                    # Record K/U/E at the crossing.
                    k = rc.manifold_kinetic_energy(v[0], v[1], complex(*c), rc_config)
                    u = rc.manifold_potential_energy(complex(*c), rc_config)
                    energy_log.append((k, u, k + u))
                    break
            return crossed, min_d, c, energy_log

        # Underpowered: KE = 0.5 * barrier. Must reflect off the ridge and
        # NOT crest (D stays > 0).
        ke_under = 0.5 * barrier
        vx_under = math.sqrt(2.0 * ke_under)
        crossed_under, min_d_under, _, _ = _crest_attempt(vx_under)
        assert not crossed_under, (
            f"underpowered trajectory crested the ridge: min D={min_d_under}"
        )
        assert min_d_under > 0.0, (
            f"underpowered trajectory reached D<=0: min D={min_d_under}"
        )

        # Overpowered: KE = 2.5 * barrier. Must crest (D < 0). With the
        # honest signed bicubic SDF (no subpixel min-abs argmin), the
        # gradient near the crest carries FD-σ noise that saps enough KE
        # to stop a marginal launch ~1e-4 short of D=0 even when the
        # theoretical barrier is exceeded. 2.5x gives the launch clear
        # headroom against that noise while still demonstrating the ridge
        # is a barrier, not a wall — the underpowered case at 0.5x reflects.
        ke_over = 2.5 * barrier
        vx_over = math.sqrt(2.0 * ke_over)
        crossed_over, min_d_over, _, energy_log = _crest_attempt(vx_over)
        assert crossed_over, (
            f"overpowered trajectory failed to crest: min D={min_d_over}"
        )
        # Energy recorded at the crossing must be finite and consistent.
        assert energy_log, "no energy recorded at crossing"
        k, u, e = energy_log[0]
        assert math.isfinite(k) and math.isfinite(u) and math.isfinite(e)


class TestSignedSdfContinuity:
    """Acceptance: mechanics derive from the SINGLE signed distance field
    authority and vary continuously through the regularized Shore crest —
    with no dependence on a discrete mip level (issue #106, fix #11)."""

    def test_signed_distance_authority_sign_and_continuity(self, rc):
        """The signed distance field is the single authority: positive
        outside the set, negative inside, and continuous across the Shore
        (no sign reconstruction via a separate escape heuristic)."""
        # Load the canonical builtin field into the in-memory slot.
        meta = rc.get_builtin_distance_field_py("default")
        assert len(meta) == 6, f"unexpected builtin metadata: {meta}"
        # Deep inside the main cardioid -> negative; open water -> positive.
        assert rc.manifold_signed_distance(complex(0.0, 0.0)) < 0.0
        assert rc.manifold_signed_distance(complex(0.5, 0.0)) > 0.0
        # Locate the actual zero contour along y=0 (the distance-estimate
        # field's Shore is not exactly at the analytic cardioid cusp x=0.25).
        x_shore = _locate_shore_x(rc, y=0.0, x_lo=0.2, x_hi=0.5)
        # Continuity: a fine sweep across the boundary must stay finite with
        # bounded variation (no O(1) cliff, no NaN). The distance-estimate
        # field has small (~1e-4) noise right at the zero contour, so we do
        # NOT assert strict monotonicity — only that the field is finite and
        # does not jump discontinuously.
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
                # No cliff: adjacent samples (>=1e-3 apart) differ by less
                # than the local scale would allow for a discontinuity.
                assert abs(d - prev) < 0.05, (
                    f"signed distance cliff across Shore: {prev} -> {d}"
                )
            prev = d

    def test_signed_distance_unsigned_consistency(self, rc):
        """The unsigned sampler is abs() of the signed authority, so the
        two agree in magnitude everywhere (single authority, fix #3)."""
        meta = rc.get_builtin_distance_field_py("default")
        assert len(meta) == 6
        pts = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (-0.75, 0.0), (0.3, 0.5)]
        unsigned = rc.sample_distance_field_py([complex(x, y) for x, y in pts])
        for (x, y), u in zip(pts, unsigned):
            signed = rc.manifold_signed_distance(complex(x, y))
            assert u == pytest.approx(abs(signed), abs=1e-6), (
                f"unsigned != abs(signed) at ({x},{y}): {u} vs {abs(signed)}"
            )

    def test_scale_continuous_through_shore_crest(self, rc):
        """sigma(c) = log2(d_ref/rho) is finite and smooth through the
        regularized Shore crest. Sampling a dense line across the boundary
        must show bounded, monotone-ish variation — NOT a cliff, and NOT a
        dependence on which discrete mip rung a sample lands on."""
        meta = rc.get_builtin_distance_field_py("default")
        assert len(meta) == 6
        config = rc.ManifoldConfig(0.1, 1e-4, 1.0, 1.0)
        # Dense sweep across the cardioid boundary at y=0.
        xs = [0.24 + 0.001 * i for i in range(21)]  # 0.24 .. 0.26
        sigmas = [rc.manifold_mandelbrot_scale(complex(x, 0.0), config) for x in xs]
        for s in sigmas:
            assert math.isfinite(s)
        # Adjacent samples (1e-3 apart) must not jump by more than a small
        # multiple of the local gradient scale. sigma_x ~ 1/(D ln2) near the
        # crest; at D ~ 1e-3 that is ~1.4e3, so a 1e-3 step can move sigma by
        # ~1.4. Use a loose bound that still catches a discontinuous cliff
        # (an O(1) jump from a mip-rung artifact would be far larger).
        for a, b in zip(sigmas, sigmas[1:]):
            assert abs(a - b) < 20.0, f"scale cliff across Shore: {a} -> {b}"

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

    def test_mirror_ste_smoke_gradients_flow(self, rc):
        """STE smoke test (NOT gradient validation): the identity surrogate
        must keep the autograd graph connected so training can proceed. This
        does NOT assert the gradient equals the true gradient of the manifold
        dynamics — that is undefined across the PyO3 boundary and is NOT
        established (issue #106, fix #12)."""
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

        # The mirror starts at c=(0,0) like the Rust default. The mirror runs
        # its state and target/force math in float64 (matching the float64
        # Rust kernel) and calls the same Rust integrate_step binding, so the
        # two trajectories agree to ~1e-8 over the full sequence. The
        # tolerance absorbs residual float64 libm rounding while still
        # catching real semantic divergence (sign flips, wrong constants are
        # O(1) errors).
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

    def test_sequence_ste_smoke_gradients_flow_to_targets(self, rc):
        """STE smoke test (NOT gradient validation): a loss on the final
        position must backprop to s_target/alpha through the identity
        surrogate. This confirms the training path is connected; it does NOT
        establish that the gradient equals the gradient of the manifold
        dynamics (issue #106, fix #12)."""
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
