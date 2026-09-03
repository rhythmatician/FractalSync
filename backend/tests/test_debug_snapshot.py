"""Tests for the read-only DebugSnapshot seam (issue #111 Phase A).

The Rust binding is the canonical source of truth (ADR 0001). These tests
exercise the PyO3 surface directly: snapshot creation must be read-only,
physics fields must match the canonical manifold bindings, and the terrain
patch must be the authoritative Q=(x, y, lambda*sigma) embedding.

Run: python -m pytest backend/tests/test_debug_snapshot.py -q
"""

from __future__ import annotations

import math

import pytest

CANONICAL_DT = 1024.0 / 48000.0


@pytest.fixture(scope="module")
def rc(runtime_core_module):  # noqa: F811 - provided by backend/conftest.py
    """The real compiled runtime_core extension."""
    mod = runtime_core_module
    if not hasattr(mod, "debug_snapshot_from_state"):
        pytest.skip("runtime_core wheel lacks debug bindings; rebuild")
    return mod


def _config(rc):
    return rc.ManifoldConfig(0.1, 1e-4, 1.0, 1.0)


class TestDebugSnapshotFromState:
    def test_snapshot_has_versioned_sections(self, rc):
        snap = rc.debug_snapshot_from_state(0.0, 0.0, 0.0, 0.0)
        assert snap["version"] == "debug-snapshot/1"
        for section in ("timeSeconds", "map", "physics", "diagnostics"):
            assert section in snap
        # observation is deliberately absent until #108 (Phase B).
        assert "observation" not in snap

    def test_physics_fields_match_canonical_bindings(self, rc):
        c = complex(0.0, 0.0)
        v = (0.05, -0.02)
        snap = rc.debug_snapshot_from_state(c.real, c.imag, v[0], v[1])
        p = snap["physics"]

        assert p["c"] == [c.real, c.imag]
        assert p["velocity"] == [v[0], v[1]]
        assert p["signedDistance"] == pytest.approx(
            rc.manifold_signed_distance(c), abs=1e-14
        )
        assert p["sigma"] == pytest.approx(
            rc.manifold_mandelbrot_scale(c, _config(rc)), abs=1e-14
        )
        assert p["kinetic"] == pytest.approx(
            rc.manifold_kinetic_energy(v[0], v[1], c, _config(rc)),
            abs=1e-14,
        )
        assert p["potential"] == pytest.approx(
            rc.manifold_potential_energy(c, _config(rc)), abs=1e-14
        )
        assert p["total"] == pytest.approx(p["kinetic"] + p["potential"], abs=1e-12)

    def test_realm_follows_signed_distance(self, rc):
        inside = rc.debug_snapshot_from_state(0.0, 0.0, 0.0, 0.0)
        assert inside["physics"]["realm"] == -1
        outside = rc.debug_snapshot_from_state(1.0, 1.0, 0.0, 0.0)
        assert outside["physics"]["realm"] == 1

    def test_action_none_before_first_step(self, rc):
        snap = rc.debug_snapshot_from_state(0.0, 0.0, 0.0, 0.0)
        assert snap["action"] is None
        assert snap["diagnostics"]["lastDeltaTotal"] is None

    def test_raw_vs_effective_provenance(self, rc):
        raw = rc.MotionControls(2.0, 0.0, 2.0, -3.0, 5.0, 0.0)
        snap = rc.debug_snapshot_from_state(0.0, 0.0, 0.0, 0.0, motion_raw=raw)
        action = snap["action"]
        assert action["raw"]["throttle"] == pytest.approx(2.0)
        assert action["effective"]["throttle"] == pytest.approx(1.0)
        assert action["effective"]["brake"] == pytest.approx(0.0)
        assert action["effective"]["grip"] == pytest.approx(1.0)

    def test_friction_power_nonpositive_through_controller(self, rc):
        """Friction power is computed by the controller's destination step;
        verify it is non-positive via the real controller path (not the
        standalone builder, whose frictionPower is a caller-supplied pass-
        through)."""
        if not hasattr(rc, "OrbitController"):
            pytest.skip("OrbitController binding unavailable")
        ctrl = rc.OrbitController(1.0, 0.0, 1.0)
        ctrl.set_manifold_physics(True)
        raw = rc.MotionControls(1.0, 0.0, 1.0, 0.0, 0.5, 0.0)
        ctrl.step_with_controls(CANONICAL_DT, raw)
        snap = ctrl.debug_snapshot()
        assert snap["action"] is not None
        assert snap["action"]["frictionPower"] <= 1e-12

    def test_time_seconds_passthrough(self, rc):
        snap = rc.debug_snapshot_from_state(
            0.0, 0.0, 0.0, 0.0, time_seconds=7 * CANONICAL_DT
        )
        assert snap["timeSeconds"] == pytest.approx(7 * CANONICAL_DT)

    def test_last_delta_total_passthrough(self, rc):
        snap = rc.debug_snapshot_from_state(0.0, 0.0, 0.0, 0.0, last_delta_total=0.125)
        assert snap["diagnostics"]["lastDeltaTotal"] == pytest.approx(0.125)

    def test_map_unavailable_without_pyramid(self, rc):
        rc.clear_pyramid_py()
        snap = rc.debug_snapshot_from_state(0.0, 0.0, 0.0, 0.0)
        assert snap["map"]["pyramidLoaded"] is False
        assert snap["map"]["shoreProximity"] is None
        assert snap["map"]["minimapWindow"] is None


class TestDebugTerrainPatch:
    def test_patch_is_authoritative_embedding(self, rc):
        patch = rc.debug_terrain_patch(0.0, 0.0, 0.5, 33)
        assert patch["n"] == 33
        assert len(patch["positions"]) == 33 * 33 * 3
        assert len(patch["signed"]) == 33 * 33
        assert len(patch["realm"]) == 33 * 33

        # Center vertex is the embedding of the patch center.
        mid = (33 * 33 // 2) * 3
        assert patch["positions"][mid] == pytest.approx(0.0, abs=1e-14)
        assert patch["positions"][mid + 1] == pytest.approx(0.0, abs=1e-14)
        sigma = rc.manifold_mandelbrot_scale(complex(0.0, 0.0), _config(rc))
        assert patch["positions"][mid + 2] == pytest.approx(sigma, abs=1e-14)

        # Signed distances match the canonical authority.
        d = rc.manifold_signed_distance(complex(0.0, 0.0))
        assert patch["signed"][33 * 33 // 2] == pytest.approx(d, abs=1e-14)

    def test_realm_signs_follow_signed_distance(self, rc):
        patch = rc.debug_terrain_patch(0.0, 0.0, 0.5, 17)
        for d, realm in zip(patch["signed"], patch["realm"]):
            expected = -1 if d < 0 else (1 if d > 0 else 0)
            assert realm == expected

    def test_row0_is_north_edge(self, rc):
        patch = rc.debug_terrain_patch(0.0, 0.0, 0.5, 9)
        # First vertex: (re, im) = (-0.5, +0.5).
        assert patch["positions"][0] == pytest.approx(-0.5, abs=1e-14)
        assert patch["positions"][1] == pytest.approx(0.5, abs=1e-14)

    def test_rejects_degenerate_grids(self, rc):
        with pytest.raises(Exception):
            rc.debug_terrain_patch(0.0, 0.0, 0.5, 1)
        with pytest.raises(Exception):
            rc.debug_terrain_patch(0.0, 0.0, 0.5, 1000)
        with pytest.raises(Exception):
            rc.debug_terrain_patch(0.0, 0.0, -1.0, 33)

    def test_sigma_heights_vary_across_patch(self, rc):
        """The surface is not flat: sigma varies across the patch, so the
        3D skate park has real relief."""
        patch = rc.debug_terrain_patch(0.0, 0.0, 0.5, 33)
        heights = patch["positions"][2::3]
        assert max(heights) - min(heights) > 1e-3


class TestControllerSnapshotParity:
    def test_controller_snapshot_matches_standalone(self, rc):
        """The OrbitController method and the standalone builder must agree
        on every state-derived field."""
        if not hasattr(rc, "OrbitController"):
            pytest.skip("OrbitController binding unavailable")
        ctrl = rc.OrbitController(1.0, 0.0, 1.0)
        ctrl.set_manifold_physics(True)
        raw = rc.MotionControls(1.0, 0.0, 0.7, 0.0, 0.5, 0.0)
        ctrl.step_with_controls(CANONICAL_DT, raw)

        from_ctrl = ctrl.debug_snapshot()
        standalone = rc.debug_snapshot_from_state(
            ctrl.c_re,
            ctrl.c_im,
            ctrl.planar_velocity[0],
            ctrl.planar_velocity[1],
            motion_raw=raw,
            friction_beta=0.05 + 0.5 * 0.15,
            friction_power=0.0,
            manifold_drag=ctrl.manifold_drag,
            config=ctrl.manifold_config,
            last_delta_total=from_ctrl["diagnostics"]["lastDeltaTotal"],
            time_seconds=from_ctrl["timeSeconds"],
        )
        assert from_ctrl["physics"] == standalone["physics"]
        assert from_ctrl["action"]["effective"] == standalone["action"]["effective"]
        assert from_ctrl["action"]["driveCovector"] == pytest.approx(
            standalone["action"]["driveCovector"], abs=1e-12
        )

    def test_snapshot_is_read_only(self, rc):
        if not hasattr(rc, "OrbitController"):
            pytest.skip("OrbitController binding unavailable")
        ctrl = rc.OrbitController(1.0, 0.0, 1.0)
        ctrl.set_manifold_physics(True)
        raw = rc.MotionControls(1.0, 0.0, 0.5, 0.0, 0.5, 0.0)
        ctrl.step_with_controls(CANONICAL_DT, raw)

        c_before = (ctrl.c_re, ctrl.c_im)
        v_before = ctrl.planar_velocity
        s1 = ctrl.debug_snapshot()
        s2 = ctrl.debug_snapshot()
        assert s1 == s2
        assert (ctrl.c_re, ctrl.c_im) == c_before
        assert ctrl.planar_velocity == v_before

    def test_time_keyed_to_destination_steps(self, rc):
        if not hasattr(rc, "OrbitController"):
            pytest.skip("OrbitController binding unavailable")
        ctrl = rc.OrbitController(1.0, 0.0, 1.0)
        ctrl.set_manifold_physics(True)
        raw = rc.MotionControls(1.0, 0.0, 0.3, 0.0, 0.5, 0.0)
        for _ in range(7):
            ctrl.step_with_controls(CANONICAL_DT, raw)
        snap = ctrl.debug_snapshot()
        assert snap["timeSeconds"] == pytest.approx(7 * CANONICAL_DT, rel=1e-12)


class TestWireFormat:
    def test_camel_case_keys(self, rc):
        snap = rc.debug_snapshot_from_state(0.0, 0.0, 0.0, 0.0)
        assert "timeSeconds" in snap
        assert "signedDistance" in snap["physics"]
        assert "pyramidLoaded" in snap["map"]
        assert "derivativeStep" in snap["diagnostics"]

    def test_crest_potential_is_rust_owned(self, rc):
        """The crest ceiling comes from the config, not a restated constant."""
        snap = rc.debug_snapshot_from_state(0.0, 0.0, 0.0, 0.0)
        expected = 1.0 * math.log2(0.1 / 1e-4)  # kappa * log2(d_ref/epsilon)
        assert snap["diagnostics"]["crestPotential"] == pytest.approx(
            expected, rel=1e-12
        )

    def test_terrain_patch_keys(self, rc):
        patch = rc.debug_terrain_patch(0.0, 0.0, 0.5, 9)
        assert set(patch.keys()) == {
            "n",
            "center",
            "half",
            "positions",
            "signed",
            "realm",
        }
