"""Parity test for JuliaViewState: Rust/Python/WASM must produce identical state for identical deltas.

issue #107 requires deterministic shared action-to-view-state semantics to live in Rust
(ADR 0001) with browser and trainer consuming the same canonical contract. The previous
browser implementation used a JavaScript mirror of the Rust clamping/rates/harmony
logic; this test pins that the Rust authority and its Python/WASM projections agree
for a fixed initial state + fixed delta sequence (deterministic evolution).

The canonical semantics are in runtime-core/src/controls.rs::JuliaViewState::apply_controls:
- zoom = clamp(zoom * exp(zoom_delta * 0.05), 0.5, 8.0)
- rotation = wrap_angle(rotation + rotation_delta * 0.08)
- anchor_hue = wrap01(anchor_hue + hue_delta * 0.02)
- chroma = clamp(chroma + chroma_delta * 0.03, 0.0, 0.4)
- lightness = clamp(lightness + lightness_delta * 0.03, 0.2, 0.9)
- accent_weight = clamp(accent_weight + accent_delta * 0.04, 0.0, 1.0)
- harmony: edge-triggered with threshold 0.6, release 0.3, cooldown 15

This test exercises the Python projection (PyO3) and asserts it matches the
Rust-computed expected values for a short deterministic sequence. The WASM
projection is exercised by the sibling frontend test
frontend/src/lib/__tests__/juliaViewStateParity.test.ts which uses the same
initial state and delta sequence against the mock (which now mirrors Rust).
"""

from __future__ import annotations

import math

import pytest


def _wrap_angle(theta: float) -> float:
    tau = 2 * math.pi
    w = theta % tau
    if w > math.pi:
        w -= tau
    return w


def _wrap01(x: float) -> float:
    return x % 1.0


def _rust_expected_sequence():
    """Compute expected Rust-level evolution for the canonical fixture.

    This is a pure-Python re-derivation of the Rust logic for test-oracle
    purposes; the assertion is that the Rust binding produces these same
    values (parity), not that the Python re-derivation is authoritative.
    """
    zoom = 1.0
    rotation = 0.0
    anchor_hue = 0.0
    chroma = 0.18
    lightness = 0.55
    accent_weight = 0.35
    harmony = "analogous"
    harmony_cooldown = 0
    harmony_armed = True

    # Small deterministic delta sequence (5 ticks)
    deltas = [
        (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.7),  # first triggers harmony shift (analogous -> opponent)
        (0.2, -0.3, 0.1, -0.2, 0.1, -0.1, 0.0),
        (-0.4, 0.2, -0.5, 0.3, -0.3, 0.2, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),  # idle - cooldown still active if harmony triggered
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ]

    modes = ["monochrome", "analogous", "opponent"]
    for zoom_d, rot_d, hue_d, chroma_d, light_d, accent_d, harmony_shift in deltas:
        if harmony_cooldown > 0:
            harmony_cooldown -= 1
        # clamp11 on deltas (Rust does this via clamped())
        def c11(v: float) -> float:
            return max(-1.0, min(1.0, v))

        zoom_d = c11(zoom_d)
        rot_d = c11(rot_d)
        hue_d = c11(hue_d)
        chroma_d = c11(chroma_d)
        light_d = c11(light_d)
        accent_d = c11(accent_d)
        harmony_shift = c11(harmony_shift)

        zoom = max(0.5, min(8.0, zoom * math.exp(zoom_d * 0.05)))
        rotation = _wrap_angle(rotation + rot_d * 0.08)
        anchor_hue = _wrap01(anchor_hue + hue_d * 0.02)
        chroma = max(0.0, min(0.4, chroma + chroma_d * 0.03))
        lightness = max(0.2, min(0.9, lightness + light_d * 0.03))
        accent_weight = max(0.0, min(1.0, accent_weight + accent_d * 0.04))
        if abs(harmony_shift) < 0.3:
            harmony_armed = True
        if abs(harmony_shift) > 0.6 and harmony_armed and harmony_cooldown == 0:
            idx = modes.index(harmony)
            direction = 1 if harmony_shift > 0 else 2
            harmony = modes[(idx + direction) % 3]
            harmony_cooldown = 15
            harmony_armed = False

    return {
        "zoom": zoom,
        "rotation": rotation,
        "anchor_hue": anchor_hue,
        "chroma": chroma,
        "lightness": lightness,
        "accent_weight": accent_weight,
        "harmony": harmony,
        "harmony_cooldown": harmony_cooldown,
        "harmony_armed": harmony_armed,
    }


def test_julia_view_state_parity_python_matches_rust(runtime_core_module):  # type: ignore
    rc = runtime_core_module
    if not hasattr(rc, "JuliaViewState") or not hasattr(rc, "JuliaViewControls"):
        pytest.skip("runtime_core wheel lacks JuliaViewState bindings; rebuild")

    expected = _rust_expected_sequence()

    # Build initial state via Rust binding (Python projection)
    # ColorIntent(anchor_hue, chroma, lightness, harmony_str, accent_weight)
    color = rc.ColorIntent(0.0, 0.18, 0.55, "analogous", 0.35)
    state = rc.JuliaViewState(1.0, 0.0, color, 0, True)

    deltas = [
        (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.7),
        (0.2, -0.3, 0.1, -0.2, 0.1, -0.1, 0.0),
        (-0.4, 0.2, -0.5, 0.3, -0.3, 0.2, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ]
    for d in deltas:
        ctrl = rc.JuliaViewControls(*d)
        state.apply_controls(ctrl)

    # Read back through getters (Python projection)
    assert state.zoom == pytest.approx(expected["zoom"], rel=1e-12, abs=1e-12)
    assert state.rotation == pytest.approx(expected["rotation"], rel=1e-12, abs=1e-12)
    # Color is a ColorIntent; read its fields
    py_color = state.color
    # py_color may be a Python object with attributes anchor_hue etc or dict-like
    try:
        ah = float(py_color.anchor_hue)  # type: ignore[attr-defined]
        ch = float(py_color.chroma)  # type: ignore[attr-defined]
        li = float(py_color.lightness)  # type: ignore[attr-defined]
        aw = float(py_color.accent_weight)  # type: ignore[attr-defined]
        harm = str(py_color.harmony)  # type: ignore[attr-defined]
        # Harmony may be enum; normalize to string
        if hasattr(py_color, "harmony") and hasattr(py_color.harmony, "name"):
            # Enum case
            harm = py_color.harmony.name.lower()  # type: ignore
    except Exception:
        # Fallback dict-like
        ah = float(py_color["anchor_hue"])  # type: ignore[index]
        ch = float(py_color["chroma"])  # type: ignore[index]
        li = float(py_color["lightness"])  # type: ignore[index]
        aw = float(py_color["accent_weight"])  # type: ignore[index]
        harm = str(py_color["harmony"])  # type: ignore[index]

    assert ah == pytest.approx(expected["anchor_hue"], rel=1e-12, abs=1e-12)
    assert ch == pytest.approx(expected["chroma"], rel=1e-12, abs=1e-12)
    assert li == pytest.approx(expected["lightness"], rel=1e-12, abs=1e-12)
    assert aw == pytest.approx(expected["accent_weight"], rel=1e-12, abs=1e-12)
    # Harmony string compare (case-insensitive)
    assert harm.lower() == expected["harmony"]
    # Cooldown/armed are not directly observable via Python getters in some builds; check if present
    if hasattr(state, "harmony_cooldown"):
        assert int(state.harmony_cooldown) == expected["harmony_cooldown"]  # type: ignore[attr-defined]
    if hasattr(state, "harmony_armed"):
        assert bool(state.harmony_armed) == expected["harmony_armed"]  # type: ignore[attr-defined]


def test_julia_view_state_deterministic(runtime_core_module):  # type: ignore
    """Fixed initial state + fixed delta sequence yields deterministic evolution."""
    rc = runtime_core_module
    if not hasattr(rc, "JuliaViewState"):
        pytest.skip("no JuliaViewState")

    def run_once():
        color = rc.ColorIntent(0.1, 0.2, 0.6, "monochrome", 0.4)
        s = rc.JuliaViewState(2.0, 0.5, color, 0, True)
        for _ in range(3):
            s.apply_controls(rc.JuliaViewControls(0.3, -0.2, 0.4, 0.1, -0.1, 0.2, 0.8))
        return s

    a = run_once()
    b = run_once()
    assert a.zoom == pytest.approx(b.zoom, rel=0, abs=1e-12)
    assert a.rotation == pytest.approx(b.rotation, rel=0, abs=1e-12)
