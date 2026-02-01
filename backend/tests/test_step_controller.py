"""Integration tests for the new step-based controller via runtime_core bindings."""

from __future__ import annotations

import runtime_core as rc


def test_step_controller_step_and_context():
    stc = rc.StepController()
    state = rc.StepState()

    # Initial state
    assert isinstance(state.c_real, float)
    assert isinstance(state.c_imag, float)

    # Small model step
    res = stc.step(state, 0.01, 0.0)

    # Result should have expected attributes
    assert hasattr(res, "c_real")
    assert hasattr(res, "c_imag")
    assert hasattr(res, "delta_real")
    assert hasattr(res, "delta_imag")
    assert hasattr(res, "context")

    # Context may be exposed at the result or via context_for_state; prefer
    # the explicit API if the StepResult wrapper does not carry it.
    ctx = res.context
    if ctx is None:
        ctx = stc.context_for_state(state)

    # Accept either a `feature_vec` attribute/dict entry or an `as_feature_vec()` method
    if hasattr(ctx, "feature_vec"):
        feature_vec = ctx.feature_vec
    elif isinstance(ctx, dict) and "feature_vec" in ctx:
        feature_vec = ctx["feature_vec"]
    elif hasattr(ctx, "as_feature_vec"):
        feature_vec = ctx.as_feature_vec()
    else:
        raise AssertionError("StepContext lacks feature_vec or as_feature_vec()")

    assert isinstance(feature_vec, (list, tuple))
    assert len(feature_vec) >= 16

    # If we call context_for_state() directly it should return a StepContext
    ctx2 = stc.context_for_state(state)
    assert hasattr(ctx2, "as_feature_vec") or hasattr(ctx2, "feature_vec")
