import pytest

from src import runtime_core_bridge as rcb


def test_transition_completion_and_cooldown_behavior():
    """Verify that a Rust LobeState transitions when driven with strong logits,
    completes the transition, and prevents immediate re-transition (cooldown).

    This test does NOT reimplement legacy logic; it exercises externally
    observable behavior of the Rust FSM only.
    """

    ls = rcb.make_lobe_state(n_lobes=2)
    assert ls.current_lobe == 0

    # Drive logits strongly toward lobe 1; a target should be set and progress > 0
    ls.step([0.0, 10.0], dt=0.5, transient=0.0)
    assert ls.target_lobe is not None
    assert ls.transition_progress > 0.0

    # Continue stepping until transition completes and current_lobe updates
    # (protect against infinite loops by bounding iterations)
    for _ in range(10):
        if ls.target_lobe is None:
            break
        ls.step([0.0, 10.0], dt=0.5, transient=0.0)

    assert ls.target_lobe is None
    assert ls.current_lobe == 1

    # Immediately attempt to flip back; cooldown should prevent an immediate transition
    ls.step([10.0, 0.0], dt=1.0, transient=0.0)
    assert ls.current_lobe == 1
    # If a transition is prevented, we expect either no target or zero progress
    assert ls.target_lobe is None or ls.transition_progress == pytest.approx(0.0)


def test_transient_accelerates_progress():
    """Transient signals should accelerate transition progress (observable).

    We create two fresh states and apply identical logits with different transient
    magnitudes; the state receiving a higher transient should make more progress
    in the same dt.
    """

    strong_scores = [0.0, 10.0]

    ls_no_trans = rcb.make_lobe_state(n_lobes=2)
    ls_no_trans.step(strong_scores, dt=0.1, transient=0.0)
    p_no = ls_no_trans.transition_progress

    ls_high_trans = rcb.make_lobe_state(n_lobes=2)
    ls_high_trans.step(strong_scores, dt=0.1, transient=1.0)
    p_high = ls_high_trans.transition_progress

    assert p_high > p_no
