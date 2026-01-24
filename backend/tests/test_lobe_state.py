from src.lobe_state import LobeState


# Simple sanity tests for the LobeState FSM


def test_basic_switch_and_cooldown():
    ls = LobeState(current_lobe=0, n_lobes=3)
    # Incoming scores favor lobe 1
    scores = [0.1, 2.0, 0.0]
    # Step should start a transition
    ls.step(scores, dt=0.1)
    assert ls.target_lobe == 1
    # Finish transition by stepping enough time
    ls.step(scores, dt=ls.transition_time)
    assert ls.current_lobe == 1
    # Immediately attempt to switch back should be blocked by cooldown
    scores2 = [5.0, 0.1, 0.0]
    ls.step(scores2, dt=0.1)
    assert ls.target_lobe is None


def test_transient_fast_switch():
    ls = LobeState(
        current_lobe=0, n_lobes=2, transition_time=1.0, transient_threshold=0.5
    )
    scores = [0.0, 3.0]
    # Provide a high transient to accelerate
    ls.step(scores, dt=0.1, transient=1.0)
    # Transition should be in progress
    assert ls.target_lobe == 1
    # High transient reduces effective time by factor -> finish faster
    ls.step(scores, dt=0.25, transient=1.0)
    assert ls.current_lobe == 1
