import pytest

from src import runtime_core_bridge as rcb
from src.lobe_state import LobeState as PyLobeState


def run_sequence(state, seq):
    # seq: list of (scores, dt, transient)
    out = []
    for scores, dt, transient in seq:
        state.step(scores, dt=dt, transient=transient)
        out.append((state.current_lobe, state.target_lobe, state.transition_progress))
    return out


def test_lobe_state_parity():
    # Build a deterministic sample sequence
    seq = [
        ([0.1, 2.0], 0.1, 0.0),
        ([0.1, 2.0], 1.0, 0.0),
        ([5.0, 0.1], 0.1, 0.0),
        ([0.0, 3.0], 0.25, 1.0),
    ]

    # Python implementation
    py_ls = PyLobeState(current_lobe=0, n_lobes=2)
    py_out = run_sequence(py_ls, seq)

    # Rust-backed via factory
    rc_ls = rcb.make_lobe_state(n_lobes=2)
    rc_out = run_sequence(rc_ls, seq)

    assert py_out == rc_out
