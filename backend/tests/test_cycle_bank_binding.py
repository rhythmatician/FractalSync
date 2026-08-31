"""PyO3 surface smoke test for the canonical observed-ridge CycleBank (#92).

This proves the SAME Rust CycleBank is reachable from the training/offline
surface (PyO3).  Python only orchestrates: it synthesizes a scalar evidence
stream, feeds explicit observations through the binding, and reads observed
modes / relations / predictions back.  No transform, ridge, frequency, phase,
confidence, relation, or prediction math is done here — all of it is Rust.

The production path is ``CycleBank.observe_tick(canonical_tick)`` (fed by
``AnalysisTimebase.ingest``); ``observe`` exists for synthetic diagnostics.
"""

from __future__ import annotations

import math

import pytest

runtime_core = pytest.importorskip("runtime_core")

TWO_PI = 2.0 * math.pi
DT = runtime_core.HOP_LENGTH / runtime_core.SAMPLE_RATE


def _feed_sinusoid(
    bank: "runtime_core.CycleBank",
    frequency_hz: float,
    seconds: float,
    amplitude: float = 0.8,
    phase0: float = 0.3,
) -> None:
    n = int(round(seconds / DT))
    for k in range(1, n + 1):
        t = k * DT
        value = amplitude * math.cos(TWO_PI * frequency_hz * t + phase0)
        bank.observe(k * runtime_core.HOP_LENGTH, DT, 1, [("mono", value)])


def test_cycle_bank_version_is_rust_owned_and_exposed():
    assert isinstance(runtime_core.CYCLE_BANK_VERSION, str)
    assert runtime_core.CYCLE_BANK_VERSION.startswith("cycle-bank/")
    bank = runtime_core.CycleBank()
    assert bank.version == runtime_core.CYCLE_BANK_VERSION


def test_cycle_bank_observes_off_grid_ridge():
    """A 2.1667 Hz sinusoid is recovered near 2.1667 Hz, not snapped to a
    numerical scale center (the architecture's central acceptance property)."""
    bank = runtime_core.CycleBank(
        {
            "f_min_hz": 0.5,
            "f_max_hz": 4.0,
            "birth_persistence": 2,
        }
    )
    target = 2.1667
    _feed_sinusoid(bank, target, seconds=14.0)

    modes = bank.modes()
    assert modes, "no observed modes reported"
    closest = min(modes, key=lambda m: abs(m.frequency_hz - target))
    assert abs(closest.frequency_hz - target) < 0.05, (
        f"recovered {closest.frequency_hz} Hz, expected near {target} Hz"
    )
    # Causal free-running prediction is a method on the emitted mode.
    predicted = closest.phase_at(DT)
    assert -math.pi <= predicted <= math.pi
    # time_to_next is consistent with phase_at for a quarter-period lead.
    reference = closest.phase + math.pi / 2.0
    lead = closest.time_to_next(reference)
    assert lead is not None and lead > 0.0

    # The camelCase wire dict matches the browser's CycleMode interface.
    wire = closest.to_dict()
    for key in (
        "id",
        "frequencyHz",
        "phase",
        "strength",
        "confidence",
        "channelSupport",
        "age",
        "missingObservations",
    ):
        assert key in wire, f"mode wire dict missing {key}"


def test_cycle_bank_tick_seam_round_trip():
    """observe_tick consumes the canonical AnalysisTimebase tick shape and the
    newest-frame extraction happens in Rust, not in Python."""
    tb = runtime_core.AnalysisTimebase()
    bank = runtime_core.CycleBank({"f_min_hz": 0.5, "f_max_hz": 4.0})

    # Feed a real 2 Hz tone through the canonical timebase so ticks are the
    # genuine production shape.
    sr = runtime_core.SAMPLE_RATE
    seconds = 12.0
    total = int(sr * seconds)
    pcm = [
        0.6 * math.sin(TWO_PI * 220.0 * i / sr)  # audible tone
        for i in range(total)
    ]
    # Chunked ingestion (arbitrary block size) to exercise the seam.
    block = 8192
    modes = []
    for start in range(0, total, block):
        chunk = pcm[start : start + block]
        ticks = tb.ingest(chunk, sr, start)
        for tick in ticks:
            modes = bank.observe_tick(tick)

    assert modes, "tick-driven CycleBank produced no modes"
    wire = [m.to_dict() for m in modes]
    assert all(
        set(w)
        >= {
            "id",
            "frequencyHz",
            "phase",
            "strength",
            "confidence",
            "channelSupport",
            "age",
            "missingObservations",
        }
        for w in wire
    ), "mode wire dict missing canonical keys"


def test_cycle_bank_epoch_reset_clears_modes():
    bank = runtime_core.CycleBank({"f_min_hz": 0.5, "f_max_hz": 4.0, "birth_persistence": 2})
    _feed_sinusoid(bank, 2.0, seconds=10.0)
    assert bank.num_modes() > 0

    # A stream-epoch change resets temporal analysis deterministically.
    bank.observe(runtime_core.HOP_LENGTH, DT, 2, [("mono", 0.0)])
    assert bank.num_modes() == 0
