"""Offline comparison of sparse impulse gates on canonical-tick evidence fixtures."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise, product
import json
from random import Random
from typing import Literal

Mechanism = Literal["plain", "hysteresis", "lif", "alif"]
TICK_SECONDS = 1024 / 48000
MATCH_EARLY = 2
MATCH_LATE = 6
BURST_GAP = 8
ADAPTATION_DECAY = 0.92


@dataclass(frozen=True)
class Gate:
    mechanism: Mechanism
    threshold: float
    leak: float = 0.0
    refractory: int = 0
    release_ratio: float = 1.0
    adaptation_gain: float = 0.0


@dataclass
class GateState:
    voltage: float = 0.0
    adaptation: float = 0.0
    armed: bool = True
    cooldown: int = 0

    def step(self, gate: Gate, evidence: float) -> bool:
        if gate.mechanism == "plain":
            return evidence >= gate.threshold
        if gate.mechanism == "hysteresis":
            if evidence < gate.threshold * gate.release_ratio:
                self.armed = True
            fired = self.armed and self.cooldown == 0 and evidence >= gate.threshold
            if fired:
                self.armed = False
                self.cooldown = gate.refractory
            else:
                self.cooldown = max(0, self.cooldown - 1)
            return fired

        self.voltage = gate.leak * self.voltage + evidence
        threshold = gate.threshold + gate.adaptation_gain * self.adaptation
        fired = self.voltage >= threshold
        if fired:
            self.voltage = 0.0
        self.adaptation = (
            ADAPTATION_DECAY * self.adaptation + float(fired)
            if gate.mechanism == "alif"
            else 0.0
        )
        return fired


@dataclass(frozen=True)
class Fixture:
    name: str
    evidence: tuple[float, ...]
    targets: tuple[int, ...]


def fire_times(
    gate: Gate, evidence: Sequence[float], state: GateState | None = None
) -> tuple[int, ...]:
    if state is None:
        state = GateState()
    return tuple(tick for tick, value in enumerate(evidence) if state.step(gate, value))


def synthetic_suite(seed: int, repetitions: int) -> list[Fixture]:
    rng = Random(seed)
    fixtures: list[Fixture] = []

    def add(
        name: str, shape: list[float], targets: tuple[int, ...], noise: float
    ) -> None:
        onset = rng.randrange(20, 45)
        background = [rng.uniform(0.0, 0.10) for _ in range(112)]
        for offset, value in enumerate(shape):
            background[onset + offset] = max(
                0.0, min(1.0, value + rng.uniform(-noise, noise))
            )
        fixtures.append(
            Fixture(name, tuple(background), tuple(onset + t for t in targets))
        )

    for _ in range(repetitions):
        add("isolated", [rng.uniform(0.80, 1.0)], (0,), 0.04)
        add("weak_accumulation", [0.36, 0.05, 0.38, 0.05, 0.41, 0.05, 0.42], (4,), 0.05)
        add("slow_ramp", [0.07 * i for i in range(12)] + [0.0], (8,), 0.04)
        add("noisy_stable", [0.41 + 0.10 * (i % 2) for i in range(22)], (), 0.13)
        add("sustained", [0.90] * 24, (0,), 0.08)
        gap = rng.choice((2, 4, 6, 8, 12, 18))
        add("legitimate_repeat", [0.88] + [0.04] * (gap - 1) + [0.91], (0, gap), 0.04)
        add("stale_then_new", [0.33, 0.34] + [0.02] * 24 + [0.91], (26,), 0.04)
        add("alternating", [0.27, 0.83, 0.27, 0.84, 0.27, 0.82], (1, 5), 0.04)
    return fixtures


@dataclass(frozen=True)
class Score:
    target_count: int
    fire_count: int
    missed: int
    unmatched: int
    duplicate: int
    premature: int
    late: int
    signed_lag: int
    absolute_lag: int
    burst_histogram: Counter[int]
    minimum_retrigger: int | None


def score(fires: tuple[int, ...], targets: tuple[int, ...]) -> Score:
    # Ordered, earliest feasible one-to-one matches maximize match count under
    # equally sized timing windows and prevent one fire from satisfying both targets.
    matched: list[tuple[int, int]] = []
    unmatched: list[int] = []
    next_target = 0
    for tick in fires:
        while next_target < len(targets) and tick > targets[next_target] + MATCH_LATE:
            next_target += 1
        if next_target < len(targets) and tick >= targets[next_target] - MATCH_EARLY:
            matched.append((tick, targets[next_target]))
            next_target += 1
        else:
            unmatched.append(tick)

    duplicate, premature, late = 0, 0, 0
    for tick in unmatched:
        if any(
            matched_tick < tick <= matched_tick + BURST_GAP
            for matched_tick, _ in matched
        ):
            duplicate += 1
        elif targets:
            nearest = min(targets, key=lambda target: abs(tick - target))
            premature += -BURST_GAP <= tick - nearest < -MATCH_EARLY
            late += MATCH_LATE < tick - nearest <= 3 * BURST_GAP

    histogram: Counter[int] = Counter()
    if fires:
        length = 1
        for previous, tick in pairwise(fires):
            if tick - previous <= BURST_GAP:
                length += 1
            else:
                histogram[length] += 1
                length = 1
        histogram[length] += 1
    return Score(
        len(targets),
        len(fires),
        len(targets) - len(matched),
        len(unmatched),
        duplicate,
        premature,
        late,
        sum(tick - target for tick, target in matched),
        sum(abs(tick - target) for tick, target in matched),
        histogram,
        min((tick - previous for previous, tick in pairwise(fires)), default=None),
    )


def summarize(gate: Gate, fixtures: Iterable[Fixture]) -> dict[str, object]:
    fixtures = list(fixtures)
    scores = [
        score(fire_times(gate, fixture.evidence), fixture.targets)
        for fixture in fixtures
    ]
    targets = sum(s.target_count for s in scores)
    fired = sum(s.fire_count for s in scores)
    missed = sum(s.missed for s in scores)
    matches = targets - missed
    bursts: Counter[int] = Counter()
    for result in scores:
        bursts.update(result.burst_histogram)
    return {
        "targets": targets,
        "fires": fired,
        "missed": missed,
        "matched": matches,
        "unmatched": sum(s.unmatched for s in scores),
        "duplicates": sum(s.duplicate for s in scores),
        "premature": sum(s.premature for s in scores),
        "late": sum(s.late for s in scores),
        "mean_signed_lag_ticks": sum(s.signed_lag for s in scores) / matches
        if matches
        else None,
        "mean_absolute_lag_ticks": sum(s.absolute_lag for s in scores) / matches
        if matches
        else None,
        "minimum_retrigger_ticks": min(
            (s.minimum_retrigger for s in scores if s.minimum_retrigger is not None),
            default=None,
        ),
        "events_per_minute": fired
        * 60
        / (sum(len(f.evidence) for f in fixtures) * TICK_SECONDS),
        "burst_size_histogram": dict(sorted(bursts.items())),
    }


def calibration_loss(gate: Gate, fixtures: Iterable[Fixture]) -> tuple[int, int, int]:
    scores = [
        score(fire_times(gate, fixture.evidence), fixture.targets)
        for fixture in fixtures
    ]
    return (
        sum(s.missed + s.unmatched for s in scores),
        sum(s.duplicate for s in scores),
        sum(s.absolute_lag for s in scores),
    )


def calibration_target_matches(gate: Gate, fixtures: Iterable[Fixture]) -> int:
    scores = [
        score(fire_times(gate, fixture.evidence), fixture.targets)
        for fixture in fixtures
    ]
    return sum(s.target_count - s.missed for s in scores)


def semantic_checks() -> None:
    plain = Gate("plain", 0.5)
    schmitt = Gate("hysteresis", 0.5, refractory=2, release_ratio=0.5)
    lif = Gate("lif", 0.75, leak=0.8)
    alif = Gate("alif", 0.75, leak=0.8, adaptation_gain=0.2)
    assert fire_times(plain, (0.5, 0.5, 0.5)) == (0, 1, 2)
    assert fire_times(schmitt, (0.8, 0.8, 0.4, 0.8, 0.2, 0.8)) == (0, 5)
    assert fire_times(schmitt, (0.8, 0.0, 0.8, 0.8)) == (0, 3)
    assert fire_times(lif, (0.4, 0.45, 0.25, 0.55)) == (1, 3)
    assert fire_times(lif, (0.4, *([0.0] * 24), 0.4)) == ()
    assert fire_times(alif, (0.85, 0.86, 0.25)) == (0, 2)
    assert fire_times(alif, (0.85, 0.98, 0.0)) == (0, 1)
    assert fire_times(alif, (0.85, *([0.0] * 24), 0.86)) == (0, 25)
    prefix = (0.4, 0.45, 0.3)
    suffix = (0.65, 0.85, 0.1, 0.9, 0.99)
    for gate in (plain, schmitt, lif, alif):
        state = GateState()
        first = fire_times(gate, prefix, state)
        second = tuple(t + len(prefix) for t in fire_times(gate, suffix, state))
        assert first + second == fire_times(gate, prefix + suffix)
        future_flip = tuple(1.0 - value for value in suffix)
        assert first == tuple(
            t for t in fire_times(gate, prefix + future_flip) if t < len(prefix)
        )
        payload = (0.1, 0.7, 0.3, 1.0, 0.55, 0.2, 0.9, 0.4)
        fires = fire_times(gate, prefix + suffix)
        gated = tuple(value if t in fires else 0.0 for t, value in enumerate(payload))
        assert tuple(gated[t] for t in fires) == tuple(payload[t] for t in fires)
        assert all(value == 0.0 for t, value in enumerate(gated) if t not in fires)
    assert score((20, 24), (20, 24)).unmatched == 0
    assert score((20, 21), (20,)).duplicate == 1


def main() -> None:
    semantic_checks()
    calibration = synthetic_suite(seed=15201, repetitions=12)
    evaluation = synthetic_suite(seed=15202, repetitions=24)
    logit_thresholds = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85)
    accumulator_thresholds = (0.65, 0.75, 0.85, 1.05, 1.25, 1.50)
    leaks = (0.35, 0.60, 0.80, 0.93)
    candidates: dict[str, list[Gate]] = {
        "plain": [Gate("plain", threshold) for threshold in logit_thresholds],
        "hysteresis": [
            Gate("hysteresis", threshold, refractory=cooldown, release_ratio=ratio)
            for threshold, cooldown, ratio in product(
                logit_thresholds, (0, 2, 6, 12), (0.5, 0.8)
            )
        ],
        "lif": [
            Gate("lif", threshold, leak=leak)
            for threshold, leak in product(accumulator_thresholds, leaks)
        ],
        "alif": [
            Gate("alif", threshold, leak=leak, adaptation_gain=gain)
            for threshold, leak, gain in product(
                accumulator_thresholds, leaks, (0.12, 0.35, 0.70)
            )
        ],
    }
    hand_set = (
        Gate("plain", 0.60),
        Gate("hysteresis", 0.60, refractory=2, release_ratio=0.70),
        Gate("lif", 0.75, leak=0.80),
        Gate("alif", 0.75, leak=0.80, adaptation_gain=0.35),
    )
    names = sorted({f.name for f in evaluation})
    repeats = [f for f in evaluation if f.name == "legitimate_repeat"]
    repeat_gaps = sorted({f.targets[1] - f.targets[0] for f in repeats})
    report: dict[str, object] = {}
    for hand in hand_set:
        pool = candidates[hand.mechanism]
        selected = min(pool, key=lambda entry: calibration_loss(entry, calibration))
        active = [
            entry
            for entry in pool
            if calibration_target_matches(entry, calibration) > 0
        ]
        best_active = min(
            active, key=lambda entry: calibration_loss(entry, calibration)
        )
        report[hand.mechanism] = {
            "candidates": len(pool),
            "hand_parameters": asdict(hand),
            "selected_parameters": asdict(selected),
            "calibration_loss": calibration_loss(selected, calibration),
            "evaluation_hand_set": summarize(hand, evaluation),
            "evaluation_calibrated": summarize(selected, evaluation),
            "best_active_parameters": asdict(best_active),
            "evaluation_best_active": summarize(best_active, evaluation),
            "evaluation_by_fixture": {
                name: summarize(selected, (f for f in evaluation if f.name == name))
                for name in names
            },
            "evaluation_by_repeat_gap": {
                gap: summarize(
                    selected, (f for f in repeats if f.targets[1] - f.targets[0] == gap)
                )
                for gap in repeat_gaps
            },
        }
    print(
        json.dumps(
            {
                "semantic_checks": "passed",
                "tick_seconds": TICK_SECONDS,
                "calibration_count": len(calibration),
                "evaluation_count": len(evaluation),
                "mechanisms": report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
