"""Deterministic lobe switching finite-state machine.

LobeState receives continuous per-lobe scores (e.g. softmaxed logits) and
converts them to deterministic discrete lobe switches with hysteresis and
cooldown. This is suitable for runtime usage and for unit testing before
mirroring the logic in runtime-core (Rust).

The class is intentionally small and pure-Python to ease unit testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import math


@dataclass
class LobeState:
    current_lobe: int = 0
    target_lobe: Optional[int] = None
    transition_progress: float = 0.0
    cooldown_timer: float = 0.0
    hold_timer: float = 0.0

    # Tunables
    transition_time: float = 1.0
    cooldown: float = 2.0
    min_hold: float = 1.0
    threshold_on: float = 0.6
    threshold_off: float = 0.4
    transient_threshold: float = 0.6

    # internal
    n_lobes: int = field(default=2)

    def step(
        self, scores: List[float], dt: float = 1.0, transient: float = 0.0
    ) -> None:
        """Advance the FSM by dt seconds given per-lobe scores.

        scores: raw scores (not necessarily normalized). We normalize with softmax
        internally so callers may pass logits or probabilities.
        """
        # Update timers
        if self.cooldown_timer > 0.0:
            self.cooldown_timer = max(0.0, self.cooldown_timer - dt)
        if self.hold_timer > 0.0:
            self.hold_timer = max(0.0, self.hold_timer - dt)

        # Normalize scores with softmax for stability
        exps = [math.exp(float(s)) for s in scores]
        ssum = sum(exps)
        if ssum == 0.0:
            probs = [1.0 / len(scores)] * len(scores)
        else:
            probs = [e / ssum for e in exps]

        # Candidate lobe
        cand = int(max(range(len(probs)), key=lambda i: probs[i]))
        cand_score = probs[cand]

        # If in transition, advance
        if self.target_lobe is not None and self.target_lobe != self.current_lobe:
            # Progress faster if transient large
            eff_time = self.transition_time * (
                0.25 if transient >= self.transient_threshold else 1.0
            )
            self.transition_progress += dt / max(1e-6, eff_time)
            if self.transition_progress >= 1.0:
                # Finish transition
                self.current_lobe = self.target_lobe
                self.target_lobe = None
                self.transition_progress = 0.0
                self.cooldown_timer = self.cooldown
                self.hold_timer = self.min_hold
            return

        # Not in transition: consider starting one
        if (
            cand != self.current_lobe
            and self.cooldown_timer <= 0.0
            and self.hold_timer <= 0.0
        ):
            if cand_score >= self.threshold_on:
                # Start transition and immediately progress using available dt
                self.target_lobe = cand
                self.transition_progress = 0.0
                eff_time = self.transition_time * (
                    0.25 if transient >= self.transient_threshold else 1.0
                )
                self.transition_progress += dt / max(1e-6, eff_time)
                if self.transition_progress >= 1.0:
                    # Finish transition immediately
                    self.current_lobe = self.target_lobe
                    self.target_lobe = None
                    self.transition_progress = 0.0
                    self.cooldown_timer = self.cooldown
                    self.hold_timer = self.min_hold
                return

        # Optionally allow falling back if candidate score drops below off threshold
        # (no-op here; hysteresis enforced by thresholds and timers)
        return

    def get_mix(self) -> float:
        """Return mix factor (0..1) indicating progress toward target lobe."""
        return float(self.transition_progress)
