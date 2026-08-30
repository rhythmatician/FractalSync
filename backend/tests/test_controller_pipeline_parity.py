"""Production-path controller pipeline parity (issue #93 regression test).

The #93 incident, physics side: the preflight controller parity check
advanced both Rust and the PyTorch mirror at a hardcoded ``1.0 / 60.0`` —
proving they agree at an obsolete timestep while the browser (post-#91)
supplies ``AnalysisTick.dt_seconds = 1024/48000``. Green test, divergent
production paths.

This test enforces the stronger invariant:

  same controls + same AnalysisTicks
       ├─ runtime controller (Rust OrbitController via PyO3 — the same
       │   code the browser drives through wasm)
       └─ differentiable training mirror (src.cspace_proxies)

  → same c trajectory

Rules enforced here (per the canonical-surfaces contract):
  - The timestep is derived from the deployed contract
    (``canonical_hop_dt()`` = HOP_LENGTH / SAMPLE_RATE), never restated.
  - The mirror's ``dt`` parameter is REQUIRED; this test would fail to
    compile/run against any mirror that reintroduces a ``dt=1/60`` default
    inviting drift.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import runtime_core
from runtime_core import HOP_LENGTH, SAMPLE_RATE

from src.cspace_proxies import (
    canonical_hop_dt,
    orbit_controller_sequence,
    orbit_controller_momentum_sequence,
)


DT = canonical_hop_dt()


class TestControllerPipelineParity:
    """Trainer mirror vs runtime controller at the CANONICAL timestep."""

    def test_canonical_dt_is_hop_duration(self):
        """The contract-derived dt must be the canonical hop duration —
        this is the value the browser supplies via AnalysisTick.dt_seconds."""
        assert DT == pytest.approx(HOP_LENGTH / SAMPLE_RATE, abs=1e-15)
        assert DT == pytest.approx(1024 / 48000, abs=1e-15)

    def test_flags_off_mirror_matches_rust_at_canonical_dt(self):
        rng = np.random.RandomState(0)
        n_steps = 120
        s_vals = torch.tensor(
            np.clip(rng.uniform(0.2, 3.0, n_steps), 0.2, 3.0), dtype=torch.float32
        )
        a_vals = torch.tensor(rng.uniform(0.0, 1.0, n_steps), dtype=torch.float32)
        gates = torch.tensor(rng.uniform(0.0, 1.0, (n_steps, 6)), dtype=torch.float32)
        seg = torch.zeros(n_steps, dtype=torch.int64)

        c = runtime_core.OrbitController(float(s_vals[0]), float(a_vals[0]), 1.0)
        rust_re = rust_im = 0.0
        for i in range(n_steps):
            c.apply_controls(float(s_vals[i]), float(a_vals[i]))
            rust_re, rust_im = c.step(DT, gates[i].tolist())

        pt_c = orbit_controller_sequence(
            s_target=s_vals,
            alpha=a_vals,
            omega=1.0,
            band_gates=gates,
            segment_ids=seg,
            dt=DT,
        )
        err = max(
            abs(rust_re - pt_c[-1].real.item()),
            abs(rust_im - pt_c[-1].imag.item()),
        )
        assert err < 1e-5, (
            f"OrbitController mirror diverged from Rust at canonical dt: "
            f"err={err} (trainer would supervise wrong physics)"
        )

    def test_momentum_mirror_matches_rust_at_canonical_dt(self):
        rng = np.random.RandomState(7)
        n_steps = 120
        s_vals = torch.tensor(
            np.clip(rng.uniform(0.2, 3.0, n_steps), 0.2, 3.0), dtype=torch.float32
        )
        a_vals = torch.tensor(rng.uniform(0.0, 1.0, n_steps), dtype=torch.float32)
        gates = torch.tensor(rng.uniform(0.0, 1.0, (n_steps, 6)), dtype=torch.float32)
        seg = torch.zeros(n_steps, dtype=torch.int64)

        cm = runtime_core.OrbitController(float(s_vals[0]), float(a_vals[0]), 1.0)
        cm.set_momentum(True)
        cm.set_drag(0.90)
        rust_re = rust_im = 0.0
        for i in range(n_steps):
            cm.apply_controls(float(s_vals[i]), float(a_vals[i]))
            rust_re, rust_im = cm.step(DT, gates[i].tolist())

        pt_cm = orbit_controller_momentum_sequence(
            s_target=s_vals,
            alpha=a_vals,
            omega=1.0,
            band_gates=gates,
            segment_ids=seg,
            dt=DT,
            drag=0.90,
        )
        err = max(
            abs(rust_re - pt_cm[-1].real.item()),
            abs(rust_im - pt_cm[-1].imag.item()),
        )
        assert err < 1e-5, (
            f"momentum mirror diverged from Rust at canonical dt: err={err}"
        )

    def test_mirror_dt_is_required_no_default(self):
        """The mirror must not carry a default dt: a function with
        ``dt=1/60`` sitting in a training mirror is an invitation to
        exactly the drift that got through green in #93."""
        import inspect

        for fn in (orbit_controller_sequence, orbit_controller_momentum_sequence):
            sig = inspect.signature(fn)
            assert "dt" in sig.parameters, f"{fn.__name__} lost its dt parameter"
            assert sig.parameters["dt"].default is inspect.Parameter.empty, (
                f"{fn.__name__} has a default dt — reintroducing a hardcoded "
                "timestep in a training mirror is structurally wrong"
            )

    def test_canonical_dt_differs_from_legacy_frame_rate(self):
        """Guard the incident itself: the canonical dt must NOT be 1/60.
        If the hop contract ever changes to exactly 60 Hz, this test must
        be consciously updated alongside the ADR — not silently."""
        assert abs(DT - 1.0 / 60.0) > 1e-3, (
            "canonical dt collapsed to the legacy 1/60 frame step; the "
            "sample-clock timebase contract has changed — review ADR 0001 "
            "and the parity surfaces manifest"
        )
