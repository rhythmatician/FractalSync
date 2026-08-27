# 0001 — Rust-first parity for runtime-consumed math

Status: Accepted (2026-08-27, authorized by repository owner)

## Context

All orbit-synthesis, geometry, feature-extraction, minimap, and visual-metric math
consumed at runtime — in the browser via `wasm-orbit` and in the backend via the
`runtime_core` Python bindings — is implemented in the Rust crate `runtime-core`.
Other implementations of that same math exist as derived mirrors: the differentiable
PyTorch mirrors in `backend/src/cspace_proxies.py`, TypeScript mocks such as
`frontend/src/lib/__tests__/orbitSynthesizer.mock.ts`, and formulas restated in
documentation.

A training session (~90 minutes) was wasted because a PyTorch mirror and the Rust
runtime had diverged. The divergence was invisible until runtime evaluation of the
trained model. Nothing mechanically prevented training from starting against a
diverged mirror.

## Decision

`runtime-core` is the single source of truth for all runtime-consumed synthesis,
geometry, feature-extraction, minimap, and visual-metric math.

- **Forbidden:** implementing runtime-consumed synthesis/geometry/feature math
  outside `runtime-core`, except as an explicitly maintained mirror.
- **Required:** every mirror must be kept in behavioral parity with the Rust
  implementation, verified by automated parity tests. A training run must not begin
  unless the training-time mirrors pass a mechanical parity preflight against the
  installed `runtime_core`. The gate is `scripts/preflight_parity.py`, wired into
  `backend/train.py`; training aborts automatically if mirrors diverge.
- **Exceptions:** none currently.

## Enforcement: golden vectors

Parity is enforced through a single shared data artifact, not by inspection:

1. `runtime-core/src/bin/generate_golden_vectors.rs` records deterministic
   input/output pairs of the canonical math (carrier geometry, full orbit
   synthesis with residuals and gates, PlayerState momentum trajectories,
   cardioid proximity) into `shared/golden_vectors.json`.
2. Every mirror has a golden parity test that replays those vectors:
   - Backend: `backend/tests/test_golden_parity.py` (PyTorch mirror + Python
     bindings).
   - Frontend: `frontend/src/lib/__tests__/goldenParity.test.ts` (TS mock).
3. Changing canonical Rust math without regenerating the file and updating all
   mirrors in the same commit fails these tests immediately.

Workflow for a canonical-math change:
```
cargo run --release -p runtime_core --bin generate_golden_vectors   # regenerate
# update mirrors in the same commit
python -m pytest backend/tests/test_golden_parity.py -q             # backend proof
npx vitest run src/lib/__tests__/goldenParity.test.ts               # frontend proof (in frontend/)
```

## Consequences

- Any change to mirror code must be accompanied by updates to the corresponding
  parity tests.
- Any change to canonical Rust math requires regenerating
  `shared/golden_vectors.json` and updating all mirrors in the same commit;
  divergence is caught by tests, not discovered at runtime.
- Training sessions are gated on preflight success; this adds a small startup cost
  but prevents multi-hour runs from being silently invalidated.
