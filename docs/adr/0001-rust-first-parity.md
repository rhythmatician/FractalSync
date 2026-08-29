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

## Amendment: controller version contract (2026-08-27)

**Invariant:** an `orbit_control` model deployed to the runtime must have been
trained against the same controller semantics the runtime executes.

- `runtime-core/src/controller.rs` defines `CONTROLLER_VERSION` (single source).
  Bump it in the SAME commit as any change to the flags-off semantics of
  `OrbitController::step`, together with regenerated goldens and updated mirrors.
- Training stamps the exported ONNX metadata with `controller_version`
  (read from the installed `runtime_core.CONTROLLER_VERSION` — the same source
  the preflight verified against).
- The browser **refuses to load** an orbit_control model whose
  `controller_version` differs from its runtime's version. A model missing the
  field (pre-contract legacy) loads with a loud warning and cannot be verified.
- The preflight (check f) fails if `shared/golden_vectors.json` was generated
  by a different controller version than the installed runtime — stale goldens
  cannot silently verify the wrong contract.

## Amendment: feature-extraction contract (2026-08-27)

**Invariant:** an `orbit_control` model deployed to the runtime must have been
trained on features produced by the same extraction pipeline the runtime
executes, with identical window layout and normalization semantics.

**Single implementation, executed twice.** The Rust extractor
(`runtime-core/src/features.rs`) is the only implementation of feature
extraction. Training runs it via the Python bindings; the browser runs it via
a new `FeatureExtractor` binding in `wasm-orbit`, fed raw PCM from
`AnalyserNode.getFloatTimeDomainData` (resampled to 48 kHz). The former
JavaScript reimplementation (`frontend/src/lib/audioFeatures.ts` extraction logic) is retired and deleted —
it had drifted from training on FFT size (2048 vs 4096), smoothing,
dB-domain math, and per-file min-max normalization that the browser cannot
reproduce.

- `FEATURE_VERSION` in `features.rs` is the single source (`features/2`; history: `features/1` = causal baseline with frame-major/causal transforms, `features/2` = flux-mean fix).
  Bump it in the SAME commit as any change to feature definitions, fixed
  transforms, window layout, or STFT defaults, together with regenerated
  goldens and updated mirrors.
- `NORM_EPS` is pinned in `features.rs`; trainer mirror and browser read it
  from bindings rather than hard-coding their own epsilon.
- **Causal transforms replace per-file min-max**: energy-like features
  (flux/rms/onset) use `log1p(100·x)/log1p(100)`. Min-max depended on the
  whole file (impossible at runtime) and made training inputs depend on
  dataset composition.
- **Frame-major window layout** everywhere:
  `[f0(t0)..f5(t0), f0(t1)..f5(t1), ...]`. The Rust extractor previously
  flattened feature-major while Python/browser used frame-major — a silent
  input-permutation bug.
- Training stamps ONNX metadata with `feature_version`; the browser refuses
  to load an orbit_control model whose `feature_version` differs from its
  runtime's (same mechanism as `controller_version`).
- Golden vectors now include `feature_cases`: deterministic synthetic audio
  windows recorded from the canonical extractor. Preflight check (g) replays
  them through the Python mirror (must match within 5e-3; measured 1e-15);
  check (h) fails on stale `feature_version` in the goldens.

**Note:** this contract changes what models learn (inputs differ from all
previously trained checkpoints). Models trained before this amendment are
invalid for the new pipeline and must be retrained.

## Version increment policy

`FEATURE_VERSION` and `CONTROLLER_VERSION` are pinned in `runtime-core` (`features.rs` and `controller.rs`). Bump rules:

- Bump `FEATURE_VERSION` whenever ANY of these change: feature definitions / formulas (centroid/flux/rms/zcr/onset/rolloff), fixed transforms (log/clamp/scaling), window flattening layout (frame-major vs feature-major), or STFT defaults (`n_fft`, `hop_length`). See the doc comment on `FEATURE_VERSION` in `features.rs` for the authoritative list and version history (`1` causal baseline, `2` flux-mean fix).
- Bump `CONTROLLER_VERSION` whenever flags-off semantics of `OrbitController::step` change (constants, formulas, or order of operations).
- The bump, `cargo run --release -p runtime_core --bin generate_golden_vectors` (regenerate `shared/golden_vectors.json`), and updates to ALL mirrors (`backend/src/cspace_proxies.py`, `backend/src/python_feature_extractor.py`, `frontend/src/lib/canonicalFeatures.ts` / `wasm-orbit`) MUST land in the SAME commit. Preflight checks (f) and (h) fail on stale versions in goldens; (b) and (g) fail on numeric drift.

CI enforces this via `.github/workflows/pytest.yml` (runs `scripts/preflight_parity.py` via `backend/tests/test_preflight_parity.py` and `backend/tests/test_golden_parity.py`) and the `train.py` preflight gate.

## Amendment: machine-enforced ownership guardrail (2026-08-28, issue #90)

**Rule:** shared cross-runtime behavior has one authority: `runtime-core`.
Other languages consume bindings. Intentional mirrors require an explicit
documented exception and parity enforcement.

### The guardrail

`scripts/architecture_guardrail.py` is a deterministic, dependency-free
check that scans `frontend/src`, `backend/src`, `backend/api`, `scripts`,
and `shared` for re-implementations of Rust-owned runtime concepts:

- competing `OrbitController` / `OrbitSynthesizer` / `PlayerState` classes
  with a `step` method (binding wrappers are exempt — wrapping is their job)
- the cardioid parameterization `mu = 1 - sqrt(1-4c)` (import
  `cspace_proxies.cardioid_mu` instead of re-deriving it)
- the cardioid boundary outline `c = e^{it}/2 - e^{2it}/4`
- residual epicycle amplitude ladder `2^(k+1)` with per-band gates
- Julia/Mandelbrot escape iteration (`z = z^2 + c` with bailout)
- the features/2 causal transform `log1p(100x)`
- `PhaseTracker` / `CycleBank` (planned Rust-owned concepts — they must be
  born in Rust, not prototyped in TS/Python)
- shore/minimap physics named APIs (`contour_biased_step`,
  `MUSIC_PUSH_GAIN`, ...) in code (comment mentions are fine)

The rules guard architectural concepts with corroborating evidence, not
ordinary math tokens, so they stay low-noise. Generated binding
declarations (`*.d.ts`) and the guardrail/preflight scripts themselves are
excluded.

### The exception manifest

`shared/architecture_mirrors.json` is the checked-in allowlist of
intentional mirrors. Each entry states:

- `path` — the mirror file
- `rust_authority` — the Rust module that owns the canonical behavior
- `reason` — why the mirror exists (e.g. differentiable training surrogate)
- `parity` — the checks/tests that pin it to the authority

The guardrail fails if a manifest entry's path does not exist, lacks a
`rust_authority`, or lacks a `parity` list. `backend/tests/
test_architecture_guardrail.py` additionally verifies every parity entry
references a real check or test file.

### How to add a legitimate exception

1. Implement the mirror with a header comment naming the Rust authority.
2. Add an entry to `shared/architecture_mirrors.json` with `path`,
   `rust_authority`, `reason`, and `parity`.
3. Add the parity check (preflight check, golden test, or pytest/vitest
   parity test) and list it in the entry's `parity`.
4. Run `python scripts/architecture_guardrail.py` and the parity tests —
   both must pass in the same commit.

Do not work around the guardrail by renaming symbols or splitting formulas
across lines; that converts an accidental second authority into a
deliberate one, which is exactly what the manifest exists to make visible.

### Current mirror inventory (audited 2026-08-28)

| Mirror | Authority | Parity |
|---|---|---|
| `backend/src/cspace_proxies.py` | `controller.rs`, `proxies.rs` | preflight (b)(e)(e3)(e4), golden tests |
| `backend/src/python_feature_extractor.py` | `features.rs` | preflight (g)(h) |
| `frontend/src/lib/__tests__/orbitSynthesizer.mock.ts` | `controller.rs` | vitest goldenParity |
| `backend/src/visual_metrics.py` | `visual_metrics.rs` | test_visual_metrics* |
| `backend/src/julia_gpu.py` | `visual_metrics.rs` | test_visual_metrics* |
| `backend/src/c_trace_plot.py` | minimap geometry | diagnostic-only (stated in manifest) |
| `backend/src/live_controller.py` | `features.rs` | experimental, NOT parity-pinned — must not be promoted to training/runtime without delegating to `runtime_core.FeatureExtractor` or adding a real parity check |

Clean adapters (verified, no formulas): `frontend/src/lib/orbitSynthesizer.ts`,
`frontend/src/lib/canonicalFeatures.ts`, `frontend/src/lib/modelInference.ts`,
`backend/src/distance_utils.py`, `backend/src/runtime_core_bridge.py`.

