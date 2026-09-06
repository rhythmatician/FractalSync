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

## Amendment: capability-constrained ownership (2026-08-29)

The earlier amendments in this ADR named the categories of math that must
live in `runtime-core` ("synthesis, geometry, feature-extraction, minimap,
visual metrics"). That wording was too narrow. It invited an agent — or a
human — to reason:

> "This new code is orchestration / preprocessing / tensor packing / timing /
> policy glue, not one of the listed mathematical categories, so Python or
> TypeScript is fine."

That is exactly how domain behavior leaked outward, sometimes silently, and
always at the cost of parity drift.

### The broadened ownership rule

`runtime-core` is the default home of **all deterministic FractalSync domain
behavior and all semantics shared between training and runtime**. Code may
live outside Rust only when its responsibility is intrinsically
surface-specific:

- **Python owns training.** Models, optimizers, batching, training loops,
  loss aggregation, experiment configuration, dataset orchestration,
  checkpointing, offline plots/reports, and differentiable surrogates
  *only when differentiation prevents directly calling Rust*.
- **TypeScript owns the browser.** WebAudio / AudioWorklet, DOM, React,
  file selection, microphone permission, WebGL, browser lifecycle, ONNX
  Runtime Web invocation, WASM loading.

Non-Rust implementations of Rust-owned behavior are **exceptions, not
peers**. Each exception requires an explicit reason and, where behavior is
mirrored, mechanical parity enforcement.

### The decision test

> **Would changing this code alter what the Player sees, decides, or does?**

If yes, it almost certainly belongs in Rust. That test covers far more
than "math". Examples that have been or would be Rust-owned under the
broadened rule:

- how an `AnalysisTick` is constructed;
- the canonical audio timebase / sample clock (see PR #91);
- how a `CycleBank` hypothesis is updated and which of its values become
  Player inputs (see ADR 0002 for the superseded hypothesis-bank framing; the
  currently implemented cycle-bank architecture is in ADR 0003);
- how `PlayerObservation` is packed, normalized, and ordered;
- model input / output schema, control ranges, and control interpretation;
- how Controls affect Physics, Map observations, realm and Shore
  calculations;
- state-reset semantics;
- any deterministic diagnostics used to evaluate those things.

### "If both Python and TypeScript need to know something, that thing belongs in Rust."

This is the more powerful heuristic and the one most likely to prevent
the next drift. Whenever a fact has to be consistent across both surfaces
— and in FractalSync it almost always does — there must be exactly one
implementation of that fact, in Rust, exposed via PyO3 and `wasm-bindgen`.
Python and TypeScript call the same Rust pipeline, not equivalent
compositions of the same Rust primitives.

### Share pipelines, not just primitives

The original 0001 amendment (2026-08-27) made the *primitive* authoritative
and let training and runtime compose those primitives differently. That was
the actual failure mode we kept hitting: identical ingredients, two
recipes, drift by composition. The destination is a single Rust pipeline
called from both surfaces:

```text
             Rust pipeline (runtime-core)
                /            \
             PyO3            wasm-bindgen
              /                \
          Python                TS
```

Not equivalent compositions of the same Rust primitives. Parity tests
remain for the few unavoidable mirrors and for binding / surface
equivalence, but the goal is to **eliminate most parity tests by
construction**.

### Capabilities: what Python and TypeScript may contain

Each non-Rust language has an explicit set of allowed responsibilities
(`shared/architecture_mirrors.json::capabilities` block). The list is
deliberately small:

```jsonc
{
  "typescript": {
    "allowed_responsibilities": [
      "browser_api",          // WebAudio, AudioWorklet, DOM, WebGL
      "ui",                   // React, file selection, mic permission
      "binding_adapter",      // wraps wasm-orbit / onnxruntime-web
      "onnx_transport"        // model + metadata IO + lifecycle
    ]
  },
  "python": {
    "allowed_responsibilities": [
      "training",             // models, optimizers, batching, loops
      "autograd",             // PyTorch graphs, differentiable mirrors
      "experiment_orchestration", // config, checkpoints, reports
      "diagnostics"           // offline plots, NEVER consumed by model
    ]
  }
}
```

Any file outside `runtime-core/` that contains domain behavior MUST either
(a) be a `binding_adapter` that consumes the Rust authority (no formulas),
(b) be one of the small set of declared mirrors with an explicit
`why_outside_rust` reason referencing one of the capabilities above, or
(c) be a `diagnostic` that is provably not on a training or runtime path.

### What TypeScript must not contain

```ts
energy = sigmoid(...)
thrust = energy * 0.06
if (onset > ...) sectionChanged = hueDelta > ...
dt = 1 / 60
pack observation in this particular ordering
interpret output index 3 as ...
```

All of those are product semantics. This list records the split authority
found during PR #93: `frontend/src/lib/modelInference.ts` still contained
those decisions at that time. The later model-I/O consolidation resolved
that finding by moving schemas, decoding, audio summaries, and visual/drive
projections into `runtime-core/src/model_io.rs`. The TypeScript file now owns
ONNX transport and delegates product interpretation through the generated
WASM bindings.

### What Python must not independently decide

- feature semantics;
- audio cadence / sample-clock;
- `CycleBank` behavior;
- `PlayerObservation` packing;
- model I/O ordering;
- runtime timestep;
- control semantics;
- Map geometry;
- Physics behavior.

Even training data should ideally flow as:

```text
audio
  ↓
runtime_core canonical pipeline
  ↓
PlayerObservation
  ↓
Python converts to torch.Tensor
  ↓
model
```

not:

```text
audio
  ↓
Python approximately reconstructs runtime state
  ↓
model
```

The second architecture guarantees drift.

### Differentiable mirrors are still the one major exception

Some places cannot be served by `runtime-core` directly because PyTorch
needs an autograd graph. Those are legitimate Python implementations and
remain `differentiable_mirror` entries. They are derived artifacts,
almost like generated code:

```text
Rust authority  →  explicit Python differentiable mirror  →  mandatory parity proof
```

The mistake to avoid is letting the exception expand into ordinary
Python implementations just because Python happens to be where training
lives.

### Mechanical enforcement: capability checks

ADR language alone has not been enough. The architecture guardrail
(`scripts/architecture_guardrail.py`) evolves from "detect duplicated
known formulas" to "non-Rust code is capability-constrained". Three new
classes of structural check are added under this amendment:

1. **Domain-type re-declaration.** TypeScript or Python declaring a
   domain-state type that is already exported by Rust
   (`OrbitState`, `PlayerObservation`, `CycleHypothesis`, `AnalysisTick`,
   `ResidualParams`, …). The check requires corroborating evidence
   (the type name AND a structural-field token in the same file), so
   genuinely new helper types are not flagged.

2. **Hard-coded shared constants outside Rust.** Magic-number copies of
   `SAMPLE_RATE`, `WINDOW_FRAMES`, `FEATURE_VERSION`,
   `CONTROLLER_VERSION`, `NORM_EPS`, `K_RESIDUALS`, etc. in non-binding
   non-manifested code. Bindings are exempt because bindings are how the
   constant reaches the other surface; consumers must read it from the
   binding rather than restating it.

3. **Duplicate feature / normalization / observation-packing functions.**
   Functions whose names + signatures indicate they perform the canonical
   extraction, normalization, or packing pipeline — outside a manifested
   mirror or binding adapter — fail. Naming and signature-shape evidence
   is required so ordinary helpers (e.g. a `pack_batches` for tensors)
   are not flagged.

`shared/architecture_mirrors.json` gains a top-level `capabilities`
block, and every mirror entry gains a `why_outside_rust` field naming the
capability that justifies it. The manifest validator and the
`test_architecture_guardrail.py` test are extended to enforce both.

The full per-kind parity contract from the issue-#90 amendment still
applies; this amendment adds a new `transitional_mirror` kind so files
like `modelInference.ts` can be tracked with an explicit sunset path
rather than collapsed into the behavioral-mirror category.

### Consequences

- Any agent deciding where a new piece of code belongs runs the decision
  test above and, if non-Rust, names the capability it relies on.
- Any future change that introduces a parallel pipeline (training calling
  primitives differently from runtime) is a regression of this amendment
  and must be justified with a new ADR.
- The destination is fewer parity tests, not more. The guardrail grows
  toward stronger pipeline-equivalence checks; mirrors shrink as their
  Rust pipelines absorb them.

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
- `PhaseTracker` / `CycleBank` (planned Rust-owned concepts — only
  *implementation-shaped* declarations (`class`/`def`/`function`/
  `interface`/`type`) are flagged; binding consumers and manifest adapters
  are exempt, mirroring the controller rule)
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
- `kind` — one of `differentiable_mirror`, `behavioral_mirror`,
  `diagnostic`, `experimental`
- `rust_authority` — the Rust module that owns the canonical behavior
- `reason` — why the mirror exists (e.g. differentiable training surrogate)
- `parity` — the checks/tests that pin it to the authority

The parity contract is enforced per kind, so "parity required" is real
rather than satisfied by a `"none (diagnostic-only)"` string:

- `differentiable_mirror` / `behavioral_mirror` — MUST have a nonempty
  `parity` list of real executable checks; a `"none"` entry fails.
- `diagnostic` / `experimental` — MUST declare `"none"` parity with a
  reason; claiming real checks fails (an experimental mirror must not
  pretend to be pinned).

The guardrail fails if a manifest entry's path does not exist, lacks a
`rust_authority`, has a missing/invalid `kind`, or violates its kind's
parity contract. `backend/tests/test_architecture_guardrail.py`
additionally verifies every behavioral mirror's parity entry references a
real check or test file.

### How to add a legitimate exception

1. Implement the mirror with a header comment naming the Rust authority.
2. Add an entry to `shared/architecture_mirrors.json` with `path`, `kind`,
   `rust_authority`, `reason`, and `parity`.
3. Add the parity check (preflight check, golden test, or pytest/vitest
   parity test) and list it in the entry's `parity`.
4. Run `python scripts/architecture_guardrail.py` and the parity tests —
   both must pass in the same commit.

Do not work around the guardrail by renaming symbols or splitting formulas
across lines; that converts an accidental second authority into a
deliberate one, which is exactly what the manifest exists to make visible.

### Current mirror inventory (audited 2026-08-29)

| Mirror | Kind | Authority | `why_outside_rust` | Parity |
|---|---|---|---|---|
| `backend/src/cspace_proxies.py` | differentiable_mirror | `controller.rs`, `proxies.rs` | `python:autograd` | preflight (b)(e)(e3)(e4), golden tests |
| `backend/src/python_feature_extractor.py` | diagnostic | `features.rs` | `python:diagnostics` | preflight (g)(h) |
| `frontend/src/lib/__tests__/orbitSynthesizer.mock.ts` | behavioral_mirror | `controller.rs` | `typescript:binding_adapter` | vitest goldenParity |
| `backend/src/visual_metrics.py` | behavioral_mirror | `visual_metrics.rs` | `python:training` | test_visual_metrics* |
| `backend/src/julia_gpu.py` | behavioral_mirror | `visual_metrics.rs` | `python:training` | test_visual_metrics* |
| `backend/src/c_trace_plot.py` | diagnostic | minimap geometry | `python:diagnostics` | none (diagnostic-only) |
| `backend/src/live_controller.py` | experimental | `features.rs` | `python:experiment_orchestration` | none — must not be promoted to training/runtime without delegating to `runtime_core.FeatureExtractor` or adding a real parity check |
| `frontend/src/lib/modelInference.ts` | adapter | `model_io.rs` + `controller.rs` + `features.rs` | `typescript:onnx_transport` | modelOutputWasmParity, modelInferenceWasm, preflight (f) |

Clean adapters (verified, no formulas): `frontend/src/lib/orbitSynthesizer.ts`,
`frontend/src/lib/canonicalFeatures.ts`, `backend/src/distance_utils.py`,
`backend/src/runtime_core_bridge.py`, `frontend/src/lib/modelInference.ts`.

> Historical note: the 2026-08-29 amendment temporarily classified
> `modelInference.ts` as a `transitional_mirror` after PR #93 fixed its
> sample-clock timing but left product interpretation in TypeScript. The
> model-I/O consolidation subsequently moved that interpretation to Rust and
> restored the file's adapter classification. The manifest and compiled-WASM
> tests now enforce that boundary.


## Amendment: canonical pipeline boundary and analysis-pipeline version (2026-08-30, issue #93)

**Canonical pipeline boundary.** The canonical shared pipeline BEGINS at
decoded PCM + native sample rate:

    decoded PCM (native rate) + native sample rate
                |
            runtime-core
                |
    AnalysisTimebase.ingest(samples, source_sample_rate, source_start_frame)

Media DECODING is surface-specific and legitimately NOT shared: Python may
use librosa/soundfile, the browser may use Web Audio
(`decodeAudioData` / `MediaStream`). Everything downstream MUST be
identical: native-rate PCM enters `runtime-core` and the Rust
`StreamingResampler` owns ALL rate conversion on every surface. Never
pre-resample in Python (`librosa.load(sr=SAMPLE_RATE)`) or in TypeScript
before the timebase — that reintroduces a second resampler authority and
is exactly the divergence class the production-path parity tests
(`backend/tests/test_audio_pipeline_parity.py`) exist to catch.

Phase-critical file-mode comparisons must use PCM/WAV fixtures: MP3/AAC
decoders apply codec-specific priming/delay that shifts sample alignment
between decoders. Live microphone input is already PCM, so the issue does
not exist there.

**Analysis-pipeline version contract.** `ANALYSIS_PIPELINE_VERSION`
(`runtime-core/src/timebase.rs`, currently `analysis/1`) versions HOW audio
reaches the extractor: resampling ownership, hop scheduling, epoch
semantics, and window anchoring. It is deliberately DISTINCT from
`FEATURE_VERSION`, which versions the feature FORMULAS — a model trained
against a different pipeline consumes inputs with different semantics even
when the formulas are identical.

- Bump `ANALYSIS_PIPELINE_VERSION` whenever resampling ownership, hop
  scheduling, window anchoring (`TICK_WINDOW_SAMPLES`), or epoch/reset
  semantics change.
- It is exported through PyO3 and wasm `constants()`, stamped into ONNX
  metadata as `analysis_pipeline_version`, and the browser REFUSES models
  with a missing or mismatched stamp (no legacy warning path: every
  pre-timebase model was trained on a pipeline the runtime no longer
  executes).
- Consumers must ALIAS the binding value, never restate it:
  `data_loader.PIPELINE_VERSION = ANALYSIS_PIPELINE_VERSION` (Python) and
  `constants().analysis_pipeline_version` (browser). The architecture
  guardrail (`hardcoded-pipeline-version`,
  `hardcoded-shared-constant`, `hardcoded-hop-length`) rejects restated
  constants and version strings outside runtime-core.

**Constant authority in TypeScript.** `SAMPLE_RATE`, `HOP_LENGTH`, and
`N_FFT` have NO TypeScript literal. Runtime code reads them from
`wasm.constants()` (`getWasmConstants()` + `getRuntimeConstants()`); tests
inject/mock the binding constants. A missing binding throws — it must
never silently fall back to a second authority.
