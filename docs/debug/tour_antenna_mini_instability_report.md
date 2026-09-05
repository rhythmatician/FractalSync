# tour_antenna_mini instability — evidence report (2026-09-04)

Branch: `prototype/82-native-shore-crossing`. Diagnostic harness:
`runtime-core/src/bin/diagnose_antenna_mini.rs` (native) +
`tmp/wasm_replay/replay.mjs` (deployed wasm under Node). No production
Physics behavior changed except the fix below, which the user requested.

## 0. Root cause found and fixed

`manifold::integrate_step` computed

    Q_total = Q_potential + Q_control + Q_drag

and **never added `Q_wall`**, even though `wall_potential` was included in
`total_energy` (the ledger) and `wall_force()` was exported as a binding.
The outer wall participated in energy accounting but not in the dynamics.

**Fix applied (this commit):**

    Q_total = Q_potential + Q_wall + Q_control + Q_drag

Consistency updates in the same change:
- `runtime-core/src/manifold.rs` — kernel force sum.
- `runtime-core/src/debug.rs` — `snapshot_from_state` net-accel
  reconstruction mirrors the same sum (it must always track the kernel).
- `backend/src/cspace_proxies.py` — mirror energy ledger now E = K +
  U_sigma + U_wall; `_ControlsStep` docstring updated.
- `backend/tests/test_manifold_physics.py` — drift rollouts compute the
  full ledger (wall is conservative; omitting U_wall would show wall-KE
  exchange as fake drift).

## 1. Ignition frames (native, post-fix, authoritative seam)

| event | frame |
|---|---|
| last ordinary frame | 137 |
| first anomalous accel (|a| > 10) | **138** |
| largest accel | 140, |a| = 90.7 |
| first hard-guard rejection | **142** (`|c_new|^2 = 4.0366 >= 4`) |

Velocity history (matches the browser exactly):
`(-0.07, 0.10) @137 -> (-0.46, 0.44) @139 -> (-1.83, 1.24) @140 ->
(-2.75, 2.73) @142`. The proposed step at 142 exits the disk and the hard
guard rejects it; the controller holds the last valid state.

## 2. Critical window decomposition (a_total = -Γ(v,v) + G⁻¹Q_total)

| tick | D | sigma | \|grad σ\| | \|Γ(v,v)\| | \|a_force\| | \|a_total\| | K |
|---|---|---|---|---|---|---|---|
| 135 | -9.8e-4 | 6.66 | 1285 | 1.04 | 0.01 | 1.04 | 155 |
| 136 | -2.2e-4 | 8.69 | 5508 | 6.04 | 0.01 | 6.05 | **36425** |
| 137 | -1.7e-3 | 5.84 | 685 | 1.86 | 0.01 | 1.85 | 1625 |
| 138 | -3.9e-3 | 4.67 | 157 | 24.49 | 0.02 | 24.48 | 173 |
| 139 | +6.2e-3 | 4.01 | 206 | 74.21 | 0.09 | 74.10 | 8230 |
| 140 | +5.6e-2 | 0.84 | 30 | 91.06 | 0.29 | 90.72 | 1778 |
| 142 | +2.5e-1 | -1.33 | 10 | 37.36 | 0.15 | 36.87 | 554 |

The explosion is **entirely the geodesic term** `-Γ(v,v) = -λ²σ_i σ_jk v^j
v^k / (1 + λ²|∇σ|²)`: the Hessian `σ_jk` (second FD difference of the
sampled field) reaches O(10⁶)–O(10⁷) near D≈0 and multiplies v². Q_wall is
O(1) throughout this window (0.7 at tick 135) — it is NOT the cause of the
launch, but it IS the restoring force that should have been braking the
reflected coast toward |c|=2 (and was absent before the fix).

## 3. Wall-force wiring (traced, not inferred)

Pre-fix `integrate_step` (manifold.rs): force sum contained exactly three
covectors — `potential_force`, caller-supplied `q_control`, `drag_force`.
`wall_force` appeared nowhere in the executed sum; it appeared only in
`total_energy` (ledger) and as a standalone binding. CONFIRMED and now
FIXED. Post-fix trace shows `Qwall=(0.685, 0.001)` contributing to
`a_force` at tick 135 — wired and executing.

## 4. FD convergence at ignition (h = production step ≈ 1.0065e-4)

At tick 138 (first anomalous):

| h× | \|grad σ\| | Hxx | Hyy |
|---|---|---|---|
| 0.25 | 120.44 | 4.530e5 | 1.616e5 |
| 0.5 | 120.25 | 4.532e5 | 1.617e5 |
| 1 | 119.78 | 4.454e5 | 1.618e5 |
| 2 | 119.35 | 4.037e5 | 1.624e5 |
| 4 | 124.14 | 3.567e5 | 1.750e5 |
| 8 | 135.68 | 2.973e5 | 1.941e5 |

grad σ is reasonably converged (±1% over 0.25h–2h). The Hessian is NOT
converged: Hxx moves 35% from h/4 to 8h, and Γ(v,v) inherits that noise
through the analytic connection. This is the classic second-difference
noise amplification on an f32-sampled field (see memory: h must clear the
f32 noise floor; it does, but curvature σ_xx ~ 1/D² still leaves the
sampled second derivative poorly resolved at D ~ 1e-3).

Verdict: **gradient converged, Hessian/Γ unstable** — derivative-noise
dominated curvature.

## 5. Substep experiment from tick 137 (pre-ignition, controls held)

| dt | final \|v\| | final K | E | max mid-step \|a\| |
|---|---|---|---|---|
| dt (1 step) | 0.123 | 172.9 | 177.6 | — |
| dt/2 ×2 | 0.141 | 180.0 | 184.8 | 3.45 |
| dt/4 ×4 | 0.173 | 230.2 | 235.0 | 9.76 |
| dt/8 ×8 | 0.229 | 291.3 | 296.1 | 27.7 |
| dt/16 ×16 | 0.311 | 387.7 | 392.4 | 71.5 |

Substeps converge toward a common (non-explosive) one-tick answer and do
NOT diverge; canonical dt is adequate for one tick of this local
geometry. The instability is a multi-frame feedback loop, not a
single-step dt blowup.

## 6. WASM vs native divergence

With the identical fix in both runtimes (wasm rebuilt from source,
Node-replayed from the same .wasm bytes the browser serves):

- first divergence: tick 44, magnitude dc ≈ 3.2e-17, dv ≈ 1.1e-15
  (fp-rounding level, expected).
- growth: dc 3e-17 @44 → 1.2e-14 @120 → 1.7e-12 @138 → 3.5e-10 @142.
- final states at rejection agree to 9 decimal places
  (c ≈ (-1.940279098, 0.155605486) in both).

So the earlier "1-ulp chaos" attribution is **contradicted**: the two
runtimes followed the SAME trajectory; the pre-fix discrepancy was the
missing wall force plus my own trace-rounding artifact. The discrete map
is unstable near the crest (divergence amplifies ~10⁷ over 100 frames —
Lyapunov-like), but both runtimes ride the same instability together.

## 7. Eikonal check (|grad D| via FD of the SDF)

| tick | h/2 | h | 2h |
|---|---|---|---|
| 137 | 0.8291 | 0.8289 | 0.8282 |
| 138 | 0.4271 | 0.4261 | 0.4239 |
| 139 | 0.8837 | 0.8840 | 0.8890 |

|grad D| − 1 ∈ [−0.57, −0.11] across the window and varies smoothly with
h. The sampled SDF is NOT behaving like a distance field near the mini's
west ridge (contract violated by up to 57%): the baked field's bicubic
interpolation under-resolves the antenna. This compresses σ's dynamic
range but is stable under h-refinement — a smooth defect, not a
non-smooth one.

## 8. Energy sanity

| tick | dE (total) | dK | dU_sigma | dU_wall |
|---|---|---|---|---|
| 136 | **+36272** | +36270 | +2.03 | +0.0003 |
| 137 | -34803 | -34800 | -2.85 | +0.0009 |
| 139 | +8056 | +8057 | -0.66 | +0.007 |
| 140 | -6455 | -6452 | -3.17 | +0.040 |

dE ≈ dK every frame; U_sigma and U_wall barely move (≤ 3.2). The kinetic
energy swings of O(10³)–O(10⁴) per frame are NOT conservative
potential-energy conversion — they are the FD-noisy Γ term injecting and
extracting energy frame-to-frame. Total E itself jumps anomalously
(+3.6e4 then −3.5e4): **this is the bug signature** — a huge velocity is
not legitimate conversion of potential drop.

## Verdict ranking

| explanation | verdict |
|---|---|
| C. FD Hessian / Christoffel instability | **strongly supported** (dominant summand in every explosion frame; Hessian unconverged at h; Γ inherits noise) |
| E. fp sensitivity exposing an unstable map | **supported** (both runtimes amplify 1e-16 → 1e-10 through the crest; but identical trajectories — not chaos, shared instability) |
| B. sampled-SDF defect | **supported** (eikonal violated up to 57%; smooth but wrong curvature) |
| F. outer wall involvement | **was true, now fixed** (Q_wall was missing from dynamics; it is the correct restoring force for the reflected coast) |
| D. canonical dt too large | **contradicted** (substeps converge; no single-step blowup) |
| A. legitimate potential conversion | **contradicted** (dE ≈ dK with U nearly constant; energy ledger shows injection, not conversion) |

## Regression case retained

- `runtime-core/src/bin/diagnose_antenna_mini.rs` — deterministic native
  harness reproducing the launch + guard rejection (ticks 138–142).
- `tmp/wasm_replay/replay.mjs` — Node replay of the deployed wasm.
- `frontend/src/lib/shoreCrossingVariants.ts` — `tour_antenna_mini`
  remains in the cockpit list; it is the eyes-on repro. Ordinary
  navigable terrain (the mini's west shore band) still turns a
  near-crest pass into a catastrophic launch; the fix for THAT is the
  Hessian/curvature resolution (future work), not terrain avoidance.

## Follow-ups (not done here, per scope)

1. Hessian quality: derive σ_jk analytically from the SDF gradient
   structure, or raise the FD stencil to a 5-point rule, before trusting
   Γ near D ≲ 1e-3.
2. Eikonal correction of the baked field (or re-bake with subpixel
   refinement near minis).
3. Regenerate `runtime-core/runtime_core.pyi` (stale, pre-existing —
   missing wall bindings since fc5c09b).
4. Trainer mirror preflight re-run after wheel rebuild (out of scope here;
   no CONTROLLER_VERSION bump — flags-off semantics unchanged for the May
   path, but ManifoldConfig dynamics changed → goldens unaffected since
   manifold path is not in golden vectors).
