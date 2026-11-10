# Geometry #145: correct the neighboring-derivative verifier

October 30, 2026. The corrected differential self-consistency check passes on
native and WASM at the unchanged `1e-4` tolerance. The original central-difference
measurements remain recorded, including their failed verdict. Field coefficients,
Physics profiles, test fixtures and crossing thresholds are unchanged.
PR #148 remains unready because of reference-fidelity failures and pending adoption.

## Reproduce the verifier defect independently

The old checker compared a central difference directly with the returned jet.
A finite central difference has a leading truncation term proportional to the
square of its increment. An accurate jet can therefore fail the check when the
field has large higher derivatives.

The analytic control is

```text
F(x,y) = x^2 + 3xy + 2y^2 + 1e8 (x^4 + y^4).
```

At the origin its exact Hessian is `[[2,3],[3,4]]`. With increment `1e-6`, the
old normalized Hessian error is approximately `0.0002`, above `0.0001`, despite
the exact analytic derivatives. The regression was observed red before correcting
the estimator. It does not depend on Mandelbrot preparation or this candidate's
coefficients.

The new `richardson-neighbor/1` checker evaluates central differences at `h`,
`h/2` and `h/4`. It forms two successive estimates using
`R(h) = (4*C(h/2) - C(h))/3`. This cancels the leading truncation term in smooth
neighborhoods. Both estimates must agree with the returned jet and with each
other at the original tolerance. It never selects whichever increment looks best.

The original 1,000 query points, both base increments `1e-6` and `1e-7`, gradient
absolute-error convention and Hessian normalization `max(1, abs(returned H))`
remain unchanged. Nested probes add evidence. No point or base increment was
removed. Existing patch-join checks also remain intact; this does not assume
uniform fourth-order convergence across joins.

The analytic quartic now passes. Controls with a constant gradient corruption
of 0.001 or symmetric off-diagonal Hessian corruption of 0.01 still fail.
Unavailable support, nonfinite complete jets, invalid increments and unresolved
coordinate increments refuse rather than produce passing evidence.

## Frozen-field results

| Measured quantity | Maximum error | Retained tolerance |
| --- | --- | --- |
| Corrected scalar-to-gradient consistency | `1.3368e-8` absolute | `1e-4` |
| Corrected gradient-to-Hessian consistency | `2.8768e-6` normalized | `1e-4` |
| Gradient agreement between successive resolutions | `2.0169e-8` absolute | `1e-4` |
| Hessian agreement between successive resolutions | `3.5833e-6` normalized | `1e-4` |
| Preserved original central Hessian comparison | `2.0479e-4` normalized | `1e-4`, fails |

Both native and WASM execute the same Rust checker and pass. The browser verifier
also requires matching checker version, point count, increments, tolerance and
verdict, preventing mixed verification rules from being presented as parity.
Jet/state/ledger comparisons retain the declared numerical tolerances; no
cross-target bitwise requirement was added.

This changes the numerical differentiation estimator. It does not change the
scalar function or the derivatives returned to Physics. The legacy central
result is preserved under `legacy_central_differences`; it is not relabeled as
passing. `derivative_checks_pass` now names the validated estimator explicitly
through `derivative_verifier`.

The field remains `b45293bd1d6a1bd6`, asset SHA-256
`2a7afa0f10119b508bde3fdfb80438c3724137d824ed037239eb8173f9330dfc`.
The four-substep profile and frozen reference policy are unchanged from the
[integrator report](geometry-145-operational-integrators.md). All 35 native
trajectories exactly reproduce their prior initial/final mechanical states,
complete profiles, crossing times, work counts and reference pairs. The original
energy-run records also match exactly. This comparison uses the prior archived
report's verified checksum.

All **27 Rust controls pass**. The full unchanged Python physics file remains
**26 passed, 1 failed**, with only scalar-error certification failing. Its full
file SHA-256 remains `2451ab5c7ffa18f9cc8259ef109a4bca27cd1af846f9ac6843a626ed93c84536`.
The original native crossing body passes unchanged and retains conclusive
reference evidence. All 35 per-run expectations pass, with 15 verified reference
crossings; all three original above-barrier cases have reference evidence.
Known reference-cut refusal and complete-state rollback remain enforced.

## Cost and remaining claims

Verification runs offline through the diagnostic report. No checker calls,
fits, proofs or new work enter a prepared query or Physics step. The timed step
still uses 160 geometry evaluations and at most 5,760 coefficient reads.

Device: Intel Core i9-14900, Windows; headless Chromium 153.0.8010.12. Native
cache-stressed query p99 is 2.8 / 2.8 / 2.9 microseconds across three fresh
processes. Complete-step p99 is 106.7 / 51.5 / 106.7 microseconds. Browser query
p99 is 15 microseconds and complete-step p99 is 430 microseconds. These meet the
provisional 100 microsecond query p99 and 2 ms complete-step p99 targets.
Unavailable lookups also meet the query p99 target: 0.1 microseconds on native
and 5 microseconds in the browser, with observed maxima of 5.9 and 365 microseconds.

Browser first prepared query is 180 microseconds; cache-stressed query maximum
is 2.220 ms. First/maximum complete-step latency is 2.435 ms, exceeding 2 ms.
Passing p99 does not establish a hard deadline. Full first/p50/p99/max distributions
are preserved rather than selecting the earlier faster run.

Loading and preparation remain outside those timers. Native loading is 0.125 to
0.128 s. Browser initialization, local asset fetch, decode and declaration fetch
are 28.16, 137.33, 174.15 and 7.15 ms. Coefficients remain 77,405,000 bytes;
browser WASM capacity remains 160,301,056 bytes plus JS allocations. Native replay
peak working sets are 245,698,560 / 248,307,712 / 248,377,344 bytes. Prior offline
preparation measurements apply to the identical asset bytes and remain separate.

The [machine-readable record](geometry-145-operational-verifier.json) and
[raw evidence archive](experiments/geometry-145-operational/verifier-evidence.zip)
preserve the initial analytic red result, original/corrected derivative witnesses,
fresh native/browser reports, protected original-test results, scalar audit and
source/binary hashes. Experimental native/Python/WASM release builds, default
native compilation, syntax and formatting checks pass.

The two certified scalar counterexamples still fail the retained budget. Scalar
upper bounds, reference derivative fidelity, complete reference-singularity
information, shipping adoption, compatibility/version work and deployment/training
validation remain unresolved. Differential self-consistency establishes a coherent
operational field under these checks; it supplies no reference-distance certificate
or permission to move through an authoritative reference singularity.
