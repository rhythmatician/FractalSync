# #145 requested-width distance bounds — 2026-10-26

PR #148 remains unready to merge. The experimental scalar enclosure can now
refine to a caller's requested width or refuse, rather than merely return a
positive interior clearance. Both high-period queries pass two requested widths.
The work also fixes a certified-arc closure refusal exposed by refinement.
Production geometry, thresholds and original crossing tests are unchanged.

## Construction and final authority

The [distance-bracket argument](geometry-145-distance-brackets.md) is unchanged:
an attracting closed cover supplies an interior lower bound, and a separately
proved escaping parameter supplies an upper bound at the same exact query.

The new proposal screen evaluates at most 567 binary64 critical orbits and ranks
suggested exterior points by endpoint distance. It uses the strict, pre-update
escape convention and fixed 32768-update limit. Screening has no proof authority:
at most 32 selected points are tried at 128 and 256 bits using the existing
escape proof. Fabricated screen metadata and nonfinite/interior proposals cannot
issue an `OutsideWitness`. The first ranked proposal certifies at both final
high-period queries, escaping at updates 17192 and 30890 respectively.

For a requested width W, the proved exterior endpoint distance U suggests a
minimum clearance near U-W. This target only guides subdivision. Every arc and
connector still passes the existing strict inclusion, contraction, identity,
attraction and separation checks. The completed cover's directed clearance is
checked against the target again. Finally, the actual bracket is consumed by
`require_width`, which independently verifies its upward-rounded width against
the positive finite request. Rounded proposal targets cannot issue a false
accuracy claim. An unattainable target or exhausted budget refuses.

The exercise widths below are inputs to this new experiment. They do not replace
or relax any Physics acceptance tolerance.

## Unequal-arc closure repair

Clearance-driven subdivision exposed a case where the first and last arcs had
different error domains. Neither direct glue direction could establish their
shared endpoint identity. The fallback now proposes a root at the exact shared
multiplier and certifies it with zero multiplier radius. Both arcs must then
independently glue to that same immutable point certificate through the original
domain, parameter-identity and cycle-phase checks. Numerical agreement alone
does not establish either link.

Cycle identity up to phase is transitive. The singleton certificate proves only
endpoint identity; the original edge certificates still establish continuity and
attraction. The fallback fixes the asymmetric refinement control without
accepting distinct components. Both links are protected by independent negative
tests and mutations using genuine period-three certificates.

## Final frozen results

All bounds and widths below are approximate decimal renderings of directed
multiprecision results. Query coordinates and periods match the preceding report.

| Query | Requested width | Proved width | Lower bound | Upper bound | Edges | Cold wall time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Native | 2e-7 | 1.6872368141395535e-7 | 7.500128320338055e-6 | 7.66885200175201e-6 | 238 | 177.35 s |
| Native, tighter | 5e-8 | 4.9181703949295564e-8 | 7.619670297802715e-6 | 7.66885200175201e-6 | 248 | 184.21 s |
| Python | 2e-9 | 6.855960717739659e-10 | 1.7240936517760065e-7 | 1.7309496124937462e-7 | 172 | 182.40 s |
| Python, tighter | 5e-10 | 4.706751315128663e-10 | 1.7262428611786175e-7 | 1.7309496124937462e-7 | 173 | 184.41 s |

The four high-period runs executed concurrently with mutation compilation. The
cardioid end-to-end control also passes. The final executable includes the point
bridge and both independent completion gates; earlier frozen variants are
archived separately rather than substituted for these results.

## Cost breakdown and limits

| Final run | Jet/check/screen preparation | Witness proof | Cover and final curve validation | Final width validation |
| --- | ---: | ---: | ---: | ---: |
| Native | 0.426 s | 0.919 s | 175.412 s | 11.9 microseconds |
| Native, tighter | 0.427 s | 0.896 s | 182.338 s | 10.6 microseconds |
| Python | 0.743 s | 1.591 s | 179.467 s | 12.2 microseconds |
| Python, tighter | 0.735 s | 1.578 s | 181.546 s | 10.4 microseconds |

Cold time remains unsuitable for a runtime miss. Region counts do not capture
all work: each region can attempt multiple branch certificates, and identity
joins can require phase checks and a point bridge. These CLI runs allow 8192
regions, depth 24 and twelve attempts per `certify_auto` call; a bridge can make
an additional call. An external 1200-second process limit bounds each high-period
run.

The retained compact-domain benchmark takes 0.183–0.204 seconds for 10,000
nearby scalar intervals in these runs. It repeats 323 positions inside a box of
half-width `0.001 * clearance`. It excludes cold construction, full trajectory
coverage, geometry jets and cache replacement. The numerical local jets continue
to pass their retained neighboring-gradient/Hessian checks, but the scalar
certificate does not establish a globally nearest branch or derivative errors.

## Verification and reproducibility

The isolated crate passes 54 tests and catches all 44 mutations by assertion
failures, not compilation failures. New coverage exercises actual refinement
above a coarse cover's bound, invalid and unattainable requests, false numerical
escape metadata, both bridge links, and independent final accuracy gates.

The final-clearance fixture is deliberately stricter than an early-failure test:
all edge capsules and preceding connectors exceed the target, while the closing
connector stays positive below it. `closed()` succeeds, but `finish_cover` must
refuse the requested target. A valid overly broad bracket similarly refuses
through `require_width`, independently of cover construction.

```text
cargo test --release --jobs 1 --manifest-path docs/validation/experiments/geometry-145-point-proof/Cargo.toml --target-dir target/geometry145-precision
cargo build --release --jobs 1 --manifest-path docs/validation/experiments/geometry-145-point-proof/Cargo.toml --target-dir target/geometry145-precision --bin component_probe
target/geometry145-precision/release/component_probe tight-bracket-native 0.999999 128 0.9 0.05 5e-8
target/geometry145-precision/release/component_probe tight-bracket-python 0.999999 128 0.9 0.05 5e-10
```

The accompanying JSON contains fourteen terminal runs, exact source/manifest and
binary hashes, separate phase timings, final test output and mutation outcomes.
The final frozen executable matches the built executable byte for byte.

## Remaining merge work

These are mathematical Mandelbrot distance intervals, without established
equivalence to the canonical finite floating orbit. They are not a coherent
distance value/gradient/Hessian field. Original native/Python crossing failures,
live training validity and full query cost remain unresolved.

The next implementation step is to reuse a validated cover over its whole
interior, rather than recertify at every anchor or rely only on a tiny transported
ball, then connect certified node values to the coherent field prototype.
That integration must preserve the original classification and energy tests and
complete the version, golden, parity, Rust, Python, browser/WASM and CI gates
before this PR can be considered ready.
