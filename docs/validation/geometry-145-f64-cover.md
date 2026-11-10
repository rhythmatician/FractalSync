# Outward f64 closed attracting covers for #145

PR #148 remains unready to merge. This test-only follow-up to the
[local branch proof](geometry-145-f64-branch.md) constructs closed attracting
curves and certifies interior distance lower bounds for all 162 archived scalar
nodes. Production code, thresholds and original crossing tests are unchanged.

Each edge uses a certified coupled-root family over an exact represented
multiplier segment. Both segment endpoints are strictly inside the unit disk;
convexity therefore gives attraction along the entire segment. Endpoint
enclosures and the affine predictor bound the parameter curve in a capsule.
Every adjacent pair, including the final pair, must prove the same endpoint root
through phase-aware uniqueness containment or a separately proved singleton
connector. A numerical agreement or accepted prefix cannot issue a closed cover.

Outward capsule separation and interval orientation prove nonzero winding at
each query. The existing multiprecision cover's fullness argument then gives an
open disk inside M, hence a lower bound on distance to its boundary. This does
not identify the full boundary of a hyperbolic component, require least period,
or supply a distance upper bound or jet.

Two shared curves meet every original lower-error target: numerical distance
proposal minus `5e-8` at the native nodes and minus `5e-10` at the Python nodes.
The targets are rounded upward. All clearances also remain below the upper ends
of the archived reference intervals. The native curve has 3,148 pieces; the Python
curve has 1,201. The recorded focused run takes approximately 9.7 and 7.9 seconds
respectively, including scalar proposals and reference-data emission. These
times exclude exterior proofs, field refinement, Physics and a complete query.

The existing 128-bit implementation independently validates all 4,349 root
domains with the same dyadic preconditioners, encloses the reported inclusion
and contraction bounds, proves every attracting edge and join, closes both paths,
and checks all 162 query clearances. Its geometric lower bounds enclose the f64
bounds from above and its winding numbers agree. Both independent tests pass.

Focused validation passes 38 tests and catches 17 injected faults. Five new
controls remove attraction, exact closure, nonzero winding, capsule radius or
the final clearance target; each causes a test assertion failure. Invalid input,
ambiguous orientation, exhausted work and unachievable clearance requests refuse.
The final provider suite passes 102 tests. The original branch-only harness also
still passes its 32 tests and 12 mutation controls.

Early broad or deep bends near the parabolic root could not achieve the original
clearance targets even after subdivision. A narrower, shallower bend and denser
initial multiplier polygon solve this proposal problem without changing proof
gates. A separate rejected exterior-witness trial using the existing outward
f64 orbit ball accepted only 16 of 162 nodes. Its failure supplies no exterior
classification for the other 146 nodes; it is not part of the retained patch.
`experiments/geometry-145-f64-exterior-rejected.patch` reproduces that failing
trial when applied after the cover patch; its all-node test is expected to fail.

Still required: effective exterior upper witnesses, complete scalar error
intervals, coherent runtime integration, global nearest-feature/cut handling,
and the original crossing, inward convergence, energy, live training and query
cost validation. The lower-error windows here alone do not certify scalar
distance estimates within those error windows.

Apply `experiments/geometry-145-f64-cover.patch` after the f64-branch patch and its
documented prerequisite chain. Build the reconstructed worktree, then run:

```powershell
rtk cargo test --release --jobs 1 --manifest-path tmp/geometry145-runtime-integration/runtime-core/Cargo.toml --target-dir target/geometry145-runtime-integration geometry_provider::
.venv/Scripts/python.exe docs/validation/experiments/geometry-145-generic-component-check.py --source-root tmp/geometry145-runtime-integration --target-dir target/geometry145-runtime-integration --output tmp/geometry-145-f64-cover-reference.json --emit-cover
.venv/Scripts/python.exe docs/validation/experiments/geometry-145-branch-hp-check.py --input tmp/geometry-145-f64-cover-reference.json --target-dir target/geometry145-precision --output tmp/geometry-145-f64-cover-independent.json
```

The last command uses the already-built `astro-float` dependency of the existing
point-proof experiment. The compact evidence JSON omits the 4,511 emitted cover
domain/target rows, records their stream and input hashes, and retains terminal
validation results. `--emit-cover` recreates the full independent-checker input;
both harnesses modify temporary copies only.
