# Certified point membership experiment

This experiment is not adopted by the production geometry provider. Original crossing tests and all acceptance thresholds are unchanged.

The retained-node investigation found one contour-stability blocker at `(0.3204946517944336, 0.04734092950820923)`. It survives 16,384 updates but escapes at update 32,747. The budget-aware field query still timed out after 1,200.28 seconds. Increasing proof budgets alone has not solved the crossing refusal or query cost.

The isolated Rust crate uses pinned AstroFloat 0.9.6 with directed component bounds and propagated roundoff disks. It proves this exterior escape at 128, 256 and 512 bits. Strict trapping disks prove that the recorded native query is interior with period 85 at prefix 1,024, and the Python crossing query is interior with period 308 at prefix 2,048. These certificates concern the exact mathematical parameter supplied as binary64 coordinates. They do not certify equivalence to the canonical machine orbit.

Four required tests pass. They cover strict pre-update budgets, radius equality, neutral boundary points, successful interior traps, invalid requests, and required outcomes for the three recorded points. Three mutations are rejected: including the budget endpoint, accepting escape at radius equality, and allowing trapping-disk equality. The initial neutral-point tests did not catch the last mutation; an explicit equality case for disk containment now catches it.

Orbit history retains at most 4,097 centers. Precision is restricted to 128, 256 or 512 bits and the update budget to 32,768. Failed enclosures refuse certification.

Run from the repository root:

```powershell
rtk cargo test --release --jobs 1 --manifest-path docs/validation/experiments/geometry-145-point-proof/Cargo.toml --target-dir target/geometry145-precision
rtk cargo run --release --jobs 1 --manifest-path docs/validation/experiments/geometry-145-point-proof/Cargo.toml --target-dir target/geometry145-precision -- proof
.venv/Scripts/python.exe docs/validation/experiments/geometry-145-point-proof-mutations.py
```

The mutation script restores the source in a finally block and writes logs under `tmp`. Do not run it concurrently with builds of the same experimental crate.

Remaining work is to obtain tight certified distance intervals at the difficult nodes, feed those intervals into one coherent distance/gradient/Hessian representation, and validate the original crossings, live training and query cost. Point membership alone does not close these requirements. Runtime adoption also requires dependency and WASM review, version and golden-vector updates, and the full project validation.
