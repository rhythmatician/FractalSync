# Binary64 error on one retained operational-field cell

This certificate covers the immutable experimental field `9721bb7647a0b8e6`
used in the retained fine replay. It bounds the compiled evaluator's value,
gradient and Hessian against the exact polynomial of the **stored** binary64
coefficients. The preparation recipe used a finite-orbit contour. True-Shore
distance and derivative fidelity remain unknown. The operational engine stays
opt-in. [Replay evidence and preparation](../validation/bounded-operational-engine.md)

## Patch and scale allocation

The patch is the closed square `||p − c||∞ ≤ s`, with
`c = (9839/32768, 1647/32768)` and `s = 1/16384`. It lies beside the recorded
near-Shore replay start `(0.3, 0.05)`, inside cell `(2604, 1504)`. The claim
uses the exact real value of each binary64 coordinate passed to the provider.
Any error in generating those coordinates for a Map window belongs in a
separate allocation. [Replay coordinates](../validation/bounded-operational-engine-evidence.json)

Decoding `FSOP0001` gives a `3425 × 2825` row-major array with no availability
mask. All 36 stencil coefficients in columns `2602..2607` and rows
`1502..1507` have magnitude below `0.019`. Both exact and machine-normalized
coordinates stay strictly inside the same unit cell throughout the closed
patch. The exact checker verifies the strict inequalities after expanding
by the coordinate arithmetic error. The selected indices are exact binary64
integers, fit in `usize` on the checked target and select the same addresses
as the real query. The final local subtraction is exact by Sterbenz's lemma.
[`PreparedField::query`](../../runtime-core/src/operational_field.rs)

For `M(p) = (F(p) − F(c))/s`, where `F` is the literal spline, two scalar
errors of size `e₀` contribute at most `2e₀/s`. The checker also charges the
final subtraction and division in a two-query evaluation of `M`. Gradient
and Hessian errors are converted from components using the maximum absolute
gradient entry and maximum absolute Hessian row sum. In the normalized
coordinates `u = (p − c)/s`, the corresponding reported jet errors are `e₁`
and `s·e₂`. All three dimensionless arithmetic allocations fit below `10⁻⁸`.
This is an illustrative local tolerance for the operational field. A reference-Map
acceptance verdict needs independent true-Shore reference evidence.
[#167 scale-relative budget](https://github.com/rhythmatician/FractalSync/issues/167)

| Quantity | Regional bound rounded upward | Maximum observed over the 81 jet samples |
| --- | ---: | ---: |
| Value | `1.372e-13` | `6.701e-17` |
| Gradient, maximum component | `5.657e-10` | `7.511e-15` |
| Hessian, maximum absolute row sum | `4.683e-6` | ≤ `2.814e-11` |
| Normalized value, including final arithmetic | `4.496e-9` | |
| Scaled Hessian | `2.859e-10` | |

## Regional calculation

For each real operation let `B` bound the magnitude of its exact result and
`P` bound the difference between that result and the operation applied in
exact arithmetic to its machine operands. Use separately rounded binary64,
round-to-nearest ties-to-even, with gradual underflow and exact literal
decoding. Given `B + P < 2¹⁰²³`, rounding contributes at most
`2⁻⁵³(B + P) + 2⁻⁸⁰`, where `2⁻⁸⁰` conservatively covers the subnormal rounding
allowance. The error after the operation is bounded by the sum of `P` and
that contribution. The checker verifies finiteness and division floors in
exact rational arithmetic.

For machine operand errors `E₁, E₂` and exact magnitudes `B₁, B₂`, addition
and subtraction use `P = E₁ + E₂`. Multiplication uses
`P = E₁(B₂ + E₂) + B₁E₂`. For division by exact target constant `d`, with
machine denominator error `E₂ < |d|`, use
`P = (E₁ + (B₁/|d|)E₂)/(|d| − E₂)`. This includes the rounded `h*h` used
as the Hessian divisor. The Horner recurrence uses exact doubling of finite
binary64 values in its proven range. The rest of the calculation follows the
source order through the two bases, the two products per coefficient, the
row-major sum and physical derivative scaling. The arithmetic analysis is
conditional on those compiler and floating-point assumptions and covers every
binary64 input in the stated patch. The checker fingerprints the Rust query
and basis source so a changed evaluation order prompts a new calculation.
[`check_operational_patch.py`](../../formal/fixtures/check_operational_patch.py)

The basis and physical jet correspondence for the polynomial in exact
arithmetic is already checked in Lean. Here the rational reference evaluator
uses differentiated monomials, independent of the runtime Horner recurrence.
The numerical envelope is a conventional forward error argument evaluated
with exact rationals. The compiled comparison samples the patch and checks
the round-trip asset bytes, field ID, and unknown reference-distance status.
[Exact-real jet and address correspondence](162-global-operational-field.md)

## Reproduce

Use the retained fine asset with SHA-256
`bc2604771c2cc48fb71cda6cb22c6e4338a776b6591fa923dd065977817ad9fb`.
The asset is generated in a separate checkout at source `e780e795` using the
preparation command in the replay notes. The checker rejects a different
checksum. [`bounded-operational-engine.md`](../validation/bounded-operational-engine.md)

```powershell
.venv/Scripts/python.exe formal/fixtures/check_operational_patch.py tmp/geometry145-retained-fine.bin
```

The script builds a temporary release-mode Cargo probe against the current
checkout's `runtime_core` with `experimental-operational-field`. It was run
with `--offline` using Rust `1.98.0` on `x86_64-pc-windows-msvc`, source
`c9ea382bf1f620f72f0f8b4bd19583a06d56f66a`. The probe compares all six independent
jet components at 81 exact dyadic points, including the patch edges and
corners. The frozen-engine probe uses a default manifold configuration with
the archived reference-refusal policy. It checks the scalar sent through
`GeometryProvider`, the unknown reference status, and refusal at the canonical
cut point `(-1, 0)`. The sample results establish compiled correspondence at
the chosen points. The uniform bound comes from the regional arithmetic
argument and its assumptions.
