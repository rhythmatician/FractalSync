# #162: a concrete finite operational-field instance

[ConcreteOperationalAsset.lean](../../formal/FractalSync/ConcreteOperationalAsset.lean)
instantiates the finite-asset premises with a literal nonconstant 16 by 16 test
fixture. Its 23 theorem declarations check the stored coefficient bound,
sample-reference error, physical halo margin, initial phase data, and every
admitted mathematical query address. These are exact-real statements about a
specific table, rather than hypotheses about an unspecified asset.

[ConcreteAssetFidelity.lean](../../formal/FractalSync/ConcreteAssetFidelity.lean)
adds eleven declarations connecting those literal samples to the actual
reconstructed value, differential, Hessian, and Euclidean gradient throughout
the halo.

## 1. Provenance and exact data

The source is the existing `tests::field` fixture in
[`runtime-core/src/operational_field.rs`](../../runtime-core/src/operational_field.rs)
at commit `57a1650fe5a2d8ce6254434252ecc0ce2e24cdd2`. It constructs coefficients
from the binary64 expression

```rust
let (x, y) = (x as f64 * 0.1, y as f64 * 0.1);
c.push(x * x + 2.0 * x * y - 3.0 * y * y);
```

for integer columns and rows from 0 through 15, then creates a prepared field
with origin `[0.0; 2]`, spacing `0.1`, and shape `[16; 2]`.

No prepared operational coefficient asset is committed in the inspected source
tree. The existing Mandelbrot raster binaries have other preparation and
coordinate obligations. This instance therefore uses the existing quadratic
test fixture. It does not claim production asset fidelity or identification
with Mandelbrot signed distance.

The companion
[`operational-quadratic.json`](../../formal/fixtures/operational-quadratic.json)
contains all 256 binary64 bit patterns, their exact common-denominator integer
representation, the spacing bit pattern, and source-function SHA-256. The
spacing is retained exactly:

\[
h=\frac{3602879701896397}{36028797018963968},
\qquad \operatorname{bits}(h)=\mathtt{0x3fb999999999999a}.
\]

Lean proves

\[
\frac1{10}\le h\le\frac18,
\qquad h>0.
\]

The table uses denominator

\[
2^{57}=144115188075855872.
\]

The generated Lean table is indexed by row, then column. The public
`coefficient i j` and integer-lattice `coefficients i j` use column, then row,
matching the mathematical field interface. `coefficients_of_fin` checks that
these indexing conventions agree. The lattice is zero outside the finite
rectangle, so the global bound used by the existing analytic estimates follows
from actual finite data.

### Extraction evidence and its boundary

[`check_operational_quadratic.py`](../../formal/fixtures/check_operational_quadratic.py)
checks that the source function has the expected text, materializes its
operation sequence as separately rounded, non-fused binary64 multiplication,
addition, and subtraction, and independently checks every arithmetic result
using exact rational round-to-nearest, ties-to-even arithmetic. It also checks
that the JSON and the generated Lean table match the regenerated data.

The check was run successfully. Rust and Cargo were not available in this
environment, so no Rust execution is asserted. The extraction check is external
evidence connecting the literal data to the source expression. It is not a
kernel proof of IEEE arithmetic, Rust compiler behavior, decimal parsing,
compiler contraction or FMA equivalence, machine indexing, or field evaluation.
The Lean results below depend on the
literal integers in the checked module, not on trusting this extraction script.

## 2. Finite magnitude and sample-reference certificates

`numerator_bound` kernel-reduces the finite integer inequality for all 256
entries. `coefficient_bound` transfers it to real coefficients, and
`coefficients_bound` proves

\[
\boxed{|C_{ij}|\le7\qquad(i,j\in\mathbb Z).}
\]

The explicit values

\[
C_{0,0}=0,\qquad C_{15,0}=\frac94,
\qquad C_{0,15}=-\frac{27}{4}
\]

are checked as well. Thus the finite data are neither constant nor zero.

For the quadratic reference

\[
P(x,y)=x^2+2xy-3y^2,
\]

`coefficient_sample_error` proves the following bound for every stored entry:

\[
\boxed{
\left|C_{ij}-P(ih,jh)\right|\le2^{-48},
\qquad 0\le i,j<16.
}
\]

This is an exact rational certificate, including all rounding already present
in the literal table. Its finite integer proof uses ordinary kernel reduction
with `decide`; it uses no native-evaluation axiom. The mathematical nodes here
are exactly \((ih,jh)\), with the exact dyadic spacing above.

The sample bound must be distinguished from a zero-bias reconstruction claim.
The existing `field_quadratic` theorem proves that ideal raw samples of this
quadratic reconstruct \(P-h^2\), owing to the quintic basis second moment.
The new whole-halo fidelity result below uses this biased reference explicitly.

## 3. A concrete phase region with physical margin

Use the actual Euclidean plane for configurations and velocities, and the
product maximum norm for phase space. The chosen data are

| Parameter | Exact value |
| --- | --- |
| Configuration center \(x_0\) | \((1/2,1/2)\) |
| Initial velocity \(v_0\) | \((1/4,0)\) |
| Phase center \(p\) | \((x_0,v_0)\) |
| Phase radius \(R\) | \(1/16\) |
| Admitted coordinate-query displacement \(\delta\) | \(1/64\) |
| Origin | \((0,0)\) |
| Shape | \(16\times16\) |

The following are checked:

\[
\|x_0\|_2^2=\frac12,\qquad
\|x_0\|_2\le\frac34,\qquad
\|v_0\|_2=\frac14.
\]

Every phase state \(z\in\overline B(p,R)\) has
\(\|z_x\|_2\le1\). This supplies the unit-ball configuration premise for a
nonzero secant-wall force estimate without changing the Euclidean norm to the
coordinate maximum norm.

The open physical halo is

\[
(2h,13h)\times(2h,13h).
\]

Its exact four-face margin at \(x_0\) is

\[
\boxed{
m(x_0)=\frac12-2h
=\frac{5404319552844595}{18014398509481984}
>\frac14.
}
\]

In particular,

\[
R+\delta=\frac5{64}<m(x_0).
\]

`safe_reported_query` consequently proves that for every
\(z\in\overline B(p,R)\) and every reported coordinate query \(q\) with

\[
\|q-\operatorname{chart}(z_x)\|_\infty\le\frac1{64},
\]

the query belongs to the open halo. All 36 selected mathematical stencil
indices satisfy \(0\le k_x,k_y<16\), and every row-major address satisfies
\(0\le16k_y+k_x<256\). This includes queries crossing cell boundaries.
The admissible displacement is an input budget; this theorem does not certify
the actual error of a machine query calculation.

## 4. Actual value and derivative fidelity throughout the halo

Write \(F\) for the actual floor-selected physical spline constructed from the
literal table, and define

\[
P_h(q)=P(q)-h^2.
\]

`selected_sample_error` proves that every active stencil entry satisfies the
finite sample certificate at the actual physical node. It uses the halo index
bounds to discharge every stored-data premise and checks the correspondence
between integer lattice nodes and the selected cell's physical sample nodes.
This remains true when the selected cell changes.

The nonnegative value weights and the existing derivative coefficient gains
then yield, for every query in the complete half-open physical halo,

\[
\boxed{|F(q)-P_h(q)|\le2^{-48},}
\]

\[
\boxed{
\|DF(q)-DP_h(q)\|
\le\frac{8}{3h}\,2^{-48}
\le2^{-43},
}
\]

\[
\boxed{
\|D^2F(q)-D^2P_h(q)\|
\le\frac{68}{9h^2}\,2^{-48}
\le2^{-38}.
}
\]

`actual_reference_errors` packages these three statements. They concern actual
Fréchet derivatives of the actual reconstructed field, including grid lines
and vertices, rather than merely the provider's named derivative outputs.
The coordinate derivative norms use the product maximum norm already used by
the physical field interface. The dyadic majorants are checked using the actual
spacing, not a substitution of \(h=1/10\).

`euclidean_gradient_reference_error` also proves the gradient estimate after
pulling both fields into the actual Euclidean plane:

\[
\boxed{
\|\nabla(F\circ\operatorname{chart})(x)
-\nabla(P_h\circ\operatorname{chart})(x)\|_2
\le2^{-43}.
}
\]

The proof uses the contractive Euclidean-to-coordinate chart and the norm
preservation of Riesz dualization. No coordinate maximum norm is silently
substituted for the Euclidean norm.

These bounds establish a concrete reference-fidelity result for this exact
quadratic fixture. They do not identify the fixture or its reference polynomial
with Mandelbrot signed distance. They also do not include rounding introduced
while evaluating a machine query, or the effect of replacing its physical query
by a rounded query.

## 5. Reproduction and remaining boundary

From the repository root, check extraction and literal-table correspondence:

```sh
python formal/fixtures/check_operational_quadratic.py
```

From `formal/`, check this module:

```sh
lake build --wfail FractalSync.ConcreteAssetFidelity
```

Both checks passed. The module's bound and margin lemmas supply the
[finite-asset trajectory certificate](159-operational-trajectory-certificates.md).
The [concrete catalog and trajectory instance](159-concrete-trajectory-certificate.md)
now discharges those data and force premises, constructs an actual solution at
an explicit horizon, and certifies an exact-real 1024-step run. The horizon
remains extremely conservative because it uses global coefficient majorants.

This fixture does not settle production asset/reference fidelity, provider
query rounding, storage or mask correspondence, IEEE evaluation errors, or
runtime integrator behavior. Runtime authority, training, and canonical
parameters are unchanged.
