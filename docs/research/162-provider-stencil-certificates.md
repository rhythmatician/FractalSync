# #162/#159: finite-difference jet error contracts

The exact-real central gradient and nine-point Hessian stencil now have checked
error bounds against the actual derivatives of one field. The bounds combine
analytic truncation, sampled-value error, and an explicitly supplied arithmetic
error. They accept the actual positive query step, including a step selected by
clamping. They do not certify the runtime raster, its `Regular` flag, or IEEE
operations without additional evidence.

Source: [ProviderCertificates.lean](../../formal/FractalSync/ProviderCertificates.lean).
The module contains 25 explicit proved theorem declarations.
This extends the [computed-scale Map bounds](162-normalized-distance-fields.md)
and supplies a derivative-level contract relevant to
[continuous mechanics](159-differential-mechanics.md).

## 1. Actual local derivative hypotheses

Let \(f:E\to\mathbb R\) on a real normed space. On a convex region \(S\)
containing the center \(c\) and every query used by a stencil, assume actual
Fréchet derivatives
\[
df_z=D_z,\qquad dD_z=H_z.
\]
For a nonnegative constant \(L\), assume the anchored Hessian variation bound
\[
\|H_z-H_c\|\le L\|z-c\|,\qquad z\in S.
\tag{1}
\]
The norm of \(H_z\) is the operator norm of the continuous bilinear map written
as a continuous linear map into the space of continuous linear maps.
An actual Lipschitz Hessian on the region is a sufficient condition for (1).
No fourth derivative is needed. The `LocalJetControl` structure records these
analytic hypotheses; constructing one requires proofs of them.

For the mixed derivative, \(S\) is additionally a neighborhood of \(c\).
The theorem `LocalJetControl.hessian_symm` then derives
\(H_c(u,v)=H_c(v,u)\) from Mathlib's actual second-derivative symmetry theorem.
Symmetry is not inferred from a numerical matrix being stored symmetrically.

The main analytic lemma proves the second-order Taylor remainder
\[
\boxed{
|f(c+w)-f(c)-D_cw-\tfrac12H_c(w,w)|
\le\frac L6\|w\|^3.
}
\tag{2}
\]
Only the radial query segment is needed for the underlying lemma. The convex
region packages the segment requirements for multiple stencil queries.

The proof first derives a vector-valued first-order remainder for the actual
differential field, with coefficient \(1/2\). It then differentiates the
second-order scalar remainder along \(c+tw\), bounds its derivative using
that first result, and applies the mean-value boundary estimate to the cubic
majorant. Thus (2) is proved from derivative data, not assumed as a provider
remainder certificate.

## 2. Actual stencil truncation bounds

For \(h>0\), define
\[
G_h(u)=\frac{f(c+hu)-f(c-hu)}{2h},
\]
\[
H_h(u,u)=\frac{f(c+hu)-2f(c)+f(c-hu)}{h^2},
\]
and
\[
H_h(u,v)=
\frac{f(c+h(u+v))-f(c+h(u-v))
-f(c-h(u-v))+f(c-h(u+v))}{4h^2}.
\]
These are the actual functions `centralFirst`, `centralSecond`, and
`centralMixed`. The field is sampled at those arguments; no abstract numerical
derivative operator is substituted.

The checked bounds are
\[
\boxed{|G_h(u)-D_cu|\le\frac L6h^2\|u\|^3,}
\tag{3}
\]
\[
\boxed{|H_h(u,u)-H_c(u,u)|\le\frac L3h\|u\|^3,}
\tag{4}
\]
\[
\boxed{
|H_h(u,v)-H_c(u,v)|
\le\frac L{12}h\bigl(\|u+v\|^3+\|u-v\|^3\bigr).
}
\tag{5}
\]
The proofs expand the true Taylor polynomial at all queries and bound the
remaining signed sum using (2). For two orthonormal coordinate directions,
(5) specializes to \(\sqrt2 Lh/3\).

The gradient bound is quadratic in \(h\); the Hessian bounds are linear under
the stated Lipschitz-Hessian regularity. A quadratic Hessian truncation bound
would require stronger regularity or additional cancellation control. It is
not claimed here.

## 3. Sampled-value errors and arithmetic errors

Let the values used by the exact stencil be \(\widehat f=f+e\). No smoothness
or Lipschitz property is required of \(e\). Linearity of each actual stencil
gives correlated error bounds:

| Stencil | Required error combination | Resulting stencil error |
| --- | --- | --- |
| Central gradient | \(\lvert e(c+hu)-e(c-hu)\rvert\le\eta_g\) | \(\eta_g/(2h)\) |
| Diagonal Hessian | \(\lvert e(c+hu)-2e(c)+e(c-hu)\rvert\le\eta_d\) | \(\eta_d/h^2\) |
| Mixed Hessian | Absolute four-corner signed error sum at most \(\eta_m\) | \(\eta_m/(4h^2)\) |

Separate endpoint bounds also have checked corollaries. The diagonal numerator
budget is \(\eta_++2\eta_0+\eta_-\); the mixed numerator budget is the sum of
the four corner budgets. Correlated certificates can be tighter than these
independent absolute-value sums. A common additive bias cancels in each
stencil's signed numerator.

Each `*_total_error` theorem also accepts an explicit premise
\[
|\text{computed component}-\text{exact stencil of }\widehat f|\le\tau.
\tag{6}
\]
This is a separate arithmetic certificate. The theorem does not manufacture
\(\tau\) from the fact that the Rust calculation uses `f64`.

For unit coordinate directions, uniform query-value error \(|e|\le\eta\),
and arithmetic component budgets \(\tau_g,\tau_d,\tau_m\), the checked
theorems imply the following specializations:

| Component | Total error bound |
| --- | --- |
| Gradient coordinate | \(Lh^2/6+\eta/h+\tau_g\) |
| Diagonal Hessian coordinate | \(Lh/3+4\eta/h^2+\tau_d\) |
| Mixed Hessian coordinate for orthonormal axes | \(\sqrt2 Lh/3+\eta/h^2+\tau_m\) |

These formulas expose why reducing a stencil step can increase derivative
error when the sampled values have a fixed error floor. Choosing the smallest
permitted step is not justified by truncation analysis alone.

The generic `euclidean_error_bound` assembles component bounds into Mathlib's
actual Euclidean norm:
\[
|x_i-y_i|\le b_i\quad\Longrightarrow\quad
\|x-y\|_2\le\sqrt{\sum_i b_i^2}.
\tag{7}
\]
A product index gives the Frobenius norm of matrix entries. For the symmetric
two-dimensional Hessian with diagonal budget \(B_d\) and mixed budget \(B_m\),
the resulting bound is \(\sqrt{2B_d^2+2B_m^2}\). These are component-coordinate
and Frobenius guarantees; a separate Riesz-gradient or matrix-operator API is
not packaged here.

## 4. Perturbed query coordinates and denominators

`sampled_value_error` provides a way to construct the per-query value budget.
If \(f\) is \(K\)-Lipschitz, the nominal query is \(z\), the actual query is
\(\widehat z\), and
\[
|\widehat v-f(\widehat z)|\le\eta,
\qquad\|\widehat z-z\|\le\delta,
\]
then
\[
\boxed{|\widehat v-f(z)|\le\eta+K\delta.}
\tag{8}
\]
Thus position-rounding error can enter the value-error part of the stencil
contract, provided an actual query displacement bound is available.

`division_perturbation_bound` separately proves, for positive \(d,\widehat d\),
\[
|\widehat n-n|\le\eta_n,\qquad
|\widehat d-d|\le\eta_d
\quad\Longrightarrow\quad
\boxed{
\left|\frac{\widehat n}{\widehat d}-\frac nd\right|
\le\frac{\eta_n}{\widehat d}
+\frac{|n|\eta_d}{d\widehat d}.
}
\tag{9}
\]
This can bound the effect of rounded numerator and denominator construction.
The final division's own rounding error must also be included in (6). No IEEE
rounding-mode, subnormal, overflow, or non-finite-value theorem is asserted.

## 5. Runtime correspondence and remaining evidence

The inspected implementation is
[`runtime-core/src/geometry_provider.rs`](../../runtime-core/src/geometry_provider.rs).
For the raster bridge it:

1. Samples the center value used to compute the ruler.
2. Sets the derivative step to the requested scale clamped between the configured
   minimum and maximum.
3. Samples the two points on each axis and four corner points.
4. Evaluates exactly the central-gradient, diagonal-Hessian, and mixed-Hessian
   formulas above, using values returned as `f32` and arithmetic in `f64`.

The real-valued theorem parameter \(h\) must be the actual selected step, not
the unclamped request. The samples and regularity region must cover the queries
at that step. The center used in the Hessian formula is the earlier center
sample `d0`; the code does not use `vals[0]` in its derivative formulas.

These stencil hypotheses are not established for the actual raster by this
module. The separate [concrete operational-field proofs](162-concrete-provider-certificates.md)
now identify the experimental quintic cell's actual real derivatives and
bound its coefficient sensitivity. Those results concern the polynomial
represented by stored coefficients, and do not certify its reference-distance
fidelity. In particular:

- The raster's interpolation error against true signed Shore distance is not
  certified. Raster interpolation need not have the required differentiability.
- A cut-locus point need not admit the assumed actual Hessian. A positive ruler
  does not supply this missing derivative.
- Eikonal and Hessian-size flags are diagnostics, not proofs of (1) or of query
  value accuracy.
- The operational field's exact derivatives are derivatives of that operational
  field. Its `reference-distance-unknown` contract still does not certify
  agreement with true Shore distance.
- Floating query construction, sample conversion, arithmetic, and final division
  require numerical certificates before the real inequalities imply executable
  error guarantees.

The new results identify concrete sufficient evidence for derivative fidelity.
They do not change the runtime provider contract, canonical parameters, training
requirements, or validity classifications.

The targeted command `lake build --wfail FractalSync.ProviderCertificates`
passes under the repository's pinned Lean 4.24.0 and Mathlib environment.
Package-wide build and axiom-audit results are recorded in
[verification.md](verification.md).
