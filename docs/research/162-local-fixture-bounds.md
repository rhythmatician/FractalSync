# #162: local jets of the concrete quadratic fixture

[ConcreteLocalFieldBounds.lean](../../formal/FractalSync/ConcreteLocalFieldBounds.lean)
adds 27 proved declarations for the same literal 16 by 16 coefficient table
used in the [concrete asset instance](162-concrete-asset-instance.md). It proves
local bounds for the actual reconstructed field, its Euclidean differential,
its Euclidean Hessian, and Hessian variation. The proof retains the polynomial
reference and the finite sample-error certificate instead of estimating every
derivative from the largest coefficient in the asset.

## 1. Fixed data and local conclusions

The spacing, asset, and configuration are unchanged:

\[
h=\frac{3602879701896397}{36028797018963968},\qquad
x_0=(1/2,1/2),\qquad R=1/16.
\]

Let \(f\) be the actual floor-selected field, pulled back to the Euclidean
plane through the existing global coordinate map. On the closed Euclidean
ball \(B=\overline B(x_0,R)\), the checked estimates are

| Quantity | Checked bound | Lean declaration |
| --- | --- | --- |
| Field value | \(\lvert f(x)\rvert\le1/4\) | `local_field_abs_le` |
| Actual differential | \(\|Df(x)\|\le4\) | `local_differential_norm_le` |
| Actual bilinear Hessian | \(\|D^2f(x)\|\le12\) | `local_hessian_norm_le` |
| Hessian variation | \(\|D^2f(x)-D^2f(y)\|\le\|x-y\|\) | `local_hessian_sub_le` |

Both points in the last row may lie on grid lines, at grid vertices, or in
different selected cells. `closedBall_openHalo` proves that the entire ball
lies in the strict physical halo of the finite asset. No new asset, reference,
or derivative premise is supplied by a caller.

The named jet functions are `euclideanDifferential` and
`euclideanHessianForm`. Their identification with actual Fréchet derivatives
comes from the existing global spline correspondence and
`hasFDerivAt_euclideanDifferential`. The estimates therefore concern derivatives
of the reconstructed function, including across knots. They are not bounds on
independently specified provider outputs.

The Euclidean qualification matters. The ordinary coordinate product
\(\mathbb R\times\mathbb R\) carries the maximum norm, whereas `Plane` carries
the \(L^2\) norm. The pullback into the coordinate product is contractive. The
new bounds explicitly use the Euclidean dual norm and do not identify those
two norms.

## 2. Exact cancellation in Hessian variation

The ideal reference and the already certified finite coefficient error are

\[
P(x,y)=x^2+2xy-3y^2,\qquad
\eta=2^{-48},\qquad
|C_{ij}-P(ih,jh)|\le\eta
\]

for all 256 stored entries. Ideal quadratic samples reconstruct
\(P-h^2\), so their actual Hessian is the constant matrix

\[
H_P=\begin{pmatrix}2&2\\2&-6\end{pmatrix}.
\]

Define an auxiliary lattice \(E\) by extending the finite errors by zero:

\[
E_{ij}=\begin{cases}
C_{ij}-P(ih,jh),&0\le i,j<16,\\
0,&\text{otherwise}.
\end{cases}
\]

`errorCoefficients_bound` proves the global bound \(|E_{ij}|\le\eta\).
The auxiliary extension supplies an everywhere defined spline to which the
existing global Hessian query bound applies. The selected stencil uses only
stored entries throughout the physical halo, so exact jet linearity gives

\[
D^2f_E(q)=D^2f_C(q)-H_P
\]

there. This is `physicalHessian_error`; its proof uses actual selected-jet
correspondence, exact quadratic reproduction, and finite-asset extension
independence. Consequently `physicalHessian_sub_le` proves the stronger bound

\[
\boxed{
\|D^2f_C(q')-D^2f_C(q)\|
\le \frac{32\eta}{h^3}\,\|q'-q\|
}
\]

for every pair of coordinate queries in the full physical halo. The norms in
this displayed coordinate statement are the coordinate maximum norm and its
induced operator norm. Pulling back both Hessian arguments to the Euclidean
plane can only reduce the bound. `third_error_gain` checks
\(32\eta/h^3\le1\), yielding the simple local Euclidean constant in the table.
The more precise coefficient remains available in the preceding theorem.

This cancellation is why Hessian variation becomes small. Applying the global
third-order query gain directly to the coefficient magnitude \(7\) gives
\(32\cdot7/h^3\). Applying it to the bounded error lattice gives
\(32\eta/h^3\); the quadratic Hessian contributes no variation. No unproved
third-derivative bound and no single-cell assumption enter this argument.

The identity with the finite error lattice is asserted only on the valid halo.
The zero extension is a proof device, not a claim that the original finite
asset globally equals its polynomial reference.

## 3. Local value and norm estimates

The module proves the Euclidean covector identity

\[
\|a\,dx+b\,dy\|^2=a^2+b^2
\]

using Riesz duality, then obtains \(\|H_P\|\le8\) by an explicit quadratic
inequality. This estimate is sufficient; optimality of the matrix norm is not
required. The derivative at the center is \((2,-2)\), with norm at most \(3\).
The actual derivative of \(DP\) is the constant Hessian, so the convex mean
value theorem gives

\[
\|DP(x)\|\le3+8R=7/2\qquad(x\in B).
\]

Combining this with the existing Euclidean pullback of the derivative error
\(2^{-43}\) proves \(\|Df\|\le4\). Combining the reference Hessian bound with
the existing Hessian error \(2^{-38}\) proves \(\|D^2f\|\le12\).

For the value estimate, \(P(x_0)=0\) and the same convex mean value theorem give

\[
|P(x)|\le(7/2)R=7/32.
\]

The reconstruction bias is retained explicitly:

\[
|f(x)|\le |P(x)|+h^2+\eta.
\]

Exact rational arithmetic verifies \(h^2+\eta\le1/32\), which closes the
bound \(|f(x)|\le1/4\). Omitting the quadratic reconstruction bias would
not justify this conclusion from the existing sample certificate.

## 4. Use and remaining boundary

These raw field estimates are inputs to the local ruler and mechanical
certificate in [the improved horizon note](159-local-horizon.md). Convex mean
value estimates also give field and differential Lipschitz constants \(4\)
and \(12\) on the same ball. The integration module packages those results
with the norm bounds; it does not require a stronger finite-data premise.

The results concern exact-real reconstruction of the existing literal fixture.
They do not certify production distance assets, IEEE query arithmetic, machine
indexing, masks, compiled extraction, or numerical evaluation errors. They also
do not turn the quadratic fixture into a signed-distance field. The provenance
and executable correspondence boundaries remain those stated in the
[concrete asset note](162-concrete-asset-instance.md).

Validation of this module: `lake build --wfail
FractalSync.ConcreteLocalFieldBounds` passed with 2,645 Lake jobs and no warnings.
The parent increment runs the complete package build and transitive axiom
audit after importing the new modules.
