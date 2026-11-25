# #160: curvature in a curved normal-coordinate metric

The curved-Shore normal-coordinate model now has a checked connection and an
actual curvature-commutator calculation. The formal result covers a varying
curvature profile, at Shore and away from it, subject to explicit local
derivative and nondegeneracy hypotheses. It does not construct a normal chart
of an arbitrary embedded Shore.

Sources:
[CurvedShoreGeometry.lean](../../formal/FractalSync/CurvedShoreGeometry.lean)
(19 theorem declarations) and
[CurvedShoreCurvature.lean](../../formal/FractalSync/CurvedShoreCurvature.lean)
(5 theorem declarations). The
[flat-Shore note](160-flat-shore.md) retains the separate classical arguments
for embedded Shores, the global cut locus, and curved-Shore bifurcation.

## 1. Coordinate model and its exact domain

Use coordinates \((r,y)\), where the intended classical interpretation is signed
normal distance and Euclidean arclength along Shore. Let \(k(y)\) be the supplied
signed Euclidean curvature profile, with the normal-orientation convention
\[
h(r,y)=1-k(y)r.
\]
For \(\epsilon>0\), \(a\in\mathbb R\), put
\[
\rho^2=\epsilon^2+r^2,\qquad b^2=1+a^2,\qquad
Q^2=\epsilon^2+b^2r^2.
\]
The actual bilinear metric defined in Lean is
\[
\boxed{g=A(r)\,dr^2+B(r)h(r,y)^2\,dy^2,
\qquad A=\frac{Q^2}{\rho^4},\quad B=\frac1{\rho^2}.}
\tag{1}
\]

`metric_eq_flat_pullback` proves the pointwise bilinear identity
\[
g(v,w)=g_{\rm flat}\bigl((v_r,hv_y),(w_r,hw_y)\bigr).
\tag{2}
\]
This is a change of tangent components at a point. It is not a claimed
coordinate diffeomorphism or a local isometry to the flat-Shore surface.
The metric is symmetric and positive definite whenever \(h\ne0\). At \(h=0\)
the tangential coefficient vanishes, so the Riemannian interpretation is
excluded. The formal positivity theorem permits either sign of \(h\); a
classical tubular chart ordinarily uses the connected region with \(h>0\).

For arbitrary coordinate vectors \(v,w\), Lean also proves the Gram identity
\[
\boxed{
g(v,v)g(w,w)-g(v,w)^2
=A B h^2(v_rw_y-v_yw_r)^2.
}
\tag{3}
\]
Thus a nondegenerate coordinate frame has positive Gram denominator under
the same metric hypotheses. This algebraic frame identity is not a formal
coordinate-invariance theorem for the curvature tensor.

When \(k=0\), the bilinear metric reduces exactly to the flat model. The
global product norm on \(\mathbb R\times\mathbb R\) is used for ordinary
Fréchet differentiation; the physical metric is explicitly (1).

## 2. Actual connection and metric compatibility

The nonzero Christoffel coefficients are
\[
\Gamma^r_{rr}=\frac{b^2r}{Q^2}-\frac{2r}{\rho^2},
\qquad
\Gamma^r_{yy}=\frac{h(r+k\epsilon^2)}{Q^2},
\]
\[
\Gamma^y_{ry}=\Gamma^y_{yr}
=-\frac r{\rho^2}-\frac{k}{h},
\qquad
\Gamma^y_{yy}=-\frac{r k'}h.
\tag{4}
\]
Here \(k,k'\) are evaluated at \(y\). The \(k'\) term is retained for a
varying Shore curvature; it is not silently treated as a constant-curvature
model.

The covariant derivative uses the actual Fréchet derivative:
\[
\nabla_XY=DY[X]+\Gamma(X,Y).
\]
The package checks pointwise linearity in the first field, additivity and
the Leibniz rule in the differentiated field, and zero torsion relative to
Mathlib's actual vector-field Lie bracket. It also differentiates the actual
metric pairing along a curve and proves
\[
\frac d{dt}g(Y,Z)
=g(\nabla_XY,Z)+g(Y,\nabla_XZ)
\]
for a curve tangent to \(X\), with the stated field differentiability,
\(\epsilon>0\), and \(h\ne0\). This proves actual metric compatibility,
not only a polynomial identity in nominal metric derivatives.

At Shore the connection simplifies to
\[
\boxed{
\Gamma(v,w)\big|_{r=0}
=\bigl(kv_yw_y,-k(v_rw_y+v_yw_r)\bigr).
}
\tag{5}
\]
In particular the tangential connection term generally does not vanish on a
curved Shore. Curvature \(+1\) at a Shore point does not imply that the Shore
itself is geodesic.

## 3. The full curvature commutator

The definition uses
\[
R(X,Y)Z=\nabla_X\nabla_YZ-\nabla_Y\nabla_XZ-\nabla_{[X,Y]}Z.
\tag{6}
\]
For the coordinate directions \(e_r,e_y\), the formal proof differentiates
both nonconstant intermediate fields. At an arbitrary point \((r,y)\), set
\[
D_{yy}=
\frac{(1-2kr-k^2\epsilon^2)Q^2
-2b^2rh(r+k\epsilon^2)}{Q^4}.
\]
The two actual directional evaluations of their Fréchet derivatives are
\[
\partial_r(\Gamma^r_{yy},\Gamma^y_{yy})
=\left(D_{yy},-\frac{k'}{h^2}\right),
\qquad
\partial_y(0,\Gamma^y_{ry})
=\left(0,-\frac{k'}{h^2}\right).
\tag{7}
\]
The tangential derivative terms cancel. The tangential connection-product
terms cancel as well, giving the checked vector identity
\[
\boxed{
R(e_r,e_y)e_y
=\bigl(D_{yy}+\Gamma^r_{rr}\Gamma^r_{yy}
-\Gamma^r_{yy}\Gamma^y_{ry},0\bigr).
}
\tag{8}
\]

The exact hypotheses are \(\epsilon>0\), \(1-k(y)r\ne0\), and actual
derivatives `HasDerivAt k dk y` and `HasDerivAt (deriv k) ddk y`.
A locally \(C^2\) curvature profile is sufficient. Although \(k''\) does not
appear in (7), its local derivative hypothesis supplies the actual full
Fréchet derivative of the intermediate vector field containing \(k'\).
The proof does not discard that hypothesis merely because the final formula
cancels the derivative terms.

## 4. Exact curvature and its normal slope

The coordinate sectional curvature is the actual metric Gram ratio
\[
K(r,y)=\frac{g(R(e_r,e_y)e_y,e_r)}
{g(e_r,e_r)g(e_y,e_y)-g(e_r,e_y)^2}.
\]
Lean proves
\[
\boxed{
K(r,y)=
\frac{\epsilon^4-b^2r^4}{Q^4}
-\frac{k(y)b^2r\rho^4}{(1-k(y)r)Q^4}.
}
\tag{9}
\]
The first term is the already checked flat curvature. The second is the
exact correction for the supplied curved normal chart. Its dependence is
on the value \(k(y)\), while the derivatives of \(k\) cancel in the actual
commutator. The proof applies to varying \(k\), not only to circular Shores.

At \(r=0\), the actual commutator is
\[
R(e_r,e_y)e_y=(\epsilon^{-2},0),
\]
and the checked Gram ratio gives
\[
\boxed{K(0,y)=1.}
\tag{10}
\]
The package further differentiates the actual curvature function in the normal
coordinate and proves
\[
\boxed{\partial_rK(0,y)=-(1+a^2)k(y).}
\tag{11}
\]
This is `hasDerivAt_sectionalCurvature_normal_on_shore`. It first differentiates
the rational expression, then transfers the derivative through eventual
equality with the actual commutator-defined curvature. Continuity of
\(1-k(y)r\) supplies the required nonzero neighborhood of \(r=0\).

For a metric-unit normal velocity at Shore, \(dr/ds=\epsilon\), so the
corresponding slope is \(-\epsilon(1+a^2)k(y)\). This last conversion is an
immediate conventional chain-rule consequence, not a separately packaged
Lean arclength theorem. The local normal asymmetry does not by itself prove
the curved-Shore bifurcation or classify global minimizing paths.

## 5. Embedded Shore and regularity boundary

The new formal objects are defined directly in a supplied normal-coordinate
model. To apply them to a particular planar embedded Shore, one must still
construct a tubular chart, prove that its signed coordinate is actual signed
distance, identify the Euclidean metric factor \(1-k(y)r\), and transfer the
geometry through that chart. The condition \(1-k(y)r\ne0\) alone does not prove
chart injectivity, nearest-point uniqueness, or the distance identification.

The earlier [classical smooth-Shore argument](160-flat-shore.md#2-universal-curvature-at-a-smooth-shore)
assumes a tubular neighborhood with a \(C^3\) signed-distance function and
proves curvature \(+1\) at Shore through graph and conformal curvature
formulas. That embedded statement remains conventional. Its sufficient
regularity assumption should not be silently equated with the particular
local \(k,k'\) derivative hypotheses used by this coordinate proof. No
statement extends to fractal Shore points or cut loci without the required
geometry and regularity.

The remaining obligations include a bundled curvature tensor and naturality
under general nonlinear chart changes, the embedded tubular-chart construction,
smooth-minimizer ODE identification, global cut-locus arguments, and the functional-analytic
curved-Shore bifurcation proof. The new coordinate curvature calculation is
independent of those unresolved steps.

No runtime behavior, provider contract, canonical parameter, or training gate
is changed. Package-wide verification is recorded in
[verification.md](verification.md).
