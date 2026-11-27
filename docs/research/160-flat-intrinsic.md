# #160: the Euclidean flat-Shore metric and the coordinate integral

The explicit flat coordinate metric is now connected to the general
AC graph-length and path-infimum construction in Lean. This is a statement
about the full ordinary absolutely continuous path class, including paths
that cross Shore, have distance-trace cusps, or rest on Shore.

Source: [FlatShoreIntrinsic.lean](../../formal/FractalSync/FlatShoreIntrinsic.lean).
The module adds fifteen proved theorem declarations. It uses the existing
[coordinate metric and connection](160-flat-shore.md) and
[metric-length and attainment package](160-intrinsic-length.md).

## 1. The ambient norm is Euclidean

The ambient plane is `WithLp 2 (ℝ × ℝ)`, with
\[
\|(r,y)\|^2=r^2+y^2.
\]
The flat Shore is the actual subset
\[
S=\{(r,y):r=0\}.
\]
`infDist_shore` proves, for every point in this plane,
\[
\boxed{\operatorname{infDist}((r,y),S)=|r|.}
\]
The upper bound uses the actual projection \((0,y)\in S\). The lower bound
uses the norm bound for the first coordinate against every Shore point.
Thus the proof uses Mathlib's actual infimum distance, not a supplied
signed-distance oracle.

A continuous linear equivalence `chart` maps this Euclidean plane to the
ordinary coordinate product used by `FlatShoreGeometry`. This chart is not
claimed to be an isometry for the product's default norm. Both it and its
inverse are Lipschitz, which proves equivalence of the full AC curve classes.
Actual differentiation through the chart identifies coordinate velocities.

## 2. Equality of the densities, including Shore crossings

Fix \(\epsilon>0\) and any \(a\in\mathbb R\), and put
\[
A(r)=\frac{\epsilon^2+(1+a^2)r^2}{(\epsilon^2+r^2)^2},
\qquad B(r)=\frac1{\epsilon^2+r^2}.
\]
The explicit metric is
\[
g=A(r)\,dr^2+B(r)\,dy^2.
\]
For an AC Euclidean path \(\gamma\), write its signed normal coordinate
as \(r(t)\), its tangential coordinate as \(y(t)\), and its actual distance
trace as \(d(t)=\operatorname{infDist}(\gamma(t),S)=|r(t)|\).
The unrestricted graph density is
\[
\ell_\gamma(t)=
\sqrt{\frac{\|\gamma'(t)\|^2}{\epsilon^2+d(t)^2}
+\frac{a^2d(t)^2d'(t)^2}{(\epsilon^2+d(t)^2)^2}}.
\]

The pointwise lemma `distance_weighted_deriv_eq` proves
\[
d(t)d'(t)=r(t)r'(t)
\]
whenever the two trace derivatives exist. Its proof differentiates
\(d^2=r^2\) and uses uniqueness of the derivative. It never divides by
\(d\) or \(r\). In particular, the equality remains valid when the signed
normal coordinate is zero; both weighted terms then vanish.

The prior AC calculus supplies the two trace derivatives almost everywhere.
Combining the weighted identity with the actual Euclidean norm gives
\[
\boxed{\ell_\gamma(t)=
\sqrt{A(r(t))r'(t)^2+B(r(t))y'(t)^2}
\quad\text{almost everywhere}.}
\]
`graphSpeed_eq_coordinateSpeed` and `ae_graphSpeed_eq_coordinateSpeed`
package the pointwise and AC versions. No derivative of the absolute-value
function at zero is assumed. A crossing cusp is handled by the same a.e.
statement as any other AC trace.

## 3. Equality of the actual integrals and infima

`coordinateLength` integrates the explicit bilinear metric evaluated on the
actual Euclidean velocity expressed in the chart. The previous integrability
theorem and the a.e. density identity prove integrability of this coordinate
speed and
\[
\boxed{L_g(\gamma)=L_{\epsilon,a}(\gamma).}
\]
Here \(L_{\epsilon,a}\) is the unchanged graph integral from the general
closed-set geometry package.

The proof then identifies the entire sets of admissible endpoint lengths,
not only one preferred family of curves. Therefore
\[
\boxed{
\inf_{\substack{\gamma\text{ AC}\\\gamma(0)=x,\,\gamma(1)=y}}
L_g(\gamma)
=D_{\epsilon,a}(x,y).
}
\]
There is no smooth-curve density hypothesis and no assumed minimizing path.

The module also defines `chartLength` directly on curves
\(q:[0,1]\to\mathbb R\times\mathbb R\), using their actual derivative:
\[
L_{\rm chart}(q)=\int_0^1\sqrt{g_{q(t)}(q'(t),q'(t))}\,dt.
\]
`coordinateLength_eq_chartLength` checks the derivative transfer.
`coordinatePathLengths_eq_chartPathLengths` checks both directions of the
AC candidate correspondence. The final theorem
`intrinsicDistance_eq_coordinate_path_infimum` consequently writes the
general Euclidean intrinsic distance as this literal coordinate-curve
infimum.

## 4. Metric length and attained minima transfer

The existing intrinsic metric-length theorem now gives
\[
\boxed{\operatorname{Len}_{D_{\epsilon,a}}(\gamma)=L_g(\gamma)}
\]
for every ordinary AC Euclidean path. `eVariationOn_eq_coordinateLength`
uses Mathlib's actual partition-supremum `eVariationOn`, with the intrinsic
metric supplied explicitly.

`exists_coordinateLength_minimizer` proves that every pair \(x,y\) has an
ordinary AC path attaining the coordinate integral's infimum and satisfying
\[
L_g(\gamma)=D_{\epsilon,a}(x,y),\qquad
D_{\epsilon,a}(\gamma(s),\gamma(t))
=D_{\epsilon,a}(x,y)|s-t|.
\]
The assertion includes coincident endpoints. It transfers the previously
proved existence result through the actual integral and infimum equalities.

## 5. Exact formal boundary

The flat Euclidean specialization is no longer a missing interface: the
coordinate integral, graph integral, endpoint path infima, and actual metric
curve length are connected for the unrestricted AC class.

This does not yet prove that every minimizing curve is smooth or satisfies
the coordinate geodesic ODE. It does not prove Jacobi-field theory,
conjugate-point criteria, the cut locus, or the curved-Shore bifurcation.
The coordinate connection/curvature theorem and the path-distance theorem
now refer to the same explicit flat metric, but those additional geometric
arguments still require their own proofs.

No dependency pin, runtime implementation, provider contract, training gate,
or canonical parameter is changed. The targeted command
`lake build --wfail FractalSync.FlatShoreIntrinsic` passes under the pinned
Lean 4.24.0 / Mathlib environment. Package-wide verification is recorded in
[verification.md](verification.md).
