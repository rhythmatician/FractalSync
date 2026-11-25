# Issue 160: actual distance and metric in certified normal neighborhoods

The coordinate model in [curved Shore geometry](160-curved-coordinate-geometry.md)
now has a checked connection to embedded geometry. The new proofs construct an
actual local normal chart, identify the absolute normal coordinate with
Mathlib's distance to the full Shore set, differentiate the actual regularized
distance in the ambient space, and identify its metric pullback.

The sufficient hypotheses include a quantitative normal-chord certificate.
They do not assume the desired distance identity or a pre-existing tubular
chart. They also do not derive such a certificate for every smooth embedded
curve. This distinction matters when another part of Shore is nearby.

The continuation in [derived normal radii](160-derived-normal-radii.md) now
derives a uniform certificate from actual curve jets and remote separation,
including a compact injective-curve instance. Its local `NormalPatch`
interface removes the global-parameter requirement from the chart and metric
bridge below. The original `NormalTube` results remain available unchanged.

The checked sources are
[TubularNeighborhood.lean](../../formal/FractalSync/TubularNeighborhood.lean)
and [EmbeddedShoreMetric.lean](../../formal/FractalSync/EmbeddedShoreMetric.lean),
with 9 and 18 explicit theorem declarations respectively. They use the existing
Lean 4.24.0/Mathlib pin. The package build and transitive audit are recorded in
[verification.md](verification.md).

## 1. A geometric certificate for an actual normal radius

Let \(E\) be a real inner-product space, \(p\in S\), \(\|n\|=1\), and
\(R>0\). The predicate `NormalChordBound S R p n` is

\[
2R\,|\langle n,q-p\rangle|\le \|q-p\|^2
\qquad\text{for every }q\in S.
\tag{1}
\]

This is a condition on the actual embedded set. In particular, the quantifier
ranges over all of \(S\), including points outside the selected curve patch.
No distance infimum or nearest-point conclusion appears in the predicate.

For the normal offset \(x=p+rn\), Lean first proves the exact expansion

\[
\|x-q\|^2-r^2
=\|q-p\|^2-2r\langle n,q-p\rangle,
\]

then derives the quantitative separation bound

\[
\boxed{\quad
\|x-q\|^2-r^2
\ge \left(1-\frac{|r|}{R}\right)\|q-p\|^2.
\quad}
\tag{2}
\]

Consequently,

\[
|r|\le R\quad\Longrightarrow\quad d(p+rn,S)=|r|.
\tag{3}
\]

When \(|r|<R\), equality in the distance minimum forces \(q=p\). The foot
point is therefore the unique nearest point. No separate nearest-point
existence theorem is needed: \(p\) is the exhibited minimizer. Neither
closedness nor completeness is required for these conclusions from (1).

The distinction between \(|r|\le R\) in (3) and \(|r|<R\) for uniqueness is
intentional. At a limiting radius, different normal rays may meet.

Principal declarations are `normalOffset_dist_sq`,
`normalOffset_dist_sq_gap`, `infDist_normalOffset`, and
`normalOffset_unique_nearest`.

## 2. A concrete certificate and injectivity of a normal strip

For the actual sphere \(S=\{q:\|q\|=R\}\), choose the inward normal
\(n=-p/R\). Lean proves both its unit norm and

\[
2R\langle n,q-p\rangle=\|q-p\|^2.
\]

This supplies (1), and the sphere wrappers establish (3) and unique nearest
points throughout the strict normal radius. At \(r=R\), the normal offset is
the sphere center: the distance identity still holds, while the uniqueness
theorem correctly does not apply. These sphere statements work in the same
general real inner-product space, not just the plane.

For a parameterized Shore \(C(y)\) and unit normal field \(N(y)\), define

\[
\Phi(r,y)=C(y)+rN(y).
\]

If \(C\) is injective on a parameter set \(U\), its image lies in \(S\), and
the same radius \(R\) satisfies (1) at every \(C(y)\), \(y\in U\), then
`normalMap_injOn_strip` proves that \(\Phi\) is injective on

\[
\{(r,y):y\in U,\ |r|<R\}.
\]

The proof first uses uniqueness of the nearest foot to identify \(C(y)\),
then injectivity of \(C\) to identify \(y\), and finally the nonzero unit
normal to identify \(r\). A periodic circle parametrization must be restricted
to an injective parameter patch before using this strip theorem.

## 3. Actual Frenet derivatives and a constructed local chart

At \(z=(r,y)\), suppose the actual derivatives satisfy

\[
C'(y)=T,\qquad N'(y)=-\kappa T,
\qquad \|T\|=\|N(y)\|=1,
\qquad \langle N(y),T\rangle=0.
\]

The derivative of the actual normal map is checked as

\[
D\Phi_z(v_r,v_y)
=v_rN(y)+(1-\kappa r)v_yT.
\tag{4}
\]

Thus its Euclidean metric pullback is

\[
\langle D\Phi_zv,D\Phi_zw\rangle
=v_rw_r+(1-\kappa r)^2v_yw_y.
\tag{5}
\]

For the local inverse construction, the theorem receives an actual continuous
linear equivalence \(B:\mathbb R^2\simeq E\) whose columns are \(N(y)\) and
\(T\). This is the planar spanning hypothesis; an arbitrary pair of vectors
in a higher-dimensional space would not suffice. If \(C,N\) are \(C^1\) at
\(y\) and \(1-\kappa r\ne0\), the explicit derivative in (4) is an invertible
linear map. Mathlib's inverse function theorem constructs an
`OpenPartialHomeomorph` whose forward map is \(\Phi\), whose source contains
\(z\), and whose inverse has the inverse of (4) as its actual derivative.

This is `exists_normalChart`; it does not assume an inverse map. It is a local
chart theorem, separate from the injectivity-on-a-supplied-strip theorem.
The formal statements do not assert that the local chart's source is the
entire normal strip.

## 4. Actual signed coordinate and ambient ruler calculus

The `NormalTube` structure packages \(R>0\), \(C(y)\in S\), unit normals,
and (1) for every parameter \(y\). This version deliberately uses global
parameter hypotheses, while the chart and Frenet regularity are local.

Let \(e\) be the constructed normal chart and write
\(\sigma(x)=(e^{-1}(x))_r\). For \(|r|<R\), Lean proves the following
identity on an actual ambient neighborhood of \(e(z)\):

\[
d(x,S)=|\sigma(x)|.
\tag{6}
\]

The normal orientation fixes the sign of \(\sigma\). Equation (6) identifies
its absolute value with actual distance, rather than postulating a smooth
signed-distance function. The theorem uses the open target of the actual
chart and continuity of its inverse; it is not restricted to a path through
the base point.

For \(\epsilon>0\), define the actual ambient regularized-distance function

\[
\rho(x)=\sqrt{\epsilon^2+d(x,S)^2}.
\]

From (6), it agrees locally with
\(\sqrt{\epsilon^2+\sigma(x)^2}\). If \(D(e^{-1})_{e(z)}=A\), the checked
ambient derivative is

\[
D\rho_{e(z)}=
\frac{r}{\sqrt{\epsilon^2+r^2}}\,\operatorname{fst}\circ A.
\tag{7}
\]

This includes \(r=0\). The unsigned distance itself may have a cusp there;
the proof differentiates the smooth regularized expression after squaring
the absolute value. It does not infer smoothness of arbitrary regularized
distance fields away from a certified chart.

The main statements are `infDist_normalChart`,
`eventually_infDist_normalChart`, and `hasFDerivAt_ambientRuler`.

## 5. Transfer of the actual metric

The new `ambientGraphMetric` is defined using the actual Fréchet derivative
of the ambient function \(\rho\):

\[
g_x(v,w)=
\frac{\langle v,w\rangle+a^2D\rho_x(v)D\rho_x(w)}{\rho(x)^2}.
\tag{8}
\]

`ambientGraphMetric_eq_scaleMetric` identifies (8) with the existing
`ScaleMetric.metric` whenever \(u\) represents the actual derivative by
\(D\rho_x(v)=\langle u,v\rangle\). Thus this is the existing rank-one metric
written directly through actual calculus.

Combining (4), (5), and (7) gives the checked pullback

\[
\boxed{
\Phi^*g
=\frac{\epsilon^2+(1+a^2)r^2}{(\epsilon^2+r^2)^2}\,dr^2
+\frac{(1-\kappa r)^2}{\epsilon^2+r^2}\,dy^2.
}
\tag{9}
\]

The right side is exactly `CurvedShoreGeometry.metric`, not a new coordinate
metric. `ambientGraphMetric_pullback` proves (9) with the actual ambient
derivative discharged by (7). The combined
`exists_normalChart_with_metric` constructs the chart and supplies both the
ambient-neighborhood distance identity and the metric pullback in one result.

## 6. Remaining transfer obligations

These results establish embedded distance and metric correspondence under
explicit sufficient hypotheses. The [derived-radius continuation](160-derived-normal-radii.md)
now supplies R=min(m²/L,d/2), with m=v-L*delta/2>0, from actual curve-jet and
remote-separation estimates, and derives the latter separation for compact
injective parameterizations.

The [ambient-connection continuation](160-ambient-shore-connection.md) also
identifies the connection formed from the actual regularized distance ruler
and its Riesz gradient with the curved-coordinate connection. It derives
ruler smoothness from the local inverse chart and transfers the curvature
formula, including curvature \(+1\) on Shore. Its sufficient hypotheses
include the normal-patch and Frenet data, a \(C^2\) normal map with a
differentiable curvature function for connection identification, and a
\(C^3\) normal map with a \(C^2\) curvature function for the curvature formula.
The following are still separate:

- Compute usable numerical jet and separation bounds for a particular Shore
  representation, and extend reach results to other embedding hypotheses.
  A local curvature bound alone does not control another nearby component.
- Package charts and overlaps into a manifold-level construction. The current
  object is an actual local open partial homeomorphism with an inverse
  derivative, plus the optional injectivity result on a supplied strip.
- Prove smooth regularity of attained AC minimizers and the functional-analytic
  curved-Shore bifurcation. Neither follows just from a pointwise metric
  correspondence.

The separate [nonlinear covariance and tensor note](160-nonlinear-geometry.md)
provides general connection/curvature transformation and extension-independence
tools used in that transfer. Runtime behavior, provider contracts, canonical
parameters, and the status of draft PR #164 are unchanged by these proofs.
