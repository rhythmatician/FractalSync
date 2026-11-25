# Issue 160: normal radii derived from curve jets and separation

The previous [normal-neighborhood result](160-embedded-normal-neighborhoods.md)
assumed a quadratic normal-chord certificate for the full Shore set. This
increment derives that certificate from actual curve derivatives, a small
parameter patch, and separation from the rest of Shore. It also localizes the
chart interface: a certificate on a finite parameter patch suffices for the
actual ambient signed-distance and metric identities near an interior point.

The checked sources are
[NormalRadius.lean](../../formal/FractalSync/NormalRadius.lean) and
[EmbeddedNormalPatch.lean](../../formal/FractalSync/EmbeddedNormalPatch.lean),
with 12 explicit theorem declarations each. All statements below described as
checked are in these files or the earlier normal-neighborhood modules. The
package uses the existing Lean 4.24.0/Mathlib pin; full verification is recorded
in [verification.md](verification.md).

## 1. From actual curve derivatives to a normal-chord bound

Let \(C:\mathbb R\to E\), where \(E\) is a real inner-product space. Let
\(U\subset\mathbb R\) be convex, \(s,t\in U\), and let \(C'_z\) be the
actual Fréchet derivative of \(C\) at each \(z\in U\). Suppose

\[
\|C'_z-C'_s\|\le L|z-s|.
\tag{1}
\]

The derivative norm in (1) is the operator norm. The proof reuses the actual
vector-valued Taylor estimate from `ProviderCertificates`, giving

\[
\|C(t)-C(s)-C'_s(t-s)\|\le \frac L2|t-s|^2.
\tag{2}
\]

For a unit vector \(n\) orthogonal to the actual tangent \(C'_s(1)\),
the linear term has zero normal projection. Cauchy--Schwarz and (2) yield

\[
|\langle n,C(t)-C(s)\rangle|\le \frac L2|t-s|^2.
\tag{3}
\]

This is `normal_chord_taylor`. No distance identity or nearest-point statement
appears in its assumptions.

If the curve also satisfies a chord lower bound

\[
m|t-s|\le\|C(t)-C(s)\|,\qquad m\ge0,
\tag{4}
\]

then every nonnegative radius satisfying \(RL\le m^2\) obeys

\[
2R|\langle n,q-C(s)\rangle|\le\|q-C(s)\|^2,
\qquad q\in C(U).
\tag{5}
\]

The checked result `normalChordBound_image` derives (5) by squaring (4) and
combining it with (3).

## 2. The chord lower bound is derived from the tangent

The main construction does not leave (4) as an unexplained geometric input.
Suppose

\[
\|C'_s(1)\|\ge v,\qquad |t-s|\le\delta,
\qquad L\ge0.
\]

The reverse triangle inequality and (2) give

\[
\begin{aligned}
\|C(t)-C(s)\|
&\ge v|t-s|-\frac L2|t-s|^2\\
&\ge\left(v-\frac{L\delta}{2}\right)|t-s|.
\end{aligned}
\tag{6}
\]

`chord_lower_bound_of_derivative` checks this estimate, including the
zero-length chord. Consequently, the useful positive constant is

\[
m=v-\frac{L\delta}{2}>0.
\tag{7}
\]

The same estimate proves injectivity on a base patch when the bounds hold
uniformly there. This is `injOn_of_curve_jet`; injectivity of that patch is a
conclusion, not a hypothesis of the local construction.

For an arc-length parameterization one can take \(v=1\). The condition in
(7) then requires \(L\delta<2\). This is a sufficient quantitative
condition, not a sharp maximal reach or a classification of all valid normal
radii.

There are two checked sources for (1):

* `derivative_variation_of_second_derivative` derives it from an actual second
  Fréchet derivative bounded by \(L\) on the convex parameter patch.
* `exists_derivative_variation_bound` derives the existence of a positive,
  finite \(L\) from actual \(C^2\) regularity on a compact convex patch.
  Its formal premise is `ContDiffAt` at every patch point, including endpoints,
  rather than only within-set differentiability.
  Compactness supplies a constant; this theorem does not calculate a numerical
  value for it.

## 3. Remote Shore controls the other part of the radius

Local curvature control alone cannot exclude another part of Shore from being
closer to the normal offset. The full-set certificate needs a remote
separation condition. Suppose

\[
\|q-C(s)\|\ge d>0,
\qquad q\in S\setminus C(U).
\tag{8}
\]

For those remote points, Cauchy--Schwarz gives

\[
2R|\langle n,q-C(s)\rangle|
\le2R\|q-C(s)\|
\le\|q-C(s)\|^2
\]

whenever \(2R\le d\). Combining this with the local estimate gives the
explicit checked radius

\[
\boxed{
R=\min\left\{\frac{m^2}{L},\frac d2\right\}>0,
\qquad m=v-\frac{L\delta}{2}>0,\quad L,d>0.
}
\tag{9}
\]

`certifiedRadius_normalChordBound` proves the version with a supplied chord
lower bound; `normal_radius_of_curve_jet` substitutes the derived bound (6).
`uniform_normal_radius` proves that the same \(R\) works for every base
parameter \(s\in V\subset U\) when all constants and hypotheses are uniform
over \(V\). The larger set \(U\) controls the local chords, while \(V\) is
the set of admissible feet.

The assumptions are meaningful for arbitrary ambient dimension. A continuous
linear equivalence with \(\mathbb R^2\) is needed only later, for a planar
normal chart.

## 4. Obtaining remote separation from an embedded compact curve

The first compactness wrapper is general. If \(V\) is compact, \(C\) is
continuous on \(V\), and

\[
C(V)\cap\overline{S\setminus C(U)}=\varnothing,
\tag{10}
\]

then `exists_remote_separation` proves a common \(d>0\) satisfying (8)
for every \(s\in V\). The closed remote set may be empty. Requiring
disjointness from its closure prevents remote points from approaching the
base patch with zero separation.

A second theorem derives (10) for an injectively parameterized compact curve.
Let \(P\) be compact, \(C\) continuous and injective on \(P\),
\(S=C(P)\), and \(V\) compact with

\[
V\subset P,\qquad V\subset U,\qquad U\text{ open in }\mathbb R.
\]

Then the remote part lies in the compact set \(C(P\setminus U)\).
Injectivity separates that set from \(C(V)\). The proof first derives the
disjointness and then obtains a positive separation by compactness. This is
`exists_remote_separation_of_compact_injective`.

Combining this theorem with the local jet estimates yields
`exists_normalPatch_of_compact_injective`: an actual positive normal radius
and an injective normal strip exist under these explicit hypotheses, without
a separately assumed remote-distance constant.

This compact theorem applies to an injective parameterized arc. It does not
apply directly to a periodic parameterization that identifies the two
endpoints of a circle, or to a self-intersecting parameterization. The general
full-Shore separation theorem remains available for a local patch of a closed
curve when its other parts have a certified separation. No reach theorem for
arbitrary fractal or noncompact Shore sets is asserted.

## 5. Actual distance, uniqueness, and a local normal chart

`NormalPatch S R C N V` packages positivity of \(R\), membership of each
foot \(C(s)\) in \(S\), unit normals, and the full-Shore normal-chord
bound for \(s\in V\). `normalPatch_of_curve_jet` constructs this predicate
from the estimates above and simultaneously proves injectivity of

\[
\Phi(r,s)=C(s)+rN(s)
\quad\text{on}\quad
\{(r,s):s\in V,\ |r|<R\}.
\tag{11}
\]

The earlier squared-distance gap theorem now gives

\[
d(\Phi(r,s),S)=|r|\quad (s\in V,\ |r|\le R).
\tag{12}
\]

For \(|r|<R\), the nearest foot is uniquely \(C(s)\). Uniqueness is not
claimed at the boundary radius. These conclusions use the actual
`Metric.infDist`, not a supplied scalar distance trace.

The local chart interface requires \(V\) to be a neighborhood of the chosen
base parameter \(s\). It no longer requires a normal certificate for every
real parameter. Given actual planar Frenet data

\[
C'(s)=T,\qquad N'(s)=-\kappa T,
\qquad \|T\|=\|N(s)\|=1,\qquad\langle N(s),T\rangle=0,
\]

an actual frame equivalence \(B:\mathbb R^2\simeq E\), local \(C^1\)
regularity of \(C,N\), and \(1-\kappa r\ne0\), the previously checked
inverse function theorem constructs a local chart \(e\) with forward map
\(\Phi\). The new localized theorem proves on a full ambient neighborhood
of \(e(r,s)\) that

\[
d(x,S)=|(e^{-1}(x))_r|.
\tag{13}
\]

For \(\epsilon>0\), the actual ambient ruler
\(\rho(x)=\sqrt{\epsilon^2+d(x,S)^2}\) therefore has derivative

\[
D\rho_{e(r,s)}=
\frac r{\sqrt{\epsilon^2+r^2}}\,\operatorname{fst}\circ D(e^{-1})_{e(r,s)}.
\tag{14}
\]

This includes \(r=0\); differentiating the squared distance removes the
unsigned-distance cusp within this certified chart. Combining (14) with the
actual normal-map derivative gives the existing curved-coordinate metric as
the pullback of the actual ambient graph metric:

\[
\boxed{
\Phi^*g=
\frac{\epsilon^2+(1+a^2)r^2}{(\epsilon^2+r^2)^2}\,dr^2
+\frac{(1-\kappa r)^2}{\epsilon^2+r^2}\,ds^2.
}
\tag{15}
\]

`exists_normalChart_with_metric` packages the constructed chart, the ambient
distance identity, and equality of (15) on every pair of tangent vectors at
the represented point. The formal result does not identify the whole strip
with the source of one inverse-function chart.

## 6. Remaining embedded-geometry work

The normal radius is now derived under explicit jet and separation hypotheses,
and the metric/chart bridge accepts a genuinely local curve patch. What
remains includes the following.

* Establishing usable numerical jet and separation constants for a specific
  embedded Shore representation, or a general reach theorem under a chosen
  class of embedding hypotheses.
* Identifying the ambient Levi-Civita connection with the transformed curved
  coordinate connection and transporting its curvature calculation. Metric
  pullback alone is not presented as that connection/curvature theorem.
* Regularity and the geodesic ODE for attained AC minimizers, general second
  variation and conjugacy, and the global cut-locus and equality-case results.
* The functional-analytic curved-Shore bifurcation reduction and its local
  branch/minimum theorem.

The runtime, training, canonical parameters, and provider contracts are
unchanged. The research issues remain open and PR #164 remains a draft.
