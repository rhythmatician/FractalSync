# Absolutely continuous paths and the exact Shore-distance infimum

This continuation closes the absolutely continuous calibration and
Shore-distance path-infimum gaps in #160. The primary results in Sections 1--5
are checked in Lean. Section 7 gives an additional conventional classification
of all minimizing paths and labels its different proof status explicitly.

The source modules are
[`AbsolutelyContinuousPaths.lean`](../../formal/FractalSync/AbsolutelyContinuousPaths.lean),
[`IntrinsicShoreDistance.lean`](../../formal/FractalSync/IntrinsicShoreDistance.lean),
and [`RadialLipschitz.lean`](../../formal/FractalSync/RadialLipschitz.lean).
The duration/action corollaries are in
[`CrossingAction.lean`](../../formal/FractalSync/CrossingAction.lean).
They extend the existing definitions and the actual interval-integral radial
profile. They do not replace the path problem with a stipulated formula.

## 1. The curve class and length functional

Let E be a finite-dimensional real normed space, let S be a nonempty closed
subset, and put

\[
d(x)=\operatorname{dist}(x,S),\qquad
\rho_\epsilon(x)=\sqrt{d(x)^2+\epsilon^2},\qquad \epsilon>0.
\]

The parameter a may be any real number because the length depends on a squared.
The physical convention uses a>=0. For an absolutely continuous curve gamma
on [t0,t1], the original graph length is

\[
L_{\epsilon,a}(\gamma)=\int_{t_0}^{t_1}
\frac{\sqrt{\|\gamma'(t)\|^2+
a^2((\rho_\epsilon\circ\gamma)'(t))^2}}
{\rho_\epsilon(\gamma(t))}\,dt.
\tag{1}
\]

The formal statements use Mathlib's epsilon-delta definition
`AbsolutelyContinuousOnInterval`, the actual `deriv` functions, and Lebesgue
interval integrals. They derive almost-everywhere differentiability and
integrability from absolute continuity. No pointwise derivative, separately
integrable velocity, or separately integrable graph-speed hypothesis remains.

Write r=d composed with gamma, v=norm(gamma'), and w=r'. The expanded speed
used in the formal length definition is

\[
\ell(t)=\sqrt{\frac{v(t)^2}{r(t)^2+\epsilon^2}
+\frac{a^2r(t)^2w(t)^2}{(r(t)^2+\epsilon^2)^2}}.
\tag{2}
\]

`acGraphLength_eq_ruler_integral` checks equality with (1), using the actual
almost-everywhere chain rule for the ruler. Values assigned by `deriv` at
nondifferentiable times occur on a null set and do not affect either integral.

## 2. Unrestricted calibration

The distance function is one-Lipschitz. Lipschitz composition preserves
absolute continuity, and the checked derivative comparison gives

\[
r\in AC,\qquad |r'|\le\|\gamma'\|\quad\text{almost everywhere}.
\tag{3}
\]

The positive-floor ruler is also one-Lipschitz. In particular,
rho composed with gamma is absolutely continuous even when the ambient
distance field has cut loci. There is no global differentiability premise
on the Shore or its distance function.

Let F be the previously defined actual radial integral,

\[
F_{\epsilon,a}(r)=\int_0^r
\frac{\sqrt{\epsilon^2+(1+a^2)s^2}}{\epsilon^2+s^2}\,ds.
\]

Its global Lipschitz bound makes F composed with r absolutely continuous.
The checked scalar chain rule and the absolute-continuity FTC yield

\[
F(r(t_1))-F(r(t_0))=\int_{t_0}^{t_1}F'(r(t))r'(t)\,dt.
\tag{4}
\]

The existing exact squared remainder is

\[
\ell^2-(F'(r)r')^2
=\frac{\|\gamma'\|^2-(r')^2}{r^2+\epsilon^2}\ge0.
\tag{5}
\]

Moreover,

\[
0\le\ell\le\frac{\sqrt{1+a^2}}{\epsilon}\|\gamma'\|.
\tag{6}
\]

Finite-dimensional absolute continuity supplies integrability of gamma'.
Thus (6) establishes graph-speed integrability. Integrating (5) and using
(4) proves the unrestricted endpoint theorem:

\[
\boxed{|F(d(\gamma(t_1)))-F(d(\gamma(t_0)))|
\le L_{\epsilon,a}(\gamma).}
\tag{7}
\]

If gamma(c) belongs to S for t0<=c<=t1, apply (7) to the two subintervals:

\[
\boxed{F(d(\gamma(t_0)))+F(d(\gamma(t_1)))
\le L_{\epsilon,a}(\gamma).}
\tag{8}
\]

The theorem requires no derivative at c, or at any other specified time.
It admits distance-trace cusps and arbitrary null exceptional sets, rather
than only a prescribed piecewise-differentiable decomposition.

The principal theorem identifiers are `ac_distance_trace_deriv_bound`,
`ac_radialProfile_integral`, `ac_graphSpeed_intervalIntegrable`,
`ac_distance_trace_endpoint_bound`, and `ac_distance_trace_crossing_bound`.

## 3. An attained minimum over the actual path class

Define the set of admissible lengths by

\[
\mathcal A_x=\{L_{\epsilon,a}(\gamma):\gamma\in AC([0,1],E),
\ \gamma(0)\in S,\ \gamma(1)=x\}.
\]

The Lean definition `intrinsicShoreDistance` is the real infimum of this set.
The proof establishes nonemptiness and a lower bound before using the
conditionally complete order API. In fact it proves the stronger statement

\[
\boxed{F_{\epsilon,a}(d(x))=\min\mathcal A_x.}
\tag{9}
\]

For the lower bound use (7) and d(gamma(0))=0. For attainment, a Euclidean or
normed-space nearest point p exists because S is closed and nonempty in a
finite-dimensional space. The affine path p+t(x-p) is absolutely continuous,
and the earlier nearest-segment theorem gives d(gamma(t))=t d(x). The new
proof identifies its actual derivatives almost everywhere and its actual
length integral with F(d(x)). The zero-length case is included.

`radialProfile_isLeast_shorePathLengths` expresses (9) as an `IsLeast`
statement. `intrinsicShoreDistance_eq_radialProfile` then identifies the
actual infimum. No convexity of S, uniqueness of its nearest point, or
restriction on the interior of an admissible path is imposed.

The opposite orientation is checked independently. `pointToShorePathLengths`
uses gamma(0)=x and gamma(1) in S; a reversed nearest segment attains the same
minimum. `pointToShorePathLengths_sInf_eq_intrinsicShoreDistance` equates the
two infima. This does not rely on assuming symmetry of a general pairwise
metric that has not yet been formally constructed.

The same module proves that the infimum is zero exactly when x belongs to S.
For every fixed x outside S it also proves

\[
\operatorname{intrinsicShoreDistance}_{\epsilon,a}(x,S)
\longrightarrow+\infty\qquad(\epsilon\downarrow0).
\tag{10}
\]

Equation (10) is now a theorem about a path infimum. It is not the complete
componentwise limit of distances between two arbitrary off-Shore points.

## 4. The optimal global Lipschitz constant for F

For a>=0 define

\[
M(a)=\begin{cases}
1,&0\le a\le1,\\
(1+a^2)/(2a),&a>1.
\end{cases}
\]

The earlier density bounds and their attainment now give a checked
`LipschitzWith` theorem for F with constant M(a)/epsilon. Conversely every
global Lipschitz constant is at least M(a)/epsilon: differentiating at a
point where the density attains that value gives the necessary lower bound.
The principal identifiers are `radialProfile_lipschitz_sharp` and
`radialProfile_lipschitz_constant_ge`.

For the AC composition argument, the convenient bound
sqrt(1+a^2)/epsilon works for every real a. The sharper result is useful
independently and completes the previously conventional mean-value corollary.

## 5. Checked crossing duration and kinetic action

For a Shore-crossing curve, put

\[
B=F(d(\gamma(t_0)))+F(d(\gamma(t_1))),\qquad T=t_1-t_0.
\]

If the intrinsic graph speed satisfies ell(t)<=V almost everywhere, Lean
proves B<=VT. For V>0 this gives the necessary duration bound T>=B/V.
The checked product-form statement also handles zero-duration intervals.

If T>0 and ell squared is integrable, the actual integral Cauchy-Schwarz
argument gives

\[
\boxed{\int_{t_0}^{t_1}\frac{\ell(t)^2}{2}\,dt
\ge\frac{B^2}{2T}.}
\tag{11}
\]

The source proves Cauchy-Schwarz directly by integrating the nonnegative
square `(T ell - integral ell)^2`. The speed is already integrable by AC;
square integrability is the explicit extra finite-action assumption.
The principal identifiers are `ac_crossing_duration_bound` and
`ac_crossing_kinetic_action_bound`.

At nonsmooth points, kinetic density means ell squared divided by two.
Where the local metric exists it is the usual g(v,v)/2. Equation (11) bounds
integrated kinetic action, not mechanical work, and is not a sufficient
dynamical crossing criterion.

## 6. Analysis support and provenance

The pinned Mathlib supplied the definition of absolute continuity and bounded
variation, but lacked its fundamental theorem of calculus. The necessary
upstream proofs were adapted from Mathlib commit
`aad247f928b1eb7c58478742119fdd15a37f0ffd`, with their original Yizheng Zhu
authorship and Apache 2.0 license retained. The
[support inventory](../../formal/FractalSync/Analysis/README.md) records the
source mapping and changes. The project pins remain unchanged.

The decisive theorem is `AbsolutelyContinuousOnInterval.integral_deriv_eq_sub`.
Its proof controls the gaps in a Vitali covering by absolute continuity.
That step excludes the singular variation that mere almost-everywhere
differentiability would permit. Finite-dimensional vector derivative
integrability follows by continuous linear coordinates and the scalar theorem.

These analysis results are attributed infrastructure, not a novelty claim
for FractalSync. They are kernel checked together with the project results.

## 7. Conventional strengthening: all minimizing paths

The following classification has a conventional proof, independently reviewed
in this increment, but is **not a Lean theorem in this package**.

For Euclidean E and x outside S, every minimizing admissible path is

\[
\gamma(t)=p+\lambda(t)(x-p),
\]

where p is a Euclidean nearest point in S, and lambda is absolutely continuous,
nondecreasing, and runs from 0 to 1. Conversely every such path minimizes.
If x belongs to S, the only minimizing path is constant.

To prove necessity, let c=F'(r)>0. For a minimizing path the two nonnegative
defects in

\[
\ell\ge c|r'|\ge c r'
\]

have zero integrals. They therefore vanish almost everywhere. The second
forces r'>=0; equation (5) then forces norm(gamma')=r'. Thus the Euclidean
length is integral r'=d(x). For p=gamma(0),

\[
d(x)\le\|x-p\|\le\int_0^1\|\gamma'\|=d(x),
\]

so p is nearest. At each intermediate time, apply the triangle inequality
to p, gamma(t), x and bound the two displacements by the corresponding
subpath lengths. All inequalities are equalities. Euclidean strict convexity
then puts gamma(t) on [p,x]. The nearest-segment trace identity identifies
lambda(t)=r(t)/d(x), which is AC and nondecreasing. The converse follows by
the same chain rule and radial length integral. When d(x)=0, zero graph length
forces gamma'=0 almost everywhere, and absolute continuity makes gamma constant.

Consequently the geometric minimizing path is unique precisely when the
Euclidean nearest point is unique. Parameterized minimizers need not be unique:
they may pause or traverse the segment at different nonnegative speeds.
This classification requires Euclidean or suitable strictly convex norm
geometry; it is not asserted for arbitrary norms. S itself need not be convex.

The closest-Shore minimizing images are therefore independent of epsilon
and a, even though their intrinsic lengths change with both parameters.
This says nothing comparable about minimizing paths between two general
points; the flat-Shore shortcut and cut-locus phenomena concern that different
problem.

## 8. Remaining formal boundary

The general pairwise intrinsic metric, its triangle inequality, equivalence
with the path-length metric, properness, global minimizing curves between
arbitrary endpoints, and the complete componentwise zero-floor limit still
need their own formal construction. Their conventional arguments remain in
the [distance-geometry note](160-distance-geometry.md).

The minimizer classification in Section 7, the smooth curvature interpretation,
the flat cut locus, and the curved-Shore bifurcation remain conventional.
The exact Shore-path minimum proved here does not silently establish those
different claims. Restricted GeometryProvider validity domains can exclude
the nearest segment; no unrestricted theorem is transferred to such a domain
without an admissibility premise.
