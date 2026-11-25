# Regularized distance graphs: exact radial distance and a singular limit

Research progress for [#160](https://github.com/rhythmatician/FractalSync/issues/160),
2026-11-24. These are exact mathematical results under the assumptions below.
The accompanying Lean package now checks the scalar integral calculus, closed
primitive, sharp additive constant, optimal Lipschitz constant, scalar
asymptotics, unrestricted absolutely continuous calibration, and the exact
attained Shore-distance path infimum in both orientations. Crossing-duration
and kinetic-action bounds are also checked. The package now constructs the
actual pairwise AC path infimum as a metric, identifies its topology, proves
properness and completeness, and checks every branch of its pointwise zero-floor
limit. Metric curve length is now identified with the graph integral, every
pair has an attained AC minimum, and all nearest-Shore minimizers are
classified in the strictly convex setting. The flat coordinate curvature
formula is also derived from an actual covariant-derivative commutator.
No novelty claim is made. See the
[radial formalization](160-radial-profile.md),
[AC path and infimum formalization](160-absolute-continuous-paths.md),
[pairwise metric and limit formalization](160-pairwise-metric-limit.md),
[intrinsic curve length and minimizers](160-intrinsic-length.md),
[flat-Shore geometry](160-flat-shore.md), and
[exact theorem inventory](../../formal/README.md).

## 1. Setting

Let S be a nonempty closed subset of finite-dimensional Euclidean space R^n,
let d(x)=dist(x,S), epsilon>0, and a>=0. Put

$$
\rho_\epsilon(x)=\sqrt{d(x)^2+\epsilon^2}.
$$

For an absolutely continuous curve gamma, define

$$
L_{\epsilon,a}(\gamma)=\int
\frac{\sqrt{|\dot\gamma|^2+a^2|\dot\rho_\epsilon|^2}}
{\rho_\epsilon}\,dt,
\qquad
d_{\epsilon,a}(x,y)=\inf_{\gamma:x\to y}L_{\epsilon,a}(\gamma).
\tag{1}
$$

All absolutely continuous curves in R^n are admissible here. A runtime validity
domain is an additional restriction and can change the distance. No differentiable boundary, unique
nearest point, or globally smooth distance field is assumed. The scalar
composition d(gamma) is absolutely continuous and satisfies
`|d'| <= |gamma'|` almost everywhere. Also
`rho' = d d'/rho` almost everywhere, including the regularized crest.

For FractalSync take S to be the Shore and `a=lambda/ln(2)`. The sign of the
signed distance D drops out of this length because d=|D| and rho depends on D^2.
This does not remove sign or validity information from the Physics contract.

## 2. The exact shortest distance to S

Define

$$
F_{\epsilon,a}(r)=\int_0^r
\frac{\sqrt{\epsilon^2+(1+a^2)t^2}}{\epsilon^2+t^2}\,dt,
\qquad r\ge0.
\tag{2}
$$

**Theorem 1.** On the unrestricted graph,

$$
\boxed{\operatorname{dist}_{\epsilon,a}(x,S)
=F_{\epsilon,a}(d(x)).}
\tag{3}
$$

More generally, any curve from x to y obeys

$$
L_{\epsilon,a}(\gamma)\ge
|F_{\epsilon,a}(d(y))-F_{\epsilon,a}(d(x))|.
\tag{4}
$$

**Proof.** Write r=d(gamma), V=|gamma'|, and w=r'. The square of the
length density, minus the square of `F'(r)|w|`, is exactly

$$
\frac{V^2}{r^2+\epsilon^2}
+\frac{a^2r^2w^2}{(r^2+\epsilon^2)^2}
-\frac{[\epsilon^2+(1+a^2)r^2]w^2}{(r^2+\epsilon^2)^2}
=\frac{V^2-w^2}{r^2+\epsilon^2}\ge0.
\tag{5}
$$

Both densities are nonnegative. Integrating and applying the scalar chain rule
gives (4), hence the lower bound in (3).

Closedness, nonemptiness, and finite dimension supply a nearest point p in S.
If r=d(x)>0, parameterize the segment from p to x by Euclidean arclength s in
[0,r]. Its distance to S is exactly s: p gives the upper bound, while the
1-Lipschitz property and the endpoint x give `d(gamma(s)) >= r-(r-s)=s`.
Thus V=|w|=1 and equality holds in (5). This segment has length F(r).
For x in S the constant curve gives zero. This proves (3).

The same equality holds in a restricted domain if at least one such nearest-point
segment is admissible. Without that hypothesis (4) remains a lower bound, but
the equality need not hold. This is a geometric shortest-path statement, not a
claim that Controls can follow the segment or that conservative Physics will.

## 3. A closed form and a sharp additive approximation

Write b=sqrt(1+a^2) and q=r/epsilon. Then

$$
\boxed{
F_{\epsilon,a}(r)
=b\operatorname{asinh}(bq)
-a\operatorname{atanh}\!\left(
\frac{aq}{\sqrt{1+b^2q^2}}\right).}
\tag{6}
$$

The argument of atanh has magnitude strictly less than one. Formula (6) also
covers a=0, giving `F=asinh(r/epsilon)`.

**Derivation.** Substitute `t/epsilon=sinh(u)/b` in (2). The integrand becomes

$$
b\left(1-\frac{a^2}{b^2+\sinh^2u}\right)du.
$$

For a>0 the second primitive is
`(ab)^(-1) atanh((a/b)tanh(u))`; a=0 follows directly. Evaluation at zero
fixes the integration constant. Independent symbolic differentiation gives the
original integrand, but the substitution is the proof.

There is also a convenient quadrature form:

$$
F_{\epsilon,a}(r)
=\int_0^{\operatorname{asinh}(q)}\sqrt{1+a^2\tanh^2u}\,du.
\tag{7}
$$

In particular,

$$
\operatorname{asinh}(q)\le F_{\epsilon,a}(r)
\le b\operatorname{asinh}(q).
\tag{8}
$$

**Theorem 2 (uniform additive error).** Define

$$
C(a)=a\operatorname{asinh}(a)-b\log b.
$$

Then C(a)>=0 and

$$
\boxed{0\le b\operatorname{asinh}(q)-F_{\epsilon,a}(r)\le C(a).}
\tag{9}
$$

The upper constant is optimal for a fixed a: the difference tends to C(a) as
q tends to infinity.

**Proof.** The difference vanishes at zero and its derivative with respect to q is

$$
\frac{a^2}
{(1+q^2)\,[b\sqrt{1+q^2}+\sqrt{1+b^2q^2}]}\ge0.
$$

Using (6), `asinh(bq)-asinh(q) -> log b`, and
`atanh(a/b)=asinh(a)`, its limit is C(a). Monotonicity proves (9) and
nonnegativity of the constant. Consequently

$$
F_{\epsilon,a}(r)
=b\log(r/\epsilon)
+b\log(2b)-a\operatorname{asinh}(a)+o(1)
\tag{10}
$$

as r/epsilon tends to infinity. Near the crest,

$$
F_{\epsilon,a}(r)=q+\frac{a^2-1}{6}q^3+O(q^5).
$$

These formulas describe the exact model. Formula (6) can suffer floating-point
cancellation for extreme parameters; it is not supplied as a production numerical
algorithm. The log-coordinate quadrature (7) is an independent check, not a
replacement Rust authority.

## 4. Checked metric, topology, and properness

Let k_epsilon be the length metric with density `|gamma'|/rho_epsilon`.
Since rho_epsilon is 1-Lipschitz,

$$
k_\epsilon\le d_{\epsilon,a}\le b k_\epsilon.
\tag{11}
$$

A particularly useful quantitative estimate is

$$
\boxed{
\log\!\left(1+\frac{|x-y|}{\rho_\epsilon(x)}\right)
\le d_{\epsilon,a}(x,y)
\le \frac{b}{\epsilon}|x-y|.}
\tag{12}
$$

**Checked proof.** The anchored scalar trace
`log(rho(x)+|gamma(t)-x|)` is absolutely continuous. The one-Lipschitz ruler
satisfies `rho(gamma(t)) <= rho(x)+|gamma(t)-x|`, and the actual a.e. derivative
of this logarithmic trace is bounded above by the graph speed.
The AC fundamental theorem of calculus gives the lower bound in (12) for
every path, hence for its infimum. The straight segment and `rho>=epsilon`
give the upper bound. Inequality (11) follows from the actual a.e. comparison
`|rho'|<=|gamma'|`, integration, and the common path class. These results are in
[`DisplacementBounds.lean`](../../formal/FractalSync/DisplacementBounds.lean),
[`PairwiseDistance.lean`](../../formal/FractalSync/PairwiseDistance.lean), and
[`MetricComparison.lean`](../../formal/FractalSync/MetricComparison.lean).

Reversal preserves actual AC graph length, and concatenation adds the two
actual lengths. Together with the nonempty, lower-bounded admissible length
classes, these facts prove symmetry and the triangle inequality of the
infimum. Equation (12) gives strict positivity for distinct endpoints.
[`PathOperations.lean`](../../formal/FractalSync/PathOperations.lean) and
[`IntrinsicMetric.lean`](../../formal/FractalSync/IntrinsicMetric.lean) construct
an explicit `MetricSpace` whose distance is this infimum.

The same bounds identify its topology with the Euclidean topology. A closed
intrinsic ball of radius R about x lies inside the Euclidean ball of radius
`rho(x)(exp(R)-1)`. The distance-to-x function is Euclidean continuous, so the
intrinsic ball is Euclidean closed and compact. The topology identification
then proves properness of the actual intrinsic metric, and properness supplies
completeness for its own uniform structure. All these steps are checked in
[`IntrinsicTopology.lean`](../../formal/FractalSync/IntrinsicTopology.lean).
Equality of the two topologies does not assert equality of their uniformities
at infinity. These positive-floor results also hold in finite-dimensional real
normed spaces and for every real a.

**Checked continuation: minimizing curves.** The
[intrinsic-length extension](160-intrinsic-length.md) now identifies metric
curve length with (1) and proves the length-space approximation property.
It uses the ordinary product lift `x -> (x,a rho(x))`, a checked chord bound,
and the AC intrinsic-variation trace to recover the full graph-speed density.
This single proof handles every real a, including zero, without importing
a smooth hyperbolic embedding argument.

Nearly optimal paths are reparameterized using cumulative length. Their
bounded-speed families are compact in the pointwise product topology, so
properness yields an exact constant-speed metric geodesic. An explicit
ordinary Lipschitz bound makes it AC in the original norm, and the graph
length equals the endpoint intrinsic distance. This is the standard
proper-length-space principle [2, Theorem 2.5.23], now proved directly for
the actual definitions. Neither uniqueness nor a classical geodesic ODE
at a cut locus follows.

Simultaneously scaling S, x, y, and epsilon by any beta>0 preserves every
length in (1). Hence

$$
d^{\beta S}_{\beta\epsilon,a}(\beta x,\beta y)
=d^S_{\epsilon,a}(x,y).
\tag{13}
$$

This pairwise dilation statement remains conventional; the scalar radial
dilation law is checked separately.

## 5. The zero-floor limit splits the space

For a curve staying off S define the unregularized length

$$
L_{0,a}(\gamma)=\int
\frac{\sqrt{|\gamma'|^2+a^2|(d\circ\gamma)'|^2}}{d(\gamma)}dt.
$$

For x,y in one connected component of R^n minus S, define d_0,a as the
infimum of these actual integrals over all AC unit-interval paths from x to y
that avoid S at every time. The Lean definition is `zeroFloorDistance`, a real
`sInf` of `zeroFloorPathLengths`. The existence of such a path follows from
connectedness of the open component: AC reachability is relatively clopen
because small balls admit affine segments and AC paths concatenate. This is
checked in [`ACComponentPaths.lean`](../../formal/FractalSync/ACComponentPaths.lean).
Compactness of an off-Shore AC path supplies positive clearance from S, and
that clearance supplies integrability of its actual zero-floor density.
Thus the admissible class is nonempty and d_0,a is finite in this case.

The definition is used here only for nonempty admissible classes. A real
`sInf` on an empty class does not represent positive infinity; cross-component
divergence is stated separately as an `atTop` limit. Conventionally, for a=0
the componentwise metric is the usual quasihyperbolic metric, since the
component boundary has the same distance function there. At a>0 it includes
the graph correction.

**Theorem 3.** As epsilon decreases to zero:

1. For x,y in one complementary component,
   `d_epsilon,a(x,y)` increases to `d_0,a(x,y)`.
2. For x,y in different complementary components,
   `d_epsilon,a(x,y)` tends to infinity.
3. If x and y are distinct and at least one lies in S,
   `d_epsilon,a(x,y)` tends to infinity.
4. For x=y, including points of S, `d_epsilon,a(x,x)=0` for every epsilon>0.

These four cases give the complete classification for every fixed endpoint
pair. Their stated branches are checked in Lean.

**Proof of monotonicity.** The square of the density is

$$
\ell_\epsilon^2=
\frac{|\gamma'|^2}{d^2+\epsilon^2}
+\frac{a^2d^2(d')^2}{(d^2+\epsilon^2)^2}.
\tag{14}
$$

Both nonnegative terms increase as epsilon decreases. Taking infima preserves
monotonicity. Every path off S also satisfies `L_epsilon<=L_0`, so the same-component
limit is at most d_0.

**Proof of same-component equality, with a quantitative bound.** Choose any
off-S path from x to y with unregularized length at most B<infinity. Set

$$
q_\epsilon=\frac{\epsilon e^B}{d(x)}.
$$

Whenever q_epsilon<1,

$$
\boxed{(1-q_\epsilon^2)d_{0,a}(x,y)
\le d_{\epsilon,a}(x,y)\le d_{0,a}(x,y).}
\tag{15}
$$

To prove the lower bound, consider every admissible regularized candidate
gamma and split at the exact reference bound B. If `L_epsilon(gamma)<=B`,
the checked AC logarithmic-ruler calibration gives
`rho>=rho(x)exp(-B)>=d(x)exp(-B)>epsilon` at every time. Therefore gamma avoids
S. Put `t=d^2/rho^2`; then `t>=1-q_epsilon^2`. Comparing (14) with its
epsilon=0 version gives

$$
\ell_\epsilon^2=t A+t^2 C\ge t^2(A+C)=t^2\ell_0^2,
$$

where `A=|gamma'|^2/d^2` and `C=a^2(d')^2/d^2` are nonnegative.
Integrating gives `L_epsilon(gamma)>=(1-q_epsilon^2)L_0(gamma)` and hence
`L_epsilon(gamma)>=(1-q_epsilon^2)d_0,a(x,y)`. If instead
`L_epsilon(gamma)>B`, the same lower bound follows from `d_0,a(x,y)<=B`
and `0<=1-q_epsilon^2<=1`. The bound therefore holds for every candidate
and passes through its infimum. The upper bound follows by comparing
`L_epsilon<=L_0` on every off-Shore path. No minimizing curve or
approximate-minimizer choice is required.

The actual density comparisons, logarithmic confinement, integrability,
infimum sandwich, and resulting limit are checked in
[`ZeroFloorLimit.lean`](../../formal/FractalSync/ZeroFloorLimit.lean).
This proves convergence and an explicit relative O(epsilon^2) bound for fixed
endpoints with a known off-Shore path-length bound.
The factor exp(2B) makes this a conservative bound for long routes; it is not
a uniform rate over arbitrary endpoint pairs or a general efficient distance
approximation algorithm.

For different components every continuous connecting curve hits S: otherwise
its connected image in the complement would place both endpoints in the same
component. This topological bridge is checked in
[`ComponentCrossing.lean`](../../formal/FractalSync/ComponentCrossing.lean).
The unrestricted AC crossing bound applies to every candidate and passes
through the infimum, giving

$$
d_{\epsilon,a}(x,y)\ge
F_{\epsilon,a}(d(x))+F_{\epsilon,a}(d(y))\longrightarrow\infty.
\tag{16}
$$

The inequality and divergence in (16) are checked in `PairwiseDistance.lean`.
If x lies in S and x differs from y, the lower bound in (12) becomes
`log(1+|x-y|/epsilon)` and diverges. `DisplacementBounds.lean` checks this
argument; symmetry gives the other endpoint orientation in
`IntrinsicMetric.lean`, including the case of two distinct Shore points.
The constant path proves the diagonal case. These are pointwise finite-limit
and divergence statements. No uniform limit over arbitrary x,y or
Gromov--Hausdorff convergence is claimed.

## 6. Consequences for FractalSync and remaining obligations

The regularization floor is part of the cross-Shore geometry, not merely a
numerical denominator guard. Setting it to zero makes crossings between
complementary components infinitely long. This statement does not prescribe
a new epsilon value or modify the adopted metric.

Every Shore-crossing path gamma satisfies
`L_epsilon,a(gamma) >= F_epsilon,a(d(x))+F_epsilon,a(d(y))`, including when its
endpoints lie in the same component. This bounds that path's length, not the
unrestricted same-component pairwise infimum. If intrinsic speed is bounded by U>0,
then its duration is at least `[F(d(x))+F(d(y))]/U`. More generally, for a
duration T>0 and `K=|gamma'|_G^2/2`, Cauchy--Schwarz gives

$$
\int_0^T K\,dt\ge
\frac{[F(d(x))+F(d(y))]^2}{2T}.
\tag{17}
$$

At nonsmooth points define K as half the squared length density in (1); wherever
the differential metric exists this equals `v^T G v/2`. The integral argument
uses the length density and does not require a global smooth tensor G.

This bounds integrated kinetic energy along an admissible path. It is not
mechanical work, potential-energy expenditure, a sufficient crossing condition,
or an executable travel-time prediction under the actual Controls.

Lean now verifies the pointwise identity (5), the actual radial integral and
its closed primitive, the sharp constant in (9) and its optimality, and the
exact logarithmic scalar asymptotic. It also verifies nearest-point existence,
the actual nearest-segment length, and unrestricted absolutely continuous
endpoint/crossing bounds, deriving a.e. differentiation and integrability.
The actual path infimum to S equals F(d(x)) and is attained in either path
orientation. Its zero-floor divergence off S is checked. The duration bound
and kinetic-action consequence (17) are checked as well, with the explicit
finite-action hypothesis for (17).

The pairwise AC path infimum, reversal and concatenation, metric axioms,
quantitative comparisons, topology identification, properness, completeness,
and complete fixed-pair zero-floor classification are now checked. The
[pairwise continuation](160-pairwise-metric-limit.md) records the precise
statements and proof dependencies.

The [intrinsic-length extension](160-intrinsic-length.md) now checks metric
curve-length equality, the length-space property, arbitrary-endpoint
minimizing AC paths, and the full nearest-Shore segment classification in
strictly convex finite-dimensional normed spaces. Cut-locus results and the
curved-Shore bifurcation remain conventional, with the precise
coordinate-curvature boundary labeled in the
[flat-Shore note](160-flat-shore.md). The exact formal boundary is
maintained in [formal/README.md](../../formal/README.md).

The [flat Euclidean bridge](160-flat-intrinsic.md) now connects that explicit
coordinate metric to this AC path-infimum construction. The subsequent
[geodesic solution contracts](160-flat-geodesics.md) check conserved energy
and momentum and an actual Shore Jacobi field. The
[curved normal-coordinate calculation](160-curved-coordinate-geometry.md)
is separate from construction of a chart for a general embedded Shore.

## 7. Prior art and originality assessment

The inverse-distance conformal length is classical quasihyperbolic geometry
[1]. Proper-length-space existence of minimizers is standard [2]. Inverse-distance
squared condition metrics also appear in numerical-analysis geometry [3]; its
self-convexity results require hypotheses such as a complete C^2 submanifold
and a unique-nearest-point region. They do not transfer automatically to an
arbitrary closed fractal set, a positive regularization floor, or the graph correction.

The closed form, sharp additive constant, and estimate (15) have been derived
here for this exact family. Targeted searches did not establish their prior
publication status. Their elementary proofs are not evidence of novelty. The
defensible outcome is a useful, self-contained theorem package with an open
literature comparison, not an announced new mathematical result.

### References

1. F. W. Gehring and B. P. Palka, *Quasiconformally homogeneous domains*,
   Journal d'Analyse Mathematique 30 (1976), 172--199.
   [DOI](https://doi.org/10.1007/BF02786713).
   Bibliographic reference; the original full text was not retrieved in this session.
2. D. Burago, Y. Burago, S. Ivanov, *A Course in Metric Geometry*, AMS,
   Graduate Studies in Mathematics 33 (2001), Section 2.5.
   [Publisher](https://doi.org/10.1090/gsm/033).
3. C. Beltran, J.-P. Dedieu, G. Malajovich, M. Shub,
   *Convexity Properties of the Condition Number*, SIAM Journal on Matrix
   Analysis and Applications 31 (2010), 1491--1506.
   [DOI](https://doi.org/10.1137/080718681),
   [author manuscript](https://personales.unican.es/beltranc/archivos/LogConvexCN.pdf).
   The author manuscript's Theorem 2 and its unique-nearest-point hypotheses
   were inspected.
