# Regularized distance graphs: exact radial distance and a singular limit

Research progress for [#160](https://github.com/rhythmatician/FractalSync/issues/160),
2026-11-24. These are exact mathematical results under the assumptions below.
The accompanying Lean module checks a smaller pointwise algebraic subset, not
the complete length-space or limiting arguments. No novelty claim is made.

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

All curves in R^n are admissible here. A runtime validity domain is an additional
restriction and can change the distance. No differentiable boundary, unique
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

## 4. Properness and minimizing paths do not require a smooth Shore

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

**Proof.** For a path of Euclidean arclength ell beginning at x,
`rho(gamma(s)) <= rho(x)+s`. Its weighted length is therefore at least
`log(1+ell/rho(x))`, which gives the lower bound. The straight segment and
`rho>=epsilon` give the upper bound. Inequality (11) follows pointwise from
`|rho'|<=|gamma'|`.

Equation (12) shows that d_epsilon,a is a genuine finite metric and induces the
Euclidean topology. A closed metric ball of radius R about x lies inside the
Euclidean ball of radius `rho(x)(exp(R)-1)`. It is Euclidean closed, hence compact.
Thus the space is proper and complete. It is a length space by construction;
the length-space Hopf--Rinow theorem gives a minimizing curve between any two
points [2, Theorem 2.5.23]. Local equivalence with Euclidean length makes such a
curve Euclidean absolutely continuous after arclength parameterization. Its
metric length equals the functional (1): for a>0 embed the graph by
`x -> (x,a rho(x))` in scaled hyperbolic upper half-space. The ambient distance
is bounded above by d_epsilon,a, whereas each d_epsilon,a increment is bounded
above by the corresponding subpath's ambient length. Taking partition suprema
gives equality of the two curve lengths. For a=0 the same equality is the
standard continuous positive weighted-length construction. Thus the minimizing
curve is admissible for (1). Neither uniqueness nor a classical geodesic ODE
at a cut locus follows.

Simultaneously scaling S, x, y, and epsilon by any beta>0 preserves every
length in (1). Hence

$$
d^{\beta S}_{\beta\epsilon,a}(\beta x,\beta y)
=d^S_{\epsilon,a}(x,y).
\tag{13}
$$

## 5. The zero-floor limit splits the space

For a curve staying off S define the unregularized length

$$
L_{0,a}(\gamma)=\int
\frac{\sqrt{|\gamma'|^2+a^2|(d\circ\gamma)'|^2}}{d(\gamma)}dt.
$$

Let d_0,a be its intrinsic metric within each connected component of R^n minus S.
Such a component is open and polygonally connected, so the distance between
any two of its points is finite. For a=0 this is the usual quasihyperbolic
metric of the component; the component boundary has the same distance function
there. At a>0 it includes the graph correction.

**Theorem 3.** As epsilon decreases to zero:

1. For x,y in one complementary component,
   `d_epsilon,a(x,y)` increases to `d_0,a(x,y)`.
2. For x,y in different complementary components,
   `d_epsilon,a(x,y)` tends to infinity.
3. If x and y are distinct and at least one lies in S,
   `d_epsilon,a(x,y)` tends to infinity.

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

Indeed choose an epsilon-minimizing curve, whose length is at most B by the
previous upper bound. Along it, `|(log rho)'|<=ell_epsilon`, so
`rho>=rho(x)exp(-B)>=d(x)exp(-B)>epsilon`. The curve avoids S and remains in
the original component. Put `t=d^2/rho^2`. Then `t>=1-q_epsilon^2`. Comparing
(14) with its epsilon=0 version gives

$$
\ell_\epsilon^2=t A+t^2 C\ge t^2(A+C)=t^2\ell_0^2,
$$

where `A=|gamma'|^2/d^2` and `C=a^2(d')^2/d^2` are nonnegative.
Integrate to obtain (15). This proves convergence and an explicit relative
O(epsilon^2) bound for fixed endpoints with a known off-S path-length bound.
The factor exp(2B) makes this a conservative bound for long routes; it is not
a general efficient distance approximation algorithm.

For different components every connecting curve hits S, and (3)--(4) on the
two portions give

$$
d_{\epsilon,a}(x,y)\ge
F_{\epsilon,a}(d(x))+F_{\epsilon,a}(d(y))\longrightarrow\infty.
\tag{16}
$$

If x lies in S, the lower bound in (12) is
`log(1+|x-y|/epsilon)`, proving the third claim. This is convergence of pointwise
distances to an extended metric, not a claimed Gromov--Hausdorff theorem.

## 6. Consequences for FractalSync and remaining obligations

The regularization floor is part of the cross-Shore geometry, not merely a
numerical denominator guard. Setting it to zero makes crossings between
complementary components infinitely long. This statement does not prescribe
a new epsilon value or modify the adopted metric.

For any path that hits the Shore, (16) is a length lower bound even when its
endpoints are in the same component. If intrinsic speed is bounded by U>0,
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

The pointwise identity (5), density comparisons, and rational regularization
algebra are tractable Lean targets. Theorems about path integration, nearest
points, length-space completeness, and the singular limit remain conventional
proofs in this increment. A full formalization must explicitly supply the
absolute-continuity, path-length, and infimum machinery.

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
