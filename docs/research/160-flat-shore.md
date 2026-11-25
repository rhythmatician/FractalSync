# Flat-Shore curvature, the cut locus, and curved-Shore bifurcation

Status: mathematical research for #160, with scalar Lean support from
`formal/FractalSync/FlatShore.lean` and an actual global-coordinate connection
and curvature calculation in `formal/FractalSync/FlatShoreGeometry.lean`.
The planar flat curvature identification is checked through the actual
covariant-derivative commutator. The [Euclidean flat bridge](160-flat-intrinsic.md)
now identifies its coordinate integral with the unrestricted AC path metric.
[Smooth solution contracts](160-flat-geodesics.md) check energy and momentum
conservation and the actual Shore Jacobi field with first positive zero pi.
The [curved normal-coordinate extension](160-curved-coordinate-geometry.md)
checks curvature for a supplied varying-curvature chart metric. The global
cut-locus/minimizer arguments, construction of a chart for an embedded Shore,
and Banach-space bifurcation reduction below remain conventional. No bundled
manifold curvature tensor, cut locus, or Lyapunov–Schmidt theorem is claimed.
No runtime or training contract changes.

The unrestricted flat model is exactly tractable. Its global cut-locus structure
is a specialization of existing results for reflective warped-product surfaces,
not an original general cut-locus theorem. The explicit formulas below make that
specialization useful for FractalSync.

## 1. Model and scalar curvature calculation

Let the Shore be the line `r=0`, with Euclidean tangential coordinate `y`. Put

$$
\rho=\sqrt{\epsilon^2+r^2},\quad b=\sqrt{1+a^2},\quad
Q=\sqrt{\epsilon^2+b^2r^2},\qquad \epsilon>0,\ a\ge0.
$$

The induced metric, on the **whole plane**, is

$$
g=\frac{Q^2}{\rho^4}\,dr^2+\rho^{-2}\,dy^2.
\tag{1}
$$

Define the signed intrinsic normal coordinate
`t(r)=sgn(r) F_{epsilon,a}(|r|)`. The exact radial result in the companion
[distance note](160-distance-geometry.md) gives `dt/dr=Q/rho²>0` and shows
`t:R→R` is an odd diffeomorphism. Equation (1) becomes

$$
g=dt^2+f(t)^2dy^2,\qquad f(t(r))=\rho(r)^{-1}.
$$

Differentiating explicitly,

$$
f_t=-\frac{r}{\rho Q},\qquad
\frac{f_{tt}}f=\frac{b^2r^4-\epsilon^4}{Q^4}.
$$

The standard warped-product curvature formula `K=-f_tt/f` therefore gives

$$
\boxed{K_{\rm rad}(r)=
\frac{\epsilon^4-b^2r^4}{(\epsilon^2+b^2r^2)^2}.}
\tag{2}
$$

For the analogous flat hyperplane in dimension at least three, purely
tangential sectional planes have

$$
K_{\rm tan}(r)=-\frac{r^2}{\epsilon^2+b^2r^2}.
\tag{3}
$$

Several useful exact facts follow:

$$
-\frac1{b^2}<K_{\rm rad}\le1,\qquad
-\frac1{b^2}<K_{\rm tan}\le0,\qquad
K_{\rm rad}-K_{\rm tan}=\frac{\epsilon^2\rho^2}{Q^4}>0,
$$

and

$$
\frac{dK_{\rm rad}}{dr}
=-\frac{4\epsilon^2b^2r(\epsilon^2+r^2)}
        {(\epsilon^2+b^2r^2)^3}<0\quad(r>0).
\tag{4}
$$

Thus curvature is strictly decreasing along either outward normal ray. It is
`+1` on Shore, zero at `|r|=epsilon/sqrt(b)`, and approaches `-1/b²` at infinity.
The intrinsic half-width of its positive-curvature strip is
`F_{epsilon,a}(epsilon/sqrt(b))`, independent of epsilon by dilation covariance.
The ambient hyperbolic metric does not make the induced graph intrinsically
constant-curvature.

In two dimensions the positive curvature-area integral across this strip, per
unit Euclidean tangential length, is `2/[epsilon(1+b)]`. Indeed `K f=-f_tt`,
and `-f_t` reaches its maximum `1/[epsilon(1+b)]` at its edge. The signed
curvature-area integral across the whole normal line is zero because `f_t`
tends to zero at both ends. These are area integrals, not trajectory integrals.

### Checked planar connection and curvature identification

The new coordinate module works on `Coord = R x R`, with the displayed
bilinear metric (1). Its product norm is used only for ordinary Fréchet
calculus; the geometric pairings are always made with (1). Lean checks
positive definiteness for every \(\epsilon>0\), every real \(a\), and every
signed normal coordinate \(r\).

The nonzero coordinate Christoffel coefficients are
\[
\Gamma^r_{rr}=\frac{b^2r}{Q^2}-\frac{2r}{\rho^2},\qquad
\Gamma^r_{yy}=\frac r{Q^2},\qquad
\Gamma^y_{ry}=\Gamma^y_{yr}=-\frac r{\rho^2}.
\]
They are connected to actual metric derivatives, not only to an algebraic
array named a connection. For actual coordinate fields define
\[
\nabla_XY(x)=dY_x[X(x)]+\Gamma_x(X(x),Y(x)).
\]
`covariantDerivative` uses Lean's actual `fderiv`. The module checks
zero torsion against Mathlib's `VectorField.lieBracket`, linearity in the
first direction, additivity and the Leibniz rule in the differentiated field
under their actual differentiability premises, and genuine metric compatibility: differentiating the
pairing of differentiable vector fields along a curve tangent to \(X\)
gives the sum of the two covariant-derivative pairings.

The curvature definition is the actual commutator
\[
R(X,Y)Z=\nabla_X\nabla_YZ-\nabla_Y\nabla_XZ-\nabla_{[X,Y]}Z.
\]
For the constant coordinate fields \(e_r,e_y\), the checked computation is
\[
R(e_r,e_y)e_y=\frac{K_{\rm rad}(r)}{\rho^2}\,e_r.
\]
The module then uses the actual metric Gram determinant to prove
\[
\boxed{
\frac{g(R(e_r,e_y)e_y,e_r)}
     {g(e_r,e_r)g(e_y,e_y)-g(e_r,e_y)^2}
=\frac{\epsilon^4-b^2r^4}{(\epsilon^2+b^2r^2)^2}.
}
\]
`sectionalCurvature_eq_radialCurvature` is this identification. The
commutator proof differentiates both nonconstant first covariant derivatives
and checks the zero bracket. Thus the sign convention gives curvature
\(+1\) at \(r=0\), consistent with (2); `sectionalCurvature_on_shore`
packages that specialization explicitly.

Source: [FlatShoreGeometry.lean](../../formal/FractalSync/FlatShoreGeometry.lean).
This is a checked curvature calculation in the specified global plane
chart. The [affine-coordinate extension](160-affine-covariance.md) now
recomputes the actual connection, bracket and curvature commutator under
every invertible affine chart and proves the same curvature Gram ratio.
Nonlinear-chart and variable-frame naturality, arbitrary-extension
tensoriality, a bundled Levi-Civita object, higher-dimensional identification
of (3), and the embedded-Shore argument below remain separate.
The separate [Euclidean flat bridge](160-flat-intrinsic.md) now proves the
coordinate integral, graph integral, actual endpoint infima and metric curve
length agree on the full AC path class. Its ambient space is `WithLp 2 (ℝ × ℝ)`;
the global coordinate chart preserves absolute continuity without being
misidentified as an isometry for the coordinate product's default norm.
The [geodesic extension](160-flat-geodesics.md) supplies actual solution-level
conservation laws and the Shore Jacobi equation. Those results do not yet
prove smoothness of every AC minimizer or the global cut locus.

## 2. Universal curvature at a smooth Shore

The following embedded-Shore and higher-dimensional argument remains
conventional. A separate [checked coordinate calculation](160-curved-coordinate-geometry.md)
now establishes the planar result for the explicit normal-chart metric
with a varying curvature profile, under its stated local derivative hypotheses.
The construction of that chart from an arbitrary embedded Shore has not
been formalized.

For a general Shore assume a tubular neighborhood with a **C³ signed-distance
function** `D`; this is a sufficient classical regularity hypothesis, not a
claim about a fractal boundary or a cut locus. Write `n=grad D` on Shore,
`rho=sqrt(D²+epsilon²)`, `h=I+a² d rho⊗d rho`, and `g=rho⁻² h`.
At Shore,

$$
d\rho=0,\qquad \operatorname{Hess}\rho=\epsilon^{-1}n\otimes n.
$$

The Euclidean graph metric `h` has zero intrinsic curvature tensor there: the
second fundamental form of the graph `a rho` has rank at most one, and its
Gauss quadratic terms cancel. This also holds at `a=0`. For
`phi=-log rho`, its gradient vanishes and
`Hess_h phi=-epsilon⁻² n⊗n`. The conformal curvature formula gives, for
Euclidean orthonormal `X,Y`,

$$
\boxed{K_g(X\wedge Y)=\langle X,n\rangle^2+\langle Y,n\rangle^2.}
\tag{5}
$$

Consequently Gaussian curvature in dimension two equals `+1` at every smooth
Shore point, independently of the Shore's Euclidean curvature, epsilon, and a.
Purely tangent sectional planes in higher dimensions have curvature zero.
Neither statement extends automatically to nonsmooth points.

A curved Shore itself need not be a geodesic. This is separate from its
universal induced sectional curvature at its points; see Section 5.

## 3. Flat Shore geodesics and an exact shortcut threshold

For a unit-speed geodesic of (1), translation in `y` conserves
`p=y'/rho²`. For endpoints with positive tangential separation, `p>0`, and

$$
(r')^2=\frac{\rho^4(1-p^2\rho^2)}{Q^2}.
\tag{6}
$$

A non-Shore excursion from Shore back to Shore reaches a turning radius
`R>0`, where `p=1/sqrt(epsilon²+R²)`. Integrating `dy/dr` and substituting
`r=R sin(theta)` gives its tangential displacement

$$
\boxed{\mathcal L(R)=2\int_0^{\pi/2}
\sqrt{\epsilon^2+b^2R^2\sin^2\theta}\,d\theta.}
\tag{7}
$$

It is continuous and strictly increasing for `R>=0`, starts at
`mathcal L(0)=pi epsilon`, and tends to infinity. Strictness follows from
strict increase of the integrand on the open angle interval when the radius
increases. Its common excursion length on either side is

$$
\mathcal D(R)=2\sqrt{\epsilon^2+R^2}
\int_0^{\pi/2}
\frac{\sqrt{\epsilon^2+b^2R^2\sin^2\theta}}
     {\epsilon^2+R^2\sin^2\theta}\,d\theta.
\tag{8}
$$

**Theorem 1, Shore-to-Shore minimizers.** For two Shore points whose Euclidean
tangential separation is `L>0`:

* If `L<=pi epsilon`, the straight Shore segment is the unique minimizing
  curve up to monotone reparameterization, including equality.
* If `L>pi epsilon`, exactly two minimizing curves exist up to monotone
  reparameterization. They are reflected one-excursion geodesics, with the
  unique radius `R>0` satisfying `mathcal L(R)=L`.

**Proof.** The unrestricted metric is proper by the distance note, so a
minimizer exists. Smoothness makes it a geodesic. A nonconstant radial
component cannot touch Shore tangentially, because uniqueness for the smooth
geodesic ODE would then force it to remain in Shore. Every excursion has
displacement (7), strictly exceeding `pi epsilon`. This rules out an
alternative geodesic below or at the threshold.

The straight unit-speed Shore geodesic has a parallel unit normal `N` and
the normal Jacobi field `sin(s)N`. Its actual Jacobi equation and first positive field zero pi are
now [Lean checked](160-flat-geodesics.md). The identification of this zero
with the first conjugate point and the following index-form/minimality
arguments remain conventional. Its first conjugate point is at intrinsic length pi. For a segment
of length `ell=L/epsilon>pi`, the normal test variation
`sin(pi s/ell)N` has index form
`(ell/2)[(pi/ell)²-1]<0`, so the straight segment cannot minimize.
A minimizing non-Shore geodesic cannot have more than one excursion:
reflecting each negative excursion into the positive half-plane preserves
length but creates a corner at every interior Shore crossing, impossible for
a minimizing curve in a smooth Riemannian metric. Strict monotonicity of (7)
and ODE uniqueness give one one-excursion curve on each side. Reflection makes
their lengths equal. Existence then proves both minimize. ∎

This argument concerns all admissible curves, not merely paths that are graphs
of `r(y)`. In higher dimensions conserved tangential momentum fixes a plane
containing the normal direction and the endpoint displacement, giving the same
Shore-to-Shore result.

### Quantitative emergence of the shortcuts

Put `u=R/epsilon`, `ell=L/epsilon`, and let `D(ell)` denote the minimizing
intrinsic distance for `ell>pi`. Taylor expansion of (7), with a uniform
remainder on its compact integration interval, gives

$$
\ell=\pi+\frac{\pi b^2}{4}u^2+O(u^4),\qquad
u\sim\frac2b\sqrt{\frac\ell\pi-1}.
\tag{9}
$$

The first variation of length in the endpoint gives
`D'(ell)=1/sqrt(1+u(ell)²)`. Hence the *saving* relative to the Shore segment is

$$
\boxed{\ell-D(\ell)=\frac{(\ell-\pi)^2}{\pi b^2}
+o((\ell-\pi)^2).}
\tag{10}
$$

The paths separate with square-root amplitude, while the saving starts
quadratically. At large separation, dominated convergence in (7) gives
`mathcal L(R)/R→2b`, so `R/L→1/(2b)`.

## 4. The entire two-dimensional cut locus is already covered by prior art

Chitsakul [2, Proposition 5.4] identifies the reflected-parallel cut locus of
the universal covering plane in the globally decreasing-curvature case.
Boonnam [1, Theorem B] gives a later formulation for complete planes
`dt²+m(t)²dy²` with positive even `m`, allowing curvature that only decreases
until a nonpositive tail.
Our `m=f` satisfies its hypotheses: equation (4) gives strict curvature
monotonicity for all `t>0`, and `f_t<0` for every `t>0`, so the theorem's
`t0=sup{t>0:m'(t)<0}` is infinity.

For a source `(r0,y0)`, its half-period formula is exactly (7) with radius
`|r0|`. Thus the complete cut locus in the full flat plane is

$$
\boxed{\operatorname{Cut}(r_0,y_0)=
\{(-r_0,y): |y-y_0|\ge\mathcal L(|r_0|)\}.}
\tag{11}
$$

At `r0=0`, the earlier theorem independently supplies the exact endpoint
value `mathcal L(0)=pi epsilon`. Chitsakul's Lemma 4.5, together with reflection,
excludes off-Shore cut points for a Shore source; this avoids substituting an
endpoint into a half-period formula originally defined on an open interval.
This includes the conjugate endpoints for a Shore source, even though Theorem 1
gives a unique minimizing curve at equality. Away from the reflected parallel
the source has no cut points. Equation (11) is a stronger and more precise
conclusion than a Shore-to-Shore shortcut threshold alone.

The globally decreasing-curvature case was already treated by Chitsakul [2];
Boonnam explicitly identifies it as prior work. Accordingly, neither the
general cut-locus structure nor the method through Clairaut half-periods is a
novel contribution here. Equations (2), (7), and (11) provide an explicit
specialization and reproducible benchmark for the adopted family.

This theorem concerns the whole smooth plane. A restricted GeometryProvider
domain can remove minimizing paths, and a curved or fractal Shore generally
breaks the reflection and translation symmetries used throughout.

## 5. Curved Shore: a rigorous local cubic-root branch

This section upgrades the one-mode calculation in the issue discussion to a
local conventional result. It does not identify a global cut locus for a
circle or an arbitrary curved Shore.

In signed tubular coordinates `(s,r)` with `N'=-kappa T`, a smooth planar
Shore of curvature `kappa(s)` has

$$
g_{rr}=\frac{\epsilon^2+b^2r^2}{(\epsilon^2+r^2)^2},\qquad
g_{ss}=\frac{(1-\kappa(s)r)^2}{\epsilon^2+r^2}.
$$

At Shore, `Gamma^r_ss=kappa`. A Shore segment with nonzero curvature is
therefore not a geodesic. For a constant-curvature segment set
`x=s/epsilon`, `q=r/epsilon`, `delta=epsilon kappa`, `ell=L/epsilon`.
The exact length functional for a graph path in the tubular chart is

$$
J_{\ell,\delta}(q)=\int_0^\ell
\sqrt{\frac{1+b^2q^2}{(1+q^2)^2}(q')^2+
      \frac{(1-\delta q)^2}{1+q^2}}\,dx,
\qquad q(0)=q(\ell)=0.
\tag{12}
$$

For fixed `0<ell<pi`, the linearized equation is
`q''+q=-delta`. Its Dirichlet solution and the analytic implicit function
theorem give the unique nearby critical branch

$$
q(x)=\delta\left[\frac{\cos(x-\ell/2)}{\cos(\ell/2)}-1\right]+O(\delta^2).
\tag{13}
$$

This branch is a strict local minimum by the Dirichlet Poincare inequality.
For positive curvature it bends toward the center of curvature. The response
is not uniform as `ell` approaches pi.

### Theorem 2, the local branch at the flat threshold

For fixed `a>=0`, at `ell=pi`, sufficiently small signed `delta` gives a
unique sufficiently small critical graph of (12). Writing its sine coefficient
as `A`, the branch satisfies in `H²(0,pi)`

$$
\boxed{
q=A\sin x+O(A^3),\qquad
\delta=\frac{\pi b^2}{8}A^3+O(A^5).}
\tag{14}
$$

Consequently, with the real cube root,

$$
A\sim\left(\frac{8\delta}{\pi b^2}\right)^{1/3}.
\tag{15}
$$

For nonzero sufficiently small delta the branch is a strict local length
minimum among H²-small graph perturbations. The estimates here are local near
the zero graph, for fixed `a`.

**Proof.** Rescale the domain to `[0,pi]` by `x=ell t/pi`, and set
`mu=ell-pi`. On `X=H²∩H¹₀`, the Euler–Lagrange gradient of (12) is an analytic
map to `Y=L²` near `(q,mu,delta)=(0,0,0)`: one-dimensional Sobolev embedding
makes `q,q'` small uniformly, and the square-root argument stays positive.
Its linearization is `A0=-d²/dt²-1`. The kernel is spanned by `phi=sin t`;
on its L²-orthogonal complement the Dirichlet spectrum is `n²-1>=3`.

Write `q=A phi+w`, with `w` orthogonal to phi. The analytic implicit function
theorem solves the complementary Euler–Lagrange equation uniquely for
`w=w(A,mu,delta)`. It gives
`w=O(|delta|+|mu A|+|A|³)`. Reflection gives
`w(-A,mu,-delta)=-w(A,mu,delta)`.

At `mu=delta=0`, the fourth-order integrand on `q=A sin t` is

$$
A^4\left[\frac{b^2-2}{2}\sin^2t\cos^2t+
\frac12\sin^4t-\frac18(\cos^2t-\sin^2t)^2\right].
$$

Its integral equals `pi b² A⁴/16`. The correction `w=O(A³)` cannot alter
this coefficient: the quadratic cross-term with the kernel is zero and all
remaining corrections start at degree six. The reduced energy has leading
terms `-2 delta A-mu A²/2+pi b² A⁴/16` plus an A-independent constant and
higher analytic terms. Its derivative is

$$
g(A,\mu,\delta)=\frac{\pi b^2}{4}A^3-\mu A-2\delta+
O\bigl(A^5+|\mu|A^3+\mu^2|A|+|\delta|A^2+
|\mu\delta|+\delta^2|A|+|\delta|^3\bigr).
\tag{16}
$$

The stated remainder follows from analytic Taylor expansion and the symmetry
`g(-A,mu,-delta)=-g(A,mu,delta)`; the first displayed coefficients were
computed above. At `mu=0`, `g_delta(0,0,0)=-2`, so the implicit function theorem
solves `delta=d(A)` analytically. Its odd expansion is exactly (14). The
positive cubic coefficient implies `d'(A)>0` for all sufficiently small
nonzero A, hence strict monotonicity through zero and uniqueness of the small
branch. Substitution into the complementary solution gives the first part
of (14).

Finally, at this branch the reduced Hessian is
`g_A=3 pi b² A²/4+O(A⁴)>0` for A nonzero. On the complementary subspace,
`∫(v'²-v²)>=3/4 ∫v'²` by the Dirichlet spectrum, so the complementary
Hessian retains coercivity in H¹. Its Schur complement is the positive
reduced Hessian. For each fixed nonzero sufficiently small A, the full second
variation consequently bounds `c(A)||v||²_H¹` for some `c(A)>0`. The coefficients
of the second variation vary continuously with the graph in C¹. H²-small
perturbations are C¹-small, so this lower bound persists, with `c(A)/2`, along
the interpolation to a sufficiently nearby graph. The integral Taylor formula
therefore gives a strict local length minimum. The size of this neighborhood
may shrink as A tends to zero. This establishes (14)–(15) for actual nearby
critical graph paths, not just a one-mode ansatz. ∎

There is no global minimizing claim here. Other paths may leave the tubular
chart or the small graph neighborhood, and the potential, friction, and
Controls change the dynamical problem. A separate global analysis is needed
before interpreting this local bifurcation as a curved-Shore cut locus.

## References and formal boundary

1. N. Boonnam, *On the Structure of the Cut Locus in Certain Classes of Warped
   Product Manifolds*, arXiv:2610.15279, 2026, Theorem B and Proposition 4.5.
   [Primary manuscript](https://arxiv.org/abs/2610.15279).
2. P. Chitsakul, *The Structure Theorem for the Cut Locus of a Certain Class of
   Cylinders of Revolution I*, Tokyo Journal of Mathematics 37 (2014),
   473–484; preprint arXiv:1311.7475 (2013), Proposition 5.4, Lemma 4.5,
   and Section 3. [Primary manuscript](https://arxiv.org/abs/1311.7475).
   The universal-cover argument was inspected directly. The preprint has
   one-sided notation in the displayed Proposition 5.4; its reflection
   argument and [1, Theorem B] give the two-sided form used here.

The scalar identities and inequalities and the planar coordinate connection
and curvature identification whose exact Lean names appear in
`formal/README.md` are kernel checked. Arbitrary-endpoint AC minimizing
curves and the complete nearest-Shore segment classification are also now
checked in [the intrinsic-length extension](160-intrinsic-length.md).
The flat AC coordinate/path-metric correspondence, smooth-solution first
integrals, actual Shore Jacobi equation and first positive field zero are
checked in the linked extensions. Actual affine covariance is also checked.
The supplied curved normal-chart metric has a checked connection, full
off-Shore curvature and actual normal curvature derivative at Shore. The further
arguments invoking embedded-chart construction, general Jacobi/conjugacy
theory, smooth minimizing-curve regularity, the cut-locus theorem, Sobolev
spaces, or the analytic implicit function theorem are **not** reported as
Lean formalizations. Existing theorem reuse is a
mathematical proof step, not a machine-checked dependency until formalized.
