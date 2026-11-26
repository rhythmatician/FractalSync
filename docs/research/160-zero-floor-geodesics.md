# #160: attained zero-floor geodesics and convergence of selected minimizers

The same-component zero-floor limit now includes **actual minimizing paths**.
The actual off-Shore AC path infimum is attained, and every selected sequence
of positive-floor constant-speed geodesics has a subsequence converging
uniformly to a constant-speed zero-floor minimizer. The proof does not assume
uniqueness or convergence of derivatives.

The increment contains twelve proved theorem declarations:

| Source | Declarations | Checked scope |
| --- | ---: | --- |
| [ZeroFloorGeodesics.lean](../../formal/FractalSync/ZeroFloorGeodesics.lean) | 11 | Actual clearance and zero-floor integral bounds, uniform subsequence extraction, constant-speed zero-floor attainment, and convergence of selected minimizers' zero-floor lengths |
| [Analysis/BoundedSpeedCompactness.lean](../../formal/FractalSync/Analysis/BoundedSpeedCompactness.lean) | 1 | Uniformly convergent subsequences of endpoint paths with a common speed bound in a proper metric space |

This extends the [fixed-pair zero-floor limit](160-pairwise-metric-limit.md)
and [positive-floor intrinsic length and attainment](160-intrinsic-length.md).
It uses their existing integral and path-infimum definitions.

## 1. Setting and exact conclusion

Let \(E\) be a finite-dimensional real normed space, \(S\subset E\) a
nonempty closed set, and \(a\in\mathbb R\). Fix \(x\notin S\) and
\(y\) in the connected component of \(E\setminus S\) containing \(x\).
Write

\[
d(z)=\operatorname{infDist}(z,S),\qquad
D=D_{0,a}(x,y),\qquad m=d(x)e^{-D}>0.
\]

Here \(D_{0,a}\) is the existing `zeroFloorDistance`: the actual infimum of
`acGraphLength S 0 a` over ordinary AC paths whose whole unit-interval image
avoids \(S\). The same-component theorem already supplies a nonempty
admissible class and a finite value. The result makes no smoothness,
convexity, strict convexity, or nearest-point uniqueness assumption on Shore.

`exists_zeroFloor_minimizer` now proves that there is an ordinary AC path
\(\gamma:[0,1]\to E\) satisfying

\[
\gamma(0)=x,\qquad \gamma(1)=y,
\qquad \gamma(t)\in\operatorname{Comp}_{E\setminus S}(x),
\]

\[
\boxed{d(\gamma(t))\ge m\qquad(0\le t\le1),}
\tag{1}
\]

\[
\boxed{L_{0,a}(\gamma)=D_{0,a}(x,y),}
\tag{2}
\]

and

\[
\boxed{
D_{0,a}(\gamma(s),\gamma(t))=D_{0,a}(x,y)|s-t|
\qquad(0\le s,t\le1).
}
\tag{3}
\]

The named theorem `zeroFloorDistance_isLeast` states attainment directly as
`IsLeast` in the original `zeroFloorPathLengths` set. The path is admissible
for that original integral, rather than only for a newly introduced abstract
metric. Coincident endpoints are included; equation (3), together with the
positive-floor metric comparison, forces the constant trace in that case.

## 2. Uniform subsequences for arbitrary selections

Let \(\epsilon_n>0\) tend to zero, with no monotonicity assumption. For each
\(n\), select any constant-speed positive-floor metric geodesic \(f_n\)
between the fixed endpoints, parameterized on \([0,1]\):

\[
D_{\epsilon_n,a}(f_n(s),f_n(t))
=D_{\epsilon_n,a}(x,y)|s-t|.
\tag{4}
\]

The earlier positive-floor package supplies such geodesics and proves their
ordinary AC admissibility and graph-length minimality.

`selected_geodesics_subsequence_zero_floor` proves that there is a strictly
increasing index map \(\varphi:\mathbb N\to\mathbb N\) and a path satisfying
(1)--(3) such that

\[
\boxed{f_{\varphi(n)}\longrightarrow\gamma
\quad\text{uniformly on }[0,1]\text{ in the original ambient metric}.}
\tag{5}
\]

The Lean conclusion is the actual `TendstoUniformly` predicate on
`unitInterval`, with a real-parameter AC representative of the limit.
There is no premise choosing a particularly favorable family of geodesics.
Different subsequences or selections can converge to different minimizers.

### The compactness argument

Eventually \(\epsilon_n\le1\). Floor monotonicity and the existing
zero-floor upper comparison give

\[
D_{1,a}(f_n(s),f_n(t))
\le D_{\epsilon_n,a}(f_n(s),f_n(t))
\le D|s-t|.
\tag{6}
\]

The already checked intrinsic displacement and ruler estimates turn (6)
into a common ordinary Lipschitz bound

\[
K=\rho_1(x)e^{2D}D,\qquad
\rho_1(x)=\sqrt{1+d(x)^2}.
\tag{7}
\]

Every path is therefore contained in the fixed ordinary closed ball
\(\overline B(x,K)\). Finite dimensionality makes that ball compact.

The generic support theorem applies the pinned Mathlib
[bounded-continuous-function Arzelà–Ascoli theorem](https://github.com/leanprover-community/mathlib4/blob/f897ebcf72cd16f89ab4577d0c826cd14afaafc7/Mathlib/Topology/ContinuousMap/Bounded/ArzelaAscoli.lean)
to the closed family of endpoint paths with common Lipschitz bound. Its
metric function-space topology gives an actual uniformly convergent
subsequence through `IsCompact.tendsto_subseq`. This avoids treating
pointwise compactness of an uncountable function product as sequential
compactness. Discarding the finite initial part of the floor sequence is
included in the formal argument.

## 3. Passing through every fixed positive floor

Fix \(\delta>0\). Eventually \(\epsilon_{\varphi(n)}\le\delta\), hence

\[
D_{\delta,a}(f_{\varphi(n)}(s),f_{\varphi(n)}(t))\le D|s-t|.
\]

The positive-floor intrinsic metric induces the original ambient topology.
The new helper `continuous_intrinsicDistance_pair` packages its joint
continuity in the two endpoints. Passing to the uniformly convergent
subsequence therefore gives

\[
D_{\delta,a}(\gamma(s),\gamma(t))\le D|s-t|
\quad\text{for every }\delta>0.
\tag{8}
\]

At \(\delta=1\), the ordinary Lipschitz estimate makes the limit an ordinary
AC path. The existing metric-length bridge then proves

\[
L_{\delta,a}(\gamma)\le D\qquad(\delta>0).
\tag{9}
\]

The logarithmic ruler calibration yields

\[
d(x)e^{-D}\le\sqrt{\delta^2+d(\gamma(t))^2}.
\]

Taking \(\delta\downarrow0\) proves (1), so the limit avoids Shore with
strictly positive clearance. Its connected image stays in the original
complement component.

The existing quantitative density comparison now applies to this same
limit path. For \(0<\delta<m\),

\[
\left(1-\frac{\delta^2}{m^2}\right)L_{0,a}(\gamma)
\le L_{\delta,a}(\gamma)\le D.
\tag{10}
\]

Letting \(\delta\downarrow0\) gives \(L_{0,a}(\gamma)\le D\); admissibility
gives the opposite inequality and proves (2). This argument uses actual
integrability and the existing scalar density comparison. It does not assume
weak convergence of derivatives or a lower-semicontinuity theorem for the
derivative-dependent integral.

For every pair of points on the limit curve, continuity and Shore avoidance
supply membership in one complement component. The existing fixed-pair
zero-floor limit transfers (8) to the zero-floor increment bound. The
positive-floor four-point triangle inequality also passes to that limit.
Together with the endpoint distance, it forces equality on every
subinterval, proving (3).

## 4. The selected curves' actual zero-floor lengths also converge

A separate result does not require constant-speed parameterization. Suppose
\(\gamma_n\) is any selected ordinary AC positive-floor minimizer from
\(x\) to \(y\), so that

\[
L_{\epsilon_n,a}(\gamma_n)=D_{\epsilon_n,a}(x,y).
\]

`selected_minimizer_lengths_tendsto_zero_floor` proves both eventual Shore
avoidance and

\[
\boxed{L_{0,a}(\gamma_n)\longrightarrow D_{0,a}(x,y).}
\tag{11}
\]

These are the actual **zero-floor integrals of the selected curves**, not
only their regularized lengths. The proof uses the uniform bound
\(L_{\epsilon_n,a}(\gamma_n)\le D\). Once \(\epsilon_n<m\), the existing
ruler-confinement theorem prevents a Shore crossing, and the checked
comparison gives

\[
D\le L_{0,a}(\gamma_n)
\le\frac{D}{1-\epsilon_n^2/m^2}.
\tag{12}
\]

The positive denominator and the limiting upper bound are checked in the
proof. Early selected paths need not avoid Shore. The statement concerns
the eventual tail, where the zero-floor integrals have their intended
off-Shore interpretation.

## 5. Verification and remaining boundary

The targeted command
`lake build --wfail FractalSync.ZeroFloorGeodesics FractalSync.Analysis.BoundedSpeedCompactness`
passed with **2,567 Lake jobs** and no warnings or errors, using the existing
Lean 4.24.0 and pinned Mathlib environment. The new Arzelà–Ascoli import was
obtained from that pin's precompiled cache; no dependency version changed.
The complete package build and unchanged transitive axiom audit are recorded
in [verification.md](verification.md).

The previous boundary excluding convergence of selected minimizing paths is
now narrowed to the following distinctions:

- The result is subsequential uniform convergence for fixed endpoints,
  fixed \(a\), and constant-speed selected positive-floor geodesics. It does
  not prove uniqueness or convergence of every selection without taking a
  subsequence.
- Integral convergence in (11) does not give convergence of derivatives,
  smooth geodesic regularity, or a geodesic ODE at nonsmooth distance points.
- No convergence uniform over varying endpoint pairs, convergence of cut
  loci, or result for selected paths between different complement components
  is inferred.
- A separately bundled zero-floor component metric and its
  properness/completeness interfaces remain separate constructions, as do
  differential-geometric structure, global cut-locus and curved-Shore
  bifurcation proofs.

The compactness and limiting-geodesic method are standard; no independent
publication-level novelty is claimed. Runtime behavior, training, provider
contracts, canonical parameters and dependency pins are unchanged. The
research issues remain open and PR #164 remains a draft.
