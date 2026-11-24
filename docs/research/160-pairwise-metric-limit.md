# #160: the actual pairwise metric and its complete pointwise zero-floor limit

This continuation proves the different-component result proposed after the
[AC Shore-distance theorem](160-absolute-continuous-paths.md), then supplies the
metric foundations and the other branches of the fixed-pair singular limit.
All results labeled checked below refer to actual Lean definitions and proofs.
They assume no existence of minimizing curves between arbitrary endpoints.

The main setting is a finite-dimensional real normed space \(E\), a nonempty
closed set \(S\), a real parameter \(a\), and a positive floor \(\epsilon\).
The physical convention \(a\ge0\) is unnecessary for these results. Write
\[
d(x)=\operatorname{dist}(x,S),\qquad
\rho_\epsilon(x)=\sqrt{\epsilon^2+d(x)^2},\qquad b=\sqrt{1+a^2}.
\]
The positive-floor metric and properness theorems actually allow arbitrary
\(S\). Closedness and nonemptiness enter the zero-floor results through
positive distance from an off-Shore point and compact path clearance.

## 1. Definitions: actual integrals and actual path infima

For an absolutely continuous path \(\gamma:[0,1]\to E\), let \(r=d\circ\gamma\).
The checked functional is
\[
L_{\epsilon,a}(\gamma)=
\int_0^1
\sqrt{\frac{\|\gamma'(t)\|^2}{\epsilon^2+r(t)^2}
      +\frac{a^2r(t)^2r'(t)^2}{(\epsilon^2+r(t)^2)^2}}\,dt.
\tag{1}
\]
The existing AC package proves the actual distance-trace derivative bound
\(|r'|\le\|\gamma'\|\) almost everywhere and integrability of (1) from AC alone
when \(\epsilon>0\). It also identifies (1) with the original
ruler-derivative graph length. No differentiability of the ambient distance
field, or derivative at a prescribed crossing time, is required.

The new definition is
\[
D_{\epsilon,a}(x,y)=
\inf\{L_{\epsilon,a}(\gamma):
 \gamma\text{ is AC},\ \gamma(0)=x,\ \gamma(1)=y\}.
\tag{2}
\]
In Lean, `pairwisePathLengths` is the set displayed in (2), and
`intrinsicDistance` is its real `sInf`. An affine segment supplies a member,
and nonnegative graph speed supplies a lower bound. For positive floor every
admissible length is a genuine finite integral.

The zero-floor definition uses a different admissible class:
\[
D_{0,a}(x,y)=
\inf\{L_{0,a}(\gamma):
 \gamma\text{ is AC},\ \gamma(0)=x,\ \gamma(1)=y,\
 \gamma([0,1])\cap S=\varnothing\}.
\tag{3}
\]
Here the integrand simplifies almost everywhere to
\[
\frac{\sqrt{\|\gamma'\|^2+a^2r'^2}}{r}.
\]
`zeroFloorPathLengths` and `zeroFloorDistance` implement (3). A continuous
path with compact image avoiding nonempty closed \(S\) has positive clearance.
The formal proof uses that clearance to derive integrability of the actual
zero-floor density, rather than imposing integrability as an extra premise.

Equation (3) is used only when its admissible class is nonempty. Its real
`sInf` is not an extended distance with value \(+\infty\) on empty path classes.
In particular, substituting \(\epsilon=0\) into (2), where paths through Shore
are still allowed, is not the definition of (3).

Sources: [PairwiseDistance.lean](../../formal/FractalSync/PairwiseDistance.lean),
[ZeroFloorLimit.lean](../../formal/FractalSync/ZeroFloorLimit.lean), and the
[earlier AC functional](../../formal/FractalSync/IntrinsicShoreDistance.lean).

## 2. The requested different-component theorem

Let \(x\notin S\) and suppose
\[
y\notin\operatorname{connectedComponentIn}(E\setminus S,x).
\]
For the different-component case both endpoints are outside \(S\).
The topological lemma also permits \(y\in S\).

If a continuous path from \(x\) to \(y\) avoided \(S\), its image would be a
preconnected subset of \(E\setminus S\) containing \(x\). It would therefore lie
inside the connected component of \(x\), contradicting the endpoint premise.
Lean proves this argument for a continuous map from any closed real interval
into an arbitrary topological space. The AC version follows by continuity.

Apply the existing AC crossing calibration at the resulting time \(c\):
\[
F_{\epsilon,a}(d(x))+F_{\epsilon,a}(d(y))
\le L_{\epsilon,a}(\gamma).
\]
The lower bound is independent of the candidate path, so it passes directly
through the nonempty, bounded-below length set in (2):
\[
\boxed{
F_{\epsilon,a}(d(x))+F_{\epsilon,a}(d(y))
\le D_{\epsilon,a}(x,y).
}
\tag{4}
\]
This is `radialProfile_sum_le_intrinsicDistance_of_separated_components`.
It does not choose a shortest path.

For nonempty closed \(S\), \(x\notin S\) implies \(d(x)>0\). The checked radial
divergence and (4) imply
\[
\boxed{D_{\epsilon,a}(x,y)\longrightarrow+\infty
       \quad(\epsilon\downarrow0).}
\tag{5}
\]
The precise Lean limit uses `𝓝[>] 0` and `Filter.atTop`. This is
`intrinsicDistance_tendsto_zero_floor_of_separated_components`.

Sources: [ComponentCrossing.lean](../../formal/FractalSync/ComponentCrossing.lean)
and [PairwiseDistance.lean](../../formal/FractalSync/PairwiseDistance.lean).

## 3. Reversal, concatenation, and the metric axioms

The new path operations are actual functions:
\[
\operatorname{reverse}\gamma(t)=\gamma(1-t),
\]
\[
(\gamma*\eta)(t)=
\begin{cases}
\gamma(2t),&t\le\tfrac12,\\
\eta(2t-1),&t>\tfrac12.
\end{cases}
\]
For matching endpoints, Lean proves AC of the concatenation and the exact
length laws
\[
L_{\epsilon,a}(\operatorname{reverse}\gamma)=L_{\epsilon,a}(\gamma),
\qquad
L_{\epsilon,a}(\gamma*\eta)=L_{\epsilon,a}(\gamma)+L_{\epsilon,a}(\eta).
\tag{6}
\]
The concatenation theorem uses positive floor and finite-dimensional AC
integrability. Its almost-everywhere proof does not require a derivative at
the join. The reversal identity is algebraically more general.

The supporting `ACComposition` module proves AC preservation under monotone
or antitone Lipschitz precomposition and proves AC gluing. These are original
compatibility proofs for the pinned environment, separate from the attributed
upstream FTC backport. The package does not infer a general length-invariance
theorem for every nonlinear reparameterization from those AC results.

Reversal proves symmetry of (2). To prove the triangle inequality, choose
paths within \(\delta/2\) of each of two infima, concatenate them using (6),
and let \(\delta\) decrease to zero. The proof needs approximate candidates,
not attainment. The constant path proves \(D(x,x)=0\).

Positive definiteness follows from a new logarithmic displacement estimate:
\[
\boxed{
\log\!\left(1+\frac{\|x-y\|}{\rho_\epsilon(x)}\right)
\le D_{\epsilon,a}(x,y)
\le \frac{b}{\epsilon}\|x-y\|.
}
\tag{7}
\]
For the lower bound, fix the anchor \(x\) and put
\(f(t)=\rho_\epsilon(x)+\|\gamma(t)-x\|\).
The one-Lipschitz ruler gives \(\rho_\epsilon(\gamma(t))\le f(t)\).
The actual AC calculus then yields almost everywhere
\[
(\log f)' \le \frac{\|\gamma'\|}{f}
\le \frac{\|\gamma'\|}{\rho_\epsilon(\gamma)}
\le \text{the density in (1)}.
\]
Integrating and taking the infimum proves the lower inequality in (7).
The upper inequality uses the affine segment and the global density bound.

Consequently `intrinsicMetricSpace` is an actual `MetricSpace E` whose distance
is definitionally (2). It is an explicit value rather than a global instance,
so the original normed-space metric remains available.

The package also proves the useful parameter comparisons
\[
D_{\epsilon,0}\le D_{\epsilon,a}\le bD_{\epsilon,0},
\tag{8}
\]
and
\[
0<\epsilon_1\le\epsilon_2
\quad\Longrightarrow\quad
D_{\epsilon_2,a}\le D_{\epsilon_1,a}.
\tag{9}
\]
These are comparisons of the actual infima, derived from their path
functionals. Decreasing the floor increases the distance.

Sources: [PathOperations.lean](../../formal/FractalSync/PathOperations.lean),
[DisplacementBounds.lean](../../formal/FractalSync/DisplacementBounds.lean),
[IntrinsicMetric.lean](../../formal/FractalSync/IntrinsicMetric.lean), and
[MetricComparison.lean](../../formal/FractalSync/MetricComparison.lean).

## 4. The original topology, properness, and completeness

Exponentiating (7) gives the checked displacement modulus
\[
\|x-y\|\le
\rho_\epsilon(x)\bigl(e^{D_{\epsilon,a}(x,y)}-1\bigr).
\tag{10}
\]
Together with the upper bound in (7), it identifies the topology of the
intrinsic metric with the original normed-space topology.

The triangle inequality and (7) also prove
\[
|D_{\epsilon,a}(x,y)-D_{\epsilon,a}(x,z)|
\le D_{\epsilon,a}(y,z)
\le \frac b\epsilon\|y-z\|.
\tag{11}
\]
Thus intrinsic closed balls are closed in the original topology. Equation
(10) confines them to ordinary bounded balls:
\[
\{y:D_{\epsilon,a}(x,y)\le R\}
\subseteq
\overline B_{\|\cdot\|}
  \left(x,\rho_\epsilon(x)(e^R-1)\right).
\tag{12}
\]
Finite dimensionality makes the latter closed ball compact. Lean transfers
that compactness to the actual intrinsic metric and constructs both
`ProperSpace` and `CompleteSpace` proofs.

Only the topologies are identified. The intrinsic metric retains its own
uniform structure, and its completeness is proved for that uniform structure.
No equality of the two uniformities is asserted.

These results hold for arbitrary \(S\) and real \(a\), with \(\epsilon>0\).
Source: [IntrinsicTopology.lean](../../formal/FractalSync/IntrinsicTopology.lean).

## 5. The same-component limit, with a quantitative comparison

### Actual AC paths inside connected components

For any open set \(U\) in a real normed space, define the points reachable from
\(x\) by AC paths staying in \(U\). Small balls provide affine paths. The
concatenation theorem shows that the reachable class is open, and that its
relative complement in \(U\) is open. Connectedness then forces the entire
component of \(x\) into the reachable class.

The checked specialization to \(U=E\setminus S\) supplies an actual AC path
avoiding \(S\) whenever \(S\) is closed, \(x\notin S\), and \(y\) belongs to the
same component. This topological construction needs no finite-dimensional or
nonemptiness assumption on \(S\). Those hypotheses re-enter when its path is
used for zero-floor integration and convergence.

Source: [ACComponentPaths.lean](../../formal/FractalSync/ACComponentPaths.lean).

### The exact reference-path bound

Fix any off-Shore AC reference path of zero-floor length at most \(B\), and set
\[
m=d(x)e^{-B}>0.
\]
Lean proves, whenever \(0<\epsilon<m\),
\[
\boxed{
\left(1-\frac{\epsilon^2}{m^2}\right)D_{0,a}(x,y)
\le D_{\epsilon,a}(x,y)
\le D_{0,a}(x,y).
}
\tag{13}
\]
The constant uses the actual reference bound \(B\), without an added margin
for approximate minimizers. The proof examines every candidate in (2).

First, the actual log-ruler trace is AC and satisfies
\[
\left|\log\rho_\epsilon(\gamma(t))
      -\log\rho_\epsilon(\gamma(0))\right|
\le L_{\epsilon,a}(\gamma|_{[0,t]}).
\]
If \(L_{\epsilon,a}(\gamma)\le B\), this yields
\[
\rho_\epsilon(\gamma(t))\ge
\rho_\epsilon(x)e^{-B}\ge d(x)e^{-B}=m>\epsilon.
\]
Such a path cannot meet \(S\). At each point of this path the scalar density
comparison implies
\[
\left(1-\frac{\epsilon^2}{m^2}\right)\ell_{0,a}
\le\ell_{\epsilon,a}.
\]
Its integrated zero-floor length is at least \(D_{0,a}\), proving the desired
lower bound for this candidate.

If \(L_{\epsilon,a}(\gamma)>B\), the same lower bound is immediate from
\[
0\le D_{0,a}\le B,\qquad
0<1-\epsilon^2/m^2\le1.
\]
Both classes of candidate therefore satisfy the uniform lower bound in (13).
Taking their infimum proves it. The upper bound follows by regularizing each
off-Shore candidate in (3), since regularization decreases its length.

The named theorem is
`intrinsicDistance_zero_floor_sandwich_of_reference`. By rearrangement, (13)
also gives the error form
\[
0\le D_{0,a}(x,y)-D_{\epsilon,a}(x,y)
\le \frac{\epsilon^2e^{2B}}{d(x)^2}D_{0,a}(x,y).
\tag{14}
\]
Equation (14) is an immediate algebraic consequence of the checked sandwich;
it is not separately packaged as a named Lean theorem.

For a fixed pair in one component, the preceding topological theorem supplies
one finite reference path. The lower factor in (13) tends to one. The squeeze
theorem therefore proves
\[
\boxed{
D_{\epsilon,a}(x,y)\longrightarrow D_{0,a}(x,y)
\quad(\epsilon\downarrow0).
}
\tag{15}
\]
`intrinsicDistance_tendsto_zero_floor_of_mem_connectedComponentIn` has actual
component membership as its hypothesis. It requires no reference-path,
clearance, integrability, or minimizing-curve premise.

Source: [ZeroFloorLimit.lean](../../formal/FractalSync/ZeroFloorLimit.lean).

## 6. Every fixed endpoint pair is covered

If \(x\in S\), then \(\rho_\epsilon(x)=\epsilon\). The lower bound in (7) becomes
\[
D_{\epsilon,a}(x,y)\ge
\log\!\left(1+\frac{\|x-y\|}{\epsilon}\right).
\]
It diverges for every \(y\ne x\), including another point of \(S\). Symmetry
handles a Shore endpoint in the other orientation. This resolves the branch
that the radial endpoint calibration alone cannot detect when both endpoint
distances to Shore vanish.

For nonempty closed \(S\), fixed real \(a\), and a finite-dimensional real
normed space, the checked statements give the complete pointwise
classification:

| Endpoint configuration | Limit as \(\epsilon\downarrow0\) | Principal checked statement |
| --- | --- | --- |
| \(x=y\), including on Shore | \(0\), identically for every positive floor | `intrinsicDistance_self` |
| \(x,y\notin S\), in the same connected component of \(E\setminus S\) | The finite actual zero-floor infimum \(D_{0,a}(x,y)\) | `intrinsicDistance_tendsto_zero_floor_of_mem_connectedComponentIn` |
| \(x,y\notin S\), in different connected components | \(+\infty\) | `intrinsicDistance_tendsto_zero_floor_of_separated_components` |
| \(x\ne y\), with at least one endpoint in \(S\) | \(+\infty\) | `intrinsicDistance_tendsto_zero_floor_of_mem_or_mem` |

This is a family of real-valued convergence or divergence theorems, not a
single bundled extended metric. No uniform convergence over varying endpoint
pairs is claimed. The estimate (14) depends on the chosen reference bound
and the initial endpoint's positive clearance.

## 7. Remaining boundary and the next formal step

The new results prove a proper, complete positive-floor metric and all
fixed-pair branches of its zero-floor limit. They do not yet identify the
integral (1) with the metric length of every admissible curve. Consequently,
the length-space construction and existence of minimizing curves between
arbitrary endpoints remain separate formal work. Properness alone is not
being used as a substitute for that missing bridge.

The next focused geometry step is to prove the needed metric-length
comparison or length-space property from the actual AC functional, then use
properness to obtain minimizing curves. The proof must retain the
almost-everywhere treatment of distance traces at nonsmooth points.
The earlier nearest-Shore minimum is already attained by the checked
nearest segment, independently of this general pairwise obligation.

Riemannian curvature identification, the flat cut locus, the curved-Shore
bifurcation, and the classification of all nearest-Shore minimizers remain
conventional results in the other notes. These formal additions make no
publication-level novelty claim: the topology and metric arguments use
standard methods, while the exact regularized family and its quantitative
comparison are recorded with their actual definitions and hypotheses.

PR #164 remains a draft, and the research issues remain open. Runtime,
training, canonical parameters, and numerical provider contracts are unchanged.
