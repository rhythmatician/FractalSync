# Intrinsic target and search-budget bounds

This proof slice for [#174](https://github.com/rhythmatician/FractalSync/issues/174) applies the existing positive-floor path metric to a set of acceptable endpoints. It gives lower bounds for ambient and intrinsic tolerances, an edge consistency theorem for the fixed-target radial bound, and an outer search ball from a certified budget. [Lean module](../../formal/FractalSync/IntrinsicTargetBounds.lean).

## Accepted endpoint sets

Write \(D(x,y)\) for the existing unrestricted intrinsic infimum of absolutely continuous graph-path lengths, with floor \(\epsilon>0\) and graph parameter \(a\). For a nonempty target \(T\) define

\[
D_T(x)=\inf_{y\in T}D(x,y).
\]

The file checks \(D_T(x)\ge0\), \(D_T(x)=0\) for \(x\in T\), and the equivalence between a lower bound on \(D_T(x)\) and a lower bound that holds for every \(y\in T\). A nonempty sampled subset has an infimum greater than or equal to that of the full target. A closest sampled endpoint therefore can overestimate the required distance. The lower-bound interface assumes a nonempty acceptable set. Zero remaining distance alone does not certify target membership, since a source can lie in the closure of a nonclosed set. [Existing pairwise path metric](160-pairwise-metric-limit.md).

## Lower bounds from proved enclosures

Let \(d(x)=\operatorname{dist}(x,S)\), \(\rho(x)=\sqrt{\epsilon^2+d(x)^2}\), and \(f(x)=F_{\epsilon,a}(d(x))\). Suppose the true \(f(x)\) lies in \([\ell_x,u_x]\), every accepted endpoint has \(f(y)\) in \([\ell_T,u_T]\), \(b\le\|x-y\|\) holds for every accepted \(y\), and \(R_x\ge\rho(x)\). The checked bound is

\[
\max\left\{
\log\!\left(1+\frac{\max(0,b)}{R_x}\right),
\max(0,\ell_x-u_T,\ell_T-u_x)
\right\}\le D_T(x).
\]

The radial interval term is zero when the intervals overlap. The code applies the displacement and radial calibrations pointwise and then passes the uniform bound through the infimum. The denominator uses a certified upper ruler and the numerator uses a certified lower displacement. The formula takes real-valued enclosures as inputs. A provider that returns a smooth approximate field still needs an independent correspondence or a conservative comparison before those enclosures can be used for pruning. [Radial profile](160-radial-profile.md), [logarithmic displacement proof](../../formal/FractalSync/DisplacementBounds.lean).

The target tolerance can be measured in either metric. If every target lies in the ambient closed ball of radius \(\tau\) about \(p\), use \(b=\|x-p\|-\tau\) in the logarithmic bound. If every target instead satisfies \(D(y,p)\le\tau\), any certified lower bound \(h_p\le D(x,p)\) gives \(\max(0,h_p-\tau)\le D_T(x)\). These are two separate checked statements with different units.

## Edges and budgets

The [two-ended displacement bound](174-two-ended-displacement.md) strengthens both one-ended bounds for a fixed goal. It proves consistency with exact rulers and upper-bounding graph edges. It also gives a lower bound under compatible ruler enclosures, and the flat-Shore graph experiment compares the stronger heuristic against Dijkstra.

For an edge \(x\to y\) with cost \(c\ge D(x,y)\), Lean checks \(D_T(x)\le c+D_T(y)\). It also checks the same inequality for the lower bound \(h(x)=\max(0,f(x)-u_T,\ell_T-f(x))\) formed with an exact source profile and one fixed target interval covering every accepted endpoint. The scalar interval-gap change is bounded by \(|f(x)-f(y)|\), which the existing radial calibration bounds by \(D(x,y)\). Using independently varying source intervals or combining this term with the logarithmic bound needs a separate consistency argument. A sufficient edge-cost premise is an upper bound on the intrinsic length of an admissible matching path. [Intrinsic triangle inequality](../../formal/FractalSync/IntrinsicMetric.lean).

If a route can reach a point \(y\) and then an accepted endpoint with certified prefix and suffix costs summing to at most \(B\), that point satisfies \(D(x,y)+D_T(y)\le B\). Each point of this combined sublevel lies in the ambient closed ball

\[
\overline B\!\left(x,R_x(\exp B-1)\right).
\]

The radius follows from the checked logarithmic displacement estimate, nonnegativity of remaining cost, and the upper ruler certificate. This is an outer envelope. A point inside it still needs a valid route and valid geometry. Restricted path domains can impose additional costs or remove routes. [Intrinsic sublevel geometry](../../formal/FractalSync/IntrinsicTopology.lean).

## Rust implementation and benchmark

The [Rust bounds](../../runtime-core/src/intrinsic_bounds.rs) accept certified
endpoint radius intervals, upper rulers, and a lower displacement. They
combine the compatible two-ended expression with a conservative radial
contribution. The radial lower bound integrates a piecewise lower bound on the
density. The canonical radial formula is also returned separately as a
diagnostic estimate. A Euclidean target ball widens the ruler and radius
enclosures to cover the whole set, and drops the center's crossing assumption.

The numerical bounds use outward-rounded binary64 arithmetic with bounded
series for logarithms and the search envelope. They require the caller to
establish the geometric premises. The [flat-Shore benchmark](174-route-benchmark.md)
tests cost agreement, target balls, masked edges and reopening against
Dijkstra with upper-bounded path costs. It disables heuristic pruning when
metric evidence is missing. Provider correspondence, actual Map query savings
and integration into a navigation consumer remain with #174 and #138.

`lake build +FractalSync.IntrinsicTargetBounds --wfail` passed with the pinned Lean environment. The module contains 18 proved theorems, including two private lemmas. The transitive axiom check for all 16 public theorems reported only `propext`, `Classical.choice`, and `Quot.sound`.
