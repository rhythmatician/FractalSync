# A path-length bound from both endpoint rulers

The two-ended displacement bound strengthens the earlier logarithmic bound from either endpoint. On a finite flat-Shore graph it reduced A* expansions from 54,651 to 49,593 across 36 searches. All searches returned the same graph cost as Dijkstra. The result supports further testing with certified provider inputs. [Target bounds](174-intrinsic-target-bounds.md), [navigation issue](https://github.com/rhythmatician/FractalSync/issues/174).

## Proof

Let \(r=\rho(x)\), \(s=\rho(y)\), \(d=\|x-y\|\), and write \(D(x,y)\) for the ideal intrinsic path infimum. Then
\[
H_2(x,y)=2\log\frac{d+r+s}{2}-\log r-\log s\le D(x,y).
\]
The ruler is 1-Lipschitz, so \(|r-s|\le d\). For a path from \(x\) to \(y\), consider the continuous functions \(r+\|\gamma(t)-x\|\) and \(s+\|\gamma(t)-y\|\). Their ordering reverses weakly at the endpoints, so they agree somewhere on the path. At that point their common value \(m\) is at least \((d+r+s)/2\) by the ambient triangle inequality. The two subpaths pay at least \(\log(m/r)\) and \(\log(m/s)\) by the existing one-ended calibration. The bound survives the infimum over all absolutely continuous paths. [Two-ended proof](../../formal/FractalSync/TwoEndedDisplacement.lean), [one-ended calibration](../../formal/FractalSync/DisplacementBounds.lean).

The arithmetic–geometric mean inequality for \(d+r\) and \(s\) shows that \(H_2\ge\log(1+d/r)\). Swapping endpoints gives the other one-ended comparison. The bound is strict on both sides when \(|r-s|<d\). Across three points with endpoint rulers \(r,s,u\) and ambient lengths \(d=\|x-y\|\), \(e=\|y-z\|\), \(f=\|x-z\|\), the scalar inequality
\[
(d+r+s)(e+s+u)-2s(f+r+u)
=(d+r-s)(e+u-s)+2s(d+e-f)\ge0
\]
proves the triangle inequality for \(H_2\). Therefore for a fixed exact goal the heuristic is consistent when each represented edge cost upper-bounds the corresponding ideal pairwise distance. A numerical graph must still justify that edge premise.

## What to do with uncertain rulers

For actual endpoint rulers bounded above by \(R,S\) and actual displacement bounded below by \(b\ge0\), Lean checks the same formula using \(R,S,b\) when \(|R-S|\le b\). This condition gives a balancing point for the supplied upper rulers. With \(R\le S\), a convenient evaluation is
\[
\log1p\!\left(\frac{b+S-R}{2R}\right)
+\log1p\!\left(\frac{b-(S-R)}{2S}\right).
\]
Without the compatibility condition use the one-ended lower bound from the smaller supplied upper ruler. For example, at coincident points the true displacement is zero. Substituting unequal ruler upper bounds into the unguarded two-ended expression would return a positive number. The exact-data consistency proof does not transfer to arbitrarily varying intervals. Provider correspondence, outward rounding and planner integration remain open.

## Analytic and finite-graph checks

For two points on a flat Shore, \(\rho=\varepsilon\) at both ends and \(H_2=2\log(1+d/(2\varepsilon))\). At separation \(6\varepsilon\), the old bound is 1.946 and the two-ended bound is 2.773, while the independently integrated zero-anisotropy excursion has length 4.815. The numerical probe checks nine spans against the flat threshold and the excursion integrals at three graph weights. On 192 independently drawn endpoint pairs, the stronger term improved 257 of 576 comparisons when combined with the radial difference or the known crossing sum. Seven incompatible input triples took the one-ended fallback. [Flat-Shore reference](172-shore-shortcut-benchmark.md).

The search experiment uses a \(49\times97\) grid at spacing \(\varepsilon/4\) in both directions. Tangential edges use the analytic constant-ruler length. Normal edges use differences of the signed radial primitive, including edges incident to the Shore. Both A* variants use the same radial bound and tie rule. Only the displacement term changes. The check compares twelve fixed start–goal pairs at each of \(a=0,1,3\), including Shore, same-side and crossing pairs. It checks the consistency inequality on every directed edge for each goal, within a numerical tolerance.

| Search | Total expansions |
| --- | ---: |
| Dijkstra | 94,104 |
| A*, one-ended displacement and radial bound | 54,651 |
| A*, two-ended displacement and radial bound | 49,593 |

The two-ended bound reduced expansions in 27 queries and tied in nine. All returned costs agree within \(2\times10^{-12}\). These are expansions on a fixed finite graph with binary64 edge evaluations. They give no query savings for an adaptive provider and no claim that the graph-optimal route minimizes continuous path length. [Reproduction script](../../formal/checks/check_two_ended_displacement.py).

~~~powershell
rtk proxy .\.venv\Scripts\python.exe formal/checks/check_two_ended_displacement.py
~~~
