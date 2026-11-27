# Route length enclosures on the retained Map field

The retained fine asset can bound costs for a finite graph near the cusp.
The bounds use the exact real spline determined by the stored coefficients
and chart, with the default manifold profile. This gives #174 a second
search fixture, alongside the flat Shore graph and the Controls replay menu.
The preparation recipe and asset SHA-256 are pinned in the replay evidence.
[Flat and replay comparisons](174-route-benchmark.md) ·
[Retained asset and preparation](../validation/bounded-operational-engine.md)

## Segment certificates

For a straight segment with displacement \(v\), write \(F\) for the
prepared scalar field, \(\rho=\sqrt{F^2+\epsilon^2}\), and
\(a^2=\lambda^2/(\ln 2)^2\). Its speed over a unit parameter interval is

\[
\frac{\sqrt{\|v\|^2+a^2 (\rho')^2}}{\rho},\qquad
\rho'=\frac{F}{\rho}\nabla F\cdot v.
\]

The Rust check encloses the entire coordinate rectangle of the segment in
floor-selected chart cells, rounding the normalization outwards. Missing
stencil data makes the segment unavailable. In each cell, a separable
change of basis turns the six-by-six spline into a tensor Bernstein
polynomial. The scalar lies between the smallest and largest control values.
Each physical derivative is bounded by the corresponding adjacent control
differences multiplied by five over the lattice spacing. Nonnegative
Bernstein weights sum to one in both applications. The table is checked
against the literal evaluator on an exact dyadic grid.

The enclosing cells give a positive ruler interval and a bound on
\(|\rho'|\) for the entire segment. Dropping the scale term gives a lower
length. Using the largest speed numerator and the smallest ruler gives an
upper length. Lean checks the straight-segment integral inequality and the
ambient target-ball bound. The coefficient conversion and binary64
operations have their own executable tests.
[Checked integral bounds](../../formal/FractalSync/OperationalRouteBounds.lean) ·
[Rust cell computation](../../runtime-core/src/operational_field/bernstein.rs)

## Retained cusp window

The fixture uses a 101 by 81 dyadic lattice at spacing \(2^{-11}\) in
\([450,550]/2048\times[-41,39]/2048\). It encloses 32,180 undirected
segments, including diagonals. For 332 deterministic samples, midpoint
quadrature of the compiled field jet lay within the intervals. Refining from
32 to 128 panels changed lengths by at most 0.000249. That comparison is a
numerical cross-check. The enclosure comes from the coefficient arithmetic.

On these same edges, a first pass using coefficient extrema and the
centered spline gradient bound had a largest interval width of 112.757.
The Bernstein controls reduced it to 11.345. Edges near the smallest ruler
still have wide intervals. For the horizontal cusp case, the eight-direction
graph puts the minimum over its exact spline edge lengths in
\([7.18535,38.10322]\). The same calculation for the second point case
gives \([4.16024,6.82602]\).

The search rounds edge weights outwards to multiples of \(2^{-24}\) and
checks that sums stay exact in binary64. Dijkstra on the lower weights bounds
the minimum over the admitted graph. Dijkstra and bounded A* on the upper
weights agree on all 32 comparisons across two neighbor sets, point and ball
targets, and four budgets per query. The heuristic divides remaining ambient
distance by a regional upper bound on the ruler. The outer budget radius uses
the same ruler bound. Neither calculation requires an ideal Shore distance.

| Neighbors | Dijkstra expansions | Bounded A* expansions |
| --- | ---: | ---: |
| Four | 89,752 | 88,956 |
| Eight | 89,812 | 88,950 |

Both searches see the same graph and precomputed edges. The expansion savings
are under one percent. The interval gives graph costs within a declared
window. Directions, spacing, and available cells constrain the possible
paths. The metric describes the stored spline in real arithmetic. Compiled
Physics error, Controls reachability, and comparison to the mathematical
Shore require separate evidence.

The same regional ruler bound gives a nominal length lower bound of 1.08825
for the first point target. With a kinetic-action budget of 0.05, the
one-second check rules it out at a lower bound of 0.59214. The action lower
bound over 100 seconds is 0.00592, leaving the task eligible. The deadline
assessment has insufficient evidence in both cases because this example has
no applicable speed cap. This check assumes the entire path stays in the
window and uses the exact spline metric. [Curriculum checks](169-navigation-feasibility.md)

```powershell
rtk proxy cargo run --release -p runtime_core --features experimental-operational-field --example check_operational_navigation -- tmp/geometry145-retained-fine.bin
```
