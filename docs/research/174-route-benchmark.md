# Intrinsic bounds on a flat-Shore route graph

The Rust bounds reduced vertex expansions on the same graph used by Dijkstra. Bounded A* found exactly the same accepted costs under every tested budget. Search time rose on this cheap synthetic graph because evaluating a bound costs more than visiting a few neighbors. The experiment exercises [#174](https://github.com/rhythmatician/FractalSync/issues/174) with the target membership contract from [#138](https://github.com/rhythmatician/FractalSync/issues/138).

## Graph and search

The fixture uses the full line \(y=0\) as Shore, so distance is \(|y|\) and a strict sign change certifies a crossing for point targets. The graph has \(65\times33\) vertices with four cardinal neighbors at spacing \(\varepsilon/4\). It covers two floors (\(1/2,1\)) and three graph weights (\(0,1,3\)). The 21 fixed queries use same-side, crossing and Shore endpoints, point goals and ambient arrival balls. Two queries begin inside their accepted sets. A wall masks edges between columns 0 and 1 except at \(|y|\ge13\), forcing a detour beyond a local strip around the Shore. A fully masked wall disconnects one target. The missing-certificate case keeps the reference edge weights but supplies no heuristic or envelope evidence.

Each edge represents a straight segment. Horizontal edges have length \(\Delta x/\sqrt{\varepsilon^2+y^2}\). On a vertical segment the density is \(\sqrt{\varepsilon^2+(1+a^2)y^2}/(\varepsilon^2+y^2)\). Each of 32 subinterval rectangles uses the largest numerator and smallest denominator on that subinterval to enclose the segment length. The arithmetic uses outward binary64 rounding for the square root, division, multiplication and sum. Input coordinates and the intermediate squared terms are exact dyadic values for this fixture. The final weights round up to a multiple of \(2^{-24}\), making represented graph costs and their sums exactly comparable across search modes. The continuum reference is [the checked flat length formula](160-flat-intrinsic.md).

All three modes enumerate the same permitted edges. Dijkstra orders by accumulated cost. A* uses the certified Rust point or ambient-ball lower bound and allows reopens. Bounded A* also rejects a candidate whose lower bound plus spent cost exceeds the budget, or whose start displacement falls outside the certified envelope. The priority sum rounds down. In all modes, goal membership is checked directly against the point or closed ambient ball. The ball heuristic applies to every point of the ambient ball, including points beyond the sampled vertices.

Each query has an unlimited run, a zero budget and a budget of 3. Reachable queries with positive cost also run at the Dijkstra optimum and one cost unit below it. All 594 budgeted and unlimited comparisons were checked for exact graph cost agreement or a shared failure to reach an accepted vertex. The graph's restricted direction set, edge enclosures and finite window limit what a graph cost says about the continuous minimum.

## Results

| Case | Searches | Accepted | Dijkstra expansions | A* expansions | Bounded A* expansions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Same side | 120 | 52 | 82,188 | 56,812 | 35,364 |
| Cross Shore | 120 | 54 | 114,388 | 100,036 | 80,734 |
| On Shore | 138 | 72 | 73,596 | 55,808 | 38,664 |
| Arrival ball | 108 | 60 | 55,686 | 43,140 | 31,678 |
| Masked detour | 60 | 24 | 69,728 | 55,320 | 42,420 |
| Disconnected | 18 | 0 | 8,948 | 8,948 | 7,230 |
| Missing certificate | 30 | 12 | 32,724 | 32,724 | 32,724 |
| Total | 594 | 274 | 437,258 | 352,788 | 268,814 |

An expansion retrieves one vertex's adjacency list. A* reduced those retrievals by 19%, and bounded A* by 39%. A* evaluated 331,252 node bounds. Bounded A* evaluated 254,174 and rejected 15,142 candidates. One Windows release run spent 36 ms in Dijkstra, 206 ms in A* and 162 ms in bounded A*. These elapsed times exclude graph construction and the reference and control checks. These counts measure fixture operations. An adaptive Map would have a separate cost for obtaining geometry and edge certificates.

The envelope check uses known axis-aligned paths to every vertex on the clear grid. It checks that all endpoints with constructive prefix cost within budget lie inside the outer radius. Across 24 sweeps at budgets \(0, 2^{-10}, 1/2, 3\), 38,376 of 51,480 closed square cell footprints are certified outside the envelope. The square's half-width is included in the distance test. This measures potential restriction before fetching geometry. The graph search already knows each candidate's prefix cost when applying its envelope.

The separate reopening test uses an admissible heuristic that violates edge consistency and checks that the better route reopens a previously expanded vertex. A negative control adds an edge weighted below the corresponding ideal-distance lower bound. Dijkstra accepts the shortcut, and the bounded search discards it. The edge-cost premise therefore matters even when every heuristic input is valid.

The Shore queries include separation 0, \(3\varepsilon\) and \(3.25\varepsilon\), on either side of the checked \(\pi\varepsilon\) continuum shortcut threshold. The [flat minimum](160-flat-global-minimum.md) and [shortcut experiment](172-shore-shortcut-benchmark.md) cover equality and the continuum excursion lengths. A separate fixed same-component reference compares the exact vertical distances from heights 1 to 2 with \(a=0\) at four floors from 1 to \(1/8\). These lengths increase toward the zero-floor value \(\log 2\). The straight normal minimizer has clearance 1, consistent with the guaranteed zero-floor clearance \(1\exp(-\log 2)=1/2\). This verifies that particular reference and leaves general route convergence to the [zero-floor analysis](160-zero-floor-geodesics.md).

Run the [fixture](../../runtime-core/examples/intrinsic_route_benchmark.rs) with

~~~powershell
rtk proxy cargo run --release --manifest-path runtime-core/Cargo.toml --example intrinsic_route_benchmark
~~~
