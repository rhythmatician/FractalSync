# Arrival bounds for the default raster bridge

A full-support window in the loaded raster gives a displacement lower bound
for the pointwise metric used by the default Physics profile. In the built-in
Map trial below, an ambient target ball needs at least 0.06077 units of
kinetic action to reach within one second. For a path that leaves the window
first, the length to the boundary supplies the bound.
[Rust certificate](../../runtime-core/src/distance_field/raster_navigation.rs)
and [action assessment](../../runtime-core/src/crossing_feasibility.rs).

## Evidence for the ruler

`RasterNavigationRegion::for_default_profile` reads the currently loaded
raster and its process-local generation under the same lock. It covers the
center sample and every possible point of the adaptive nine-point stencil.
The center rectangle must lie strictly inside the hard disk. Each coordinate
uses the sampler's rounded, monotone chart. The selected cells and their
full four-by-four coefficient supports must lie within the chart for the
center rectangle and the padded stencil rectangle. The scan rejects
nonfinite support, bilinear sampling and edge clamping or extrapolation.
The step padding comes from the center ruler upper bound and the provider's
production step clamp.
[Raster sampler](../../runtime-core/src/distance_field.rs) and
[provider query](../../runtime-core/src/geometry_provider.rs).

For a fractional coordinate \(t\in[0,1]\), the Catmull–Rom weights sum
to one. The two outer weights are
\(-t(1-t)^2/2\) and \(-t^2(1-t)/2\), while the two inner weights are
nonnegative. Their absolute sum is \(1+t(1-t)\le 5/4\), so the tensor
absolute sum is at most \(25/16\). With \(M\) the maximum absolute
coefficient, \((25/16)M\) bounds the exact interpolated magnitude.

The implementation uses \(101M/64\) before the monotone binary32 cast.
Here is the binary64 error allowance under round to nearest. The floor
selected by the chart is exact, and subtraction to form the fractional
coordinate is exact by Sterbenz's lemma. Rounding the subsequent kernel
argument moves it by at most \(2^{-52}\). The continuous piecewise
kernel has slope magnitude below 20 across its range. Each polynomial
operation has magnitude below 32 and absolute rounding error at most
\(2^{-47}\). Propagating at most 16 times that error through the degree-three
expression, then allowing for the rounded argument, gives less than
\(2^{-40}\) error per kernel weight. An individual tensor weight then has
error below \(2^{-37}\). Products of coefficients and rounded weights and
their 16-term partial sums have magnitude at most \(32M\). Their arithmetic
rounds by at most \(32M\cdot2^{-52}+2^{-1074}\) per operation.
Thus with \(\delta=2^{-30}\), the rounded numerator has magnitude at most
\((25/16+\delta)M\), and the rounded normalizer differs from one by at most
\(\delta\). The resulting quotient, including its final binary64 rounding,
has magnitude at most \(101M/64\). The absolute underflow allowance fits
when \(M>0\) because \(M\ge2^{-149}\), and a zero window gives a zero
sample. An upward binary64 product followed by the same binary32 cast as
the sampler encloses the returned value, including subnormals. An infinite
cast invalidates the bound. Inside the raster the extra outside distance
is zero. Taking the square root of the outward sum of squares bounds
the compiled ruler \(\widehat\rho\).

Finite stencil values and the provider's positive minimum step keep its
gradient and Hessian arithmetic finite. The bridge may still classify a jet
as unresolved or singular. The default metric assembly accepts those finite
raster jets.

## From the ruler to the rounded metric

Write \(R\) for the ruler upper bound. In the default profile
\(\lambda^2=1\), and the code checks \(10^{-8}\le\varepsilon\le1\).
The finite binary32 magnitudes and this floor imply
\(\varepsilon/2\le\widehat\rho<2^{130}\).
The positive base term \(b=\operatorname{fl}(1/\operatorname{fl}(\widehat\rho^2))\)
is normal and \(b\ge(1-2^{-50})/\widehat\rho^2\).
The outward ratio of the stencil sample bound to the minimum allowed
adaptive step bounds each compiled centered gradient component. The
certificate requires that ratio to be at most \(2^{18}\).

The square of each widened binary32 center sample is exact in binary64,
which gives \(|D|\le\widehat\rho\). The denominator used for the scale
gradient is at least \(\widehat\rho/2\), and the rounded scale gradient
components therefore have magnitudes at most \(2^{20}/\widehat\rho\).
For these components \(g_i\), \(|g_i g_j|\le2^{41}b\).
Relative to \(bI+gg^{\mathsf T}\), each assembled matrix entry has absolute
error at most \(2^{-48}(b+|g_i g_j|)+4\cdot 2^{-1074}<b/64\).
This allowance covers the rank-one multiplication, the diagonal addition
and gradual underflow. The argument bounds the error in the entries,
including cancellation in the quadratic form. In two dimensions their
quadratic error is at most \(b\|v\|^2/32\), while \(gg^{\mathsf T}\) is
positive semidefinite. Consequently the compiled matrix, interpreted over
the reals, satisfies

```math
v^{\mathsf T}\widehat G(c)v\ge
\frac{\|v\|^2}{2\widehat\rho(c)^2}\ge\frac{\|v\|^2}{2R^2}.
```

These floating-point estimates are conventional arithmetic arguments
against the cited implementation. Lean checks the ambient target triangle
inequality and the speed-to-action implication used afterwards.
[Metric assembly](../../runtime-core/src/manifold.rs),
[target inequality](../../formal/FractalSync/OperationalRouteBounds.lean),
[kinetic-action implication](../../formal/FractalSync/CrossingAction.lean).

## Arrival before or after leaving the window

For an absolutely continuous path and a closed ambient ball centered at
\(z\) with radius \(r\), arrival within the window requires accumulated
length at least \(\max(0,\|s-z\|-r)/(\sqrt 2R)\).
The metric at a real-valued position is the compiled matrix at its rounded
position. Rounding preserves membership in the closed binary64 rectangle.
If the path leaves before arrival, continuity gives a first contact with a
window face, and length up to contact is at least the Euclidean distance to
the nearest face divided by \(\sqrt 2R\). The lower of these two bounds
applies to either case, including a target wholly beyond the window. All
arithmetic used to return the bound rounds outward. For a positive
deadline \(T\), a length lower bound \(B\) requires at least \(B^2/(2T)\)
kinetic action on the corresponding prefix. Applying this bound to a
whole-rollout budget requires nonnegative kinetic action on the remainder.

The evidence requires the same raster generation and default profile
throughout the prefix. A live check refuses an unloaded or replaced raster
or an active prepared-field override. Continuous provider identity during
an external rollout remains a caller premise. The result describes ambient
arrival under the default runtime metric. True Shore correspondence,
intrinsic target tolerance, and compiled kinetic evaluation or RK4
trajectory error require separate evidence. Applying #169 to a rollout
still requires evidence for its accepted target,
provider and profile identity, and an actual action budget or speed cap.
The curriculum does not consume this certificate.

## Straight-segment route witnesses

For two points in the certified window, the whole straight segment stays
in the window by convexity. Let \(G_*\) be the outward bound on the
magnitude of each compiled centered gradient component. The rounded
scale gradient satisfies \(|g_i|\le 4G_*/\widehat\rho\). This uses
\(|D|\le\widehat\rho\) and the same scale-gradient denominator and
rounding allowance as the lower bound. The base term satisfies
\(b\le 2/\widehat\rho^2\). With the entry errors bounded by \(b/64\),
the matrix quadratic form is at most
\((4+32G_*^2)\|v\|^2/\widehat\rho^2\). Taking the square root and
using \(\widehat\rho\ge\varepsilon/2\) bounds length by
\((4+12G_*)/\varepsilon\) times ambient displacement.

`segment_length_upper` also encloses center and stencil samples using
the raster coefficients that support the segment's bounding rectangle.
For coefficient extrema `lo` and `hi`, the ideal interpolant ranges
from at least `lo - (9/32)(hi - lo)` to at most
`hi + (9/32)(hi - lo)`. The tensor weights sum to one and have
negative mass at most `(25/16 - 1)/2 = 9/32`. The binary64 error
allowance of `2^-26` times the maximum coefficient magnitude covers
the same kernel, sum and division errors bounded above. In particular,
with the numerator and normalizer error `delta = 2^-30`, normalization
adds at most `(1 + 25/16) delta / (1 - delta)` times that magnitude,
before the final binary64 round. Outward binary64 endpoints followed by
the sampler's monotone binary32 cast enclose the actual returned samples.

The center interval supplies a lower bound on `abs(D)` when it stays
on one side of zero. The rounded ruler dominates that bound, so it also
raises the lower bound on the adaptive step. The stencil interval's
width divided by twice this minimum step bounds the centered gradient.
Substituting these bounds into `(2 + 6 G*) / rho` gives a local metric
factor. The implementation takes the smaller of the local factor and
the original regional factor. If a local scan cannot certify its
arithmetic, the regional factor still applies.

`RasterNavigationRegion::search_route` takes a caller-supplied
directed waypoint graph in the certified window. It uses these edge
bounds, rounds their running sum upward and uses the remaining
target length lower bound to order and prune the search. The returned
path carries its length upper bound and raster generation. The method
holds that generation under a single read lock while pricing edges.

Arrival requires a waypoint whose outward ambient displacement from the
target center is within the radius. Equality of coordinates suffices for
a point target. For a positive radius, rounding can leave some points
near the boundary unaccepted. A zero remaining lower bound can occur
at the edge of the window and says nothing about waypoint membership.
An empty result means the graph yielded no certified witness within
budget. A continuous trajectory could use another route or fit the
budget with tighter segment bounds. The crossing action and deadline
assessment uses the target length lower bound separately.

The built-in raster test compares the two-route graph with Dijkstra,
checks the budget at the outward aggregate and one representable value
below it, and samples the compiled metric along each segment.

The Debug Cockpit's **Map target comparison** also searches a graph built
from the successful prefixes of matching replays. The shared launch
branches into each replay, and each branch follows only consecutive
frames. The bounding rectangle encloses the waypoints and uses the
currently loaded default-profile raster metric. The reported result is
a bound on the straight segments joining the waypoints. The observed
Controls arrival and its sampled frame are reported separately.

In the [common-start replay fixture](174-route-benchmark.md#controls-replays-on-the-map),
the raster search reaches frame 698, zero-based, of the commit 1.0
replay. Its length upper bound is about 7.47 billion because the
certificate uses one worst-case metric factor for the whole rectangle.
This demonstrates a certified path and shows how loose the window-wide
upper bound can be near the default floor.

## Reproduce the built-in raster trial

```powershell
rtk proxy cargo test --offline --release -p runtime_core --lib raster_navigation -- --nocapture
rtk proxy cargo test --offline --release -p runtime_core --features experimental-operational-field --lib raster_navigation -- --nocapture
```

The trial explicitly loads `mandelbrot_default` and reports the generation.
The SHA-256 of `mandelbrot_distance_1024.bin` is
`3105450d55222fbd5cde90a5e7757930a8536f451e5f56deeeefa1950f8de4345`.
Its `mandelbrot_distance_1024.json` chart hashes to
`2eac192de829e1d70e8c73a2061bf1d59cc747bd49466927a0e874a30532ab15`.
The window runs from \((-1.5,-0.45)\) to \((-0.4,0.45)\).
The start is \((-1,0)\) and the target center is \((-0.7,0.1)\).

| Quantity | Value shown by the test |
| --- | ---: |
| Ruler upper bound | 0.590664395214 |
| Point target length lower bound | 0.378568269159 |
| Ambient target with radius 0.025 | 0.348639819649 |
| First-exit length lower bound | 0.538712091184 |
| Kinetic-action lower bound, 48,000 samples | 0.060774861922 |
| Kinetic-action lower bound, 4,800,000 samples | 0.000607748619 |

An action budget of 0.02 rules out the shorter task and leaves the
longer task open on action. Deadline status is insufficient evidence
because the trial supplies no speed cap. A chart-edge window yields
insufficient evidence. The synthetic overshoot control returns a center
sample of 1.25 from coefficients bounded in magnitude by one.
The tests also cover missing stencil support, a nonfinite coefficient, a
bilinear tile, subnormal coefficients, an overflowing cast, raster replacement
and a prepared-field override.

Lean checks the continuous first-radius and target inequalities given a
local displacement envelope. The raster coefficient and rounded-matrix
estimates supply that premise under the numerical conditions above.
[First-exit proof](../../formal/FractalSync/OperationalWindowExit.lean)

## Timed curriculum diagnostic

The training CLI accepts an optional `defaultRasterWindow` on `runtimeMap`
manifest cases. Give the minimum and maximum corners of the window, for
example `[[-1.5, -0.45], [-0.4, 0.45]]`. This opts into loading the built-in
raster and reporting a conditional continuous-path assessment for the
default Physics profile. Both the task start and the target center must lie
in the certified window. The point length lower bound includes paths that
exit the window. Subtracting the intrinsic arrival tolerance with downward
rounding gives the lower bound for the task's intrinsic arrival ball.

The report includes `defaultRasterDiagnostic.continuousAssessment` only when
the certificate applies. Its kinetic-action result uses the task's sample
window and declared action budget. The CLI still uses the main runtime
assessment for eligibility. Turning the continuous result into a hard
filter requires a link to committed runtime action and provider identity
over the actual rollout.
