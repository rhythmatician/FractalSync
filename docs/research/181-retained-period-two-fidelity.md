# Retained prepared field versus the period-two Shore

The compiled prepared-field evaluator returned a finite jet on five of eight sampled directions at each of seven depths. The closest accepted samples showed large signed-distance and derivative errors against the exact period-two disk. This comparison addresses the independent reference-fidelity requirement in [#181](https://github.com/rhythmatician/FractalSync/issues/181). The existing arithmetic certificate instead compares compiled output to its own stored-coefficient polynomial. Both results can hold at once. [Operational polynomial correspondence](181-operational-patch.md).

## Native assets and exact reference

The earlier replay retained two prepared assets. Both have origin `(-1.002, -0.702)`, spacing `0.0005`, dimensions `3425 × 2825`, and no serialized availability mask. The fine asset's archived policy has one exact reference-cut rejection at `(-1, 0)` and leaves true-Shore fidelity unknown elsewhere. The probe decodes and re-encodes each exact asset using the compiled Rust path. It queries `PreparedField::query` on both and `FrozenRollout::probe` with the archived policy on the fine asset. The coarse asset needs its own policy for rollout claims. [Retained asset preparation](../validation/bounded-operational-engine.md), [archived reference policy](../validation/bounded-reference-policy.json).

| Retained asset | Native field ID | SHA-256 |
| --- | --- | --- |
| Fine | `9721bb7647a0b8e6` | `bc2604771c2cc48fb71cda6cb22c6e4338a776b6591fa923dd065977817ad9fb` |
| Coarse | `71eaa19d09e17bac` | `3bb6bbaca940bc98bbd4d735702d79209cda7bc862559a321491dbb990a62a63` |

The open period-two component is the disk $|c+1|<1/4$, and its circle lies on the Shore. For any interior point, every Shore point lies outside the open disk and the nearest point on the circle attains the distance to the Shore. Thus the signed interior distance is exactly $D(c)=|c+1|-1/4$. At noncentral points with $r=|c+1|$ and $n=(c+1)/r$, its gradient is $n$ and its Hessian is $(I-nn^{\mathsf T})/r$. At the center only the scalar distance has a reference value. The comparison evaluates these formulas in 70-digit Decimal arithmetic from the exact binary64 coordinates passed to Rust. [Exact period-two component and distance](184-period-two-attachment.md).

## Observations

Each depth uses four axis points and four directions from $(\pm3/5,\pm4/5)$, rounded to the nearest dyadic grid coordinate of spacing $2^{-30}$. All 56 points remain strictly inside the reference disk. Both assets return identical jets or unavailability at all 59 queried positions, including the three controls. The three westward directions in each depth band lie beyond the left side of the retained lattice and are unavailable. All 35 accepted depth-band samples are also accepted by the fine frozen probe with `reference_distance_accuracy = unknown`. All returned scalar signs in these interior samples agree with the reference.

| Nominal depth | Accepted / 8 | Maximum $|F-D|$ | Maximum $\|\nabla F-\nabla D\|_2$ | Maximum Hessian Frobenius error |
| ---: | ---: | ---: | ---: | ---: |
| $2^{-4}$ | 5 | $1.086\times10^{-5}$ | 0.0116 | 13.0 |
| $2^{-6}$ | 5 | $1.996\times10^{-5}$ | 0.0099 | 18.8 |
| $2^{-9}$ | 5 | $2.912\times10^{-5}$ | 0.0184 | 173.9 |
| $2^{-11}$ | 5 | $1.388\times10^{-4}$ | 0.2958 | 1109.9 |
| $2^{-13}$ | 5 | $3.230\times10^{-4}$ | 0.7832 | 2198.2 |
| $2^{-15}$ | 5 | $4.015\times10^{-4}$ | 0.9486 | 2318.6 |
| $2^{-17}$ | 5 | $4.235\times10^{-4}$ | 0.9909 | 2326.9 |

At the closest lower axis sample $(-1,-0.24999237060546875)$, the exact depth is $7.6294\times10^{-6}$. The prepared field reports $-4.3118\times10^{-4}$, about 56.5 times the actual magnitude, and gives gradient $(-0.11683,-0.01600)$ where the oracle gives $(0,-1)$. The largest Hessian error occurs on the right axis approaching the attachment. There the reported Hessian is approximately $\operatorname{diag}(-2065.26,1075.96)$ and the exact interior Hessian is approximately $\operatorname{diag}(0,4.00012)$. Errors in value and gradient are possible much farther inside, and the near-Shore results are enough to rule out treating this prepared asset as a true-distance oracle even where it returns a coherent jet.

For the left-halo controls, $x=-1-2^{-10}$ returns a jet and $x=-1-2^{-9}$ returns `GeometryUnavailable`. Both are inside the exact disk. At the exact center, the raw field returns a finite jet and scalar $-0.2495943$, but the archived fine policy returns `ReferenceSingularity: canonical cut or cusp`. No derivative error is reported at the center. Over these positions the raw evaluator accepts 37 of 59 queries per asset and the fine frozen policy accepts 36.

## Reproduce and remaining scope

From the repository root with the retained assets present and cached Cargo dependencies:

```powershell
rtk proxy .venv/Scripts/python.exe formal/fixtures/check_181_retained_period_two.py tmp/geometry145-retained-fine.bin tmp/geometry145-retained-coarse.bin > tmp/geometry145-retained-period-two.json
```

The [checker](../../formal/fixtures/check_181_retained_period_two.py) builds a temporary release-mode Rust consumer with `experimental-operational-field`, without loading the installed Python wheel. This run used Rust `1.98.0` on `x86_64-pc-windows-msvc`. The report retains binary query coordinates, asset and policy hashes, current source commit, both native dispositions, raw jet values and exact-reference estimates. No new asset or field policy is introduced.

These maxima describe sampled errors over two retained assets and one analytic Shore region. The following bounds cover two closed patches on the fine asset only. [Prior installed-provider sampling](120-sampled-metric-validation.md) is a separate experiment on a different field implementation.

## Closed-patch enclosures

For the fine asset, consider two closed squares of half-width $2^{-15}$ in
Map coordinates. The deeper square is centered at $(-1,-3/16)$ and covers
four cells around a grid vertex. The shallow square is centered at
$(-1,-1/4+2^{-14})$ and covers two cells. Both squares lie strictly inside
the period-two bulb away from its center. The attracting period-two
component is the disk because its multiplier is $4(c+1)$. Its boundary
lies on the Shore. Every path from inside the disk to the Shore must reach
the circle, and the radial projection attains that distance. The signed
distance inside is therefore $D=sqrt{(x+1)^2+y^2}-1/4$.
[Period-two geometry](184-period-two-attachment.md#exact-bulb-and-shore-distance) ·
[Douady and Hubbard, Exploring the Mandelbrot Set, chapter 14](https://pi.math.cornell.edu/~hubbard/OrsayEnglish.pdf) ·
[Milnor, Theorem 6.5](https://arxiv.org/pdf/math/9905169#page=29)

The table gives signed component-wise intervals for the reported jet minus
the exact Shore jet at the same represented query. The checker prints all
six components.

| Component | Deeper square | Shallow square |
| --- | ---: | ---: |
| Value | $[-6.883,-6.782]\times 10^{-7}$ | $[-4.070,-3.403]\times 10^{-4}$ |
| $y$ derivative | $[1.465,1.529]\times 10^{-5}$ | $[0.8268,0.9431]$ |
| $xx$ Hessian entry | $[3.726,3.878]$ | $[958.2,1017.6]$ |
| $yy$ Hessian entry | $[-4.876,-4.687]\times 10^{-4}$ | $[-1881.3,-1785.7]$ |

Even on the deeper square, value accuracy better than $10^{-6}$ coexists
with a Hessian error above 3.7. On the shallow square the signed field
underestimates $D$ by over $3.4\times 10^{-4}$ and its $y$ derivative differs
by over 0.82 **at every point**. These intervals include the evaluator error
allowances, below $4.4\times 10^{-14}$ for value, $2.5\times 10^{-10}$ per
gradient entry and $2.0\times 10^{-6}$ per Hessian entry across both squares.

For a common floor $\epsilon\ge0$, the reference ruler
$\sqrt{\epsilon^2+D^2}$ is at least $0.0624$ on the deeper square. The
1-Lipschitz dependence of this ruler on its scalar field gives relative
ruler error at most $1/62400$ there for the exact stored spline. For a path
contained in that square, this gives a reciprocal length-density factor
between $1/(1+1/62400)$ and $1/(1-1/62400)$ when replacing the exact Shore
field with the exact stored spline at the same floor. The comparison
requires path containment. The scalar bound gives no comparable Hessian bound.

### How the checker encloses the difference

1. Decode the checksum-locked asset, verify the no-mask header and complete
   stencil for every possible exact or rounded cell, and range the stored
   polynomial on each intersected cell. Binomial expansion about the local
   rectangle midpoint combines like powers before bounding each monomial
   with rational arithmetic. The checker differentiates the resulting
   tensor polynomial and uses the physical spacing for derivative units.
   [Checked exact-real evaluator and joins](162-global-operational-field.md)
2. Bound the reference radius by taking integer square roots on a
   $2^{-80}$ grid. Rational interval operations then bound the exact
   derivatives using $n=(x+1,y)/r$ and $(I-nn^{\mathsf T})/r$. For the
   gradient and value differences, intersect the direct interval with
   the interval inferred by integrating derivative-error bounds from the
   center along a straight segment. The spline and reference are twice
   continuously differentiable across those segments.
3. Bound the error from machine normalization by writing the rounded grid
   coordinate as an exact shifted physical point and applying the
   cross-cell spline gains to the complete union of candidate stencils.
   Add the source-order binary64 evaluation bound, which covers separately
   rounded nearest-even operations with gradual underflow and no
   contraction. The allowances begin at the coordinates passed to the
   provider. [Cross-cell arithmetic method](181-operational-patch.md#four-cells-around-a-grid-vertex) ·
   [Rust binary64](https://doc.rust-lang.org/std/primitive.f64.html)

The compiled release probe checked all six components at 45 binary64
points, including neighbors of grid faces. The separate arithmetic model
predicted different exact and machine-selected cells at ten of those
points. The probe also verifies asset round-trip bytes and the archived
policy's center refusal. The checker rejects a box containing the center,
a box crossing the circle, and an interior box without a full asset halo.
The regional intervals come from the rational bounds, with the compiled
queries checking their application to the Rust evaluator.

From the repository root, using the retained fine asset described above:

```powershell
rtk proxy .venv/Scripts/python.exe formal/fixtures/check_181_period_two_patch.py tmp/geometry145-retained-fine.bin --offline
```

Computing the rational bounds for both squares took under a second on the
verification host, apart from building and running the Rust probe. The
report records the asset and policy hashes, evaluator source hash, Rust
target, candidate cells, and outward-rounded interval endpoints.
[Checker](../../formal/fixtures/check_181_period_two_patch.py)

The certificate is scoped to this fine asset, two closed interior squares,
and the declared arithmetic model. The experimental provider keeps
`reference-distance-unknown` at runtime. Wider coverage, other assets and
Shore components, rounded query generation before the provider, and
downstream certificate consumption remain open in [#181](https://github.com/rhythmatician/FractalSync/issues/181).
