# Geometry decision after the dynamical-ruler ablation

Keep the signed-distance graph metric as the production baseline. Preserve the
quartic dynamical conformal metric as an opt-in research profile. The current
evidence supports a local operating regime, and neither the provider results
nor the rendered comparison support a replacement or a new training gate.

## Reference geometry

The reference ruler is the reciprocal Poincaré density at curvature −1 on
each component of the complement of the finite Shore. In a hyperbolic interior
component the normalized multiplier gives the formula. The finite exterior
formula uses the parameter Böttcher coordinate. Other interiors have an
abstract domain metric. A finite nonescape result carries no multiplier-chart
authority. The [conformal proof audit](184-conformal-contract-acceptance.md)
identifies the classical inputs, checked consequences, and analytic assumptions.

The main cardioid and period-two bulb give exact test cases, including the
two-component attachment at $c=-3/4$. The inland Koebe bracket gives a factor
two for conformal lengths, four for kinetic terms, and one octave for scale.
Those comparisons concern the scalar and conformal metric. Far offshore the
ruler grows like $|c|\log|c|$, so a uniform Shore-distance ratio fails.
[Cardioid](184-main-cardioid-ruler.md), [attachment](184-period-two-attachment.md),
[exterior](184-exterior-ruler-bounds.md).

A positive quadratic floor guarantees a distance ceiling along a straight
crossing and still permits discontinuous limiting curvature at the cardioid
cusp. With the quartic floor the reference curvature tends to zero along finite
Shore, under the distance-power bounds in the gluing argument. The metric
has a $C^3$ extension with the stated collar estimates, and the cusp obstructs
$C^4$. The smoothing choice therefore matters to the connection and curvature.
[Quadratic curvature](184-conformal-floor-curvature.md),
[quartic gluing](184-shore-floor-gluing.md).

## Provider and Physics comparisons

Three profiles use the same frozen Rust integrator and held inputs in each
fixture. The retained distance graph and its conformal variant isolate the
rank-one term. The two conformal profiles isolate the choice of ruler.
Period-two, cardioid, and finite-exterior runs have common admissible patches
and step-halving evidence. The retained field's reference-distance accuracy is
unknown. The exterior chart uses a fixed truncation, for which the sampled
scalar tail guard and next-iterate differences are diagnostics. The analytic
Cauchy bounds show why a small scalar discrepancy can coexist with a wide
Hessian enclosure. [Period-two](183-period-two-physics-ablation.md),
[cardioid and exterior](183-multiregion-physics-ablation.md),
[exterior jet study](184-exterior-ruler-jets.md).

At the attachment the two distance profiles use the same analytic tangent-line
field, and the quartic profile uses the analytic two-component ruler. All
three cross on the real axis at both step sizes. The candidate's measured
crossing time differs from the independent damped-axis estimate by $4.35$ μs.
The graph metric's reference ratio of largest to smallest eigenvalue at the
launch point is $1+(1/\ln^2 2)(4/5)\approx2.67$. The reference ratio at the
attachment and for the conformal metrics is one. Conformal conditioning of
the matrix leaves separate questions about the ruler scale, derivatives,
operator forces, and discrete stability. Both tangent-line profiles have a
synthetic field and all three start with equal coordinate velocity. The
launch energies consequently differ. [Attachment crossing](183-attachment-crossing-fixture.md).

## Rendered motion on the attachment paths

The frozen Julia renderer measures how these different trajectories change
the displayed image. It holds the shader, time, palette, zoom, lighting, and
iteration limit fixed. Each fine trajectory supplies 26 points at a common
40 ms spacing, including the ends. Here the change is the root mean square of
normalized RGB8 differences over the 80×80 framebuffer and three channels.
Chromium with SwiftShader rendered the tracked Julia fragment shader at
500 iterations. A repeated frame following an intervening seed was byte-identical.

| Profile | Crossing time (s) | Median frame change | Peak frame change | Start-to-end change |
| --- | ---: | ---: | ---: | ---: |
| Tangent-line distance graph | 0.426075 | 0.01966 | 0.02955 | 0.07769 |
| Tangent-line distance conformal | 0.531678 | 0.01346 | 0.02550 | 0.06574 |
| Quartic dynamical conformal | 0.526618 | 0.01482 | 0.02538 | 0.06846 |

Both conformal trajectories cross later and show smaller median frame changes
than the graph trajectory on this matched-coordinate launch. The quartic
trajectory produces slightly more median image change than the distance
conformal one, with a similar peak. Different energies, endpoint positions,
and finite display resolution affect these comparisons. The separate
[frozen stencil study](151-frozen-julia-sensitivity.md) found a near-Shore
stencil-locality benefit with the installed distance provider and no resolved
convergence across its sampled spacings. The motion table measures one display
configuration on one crossing. Whether performers experience these changes as
more controllable is a playtesting question for [#186](https://github.com/rhythmatician/FractalSync/issues/186).

Reproduce the table from the repository root in an environment with the frozen
probe prerequisites. The optional retained-asset flag in the attachment note
adds a separate provenance check. The path samples are identical in both runs.

```sh
cargo run --offline -p runtime_core --features experimental-operational-field --example attachment_ruler_crossing > attachment.json
node frontend/scripts/check_julia_sensitivity.mjs julia-motion.json attachment.json
```

Use a shell that writes UTF-8 when redirecting JSON. The second command records
the source JSON and shader hashes, renderer and presentation settings, and
per-frame measurements in `attachment_motion`.

## Decision boundary

The experimental profile retains two position coordinates, derived Mandelbrot
scale, independent Julia presentation, music-independent Physics, and the
existing held input seam. The fixtures refuse calls outside their chart or
field policy. Production continues using the distance provider. Navigation
fields and orbit traps remain independent. A future
replacement proposal needs reproducible provider correspondence at true-Shore
crossings, broader region and derivative coverage, and evidence from actual
Controls and perceptual evaluation. [#183](https://github.com/rhythmatician/FractalSync/issues/183),
[#181](https://github.com/rhythmatician/FractalSync/issues/181),
[#186](https://github.com/rhythmatician/FractalSync/issues/186).
