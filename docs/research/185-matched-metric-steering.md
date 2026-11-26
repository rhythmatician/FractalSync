# Steering demand under three metrics on the same trap arcs

The signed-distance graph metric and its conformal restriction give
different steering demands near the second-iterate trap saddle. On
these fixed arcs, the graph metric amplifies the inner turn and reduces
the outer turn relative to the distance conformal metric. Both
conformal cases have equal limiting magnitudes on the two sides.
This comparison isolates geometry and uses the same parameter-plane
paths for all three cases. [Physics ablation #183](https://github.com/rhythmatician/FractalSync/issues/183), [navigation research #185](https://github.com/rhythmatician/FractalSync/issues/185)

## Compare at equal intrinsic speed

Let $u=dc/d\theta$ for a regular prescribed path and let
$a=d^2c/d\theta^2+\Gamma(u,u)$ be its covariant phase acceleration.
For a symmetric positive metric $G$, put $S=u^\mathsf{T}Gu>0$.
The path speed per phase is $\sqrt S$. Metric compatibility gives
$dS/d\theta=2u^\mathsf{T}Ga$, so the covariant acceleration after
unit-speed reparameterization is

$$
\frac{1}{S}\left(a-u\frac{u^\mathsf{T}Ga}{S}\right).
$$

Its signed curvature is

$$
\kappa_G=\frac{\sqrt{\det G}\det(u,a)}{S^{3/2}},
$$

by the planar Gram identity. At constant intrinsic speed $v$,
normal acceleration has signed value $v^2\kappa_G$.
Lean checks the projection's orthogonality and squared norm. The
connection and arc-length derivation use the standard metric
compatibility identity. [Pointwise calculation](../../formal/FractalSync/TrapMetricComparison.lean)

The experiment uses the [exact trap lift](185-exact-trap-contours.md).
Set $\delta=|1-4r|$, $r=|P(c)|$, and
$\theta=\pi+\delta\xi$ for $\xi\in[-4,4]$. All 513 phase samples
lie within the main cardioid and the two Rust distance-metric calls
succeeded at every sample. The cases are the default distance graph,
the same Rust metric with `lambda_sq=0`, and the analytic
main-cardioid conformal metric with a quartic ruler floor. All use
$\varepsilon=10^{-4}$. The Rust calls are
`manifold_induced_metric` and `manifold_geodesic_acceleration`.
[Runtime manifold](../../runtime-core/src/manifold.rs), [cardioid quartic floor](184-quartic-ruler-floor.md)

## Closest-phase result

If $G$ approaches $G_s$ at the saddle and $\Gamma$ is bounded,
the leading curvature comes from the coordinate acceleration of the
lift. The connection's contribution to curvature is bounded, since
$\Gamma(u,u)=O(|u|^2)$ and the denominator is comparable to $|u|^3$.
For limiting Euclidean unit tangent $T$ at the closest phase, the
[Euclidean curvature limit](185-trap-saddle-curvature.md#where-the-turn-concentrates)
therefore gives

$$
\sqrt{\delta}\,\kappa_G\longrightarrow
\frac{2\operatorname{sgn}(1-4r)\sqrt{\det G_s}}
     {(T^\mathsf{T}G_sT)^{3/2}}.
$$

The inner branch has a vertical tangent and the outer a horizontal
one. The ratio of limiting magnitudes is
$(G_{s,xx}/G_{s,yy})^{3/2}$. The sampled graph metric at
$c_s=-1/2$ has diagonal entries $49.4000$ and $16.0456$,
which would give a ratio $5.4020$ under these local hypotheses.

| Metric | Inner $\sqrt{\delta}\kappa_G$ at $10^{-6}$ | Outer $\sqrt{\delta}\kappa_G$ at $10^{-6}$ | Limiting coefficients from $G_s$ |
| --- | ---: | ---: | ---: |
| Distance graph | $+0.877775$ | $-0.162174$ | $+0.876065,\ -0.162174$ |
| Distance conformal | $+0.501289$ | $-0.499277$ | $\pm0.499288$ |
| Cardioid quartic conformal | $+0.403119$ | $-0.401922$ | $\pm0.401924$ |

The ratio in the graph case comes from the metric's directional
weights. The provider outputs support the predicted coefficients in
this sample. Continuity and uniform connection bounds for the
compiled provider would need separate evidence.

## Speed and time

For a constant-speed traversal, the script uses
$v=\sqrt{1/\max|\kappa_G|}$, where the maximum is taken over sampled
points, then divides trapezoidal arc length by $v$. The table gives
these sampled traversal times at a normal acceleration budget of
one, in model-time units.

| Metric | Inner, $\delta=10^{-4}$ | Outer, $\delta=10^{-4}$ | Inner, $\delta=10^{-6}$ | Outer, $\delta=10^{-6}$ |
| --- | ---: | ---: | ---: | ---: |
| Distance graph | 0.61975 | 0.36957 | 0.19946 | 0.11683 |
| Distance conformal | 0.39760 | 0.39953 | 0.12625 | 0.12631 |
| Cardioid quartic conformal | 0.44365 | 0.44526 | 0.14073 | 0.14078 |

The travel times fall roughly as $\delta^{1/4}$ as the arc
shrinks, even though the selected speeds also fall. Doubling the number
of phase intervals changed the sampled curvature peaks by less than
0.014% across the trials.

For the analytic cardioid metric we can also bound the entire arc.
The discriminant satisfies $\delta\le |1+4P(c)|\le6\delta$ in
the sample window. Each point is within $\sqrt{6\delta}/2$ of the
saddle, in its radius-$1/4$ disk inside the cardioid. The quartic
ruler at the saddle is below $0.202$, and its gradient norm is at
most $2$ in this disk. Substituting these bounds in the curvature
formula gives

$$
|\kappa_G|\le
\frac{200}{99}(0.202+\sqrt{6\delta})
  \left(\frac1{\sqrt\delta}+\sqrt{6\delta}\right)+2.
$$

Using $\sqrt6<2.45$ gives curvature caps $12$, $48$, and $416$
at $\delta=10^{-2},10^{-4},10^{-6}$, respectively, on both sides.
At budget one, a continuous-arc constant speed $1/\sqrt K$ is
therefore sufficient in the analytic metric. The
[stop-to-stop construction](185-trap-steering-budget.md#a-stop-to-stop-benchmark)
also allows braking and restarting within the window. Enclosing
the compiled distance metrics between samples remains open.

Run the experiment with the project venv and the compiled
`geometry-provider/6` binding.

~~~powershell
.venv/Scripts/python.exe formal/checks/check_trap_steering_budget.py
~~~

The comparison concerns prescribed arcs with no potential or Controls
calculation. A Physics rollout with held controls, measurement of
Julia appearance, and comparisons across other components remain
part of the larger ablation.
