# Issue 157: proof reuse if the underlying field changes

The user has reported discussion of replacing the underlying signed-distance
field with a field intrinsic to the quadratic dynamics of the Mandelbrot
geometry, modified to permit Shore crossings. No replacement formula, derivative
contract, or implementation is specified in this PR. The user's description of
the current SDF as an approximation of that proposed field is project context;
it is not a certified value, derivative, or topology comparison.

This note records the dependencies of the existing results so that the current
PR can finish its numerical work without choosing the replacement. It adds no
Lean theorem and changes no runtime definition. Rust remains runtime authority
under [ADR 0001](../adr/0001-rust-first-parity.md).

## 1. The reusable mathematical interface

The general metric and mechanics layer accepts a positive scalar ruler
\(\rho\), its gradient \(u\), and, where derivatives are needed, its Hessian
\(H\). Its metric is

\[
G=\rho^{-2}(I+a^2u\otimes u),\qquad
W=1+a^2\|u\|^2.
\]

The pointwise algebra in
[`ScaleMetric.lean`](../../formal/FractalSync/ScaleMetric.lean) quantifies
directly over \(\rho,a,u\); it does not require a distance field. Inverse
identities require \(\rho\ne0\), and positive definiteness is stated with
\(\rho>0\). The contraction in
[`RankOneInverseBounds.lean`](../../formal/FractalSync/RankOneInverseBounds.lean),

\[
\|G^{-1}w\|\le\rho^2\|w\|,
\]

does not assume \(\|u\|\le1\). The separate conclusion
\(1\le W\le1+a^2\) does require that bound on \(u\). A replacement can retain
the rank-one algebra while requiring different anisotropy and regional
regularity bounds.

| Existing layer | Exact interface to retain or supply | Consequence of changing the field |
| --- | --- | --- |
| Metric, inverse, determinant, impulse energy | The displayed rank-one metric; nonzero or positive ruler as stated; actual impulse budget | The algebra and fixed-configuration identities remain applicable. See [ScaleMetric](../../formal/FractalSync/ScaleMetric.lean) and [MetricImpulse](../../formal/FractalSync/MetricImpulse.lean). |
| Connection and continuous work-energy | Actual identities \(u=\nabla\rho\), \(H=Du\), potential derivative, and equation of motion | The field-level theorems remain applicable after these derivative and motion premises are supplied. Arbitrary coefficient arrays alone do not identify a geometric connection or an energy derivative. See [ContinuousMechanics](../../formal/FractalSync/ContinuousMechanics.lean). |
| Local ODE existence and quantitative trajectories | Coefficient regularity or regional norm/Lipschitz certificates, positive domain radius, and ruler floor | The generic construction accepts a new acceleration field. Its numerical horizon, tube, and mesh constants must be recomputed. See [LocalDynamics](../../formal/FractalSync/LocalDynamics.lean) and [QuantitativeDynamics](../../formal/FractalSync/QuantitativeDynamics.lean). |
| Ordered-step error, interpolation, and accumulated residuals | An actual reference ODE, acceleration Lipschitz bound, step/update residuals, and domain bounds | The error machinery can be instantiated independently of distance-to-set geometry. The update formula must still match the numerical method being analyzed. See [TrajectoryError](../../formal/FractalSync/TrajectoryError.lean), [TrajectoryInterpolation](../../formal/FractalSync/TrajectoryInterpolation.lean), and [AccumulatedDynamics](../../formal/FractalSync/AccumulatedDynamics.lean). |
| Signed RK work and energy ledger | Actual stage power \(P=DE[F]\), a Lipschitz energy differential, bounded stage slopes and a certified closed ball | The quadrature and telescoping proof is field independent. Its constants and power identity require the replacement's own evidence. See [the signed-work extension](159-rk-work-ledger.md) and [RungeKuttaWork](../../formal/FractalSync/RungeKuttaWork.lean). |
| Catalog cancellation and held-control residual | The same potential/drive/drag formulas, metric raising, command assumptions, and local coefficient bounds | The symbolic cancellation survives a field substitution; its quantitative constants require new evidence. See [CancelledCatalogBounds](../../formal/FractalSync/CancelledCatalogBounds.lean) and [HeldControlResidual](../../formal/FractalSync/HeldControlResidual.lean). |
| Coordinate and frame transformations | The stated chart invertibility, differentiability, and connection/metric transformation premises | Generic covariance laws remain available. They do not supply the replacement metric's particular curvature formula or its global geometry. See [NonlinearCoordinates](../../formal/FractalSync/NonlinearCoordinates.lean) and [MovingFrames](../../formal/FractalSync/MovingFrames.lean). |

There is a useful distinction inside the ODE row. The definition
`mechanicalAcceleration` accepts separate ruler, gradient, Hessian, and force
fields. `exists_local_mechanical_certificate` proves local existence from their
\(C^1\) regularity and a nonzero ruler. To interpret that solution as the
mechanics of an actual ruler, the derivative identities must also hold. They
are explicit premises of `hasDerivAt_totalEnergy_of_fields`; the current
[operational adapter](../../formal/FractalSync/OperationalDynamics.lean)
discharges them using the actual spline derivatives.

## 2. Retaining the regularizer permits conditional reuse

If a replacement field \(f\) retains

\[
\rho(x)=\sqrt{\epsilon^2+f(x)^2},\qquad\epsilon>0,
\]

the scalar calculus in [Ruler](../../formal/FractalSync/Ruler.lean),
[RulerDifferential](../../formal/FractalSync/RulerDifferential.lean), and
[RulerJetBounds](../../formal/FractalSync/RulerJetBounds.lean) applies once the
required derivatives of \(f\) are proved. In particular, the checked chain-rule
expressions are

\[
D\rho=\frac f\rho Df,\qquad
D^2\rho=\frac f\rho D^2f+
\frac{\epsilon^2}{\rho^3}Df\otimes Df.
\]

The scalar map \(r\mapsto\sqrt{\epsilon^2+r^2}\) is one-Lipschitz. Its
composition with a field inherits that field's Lipschitz bound. The existing
one-Lipschitz distance-field instance uses Mathlib's actual `infDist` bound;
one-Lipschitz regularization does not turn an arbitrary input field into a
one-Lipschitz spatial field. Likewise, positive regularization supplies a ruler
floor but does not by itself prove that the underlying field is smooth.

The local estimates in
[`LocalRulerBounds.lean`](../../formal/FractalSync/LocalRulerBounds.lean) have
both generic and evaluator-specific parts. Its scalar and bilinear estimates
accept regional value, differential, and Hessian bounds. Its later
`physicalField` corollaries are adapters to the existing spline. A different
evaluator needs its own actual-derivative adapter and evidence for the regional
bounds. In particular, the improved fixture's small field value and small
Hessian variation are established properties of that fixture, not default
properties of every regularized field.

If the replacement also changes the definition of \(\rho\), the scalar chain
rules and their constants must change. If it changes the rank-one metric or
the force/control catalog, the affected algebraic interfaces must change as
well. The generic ODE and numerical-error results can still be applied to a
newly certified vector field, but that is a new instantiation.

## 3. Evaluator proofs and concrete fixture evidence are separate

The current operational field is a globally glued quintic spline selected by
the actual query-cell floor operation. Its smoothness, patch joins, derivative
queries, coefficient error gains, and finite-halo address facts concern that
specific evaluator. They can be reused for new coefficient data if the same
interpolation and indexing definitions are retained. They do not prove fidelity
of a different analytic or numerical evaluator. The relevant boundaries are
documented in [operational reference fidelity](162-operational-reference-fidelity.md)
and formalized in
[OperationalFieldQueryCorrespondence](../../formal/FractalSync/OperationalFieldQueryCorrespondence.lean),
[OperationalFieldFidelity](../../formal/FractalSync/OperationalFieldFidelity.lean),
and [OperationalHalo](../../formal/FractalSync/OperationalHalo.lean).

The 256-coefficient asset is a research fixture sampled from
\(P(x,y)=x^2+2xy-3y^2\). Its reconstructed reference includes the spline bias
\(P-h^2\). The proved value, differential, Hessian, local mechanical, and
trajectory bounds refer to those literal coefficients and the specified
spacing and phase region. A quadratic polynomial fixture is not a certificate
for the Mandelbrot iteration merely because both involve quadratic expressions.
See [ConcreteAssetFidelity](../../formal/FractalSync/ConcreteAssetFidelity.lean),
[ConcreteLocalFieldBounds](../../formal/FractalSync/ConcreteLocalFieldBounds.lean),
and [ImprovedConcreteTrajectory](../../formal/FractalSync/ImprovedConcreteTrajectory.lean).

Neither this fixture nor the general error contracts discharge production
asset/reference fidelity, compiled storage/mask correspondence, machine
indexing, query rounding, or IEEE arithmetic. Replacing the field does not
remove those implementation obligations.

## 4. Results that depend on actual distance to Shore

The geometric path construction takes a set \(S\) and uses
\(d(x,S)=\operatorname{Metric.infDist}(x,S)\) in the definition of its AC
length. It is a different level of specificity from the arbitrary smooth-field
mechanics above. The following conclusions must not be transferred by simply
substituting \(|f(x)|\) for \(d(x,S)\):

- **Nearest-segment calibration.**
  [DistanceToSet](../../formal/FractalSync/DistanceToSet.lean) proves
  \(d(p+t(x-p),S)=t\,d(x,S)\) for an actual nearest point. The segment's exact
  length and the attained Shore distance
  \(F_{\epsilon,a}(d(x,S))\) in
  [IntrinsicShoreDistance](../../formal/FractalSync/IntrinsicShoreDistance.lean),
  together with the [all-nearest-Shore-minimizer classification](../../formal/FractalSync/ShoreMinimizers.lean),
  use this calibration.
- **The path metric and its singular limit.**
  [PairwiseDistance](../../formal/FractalSync/PairwiseDistance.lean),
  [ZeroFloorLimit](../../formal/FractalSync/ZeroFloorLimit.lean), and
  [ZeroFloorGeodesics](../../formal/FractalSync/ZeroFloorGeodesics.lean) concern
  the specified distance-derived length functional. Their crossing lower
  bounds, componentwise limits, and selected-geodesic subsequence statements
  are not assertions about an unspecified field-derived metric.
- **Flat and embedded Shore geometry.**
  [FlatShoreIntrinsic](../../formal/FractalSync/FlatShoreIntrinsic.lean) identifies
  actual Euclidean distance to a line with its absolute normal coordinate.
  [EmbeddedShoreMetric](../../formal/FractalSync/EmbeddedShoreMetric.lean)
  identifies the distance-derived ruler in a certified normal tube. The
  resulting curvature, threshold, and minimizer formulas depend on these
  identifications. The conventional global cut-locus and bifurcation arguments
  in [the flat-Shore note](160-flat-shore.md) retain their existing formal
  boundary and are not replacement-field results.

Two pieces remain independently useful. The scalar
[radial-profile analysis](../../formal/FractalSync/RadialProfile.lean) remains a
theorem about its stated integral, although its identification with a new
metric's Shore distance would need proof. Also,
`continuousOn_exists_shore_crossing` in
[ComponentCrossing](../../formal/FractalSync/ComponentCrossing.lean) is purely
topological: a continuous interval path joining different components of
\(S^c\) meets \(S\). It requires neither a distance field nor closedness of
\(S\). Applying it to a replacement first requires identifying the relevant
set and component relation; obtaining a length lower bound then requires the
replacement's analytic estimates.

Value approximation alone supplies none of the missing derivative or
zero-set/component identifications. Even exact agreement of zero sets does not
identify the induced metric, nearest-segment lengths, curvature, or singular
limit. Those conclusions depend on how the field varies as well as where it
vanishes.

## 5. Completed exact-real RK bridge and remaining runtime correspondence

The current
[`operational_engine.rs`](../../runtime-core/src/operational_engine.rs)
implements classical RK4 on the four real mechanical coordinates
\(z=(x,y,v_x,v_y)\). For an outer step \(\Delta t\), `Rk4` uses one substep and
`Rk4FourSubsteps` uses four with \(h=\Delta t/4\). The same constructed control
covector \(q\) is supplied to every stage of every substep. At each stage, the
provider, potential, wall force, drag, inverse metric, and connection are
reevaluated at that stage's state.

The concrete numerical certificate uses the research fixture's parameters
\(\epsilon=a=\beta=1\), \(\kappa=\log 2\), \(\mu=1/\pi\), and unit intrinsic
drive in a fixed positive first-axis direction. These are not the default Rust
profile. In [`manifold.rs`](../../runtime-core/src/manifold.rs),
\(a^2=\texttt{lambda\_sq}/(\log 2)^2\), so the exact-real parameter matching
\(a=1\) is \(\texttt{lambda\_sq}=(\log 2)^2\).
[`controls.rs`](../../runtime-core/src/controls.rs) has `MAX_DRIVE_FORCE = 2`,
so unit intrinsic drive corresponds to command magnitude \(1/2\) in the
nonzero, unclipped branch. `FrozenRollout::attempt_step` accepts an already
constructed covector; it does not reconstruct a command itself. The numerical
history being certified therefore assumes that a caller supplies the fixture's
exact \(Q_d(y_i)\) at each outer anchor \(y_i\). It does not certify arbitrary
caller inputs, command construction in floating point, or a default-profile
rollout.

The source has three distinct outputs and checks:

| Quantity | Actual runtime mapping | Checked scope or remaining correspondence |
| --- | --- | --- |
| Mechanical state | Four classical RK stages and the weighted endpoint update; four-substep mode composes four such updates | The [held-RK extension](159-held-runge-kutta.md) proves exact-real stage/endpoint containment and update residuals for the fixture's stated caller policy |
| Work increment | The same RK weights applied to the fifth RHS component, \(v\cdot(q+Q_{\rm drag})\); this component does not enter the mechanical stage states | The [signed-work extension](159-rk-work-ledger.md) identifies the actual stage power, proves exact-real cumulative energy/work-ledger bounds, and proves uniform convergence of the affine ledger interpolant to the reference's actual power integral. IEEE power evaluation and summation remain separate |
| Endpoint acceptance | `checked_endpoint` at each internal substep endpoint and the final outer endpoint, followed by final finite-ledger checks before `attempt_step` assigns the new state | Provider regularity, domain and finite-arithmetic correspondence; the weighted endpoint generally differs from the fourth stage |

The reusable RK bounds depend on a vector field's norm and variation on a
region, the stage recurrence, and the held-control residual. None needs the
identity \(f=d(\cdot,S)\). For the four-substep profile the holding residual
must be measured from the outer sample state, because the runtime does not
reconstruct \(q\) at internal substep boundaries. A reference-trajectory speed
bound alone does not establish numerical stage containment.

The [held-RK extension](159-held-runge-kutta.md) completes the exact-real
stage and method-residual bridge, and the
[signed-work extension](159-rk-work-ledger.md) adds the actual stage-power
quadrature, finite energy/work-ledger estimate and uniform convergence to
the actual reference power integral. Applying these results to another
field requires its regional certificates, energy/power identity and the same
control/update semantics. They do not certify Rust floating-point
evaluation or summation, compiled provider correspondence, or operational
acceptance. `attempt_step` checks the proposed outer ledger's finiteness;
it has no energy/work-residual threshold or drift-based refusal criterion.

## Scope of this PR

This increment finishes the exact-real numerical bridge with the present
ruler, catalog, and fixture. The distance-to-Shore results remain statements
about their actual distance-derived model. The proposed field remains
a future design decision; this PR supplies reusable interfaces and explicit
instantiation boundaries without defining or implementing that replacement.
