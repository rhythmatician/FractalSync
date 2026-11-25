# #159: changing-metric steps, local energy defects, and covariant steering

The exact-real proof package now treats the velocity-then-position step with
energy evaluated at its actual new configuration. It proves a quadratic local
energy-defect bound from actual differential or Hessian certificates, whole-
segment wall-domain conditions, and the ordered impulse ledger. A separate
module proves metric-norm preservation for continuous covariant steering along
a moving configuration.

The source modules are
[MovingMetricMechanics.lean](../../formal/FractalSync/MovingMetricMechanics.lean)
(18 theorem declarations) and
[CovariantSteering.lean](../../formal/FractalSync/CovariantSteering.lean)
(8 theorem declarations). They extend the
[frozen-metric results](159-discrete-mechanics.md) and
[actual continuous mechanics](159-differential-mechanics.md). Rust remains the
runtime authority; these proofs make no floating-point correspondence claim.

## 1. Actual fields and the ordered step

Let `E` be a real inner-product space and let the configuration fields be
`rho : E → ℝ`, `gradient : E → E`, and `U : E → ℝ`. Where used geometrically,
the hypotheses identify the actual derivatives

\[
d\rho_x=\langle u,\cdot\rangle,\qquad
d(\nabla\rho)_x=H,\qquad
dU_x=\langle \nabla U,\cdot\rangle,
\quad u=\nabla\rho(x).
\]

The metric and kinetic energy remain the existing definitions

\[
g_x(v,w)=\frac{\langle v,w\rangle+a^2\langle u,v\rangle\langle u,w\rangle}{\rho(x)^2},
\qquad K_x(v)=\frac12g_x(v,v).
\]

Define the actual phase energy and proposed step by

\[
\mathcal E(x,v)=K_x(v)+U(x),\qquad
v_1=v+\Delta t\,A,\qquad x_1=x+\Delta t\,v_1.
\tag{1}
\]

This matches the algebraic order of `integrate_step` in
[manifold.rs](../../runtime-core/src/manifold.rs). The acceleration condition is

\[
A+\Gamma_x(v,v)=G_x^{-1}(Q-\nabla U).
\tag{2}
\]

Here `Q` contains the nonconservative covectors, including drag when present.
The theorem assumes (2) for the exact-real values supplied to the step; it does
not identify approximate provider derivatives with the actual field jets.
The physical specialization uses a positive ruler on the region under
consideration. The algebraic step theorem requires `rho(x)≠0` at the initial
connection evaluation and actual energy regularity along its phase segment;
the later impulse additionally requires `rho(x1)≠0`.

## 2. The actual phase differential and its cancellation

`fderiv_phaseEnergy_apply` differentiates (1) from the field hypotheses:

\[
d\mathcal E_{(x,v)}(d,w)
=\frac12\,dg_x[d](v,v)+g_x(v,w)+\langle\nabla U,d\rangle.
\tag{3}
\]

Define the position-only energy rate

\[
B_x^v(d)=\frac12\,dg_x[d](v,v)+\langle\nabla U,d\rangle.
\]

The checked metric compatibility identity gives

\[
B_x^v(d)=g_x(v,\Gamma_x(d,v))+\langle\nabla U,d\rangle.
\tag{4}
\]

In particular, (2) implies the actual differential identity

\[
\boxed{d\mathcal E_{(x,v)}(v,A)=\langle v,Q\rangle.}
\tag{5}
\]

Both the connection and conservative-potential contributions cancel in (5).
The proposed phase increment has the exact decomposition

\[
(x_1,v_1)-(x,v)
=\Delta t\,(v_1,A)
=\Delta t\,(v,A)+\Delta t^2(A,0).
\tag{6}
\]

Consequently its linearized energy change is exactly
`dt * ⟨v,Q⟩ + dt² * B(A)`. The second term is a consequence of updating
position with the new velocity.

## 3. A quadratic local energy defect for the complete step

Equip phase space `E × E` with its ordinary product max norm for the following
analytic certificate:

\[
\|(d,w)\|_{\rm phase}=\max(\|d\|,\|w\|).
\]

This norm controls the differential remainder. It does not replace the
physical ruler metric in `K` or in the equation of motion.

Assume `Ecal` is differentiable on the entire closed phase segment from
`(x,v)` to `(x1,v1)`, and that its actual differential varies from the initial
one by at most `L` times phase displacement. Then
`positionVelocityStep_energy_defect` proves

\[
\boxed{
\left|\mathcal E(x_1,v_1)-\mathcal E(x,v)
-\Delta t\langle v,Q\rangle\right|
\le\Delta t^2\left[
|B_x^v(A)|+\frac L2\max(\|v_1\|,\|A\|)^2
\right].
}
\tag{7}
\]

The proof applies the already checked quadratic Taylor remainder to the
actual phase energy and increment (6), then uses (5). The assumptions concern
actual field calculus and differential variation, rather than a presumed
energy-error bound.

`positionVelocityStep_energy_defect_of_hessian_bound` supplies the variation
condition from an actual phase-energy Hessian certificate: for every point
`z` on that same closed segment,

\[
d(d\mathcal E)_z=J(z),\qquad \|J(z)\|\le L,
\qquad L\ge0.
\tag{8}
\]

This is the Hessian of **phase energy**, not merely the ruler Hessian `H_rho`.
Obtaining (8) from ruler data can require higher field regularity, including
derivatives of `H_rho`. No such bound follows from `rho>0`, positive epsilon,
or a numerical Hessian-size flag. The formal certificate covers the entire
proposed segment, including its endpoints.

Equation (7) includes a zero step and permits signed real steps. It supplies
a uniform `O(dt²)` local estimate only when the displayed coefficients and
certificates stay bounded as the step varies. It does not establish a global
rollout convergence rate or exact energy conservation. If the initial work is
negative, it must dominate the remainder before the bound guarantees a
nonincreasing total energy.

The rate coefficient itself has a checked explicit expansion and bound:

\[
B_x^v(d)=
-\frac{\langle u,d\rangle}{\rho}g_x(v,v)
+\frac{a^2}{\rho^2}\langle Hd,v\rangle\langle u,v\rangle
+\langle\nabla U,d\rangle,
\]

\[
\boxed{
|B_x^v(d)|\le
\left[
\frac{\|u\|}{\rho}g_x(v,v)
+\frac{a^2}{\rho^2}\|H\|\|u\|\|v\|^2
+\|\nabla U\|
\right]\|d\|,
\qquad \rho>0.
}
\tag{9}
\]

Thus the first-derivative part of the coefficient can be bounded by actual
ruler, gradient, Hessian, and force magnitudes. The phase-Hessian remainder
in (8) is still a separate certificate.

## 4. Exact metric-change bookkeeping and the subsequent impulse

The previously conventional decomposition is now checked for the actual
configuration fields:

\[
\begin{aligned}
\mathcal E(x_1,v_1)-\mathcal E(x,v)
={}&\Delta t\,g_x(v,A)+\Delta t^2K_x(A)\\
&+[K_{x_1}(v_1)-K_x(v_1)]+[U(x_1)-U(x)].
\end{aligned}
\tag{10}
\]

The changing-metric and potential terms in (10) need not vanish. Equation (7)
bounds their combined effect with the velocity update under its regularity
and dynamics hypotheses.

The executable wrapper `integrate_motion_controls` in
[controls.rs](../../runtime-core/src/controls.rs) applies its impulse at `x1`.
The new checked ordered ledger uses that same location:

\[
v_2=v_1+G_{x_1}^{-1}J,
\]

\[
\boxed{
\mathcal E(x_1,v_2)-\mathcal E(x,v)
=[\mathcal E(x_1,v_1)-\mathcal E(x,v)]
+\langle v_1,J\rangle+\frac12\langle J,G_{x_1}^{-1}J\rangle.
}
\tag{11}
\]

The inverse metric in (11) is evaluated at the updated position. The old
configuration's impulse metric cannot be substituted without an additional
error estimate. The instantaneous impulse itself does not change position.

## 5. A sufficient whole-segment wall-domain condition

For any radius `R`, Lean proves that

\[
\|x\|+|\Delta t|\,\|v_1\|<R
\tag{12}
\]

keeps every point `x+t dt v1`, `0≤t≤1`, inside the open radius-`R` ball.
A condition using the old velocity and acceleration is

\[
\boxed{\|x\|+|\Delta t|\,\|v\|+\Delta t^2\|A\|<R.}
\tag{13}
\]

The triangle inequality for `v1=v+dt A` makes (13) sufficient for (12).
Setting `R=2` gives the secant wall's physical disk. Both endpoints and every
intermediate point of the proposed affine position interpolation are covered.

This does not enclose the exact ODE trajectory or establish unconditional
multi-step invariance. It also does not discharge the runtime's additional
finite-value and wall-force conditioning checks, or floating-point error in
computing the proposed position.

## 6. Covariant steering in the moving metric

For two vectors `e,f`, define

\[
J_{e,f}d=g(e,d)f-g(f,d)e.
\tag{14}
\]

Lean proves its linearity and `g(d,J_{e,f}d)=0` for arbitrary `e,f`. When
`e,f` are a metric orthonormal pair, the checked identities `J e=f` and
`J f=-e` identify the oriented quarter-turn **on their frame plane**. In a
higher-dimensional ambient space, (14) is not a quarter-turn on all vectors.

For a base trajectory with velocity `b`, the actual metric-pairing derivative
is checked to be

\[
\frac d{dt}g_t(v,w)
=g_t(\dot v+\Gamma_t(b,v),w)
+g_t(v,\dot w+\Gamma_t(b,w)).
\tag{15}
\]

Consequently the covariant steering equation

\[
\boxed{\dot d+\Gamma_t(b,d)=\omega(t)J_{e(t),f(t)}d}
\tag{16}
\]

has zero actual metric-norm-square derivative. The integrated theorem proves

\[
g_{t_1}(d(t_1),d(t_1))=g_{t_0}(d(t_0),d(t_0)).
\tag{17}
\]

In particular an initially metric-unit heading remains metric-unit. The
connection term in (16) accounts for changing ruler coefficients; it cannot
be omitted by reusing the frozen-metric angle derivative.

The trace theorem assumes the actual ruler and gradient derivatives along
the moving base, the actual heading derivative, and (16) on the open time
interval, with continuous metric-norm-square on the closed interval. A
separate field-level theorem obtains the trace derivatives by the actual
Fréchet chain rule for `gamma`, `rho`, and `gradient`. Frames and rates may
vary with time. The conclusion is conditional preservation for an actual
solution, not an ODE existence theorem, a frame-construction theorem, or a
norm-preserving discrete algorithm.

## 7. Checked declarations and remaining correspondence

All declarations use namespace `FractalSync.ScaleMetric`.

| Module | Principal declarations and exact role |
| --- | --- |
| `MovingMetricMechanics` | `differentiableAt_phaseEnergy`, `hasDerivAt_phaseEnergy_line`, `fderiv_phaseEnergy_apply`, and `fderiv_phaseEnergy_motion`: actual phase calculus and work cancellation. |
| `MovingMetricMechanics` | `positionVelocityStep_eq_add`, `positionVelocityStep_increment`, and `fderiv_phaseEnergy_step_increment`: exact ordered step and differential. |
| `MovingMetricMechanics` | `positionVelocityStep_energy_defect` and `positionVelocityStep_energy_defect_of_hessian_bound`: (7) and (8). |
| `MovingMetricMechanics` | `positionVelocityStep_energy_decomposition`, `positionEnergyRate_eq_connection`, `positionEnergyRate_expanded`, and `abs_positionEnergyRate_le`: (4), (9), and (10). |
| `MovingMetricMechanics` | `affine_segment_norm_lt`, `positionVelocityStep_segment_norm_lt`, and `positionVelocityStep_segment_norm_lt_of_acceleration_bound`: (12) and (13). |
| `MovingMetricMechanics` | `impulseAtPhase_energy_change` and `positionVelocityImpulse_energy_ledger`: actual impulse and ordered ledger (11). |
| `CovariantSteering` | `frameTurn_orthogonal`, `frameTurn_first`, `frameTurn_second`, and `frameTurn_mix`: pointwise skew action and frame-plane quarter-turn. |
| `CovariantSteering` | `hasDerivAt_metric_pairing_covariant`, `hasDerivAt_metric_self_steering`, `metric_self_steering_eq`, and `hasDerivAt_metric_self_steering_of_fields`: actual moving-metric differentiation and preservation. |

Targeted `lake build --wfail FractalSync.MovingMetricMechanics FractalSync.CovariantSteering`
checks these modules under the unchanged Lean/Mathlib pins. Package-wide build,
inventory, and transitive axiom verification are recorded in
[verification.md](verification.md).

Remaining implementation obligations include provider correspondence for the
actual fields and their derivatives, a usable certificate for the phase-energy
Hessian bound, floating-point error propagation, and control of repeated steps.
Continuous solution existence and a discrete realization of covariant steering
are also separate. No canonical parameter, integrator, runtime behavior, or
training preflight has been changed.
