# #159: exact frozen-metric velocity steps and explicit drag

[`DiscreteMechanics.lean`](../../formal/FractalSync/DiscreteMechanics.lean)
checks the energy change of an explicit velocity update while the metric is
held fixed. It proves an exact force-work correction and the sharp
nonincrease condition for explicit metric-proportional drag. These are
discrete algebraic contracts relevant to
[#159](https://github.com/rhythmatician/FractalSync/issues/159) and the
[current mechanics equations](../equations.md); the runtime's position
update and changing metric require additional estimates.

The module contains eleven proved theorems and three definitions. It reuses
[`MetricImpulse.lean`](../../formal/FractalSync/MetricImpulse.lean), including
the existing metric inverse and positive kinetic-energy form. The
[potential and control note](159-potential-controls.md) and
[continuous differential note](159-differential-mechanics.md) describe the
separate continuous work-energy contracts. No runtime behavior or canonical
parameter is changed by this formalization.

## 1. A fixed configuration and an explicit velocity update

Let `E` be a real inner-product space. Fix a positive ruler `ρ`, a real
parameter `a`, and a vector `u`, and use the shared metric

\[
G=\rho^{-2}(I+a^2u\otimes u),\qquad
g(v,w)=\langle v,Gw\rangle,\qquad
K(v)=\frac12g(v,v).
\]

The proofs do not require finite dimension or a unit-gradient assumption.
When these coefficients come from a smooth physical field, `u=∇ρ` at the
configuration being frozen. The algebra itself accepts any `u`.

For any acceleration vector `A` and real step `Δt`, define

\[
v_+=v+\Delta t\,A.
\]

The checked identity `velocityEulerStep_energy_change` is

\[
\boxed{K(v_+)-K(v)
=\Delta t\,g(v,A)+\Delta t^2K(A).}
\tag{1}
\]

It follows by expanding the same symmetric bilinear form at both
velocities. No equation of motion or interpretation of `A` is needed.
Consequently it can describe the velocity part of a runtime update even
when `A` contains several force and connection terms. Both energies in
(1) must still use the same metric coefficients.

## 2. A frozen force covector and its exact work correction

Represent a force covector `Q` by an ambient vector through the inner
product, so its power is `⟨v,Q⟩`. Raise it exactly once by the inverse
metric and set

\[
v_+=v+\Delta t\,G^{-1}Q,\qquad
\|Q\|_{G^{-1}}^2=\langle Q,G^{-1}Q\rangle.
\]

The theorem `frozenForceStep_energy_change` proves

\[
\boxed{K(v_+)-K(v)
=\Delta t\,\langle v,Q\rangle
+\frac{\Delta t^2}{2}\|Q\|_{G^{-1}}^2.}
\tag{2}
\]

The final term is nonnegative, as separately recorded by
`frozenForceStep_work_defect_nonneg`. Initial-velocity power times the
step therefore underestimates this frozen-metric energy change. Negative
initial power alone does not establish energy decrease for an arbitrary
finite step.

The equivalent checked identity
`frozenForceStep_average_velocity_work` is

\[
\boxed{K(v_+)-K(v)
=\Delta t\left\langle\frac{v+v_+}{2},Q\right\rangle.}
\tag{3}
\]

Here `Q` is held fixed during the velocity increment. It may have been
computed from the old velocity, as in explicit drag. Equation (3) is an
algebraic identity for that update, not a claim that the runtime uses an
implicit midpoint integrator or exactly integrates a varying force.
The inverse-dependent Lean statements only require `ρ≠0`; the physical
scope here uses `ρ>0`.

## 3. The sharp explicit-drag criterion

For the canonical drag covector

\[
Q_{\rm drag}=-\beta Gv,
\]

the checked inverse gives `G⁻¹Q_drag=−βv`. Thus the frozen force step is
exactly the explicit drag update

\[
\boxed{v_+=(1-\beta\Delta t)v,\qquad
K(v_+)=(1-\beta\Delta t)^2K(v).}
\tag{4}
\]

These are `frozenForceStep_metric_drag`, `explicitDragStep_eq_smul`, and
`explicitDragStep_kinetic`. Positive definiteness makes `K(v)>0` when
`v≠0`, giving the checked necessary-and-sufficient condition

\[
\boxed{v\ne0\quad\Longrightarrow\quad
\bigl(K(v_+)\le K(v)\iff0\le\beta\Delta t\le2\bigr).}
\tag{5}
\]

The complete theorem `explicitDragStep_kinetic_le_iff` includes the
stationary case: nonincrease holds exactly when `v=0` or the product
`βΔt` lies in `[0,2]`. No separate sign assumptions on `β` and `Δt` are
needed for this algebraic classification. Physical damping uses `β≥0`
and a nonnegative step.

The checked factor also resolves the boundary cases: product zero leaves
the velocity unchanged, product one stops it, and product two reverses
the velocity while preserving its kinetic energy. For nonzero velocity,
products strictly between zero and two strictly decrease energy; products
outside the interval increase it. The latter consequences are immediate
from (4); the named Lean statements package the nonincrease classification.
This condition applies to the isolated drag update at a fixed metric.
Additional forces and the subsequent change of position contribute other
energy terms.

## 4. Correspondence to the executable runtime order

The implementation in
[`manifold.rs`](../../runtime-core/src/manifold.rs), function
`integrate_step`, uses this order:

1. Check the old position lies in the open disk, then evaluate the
   connection and force quantities at that position.
2. Sum the Shore-potential, wall, control, and drag covectors; apply
   `G⁻¹` once; subtract the connection term to obtain the coordinate
   acceleration `A`.
3. Update `v₁=v₀+Δt A`, then `c₁=c₀+Δt v₁`.
4. Reject a nonfinite or out-of-disk proposed position. For an accepted
   state, evaluate energy at `(c₁,v₁)` and compare it with energy at
   `(c₀,v₀)`.
5. The wrapper `integrate_motion_controls` in
   [`controls.rs`](../../runtime-core/src/controls.rs) then computes its
   bounded impulse at `c₁`, adds `G(c₁)⁻¹J` to `v₁`, and recomputes
   kinetic energy there.

The last step follows the executable call `impulse_delta_v(c_new, config)`.
The nearby pseudocode uses `c` for that call, so it should not be used to
infer that the impulse is evaluated before the position update. The
existing `impulse_energy_jump` theorem describes the exact fixed-position
energy jump at `c₁` if its metric and impulse hypotheses are instantiated.
The force step in (2) alone does not include the connection term or this
later impulse.

For the continuous-force portion, a useful exact bookkeeping decomposition
over real arithmetic is

\[
\begin{aligned}
E(c_1,v_1)-E(c_0,v_0)
={}&[K_{c_0}(v_1)-K_{c_0}(v_0)]\\
&+[K_{c_1}(v_1)-K_{c_0}(v_1)]\\
&+[U(c_1)-U(c_0)].
\end{aligned}
\tag{6}
\]

The first bracket is covered by (1) for the chosen `A`. The second
measures the change of metric, and the third the change of potential.
Equation (6) is conventional algebraic bookkeeping here; it is not a
new checked convergence or energy-conservation theorem for the runtime.
The continuous connection cancellation proved elsewhere does not make
these finite-step remainders vanish.

## 5. Checked inventory and remaining correspondence work

All identifiers below are in namespace `FractalSync.ScaleMetric`.
The definitions are `velocityEulerStep`, `frozenForceStep`, and
`explicitDragStep`.

| Theorem | Checked result |
| --- | --- |
| `kinetic_smul` | `K(cv)=c²K(v)`. |
| `velocityEulerStep_energy_change` | Arbitrary frozen-metric velocity increment, (1). |
| `frozenForceStep_energy_change` | Force-work identity with its quadratic correction, (2). |
| `frozenForceStep_work_defect_nonneg` | Nonnegative defect relative to initial-velocity work. |
| `frozenForceStep_average_velocity_work` | Exact average-velocity work identity, (3). |
| `frozenForceStep_metric_drag` | Raising `−βGv` reduces the force step to explicit drag. |
| `explicitDragStep_eq_smul` | Exact velocity multiplier in (4). |
| `explicitDragStep_kinetic` | Exact energy factor in (4). |
| `explicitDragStep_kinetic_le` | Sufficiency of `0≤βΔt≤2`. |
| `explicitDragStep_kinetic_le_iff_of_ne_zero` | Necessity and sufficiency for `ρ>0`, `v≠0`. |
| `explicitDragStep_kinetic_le_iff` | Full classification including `v=0`. |

The targeted command `lake build --wfail FractalSync.DiscreteMechanics`
passed under the pinned Lean 4.24.0 and Mathlib environment. The module
contains no admitted proofs or custom axioms. Package-wide verification is
tracked in [verification.md](verification.md).

The remaining runtime obligations are concrete:

- The supplied field values, gradients, and Hessians need error bounds
  against one underlying differentiable field on the region traversed by
  the step. A positive ruler alone does not provide those derivatives at
  a cut locus. The bridge's finite differences and operational field's
  derivative self-consistency checks are not such correspondence proofs;
  see [the provider-error note](162-normalized-distance-fields.md).
- A varying-metric step needs estimates for the second and third brackets
  in (6), together with timestep and regularity hypotheses. The frozen
  drag interval cannot establish total-energy nonincrease for that full
  update, or global convergence of a rollout.
- Floating-point evaluation needs bounds for metric inversion, force
  summation, the updates, and energy reporting. The exact Lean identities
  currently concern real arithmetic. Rejecting an invalid proposed
  position is a runtime guard; this proof does not show that every step
  will be accepted or that every intermediate continuous path stays in
  the domain.

Existing Rust tests provide numerical seams: `drive_covector_metric_consistent_norm`,
`drift_grip_controls_cannot_inject_energy`,
`conservative_rollout_energy_drift_is_bounded`, and
`taps_are_bounded_impulses_layered_on_momentum`. These tests have narrower
empirical scopes than the real-arithmetic theorems. A focused additional
correspondence test could freeze one metric, isolate a velocity step, and
compare (1) and (2) with the computed energy difference; pure-drag cases
at products `0,1,2` and just outside `[0,2]` would exercise the sharp
boundary. Such a test would complement the proof without establishing
variable-metric conservation or a floating-point error theorem.
