# #159: residual-aware accumulated dynamics and discrete covariant steering

The exact-real package now connects the changing-metric local energy estimate
to an actual finite sequence of velocity, position, and impulse updates. It
also derives uniform phase-energy regularity on compact convex regions from
actual smooth configuration fields, and defines a concrete discrete steering
step with both exact endpoint metric norm and a checked first derivative equal
to the covariant steering rate.

The three new source modules are
[AccumulatedDynamics.lean](../../formal/FractalSync/AccumulatedDynamics.lean),
[PhaseEnergyRegularity.lean](../../formal/FractalSync/PhaseEnergyRegularity.lean),
and [DiscreteSteering.lean](../../formal/FractalSync/DiscreteSteering.lean).
They extend the [local changing-metric results](159-moving-metric-dynamics.md).
They introduce no runtime, canonical-parameter, provider, or training change.

## 1. The actual acceleration residual determines the power error

Write the existing metric, weight, and phase energy as

\[
g_x(v,w)=\frac{\langle v,w\rangle+
 a^2\langle u,v\rangle\langle u,w\rangle}{\rho(x)^2},
\qquad W=1+a^2\|u\|^2,
\qquad \mathcal E(x,v)=\tfrac12g_x(v,v)+U(x).
\]

At the initial configuration, the hypotheses identify actual derivatives

\[
D\rho_x=\langle u,\cdot\rangle,
\qquad Du_x=H,
\qquad DU_x=\langle\nabla U,\cdot\rangle,
\qquad \rho(x)\ne0.
\]

The new definition is the actual equation-of-motion residual

\[
R=A+\Gamma_x(v,v)-G_x^{-1}(Q-\nabla U).
\tag{1}
\]

There is no assumption that the supplied acceleration satisfies the ideal
equation. Differentiating the actual energy and using the checked metric
compatibility identity gives

\[
\boxed{D\mathcal E_{(x,v)}(v,A)
 =\langle v,Q\rangle+g_x(v,R).}
\tag{2}
\]

Thus an acceleration discrepancy contributes precisely the extra power
`g(v,R)`. The theorem is about the actual `fderiv` of the defined energy, not
an independently stipulated power law.

Suppose the acceleration is assembled as
`A = forceEstimate - connectionEstimate`, with estimates relative to the
actual quantities satisfying

\[
\|\mathrm{forceEstimate}-G_x^{-1}(Q-\nabla U)\|\le\eta_F,
\qquad
\|\Gamma_x(v,v)-\mathrm{connectionEstimate}\|\le\eta_\Gamma.
\]

Lean proves the residual decomposition and its consequences

\[
\|R\|\le\eta_F+\eta_\Gamma,
\qquad
\boxed{|g_x(v,R)|
 \le\frac{W}{\rho(x)^2}\|v\|(\eta_F+\eta_\Gamma).}
\tag{3}
\]

The second inequality uses a separately checked operator bound for the actual
bilinear metric. The force estimate here is an already raised acceleration
vector. Numerical errors in potential derivatives, inverse-metric evaluation,
or provider queries are included in `etaF` only after an independent fidelity
argument relates those computations to the exact term in (1). The theorem
does not infer such fidelity from a provider flag or a positive epsilon.

## 2. The ordered local step with an explicit residual

For a signed real step `h`, use the existing ordered definitions

\[
\widehat v=v+hA,\qquad x^+=x+h\widehat v,
\qquad \widehat z=(x^+,\widehat v).
\tag{4}
\]

This is the velocity-then-position algebraic order of `integrate_step` in
[manifold.rs](../../runtime-core/src/manifold.rs). Its phase displacement is

\[
\widehat z-(x,v)=h(\widehat v,A)=h(v,A)+h^2(A,0).
\]

Let

\[
B_x^v(d)=\tfrac12 Dg_x[d](v,v)+\langle\nabla U,d\rangle.
\]

Assume actual differentiability of `Ecal` throughout the closed proposed
phase segment, and actual differential variation at most `L` times phase
displacement from its initial point. The phase-space norm is the usual
product max norm. The new local theorem proves

\[
\boxed{
\begin{aligned}
|\mathcal E(\widehat z)-\mathcal E(x,v)-h\langle v,Q\rangle|
\le{}& |h|\,|g_x(v,R)|\\
&+h^2\left[
 |B_x^v(A)|+\frac L2\max(\|\widehat v\|,\|A\|)^2
\right].
\end{aligned}}
\tag{5}
\]

Both the first-order residual term and the position-only correction are
derived by differentiating the actual ordered increment. The analytic
remainder is the checked quadratic Taylor bound. Zero steps are included.

The previously checked explicit first-derivative bound remains available:

\[
|B_x^v(A)|\le
\left[
\frac{\|u\|}{\rho}g_x(v,v)
+\frac{a^2}{\rho^2}\|H\|\|u\|\|v\|^2
+\|\nabla U\|
\right]\|A\|,
\qquad \rho>0.
\tag{6}
\]

`StepJetCertificate` packages actual initial field derivatives, initial
nonzero ruler, and the whole-segment differentiability and variation data.
Its `errorBudget` is the explicit right side of (5). The record does not take
an energy-error estimate as a field: `StepJetCertificate.energy_defect`
proves that estimate from the calculus data.

## 3. Actual field smoothness supplies uniform regularity on compact regions

`contDiffAt_phaseEnergy` proves the actual arithmetic energy formula is
`C^n` at `(x,v)` whenever `rho`, `u`, and `U` are `C^n` at `x` and
`rho(x) != 0`. This holds for all regularity orders accepted by `ContDiffAt`,
including the finite order needed below.

For a compact phase region `K`, suppose those three fields are actually
`C^2` at every configuration projected from `K`, and the ruler is nonzero
there. The package proves that some finite nonnegative `L` bounds the norm
of the actual phase-energy Hessian at every point of `K`:

\[
\exists L\ge0\quad
\forall z\in K,\qquad
\mathcal E\text{ and }D\mathcal E\text{ are differentiable at }z,
\qquad \|D^2\mathcal E_z\|\le L.
\tag{7}
\]

Here the displayed derivative is the iterated `fderiv`, and its
`HasFDerivAt` property is proved. Continuity of that actual Hessian and
compactness supply the bound. Compactness is an explicit hypothesis; no
finite-dimensional assumption is required merely to state this result.

If `K` is also convex, the same `L` makes `D Ecal` Lipschitz on `K`.
`StepJetCertificate.ofLipschitzRegion` then constructs the local certificate
for any ordered step whose initial phase point and pre-impulse endpoint both
belong to `K`, provided its actual initial field jets are supplied. Convexity
puts every point of its phase segment in `K`. Thus the regularity constant
can be shared across all such steps, rather than postulated separately as an
energy-error bound.

This is an existence theorem for a finite uniform constant, not a numerical
certificate for its magnitude. In the physical specialization `u = grad rho`,
the `C^2` assumption on `u` demands higher ruler regularity than existence of
the ruler Hessian at one point. A positive ruler, positive epsilon, or a
bounded numerical Hessian does not supply that assumption. Membership in a
compact phase region also includes velocity control; the position-domain
condition below alone does not establish it.

## 4. Finite-horizon energy and work accounting

Let `z_i=(x_i,v_i)` be an actual finite history. For every index `i<N`, the
formal recurrence requires

\[
\begin{aligned}
\widehat v_i&=v_i+h_iA_i, &x_{i+1}&=x_i+h_i\widehat v_i,\\
v_{i+1}&=\widehat v_i+G_{x_{i+1}}^{-1}J_i.
\end{aligned}
\tag{8}
\]

The impulse is evaluated at the updated position, matching the algebraic
ordering in `integrate_motion_controls` in
[controls.rs](../../runtime-core/src/controls.rs). Its exact work contribution
is

\[
I_i=\langle\widehat v_i,J_i\rangle
 +\tfrac12\langle J_i,G_{x_{i+1}}^{-1}J_i\rangle.
\tag{9}
\]

With one actual `StepJetCertificate` per step and a nonzero ruler at each
impulse position, `positionVelocityImpulse_energy_defect_sum` proves

\[
\boxed{
\left|\mathcal E(z_N)-\mathcal E(z_0)
 -\sum_{i<N}\bigl[h_i\langle v_i,Q_i\rangle+I_i\bigr]\right|
\le\sum_{i<N}\mathrm{errorBudget}_i.}
\tag{10}
\]

The proof uses the actual recurrence, the exact updated-position impulse
ledger, and telescoping of the signed local energy defects. It neither
assumes the final energy bound nor introduces an exact ODE solution.
Certificates are required only for the finite set of steps. The `N=0` case
is included.

The separate sum theorem gives a useful uniform form. If `0 <= h_i <= hmax`,
every residual-power term is at most `eta`, and every bracketed coefficient
in (5) is at most a nonnegative `C`, then applying (10), termwise comparison,
and `sum_step_budget_le` gives

\[
\boxed{
|\text{accumulated energy minus accounted work}|
 \le T(\eta+h_{\max}C),
\qquad T=\sum_{i<N}h_i.}
\tag{11}
\]

The source separately checks the actual recurrence bound (10) and the
algebraic sum estimate for `sum (h_i*eta + h_i^2*C)` used in (11). It does not
introduce another wrapper theorem asserting all coefficient bounds on a
given trajectory. Those bounds must still be established for that trajectory
or its certified region.

For a fixed horizon and bounded `C`, reducing the maximum step reduces the
Taylor contribution. A fixed residual-power budget leaves the `T*eta` term.
These are energy/work estimates. They do not bound state error relative to an
exact ODE, prove convergence of a rollout, or give exact conservation. The
impulse ledger is exact-real; numerical impulse error and rounded stored
states need additional correspondence estimates.

## 5. One displacement budget controls all proposed position segments

For arbitrary finite nodal positions, Lean proves

\[
\|x_n\|\le\|x_0\|+
 \sum_{i<n}\|x_{i+1}-x_i\|.
\]

For the actual position recurrence in (8), the sufficient global condition

\[
\boxed{\|x_0\|+
 \sum_{i<N}|h_i|\,\|\widehat v_i\|<R}
\tag{12}
\]

therefore keeps every point of every proposed position segment
`x_i + t*h_i*vhat_i`, `0 <= t <= 1`, `i<N`, strictly inside the radius-`R`
ball. The theorem includes both endpoints. Taking `R=2` supplies a finite
history condition for the canonical secant-wall disk. Impulses after the
position update leave its endpoint unchanged, and their effects on subsequent
step velocities are already present in (12).

This is a sufficient budget on the actual finite proposed history. It can be
checked against a cumulative displacement allowance as steps are proposed.
It does not enclose an exact ODE path, prove an unconditional invariant for
unbounded time, or discharge floating-point finite-value and wall-force
conditioning guards. It also does not supply the velocity bound needed for a
compact phase region in Section 3.

## 6. A concrete discrete covariant steering update

The earlier continuous steering calculation uses

\[
J_{e,f}d=g(e,d)f-g(f,d)e,
\qquad
\dot d+\Gamma(b,d)=\omega J_{e,f}d.
\tag{13}
\]

`DiscreteSteering` now defines the coordinate rate and candidate step

\[
\begin{aligned}
k&=\omega J_{e,f}d-\Gamma_{x_0}(b,d),\\
d_{\rm trial}&=d+h k,\\
\boxed{d_+=
 \frac{d_{\rm trial}}{
  \sqrt{g_{x_1}(d_{\rm trial},d_{\rm trial})}}.}
\end{aligned}
\tag{14}
\]

The rate is frozen at the initial configuration, and normalization uses the
actual endpoint metric. This is a fully specified exact-real direction
update, separate from the current runtime controls.

If the endpoint ruler is positive and

\[
|h|\,\|k\|<\|d\|,
\tag{15}
\]

the trial cannot vanish. Positive definiteness then makes normalization
well-defined, and Lean proves

\[
g_{x_1}(d_+,d_+)=1.
\tag{16}
\]

For an initially unit heading, a zero step returns that same heading.

Exact endpoint norm alone would not establish consistency with the
connection. The additional theorem differentiates (14) as the step parameter
tends to zero along a genuine differentiable base curve. It assumes the
actual field derivatives for `rho` and `u`, and that the initial base velocity
is `b`. The already checked metric compatibility calculation gives

\[
\left.\frac d{dh}
 g_{\gamma(h)}(d+h k,d+h k)\right|_{h=0}=0
\]

when the initial heading is metric unit. Differentiating the actual square
root, reciprocal, and scalar product in (14) shows that the normalization
factor has derivative zero. Therefore

\[
\boxed{\left.\frac d{dh}d_+(h)\right|_{h=0}
 =\omega J_{e,f}d-\Gamma_{x_0}(b,d).}
\tag{17}
\]

Both trace-level and actual-field chain-rule versions are proved. Equation
(17) supplies first-order consistency with the covariant steering ODE, in
addition to exact-real norm preservation. The argument is the normalized
Euler construction applied to the existing metric-compatible rate; no
novelty claim is made for that numerical idea.

The skew property needed for norm preservation holds for arbitrary `e,f`.
Interpreting `J` as a quarter-turn requires an orthonormal pair and applies
on their frame plane, as in the earlier module. The new theorem does not
construct a frame, solve the steering ODE, prove a global numerical
convergence order, assert exact parallel transport, or certify IEEE
normalization. In particular it is stronger than an arbitrary endpoint
metric isometry, but still a discrete approximation to the continuous law.

## 7. Declaration inventory and verification

All new declarations are in `FractalSync.ScaleMetric`.

| Source module | Explicit theorem declarations | Main results |
| --- | ---: | --- |
| `AccumulatedDynamics` | 14 | Actual residual power and local defect; norm-to-power conversion; force/connection error decomposition; actual finite ordered recurrence with impulses; accumulated budget sum; complete proposed-position-segment domain control. |
| `PhaseEnergyRegularity` | 3 | Actual field-to-energy smoothness; uniform actual Hessian bound on compact phase regions; Lipschitz energy differential on convex compact regions. |
| `DiscreteSteering` | 8 | Actual normalized update derivative; endpoint metric unit norm; nonvanishing step condition; zero-step identity; field-level consistency. |

The definitions include the residual, local jet record and error budget,
the constructor `StepJetCertificate.ofLipschitzRegion`, and the normalized
steering algorithm. Supporting theorem declarations are included in the
counts; definitions and generated record helpers are not.

The targeted verification command is

```text
lake build --wfail FractalSync.AccumulatedDynamics \
  FractalSync.PhaseEnergyRegularity FractalSync.DiscreteSteering
```

It uses the existing Lean and Mathlib pins. The package-wide build,
declaration inventory, and transitive axiom audit are recorded in
[verification.md](verification.md).

## 8. Remaining boundaries

The finite accumulated energy/work theorem, sufficient finite-history
position-domain condition, and a concrete consistent discrete steering
construction are checked here. The [trajectory continuation](159-trajectory-error.md)
now derives actual ODE truncation and stability, simultaneous error and
position/velocity domain control from reference margins, and uniform grid-point
convergence under the stated refinement scaling. The following remain distinct obligations:

| Area | Remaining work |
| --- | --- |
| Quantitative regularity | Concrete upper bounds for phase-energy Hessians, acceleration derivatives and reference speeds from an actual field/provider, with usable phase-region margins. |
| Provider and arithmetic correspondence | Fidelity of actual force and connection evaluations, query rounding, IEEE arithmetic, rounded stored-state errors, and numerical impulse errors. |
| Rollout control | Reference-solution existence and numerical margin certificates for specific fields; continuous-interpolant, time-dependent-control and impulsive-history convergence. Conditional domain control and uniform grid-state convergence for autonomous continuous force steps are now checked. |
| Discrete steering | Frame construction and transport, solution existence and quantitative convergence, and a runtime realization with floating-point norm/error guarantees. |

No minimum-curve convergence, global cut-locus theorem, integrator energy
conservation, or provider derivative certificate follows from this increment.
