# Local operation errors in the held-control RK ledger

This closes the numerical-perturbation increment started after the
[exact-real phase/work certificate](159-rk-work-ledger.md). It adds a conditional
bridge from errors inside a four-stage computation to complete phase and work
histories. It retains the same quadratic fixture, fixed command, field, energy,
and reference ODE. No new distance-to-set assumption is used.

The accepted theorem boundary is the local residual contract below. The
results do not establish that a compiled implementation satisfies that contract.

## Recorded computation and error contract

`RungeKutta.Perturbed.Stages` records the three constructed stage states, all
four full phase slopes, and the final mechanical endpoint. For source `x`, step
size `h`, and held field `f`, the exact formulas being compared are

\[
\begin{aligned}
k_0&=f(x),&y_2&=x+\tfrac h2k_0,\\
k_2&=f(y_2),&y_3&=x+\tfrac h2k_2,\\
k_3&=f(y_3),&y_4&=x+hk_3,\\
k_4&=f(y_4),&x^+&=x+\tfrac h6(k_0+2k_2+2k_3+k_4).
\end{aligned}
\]

The recorded values need not satisfy these equalities. `Residuals` supplies
separate norm bounds for each RHS evaluation, each stage construction, and
the weighted endpoint. Its RHS errors compare against the field at the
**recorded** stage, rather than at an exact-arithmetic shadow stage.
The phase norm is `max(norm(deltaPosition), norm(deltaVelocity))`, with the
Euclidean norm on each plane; it is not the coordinatewise maximum on four scalars.

`Powers` records four signed power evaluations at those same mechanical
states. `PowerResiduals` compares each value with actual held-drive-plus-drag
power. A separate scalar bound compares the stored work increment with
`h*(p0+2*p2+2*p3+p4)/6`; this bound includes weighted work arithmetic and
accumulation into the stored ledger.

For the concrete result, write `m=N+1`, with `m>=16`, and retain

\[
T=\frac1{576},\qquad h=\frac{T}{m}.
\]

| Operation | Allowed absolute error |
| --- | ---: |
| Each full mechanical RHS evaluation | `h` in the phase maximum norm |
| Each intermediate stage construction | `h^2` |
| Weighted mechanical endpoint | `h^2` |
| Each signed power evaluation | `h` |
| Stored signed-work increment versus weighted recorded powers | `h^2` |

The source state is the stored outer state. The held covector is the exact
fixture drive constructed at that source, and stays fixed throughout the
four stages. The initial phase is the fixture's exact initial phase and
initial stored work is zero. In the finite check below, the RHS and power
comparisons include the effect of rounding in the constructed drive.

This increment uses **one RK substep per outer interval**. The preceding
exact-real results for every positive internal substep count remain unchanged.

## Containment follows from local errors

The generic stage proof proceeds in evaluation order. A containing ball,
a bound `M` for the actual field, and budgets `eta`, `delta`, `tau` give
recorded slope bounds `M+eta` and displacements

\[
\begin{aligned}
\|y_2-x\|,\|y_3-x\|&\le h(M+\eta)/2+\delta,\\
\|y_4-x\|&\le h(M+\eta)+\delta,\\
\|x^+-x\|&\le h(M+\eta)+\tau.
\end{aligned}
\]

Membership is established before applying the regional field bound at the
next stage. It is not assumed for the complete stage sequence.

For the fixture, the containing ball has radius `18*h` around the current
outer anchor. The existing held-field estimate closes its speed bound at
`M=16`, since

\[
8+1080(18h)\le\frac{647}{64}<16.
\]

The half-radius anchor region leaves room for this ball inside the original
phase region. The error budgets then imply stage and endpoint membership.
The complete trajectory theorem subsequently derives anchor containment for
every outer step. The existing reported-coordinate allowance `1/64` therefore
applies at all recorded stage and endpoint queries.

This is mathematical halo/address validity. It does not certify machine
index conversion, masks, storage contents, or provider execution.

## Phase comparison

Regional field variation and the RHS residuals bound the recorded weighted
slope's deviation from the field at the source. Endpoint arithmetic then
produces an Euler-comparison estimate. Converting to the package's existing
velocity-then-position comparison step gives

\[
\|\widehat x^+-\operatorname{orderedStep}(x)\|\le18000h^2.
\]

The proof charges this entire bound to the generic stored-step error slot,
using the exact source acceleration for its other residual slot. It does not
assume a bound for the completed RK update as an input.

The existing reference has acceleration Lipschitz constant `400` and phase
speed `8`. The resulting interpolation coefficient is at most `108`, so one
reference solution controls every admitted observed history:

\[
\|\widehat z_{\rm aff}(t)-z(t)\|\le\frac3{16m},
\qquad0\le t\le T.
\]

The affine interpolant also stays in the inner phase region. The reference
is chosen before the mesh and before the observed history.

## Signed work and actual energy

The local work proof uses the actual energy differential and actual held
power identity `P(y)=DE(y)[f(y)]`. It combines the endpoint Taylor remainder,
variation of the energy differential at the recorded stages, RHS residuals,
power-evaluation residuals, and endpoint/work arithmetic residuals. Signed
values are retained throughout.

With energy differential norm `5` and Lipschitz constant `134`, a convenient
conservative bound is

\[
|E(\widehat x^+)-E(x)-\Delta\widehat W|\le65536h^2.
\]

Telescoping the actual stored increments yields, for every prefix `k<=m`,

\[
|E(\widehat z_k)-E(\widehat z_0)-\widehat W_k|
\le\frac1{4m}.
\]

The work coordinate does not feed back into the mechanical stages. This is
an energy/work residual bound, not a conservation claim about driven energy.
Here `E` is the exact mathematical energy evaluated at the recorded phase;
rounding in the runtime's separately reported energy value is not included.

The actual reference work is

\[
W(t)=\int_0^tP(z(s))\,ds=E(z(t))-E(z(0)).
\]

The prior FTC and reference-power proofs supply this identity and a
`40`-Lipschitz work trace. Phase error contributes `15/(16*m)` to the grid work
comparison. Affine interpolation contributes `5/(144*m)`. Together with the
finite ledger, this gives

\[
|\widehat W_{\rm aff}(t)-W(t)|
\le\frac1{4m}+\frac{15}{16m}+\frac5{144m}
=\frac{11}{9m}.
\]

## Finite tolerances and refinement

Every listed allowance is positive at a finite mesh. A family satisfying
these contracts as `m` increases has the stated phase and work error rates
against one reference solution. This is a conditional refinement result.
Fixed-precision arithmetic is not proved to supply shrinking `h` and `h^2`
errors. A fixed absolute error accumulated over progressively more steps
cannot simply be substituted into this convergence statement.

The bounds concern affine interpolation of stored values. They do not assert
RK dense-output accuracy, instantaneous-power convergence, or fourth-order
convergence under held controls.
The interpolation grid uses the ideal times `k*h`. Rounding in an accumulated
machine clock is a separate implementation obligation.

## Proof entry points and remaining boundary

- [Generic observed stages and containment](../../formal/FractalSync/PerturbedRungeKutta.lean).
- [Generic actual-power and stored-work defects](../../formal/FractalSync/PerturbedRungeKuttaWork.lean).
- [Concrete stage, phase, and query histories](../../formal/FractalSync/ConcretePerturbedRungeKutta.lean).
- [Concrete signed ledger and reference-work comparison](../../formal/FractalSync/ConcretePerturbedRungeKuttaWork.lean).

The [verification record](verification.md) and [module inventory](../../formal/README.md)
give the checked package boundary. This completes the conditional local-error
bridge for the stated one-substep profile.

## Measured binary64 residuals for the literal fixture

The release regression exports the raw binary64 words of the command at each
outer anchor, all four states and slopes of each successful RK substep, its
weighted endpoint and work, and the accumulated outer work. It ties the
recorded states to the attempted provider queries and accepted mechanical
endpoints. Both profiles use 16 outer steps on the horizon `1/576`.
[Rust fixture regression](../../runtime-core/src/operational_engine/fixture_queries.rs)

The independent checker decodes the recorded words as exact dyadics and
differentiates the six spline polynomials as rational monomials using the
literal fixture coefficients. It encloses the ruler jets, normalized drive,
acceleration, and power on a grid of `2^-100` with integer arithmetic. It
rounds products and reciprocals outward, bounds square roots with integer
square roots, and bounds the sine and cosine of the wall angle with alternating
Taylor sums on `[0, 1]`. The π interval comes from the alternating series for
`16 atan(1/5) − 4 atan(1/239)`. The tangent of `4 atan(1/5)` is `120/119`,
and subtracting `atan(1/239)` gives an angle in `(0, π/2)` with tangent one.
[Exact spline evaluator](../../formal/fixtures/check_operational_patch.py)
[Recorded-trace checker](../../formal/fixtures/check_181_rounded_rk.py)

At the initial position `(1/2, 1/2)`, exact division by the stored spacing
gives cell `(4, 4)`. The corresponding binary64 division rounds to `5` on
each axis and the machine reads cell `(5, 5)`. The residual bounds include
this difference because the checker selects the cell using the exact quotient.
[Literal spacing](../../formal/fixtures/operational-quadratic.json)

For RHS and power residuals, the checker uses the exact drive from the
recorded outer anchor, held across all of that step's stages. This includes
command construction error in the measured residual. Stage construction,
weighted endpoints, and stored work increments are compared by exact rational
arithmetic using the recorded slopes and the ideal rational step. The sum of
the two coordinate error bounds in each plane bounds its Euclidean error.
[Exact held dynamics](../../formal/FractalSync/ConcreteHeldControl.lean)
[Exact held power](../../formal/FractalSync/ConcreteHeldPower.lean)

These are upward-rounded bounds from the Windows x86-64 release run with Rust 1.98.0.

| Residual | One substep, 64 stages | Four substeps, 256 stages |
| --- | ---: | ---: |
| Held drive | `3.08e-16` | `3.02e-16` |
| Mechanical RHS | `7.58e-16` | `1.060e-15` |
| Signed power | `9.3e-17` | `1.15e-16` |
| Intermediate phase construction | `1.05e-16` | `1.11e-16` |
| Weighted mechanical endpoint | `9.5e-17` | `1.01e-16` |
| Weighted substep work | `1e-18` | `1e-18` |
| Stored outer work increment | `1e-18` | `1e-18` |

The checker verifies the `h` and `h²` budgets at each substep with
`h = 1/(576*16*substep_count)`. It uses the outer step squared for the
stored outer work increment. For the four-substep profile, write \(H=4h\).
The recorded source phases, stages and endpoints stay in a ball of radius
\(12H\) around the outer anchor. The held field has speed at most 10 on this
ball. Comparing all recorded slopes with the field at the outer anchor and
summing the endpoint residuals gives an ordered-update defect at most
\(18000H^2\). The four energy defects telescope through the actual internal
endpoints. The checker bounds the stored work increment against the sum of
the four exact weighted power samples, giving an outer energy/work defect at
most \(65536H^2\).

Both sixteen-step runs supply the numerical premises of their respective
phase/work certificates. The exact-time affine phase and stored-work traces
through the outer endpoints have bounds `3/256` and `11/144` against the
reference phase and integrated work on the full horizon.
[One-substep phase and work theorem](../../formal/FractalSync/ConcretePerturbedRungeKuttaWork.lean)
[Four-substep phase and work theorem](../../formal/FractalSync/ConcreteFourSubstepRungeKutta.lean)

From the repository root with the project virtual environment available:

```powershell
$env:FRACTALSYNC_RK_FIXTURE_TRACE = Join-Path $env:TEMP 'fractalsync-181-rk-bits.json'
rtk cargo test --release -p runtime_core --features experimental-operational-field --lib literal_quadratic_rk_queries_require_exactly_the_predicted_stencil
.venv/Scripts/python.exe formal/fixtures/check_181_rounded_rk.py $env:FRACTALSYNC_RK_FIXTURE_TRACE
```

This check applies to the finite trace exported by the release regression.
A uniform arithmetic bound across targets, profiles, or meshes, the accuracy
of the reported total energy and machine clock, and production field fidelity
remain open.
