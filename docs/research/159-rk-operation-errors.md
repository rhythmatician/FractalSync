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
four stages. The initial phase is the fixture's exact initial phase; initial
stored work is zero. Errors in initial-state construction or control
construction require separate evidence and are not newly certified here.

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
bridge for the stated one-substep profile. Production field fidelity,
compiled provider correspondence, actual IEEE residual certificates, rounded
control construction, perturbed internal-substep composition, changing
commands, impulses, steering, and continuation beyond this local interval
remain separate implementation or research work.
