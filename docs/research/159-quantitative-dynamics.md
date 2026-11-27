# #159: explicit local horizons and numerical trajectory control

The [local-trajectory package](159-local-trajectories.md) constructed an actual
reference solution from C1 acceleration, with an existential horizon and phase
ball. [QuantitativeDynamics.lean](../../formal/FractalSync/QuantitativeDynamics.lean)
now starts with a specified phase ball and explicit regional bounds. It
constructs an actual solution on a stated positive time interval, supplies a
fixed domain tube, proves regional uniqueness, and applies the continuous
interpolation estimate with a step-count condition in the supplied constants.

The module contains five theorem declarations. It uses the existing pinned
Mathlib Picard–Lindelöf implementation; no existence of a reference solution,
unknown local truncation bound, or unquantified neighborhood is an input.

## 1. Inputs and explicit horizon

Let the configuration space be a complete real inner-product space. Write

\[
z=(x,v),\qquad \|z\|=\max(\|x\|,\|v\|),\qquad
F(z)=(v,a(z)).
\]

Fix an initial state \(p\), radius \(R>0\), and constants \(L,B\ge0\).
The actual field hypotheses are

\[
\|a(z)-a(w)\|\le L\|z-w\|
\quad(z,w\in\overline B(p,R)),
\tag{1}
\]

\[
\|F(z)\|\le B
\quad(z\in\overline B(p,R)).
\tag{2}
\]

No differentiability premise is needed in addition to the actual Lipschitz
condition (1). The first-order phase field has Lipschitz constant
\(\max(1,L)\), by the previously checked phase-space estimate.

The definition `quantitativeHorizon` fixes

\[
\boxed{T=\frac{R}{4(B+1)}.}
\tag{3}
\]

`quantitativeHorizon_properties` proves

\[
T>0,\qquad 2BT\le R,\qquad BT<R/4.
\tag{4}
\]

This is a conservative horizon, not a maximal-existence claim. The denominator
avoids division by a possibly zero speed bound. The Lipschitz constant affects
the numerical error amplification and uniqueness argument; the bound on field
values supplies the ball-containment condition for existence.

If only an acceleration norm bound \(\|a(z)\|\le A\) is supplied on the ball,
`phaseField_norm_le_phaseBallSpeed` derives (2) using the defined constant

\[
\boxed{B=\max(\|p_v\|+R,A).}
\tag{5}
\]

Indeed, \(\|v-p_v\|\le R\) throughout the phase ball. This accounts for the
entire permitted velocity range; the bound is not evaluated only at the initial
velocity.

## 2. The actual solution and its certified tube

`exists_quantitative_integralCurve` is a generic real Banach-space result.
It constructs the actual Picard fixed point on \([-2T,2T]\), using the given
norm and Lipschitz bounds on the closed ball. The chosen extension to the real
line is globally \(B\)-Lipschitz, and the actual ODE holds on the open interval
\((-2T,2T)\). The extension outside that interval is not claimed to solve the
ODE.

The proof retains the bounded, Lipschitz function space from the pinned
[Picard–Lindelöf source](https://github.com/leanprover-community/mathlib4/blob/f897ebcf72cd16f89ab4577d0c826cd14afaafc7/Mathlib/Analysis/ODE/PicardLindelof.lean).
Its actual fixed-point identity gives the integral equation, and the
fundamental theorem of calculus gives the derivative. Working on a larger
interval ensures that both endpoints of \([0,T]\) have ordinary derivatives,
as required by the existing numerical certificate.

`exists_quantitative_ODE_certificate` applies this construction to the actual
phase field. It produces \(z(0)=p\) and a `UniformODECertificate` on
\([0,T]\) and \(B(p,R)\), with acceleration constant exactly \(L\) and
phase-speed constant exactly \(B\). It also proves, for every \(t\in[0,T]\),

\[
\|z(t)-p\|\le Bt\le BT<R/4,
\tag{6}
\]

\[
\boxed{\|w-z(t)\|\le R/2\quad\Longrightarrow\quad w\in B(p,R).}
\tag{7}
\]

The tube is therefore a consequence of the explicit field bounds and time
horizon. It is not an independent margin premise.

The existing certificate uniqueness theorem identifies this solution with
any other actual solution on \([0,T]\) that starts at \(p\) and remains in
the same open phase ball. This regional uniqueness conclusion is included
explicitly in the combined trajectory-control theorem below. It makes no
claim about continuation outside that ball or global existence.

## 3. A step-count condition for the complete interpolated path

Let \(\eta_0,r_0,I_0\ge0\) be the supplied acceleration-residual,
stored-state-rounding, and initial-error coefficients. Retain the exact-real
velocity-then-position update and set

\[
h=\frac{T}{N+1},\qquad m=\max(1,L),
\]

\[
\mathcal C=
e^{T(m+TL)}
\left[I_0+T\left((1+T)\eta_0+r_0+B+\frac{mB}{2}\right)\right]
+\frac B2.
\tag{8}
\]

The final \(B/2\) term controls interpolation between grid points. The earlier
part is the checked recurrence bound for the actual ordered update.

For a stored history \(y_i\) and evaluated acceleration \(A_i\), assume

\[
\|y_0-p\|\le hI_0,
\qquad \|A_i-a(y_i)\|\le h\eta_0,
\]

\[
\|y_{i+1}-\operatorname{positionVelocityStep}(h,y_i,A_i)\|
\le h^2r_0.
\tag{9}
\]

The latter two contracts need hold only when \(y_i\in B(p,R)\). The theorem
does not assume membership of the complete history.

`exists_quantitative_trajectory_control` constructs one actual reference
solution, independent of the numerical history and mesh. For every history
satisfying (9), the explicit step-count condition

\[
\boxed{2T\mathcal C\le R(N+1)}
\tag{10}
\]

implies that its actual piecewise-affine interpolant \(Y_h\) is continuous and

\[
\boxed{
\|Y_h(t)-z(t)\|\le h\mathcal C,
\qquad Y_h(t)\in B(p,R)
\quad(0\le t\le T).
}
\tag{11}
\]

Condition (10) is precisely the checked sufficient inequality
\(h\mathcal C\le R/2\), written without dividing by the step count. The
previous simultaneous error/domain induction activates the conditional
provider contracts, and (7) then controls every interpolated state. Regional
uniqueness is included for the constructed reference.

The returned quantitative certificate also supplies the assumptions of the
existing `UniformODECertificate.interpolation_converges` theorem. That theorem
gives all-time uniform convergence for families obeying the same shrinking
budgets. The finite estimate (11) retains the explicit dependence on the
regional constants.

## 4. Scope and verification

| Declaration | Checked scope |
| --- | --- |
| `quantitativeHorizon_properties` | Positive horizon and the two inequalities needed for existence and the interior tube |
| `exists_quantitative_integralCurve` | Actual generic Picard solution, ordinary derivatives on a stated open interval, and a globally Lipschitz chosen extension |
| `phaseField_norm_le_phaseBallSpeed` | Acceleration norm bounds converted into a phase-field bound over the supplied ball |
| `exists_quantitative_ODE_certificate` | Actual phase solution, exact supplied certificate constants, displacement estimate, and fixed positive tube |
| `exists_quantitative_trajectory_control` | One actual reference, regional uniqueness, explicit step-count condition, continuity, all-time error, and numerical-domain control |

`lake build --wfail FractalSync.QuantitativeDynamics` passed with 2,547 jobs
and no warnings or errors. The complete package build, final module inventory,
and transitive axiom audit are recorded in [verification.md](verification.md).
No dependency pin, runtime behavior, canonical parameter, training path, or
provider contract is changed.

Concrete mechanical or operational-field bounds must establish (1) and (2)
for the selected ball. Finite-asset halo validity is a separate geometric
domain condition. The results are over exact reals and do not certify query
rounding, IEEE arithmetic, or stored-state error budgets for fixed machine
precision. Time-dependent controls, impulse jumps, event detection, and a
combined transported-steering evolution remain separate. The horizon is local;
no maximal interval, global continuation, or long-time energy conservation is
claimed.
