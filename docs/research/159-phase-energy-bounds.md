# Issue 159: explicit regional phase-energy regularity

## What is checked

`formal/FractalSync/PhaseEnergyBounds.lean` supplies an explicit operator-norm
and Lipschitz bound for the actual differential of

\[
\mathcal E(x,v)=\frac12 g_x(v,v)+U(x).
\]

The earlier `PhaseEnergyRegularity.lean` compactness theorem established
existence of some finite phase-Hessian bound. The new theorem computes a
regularity constant from norm/Lipschitz certificates for the actual ruler,
ruler gradient, ruler Hessian, potential gradient, and velocity. It does not
assume a phase-energy remainder estimate or an unspecified Hessian bound.

Let \(u=\nabla\rho\), let \(H=Du\) be the actual gradient differential, and
write \(g_U=\nabla U\). These identifications remain explicit `HasFDerivAt`
premises at every position in the phase region. No symmetry assumption on a
separately supplied matrix is used to manufacture a derivative identity.

## Bundled differential

Define the position covector

\[
P=
\frac{a^2\langle u,v\rangle}{\rho^2}
  \bigl(\langle v,\cdot\rangle\circ H\bigr)
-\frac{\|v\|^2+a^2\langle u,v\rangle^2}{\rho^3}
  \langle u,\cdot\rangle
+\langle g_U,\cdot\rangle.
\]

The complete continuous linear map on position-velocity increments is

\[
J(d,w)=P(d)+\langle Gv,w\rangle.
\]

`fderiv_phaseEnergy_eq` proves that this bundled map is the actual Fréchet
differential of the existing `phaseEnergy` definition, wherever \(\rho\ne0\)
and the actual field-jet identities hold. The proof uses the previously
checked directional phase-energy formula and verifies the algebraic equality.
The product norm is the existing max norm on \(E\times E\).

## Explicit budget arithmetic

For a field norm/Lipschitz pair \(B_f=(M_f,L_f)\), use

\[
B_f+B_g=(M_f+M_g,L_f+L_g),\qquad
B_fB_g=(M_fM_g,M_fL_g+M_gL_f).
\]

Above a ruler floor \(\rho\ge\epsilon>0\), write

\[
B_{\rho^{-1}}=(\epsilon^{-1},L_\rho/\epsilon^2),\qquad
A=(a^2,0),\qquad C=B_uB_v,
\]

\[
N=B_vB_v+(AC)C.
\]

The module's `positionCovectorBudget` is exactly

\[
B_P=(ACB_{\rho^{-1}}^2)(B_vB_H)
 +(NB_{\rho^{-1}}^3)B_u+B_{g_U}.
\]

Both additions account for norm estimates, including the subtraction in the
actual covector. Its `energyCovectorBudget` is

\[
\boxed{B_J=B_P+B_{Gv},}
\]

where `ControlForceBounds.metricActionBudget` supplies the checked budget for
the actual metric lowering \(Gv\). All operations in this displayed recursive
expression are finite real arithmetic on the supplied certificate constants.
There is no optimization or unspecified existence constant.

`innerSL_bound` uses the exact norm preservation of the ambient Riesz
covector. `covector_comp_bound` propagates the operator composition term, and
`phase_precomp_bound` proves that the two coordinate projections introduce no
additional factor. These facts establish `energyCovector_bound`.

`fderiv_phaseEnergy_bound` transfers that certificate to the actual `fderiv`
using the regional field-jet identities. The phase region need not be convex
for this pairwise certificate. Its first component bounds \(\|D\mathcal E\|\)
and its second component is the explicit differential Lipschitz constant.

## Step certificate

`stepJetCertificate` constructs the existing `StepJetCertificate` with

\[
L=(B_J)^{(2)}.
\]

The constructor requires a convex certified phase region containing the old
state and the proposed velocity-then-position state. Convexity ensures that
the complete affine step segment lies in the region. Actual differentiability
and differential variation are then derived from the field certificates and
jet identities. This connects directly to the existing checked energy-defect,
residual-power, and finite-history accumulation theorems.

The result supplies regional energy regularity for such steps. Domain inclusion,
force residuals, and field-jet fidelity remain explicit obligations for any
specific provider or trajectory. It does not prove global continuation,
IEEE arithmetic accuracy, or exact energy conservation of a numerical step.

## Verification

The module contains seven proved theorem declarations and the proved
`stepJetCertificate` constructor. It is checked by
`lake build --wfail FractalSync.PhaseEnergyBounds` under the repository's
existing Lean and Mathlib pins. Package-level build and axiom audit include
this module when imported by the aggregate proof package.

## Concrete finite-asset instantiation

`formal/FractalSync/ConcreteEnergyCertificate.lean` instantiates this result on
`ConcreteOperationalAsset` and the same phase ball as
`ConcreteTrajectoryCertificate.lean`. It uses the actual regularized spline
ruler, its actual gradient and gradient differential, and

\[
U(x)=U_{\rm Shore}(\kappa=\log2,d_{\rm ref}=1,\rho(x))
     +U_{\rm wall}(\mu=1/\pi,x),\qquad a=1.
\]

The potential-gradient identity is checked from the existing actual catalog
potential differential. The concrete phase-region certificates are

| Field | Norm bound | Lipschitz constant |
|---|---:|---:|
| Ruler \(\rho\) | 8 | 200 |
| Actual ruler gradient \(u\) | 200 | 50,000 |
| Actual ruler Hessian \(H\) | 50,000 | 30,000,000 |
| Velocity | 1 | 1 |
| Actual potential gradient \(\nabla U\) | 201 | 90,001 |

The ruler floor is one. The conservative-gradient pair follows from the
checked Shore-plus-wall estimate and preserves the nonzero wall term. No
unspecified asset, force, or energy-regularity constant is supplied to this
instantiation.

`energy_differential_bound` proves the concrete rational bounds

\[
\boxed{
\|D\mathcal E(z)\|\le20{,}000{,}000,\qquad
\|D\mathcal E(z)-D\mathcal E(w)\|
\le24{,}000{,}000{,}000\,\|z-w\|.
}
\]

These round upward from the recursive certificate's exact values
\(18{,}040{,}402\) and \(23{,}372{,}300{,}802\). The relatively large constants
are global arithmetic majorants propagated without cancellation; they are not
empirical estimates of the fixture's local numerical stiffness or timestep
recommendations.

For an actual catalog acceleration \(A(z)\), define

\[
v_+=v+hA(z),\quad z_+=(x+hv_+,v_+),\quad
\Delta z=h(v_+,A(z)).
\]

Whenever both \(z\) and \(z_+\) belong to the certified closed phase ball,
`energyStepCertificate` supplies the existing step certificate with literal
\(L=24{,}000{,}000{,}000\). `energy_step_taylor_remainder` gives

\[
\boxed{
|\mathcal E(z_+)-\mathcal E(z)-D\mathcal E(z)[\Delta z]|
\le
12{,}000{,}000{,}000\,h^2
\max(\|v_+\|,\|A(z)\|)^2.
}
\]

This is a checked full changing-configuration energy remainder. It is not an
energy-conservation statement: drive and drag can do work, and replacing the
exact-real field or update by a machine implementation requires its own error
accounting. The constructor can be used directly by the existing residual and
work-error accumulation theorems. For arbitrary proposed steps, endpoint
inclusion is retained explicitly; convexity then supplies the entire traversed
segment. The certified finite run below discharges this inclusion requirement.

The concrete module adds twelve proved theorem declarations and the
`energyStepCertificate` constructor. Its target is checked with
`lake build --wfail FractalSync.ConcreteEnergyCertificate`.


### Unconditional energy remainder along the certified finite run

The exact-real recurrence `exactTrajectory N` in
`ConcreteTrajectoryCertificate.lean` takes \(N+1\) velocity-then-position
steps of length \(h_N=T/(N+1)\). Its existing interpolant certificate keeps the
entire path in the open certified phase ball. `exactTrajectory_mem` combines
that theorem with exact interpolation at grid times to prove membership of
every stored state, including both endpoints and the last state.

Consequently `exactTrajectory_energy_remainder` applies the preceding energy
estimate at each actual step \(i<N+1\), with no supplied endpoint, field,
force, or energy-regularity premise. The next stored velocity has norm at most
one, and the old-state acceleration has norm at most \(1.1\times10^{12}\).
`exactTrajectory_energy_remainder_numeric` therefore gives

\[
|\mathcal E(z_{i+1})-\mathcal E(z_i)-D\mathcal E(z_i)[\Delta z_i]|
\le1.452\times10^{34}h_N^2.
\]

Using the exact checked horizon
\(T=1/2560000000000000000064\),
`exactTrajectory_energy_remainder_rate` proves the more readable bound

\[
\boxed{
|\mathcal E(z_{i+1})-\mathcal E(z_i)-D\mathcal E(z_i)[\Delta z_i]|
\le\frac{1}{400000000(N+1)^2},\qquad i<N+1.
}
\]

Here \(\Delta z_i=h_N(v_{i+1},A(z_i))\) is the actual full phase increment of
the recurrence. The result controls the Taylor remainder of the changing
metric and potential along the certified exact-real run; its linear term is
retained. It does not assert that driven, damped, or discretized energy is
constant. No machine execution or IEEE evaluation certificate is inferred.
