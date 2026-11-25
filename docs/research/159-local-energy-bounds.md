# Issue 159: local energy bounds at the improved concrete horizon

[`ImprovedConcreteEnergy.lean`](../../formal/FractalSync/ImprovedConcreteEnergy.lean)
propagates the new local fixture estimates to the actual phase-energy
differential. It uses the same literal asset, actual regularized ruler,
conservative potential, and unit-parameter catalog as the preceding
[concrete certificate](159-concrete-trajectory-certificate.md). The new
velocity-first recurrence runs over the improved horizon \(T=1/576\).

## 1. Actual differential bounds

Let

\[
\mathcal E(x,v)=K(x,v)+U(x),
\]

where \(U\) is the existing Shore-plus-wall potential with
\(\kappa=\log 2\), \(\mu=1/\pi\), and reference ruler one. The position
space uses its Euclidean norm. The phase-space norm is
\(\|(x,v)\|=\max(\|x\|,\|v\|)\).

On the same closed phase ball of radius \(1/16\), the new inputs and outputs
are as follows. A pair lists a uniform norm bound followed by a Lipschitz
constant.

| Actual quantity | Checked pair |
| --- | --- |
| Ruler \(\rho\) | \((2,1)\) |
| Ruler gradient \(\nabla\rho\) | \((1,20)\) |
| Ruler Hessian \(D\nabla\rho\) | \((20,200)\) |
| Velocity projection | \((1/3,1)\) |
| Potential gradient \(\nabla U\) | \((17/16,43/2)\) |
| Phase-energy differential \(D\mathcal E\) | \((5,134)\) |

The potential-gradient calculation combines the Shore-force pair
\((1,21)\) and default wall-force pair \((1/16,1/2)\). Its sign changes
when passing from force to gradient, without changing either bound.

The existing `energyCovectorBudget` evaluates exactly to

\[
\left(\frac{601}{144},\frac{267}{2}\right).
\]

The exported differential bound enlarges these numbers to \((5,134)\).
This is a bound on the actual Fréchet differential of \(\mathcal E\), using
the already proved ruler, gradient, and potential derivative identities. No
additional phase Hessian or independently assumed energy-remainder function
is supplied.

Convexity also yields the checked value estimate

\[
|\mathcal E(z)-\mathcal E(w)|\le5\,\|z-w\|
\]

for two states in the closed phase ball. This can translate a certified phase
error into an energy-value error on that region.

## 2. Complete changing-position steps

The ordered exact-real force step is

\[
v_+=v+hA(z),\qquad x_+=x+hv_+,
\qquad \Delta z=h\,(v_+,A(z)).
\]

If its two endpoints lie in the certified ball, convexity keeps the full
interpolating phase segment there. The new `energyStepCertificate` has the
literal differential Lipschitz constant \(134\), and Lean proves

\[
\boxed{
|\mathcal E(z_+)-\mathcal E(z)-D\mathcal E(z)[\Delta z]|
\le67h^2\max(\|v_+\|,\|A(z)\|)^2.
}
\]

The metric and potential in the final energy are evaluated at the new
position. This is a complete changing-configuration estimate, not a
frozen-metric kinetic-energy identity.

## 3. The actual finite recurrence

For \(m=N+1\) steps, set \(h=T/m=1/(576m)\). The trajectory certificate
proves membership in the phase ball at every stored state. Consequently the
energy theorem's endpoint conditions are discharged at every actual step.
The new velocity and acceleration bounds give

\[
\|v_{i+1}\|\le\frac13,\qquad \|A(z_i)\|\le8.
\]

Writing \(J_i=D\mathcal E(z_i)[z_{i+1}-z_i]\), the checked step bound is

\[
|\mathcal E(z_{i+1})-\mathcal E(z_i)-J_i|
\le4288h^2
\le\frac{1}{64m^2}.
\]

Finite telescoping then proves

\[
\boxed{
\left|\mathcal E(z_m)-\mathcal E(z_0)-\sum_{i=0}^{m-1}J_i\right|
\le\frac{1}{64m}.
}
\]

For the same 2,048-step mesh used by the improved trajectory certificate,
these are \(2^{-28}\) per step and \(2^{-17}\) for the complete Taylor ledger.
The refinement bounds hold on the new, much longer horizon. They are not
claimed to be smaller than the previous remainder estimates over the
microscopic horizon.

## 4. Meaning and boundary

The ledger sums actual differential increments \(J_i\). It is not an exact
physical work integral and is not a claim of energy conservation. The model
contains drive and damping, so its physical energy generally changes. A
comparison with a selected work quadrature needs the corresponding force,
connection, and time-discretization identities and residual bounds.

The results concern the certified exact-real velocity-first recurrence. They
do not identify that recurrence with the operational Runge–Kutta update. The
[holding residual](159-held-control-residual.md) separately quantifies the
state-dependent drive discrepancy; its displacement and sample-error
conditions must be supplied before applying it to numerical stages.

Machine query rounding, actual IEEE field and force evaluations, changing
commands, impulses, combined steering, and continuation beyond the local
interval retain their separate proof obligations. Runtime behavior and
parameters are unchanged.
