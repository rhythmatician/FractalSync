# Issue 159: residual from holding the constructed drive covector

The checked results in
[`HeldControlResidual.lean`](../../formal/FractalSync/HeldControlResidual.lean)
quantify the difference between two exact-real force evaluations: a normalized
fixed command reconstructed at the current state, and its covector constructed
at an earlier sample state and then held. The bound is conditional on both
states remaining in the certified region. A separate displacement estimate is
required to express it in terms of elapsed time or a numerical step size.

## 1. The runtime policy being compared

[`FrozenRollout::attempt_step`](../../runtime-core/src/operational_engine.rs)
accepts a constructed control covector `q`. `rk4_advance` passes that same `q`
to all four evaluations of `rhs`. `rk4_profile_advance` also retains `q` across
all four substeps of `Rk4FourSubsteps`. Each `rhs` call recomputes the potential
force, wall force, viscous drag, inverse metric, and connection at its current
stage state. Thus drag is not part of the frozen input in this comparison.

The separate
[`MotionControls::drive_covector`](../../runtime-core/src/controls.rs)
constructs the normalized metric-consistent drive from a command and position.
The formal `catalogForce` uses the same exact-real drive formula but
reevaluates it at each phase point. Runtime clamping, the near-zero command
branch, floating-point operations, and numerical stage construction are outside
this result. The local numerical corollaries use a fixed nonzero direction and
unit drive magnitude.

## 2. Exact cancellation of unchanged terms

Write the covector associated with the fixed command as

\[
Q_d(z)=G(x)\,\operatorname{normalize}_{G(x)}(e),
\qquad z=(x,v),\qquad e\ne0.
\]

At a current state \(z\), let \(A(z)\) be the acceleration with reevaluated
drive, and \(A_{\mathrm{hold}}(z;q_0)\) the acceleration with supplied covector
\(q_0\). Both expressions use the current conservative forces, drag, inverse
metric, and connection. Lean proves the exact identity

\[
A_{\mathrm{hold}}(z;q_0)-A(z)
=G(x)^{-1}\bigl(q_0-Q_d(z)\bigr).
\]

The theorem `frozen_catalog_acceleration_sub` contains the actual
`catalogForce` and new `frozenCatalogForce` expressions. The connection term is
arbitrary because it cancels identically. Potential strengths and the damping
coefficient are also arbitrary and equal in the two evaluations.

## 3. General holding and input-error bounds

The shared rank-one inverse satisfies

\[
\|G(x)^{-1}w\|\le \rho(x)^2\|w\|.
\]

Suppose \(|\rho(x)|\le R_\rho\) at the current point, and the constructed
covector field is \(L_Q\)-Lipschitz on a region containing the sample state
\(z_0\) and current state \(z\). Then the checked holding estimate is

\[
\boxed{
\|A_{\mathrm{hold}}(z;Q_d(z_0))-A(z)\|
\le R_\rho^2 L_Q\,d(z_0,z).
}
\]

For a separately supplied sample-construction error
\(\|q_0-Q_d(z_0)\|\le\delta_Q\), Lean proves

\[
\boxed{
\|A_{\mathrm{hold}}(z;q_0)-A(z)\|
\le R_\rho^2\delta_Q+R_\rho^2L_Q\,d(z_0,z).
}
\]

The sample error is an explicit premise. No value of \(\delta_Q\) is asserted
for a Rust query, machine-rounded position, or compiled drive evaluation.

There are two separate time interfaces. If a numerical stage satisfies
\(d(z_0,z)\le B\tau\), its holding residual is at most
\(R_\rho^2L_QB\tau\). If \(z(t)\) is an actual \(B\)-Lipschitz curve in the
region, the corresponding bound is
\(R_\rho^2L_QB|t-t_0|\). The stage interface deliberately assumes only a
stage-displacement bound; it does not identify a Runge–Kutta stage with an
exact solution point.

## 4. Numerical constants from the improved local field estimates

For the local certificates

\[
\rho:(2,1),\qquad u=\nabla\rho:(1,20),\qquad \rho\ge1,
\qquad a=F=1,
\]

where each pair gives a norm bound and Lipschitz constant, the existing
normalization and lowering bounds yield

\[
Q_d:(4,170).
\]

`normalized_drive_local_bound` checks this exact numerical specialization for
every fixed nonzero command direction. The resulting estimates are

\[
\|A_{\mathrm{hold}}(z;Q_d(z_0))-A(z)\|
\le 680\,d(z_0,z),
\]

and, if \(d(z_0,z)\le8\tau\),

\[
\boxed{
\|A_{\mathrm{hold}}(z;Q_d(z_0))-A(z)\|\le5440\tau.
}
\]

With the additional sample-error premise, the general theorem gives
\(4\delta_Q+5440\tau\). These constants need to be included in a numerical
residual budget if that budget compares held-input evolution with the
reevaluated-command reference field. A certificate permitting a unit residual
coefficient does not automatically cover the coefficient \(5440\).

The speed bound of an exact reference curve alone does not prove the
displacement bound for a held-input numerical stage. For four fixed RK4
substeps, holding age is measured from the start of the outer step, since the
runtime does not reconstruct `q` at the internal substep endpoints.

## 5. Formal boundary

All identities and inequalities above are exact-real results. The new module
contains ten theorem/lemma declarations. It supplies the previously missing
holding-policy residual as a function of regional and displacement data.

The subsequent [held-RK extension](159-held-runge-kutta.md) now supplies actual
exact-real stage displacement, stage/endpoint region membership and first-order
method residuals. It instantiates a complete finite-run certificate for the
fixture's exact outer-sampled drive policy, including one and four internal
substeps. That extension uses its own larger residual coefficients and does
not treat the reference speed bound as a numerical-stage certificate.

Neither extension proves accepted-step or refusal behavior, compiled provider
fidelity, work quadrature, or floating-point arithmetic. Changing commands, impulses,
combined steering, and continuation require their additional hypotheses and
estimates. No runtime implementation or control policy is changed.
