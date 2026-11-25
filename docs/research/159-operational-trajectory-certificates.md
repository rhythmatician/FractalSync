# #159: actual finite-asset trajectory certificates

[OperationalTrajectoryCertificate.lean](../../formal/FractalSync/OperationalTrajectoryCertificate.lean)
connects the finite operational asset to an actual mechanical solution and
continuous numerical trajectories. The reference solution, horizon, field
constants, and interior tube are constructed from the stated finite-asset,
force, and phase-region evidence. The convergence theorem no longer takes an
unspecified actual ODE solution as an input.

The module contains four theorem declarations. It integrates the
[actual operational ruler bounds](159-operational-ruler-bounds.md),
[explicit quantitative ODE construction](159-quantitative-dynamics.md), and
[physical halo and extension-independence results](162-operational-halo.md).

## 1. The finite data and force assumptions

Let the asset have dimensions \(N_x,N_y\), origin \(o\), and positive physical
spacing \(h\). Assume a bound \(B\ge0\) only on the stored coefficients:

\[
|C_{ij}|\le B
\quad(0\le i<N_x,\;0\le j<N_y).
\tag{1}
\]

The lattice outside this rectangle is arbitrary. The zero-extension theorem
provides a globally bounded representative, and equality of actual metric
jets on the open halo transfers its local acceleration bounds back to the
original extension.

Fix \(\epsilon>0\), any real anisotropy parameter \(a\), initial phase state
\(p=(p_x,p_v)\), and phase radius \(R>0\). Configuration norms are Euclidean;
the phase norm is

\[
\|(x,v)\|=\max(\|x\|_2,\|v\|_2).
\]

The supplied force field \(Q(x,v)\) must have an actual bounded-Lipschitz
certificate on \(\overline B(p,R)\):

\[
\|Q(z)\|\le Q_0,
\qquad
\|Q(z)-Q(w)\|\le Q_1\|z-w\|.
\tag{2}
\]

This is the remaining external force assumption. No supplied ruler,
gradient, Hessian, or solution field is substituted for the actual operational
definitions.

For a query-displacement budget \(\delta\ge0\), require

\[
\boxed{R+\delta<m(\operatorname{chart}(p_x)),}
\tag{3}
\]

where \(m\) is the smallest of the four signed physical halo-face distances.
The ODE-certificate theorem itself only needs \(R<m\); the stronger condition
(3) is used by the numerical/query results.

## 2. Every acceleration constant is an explicit expression

The operational scalar field \(P\) defines

\[
\rho(x)=\sqrt{\epsilon^2+P(\operatorname{chart}x)^2},
\qquad u=\nabla\rho,\qquad H=D(\nabla\rho).
\]

These are the literal C4 spline, actual Riesz gradient, and actual derivative
Hessian. The preceding bounds supply

\[
G=\frac{8B}{3h},\qquad K_P=\frac{68B}{9h^2},\qquad T_P=\frac{32B}{h^3},
\]

\[
\rho\in[\epsilon,\sqrt{\epsilon^2+B^2}],\qquad
\|u\|\le G,\qquad
\|H\|\le K_P+G^2/\epsilon,
\]

with actual Hessian variation bounded by

\[
T_P+3GK_P/\epsilon+3G^3/\epsilon^2.
\tag{4}
\]

The actual acceleration is the defined expression

\[
\mathcal A(x,v)=G_{\rho,a,u}^{-1}Q(x,v)
-\Gamma_{\rho,a,u,H}(v,v).
\]

`operationalAccelerationBudget` evaluates the checked arithmetic budget for
this expression using (2), (4), and the velocity bound \(\|v\|\le\|p_v\|+R\).
Its two entries are denoted by \(A,L\):

\[
\|\mathcal A(z)\|\le A,
\qquad
\|\mathcal A(z)-\mathcal A(w)\|\le L\|z-w\|
\quad(z,w\in\overline B(p,R)).
\tag{5}
\]

Both constants are explicit real expressions. They are propagated through
addition, subtraction, inner products, scalar multiplication, and operator
evaluation using the pair rules

\[
(M,L)+(N,K)=(M+N,L+K),
\]

\[
(M,L)\cdot(N,K)=(MN,MK+NL),
\qquad
\operatorname{inverse}_{e}(M,L)=(e^{-1},L/e^2).
\]

The reciprocal rule is used with the positive ruler floor \(e=\epsilon\)
and the actual weight floor \(W=1+a^2\|u\|^2\ge1\).
The precise expression is the definition in
[MechanicalBounds.lean](../../formal/FractalSync/MechanicalBounds.lean),
specialized and proved for finite assets in
[OperationalMechanicalBounds.lean](../../formal/FractalSync/OperationalMechanicalBounds.lean).
These are conservative certificates, not sharp constant claims.

## 3. Actual reference, explicit time interval, and uniqueness

The definitions `phaseSpeedBound` and `trajectoryHorizon` fix

\[
V=\max(\|p_v\|+R,A),
\qquad
\boxed{T=\frac{R}{4(V+1)}.}
\tag{6}
\]

`exists_finiteAsset_ODE_certificate` constructs \(z(0)=p\) satisfying the
actual equation

\[
z'(t)=F(z(t)),\qquad F(x,v)=(v,\mathcal A(x,v)),\qquad0\le t\le T.
\tag{7}
\]

Its returned uniform certificate has exactly acceleration constant \(L\)
and speed constant \(V\), and the theorem derives

\[
\|z(t)-p\|\le Vt<R/4,
\]

\[
\boxed{\|w-z(t)\|\le R/2\Longrightarrow w\in B(p,R).}
\tag{8}
\]

The time interval is positive and determined by the explicit constants. The
ordinary derivatives at both endpoints of (7) come from a constructed
solution on a larger open interval. Both trajectory theorems below also
include uniqueness among actual solutions on \([0,T]\) with the same initial
state that remain in \(B(p,R)\).

## 4. Finite mesh error and every reported query

Distinguish the physical grid spacing \(h\) from the numerical time step
\(\tau=T/(N+1)\). For the defined velocity-then-position update, assume

\[
\|y_0-p\|\le\tau I_0,
\qquad
\|A_i-\mathcal A(y_i)\|\le\tau\eta_0,
\]

\[
\|y_{i+1}-\operatorname{positionVelocityStep}(\tau,y_i,A_i)\|
\le\tau^2r_0,
\tag{9}
\]

where \(I_0,\eta_0,r_0\ge0\). The evaluation and stored-state contracts need
hold only when \(y_i\in B(p,R)\). The theorem proves the needed membership;
it does not assume the numerical rollout remains in its valid region.

Set \(m=\max(1,L)\) and

\[
\mathcal K=
e^{T(m+TL)}
\left[I_0+T\left((1+T)\eta_0+r_0+V+\frac{mV}{2}\right)\right]
+\frac V2.
\tag{10}
\]

`exists_finiteAsset_trajectory_control` constructs one actual reference,
independent of the numerical mesh and history. For every history obeying
(9), the condition

\[
\boxed{2T\mathcal K\le R(N+1)}
\tag{11}
\]

implies that its actual piecewise-affine phase interpolant \(Y_\tau\) is
continuous and satisfies, at every \(t\in[0,T]\),

\[
\boxed{
\|Y_\tau(t)-z(t)\|\le\tau\mathcal K,
\qquad Y_\tau(t)\in B(p,R).
}
\tag{12}
\]

For every reported coordinate query \(q\) with

\[
\|q-\operatorname{chart}(Y_\tau(t)_x)\|_\infty\le\delta,
\]

the same conclusion supplies `SafeReportedQuery`: the reported point belongs
to the open physical halo, and every one of its 36 mathematical stencil
addresses has valid axis indices and a row-major index in
\([0,N_xN_y)\). This includes queries made at off-grid interpolated times and
queries that cross spline-cell boundaries.

## 5. Actual uniform refinement convergence

`exists_finiteAsset_interpolation_converges` constructs the actual solution
from the same finite data, force certificate, and region. For families obeying
the same shrinking budgets (9), it proves

\[
\forall\zeta>0,\quad\text{for all sufficiently large }N,\quad
\forall t\in[0,T],\quad
\|Y_{\tau_N}(t)-z(t)\|<\zeta.
\tag{13}
\]

Continuity, whole-interpolant phase-domain membership, and the displaced-query
halo/address guarantee remain in that eventual conclusion. No independent
reference trajectory or interior-margin certificate is assumed.

The fixed query-displacement budget \(\delta\) certifies address validity.
It does not by itself supply the vanishing acceleration-residual budget in
(9). If evaluation error arises from query perturbation, a separate fidelity
estimate must establish the required total residual scaling.

## 6. Verification and remaining boundary

| Declaration | Checked result |
| --- | --- |
| `safeReportedQuery_of_phaseBall` | Combined phase/query displacement implies open-halo membership and all 36 exact integer address bounds |
| `exists_finiteAsset_ODE_certificate` | Actual finite-asset mechanical reference, positive explicit horizon, exact constants, and fixed tube |
| `exists_finiteAsset_trajectory_control` | Regional uniqueness, finite-mesh all-time error, whole-interpolant domain control, and reported-query validity |
| `exists_finiteAsset_interpolation_converges` | Uniform convergence to a constructed actual solution, retaining domain and query guarantees |

`lake build --wfail FractalSync.OperationalTrajectoryCertificate` passed with
2,641 jobs and no warnings or errors. The full package build and unchanged
transitive axiom audit are recorded in [verification.md](verification.md).
The theorem hypotheses still include the
actual force norm/Lipschitz certificate and numerical residual/rounding
contracts. They do not certify IEEE query arithmetic, machine floor or casts,
availability masks, fixed-precision shrinking budgets, or a runtime
implementation's stored-state errors. The field is the operational spline's
own exact-real geometry; no ideal signed-distance fidelity is inferred.

The conclusions are local and autonomous. Global continuation, time-dependent
controls, impulse jumps, event detection, and combined transported steering
remain separate. Runtime behavior, canonical parameters, provider contracts,
training, and dependency pins are unchanged.
