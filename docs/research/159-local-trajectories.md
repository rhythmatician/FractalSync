# #159: local ODE existence and continuous numerical trajectories

The preceding [trajectory-error package](159-trajectory-error.md) established
convergence at the stored grid points against a supplied actual ODE solution.
This increment constructs a local reference solution from actual acceleration
regularity and proves convergence of a continuous interpolation of the stored
phase states at every time in a fixed interval.

Three new modules contain twenty proved theorem declarations:

| Module | Declarations | Checked scope |
| --- | ---: | --- |
| [TrajectoryInterpolation.lean](../../formal/FractalSync/TrajectoryInterpolation.lean) | 9 | Actual continuous piecewise-affine interpolation, grid reproduction, off-grid error, explicit finite-step domain control, and uniform convergence at all times |
| [LocalDynamics.lean](../../formal/FractalSync/LocalDynamics.lean) | 6 | Actual local solution existence, a uniform ODE certificate and positive interior tube, uniqueness in the certified region, and the smooth mechanical-acceleration specialization |
| [OperationalDynamics.lean](../../formal/FractalSync/OperationalDynamics.lean) | 5 | Literal global physical spline on the Euclidean plane, actual regularized gradient/Hessian regularity, actual query-jet derivative, and local operational mechanical trajectories |

The numerical statements retain the exact-real velocity-then-position update
already analyzed. Runtime code, parameters, provider contracts, and training
are unchanged.

## 1. The continuous trajectory is an actual defined function

Use the phase-space norm

\[
\|(x,v)\|=\max(\|x\|,\|v\|).
\]

For a positive step \(h\) and stored states \(y_0,\ldots,y_n\), the defined
interpolant \(Y_h\) uses the affine segment between successive stored values:

\[
Y_h(ih+s)
=\left(1-\frac{s}{h}\right)y_i+\frac{s}{h}y_{i+1},
\qquad 0\le s\le h.
\]

`trajectoryInterpolant` constructs this function by finite recursion. The
previous interpolant is used up to its last grid time; the next affine segment
is used after that time. `trajectoryInterpolant_grid` proves
\(Y_h(ih)=y_i\) for every \(i\le n\), and
`continuous_trajectoryInterpolant` proves actual continuity. Agreement at each
join is checked using the endpoint theorem. For the global definition, the
path is constant before zero and the last affine segment extends beyond the
final time. Only \([0,nh]\) is used in the error statements.

This is an explicitly chosen interpolation of the stored numerical history.
It is not a claim that the renderer uses this interpolation, that the
interpolated position derivative equals its interpolated velocity between
nodes, or that the interpolant solves the mechanical ODE exactly.

## 2. The off-grid estimate uses both neighboring endpoints

Let \(z\) be a \(B\)-Lipschitz reference curve on \([0,nh]\). Suppose all
stored grid values have error at most \(e\):

\[
\|y_i-z(ih)\|\le e,\qquad 0\le i\le n.
\]

`trajectoryInterpolant_error` proves the uniform estimate

\[
\boxed{\|Y_h(t)-z(t)\|\le e+\frac{Bh}{2}
\qquad(0\le t\le nh).}
\tag{1}
\]

For \(t=ih+s\), write \(\theta=s/h\). The two endpoint errors contribute at
most \((1-\theta)e+\theta e=e\). The reference displacements contribute

\[
(1-\theta)Bs+\theta B(h-s)
=2Bh\theta(1-\theta)\le\frac{Bh}{2}.
\]

The vector identities, norm inequalities, division conditions, and final
scalar estimate are all checked. The generic interpolation lemmas work in a
real normed space; they do not require an inner product.

For a `UniformODECertificate`, the reference Lipschitz hypothesis in (1) is
derived from the actual equation

\[
z'(t)=F(z(t)),\qquad F(x,v)=(v,a(x,v)),\qquad \|F(z(t))\|\le B.
\]

`UniformODECertificate.reference_lipschitz` applies the mean-value theorem to
these actual derivatives on the whole interval. No independent reference
displacement estimate is assumed.

## 3. Explicit finite-mesh error and domain control

Retain the previous notation

\[
m=\max(1,L),\qquad
\mathcal C=
e^{T(m+TL)}
\left[I_0+T\left((1+T)\eta_0+r_0+B+\frac{mB}{2}\right)\right],
\]

where \(L\) is the acceleration Lipschitz constant and \(B\) the reference
phase-speed bound. For \(h=T/(N+1)\), the actual initial, evaluation, and
stored-state error contracts are

\[
\|y_0-z(0)\|\le hI_0,
\qquad \|A_i-a(y_i)\|\le h\eta_0,
\]

\[
\left\|y_{i+1}-
\operatorname{positionVelocityStep}(h,y_i,A_i)\right\|\le h^2r_0.
\]

The evaluation and stored-state contracts need hold only when \(y_i\in K\).
Assume the exact curve has an interior tube of radius \(d\):

\[
0\le t\le T,\quad \|w-z(t)\|\le d
\quad\Longrightarrow\quad w\in K.
\tag{2}
\]

`UniformODECertificate.interpolation_error_bound` proves that the explicit
step-size condition

\[
\boxed{h\left(\mathcal C+\frac B2\right)\le d}
\tag{3}
\]

implies, at every time in the interval,

\[
\boxed{
\|Y_h(t)-z(t)\|
\le h\left(\mathcal C+\frac B2\right),
\qquad Y_h(t)\in K.
}
\tag{4}
\]

The proof first bounds the previously defined numerical error envelope by
\(h\mathcal C\). Condition (3) places that envelope within the exact-curve
tube, activating the existing simultaneous grid-error/domain induction.
Equation (1) then controls the interpolation between nodes, and (2) gives its
domain membership. No membership hypothesis about the complete numerical
history is inserted into this argument. The region \(K\) need not be convex.

This is an explicit inequality in the supplied constants. It does not
compute those constants from a runtime asset, nor does it certify their
floating-point evaluation.

## 4. Uniform convergence at every time

For a family of histories satisfying the same mesh-scaled contracts,
`UniformODECertificate.interpolation_converges` proves that, for every
\(\varepsilon>0\), every sufficiently fine mesh has

\[
\forall t\in[0,T],\qquad
\|Y_{h_N}(t)-z(t)\|<\varepsilon,
\qquad Y_{h_N}(t)\in K.
\tag{5}
\]

Continuity of each interpolant is included in the conclusion. This strengthens
the preceding uniform grid-point conclusion to the complete interval. The
formal statement uses the explicit epsilon/eventually quantifiers for uniform
convergence. The reference tube is fixed as the mesh is refined.

The interpolation theorems use \(T>0\), so every step has a nonzero
denominator. The preceding grid theorem still includes the separate
zero-horizon case. No new positive-time conclusion is inferred for a
degenerate mesh.

## 5. The reference solution can now be constructed locally

Let the underlying real inner-product space be complete. If the actual
acceleration \(a:E\times E\to E\) is `ContDiffAt` of order one at the initial
phase point \(p\), then `exists_local_ODE_certificate` proves the existence of

\[
T>0,\qquad R>0,\qquad z:\mathbb R\to E\times E,
\]

such that

\[
z(0)=p,
\qquad z'(t)=F(z(t))\quad(0\le t\le T),
\]

and supplies an actual `UniformODECertificate` on the phase ball
\(K=B(p,R)\). It also proves the fixed positive tube

\[
\|w-z(t)\|\le R/2\quad\Longrightarrow\quad w\in B(p,R)
\qquad(0\le t\le T).
\tag{6}
\]

The proof uses the pinned Mathlib
[Picard–Lindelöf theorem](https://github.com/leanprover-community/mathlib4/blob/f897ebcf72cd16f89ab4577d0c826cd14afaafc7/Mathlib/Analysis/ODE/PicardLindelof.lean).
Actual C1 acceleration gives a Lipschitz neighborhood. A phase ball is chosen
inside that neighborhood, the actual integral curve is constructed, and the
time interval is shortened so that the curve lies in the ball of radius
\(R/2\). On the full phase ball, the checked speed estimate is

\[
\|F(w)\|\le \max(1,L)R+\|F(p)\|.
\]

These quantities supply the uniform certificate. The horizon and region
constants are existential; this theorem does not produce numerical values
for a particular physical configuration.

`UniformODECertificate.solution_unique` uses the pinned Mathlib
[Grönwall uniqueness theorem](https://github.com/leanprover-community/mathlib4/blob/f897ebcf72cd16f89ab4577d0c826cd14afaafc7/Mathlib/Analysis/ODE/Gronwall.lean)
to identify this reference with every other actual solution on the same
interval that remains in \(K\) and has the same initial state. It makes no
claim of global existence or continuation through a singularity.

## 6. The mechanical acceleration has a checked regularity adapter

The actual defined mechanical field is

\[
a(x,v)=G^{-1}_{\rho(x),a,u(x)}Q(x,v)
-\Gamma_{\rho(x),a,u(x),H(x)}(v,v),
\]

using the existing exact inverse and compact connection definitions.
Here \(Q\) denotes the net force covector, represented by the ambient inner
product. Conservative and nonconservative contributions can be included in
this field.

`contDiffAt_mechanicalAcceleration` proves actual C1 regularity from C1
coefficient fields \(\rho,u,H,Q\) at the initial configuration/phase point and
\(\rho(x_0)\ne0\). The denominator
\(W=1+a^2\|u\|^2\) is positive by the existing metric theorem. The proof uses
actual continuous-linear-map evaluation for \(H(x)v\), inner products,
division, and scalar multiplication.

`exists_local_mechanical_certificate` applies the local-existence theorem to
this specific acceleration and supplies the same positive time interval,
phase ball, uniform certificate, and interior tube.

This general adapter treats \(u\) and \(H\) as specified coefficient fields.
For its geometric interpretation, their identities with \(\nabla\rho\) and
its derivative must come from actual field-jet results. It does not infer
those identities from smoothness or from a provider flag. The operational
spline specialization below supplies them by using the actual derivatives
in its definition.
For a smooth ruler, obtaining C1 regularity of its Hessian normally requires
the corresponding third-order regularity. Regularization of distance alone
does not supply that at an arbitrary cut-locus point.

## 7. Local mechanics for the literal operational spline

The [global operational field](162-global-operational-field.md) is an actual
C4 spline on the entire coordinate plane, including its knots. Let
\(P_{C,o,h}\) be its physical field for coefficient lattice \(C\), origin
\(o\), and physical spacing \(h>0\). On the Euclidean plane, define

\[
\rho(x)=\sqrt{\epsilon^2+P_{C,o,h}(\operatorname{chart}x)^2},
\qquad \epsilon>0,
\]

\[
u(x)=\nabla\rho(x),\qquad H(x)=D(\nabla\rho)(x).
\]

`OperationalDynamics.rulerField` is this actual function. The chart is the
existing continuous linear equivalence from the L2 product plane to the
coordinate product. Riesz gradients use the Euclidean inner product, while
the phase-state norm remains the maximum of the position and velocity norms.

`contDiff_rulerField` proves C4 regularity of \(\rho\).
`actual_jet_regularity` then proves C3 regularity of the actual Mathlib Riesz
gradient and C2 regularity of its actual Fréchet derivative. These are not
arbitrary coefficient fields satisfying smoothness premises.
`hasFDerivAt_rulerField` additionally identifies the actual differential with
the literal physical query jet:

\[
D\rho(x)=\frac{P_{C,o,h}(\operatorname{chart}x)}{\rho(x)}
\,DP_{C,o,h}(\operatorname{chart}x)\circ\operatorname{chart}.
\]

The acceleration definition substitutes exactly these \(\rho,u,H\) into
the checked mechanical field from section 6. Given a net force field that
is C1 at the initial phase point, `contDiffAt_acceleration` derives C1
regularity of the resulting mechanical acceleration.
`exists_local_certificate` constructs a genuine local solution, a uniform
ODE certificate on a phase ball, and the positive tube in (6). Thus the
previous continuous-interpolation error and convergence theorems apply to
this specific smooth operational metric, subject to their stated
mesh-scaled numerical contracts. Regional solution uniqueness follows from
the existing certificate theorem.

The spline need not equal a signed-distance function. This is a theorem about
the operational spline's own exact-real geometry. Positive spacing gives
the intended physical interpretation; the regularity and local-existence
proofs actually hold for every real spacing under Lean's totalized division.
For a finite asset, the separate halo theorems still determine where the
chosen lattice extension agrees with the finite query. This local ODE
theorem does not itself certify that its existential phase ball stays inside
that finite query domain.

## 8. Verification and remaining boundary

The targeted command `lake build --wfail FractalSync.LocalDynamics` passed
with 2,546 Lake jobs and no warnings or errors; it builds the interpolation
module transitively. The additional target
`lake build --wfail FractalSync.OperationalDynamics` passed with 2,630 jobs
and no warnings or errors. The new Picard–Lindelöf, Grönwall, and Euclidean
product inner-product imports use the existing pinned Mathlib cache.
No dependency version changed. The complete
package build and unchanged transitive axiom audit are recorded separately in
[verification.md](verification.md).

The package now has an actual local reference solution under C1 acceleration,
an actual-metric-jet specialization for the regularized C4
operational spline, and actual continuous-interpolant convergence under the
stated numerical contracts. The next numerical obligations are concrete values for regional
derivative, speed, time, and margin constants for a particular asset and
force, together with actual provider and arithmetic error certificates.
The [quantitative operational continuation](159-operational-trajectory-certificates.md)
now derives explicit real formulas for those regional mechanical constants
from finite stored coefficient bounds, force norm/variation bounds, and an
actual physical halo margin. It constructs an explicit positive horizon and
gives a sufficient mesh inequality for whole-interpolant error and query
validity. This replaces the existential constant selection for that
specialization; it does not supply numerical evidence for a concrete asset
or force. Fixed nonzero errors do not satisfy the vanishing mesh-scaled
budgets automatically.

Time-dependent controls, event detection, impulse-jump convergence, and a
combined covariant-steering evolution remain separate. The interpolation
result concerns position and velocity values, not convergence of their
derivatives. No global existence, long-time conservation, or full runtime
integrator certification is claimed.
