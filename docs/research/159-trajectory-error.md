# #159: actual ordered-step trajectory error and refinement convergence

The formal package now proves trajectory error against an actual ODE solution
for the existing velocity-then-position step. It derives the one-step
stability and truncation bounds from the update formula and actual curve
derivatives, propagates evaluation and stored-state errors through a finite
history, and closes numerical domain membership from an exact-solution
margin. A separate theorem proves convergence of the actual histories at all
grid points as a fixed time horizon is refined.

The new modules are
[TrajectoryError.lean](../../formal/FractalSync/TrajectoryError.lean), with
18 theorem declarations, and
[TrajectoryRefinement.lean](../../formal/FractalSync/TrajectoryRefinement.lean),
with seven. Their hypotheses describe an autonomous acceleration field and
an actual ODE solution. They do not assume a local truncation estimate,
stability estimate, or numerical trajectory-error conclusion.

These results extend the
[accumulated energy/work analysis](159-accumulated-dynamics.md). Energy
accounting and trajectory approximation are distinct statements: this
increment supplies the latter for continuous force steps. The earlier
impulse ledger remains available, but the convergence theorem below does not
include impulse jumps.

## 1. The actual update and its stability constant

Let `E` be a real inner-product space and use the phase-space product norm

\[
\|(x,v)\|_{\mathrm{phase}}=\max(\|x\|,\|v\|).
\]

This is the norm used to control errors, independently of the physical ruler
metric. For an acceleration field \(a:E\times E\to E\), define

\[
F(x,v)=(v,a(x,v)).
\]

The existing `positionVelocityStep` is

\[
v^+=v+hA,\qquad x^+=x+hv^+.
\]

With the exact acceleration \(A=a(z)\), its phase update \(S_h\) satisfies
the checked identity

\[
\boxed{S_h(z)=z+hF(z)+h^2(a(z),0).}
\tag{1}
\]

This uses the same algebraic order as `integrate_step` in
[manifold.rs](../../runtime-core/src/manifold.rs). The extra position term in
(1) is retained throughout the analysis.

Suppose \(\|a(z)-a(w)\|\le L\|z-w\|\), and write \(m=\max(1,L)\).
The actual phase field obeys
\(\|F(z)-F(w)\|\le m\|z-w\|\). Expanding the two copies of (1) and
estimating their three differences gives

\[
\boxed{
\|S_h(z)-S_h(w)\|\le\kappa\|z-w\|,
\qquad \kappa=1+|h|m+h^2L.
}
\tag{2}
\]

The local stability theorem permits signed \(h\). Forward trajectory
comparison below uses \(h\ge0\), with \(L\ge0\), so \(\kappa\ge1\).

The field regularity can be supplied by actual derivatives:
`acceleration_sub_le_of_derivative_bound` proves the required Lipschitz
estimate on a convex region \(K\) from

\[
\operatorname{HasFDerivAt}(a,D a_z,z),\qquad \|D a_z\|\le L
\quad(z\in K).
\]

There is no claim that a provider capability flag or a positive ruler
supplies these derivative bounds.

## 2. Local truncation is derived from the actual ODE curve

Let \(z(t)\) be an actual solution of

\[
z'(t)=F(z(t)).
\]

The formal hypotheses use `HasDerivAt` at every point of each closed step
interval. Suppose \(\|F(z(s))\|\le B\) there and the acceleration
Lipschitz estimate holds between the reference points involved. First, the
mean-value bound gives

\[
\|z(s)-z(t)\|\le B(s-t),\qquad t\le s\le t+h.
\]

Consequently the actual derivative varies by at most \(mB(s-t)\).
`curve_quadratic_remainder` proves the vector-valued remainder estimate
directly from actual derivative variation, using a quadratic scalar barrier:

\[
\|z(t+h)-z(t)-hF(z(t))\|\le\frac{mB}{2}h^2.
\]

Combining it with the position correction in (1) gives

\[
\boxed{
\|S_h(z(t))-z(t+h)\|
\le h^2 C,\qquad C=B+\frac{mB}{2}.
}
\tag{3}
\]

Here \(B\) bounds both reference velocity and acceleration through the phase
max norm. The estimate is intentionally simple; it is not asserted to be
sharp. It includes \(h=0\) and requires no assumed local error bound or
independent formal Taylor jet of the solution.

## 3. Evaluation and stored-state errors enter separately

Let \(y_i\) be the actual stored phase state, \(A_i\) the acceleration
used for its update, and \(y_{i+1}\) the next stored state. Assume

\[
\|A_i-a(y_i)\|\le\eta,
\qquad
\|y_{i+1}-\operatorname{positionVelocityStep}(h,y_i,A_i)\|\le r.
\tag{4}
\]

The second bound can include arithmetic, storage, and any separately
accounted update discrepancy. It is a hypothesis about the actual stored
values and exact-real formula, not an IEEE certificate.

The acceleration perturbation has the exact phase difference
\((h^2(A_i-a(y_i)),h(A_i-a(y_i)))\). The checked norm estimate is
\((|h|+h^2)\eta\). Combining this with (2) and (3) proves

\[
\boxed{
e_{i+1}\le\kappa e_i+\Delta,
\quad e_i=\|y_i-z(t_i)\|,
\quad \Delta=(|h|+h^2)\eta+r+h^2C.
}
\tag{5}
\]

The connection and force discrepancies from the earlier mechanical package
fit the acceleration residual in (4). For the physical specialization

\[
a(x,v)=G_x^{-1}(Q-\nabla U)-\Gamma_x(v,v),
\]

the previously defined residual is
\(R=A+\Gamma_x(v,v)-G_x^{-1}(Q-\nabla U)=A-a(x,v)\).
The existing checked force/connection residual decomposition bounds its norm
by \(\eta_F+\eta_\Gamma\) when the separately estimated raised-force and
connection terms satisfy their stated fidelity hypotheses. Thus those
certified errors can supply \(\eta\). Numerical provider value, derivative,
and query errors must first be related to the actual terms; they are not
implicitly certified by (5).

## 4. Finite histories and domain membership are proved together

Define the scalar envelope

\[
E_0=e_0^{\rm budget},\qquad E_{i+1}=\kappa E_i+\Delta.
\]

Lean proves its exact closed form

\[
\boxed{
E_n=\kappa^n e_0^{\rm budget}
       +\Delta\sum_{j=0}^{n-1}\kappa^j.
}
\tag{6}
\]

The sum form includes \(\kappa=1\). For nonnegative inputs and
\(\kappa\ge1\), the envelope is nonnegative and nondecreasing.

`ODERegionCertificate` bundles the actual ODE equation, a uniform phase-speed
bound, reference membership in \(K\), acceleration Lipschitz regularity on
\(K\), and the time-grid relation \(t_{i+1}=t_i+h\). It contains no
numerical-history bound. For a horizon of \(N\) steps, suppose the exact
reference points have the margin

\[
\|w-z(t_i)\|\le E_N\ \Longrightarrow\ w\in K
\qquad(0\le i\le N).
\tag{7}
\]

The residual and stored-state contracts in (4) need hold only when
\(y_i\in K\). `trajectory_error_domain` proves by simultaneous induction

\[
\boxed{e_i\le E_i\quad\text{and}\quad y_i\in K
\qquad(0\le i\le N).}
\tag{8}
\]

At the initial point, the error bound and (7) establish membership. At each
successor, established membership activates the evaluation contract; the
derived local estimate controls the next error, and the reference margin
establishes the next membership. Numerical domain preservation is therefore
a conclusion rather than a premise about the whole history.

For the concrete phase region

\[
K=\{(x,v):\|x\|<R_x,\ \|v\|<R_v\},
\]

the checked `trajectory_error_phaseBox` corollary discharges (7) from the
separate margins

\[
\|x(t_i)\|+E_N<R_x,\qquad
\|v(t_i)\|+E_N<R_v.
\]

It proves both position and velocity control. If \(K\) is convex and the
next reference margin covers the derived proposed-step error,
`proposed_segment_mem` also places the entire exact affine phase segment from
\(y_i\) to the unrounded ordered endpoint in \(K\). This is the segment used
by the local phase-energy Taylor estimates. It is not a claim about every
intermediate floating-point expression or an enclosure of the ODE between
grid points.

## 5. Checked fixed-horizon refinement

Fix \(T\ge0\) and use \(n=N+1\) steps of size

\[
h_N=\frac{T}{N+1}.
\]

The positive step count avoids division by zero; \(T=0\) is included.
Assume fixed nonnegative coefficients \(\eta_0,r_0,I_0\) and the scaling

\[
\eta=h_N\eta_0,\qquad r=h_N^2r_0,\qquad
e_0^{\rm budget}=h_N I_0.
\tag{9}
\]

For \(0\le h\le T\), the checked estimates are

\[
\kappa\le1+h(m+TL),\qquad
\Delta\le h^2[(1+T)\eta_0+r_0+B+mB/2].
\]

The geometric envelope is bounded using \(1+x\le\exp x\) and an explicit
bound on its finite sum. At \(nh=T\), Lean proves

\[
\boxed{
E_n\le h\,\exp\!\bigl(T(m+TL)\bigr)
\left[I_0+T\left((1+T)\eta_0+r_0+B+\frac{mB}{2}\right)\right].
}
\tag{10}
\]

The coefficient is independent of the refinement index. The scalar envelope
therefore tends to zero, and this limit is checked using the actual sequence
\(h_N\), rather than left as an asymptotic observation.

`UniformODECertificate` supplies one actual solution, speed bound, and field
Lipschitz constant on the whole fixed interval \([0,T]\). Its
`onRefinement` construction derives the finite-step certificates for every
mesh. Finally, suppose there is a fixed \(d>0\) such that

\[
\|w-z(t)\|\le d\ \Longrightarrow\ w\in K
\qquad(0\le t\le T).
\tag{11}
\]

`UniformODECertificate.refinement_converges` proves the actual-history result:
for every \(\varepsilon>0\), all sufficiently large \(N\) satisfy

\[
\boxed{
\forall i\le N+1,\qquad
\|y_{N,i}-z(ih_N)\|<\varepsilon
\quad\text{and}\quad y_{N,i}\in K.
}
\tag{12}
\]

The provider and stored-state bounds are still conditional on membership in
\(K\). Once the derived final envelope is smaller than
\(\min(d,\varepsilon)\), the simultaneous induction proves that every
relevant state belongs to \(K\) and has the desired error. The conclusion
is uniform over all grid points of each sufficiently fine mesh, including
the terminal time.

## 6. Formal boundary and verification

The two modules contain 25 explicit theorem declarations. The targeted
command `lake build --wfail FractalSync.TrajectoryRefinement` passed with
2,497 Lake jobs and no warnings or errors; it transitively builds
`TrajectoryError`. The repository-wide build, inventory, and axiom audit are
recorded separately in [verification.md](verification.md).

The grid theorems above take an actual autonomous ODE solution and its stated
regularity, speed, and interior-margin certificates as inputs. The subsequent
[local-trajectory package](159-local-trajectories.md) constructs such a
solution and a positive tube from actual C1 acceleration, including for the
regularized literal C4 operational spline with its actual gradient and
Hessian and a C1 force. Concrete numerical regional constants, time horizons,
and finite-asset halo margins still require separate certificates.

The refinement scaling in (9) is substantive. A fixed nonzero acceleration
residual or fixed absolute stored-state error is not covered by this
vanishing-error conclusion. Fixed machine precision does not automatically
supply \(O(h^2)\) stored-state rounding as \(h\) decreases. The finite-step
bound (5) still exposes those errors, but an implementation convergence claim
requires the appropriate fidelity and arithmetic certificates.

The convergence statements in this note concern grid points; the same
local-trajectory continuation now proves uniform convergence of actual
continuous piecewise-affine interpolants at every time, with explicit
off-grid error and domain control. Time-dependent controls, impulse-jump
trajectories, event detection, and complete hybrid integrator convergence
remain separate. The earlier exact impulse energy ledger and discrete
covariant-steering consistency results do not by themselves prove these
combined evolution results. Runtime behavior, canonical parameters,
provider contracts, and training were not changed.
