# A concrete catalog, finite asset, horizon, and mesh certificate

This note retains the first fully instantiated certificate and its constants.
The subsequent [local-horizon certificate](159-local-horizon.md) uses the same
asset and actual dynamics with much tighter bounds, horizon \(1/576\), and a
2048-step error certificate.

This increment discharges the asset and regional force premises of the earlier
[finite-asset trajectory theorem](159-operational-trajectory-certificates.md)
for a specific nonconstant fixture. It also constructs an exact-real numerical
run, proves its continuous interpolation error, and certifies every admitted
reported query throughout the interval. The numerical horizon is extremely
conservative; the theorem is a complete instance of the proof interfaces, not
a practical production timestep recommendation.

The checked implementation is
[OperationalCatalogBounds.lean](../../formal/FractalSync/OperationalCatalogBounds.lean),
[RationalTrajectoryMesh.lean](../../formal/FractalSync/RationalTrajectoryMesh.lean),
and [ConcreteTrajectoryCertificate.lean](../../formal/FractalSync/ConcreteTrajectoryCertificate.lean).
The [asset note](162-concrete-asset-instance.md) records the literal data,
provenance model, and independently checked reference-fidelity results.

## 1. Actual configuration and force

Use the existing 16 by 16 quadratic runtime test fixture, materialized as
256 exact dyadic coefficients. Its spacing is the exact real value of the
binary64 literal `0.1`:

\[
h=\frac{3602879701896397}{36028797018963968}.
\]

Let \(\phi_h\) be its actual globally glued quintic field, pulled back to the
Euclidean plane. The ruler and its derivatives are the actual functions

\[
\rho(x)=\sqrt{1+\phi_h(x)^2},\qquad
u=\nabla\rho,\qquad H=Du.
\]

The example explicitly chooses

\[
\epsilon=a=F=\beta=d_{\rm ref}=1,\qquad
\kappa=\log 2,\qquad \mu=1/\pi.
\]

These are research-instance parameters, not changes to canonical parameters.
All conservative, drive, and damping terms are present. The fixed command is
\(e=(1/4,0)\); its positive scale is immaterial to metric normalization:

\[
d_g(x)=\frac{e}{\sqrt{g_x(e,e)}}.
\]

The actual total force covector, represented by its ambient Riesz vector, is

\[
Q_{\rm tot}(x,v)
=Q_{\rm Shore}(x)+Q_{\rm wall}(x)+G_xd_g(x)-G_xv.
\]

Here \(Q_{\rm tot}\) includes the conservative forces. In the work-energy
identities, \(Q\) continues to denote the nonconservative part. The two names
must not be interchanged when interpreting energy changes.

The acceleration is the existing actual mechanical expression

\[
A(x,v)=G_x^{-1}Q_{\rm tot}(x,v)-\Gamma_x(v,v).
\]

The [potential-force bounds](159-potential-force-bounds.md) prove the actual
Shore and secant-wall estimates. The wall denominator floor follows from the
position radius. The [control-force bounds](159-control-force-bounds.md)
derive regularity of the actual metric normalization for a fixed nonzero
command, then bound the actual drive and viscous covectors. No heading
Lipschitz assumption is inserted into the concrete theorem.

## 2. Region, norms, and explicit constants

The initial state and certified phase region are

\[
p=(x_0,v_0),\qquad x_0=(1/2,1/2),\qquad v_0=(1/4,0),
\qquad K=\overline B(p,1/16).
\]

Positions and velocities use the Euclidean norm; phase space uses
\(\|(x,v)\|=\max(\|x\|_2,\|v\|_2)\). Operator norms in the following table
are induced by these norms. A pair \((M,L)\) means both
\(\|f(z)\|\le M\) and
\(\|f(z)-f(w)\|\le L\|z-w\|\) on the stated region.

| Actual field on the phase region | Norm bound | Lipschitz constant |
| --- | ---: | ---: |
| \(\rho\), with \(\rho\ge1\) | 8 | 200 |
| \(u=\nabla\rho\) | 200 | 50,000 |
| \(H=Du\) | 50,000 | 30,000,000 |
| Position and velocity projections | 1 each | 1 each |
| Shore force | 200 | 90,000 |
| Wall force | \(1/16\) | \(1/2\) |
| Complete \(Q_{\rm tot}\) | 400,000 | 4,000,000,000,000 |
| Actual acceleration \(A\) | 1,100,000,000,000 | 40,000,000,000,000,000,000 |

The first three rows follow from the literal coefficient bound \(|C_{ij}|\le7\),
\(h\ge1/10\), the actual spline derivative gains, and the regularizer chain
rule. The force rows follow from the preceding actual catalog formulas.
`MechanicalBounds.acceleration_bound` then propagates those certificates
through the existing inverse metric and connection. Every displayed majorant
is checked by Lean; none is an empirical maximum.

These estimates discard substantial cancellation and use global coefficient
magnitudes even though the query region is small. In particular, the fixture's
much sharper polynomial reference certificates have not yet been used to
reduce the mechanical constants in this instance.

## 3. Constructed solution and exact horizon

For a shared acceleration-Lipschitz and phase-speed bound, choose

\[
B=4\times10^{19}.
\]

The phase-speed bound is deliberately enlarged to the Lipschitz bound. This
also keeps the exponential factor in the numerical estimate small at the
chosen horizon. With \(R=1/16\), Lean checks

\[
T=\frac{R}{4(B+1)}
=\frac1{2560000000000000000064}
\approx3.90625\times10^{-22}.
\]

This is model time; no conversion to physical seconds is supplied.
`exists_reference` constructs an actual solution

\[
z(0)=p,\qquad z'(t)=(v(t),A(z(t))),\qquad 0\le t\le T.
\]

The certificate includes the actual derivative at the endpoints,
phase-region membership, the displacement bound \(\|z(t)-p\|\le Bt\), and a
reference tube of radius \(R/2\). `exists_certified_trajectory` also packages
the regional uniqueness conclusion. A solution or minimum safe horizon is
not assumed as input.

The tiny horizon is a consequence of these coarse proof constants. It is not
evidence of measured stiffness, instability, or a necessary runtime step size.
Tighter local derivative and inverse-metric estimates are needed for a useful
operational horizon.

## 4. A numerical mesh with explicit all-time guarantees

Write \(m=N+1\ge1\) and \(\tau=T/m\). The perturbed-step theorem accepts

\[
\|y_0-p\|\le\tau,\qquad
\|A_i-A(y_i)\|\le\tau,\qquad
\|y_{i+1}-\operatorname{step}_\tau(y_i,A_i)\|\le\tau^2.
\]

The acceleration and stored-step inequalities need only be supplied when
\(y_i\) belongs to the phase region. The domain proof establishes that
membership, so the hypotheses do not assume the desired whole-history
conclusion. These are explicit exact-real error budgets. Fixed machine
precision is not claimed to meet them at this microscopic horizon.

The checked rational-horizon lemma proves that the interpolation coefficient
\(C\) satisfies \(C\le B\), and that the numerical tube condition holds for
every \(m\ge1\). Therefore the actual continuous stored-state interpolant
\(\widetilde y_m\) obeys

\[
\boxed{\quad
\|\widetilde y_m(t)-z(t)\|\le\frac1{64m},\qquad
\widetilde y_m(t)\in B(p,1/16),\quad 0\le t\le T.
\quad}
\]

In particular, 1024 steps give the explicit uniform phase-error bound
\(2^{-16}=1/65536\). The phase error controls both Euclidean position and
velocity errors. It does not assert that the interpolated velocity equals
the derivative of interpolated position.

There is also an unconditional exact-real instance. `exactTrajectory` is the
recursive velocity-first update

\[
v_{i+1}=v_i+\tau A(x_i,v_i),\qquad
x_{i+1}=x_i+\tau v_{i+1},\qquad y_0=p.
\]

Its numerical residuals vanish by its definition. The theorem
`exists_exact_scheme_certificate` proves the all-mesh conclusions for that
actual recursion; `exists_1024_step_certificate` packages the specific
1024-step result. Neither theorem requires an assumed residual certificate.

## 5. All-time query and stencil validity

The exact halo margin at \(x_0\) is

\[
m(x_0)=\frac12-2h
=\frac{5404319552844595}{18014398509481984}>\frac14.
\]

Allow a reported coordinate displacement \(\delta=1/64\), measured in the
coordinate maximum norm. Since \(R+\delta=5/64<m(x_0)\), every query within
\(\delta\) of the interpolated position lies in the open physical halo.
The theorem certifies all 36 mathematical row and column indices and their
row-major addresses throughout the complete time interval, including cell
crossings. The displacement limit is an admitted error budget, not a proved
machine query-rounding error.

## 6. Concrete phase-energy regularity

The [phase-energy extension](159-phase-energy-bounds.md) uses the actual
conservative potential and proves a numerical norm/Lipschitz certificate for
the actual derivative of

\[
E(x,v)=\tfrac12g_x(v,v)+U(x).
\]

On the same closed phase ball,

\[
\|DE\|\le20{,}000{,}000,\qquad
\operatorname{Lip}(DE)\le24{,}000{,}000{,}000.
\]

For an actual update whose old and proposed phase states belong to that ball,
convexity supplies the segment hypothesis. With \(v_+=v+\Delta t A(x,v)\),
the checked Taylor remainder is

\[
\left|E(y_+)-E(y)-DE_y[\Delta t(v_+,A(y))]\right|
\le12{,}000{,}000{,}000\,\Delta t^2
\max(\|v_+\|,\|A(y)\|)^2.
\]

For the certified finite recurrence, `exactTrajectory_mem` proves both
endpoint conditions for every actual step. The final specialization eliminates
the trajectory-dependent factor as well:

\[
\boxed{
|E(y_{i+1})-E(y_i)-DE_{y_i}[y_{i+1}-y_i]|
\le \frac1{400000000\,m^2},\qquad 0\le i<m.
}
\]

This is an actual changing-metric energy Taylor remainder relative to the
displayed linear differential term. It is not a total energy-drift bound or
conservation claim for a driven and damped system.

## 7. Exact scope and next useful tightening

The finite asset, actual catalog force bounds, actual field derivatives,
reference solution, numerical horizon, mesh condition, and exact-real update
are instantiated. The asset's reference fidelity is checked against its
quadratic reference, with the raw-sample spline bias retained. None of those
statements is a certificate for production Mandelbrot distance data.

The extraction record models separately rounded, non-fused binary64
operations and checks them with exact rational rounding. It does not prove
compiled Rust or compiler-contraction equivalence. Query rounding, machine
integer conversions, storage and masks, and IEEE field/force evaluation still
need their own evidence.

The formal force reevaluates the normalized fixed-command drive at each phase
point. The operational engine accepts an already constructed control covector
and holds it fixed through its Runge-Kutta stages. Drag is recomputed from each
stage's state. The subsequent [holding residual](159-held-control-residual.md)
proves the exact-real acceleration discrepancy under explicit displacement
and covector-error bounds. Runtime stage displacement, Runge-Kutta truncation,
and machine errors still require certificates. Changing commands, impulses,
combined steering, and continuation beyond the local interval remain separate.

The [local extension](159-local-horizon.md) now propagates the polynomial
reference jet bounds through the ruler and metric while retaining the
rank-one inverse's cancellations. It replaces the microscopic horizon for
this same fixture, while the present theorem remains a valid older bound.
No runtime behavior, training policy, canonical parameter, provider contract,
dependency pin, or audit rule changes in this increment.
