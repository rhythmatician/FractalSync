# A local horizon from the actual quadratic reference

This continuation retains the literal finite asset, actual force catalog and
initial state of the [concrete trajectory instance](159-concrete-trajectory-certificate.md).
It replaces the global coefficient propagation by local polynomial-reference
bounds and simplifies the inverse metric before estimating the raised forces.
The resulting certified horizon is

\[
T=\frac1{576}\approx0.0017361111
\]

in model time. The preceding certificate's horizon was
\(1/2560000000000000000064\). The ratio is approximately
\(4.4444\times10^{18}\). Neither value specifies physical seconds or a runtime
timestep recommendation.

The checked sources are
[LocalRulerBounds.lean](../../formal/FractalSync/LocalRulerBounds.lean),
[RankOneInverseBounds.lean](../../formal/FractalSync/RankOneInverseBounds.lean),
[CancelledCatalogBounds.lean](../../formal/FractalSync/CancelledCatalogBounds.lean),
[LocalTrajectoryMesh.lean](../../formal/FractalSync/LocalTrajectoryMesh.lean), and
[ImprovedConcreteTrajectory.lean](../../formal/FractalSync/ImprovedConcreteTrajectory.lean).
The [local fixture note](162-local-fixture-bounds.md) gives the asset-level
derivation, including crossings of spline cell boundaries.

## 1. Region and actual local jets

Positions and velocities retain their Euclidean norms. Phase space has the
maximum norm. The center, initial velocity and radius remain

\[
x_0=(1/2,1/2),\qquad v_0=(1/4,0),\qquad R=1/16.
\]

Write \(f\) for the actual floor-selected finite-asset spline pulled back to
the Euclidean plane. Its reference is \(P-h^2\), where
\(P(x,y)=x^2+2xy-3y^2\). On the closed position ball, the checked bounds are

\[
|f|\le\tfrac14,\quad \|Df\|\le4,\quad
\|D^2f\|\le12,\quad \operatorname{Lip}(D^2f)\le1.
\]

The final bound subtracts the ideal quadratic before estimating variation.
The quadratic Hessian is constant. Only the tiny coefficient-error spline
contributes to its variation, with explicit gain \(32\eta/h^3\le1\), where
\(\eta=2^{-48}\). The actual global spline differentiation and query bounds
make this valid across cell faces and vertices.

For \(\rho=\sqrt{1+f^2}\), \(u=\nabla\rho\), and \(H=Du\), the local scalar
regularizer bounds retain the small factor \(|f|\):

\[
\left|\frac{f}{\rho}\right|\le\tfrac14,\qquad
\rho^{-3}\le1,\qquad
\left|\frac{-3f}{\rho^5}\right|\le\tfrac34.
\]

The scalar second-derivative coefficient therefore has Lipschitz constant
\(3/4\) on the value interval \([-1/4,1/4]\). The actual second chain rule is

\[
D^2\rho=\frac f\rho D^2f+\frac1{\rho^3}Df\otimes Df.
\]

Its direct norm and variation budgets are \(19\) and \(769/4\). Riesz
dualization preserves the relevant Euclidean operator bounds. Convex
mean-value estimates then give the following slightly enlarged pairs. A pair
\((M,L)\) means both a norm bound \(M\) and a Lipschitz constant \(L\).

| Quantity on the closed phase ball | Previous pair | New pair |
| --- | --- | --- |
| \(\rho\), with \(\rho\ge1\) | \((8,200)\) | \((2,1)\) |
| \(u=\nabla\rho\) | \((200,50000)\) | \((1,20)\) |
| \(H=Du\) | \((50000,30000000)\) | \((20,200)\) |
| Velocity projection | \((1,1)\) | \((1/3,1)\) |
| Actual acceleration | \((1100000000000,40000000000000000000)\) | \((8,400)\) |

## 2. Preserve the inverse metric's cancellations

The inverse rank-one factor

\[
U(u)=(I+a^2u\otimes u)^{-1}
\]

is a contraction, including when \(u=0\). The proof uses the positive
quadratic form of its inverse and the checked two-sided inverse identity.
Thus

\[
\|G^{-1}w\|\le\rho^2\|w\|.
\]

The exact resolvent identity gives a variation estimate without separately
estimating the subtractive terms in the Sherman--Morrison formula:

\[
\|(U(u)-U(\widetilde u))w\|
\le a^2(\|u\|+\|\widetilde u\|)
\|u-\widetilde u\|\,\|w\|.
\]

For ruler pair \((R_\rho,L_\rho)\) and gradient pair \((U,L_u)\), the inverse
operator therefore has norm bound \(R_\rho^2\) and variation bound

\[
2R_\rho L_\rho+2R_\rho^2a^2UL_u.
\]

With the concrete parameters \(a=F=\beta=1\), \(\kappa=\log2\), and
\(\mu=1/\pi\), raising the actual catalog gives the identity

\[
G^{-1}Q_{\rm tot}
=\frac\rho W u+G^{-1}Q_{\rm wall}+d_g-v,
\qquad W=1+\|u\|^2.
\]

The raised Shore term, normalized drive and drag are simplified before
propagating bounds. The new exact arithmetic budgets are

| Expression | Norm bound | Lipschitz constant |
| --- | ---: | ---: |
| Raised catalog \(G^{-1}Q_{\rm tot}\) | \(55/12\) | \(701/4\) |
| Connection \(\Gamma(v,v)\) | \(8/3\) | \(1748/9\) |
| Acceleration before enlargement | \(29/4\) | \(13301/36\) |
| Acceleration used by the certificate | \(8\) | \(400\) |

These are regional majorants for the actual functions, not sampled maxima.

## 3. Actual reference, all-time mesh control and queries

The phase-speed bound is \(B=8\). It is kept separate from the acceleration
Lipschitz constant \(L=400\). The actual Picard construction gives

\[
T=\frac{R}{4(B+1)}=\frac1{576},\qquad
z(0)=(x_0,v_0),\qquad z'=(v,A(z)).
\]

The certificate includes regional uniqueness, the displacement estimate
\(\|z(t)-z(0)\|\le8t\), and an interior tube of radius \(R/2\).

For \(m=N+1\) steps and \(\tau=T/m\), the existing interpolation theorem
admits initial error at most \(\tau\), acceleration residual at most \(\tau\),
and stored-step error at most \(\tau^2\). Its error coefficient is

\[
K=e^{T(400+400T)}
\left[1+T\left((1+T)+1+8+\frac{400\cdot8}{2}\right)\right]+4.
\]

Lean proves \(K\le16\). Therefore every nonempty mesh fits in the certified
tube and has the uniform interpolation bound

\[
\boxed{\quad
\|\widetilde y_m(t)-z(t)\|\le\frac1{36m},\qquad
\widetilde y_m(t)\in B((x_0,v_0),1/16),\quad 0\le t\le T.
\quad}
\]

The exact-real velocity-first recurrence is instantiated at this new horizon.
Its residuals vanish by definition. For **2048 steps**, the displayed bound
is \(1/73728<2^{-16}\); the packaged theorem gives the uniform target
\(2^{-16}\) for both position and velocity in the phase norm.

Every reported coordinate query within \(1/64\) of the interpolated position
has all 36 valid mathematical stencil addresses. This follows from the same
literal physical halo and margin as before and holds throughout the interval,
including between stored times. The query displacement is an admitted error
budget, not an IEEE query-rounding certificate.

## 4. Energy on the longer run

The [local-energy extension](159-local-energy-bounds.md) proves that the actual
phase energy \(E=K+U\) has differential norm at most \(5\) and differential
Lipschitz constant at most \(134\) on the same closed phase ball. Its
contained-step Taylor coefficient is therefore \(67\).

For the actual recurrence, the stored-state theorem supplies every endpoint
condition. Let \(J_i=DE(y_i)[y_{i+1}-y_i]\). The checked finite-run estimates are

\[
|E(y_{i+1})-E(y_i)-J_i|\le\frac1{64m^2},\qquad
\left|E(y_m)-E(y_0)-\sum_{i=0}^{m-1}J_i\right|\le\frac1{64m}.
\]

At 2048 steps, these are respectively \(2^{-28}\) and \(2^{-17}\). The
linear differential sum is retained: the displayed defect measures the
Taylor approximation, while the driven and damped system can change energy.
No physical-work quadrature or conservation statement follows from this
defect alone.

## 5. Scope and next useful work

The [held-control residual](159-held-control-residual.md) separately bounds
the acceleration discrepancy from holding the supplied normalized-drive
covector. Drag, potential, ruler and inverse metric remain state dependent.
The bound requires stage displacement or an appropriate curve-speed
certificate. The subsequent [held-RK extension](159-held-runge-kutta.md) now
supplies actual stage displacement and endpoint containment, and derives
the corresponding method residuals for one or four internal RK4 substeps.
It uses a separate interpolation coefficient with all-time error<=3/(16m),
rather than inserting the larger holding coefficient into the unit-residual
mesh theorem above.

The new horizon addresses the prior microscopic constant propagation on this
fixture. It remains local and exact real. Production distance fidelity, actual
machine query and arithmetic errors, operational acceptance and work quadrature,
changing commands, impulses, combined steering, and certified
continuation beyond this interval remain open. The previous smooth-minimizer,
cut-locus, bifurcation, manifold/atlas, reach and stronger singular-limit
boundaries are unchanged.

Runtime behavior, canonical parameters, training, provider contracts,
dependency pins and audit rules are unchanged. PR #164 remains a draft.
