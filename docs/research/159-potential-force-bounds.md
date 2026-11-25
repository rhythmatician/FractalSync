# Issue 159: explicit regional bounds for the conservative force catalog

## Scope

`formal/FractalSync/PotentialForceBounds.lean` proves norm and pairwise Lipschitz
bounds for the existing `PotentialForces.shoreForce`, `wallForce`, and
`totalForce`. These are the actual real-valued catalog definitions already tied
to the potential differential in `PotentialForces.lean`. The new bounds do not
introduce substitute forces or require an unspecified compactness constant.

The estimates apply on an arbitrary region of a pseudometric parameter space.
Position, ruler, and gradient fields may therefore be composed with phase-space
projections before using the theorem. A regional certificate
`BoundedLipschitzOn f S M L` means

\[
\|f(z)\|\le M,\qquad
\|f(z)-f(w)\|\le L\,d(z,w),\qquad z,w\in S,
\]

with both constants nonnegative. Convexity of the supplied region is not
required: these are pairwise estimates propagated through explicit operations.

The coefficient and parameter choices below are mathematical instance choices.
They do not change runtime defaults, canonical parameters, provider contracts,
or training. In particular, choosing \(\kappa=\log2\) in an example is not a
claim that it is the production Shore parameter.

## Checked Shore-force bound

Suppose \(\rho\ge\epsilon>0\), and let
\((R_0,L_\rho)\), \((U_0,L_u)\) be the ruler and gradient certificates. For

\[
Q_{\rm Shore}(z)
=\frac{\kappa}{\rho(z)\log2}\,u(z),
\]

`shoreForce_bound` propagates the constant, inverse-ruler, and scalar-vector
product certificates. Its budget is

\[
Q_{S,0}=\left|\frac\kappa{\log2}\right|\frac{U_0}{\epsilon},
\qquad
L_S=\left|\frac\kappa{\log2}\right|
\left(\frac{L_u}{\epsilon}
+\frac{U_0L_\rho}{\epsilon^2}\right).
\]

The upper ruler bound \(R_0\) is carried by the input interface but is not needed
by this inverse estimate. `shoreForce_log_two_bound` packages the exact
cancellation at \(\kappa=\log2\), including the proof that \(\log2\ne0\).

## Checked wall-force bound

Keep the existing definitions

\[
s(x)=\frac{\|x\|^2}{4},\qquad
\theta(x)=\frac\pi2s(x)^4,
\]

\[
Q_{\rm wall}(x)
=-\mu\pi\,s(x)^3\sec\theta(x)\tan\theta(x)\,x.
\]

The proof rewrites the coefficient as
\(-\mu\pi s^3\sin\theta\cos^{-2}\theta\), then propagates certificates
through the actual factors. Global scalar sine and cosine difference bounds are
proved using the mean value estimate and their actual derivatives. Inverse
cosine uses a positive explicit denominator floor.

For a position certificate \((r,L_x)\) and a cosine floor \(c>0\), define

\[
\begin{aligned}
B_s&=(r^2/4,\ rL_x/2),\\
B_\theta&=\left(\frac\pi2(r^2/4)^4,\ \frac{\pi r^7L_x}{64}\right),\\
B_{\sin\theta}&=(1,B_\theta^{(2)}),\\
B_{1/\cos\theta}&=(c^{-1},B_\theta^{(2)}/c^2).
\end{aligned}
\]

`wallCoordinateBudget`, `wallPhaseBudget`, and `wallForceBudget` encode these
operations using the existing product-budget rule

\[
(M,L)(N,K)=(MN,MK+NL).
\]

Consequently the recursively encoded wall-force budget expands to

\[
Q_{W,0}=|\mu\pi|\frac{r^7}{64c^2},
\]

\[
L_W=|\mu\pi|L_x\left(
\frac{7r^6}{64c^2}
+\frac{\pi r^{14}}{4096c^2}
+\frac{\pi r^{14}}{2048c^3}\right).
\]

The Lean theorem `wallForce_bound` uses the recursive expression; the display
above is its algebraic expansion. It is intentionally conservative, since the
sine norm is bounded by one even on small radii.

### Radius supplies the denominator evidence

A position bound \(\|x\|\le r<2\) supplies the floor

\[
c(r)=\cos\left(\frac\pi2(r^2/4)^4\right)>0.
\]

`wallPhase_le_radius`, `wallCosFloor_pos`, and `wallCosFloor_le` prove the
required phase comparison, positivity, and lower bound on the actual cosine.
`wallForce_bound_of_radius` therefore needs no separately supplied
trigonometric floor. This is a quantitative regional estimate inside the
physical wall domain; the constants are allowed to deteriorate as \(r\uparrow2\).

On the unit position ball, the stronger simple interface

\[
\cos\theta(x)\ge\tfrac12
\]

is checked by `half_le_cos_wallPhase`. It uses
\(\theta(x)\le\pi/512<\pi/3\) and cosine monotonicity.

At the actual wall default \(\mu=1/\pi\),
`wallForce_default_unit_bound` gives the fully rational certificate

\[
\boxed{Q_{W,0}=\frac1{16},\qquad L_W=\frac{L_x}{2}.}
\]

The intermediate Lipschitz estimate is
\((7/16+5\pi/1024)L_x\); the checked inequality \(\pi<4\) implies the stated
rational upper bound. The wall coefficient is not discarded.

## Complete conservative force

`totalForce_bound` adds the checked bounds for the actual catalog sum.
`totalForce_log_two_default_unit_bound` packages the particularly useful
instance \(\kappa=\log2\), \(\mu=1/\pi\), \(\|x\|\le1\):

\[
\boxed{
Q_{\mathrm{cons},0}=\frac{U_0}{\epsilon}+\frac1{16},\qquad
L_{\mathrm{cons}}=
\frac{L_u}{\epsilon}+\frac{U_0L_\rho}{\epsilon^2}+\frac{L_x}{2}.
}
\]

For example, certificates \(\epsilon=1\), \((R_0,L_\rho)=(8,200)\),
\((U_0,L_u)=(200,50000)\), and \((r,L_x)=(1,1)\) yield

\[
Q_{\mathrm{cons},0}=200+\frac1{16},\qquad
L_{\mathrm{cons}}=90000+\frac12.
\]

This arithmetic example does not by itself establish the ruler and gradient
certificates for an asset. Those obligations belong to the finite-asset
instantiation. Likewise, drive and drag budgets must be added separately when
forming the full phase-force bound \((Q_0,L_Q)\).

## Verification and boundary

The module contains 16 proved theorem declarations. The target was checked
with `lake build --wfail FractalSync.PotentialForceBounds` using the repository's
existing Lean and Mathlib pins. The root proof-package build and transitive axiom
audit cover its inclusion in the aggregate package.

These are exact-real covector bounds. The inverse-metric operation is supplied
by the existing mechanical certificate layer. The results do not assert
production asset/reference fidelity, floating-point query or arithmetic
accuracy, global trajectory continuation, or energy conservation of a
particular numerical integrator.
