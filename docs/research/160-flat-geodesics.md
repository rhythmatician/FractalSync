# #160: smooth flat geodesic contracts and the Shore Jacobi field

The flat model now has checked conservation laws for actual smooth coordinate
geodesic solutions, a verified unit-speed Shore geodesic with a parallel
normal, and an actual coordinate Jacobi equation whose normal scalar form is
\(f''+f=0\). The explicit field \(J(t)=\epsilon\sin(t)e_r\) has first positive
zero at intrinsic arclength \(\pi\).

Source: [FlatShoreGeodesics.lean](../../formal/FractalSync/FlatShoreGeodesics.lean),
with twenty-five proved theorem declarations. It extends the actual
[coordinate connection and curvature](160-flat-shore.md), whose metric is
now connected to the unrestricted AC path infimum by
[the Euclidean flat bridge](160-flat-intrinsic.md).

These results do not yet show that arbitrary minimizing AC paths are smooth,
identify a bundled conjugate point, or prove the global cut locus. The
conventional threshold and cut-locus arguments remain in the flat-Shore note.

## 1. Actual solution hypotheses

For \(\epsilon>0\), any real \(a\), and coordinates \(q=(r,y)\), write
\[
g=A(r)\,dr^2+B(r)\,dy^2,
\quad
A(r)=\frac{\epsilon^2+(1+a^2)r^2}{(\epsilon^2+r^2)^2},
\quad B(r)=\frac1{\epsilon^2+r^2}.
\]
The predicate `GeodesicAt epsilon a q v t` states the two actual derivative
conditions
\[
q'(t)=v(t),\qquad
v'(t)=-\Gamma_{q(t)}(v(t),v(t)),
\]
using `HasDerivAt`. These are assumptions that a supplied path solves the
coordinate geodesic equation; existence of such an ODE solution is not built
into its name.

For
\[
E(t)=\frac12g_{q(t)}(v(t),v(t)),\qquad
p(t)=B(r(t))v_y(t),
\]
Lean proves the actual derivatives
\[
\boxed{E'(t)=0,\qquad p'(t)=0.}
\]
The energy proof differentiates the actual metric pairing and uses the
previously proved metric-compatibility identity. The momentum proof
differentiates \(B(r)v_y\), uses \(B'=2B\Gamma^y_{ry}\), and substitutes the
actual tangential equation of motion.

The mean value theorem then proves constancy of \(E\) and \(p\) between any
two parameter values in an open interval on which the solution hypotheses
hold. No constancy or conserved quantity is assumed as an input.

The checked algebraic first integrals are
\[
\boxed{v_y=(\epsilon^2+r^2)p,}
\qquad
\boxed{A(r)v_r^2=2E-(\epsilon^2+r^2)p^2.}
\]
Together with the conservation theorems these supply the familiar radial
quadrature for smooth solutions. They do not alone classify all geodesics,
solve their endpoint problem, or establish minimizing behavior.

## 2. Unit-speed Shore motion and its parallel normal

Define
\[
\gamma(t)=(0,y_0+\epsilon t),\qquad
T=(0,\epsilon),\qquad N=(\epsilon,0).
\]
Every Christoffel coefficient vanishes at \(r=0\). Lean checks the actual
velocity of \(\gamma\), its geodesic equation, and
\[
g(T,T)=g(N,N)=1,\qquad g(N,T)=0.
\]
Thus \(t\) is intrinsic arclength along this Shore line.

For an actual coordinate field \(J\) along a curve, the module defines
\[
D_tJ=J'+\Gamma_{\gamma(t)}(\gamma',J).
\]
On the Shore curve the actual covariant derivative equals the ordinary
derivative. In particular, \(D_tN=0\). This is the checked parallel normal
used in the Jacobi calculation.

## 3. The curvature action is computed from the commutator

Using the convention
\[
R(X,Y)Z=\nabla_X\nabla_YZ-\nabla_Y\nabla_XZ-\nabla_{[X,Y]}Z,
\]
`curvature_shore_normal_tangent` evaluates the actual commutator on the
constant coordinate extensions of \((j,0)\) and the unit tangent \(T\):
\[
\boxed{R((j,0),T)T=(j,0)\quad\text{at }(0,y).}
\]
The calculation differentiates both resulting connection-profile fields.
It uses the actual Fréchet derivative of the normal profile in the normal
direction and of the tangential profile in the tangential direction. The
vanishing of \(\Gamma\) at the evaluation point removes the remaining
connection terms. The factor \(\epsilon^2\) from the two unit tangent inputs
cancels the coordinate curvature coefficient \(\epsilon^{-2}\).

This supplies the curvature term in the coordinate Jacobi equation. The
definition in this module chooses constant coordinate extensions explicitly.
The later [tensoriality extension](160-nonlinear-geometry.md) proves that the
actual flat curvature commutator is independent of the regular field
extensions and identifies its full continuous trilinear tensor. It therefore
justifies this choice without restricting the resulting curvature tensor to
constant fields.

## 4. The normal Jacobi equation and its first zero

Define the actual residual
\[
\mathcal J(J)=D_t(D_tJ)+R(J,\gamma')\gamma',
\]
with the actual along-curve derivatives and the constant-extension curvature
commutator just described.

For a normal profile \(J(t)=(\epsilon f(t),0)\), the module assumes actual
first-derivative data for \(f\) and an actual derivative of that derivative
at the evaluation point. It proves
\[
\boxed{\mathcal J(J)(t)=(\epsilon(f''(t)+f(t)),0).}
\]
Since \(\epsilon>0\), the vector residual vanishes exactly when
\(f''(t)+f(t)=0\). This is derived from the actual connection and commutator;
it is not an unrelated scalar sine identity labeled as a geometric theorem.

For the explicit field
\[
\boxed{J(t)=(\epsilon\sin t,0),}
\]
Lean checks both actual derivatives and proves
\[
\mathcal J(J)=0,\qquad J(0)=0,\qquad D_tJ(0)=N.
\]
The initial derivative is metric-unit by the frame theorem. It also proves
\[
J(\pi)=0,\qquad J(t)\ne0\quad(0<t<\pi).
\]
Thus this nontrivial normal Jacobi field has first positive zero \(\pi\) in
intrinsic arclength. The actual Shore curve has tangential displacement
\[
\gamma_y(\pi)-\gamma_y(0)=\pi\epsilon.
\]
This checks the field-level origin of the familiar scale in the conventional
shortcut argument. It does not yet constitute a formal theorem about the
first conjugate point, the index form, global minimality up to that point,
or the appearance of two minimizing excursions beyond it. The separate
[actual length second variation](160-flat-second-variation.md) now proves
strict shortening by every sufficiently small positive sine amplitude for
normalized span greater than pi, with actual AC and fixed endpoints. That
concrete family result does not classify global minimizers.

## 5. Verification and remaining interfaces

`lake build --wfail FractalSync.FlatShoreGeodesics` passes with the repository's
pinned Lean 4.24.0 and Mathlib environment. No placeholders or custom axioms
are used; package-wide verification is recorded in [verification.md](verification.md).

Remaining formal steps include smooth regularity of the AC minimizers,
ODE existence and endpoint analysis, manifold-level tensor interfaces,
Jacobi/conjugacy and general second-variation criteria, and the global cut-locus
classification. The coordinate tensor and nonlinear covariance are now checked
in the extension linked above, and the sine-family second variation/shortening
is checked separately. No runtime, provider, training, or
canonical-parameter change is made by these proofs.
