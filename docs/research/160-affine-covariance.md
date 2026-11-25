# #160: actual affine coordinate covariance

The coordinate connection and curvature calculations now have a checked
interface for every constant invertible linear coordinate change and
translation. The covariant differential and curvature commutator are
recomputed using actual Fréchet derivatives. Their covariance is proved by
the chain rule, including the Lie-bracket term.

Source: [CoordinateCovariance.lean](../../formal/FractalSync/CoordinateCovariance.lean),
with fifteen proved theorem declarations. The final specialization discharges
the derivative hypotheses for the flat metric and recovers its same Gaussian
curvature formula in every such affine chart.

## 1. Coordinate convention

Let \(E,F\) be real normed spaces, let \(A:F\to E\) be a continuous linear
equivalence, and fix \(b\in E\). New coordinates \(x\in F\) represent the old
point
\[
\phi(x)=Ax+b.
\]
The transported vector field and metric are
\[
\widetilde X(x)=A^{-1}X(\phi(x)),
\qquad
\widetilde g_x(v,w)=g_{\phi(x)}(Av,Aw).
\]
The connection coefficient transformation is
\[
\widetilde\Gamma_x(v,w)
=A^{-1}\Gamma_{\phi(x)}(Av,Aw).
\]
This formula applies to affine charts because their derivative is the constant
map \(A\) and their second derivative vanishes.

The general source accepts unbundled coefficient operations \(g\) and
\(\Gamma\). Its algebraic covariance statements hold without assuming their
metric or connection laws. The geometric interpretation requires those laws;
they are available for the separately checked flat coordinate construction.
The module does not turn an arbitrary three-argument function into a
Levi-Civita connection by naming it one.

## 2. The actual covariant differential transforms

For the coordinate formula
\[
\nabla_XY(x)=dY_x[X(x)]+\Gamma_x(X(x),Y(x)),
\]
`hasFDerivAt_pullbackField` proves the actual field derivative
\[
d\widetilde Y_x=A^{-1}\circ dY_{\phi(x)}\circ A.
\]
Consequently
\[
\boxed{
\widetilde\nabla_{\widetilde X}\widetilde Y(x)
=A^{-1}(\nabla_XY(\phi(x))).
}
\tag{1}
\]
Only actual differentiability of \(Y\) at \(\phi(x)\) is needed for (1).
The derivative on the left is Lean's actual `fderiv` of the transported field,
not a separately defined formal symbol prescribed to have this behavior.

Differentiating both transported fields also proves the bracket identity
\[
\boxed{[\widetilde X,\widetilde Y](x)
=A^{-1}[X,Y](\phi(x)).}
\tag{2}
\]
Both \(X\) and \(Y\) must be differentiable at the old point. The bracket is Mathlib's actual vector-field Lie bracket.

## 3. The full curvature commutator transforms

Use the same convention as the flat and curved coordinate modules:
\[
R(X,Y)Z=\nabla_X\nabla_YZ-\nabla_Y\nabla_XZ-\nabla_{[X,Y]}Z.
\]
`curvature_pullback` proves
\[
\boxed{
\widetilde R(\widetilde X,\widetilde Y)\widetilde Z(x)
=A^{-1}(R(X,Y)Z(\phi(x))).
}
\tag{3}
\]
Its sufficient actual differentiability hypotheses are explicit:

- \(X\) and \(Y\) are differentiable at \(\phi(x)\).
- \(Z\) is differentiable everywhere, which supplies equality of the two
  inner transported covariant-derivative fields as functions.
- \(\nabla_YZ\) and \(\nabla_XZ\) are differentiable at \(\phi(x)\).

The proof uses (1) for both outer covariant derivatives and (2) for the
bracket-direction term. It does not assume the curvature transformation
formula as an input. The global hypothesis on \(Z\) is a sufficient condition
used by this theorem; a future local version could weaken it by proving the
required eventual equalities in a neighborhood.

## 4. Metric pairings and the Gram ratio

Pairings of transported fields agree exactly. Positive definiteness of an
actual metric is preserved because \(A\) is injective. The Gram expression
\[
G_g(X,Y)=g(X,X)g(Y,Y)-g(X,Y)^2
\]
is also preserved for transported fields. Combining this with (3) proves
\[
\boxed{
\frac{\widetilde g(\widetilde R(\widetilde X,\widetilde Y)\widetilde Y,
                   \widetilde X)}
     {G_{\widetilde g}(\widetilde X,\widetilde Y)}
=
\frac{g(R(X,Y)Y,X)}{G_g(X,Y)}
}
\tag{4}
\]
at the corresponding points.

As an equality of Lean's totalized real expressions, (4) does not need a
nonzero denominator. Its sectional-curvature interpretation does require a
nonzero Gram denominator and a genuine metric and connection. The flat
specialization separately proves that denominator is strictly positive.

## 5. Flat-Shore specialization

The module checks that its general covariant derivative and commutator
specialize definitionally to the existing `FlatShoreGeometry` constructions.
It then applies (4) to the old normal and tangential coordinate fields,
transporting them to the new affine chart.

Actual derivatives of \(\Gamma^r_{yy}\) and \(\Gamma^y_{ry}\) discharge the
two intermediate-field differentiability conditions. The resulting theorem
is
\[
\boxed{
\widetilde K(x)=
\frac{\epsilon^4-(1+a^2)r(\phi(x))^4}
     {(\epsilon^2+(1+a^2)r(\phi(x))^2)^2},
\qquad \epsilon>0.
}
\]
It holds for every real \(a\), every invertible constant \(A\), every
translation \(b\), and every point. Here \(r(\phi(x))\) is the old signed
normal coordinate of the represented point. An affine chart mixing normal
and tangential coordinates need not have that quantity as its first new
coordinate.

## 6. Remaining boundary

This module establishes the affine interface. The subsequent
[nonlinear covariance and tensor extension](160-nonlinear-geometry.md) now
checks the chart-Hessian contribution, actual nonlinear connection and
curvature covariance, field-extension independence, and an actual continuous
trilinear coordinate tensor. The flat specialization also proves the same
curvature ratio in every regular nondegenerate frame after nonlinear
pullback. A manifold bundle/atlas construction, moving-frame connection-form
laws, and minimizer regularity remain separate.

`lake build --wfail FractalSync.CoordinateCovariance` passes under the pinned
Lean 4.24.0 / Mathlib environment. No proof placeholders, extra axioms, runtime
changes, or provider-contract changes are introduced. Package-wide verification
is recorded in [verification.md](verification.md).
