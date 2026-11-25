# #160: nonlinear covariance and actual curvature tensors

The coordinate connection and curvature now have checked nonlinear change-of-coordinate laws. The curvature commutator is also proved to be a continuous trilinear tensor for differentiable bilinear connection coefficients. In the flat-Shore specialization, all coefficient regularity is discharged and the scalar curvature is independent of arbitrary regular nondegenerate frames, including after nonlinear coordinate pullback.

The new modules are [NonlinearCoordinates.lean](../../formal/FractalSync/NonlinearCoordinates.lean), [CurvatureTensoriality.lean](../../formal/FractalSync/CurvatureTensoriality.lean), and [FlatCurvatureTensor.lean](../../formal/FractalSync/FlatCurvatureTensor.lean). They extend the actual derivative and curvature definitions in [CoordinateCovariance.lean](../../formal/FractalSync/CoordinateCovariance.lean) and [FlatShoreGeometry.lean](../../formal/FractalSync/FlatShoreGeometry.lean). No runtime, training, canonical-parameter, or provider behavior is changed.

## 1. The nonlinear connection law contains the actual chart Hessian

Let `f : E → F` be a real coordinate map, write `J_x = Df_x`, and assume `f` is `C²` at `x` and `J_x` is invertible as a continuous linear map. The source space is complete. A global bijection is not required. The proof derives invertibility on a neighborhood from these local hypotheses.

The transported vector field is Mathlib's actual derivative-inverse pullback,

\[
(f^*Y)_x=J_x^{-1}Y_{f(x)}.
\]

`fderiv_pullback_apply` proves its actual derivative:

\[
D(f^*Y)_x[v]
 =J_x^{-1}\left(DY_{f(x)}[J_xv]-D^2f_x[v,(f^*Y)_x]\right).
\]

The definition of the transformed connection is

\[
\widetilde\Gamma_x(v,w)
 =J_x^{-1}\left(D^2f_x(v,w)
   +\Gamma_{f(x)}(J_xv,J_xw)\right).
\]

The second derivative in this formula is the iterated `fderiv` of `f`. It is not supplied as a symbolic jet or assumed to satisfy a connection transformation law. Differentiating the inverse Jacobian and canceling its Hessian contribution against the displayed connection term gives

\[
\widetilde\nabla_{f^*X}(f^*Y)=f^*(\nabla_XY)
\]

at the represented point. The connection uses the actual expression
`fderiv Y x (X x) + Gamma x (X x) (Y x)`.

The actual Lie-bracket law

\[
[f^*X,f^*Y]=f^*[X,Y]
\]

is checked using pinned Mathlib's second-derivative symmetry and pullback theorem. The geometric connection and curvature statements use invertible Jacobians even though Mathlib's totalized bracket theorem has a broader algebraic statement.

## 2. The actual curvature commutator transforms

The curvature convention remains

\[
R(X,Y)Z
 =\nabla_X\nabla_YZ-\nabla_Y\nabla_XZ-\nabla_{[X,Y]}Z.
\]

`curvature_pullback` proves

\[
\boxed{\widetilde R(f^*X,f^*Y)f^*Z=f^*(R(X,Y)Z).}
\]

The important step is a neighborhood identity for each inner covariant derivative. Equality only at one point would not justify differentiating it. The proof obtains local connection covariance from local `C²` chart regularity and local Jacobian invertibility, then uses actual equality of germs to identify the outer derivatives.

For the generic theorem, `X` and `Y` are differentiable at `f(x)`, `Z` is differentiable near `f(x)`, and both `nabla_Y Z` and `nabla_X Z` are differentiable at `f(x)`. These are explicit analytic premises, not a curvature covariance assumption. The concrete flat specialization below derives the intermediate differentiability from the actual smooth connection and `C²` regularity of `Z`.

The pullback metric is

\[
\widetilde g_x(v,w)=g_{f(x)}(J_xv,J_xw).
\]

The package proves positive-definiteness transfer, equality of pairings of transported fields, equality of Gram determinants, and invariance of the actual curvature/Gram ratio. The generic ratio equality uses Lean's total real division. The flat nondegenerate-frame theorems separately prove that the denominator is strictly positive.

### Why the local commutator theorem uses only a C² chart

The inner transformed covariant derivatives are identified with differentiable pullbacks before they are differentiated. This gives an actual commutator identity at `x` under the regularity hypotheses above, without separately differentiating every summand of `D²f`.

This does not assert that the transformed coefficient field is `C¹` whenever `f` is merely `C²`. The transformed coefficients contain `D²f`. The [moving-frame continuation](160-moving-frames.md) now proves actual `C¹` coefficient regularity from a `C³` chart and `C¹` original connection, with invertibility on a derived positive-radius neighborhood. The generic tensor-expansion theorem here assumes actual differentiability of its coefficient field; a merely `C²` chart does not discharge that premise.

## 3. Field-extension independence is proved from actual derivatives

`CurvatureTensoriality` takes a connection as a field of continuous bilinear maps,

\[
\Gamma:E\longrightarrow \mathcal L(E,\mathcal L(E,E)).
\]

At a point where `Gamma` is differentiable, `X` and `Y` are differentiable, and `Z` is `C²`, it proves the exact expansion

\[
\boxed{
R(X,Y)Z\big|_x
 =D\Gamma_x[X_x](Y_x,Z_x)-D\Gamma_x[Y_x](X_x,Z_x)
  +\Gamma_x(X_x,\Gamma_x(Y_x,Z_x))
  -\Gamma_x(Y_x,\Gamma_x(X_x,Z_x)).
}
\]

The intermediate derivative calculation retains the second derivatives of `Z` and the first derivatives of every field. The former cancel by the actual symmetry theorem, while the latter cancel with the Lie-bracket and connection terms. Thus the conclusion is derived from the defined commutator, not imposed as a definition of geometric curvature.

The right side is bundled as an actual continuous trilinear map

\[
\mathcal R_x:E\longrightarrow_L E\longrightarrow_L E\longrightarrow_L E.
\]

`curvature_eq_tensor` identifies this object with the actual commutator. `curvature_congr_values` proves that any two permitted field extensions with the same three point values give the same curvature, even when their derivatives differ. Alternation in the first two slots is also checked.

This is a bundled tensor on a coordinate vector space. It is not a manifold-level connection or curvature object in Mathlib's Riemannian bundle interface, nor an atlas-gluing construction.

## 4. Flat Shore: all coefficient hypotheses and arbitrary frames

For the previously defined flat metric,

\[
g=A(r)\,dr^2+B(r)\,dy^2,
\quad
A(r)=\frac{\epsilon^2+(1+a^2)r^2}{(\epsilon^2+r^2)^2},
\quad
B(r)=\frac1{\epsilon^2+r^2},
\]

the new module bundles the actual Christoffel coefficients as a continuous-bilinear-map field and proves that this field is `C∞` for `epsilon > 0`, with arbitrary real `a`.

In addition to the previously checked component `R(e_r,e_y)e_y`, it differentiates the actual first covariant derivatives and proves

\[
R(e_r,e_y)e_r=-A(r)K(r)e_y,
\qquad
R(e_r,e_y)e_y=B(r)K(r)e_r,
\]

where

\[
K(r)=\frac{\epsilon^4-(1+a^2)r^4}
 {(\epsilon^2+(1+a^2)r^2)^2}.
\]

Continuous trilinearity and alternation reconstruct the complete tensor:

\[
\boxed{R_x(u,v)w=K(r)\bigl(g_x(v,w)u-g_x(u,w)v\bigr).}
\]

The same identity holds for the actual commutator of arbitrary `X,Y,Z` with the local differentiability hypotheses stated above. This removes the earlier constant-coordinate-field restriction.

For any two point vectors,

\[
\operatorname{Gram}_g(u,v)
 =A(r)B(r)(u_rv_y-u_yv_r)^2.
\]

Consequently every frame with nonzero coordinate determinant has strictly positive Gram determinant, and

\[
\boxed{\frac{g(R(X,Y)Y,X)}
 {g(X,X)g(Y,Y)-g(X,Y)^2}=K(r).}
\]

The formal frame theorem requires `X` differentiable and `Y` `C²` at the point. It does not require constant fields, orthonormality, or a fixed orientation. Smooth frames satisfy these premises.

`frame_gramRatio_pullback` combines this result with nonlinear chart covariance: every locally invertible `C²` coordinate map and every such regular nondegenerate frame give `K((f x).1)` for the transformed metric and connection. It derives the neighborhood and intermediate-field differentiability hypotheses, and `frame_gram_pullback_pos` proves positivity of the transformed denominator.

## 5. Verification and exact boundary

| Module | Explicit proved theorem declarations |
| --- | ---: |
| `NonlinearCoordinates` | 14 |
| `CurvatureTensoriality` | 7 |
| `FlatCurvatureTensor` | 24 |
| Total | 45 |

These counts include supporting lemmas and exclude definitions and generated declarations. All three modules are checked with the repository's existing Lean/Mathlib pins and warnings treated as errors. The central verification inventory records the combined build and transitive axiom audit for the research increment.

The geometric interface now contains actual nonlinear derivative, connection, bracket, and curvature covariance; general pointwise extension independence; a continuous trilinear coordinate curvature object; and full flat curvature/frame independence. The subsequent [ambient-connection continuation](160-ambient-shore-connection.md) uses these interfaces to identify the actual regularized distance ruler's ambient connection with the curved-coordinate connection and transfer its curvature formula. This requires a certified normal patch and local Frenet data, with a \(C^2\) normal map and differentiable curvature function for connection identification, and a \(C^3\) normal map with a \(C^2\) curvature function for the curvature formula. The following remain distinct obligations:

- A manifold-level bundled connection/curvature interface with atlas compatibility. Actual coefficient regularity under a `C³` chart is now checked in the [moving-frame continuation](160-moving-frames.md).
- A globally glued frame-bundle construction. Actual coordinate connection-form gauge laws, gauge composition and fiber-valued curvature conjugacy are now checked in that continuation.
- Regularity and geodesic-ODE identification for attained AC minimizers, general second variation and conjugacy, global cut-locus theorems, and the curved-Shore functional-analytic bifurcation proof.

No statement here establishes convergence of selected minimizers or a numerical derivative certificate. The research issues remain open.
