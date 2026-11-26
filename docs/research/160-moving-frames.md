# #160: moving-frame connection forms and coefficient regularity

The coordinate geometry package now checks the actual moving-frame gauge law, including differentiation of the inverse frame. It also proves that a `C³` nonlinear coordinate map and `C¹` original connection give an actually `C¹` transformed coefficient field on a neighborhood of an invertible Jacobian. A fiber-valued continuous trilinear curvature tensor connects these two results to the actual curvature commutator.

The new sources are [MovingFrames.lean](../../formal/FractalSync/MovingFrames.lean), [CoordinateConnectionRegularity.lean](../../formal/FractalSync/CoordinateConnectionRegularity.lean), and [ConnectionFormCurvature.lean](../../formal/FractalSync/ConnectionFormCurvature.lean). They extend the [nonlinear covariance and tensor results](160-nonlinear-geometry.md), retaining the same actual Fréchet derivative and curvature convention. Runtime behavior, training, canonical parameters, and provider contracts are unchanged.

## 1. Base directions and fiber coordinates are separate

Let `E` be the fixed coordinate base space and `G` the space in which a section takes values. A coordinate connection form is a field of continuous bilinear maps

\[
\Gamma_x:E\times G\longrightarrow G.
\]

The Lean type is `E → E →L[ℝ] G →L[ℝ] G`. For a base direction field `X : E → E` and a section `Z : E → G`, the covariant derivative is the actual expression

\[
\nabla_X Z\big|_x=DZ_x[X_x]+\Gamma_x(X_x,Z_x).
\tag{1}
\]

A moving frame is an actual operator field `P : E → F →L[ℝ] G`, invertible at the point in question. The section's coordinates in that frame are

\[
z_x=P_x^{-1}Z_x.
\]

This changes the fiber coordinates while leaving the base coordinate directions unchanged. A general moving frame need not be the derivative of a coordinate map. Accordingly, the actual base Lie bracket remains the bracket of the original base direction fields; no false bracket-preservation law is assumed for arbitrary frame multiplication.

The inverse-frame calculus requires completeness of `F`, the source fiber of the invertible operator. The base space need not be complete for these frame statements. The coordinate-change application below supplies its own complete coordinate domain.

## 2. The inverse-frame derivative is proved

`eventually_isInvertible` derives invertibility on a neighborhood from continuity of `P` and invertibility of `P_x`. The proof uses openness of the space of continuous linear equivalences.

`fderiv_inverseFrame` proves

\[
\boxed{D(P^{-1})_x[v]=-P_x^{-1}\circ DP_x[v]\circ P_x^{-1}.}
\tag{2}
\]

This is an equality for the actual `fderiv` of the totalized inverse-operator field. The proof first obtains actual differentiability of inversion at the invertible value, then differentiates the neighborhood identity `P(y) ∘ P(y)⁻¹ = I`. It does not receive the derivative formula as input.

Applying the actual product rule to the section coordinates gives

\[
D(P^{-1}Z)_x[v]
=P_x^{-1}\left(DZ_x[v]-DP_x[v](P_x^{-1}Z_x)\right).
\tag{3}
\]

Only differentiability of `P` and `Z` at the point, together with invertibility of `P_x`, is needed for this pointwise formula. Separate regularity statements show that an invertible frame and a section of the same `C^n` regularity produce frame coordinates of that regularity.

## 3. Actual connection and curvature gauge covariance

The transformed connection form is bundled continuously bilinearly and evaluates to

\[
\boxed{\Gamma^P_x(v,w)
=P_x^{-1}\left(DP_x[v]w+\Gamma_x(v,P_xw)\right).}
\tag{4}
\]

The inhomogeneous term is the actual derivative of the frame field. Combining (3) with (4) proves the actual covariant-derivative identity

\[
\nabla^{\Gamma^P}_X(P^{-1}Z)=P^{-1}\nabla^\Gamma_X Z.
\tag{5}
\]

The curvature definition is the actual commutator

\[
R^\Gamma(X,Y)Z
=\nabla_X\nabla_YZ-\nabla_Y\nabla_XZ-\nabla_{[X,Y]}Z.
\]

`curvature_gauge` proves

\[
\boxed{R^{\Gamma^P}(X,Y)(P^{-1}Z)=P^{-1}R^\Gamma(X,Y)Z.}
\tag{6}
\]

For the generic commutator theorem, the frame is locally `C¹`, `Z` is differentiable near the point, and the two original intermediate covariant sections are differentiable at the point. The proof derives equality of the inner covariant sections as germs before differentiating them. Thus pointwise equality is not substituted for equality of derivatives. The stronger coefficient/tensor theorem below discharges the intermediate regularity premises from `C²` frame and `C¹` coefficient hypotheses.

### Successive frame changes

For `Q : E → H →L[ℝ] F`, the checked composition law is

\[
\boxed{(\Gamma^P)^Q=\Gamma^{P Q}}
\tag{7}
\]

at every point where both frame fields are differentiable and invertible. Here `(P Q)_x=P_x\circ Q_x`. The proof differentiates this actual product, retaining both derivative contributions, and verifies the inverse composition and cancellation. A constant identity frame leaves the connection form unchanged.

Equation (7) supplies the local coefficient overlap law needed when relating frame descriptions. It does not itself construct a globally glued vector bundle or a manifold connection.

## 4. Actual coefficient regularity under frame and coordinate changes

The raw commutator law (6) does not assert that a merely `C¹` frame produces `C¹` transformed coefficients. Those coefficients contain `DP`. The new regularity theorem uses the sufficient hypotheses

\[
P\in C^2,\quad \Gamma\in C^1,\quad P_x\text{ invertible},
\]

and proves that the actual bundled field `Gamma^P` is `C¹` at `x`, in the continuous-bilinear operator topology.

For a nonlinear coordinate map `f : E → F`, set `J_x=Df_x`. First pull back the base-direction argument of the old connection:

\[
\Gamma^0_x(v,w)=\Gamma_{f(x)}(J_xv,w).
\]

Then apply the frame law with `P=J`. The resulting bundled coefficients are definitionally the existing nonlinear formula

\[
\widetilde\Gamma_x(v,w)
=J_x^{-1}\left(D^2f_x[v,w]+\Gamma_{f(x)}(J_xv,J_xw)\right).
\tag{8}
\]

`pullback_apply` and `coefficients_pullback` check compatibility with `NonlinearCoordinates.pullbackConnection`. There is no second competing transformation convention.

`contDiffAt_pullback` proves the sufficient regularity statement

\[
\boxed{f\in C^3\text{ at }x,\quad
 \Gamma\in C^1\text{ at }f(x),\quad J_x\text{ invertible}
 \quad\Longrightarrow\quad\widetilde\Gamma\in C^1\text{ at }x.}
\tag{9}
\]

The coordinate domain `E` is complete. No global bijection is required. The proof derives `J ∈ C²`, pulls back the original coefficient field using actual composition, and applies the checked frame-regularity theorem.

The neighborhood theorem strengthens the conclusion to

\[
\exists\delta>0\quad\forall y\in B(x,\delta),\qquad
 Df_y\text{ is invertible and }\widetilde\Gamma\text{ is }C^1\text{ at }y.
\tag{10}
\]

This is an actual positive-radius neighborhood derived from the hypotheses. The radius is existential; the theorem supplies no numerical lower bound for it. The transformed-coordinate curvature theorem can consequently apply the existing tensor identification to arbitrary regular new-coordinate fields, with the coefficient differentiability premise discharged.

## 5. A genuine fiber-valued curvature tensor

The new tensor module allows the base and fiber spaces to differ. For a differentiable connection form, differentiable direction fields `X,Y`, and a `C²` section `Z`, it expands the actual commutator and proves

\[
R^\Gamma(X,Y)Z\big|_x=\mathcal R_x(X_x,Y_x,Z_x),
\]

where

\[
\begin{aligned}
\mathcal R_x(u,v,w)={}&D\Gamma_x[u](v,w)-D\Gamma_x[v](u,w)\\
&+\Gamma_x(u,\Gamma_x(v,w))-\Gamma_x(v,\Gamma_x(u,w)).
\end{aligned}
\tag{11}
\]

Second derivatives of the section cancel by actual Hessian symmetry. Its first derivatives and the derivatives of the direction fields cancel with the base bracket and connection terms. This establishes dependence on the three point values.

The object is bundled as

\[
\mathcal R_x:E\longrightarrow_L E\longrightarrow_L F\longrightarrow_L F.
\]

Thus continuity and multilinearity are part of its Lean type. The formula is identified with the actual commutator under the stated regularity, rather than imposed as a surrogate definition of the commutator.

For a `C²` invertible frame and a `C¹` original connection, `curvatureTensor_gauge` proves

\[
\boxed{\mathcal R^{\Gamma^P}_x(u,v,w)
=P_x^{-1}\mathcal R^\Gamma_x(u,v,P_xw).}
\tag{12}
\]

The proof derives regularity of the transformed coefficients and of the transported test section, applies (6), and uses actual tensoriality on both sides. It does not assume a tensor transformation law. When the fiber is the coordinate tangent space, both the commutator and tensor specialize definitionally to the existing `CoordinateCovariance` and `CurvatureTensoriality` objects; compatibility lemmas record those equalities.

## 6. Verification and remaining boundaries

| New module | Explicit proved theorem declarations |
| --- | ---: |
| `MovingFrames` | 13 |
| `CoordinateConnectionRegularity` | 5 |
| `ConnectionFormCurvature` | 8 |
| Total | 26 |

Supporting lemmas are included; definitions and generated helpers are not. The targeted verification is

```text
lake build --wfail FractalSync.MovingFrames \
  FractalSync.CoordinateConnectionRegularity FractalSync.ConnectionFormCurvature
```

It uses the existing Lean 4.24.0 and Mathlib pins. Combined package verification and the transitive axiom audit are recorded in [verification.md](verification.md).

The pinned Mathlib manifold tree was inspected before choosing this interface. These results operate on coordinate base spaces and trivial fiber coordinates. They do not claim an existing manifold connection/curvature instance has been constructed. Remaining obligations include:

- Construct and glue the appropriate manifold or vector-bundle objects over actual chart overlaps, with their domains, transition maps, regularity, and compatibility proofs.
- Connect an actual ambient embedded-Shore connection to the transformed curved-coordinate connection. The separate tubular-chart metric correspondence is not an ambient connection or curvature transfer theorem.
- Establish the regularity and geodesic-ODE characterization of attained AC minimizers, general second variation/conjugacy, global cut locus, and the curved-Shore functional-analytic bifurcation.
- Supply numerical frame, chart, derivative, and arithmetic error certificates before treating these exact-real transformations as verified runtime computations.

The local gauge algebra, actual inverse-frame derivative, coefficient regularity, and bundled coordinate curvature covariance are now checked. No atlas gluing, global smooth-minimizer classification, or runtime modification is asserted.
