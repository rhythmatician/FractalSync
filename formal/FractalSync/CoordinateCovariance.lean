import FractalSync.FlatShoreGeometry
import Mathlib.Analysis.Calculus.FDeriv.Comp

/-!
# Actual affine coordinate covariance

A constant continuous linear equivalence and a translation transport vector
fields, a bilinear metric's coefficients, and connection coefficients. The
covariant differential and curvature commutator are recomputed from actual
Fréchet derivatives. This module proves their affine covariance by chain rule.
It does not treat nonlinear charts, whose connection transformation includes
second derivatives, or introduce a bundled manifold connection/tensor API.
-/

namespace FractalSync.CoordinateCovariance

variable {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

/-- A constant invertible linear coordinate change followed by translation. -/
def affineMap (A : F ≃L[ℝ] E) (b : E) (x : F) : E := A x + b

/-- Coordinate components of an actual vector field in the new affine chart. -/
noncomputable def pullbackField (A : F ≃L[ℝ] E) (b : E) (X : E → E) (x : F) : F :=
  A.symm (X (affineMap A b x))

/-- The affine transformation of the connection coefficient operation. -/
noncomputable def pullbackConnection (A : F ≃L[ℝ] E) (b : E)
    (Gamma : E → E → E → E) (x v w : F) : F :=
  A.symm (Gamma (affineMap A b x) (A v) (A w))

/-- The pullback of a bilinear metric, evaluated on actual vectors. -/
def pullbackMetric (A : F ≃L[ℝ] E) (b : E) (g : E → E → E → ℝ)
    (x v w : F) : ℝ := g (affineMap A b x) (A v) (A w)

/-- Coordinate covariant differentiation, using the actual field derivative. -/
noncomputable def covariantDerivative (Gamma : E → E → E → E)
    (X Y : E → E) (x : E) : E := fderiv ℝ Y x (X x) + Gamma x (X x) (Y x)

/-- The actual coordinate curvature commutator. -/
noncomputable def curvature (Gamma : E → E → E → E) (X Y Z : E → E) (x : E) : E :=
  covariantDerivative Gamma X (covariantDerivative Gamma Y Z) x -
    covariantDerivative Gamma Y (covariantDerivative Gamma X Z) x -
    covariantDerivative Gamma (VectorField.lieBracket ℝ X Y) Z x

/-- The coordinate map has its constant linear part as actual derivative. -/
theorem hasFDerivAt_affineMap (A : F ≃L[ℝ] E) (b : E) (x : F) :
    HasFDerivAt (affineMap A b) A.toContinuousLinearMap x := by
  exact A.hasFDerivAt.add_const b

/-- Actual derivative of a transported field, with both linear factors. -/
theorem hasFDerivAt_pullbackField (A : F ≃L[ℝ] E) (b : E) (Y : E → E) (x : F)
    (D : E →L[ℝ] E) (hY : HasFDerivAt Y D (affineMap A b x)) :
    HasFDerivAt (pullbackField A b Y)
      (A.symm.toContinuousLinearMap.comp (D.comp A.toContinuousLinearMap)) x := by
  exact A.symm.hasFDerivAt.comp x (hY.comp x (hasFDerivAt_affineMap A b x))

/-- The transported field's derivative is obtained by conjugating its actual derivative. -/
theorem fderiv_pullbackField (A : F ≃L[ℝ] E) (b : E) (Y : E → E) (x : F)
    (hY : DifferentiableAt ℝ Y (affineMap A b x)) :
    fderiv ℝ (pullbackField A b Y) x =
      A.symm.toContinuousLinearMap.comp ((fderiv ℝ Y (affineMap A b x)).comp A.toContinuousLinearMap) := by
  exact (hasFDerivAt_pullbackField A b Y x _ hY.hasFDerivAt).fderiv

/-- Affine connection covariance follows from the actual chain rule. -/
theorem covariantDerivative_pullback (A : F ≃L[ℝ] E) (b : E)
    (Gamma : E → E → E → E) (X Y : E → E) (x : F)
    (hY : DifferentiableAt ℝ Y (affineMap A b x)) :
    covariantDerivative (pullbackConnection A b Gamma) (pullbackField A b X)
      (pullbackField A b Y) x =
      A.symm (covariantDerivative Gamma X Y (affineMap A b x)) := by
  rw [covariantDerivative, fderiv_pullbackField A b Y x hY]
  simp [pullbackField, pullbackConnection, covariantDerivative]

/-- The Lie bracket is recomputed using both transported field derivatives. -/
theorem lieBracket_pullback (A : F ≃L[ℝ] E) (b : E) (X Y : E → E) (x : F)
    (hX : DifferentiableAt ℝ X (affineMap A b x))
    (hY : DifferentiableAt ℝ Y (affineMap A b x)) :
    VectorField.lieBracket ℝ (pullbackField A b X) (pullbackField A b Y) x =
      A.symm (VectorField.lieBracket ℝ X Y (affineMap A b x)) := by
  rw [VectorField.lieBracket, fderiv_pullbackField A b Y x hY,
    fderiv_pullbackField A b X x hX]
  simp [pullbackField, VectorField.lieBracket]

/-- Pairings of transported fields agree in the actual pulled-back metric. -/
theorem pullbackMetric_fields (A : F ≃L[ℝ] E) (b : E) (g : E → E → E → ℝ)
    (X Y : E → E) (x : F) :
    pullbackMetric A b g x (pullbackField A b X x) (pullbackField A b Y x) =
      g (affineMap A b x) (X (affineMap A b x)) (Y (affineMap A b x)) := by
  simp [pullbackMetric, pullbackField]

/-- The bracket-direction term in the curvature commutator also transforms. -/
theorem covariantDerivative_lieBracket_pullback (A : F ≃L[ℝ] E) (b : E)
    (Gamma : E → E → E → E) (X Y Z : E → E) (x : F)
    (hX : DifferentiableAt ℝ X (affineMap A b x))
    (hY : DifferentiableAt ℝ Y (affineMap A b x))
    (hZ : DifferentiableAt ℝ Z (affineMap A b x)) :
    covariantDerivative (pullbackConnection A b Gamma)
      (VectorField.lieBracket ℝ (pullbackField A b X) (pullbackField A b Y))
      (pullbackField A b Z) x =
      A.symm (covariantDerivative Gamma (VectorField.lieBracket ℝ X Y) Z (affineMap A b x)) := by
  rw [covariantDerivative, fderiv_pullbackField A b Z x hZ,
    lieBracket_pullback A b X Y x hX hY]
  simp [pullbackConnection, pullbackField, covariantDerivative]

/-- The actual curvature commutator is conjugate under an affine chart change.
Global differentiability of Z supplies equality of the inner derivative fields;
the two resulting covariant derivative fields must be differentiable at the point. -/
theorem curvature_pullback (A : F ≃L[ℝ] E) (b : E)
    (Gamma : E → E → E → E) (X Y Z : E → E) (x : F)
    (hX : DifferentiableAt ℝ X (affineMap A b x))
    (hY : DifferentiableAt ℝ Y (affineMap A b x)) (hZ : Differentiable ℝ Z)
    (hYZ : DifferentiableAt ℝ (covariantDerivative Gamma Y Z) (affineMap A b x))
    (hXZ : DifferentiableAt ℝ (covariantDerivative Gamma X Z) (affineMap A b x)) :
    curvature (pullbackConnection A b Gamma) (pullbackField A b X) (pullbackField A b Y)
      (pullbackField A b Z) x = A.symm (curvature Gamma X Y Z (affineMap A b x)) := by
  have hinner (V : E → E) :
      covariantDerivative (pullbackConnection A b Gamma) (pullbackField A b V)
        (pullbackField A b Z) = pullbackField A b (covariantDerivative Gamma V Z) :=
    funext fun z => covariantDerivative_pullback A b Gamma V Z z (hZ (affineMap A b z))
  rw [curvature, hinner Y, hinner X,
    covariantDerivative_pullback A b Gamma X (covariantDerivative Gamma Y Z) x hYZ,
    covariantDerivative_pullback A b Gamma Y (covariantDerivative Gamma X Z) x hXZ,
    covariantDerivative_lieBracket_pullback A b Gamma X Y Z x hX hY (hZ _)]
  simp only [curvature, map_sub]

/-- Positive definiteness survives an invertible affine change of coordinates. -/
theorem pullbackMetric_pos (A : F ≃L[ℝ] E) (b : E) (g : E → E → E → ℝ)
    (hg : ∀ x v, v ≠ 0 → 0 < g x v v) (x v : F) (hv : v ≠ 0) :
    0 < pullbackMetric A b g x v v := by
  exact hg _ _ (fun h => hv (A.injective (h.trans (map_zero A).symm)))

/-- The metric Gram expression of two vectors. -/
def gram (g : E → E → E → ℝ) (x v w : E) : ℝ :=
  g x v v * g x w w - (g x v w) ^ 2

/-- Gram expressions agree for transported fields, hence nondegeneracy is preserved. -/
theorem gram_pullback (A : F ≃L[ℝ] E) (b : E) (g : E → E → E → ℝ)
    (X Y : E → E) (x : F) :
    gram (pullbackMetric A b g) x (pullbackField A b X x) (pullbackField A b Y x) =
      gram g (affineMap A b x) (X (affineMap A b x)) (Y (affineMap A b x)) := by
  simp only [gram, pullbackMetric_fields]

/-- The curvature/Gram ratio for a coordinate plane. Its geometric
interpretation uses a genuine metric, connection, and nonzero Gram denominator. -/
noncomputable def gramRatio (g : E → E → E → ℝ) (Gamma : E → E → E → E)
    (X Y : E → E) (x : E) : ℝ :=
  g x (curvature Gamma X Y Y x) (X x) / gram g x (X x) (Y x)

/-- The actual curvature/Gram ratio is invariant under affine coordinate change. -/
theorem gramRatio_pullback (A : F ≃L[ℝ] E) (b : E)
    (g : E → E → E → ℝ) (Gamma : E → E → E → E) (X Y : E → E) (x : F)
    (hX : DifferentiableAt ℝ X (affineMap A b x)) (hY : Differentiable ℝ Y)
    (hYY : DifferentiableAt ℝ (covariantDerivative Gamma Y Y) (affineMap A b x))
    (hXY : DifferentiableAt ℝ (covariantDerivative Gamma X Y) (affineMap A b x)) :
    gramRatio (pullbackMetric A b g) (pullbackConnection A b Gamma)
      (pullbackField A b X) (pullbackField A b Y) x =
      gramRatio g Gamma X Y (affineMap A b x) := by
  rw [gramRatio, curvature_pullback A b Gamma X Y Y x hX (hY _) hY hYY hXY,
    gram_pullback]
  simp only [pullbackMetric, pullbackField, ContinuousLinearEquiv.apply_symm_apply, gramRatio]

/-- The generic coordinate construction specializes definitionally to the checked flat connection. -/
theorem flat_covariantDerivative_eq (epsilon a : ℝ)
    (X Y : FlatShoreGeometry.Coord → FlatShoreGeometry.Coord) :
    covariantDerivative (fun x => FlatShoreGeometry.christoffel epsilon a x.1) X Y =
      FlatShoreGeometry.covariantDerivative epsilon a X Y := by
  rfl

/-- The generic commutator is the same actual flat curvature commutator. -/
theorem flat_curvature_eq (epsilon a : ℝ)
    (X Y Z : FlatShoreGeometry.Coord → FlatShoreGeometry.Coord) :
    curvature (fun x => FlatShoreGeometry.christoffel epsilon a x.1) X Y Z =
      FlatShoreGeometry.curvature epsilon a X Y Z := by
  rfl

/-- The transported flat coordinate plane has a strictly positive Gram denominator. -/
theorem flat_gram_pullback_pos (A : F ≃L[ℝ] FlatShoreGeometry.Coord)
    (b : FlatShoreGeometry.Coord) (epsilon a : ℝ) (he : 0 < epsilon) (x : F) :
    0 < gram (pullbackMetric A b (fun p => FlatShoreGeometry.metric epsilon a p.1)) x
      (pullbackField A b FlatShoreGeometry.normalField x)
      (pullbackField A b FlatShoreGeometry.tangentField x) := by
  rw [gram_pullback]
  simpa [gram, FlatShoreGeometry.metric, FlatShoreGeometry.normalField,
    FlatShoreGeometry.tangentField] using
    mul_pos (FlatShoreGeometry.radialMetric_pos epsilon a (affineMap A b x).1 he)
      (FlatShoreGeometry.tangentialMetric_pos epsilon (affineMap A b x).1 he)

/-- The previously checked flat Gaussian ratio is invariant in every affine chart. -/
theorem flat_gramRatio_pullback (A : F ≃L[ℝ] FlatShoreGeometry.Coord)
    (b : FlatShoreGeometry.Coord) (epsilon a : ℝ) (he : 0 < epsilon) (x : F) :
    gramRatio
      (pullbackMetric A b (fun p => FlatShoreGeometry.metric epsilon a p.1))
      (pullbackConnection A b (fun p => FlatShoreGeometry.christoffel epsilon a p.1))
      (pullbackField A b FlatShoreGeometry.normalField)
      (pullbackField A b FlatShoreGeometry.tangentField) x =
      FlatShore.radialCurvature epsilon a (affineMap A b x).1 := by
  apply Eq.trans (gramRatio_pullback A b
    (fun p => FlatShoreGeometry.metric epsilon a p.1)
    (fun p => FlatShoreGeometry.christoffel epsilon a p.1)
    FlatShoreGeometry.normalField FlatShoreGeometry.tangentField x ?_ ?_ ?_ ?_)
  · exact FlatShoreGeometry.sectionalCurvature_eq_radialCurvature epsilon a (affineMap A b x) he
  · exact differentiableAt_const _
  · exact differentiable_const _
  · rw [flat_covariantDerivative_eq, FlatShoreGeometry.covariantDerivative_tangent_tangent]
    exact (((FlatShoreGeometry.hasDerivAt_gammaYY epsilon a (affineMap A b x).1 he).comp_hasFDerivAt
      (affineMap A b x) (ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt).prodMk
        (hasFDerivAt_const (0 : ℝ) (affineMap A b x))).differentiableAt
  · rw [flat_covariantDerivative_eq, FlatShoreGeometry.covariantDerivative_normal_tangent]
    exact ((hasFDerivAt_const (0 : ℝ) (affineMap A b x)).prodMk
      ((FlatShoreGeometry.hasDerivAt_gammaRY epsilon (affineMap A b x).1 he).comp_hasFDerivAt
        (affineMap A b x) (ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt)).differentiableAt

end FractalSync.CoordinateCovariance
