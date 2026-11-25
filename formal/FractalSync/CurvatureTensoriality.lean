import FractalSync.CoordinateCovariance

/-!
# Actual curvature tensoriality in a coordinate space

Connection coefficients are a field of continuous bilinear maps. Expanding the
actual curvature commutator cancels all derivatives of the input vector fields.
The resulting tensor therefore depends only on their values at the point.
-/

namespace FractalSync.CurvatureTensoriality

open CoordinateCovariance
open scoped Topology

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Unbundled coefficients of a field of continuous bilinear connection maps. -/
def coefficients (Gamma : E → E →L[ℝ] E →L[ℝ] E) (x v w : E) : E := Gamma x v w

/-- The pointwise curvature expression from the actual connection derivative. -/
noncomputable def curvatureValue (Gamma : E → E →L[ℝ] E →L[ℝ] E)
    (x u v w : E) : E :=
  fderiv ℝ Gamma x u v w - fderiv ℝ Gamma x v u w +
    Gamma x u (Gamma x v w) - Gamma x v (Gamma x u w)

/-- Genuine differentiability of the covariant derivative follows from the
operator-valued coefficient derivative and the second derivative of its field. -/
theorem differentiableAt_covariantDerivative
    (Gamma : E → E →L[ℝ] E →L[ℝ] E) (Y Z : E → E) (x : E)
    (hG : DifferentiableAt ℝ Gamma x) (hY : DifferentiableAt ℝ Y x)
    (hZ : DifferentiableAt ℝ Z x) (hDZ : DifferentiableAt ℝ (fderiv ℝ Z) x) :
    DifferentiableAt ℝ (covariantDerivative (coefficients Gamma) Y Z) x := by
  exact (hDZ.clm_apply hY).add ((hG.clm_apply hY).clm_apply hZ)

/-- Actual derivative of the covariant derivative, with every field derivative retained. -/
theorem fderiv_covariantDerivative_apply
    (Gamma : E → E →L[ℝ] E →L[ℝ] E) (Y Z : E → E) (x u : E)
    (hG : DifferentiableAt ℝ Gamma x) (hY : DifferentiableAt ℝ Y x)
    (hZ : DifferentiableAt ℝ Z x) (hDZ : DifferentiableAt ℝ (fderiv ℝ Z) x) :
    fderiv ℝ (covariantDerivative (coefficients Gamma) Y Z) x u =
      fderiv ℝ (fderiv ℝ Z) x u (Y x) + fderiv ℝ Z x (fderiv ℝ Y x u) +
        fderiv ℝ Gamma x u (Y x) (Z x) + Gamma x (fderiv ℝ Y x u) (Z x) +
          Gamma x (Y x) (fderiv ℝ Z x u) := by
  change fderiv ℝ (fun y => (fderiv ℝ Z y) (Y y) + (Gamma y (Y y)) (Z y)) x u = _
  rw [fderiv_fun_add (hDZ.clm_apply hY) ((hG.clm_apply hY).clm_apply hZ),
    fderiv_clm_apply hDZ hY, fderiv_clm_apply (hG.clm_apply hY) hZ,
    fderiv_clm_apply hG hY]
  simp only [ContinuousLinearMap.add_apply, ContinuousLinearMap.comp_apply,
    ContinuousLinearMap.flip_apply]
  abel


/-- Actual curvature depends only on the three vector values at the point.
The second derivatives of Z cancel by the actual symmetry theorem, and all
first derivatives of X, Y, Z cancel against the Lie bracket and connection terms. -/
theorem curvature_eq_curvatureValue
    (Gamma : E → E →L[ℝ] E →L[ℝ] E) (X Y Z : E → E) (x : E)
    (hG : DifferentiableAt ℝ Gamma x) (hX : DifferentiableAt ℝ X x)
    (hY : DifferentiableAt ℝ Y x) (hZ : ContDiffAt ℝ 2 Z x) :
    curvature (coefficients Gamma) X Y Z x =
      curvatureValue Gamma x (X x) (Y x) (Z x) := by
  have hDZ : DifferentiableAt ℝ (fderiv ℝ Z) x :=
    (hZ.fderiv_right (m := 1) (by norm_num)).differentiableAt le_rfl
  have hYZ := fderiv_covariantDerivative_apply Gamma Y Z x (X x) hG hY
    (hZ.differentiableAt one_le_two) hDZ
  have hXZ := fderiv_covariantDerivative_apply Gamma X Z x (Y x) hG hX
    (hZ.differentiableAt one_le_two) hDZ
  simp only [curvature, covariantDerivative, coefficients, VectorField.lieBracket,
    curvatureValue] at hYZ hXZ ⊢
  rw [hYZ, hXZ, (hZ.isSymmSndFDerivAt (by norm_num)).eq (X x) (Y x)]
  simp only [map_add, map_sub, ContinuousLinearMap.sub_apply]
  abel


/-- Composition of the connection's endomorphisms, bundled in all three arguments. -/
noncomputable def compositionTensor (G : E →L[ℝ] E →L[ℝ] E) : E →L[ℝ] E →L[ℝ] E →L[ℝ] E :=
  (((ContinuousLinearMap.compL ℝ E E E).comp G).flip.comp G).flip

/-- The coordinate curvature is a genuine continuous trilinear map, obtained
from the actual derivative of the continuous-bilinear connection field. -/
noncomputable def curvatureTensor (Gamma : E → E →L[ℝ] E →L[ℝ] E) (x : E) :
    E →L[ℝ] E →L[ℝ] E →L[ℝ] E :=
  fderiv ℝ Gamma x - (fderiv ℝ Gamma x).flip +
    compositionTensor (Gamma x) - (compositionTensor (Gamma x)).flip

/-- Evaluation of the bundled tensor gives the connection/derivative formula. -/
theorem curvatureTensor_apply (Gamma : E → E →L[ℝ] E →L[ℝ] E) (x u v w : E) :
    curvatureTensor Gamma x u v w = curvatureValue Gamma x u v w := by
  simp [curvatureTensor, curvatureValue, compositionTensor]


/-- The bundled continuous trilinear tensor is the actual curvature commutator. -/
theorem curvature_eq_tensor
    (Gamma : E → E →L[ℝ] E →L[ℝ] E) (X Y Z : E → E) (x : E)
    (hG : DifferentiableAt ℝ Gamma x) (hX : DifferentiableAt ℝ X x)
    (hY : DifferentiableAt ℝ Y x) (hZ : ContDiffAt ℝ 2 Z x) :
    curvature (coefficients Gamma) X Y Z x =
      curvatureTensor Gamma x (X x) (Y x) (Z x) := by
  rw [curvatureTensor_apply, curvature_eq_curvatureValue Gamma X Y Z x hG hX hY hZ]


/-- Arbitrary regular vector-field extensions with the same point values
produce the same actual curvature, even when their derivatives differ. -/
theorem curvature_congr_values
    (Gamma : E → E →L[ℝ] E →L[ℝ] E) (X Y Z X' Y' Z' : E → E) (x : E)
    (hG : DifferentiableAt ℝ Gamma x)
    (hX : DifferentiableAt ℝ X x) (hY : DifferentiableAt ℝ Y x)
    (hZ : ContDiffAt ℝ 2 Z x)
    (hX' : DifferentiableAt ℝ X' x) (hY' : DifferentiableAt ℝ Y' x)
    (hZ' : ContDiffAt ℝ 2 Z' x)
    (eX : X x = X' x) (eY : Y x = Y' x) (eZ : Z x = Z' x) :
    curvature (coefficients Gamma) X Y Z x = curvature (coefficients Gamma) X' Y' Z' x := by
  rw [curvature_eq_tensor Gamma X Y Z x hG hX hY hZ,
    curvature_eq_tensor Gamma X' Y' Z' x hG hX' hY' hZ', eX, eY, eZ]


/-- The continuous curvature tensor is alternating in its first two arguments. -/
theorem curvatureTensor_swap (Gamma : E → E →L[ℝ] E →L[ℝ] E) (x u v w : E) :
    curvatureTensor Gamma x u v w = -curvatureTensor Gamma x v u w := by
  simp only [curvatureTensor_apply, curvatureValue]
  abel

end FractalSync.CurvatureTensoriality
