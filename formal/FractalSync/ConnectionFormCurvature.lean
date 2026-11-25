import FractalSync.MovingFrames

/-!
# Actual curvature tensors for coordinate connection forms

The base space and the section fiber may differ. The actual curvature
commutator is a continuous trilinear map in two base directions and one fiber
value. Moving-frame covariance is proved for this bundled tensor.
-/

namespace FractalSync.ConnectionFormCurvature

open MovingFrames
open scoped Topology

variable {E F G : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]
  [NormedAddCommGroup G] [NormedSpace ℝ G]

/-- Pointwise expression in the actual connection-form derivative. -/
noncomputable def curvatureValue (Gamma : ConnectionForm E F) (x u v : E) (w : F) : F :=
  fderiv ℝ Gamma x u v w - fderiv ℝ Gamma x v u w +
    Gamma x u (Gamma x v w) - Gamma x v (Gamma x u w)

/-- Actual derivative of the covariant derivative of a fiber-valued section. -/
theorem fderiv_covariantDerivative_apply (Gamma : ConnectionForm E F)
    (Y : E → E) (Z : E → F) (x u : E)
    (hG : DifferentiableAt ℝ Gamma x) (hY : DifferentiableAt ℝ Y x)
    (hZ : DifferentiableAt ℝ Z x) (hDZ : DifferentiableAt ℝ (fderiv ℝ Z) x) :
    fderiv ℝ (covariantDerivative Gamma Y Z) x u =
      fderiv ℝ (fderiv ℝ Z) x u (Y x) + fderiv ℝ Z x (fderiv ℝ Y x u) +
        fderiv ℝ Gamma x u (Y x) (Z x) + Gamma x (fderiv ℝ Y x u) (Z x) +
          Gamma x (Y x) (fderiv ℝ Z x u) := by
  change fderiv ℝ (fun y => fderiv ℝ Z y (Y y) + Gamma y (Y y) (Z y)) x u = _
  rw [fderiv_fun_add (hDZ.clm_apply hY) ((hG.clm_apply hY).clm_apply hZ),
    fderiv_clm_apply hDZ hY, fderiv_clm_apply (hG.clm_apply hY) hZ,
    fderiv_clm_apply hG hY]
  simp only [ContinuousLinearMap.add_apply, ContinuousLinearMap.comp_apply,
    ContinuousLinearMap.flip_apply]
  abel

/-- All section-extension derivatives cancel from the actual commutator. -/
theorem curvature_eq_curvatureValue (Gamma : ConnectionForm E F)
    (X Y : E → E) (Z : E → F) (x : E)
    (hG : DifferentiableAt ℝ Gamma x) (hX : DifferentiableAt ℝ X x)
    (hY : DifferentiableAt ℝ Y x) (hZ : ContDiffAt ℝ 2 Z x) :
    curvature Gamma X Y Z x = curvatureValue Gamma x (X x) (Y x) (Z x) := by
  have hDZ : DifferentiableAt ℝ (fderiv ℝ Z) x :=
    (hZ.fderiv_right (m := 1) (by norm_num)).differentiableAt le_rfl
  have hYZ := fderiv_covariantDerivative_apply Gamma Y Z x (X x) hG hY
    (hZ.differentiableAt one_le_two) hDZ
  have hXZ := fderiv_covariantDerivative_apply Gamma X Z x (Y x) hG hX
    (hZ.differentiableAt one_le_two) hDZ
  simp only [curvature, covariantDerivative, VectorField.lieBracket, curvatureValue] at hYZ hXZ ⊢
  rw [hYZ, hXZ, (hZ.isSymmSndFDerivAt (by norm_num)).eq (X x) (Y x)]
  simp only [map_add, map_sub, ContinuousLinearMap.sub_apply]
  abel


/-- Connection endomorphism composition, continuous bilinear in the base directions. -/
noncomputable def compositionTensor (G : E →L[ℝ] F →L[ℝ] F) :
    E →L[ℝ] E →L[ℝ] F →L[ℝ] F :=
  (((ContinuousLinearMap.compL ℝ F F F).comp G).flip.comp G).flip

/-- A genuine continuous trilinear curvature tensor for the coordinate fiber. -/
noncomputable def curvatureTensor (Gamma : ConnectionForm E F) (x : E) :
    E →L[ℝ] E →L[ℝ] F →L[ℝ] F :=
  fderiv ℝ Gamma x - (fderiv ℝ Gamma x).flip +
    compositionTensor (Gamma x) - (compositionTensor (Gamma x)).flip

/-- The bundled tensor evaluates to the actual derivative/coefficient expression. -/
theorem curvatureTensor_apply (Gamma : ConnectionForm E F) (x u v : E) (w : F) :
    curvatureTensor Gamma x u v w = curvatureValue Gamma x u v w := by
  simp [curvatureTensor, curvatureValue, compositionTensor]


/-- The actual commutator is the bundled tensor on the three point values. -/
theorem curvature_eq_tensor (Gamma : ConnectionForm E F)
    (X Y : E → E) (Z : E → F) (x : E)
    (hG : DifferentiableAt ℝ Gamma x) (hX : DifferentiableAt ℝ X x)
    (hY : DifferentiableAt ℝ Y x) (hZ : ContDiffAt ℝ 2 Z x) :
    curvature Gamma X Y Z x = curvatureTensor Gamma x (X x) (Y x) (Z x) := by
  rw [curvatureTensor_apply, curvature_eq_curvatureValue Gamma X Y Z x hG hX hY hZ]


/-- The intermediate covariant section is actually differentiable under the
stated coefficient and section regularity. -/
theorem differentiableAt_covariantDerivative (Gamma : ConnectionForm E F)
    (X : E → E) (Z : E → F) (x : E)
    (hG : DifferentiableAt ℝ Gamma x) (hX : DifferentiableAt ℝ X x)
    (hZ : ContDiffAt ℝ 2 Z x) :
    DifferentiableAt ℝ (covariantDerivative Gamma X Z) x := by
  exact (((hZ.fderiv_right (m := 1) (by norm_num)).differentiableAt le_rfl).clm_apply hX).add
    ((hG.clm_apply hX).clm_apply (hZ.differentiableAt one_le_two))


/-- The actual curvature tensor is conjugated by a C² moving frame. All
intermediate differentiability and field-extension hypotheses are discharged. -/
theorem curvatureTensor_gauge [CompleteSpace F]
    (P : E → F →L[ℝ] G) (Gamma : ConnectionForm E G) (x u v : E) (w : F)
    (hP : ContDiffAt ℝ 2 P x) (hi : (P x).IsInvertible)
    (hG : ContDiffAt ℝ 1 Gamma x) :
    curvatureTensor (gauge P Gamma) x u v w =
      (P x).inverse (curvatureTensor Gamma x u v (P x w)) := by
  have hcov := MovingFrames.curvature_gauge P Gamma (fun _ => u) (fun _ => v)
    (fun _ => P x w) x (hP.of_le (by norm_num)) hi
    (Filter.Eventually.of_forall fun _ => differentiableAt_const _)
    (differentiableAt_covariantDerivative Gamma (fun _ => v) (fun _ => P x w) x
      (hG.differentiableAt le_rfl) (differentiableAt_const _) contDiffAt_const)
    (differentiableAt_covariantDerivative Gamma (fun _ => u) (fun _ => P x w) x
      (hG.differentiableAt le_rfl) (differentiableAt_const _) contDiffAt_const)
  rw [curvature_eq_tensor (gauge P Gamma) (fun _ => u) (fun _ => v)
    (inFrame P (fun _ => P x w)) x
    ((contDiffAt_gauge P Gamma x hP hi hG).differentiableAt le_rfl)
    (differentiableAt_const _) (differentiableAt_const _)
    (contDiffAt_inFrame P (fun _ => P x w) x 2 hP hi contDiffAt_const)] at hcov
  simp only [inFrame] at hcov
  rw [curvature_eq_tensor Gamma (fun _ => u) (fun _ => v) (fun _ => P x w) x
    (hG.differentiableAt le_rfl) (differentiableAt_const _)
    (differentiableAt_const _) contDiffAt_const] at hcov
  simpa only [show (P x).inverse (P x w) = w from hi.inverse_apply_eq.mpr rfl] using hcov


/-- When the fiber is the coordinate tangent space, the actual commutator is
exactly the pre-existing tangent-coordinate definition. -/
theorem curvature_eq_coordinateCurvature (Gamma : ConnectionForm E E)
    (X Y Z : E → E) :
    curvature Gamma X Y Z =
      CoordinateCovariance.curvature (CurvatureTensoriality.coefficients Gamma) X Y Z := by
  rfl


/-- The fiber-valued tensor specializes definitionally to the existing tangent tensor. -/
theorem curvatureTensor_eq_coordinateTensor (Gamma : ConnectionForm E E) (x : E) :
    curvatureTensor Gamma x = CurvatureTensoriality.curvatureTensor Gamma x := by
  rfl

end FractalSync.ConnectionFormCurvature
