import FractalSync.MovingFrames

/-!
# Regularity of actual nonlinear coordinate connection coefficients

A C³ coordinate map and a C¹ connection field give an actually C¹ transformed
continuous-bilinear coefficient field near an invertible Jacobian. The second
chart derivative is retained, and local invertibility is derived.
-/

namespace FractalSync.CoordinateConnectionRegularity

open MovingFrames
open scoped Topology

variable {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

/-- Bundled nonlinear coordinate coefficients, obtained from the actual
Jacobian frame and the original connection with its base direction pulled back. -/
noncomputable def pullback (f : E → F) (Gamma : ConnectionForm F F) : ConnectionForm E E :=
  gauge (fderiv ℝ f) (fun x => (Gamma (f x)).comp (fderiv ℝ f x))

/-- The bundled definition is exactly the previous actual nonlinear connection law. -/
theorem pullback_apply (f : E → F) (Gamma : ConnectionForm F F) (x v w : E) :
    pullback f Gamma x v w =
      NonlinearCoordinates.pullbackConnection f (CurvatureTensoriality.coefficients Gamma) x v w := by
  rfl

/-- The old unbundled coefficient operation agrees with the new bilinear object. -/
theorem coefficients_pullback (f : E → F) (Gamma : ConnectionForm F F) :
    CurvatureTensoriality.coefficients (pullback f Gamma) =
      NonlinearCoordinates.pullbackConnection f (CurvatureTensoriality.coefficients Gamma) := by
  rfl

/-- Sufficient chart regularity genuinely supplies C¹ transformed coefficients. -/
theorem contDiffAt_pullback [CompleteSpace E] (f : E → F) (Gamma : ConnectionForm F F)
    (x : E) (hf : ContDiffAt ℝ 3 f x) (hi : (fderiv ℝ f x).IsInvertible)
    (hG : ContDiffAt ℝ 1 Gamma (f x)) :
    ContDiffAt ℝ 1 (pullback f Gamma) x := by
  have hJ : ContDiffAt ℝ 2 (fderiv ℝ f) x := hf.fderiv_right (by norm_num)
  exact contDiffAt_gauge (fderiv ℝ f) _ x hJ hi
    ((hG.comp x (hf.of_le (by norm_num))).clm_comp (hJ.of_le (by norm_num)))


/-- The hypotheses produce a genuine positive-radius neighborhood where the
actual Jacobian is invertible and transformed coefficients are C¹. -/
theorem exists_ball_invertible_contDiffAt_pullback [CompleteSpace E]
    (f : E → F) (Gamma : ConnectionForm F F) (x : E)
    (hf : ContDiffAt ℝ 3 f x) (hi : (fderiv ℝ f x).IsInvertible)
    (hG : ContDiffAt ℝ 1 Gamma (f x)) :
    ∃ delta > 0, ∀ y ∈ Metric.ball x delta,
      (fderiv ℝ f y).IsInvertible ∧ ContDiffAt ℝ 1 (pullback f Gamma) y := by
  exact Metric.eventually_nhds_iff_ball.mp
    ((NonlinearCoordinates.eventually_isInvertible_fderiv f x (hf.of_le (by norm_num)) hi).and
      ((contDiffAt_pullback f Gamma x hf hi hG).eventually (by norm_num)))

/-- Actual transformed-coordinate curvature equals its bundled trilinear
tensor on arbitrary regular fields, with coefficient regularity discharged. -/
theorem curvature_eq_tensor [CompleteSpace E]
    (f : E → F) (Gamma : ConnectionForm F F) (x : E)
    (hf : ContDiffAt ℝ 3 f x) (hi : (fderiv ℝ f x).IsInvertible)
    (hG : ContDiffAt ℝ 1 Gamma (f x))
    (X Y Z : E → E) (hX : DifferentiableAt ℝ X x) (hY : DifferentiableAt ℝ Y x)
    (hZ : ContDiffAt ℝ 2 Z x) :
    CoordinateCovariance.curvature
      (NonlinearCoordinates.pullbackConnection f (CurvatureTensoriality.coefficients Gamma))
      X Y Z x = CurvatureTensoriality.curvatureTensor (pullback f Gamma) x (X x) (Y x) (Z x) := by
  simpa only [coefficients_pullback] using
    CurvatureTensoriality.curvature_eq_tensor (pullback f Gamma) X Y Z x
      ((contDiffAt_pullback f Gamma x hf hi hG).differentiableAt le_rfl) hX hY hZ

end FractalSync.CoordinateConnectionRegularity
