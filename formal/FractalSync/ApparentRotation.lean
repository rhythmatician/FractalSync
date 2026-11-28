import Mathlib.Analysis.SpecialFunctions.Complex.LogDeriv

/-!
# Apparent turn of a marked landmark (#259)

The parameter is real-differentiable and the output is a relative complex
landmark position. This makes no regularity assertion about Julia rendering.
-/

namespace FractalSync.ApparentRotation

noncomputable section

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The imaginary part of the relative logarithmic differential. -/
def turnDifferential (w : ℂ) (dw : E →L[ℝ] ℂ) : E →L[ℝ] ℝ :=
  Complex.imCLM.comp (w⁻¹ • dw)

theorem turnDifferential_apply (w : ℂ) (dw : E →L[ℝ] ℂ) (v : E) :
    turnDifferential w dw v = (dw v / w).im := by
  simp [turnDifferential, div_eq_inv_mul, smul_eq_mul]

/-- Shorter relative landmarks can amplify the same velocity error. -/
theorem abs_turnDifferential_le (w : ℂ) (dw : E →L[ℝ] ℂ) (v : E) :
    |turnDifferential w dw v| ≤ ‖dw v‖ / ‖w‖ := by
  simpa [turnDifferential_apply, norm_div] using Complex.abs_im_le_norm (dw v / w)

/-- The branch of relative angle centered at the base landmark vector. -/
def relativeAngle (base : ℂ) (w : ℂ) : ℝ :=
  Complex.arg (base⁻¹ * w)

theorem hasFDerivAt_relativeAngle {w : E → ℂ} {dw : E →L[ℝ] ℂ} {c : E}
    (h : HasFDerivAt w dw c) (hn : w c ≠ 0) :
    HasFDerivAt (fun x => relativeAngle (w c) (w x)) (turnDifferential (w c) dw) c := by
  have hn1 : (w c)⁻¹ * w c = 1 := inv_mul_cancel₀ hn
  have hslit : (w c)⁻¹ * w c ∈ Complex.slitPlane := hn1 ▸ Complex.one_mem_slitPlane
  have hlog := (Complex.hasStrictFDerivAt_log_real hslit).hasFDerivAt.comp c
    (h.const_smul (w c)⁻¹)
  simpa [relativeAngle, turnDifferential, hn1, Function.comp_def, Complex.log_im, smul_eq_mul]
    using Complex.imCLM.hasFDerivAt.comp c hlog

end

end FractalSync.ApparentRotation
