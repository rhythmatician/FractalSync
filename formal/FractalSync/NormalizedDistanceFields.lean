import Mathlib.Analysis.Normed.Module.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

/-!
# Exact normalized-field contracts

Development-time mathematical lemmas for research issue #162. These theorems
concern real arithmetic and explicit hypotheses; they do not certify a numerical
GeometryProvider, differentiability, or finite-sample topology reconstruction.
-/

namespace FractalSync.NormalizedDistanceFields

/-- An anchored scalar difference divided by its physical scale. -/
noncomputable def normalizedDifference (sample center h : ℝ) : ℝ :=
  (sample - center) / h

/-- The continuous normalized Map field; the meaningful application has `h > 0`. -/
noncomputable def normalizedMap {E : Type*} [AddCommGroup E] [Module ℝ E]
    (D : E → ℝ) (x u : E) (h : ℝ) : ℝ :=
  normalizedDifference (D (x + h • u)) (D x) h

/-- Recentring and rescaling a continuous window is an exact chart transition. -/
theorem normalizedMap_chart_transition {E : Type*} [AddCommGroup E] [Module ℝ E]
    (D : E → ℝ) (x u v : E) (h α : ℝ) (hh : h ≠ 0) (hα : α ≠ 0) :
    normalizedMap D (x + h • v) u (α * h) =
      (normalizedMap D x (v + α • u) h - normalizedMap D x v h) / α := by
  unfold normalizedMap normalizedDifference
  simp only [smul_add, smul_smul, mul_comm h α, add_assoc]
  field_simp
  ring

/-- Provider error enters through the difference of errors at query and anchor. -/
theorem normalizedDifference_error_identity
    (sample center sampleError centerError h : ℝ) :
    normalizedDifference (sample + sampleError) (center + centerError) h -
        normalizedDifference sample center h = (sampleError - centerError) / h := by
  unfold normalizedDifference
  ring

/-- A certified anchored error increment gives the sharper fixed-scale bound. -/
theorem normalizedDifference_anchored_error_bound
    (sample center sampleError centerError h errorBound : ℝ)
    (hh : 0 < h) (he : |sampleError - centerError| ≤ errorBound) :
    |normalizedDifference (sample + sampleError) (center + centerError) h -
        normalizedDifference sample center h| ≤ errorBound / h := by
  rw [normalizedDifference_error_identity, abs_div, abs_of_pos hh]
  exact div_le_div_of_nonneg_right he (le_of_lt hh)

/-- Separate certified endpoint errors imply the usual sum-of-errors bound. -/
theorem normalizedDifference_endpoint_error_bound
    (sample center sampleError centerError h sampleBound centerBound : ℝ)
    (hh : 0 < h) (hs : |sampleError| ≤ sampleBound)
    (hc : |centerError| ≤ centerBound) :
    |normalizedDifference (sample + sampleError) (center + centerError) h -
        normalizedDifference sample center h| ≤ (sampleBound + centerBound) / h := by
  apply normalizedDifference_anchored_error_bound sample center sampleError centerError h
    (sampleBound + centerBound) hh
  simpa only [sub_eq_add_neg, abs_neg] using
    (abs_add_le sampleError (-centerError)).trans
      (add_le_add hs (by simpa only [abs_neg] using hc))

/-- A common additive field bias cancels exactly when the scale is held fixed. -/
theorem normalizedMap_add_constant {E : Type*} [AddCommGroup E] [Module ℝ E]
    (D : E → ℝ) (x u : E) (h b : ℝ) :
    normalizedMap (fun p => D p + b) x u h = normalizedMap D x u h := by
  simp [normalizedMap, normalizedDifference]

/-- Normalization preserves the ideal field's 1-Lipschitz bound in window coordinates. -/
theorem normalizedMap_lipschitz_bound {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E]
    (D : E → ℝ) (hD : LipschitzWith 1 D) (x u v : E) (h : ℝ) (hh : 0 < h) :
    |normalizedMap D x u h - normalizedMap D x v h| ≤ ‖u - v‖ := by
  have hLip := hD.norm_sub_le (x + h • u) (x + h • v)
  simp only [NNReal.coe_one, one_mul, add_sub_add_left_eq_sub, ← smul_sub,
    norm_smul, Real.norm_eq_abs, abs_of_pos hh] at hLip
  simp only [normalizedMap, normalizedDifference, div_sub_div_same,
    sub_sub_sub_cancel_right, abs_div, abs_of_pos hh]
  exact (div_le_iff₀ hh).2 (by simpa only [mul_comm h] using hLip)

/-- Every normalized Map is anchored at zero. -/
theorem normalizedMap_zero {E : Type*} [AddCommGroup E] [Module ℝ E]
    (D : E → ℝ) (x : E) (h : ℝ) : normalizedMap D x 0 h = 0 := by
  simp [normalizedMap, normalizedDifference]

/-- Values are bounded by distance from the normalized window's anchor. -/
theorem normalizedMap_value_bound {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E]
    (D : E → ℝ) (hD : LipschitzWith 1 D) (x u : E) (h : ℝ) (hh : 0 < h) :
    |normalizedMap D x u h| ≤ ‖u‖ := by
  simpa only [normalizedMap_zero, sub_zero] using
    normalizedMap_lipschitz_bound D hD x u 0 h hh

end FractalSync.NormalizedDistanceFields
