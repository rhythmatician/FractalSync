import FractalSync.RadialProfile
import Mathlib.Analysis.Calculus.MeanValue

/-! Global and optimal Lipschitz constants for the actual radial integral. -/

namespace FractalSync.DistanceGraph

open scoped NNReal

theorem radialDensity_normalize (epsilon a r : ℝ) (he : 0 < epsilon) :
    radialDensity epsilon a r = radialDensity 1 a (r / epsilon) / epsilon := by
  apply (eq_div_iff he.ne').2
  simpa only [mul_one, mul_div_cancel₀ r he.ne', mul_comm] using
    radialDensity_scale 1 a (r / epsilon) epsilon zero_lt_one he

theorem radialDensity_uniform_bound (epsilon a r : ℝ) (he : 0 < epsilon) :
    radialDensity epsilon a r ≤ Real.sqrt (1 + a ^ 2) / epsilon := by
  rw [radialDensity_normalize epsilon a r he]
  apply div_le_div_of_nonneg_right _ he.le
  exact (radialDensity_comparison 1 a (r / epsilon) zero_lt_one).2.trans
    (by simpa only [mul_one] using (mul_le_mul_of_nonneg_left
      (radialDensity_le_one 0 (r / epsilon) (by norm_num)) (Real.sqrt_nonneg _)))

/-- A convenient global bound, valid for every real graph parameter. -/
theorem radialProfile_lipschitz (epsilon a : ℝ) (he : 0 < epsilon) :
    LipschitzWith (Real.toNNReal (Real.sqrt (1 + a ^ 2) / epsilon))
      (radialProfile epsilon a) := by
  apply lipschitzWith_of_nnnorm_deriv_le
    (fun r => (hasDerivAt_radialProfile epsilon a r he).differentiableAt)
  intro r
  rw [(hasDerivAt_radialProfile epsilon a r he).deriv]
  change |radialDensity epsilon a r| ≤
    (Real.toNNReal (Real.sqrt (1 + a ^ 2) / epsilon) : ℝ)
  rw [Real.coe_toNNReal _ (by positivity), abs_of_pos (radialDensity_pos epsilon a r he)]
  exact radialDensity_uniform_bound epsilon a r he

/-- The maximum normalized radial slope for the physical convention `a ≥ 0`. -/
noncomputable def radialMaxSlope (a : ℝ) : ℝ :=
  if a ≤ 1 then 1 else (1 + a ^ 2) / (2 * a)

theorem radialMaxSlope_pos (a : ℝ) : 0 < radialMaxSlope a := by
  unfold radialMaxSlope
  split_ifs with h
  · norm_num
  · exact div_pos (by positivity) (mul_pos (by norm_num) (by linarith))

theorem radialDensity_le_radialMaxSlope (a q : ℝ) (ha : 0 ≤ a) :
    radialDensity 1 a q ≤ radialMaxSlope a := by
  unfold radialMaxSlope
  split_ifs with h
  · exact radialDensity_le_one a q (by nlinarith)
  · exact radialDensity_le_peak a q (by linarith)

theorem radialMaxSlope_attained (a : ℝ) :
    ∃ q : ℝ, 0 ≤ q ∧ radialDensity 1 a q = radialMaxSlope a := by
  by_cases h : a ≤ 1
  · exact ⟨0, le_rfl, by simpa only [radialMaxSlope, if_pos h] using
      radialDensity_one_at_zero a⟩
  · exact ⟨Real.sqrt ((a ^ 2 - 1) / (1 + a ^ 2)), Real.sqrt_nonneg _,
      by simpa only [radialMaxSlope, if_neg h] using
        radialDensity_peak_attained a (by linarith)⟩

/-- The sharp global bound obtained from the attained derivative maximum. -/
theorem radialProfile_lipschitz_sharp (epsilon a : ℝ)
    (he : 0 < epsilon) (ha : 0 ≤ a) :
    LipschitzWith (Real.toNNReal (radialMaxSlope a / epsilon))
      (radialProfile epsilon a) := by
  apply lipschitzWith_of_nnnorm_deriv_le
    (fun r => (hasDerivAt_radialProfile epsilon a r he).differentiableAt)
  intro r
  rw [(hasDerivAt_radialProfile epsilon a r he).deriv]
  change |radialDensity epsilon a r| ≤ (Real.toNNReal (radialMaxSlope a / epsilon) : ℝ)
  rw [Real.coe_toNNReal _ (div_nonneg (radialMaxSlope_pos a).le he.le),
    abs_of_pos (radialDensity_pos epsilon a r he), radialDensity_normalize epsilon a r he]
  exact div_le_div_of_nonneg_right (radialDensity_le_radialMaxSlope a (r / epsilon) ha) he.le

/-- No global Lipschitz constant can be smaller than the attained slope.
Together with `radialProfile_lipschitz_sharp`, this is optimality for `a ≥ 0`. -/
theorem radialProfile_lipschitz_constant_ge (epsilon a : ℝ) (he : 0 < epsilon)
    (C : ℝ≥0) (hC : LipschitzWith C (radialProfile epsilon a)) :
    radialMaxSlope a / epsilon ≤ (C : ℝ) := by
  obtain ⟨q, _, hq⟩ := radialMaxSlope_attained a
  have hs : radialDensity epsilon a (epsilon * q) = radialMaxSlope a / epsilon := by
    rw [radialDensity_normalize epsilon a (epsilon * q) he,
      mul_div_cancel_left₀ q he.ne', hq]
  have hb := (hasDerivAt_radialProfile epsilon a (epsilon * q) he).le_of_lipschitz hC
  rwa [hs, Real.norm_of_nonneg (div_nonneg (radialMaxSlope_pos a).le he.le)] at hb

end FractalSync.DistanceGraph
