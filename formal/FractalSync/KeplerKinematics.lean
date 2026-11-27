import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus

/-! Exact-real scalar kinematics of the focus-relative ellipse. -/
namespace FractalSync.KeplerKinematics

theorem clock_positive (e c : ℝ) (he : 0 ≤ e) (he' : e < 1)
    (hc : c ≤ 1) : 0 < 1 - e*c := by
  have h := mul_le_mul_of_nonneg_left hc he
  linarith

theorem clock_lower (e c : ℝ) (he : 0 ≤ e) (hc : c ≤ 1) :
    1-e ≤ 1-e*c := by
  have h := mul_le_mul_of_nonneg_left hc he
  linarith

theorem focus_radius_sq (a b e c s : ℝ)
    (hb : b^2 = a^2*(1-e^2)) (hunit : c^2+s^2=1) :
    (a*(c-e))^2+(b*s)^2 = (a*(1-e*c))^2 := by
  calc
    _ = a^2*(c-e)^2+b^2*s^2 := by ring
    _ = a^2*(1-e*c)^2 := by
      rw [hb]
      linear_combination (a^2*(1-e^2))*hunit
    _ = _ := by ring

theorem focus_determinant (a b e c s : ℝ) (hunit : c^2+s^2=1) :
    a*(c-e)*(b*c) - b*s*(-a*s) = a*b*(1-e*c) := by
  linear_combination (a*b)*hunit

theorem determinant_positive (a b e c s : ℝ)
    (ha : 0<a) (hb : 0<b) (he : 0≤e) (he' : e<1)
    (hc : c≤1) (hu : c^2+s^2=1) :
    0 < a*(c-e)*(b*c)-b*s*(-a*s) := by
  rw [focus_determinant a b e c s hu]
  exact mul_pos (mul_pos ha hb) (clock_positive e c he he' hc)

theorem speed_ratio_identity (e : ℝ) (he : e < 1) :
    (1+e)/(1-e)*(1-e) = 1+e := by
  field_simp [show 1-e ≠ 0 by linarith]

theorem area_clock_ratio_bounds (l u x y : ℝ)
    (hl : 0<l) (hx : l≤x) (hx' : x≤u)
    (hy : l≤y) (hy' : y≤u) :
    l/u ≤ x/y ∧ x/y ≤ u/l := by
  have hp : 0<y := lt_of_lt_of_le hl hy
  have hu : 0<u := lt_of_lt_of_le hp hy'
  constructor
  · apply (div_le_div_iff₀ hu hp).2
    exact le_trans (mul_le_mul_of_nonneg_left hy' hl.le)
      (mul_le_mul_of_nonneg_right hx hu.le)
  · apply (div_le_div_iff₀ hp hl).2
    exact le_trans (mul_le_mul_of_nonneg_right hx' hl.le)
      (mul_le_mul_of_nonneg_left hy hu.le)

theorem feedforward_balance (a u drag : ℝ) :
    -u + (a+u+drag) - drag = a := by ring

/-- Unwrapped classical mean anomaly. -/
noncomputable def meanAnomaly (e x : ℝ) : ℝ := x - e * Real.sin x

theorem hasDerivAt_meanAnomaly (e x : ℝ) :
    HasDerivAt (meanAnomaly e) (1 - e * Real.cos x) x :=
  (hasDerivAt_id x).sub ((Real.hasDerivAt_sin x).const_mul e)

theorem integral_classical_rate (e x : ℝ) :
    (∫ u in (0 : ℝ)..x, 1 - e * Real.cos u) = meanAnomaly e x := by
  rw [intervalIntegral.integral_eq_sub_of_hasDerivAt
    (fun u _ => hasDerivAt_meanAnomaly e u)
    ((continuous_const.sub (continuous_const.mul Real.continuous_cos)).intervalIntegrable _ _)]
  simp [meanAnomaly]

theorem integral_classical_rate_period (e : ℝ) :
    (∫ u in (0 : ℝ)..(2 * Real.pi), 1 - e * Real.cos u) = 2 * Real.pi := by
  simp [integral_classical_rate, meanAnomaly]

end FractalSync.KeplerKinematics
