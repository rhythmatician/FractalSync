import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Complex.Basic
import Mathlib.Topology.MetricSpace.HausdorffDistance
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

namespace FractalSync.MainCardioidRuler

/-- Parameter of the quadratic map whose fixed point has multiplier `z`. -/
noncomputable def parameter (z : ℂ) : ℂ := z * (2 - z) / 4

/-- Reciprocal curvature-minus-one hyperbolic density, in parameter coordinates. -/
noncomputable def ruler (z : ℂ) : ℝ := (1 - ‖z‖ ^ 2) * ‖1 - z‖ / 4

/-- The image of the unit circle under the multiplier parametrization. -/
def boundary : Set ℂ := parameter '' {z : ℂ | ‖z‖ = 1}

theorem fixedPoint (z : ℂ) : (z / 2) ^ 2 + parameter z = z / 2 := by
  unfold parameter
  ring

theorem hasDerivAt_parameter (z : ℂ) :
    HasDerivAt parameter ((1 - z) / 2) z := by
  convert ((hasDerivAt_id' z).mul (HasDerivAt.const_sub 2 (hasDerivAt_id' z))).div_const 4 using 1
  simp
  ring

theorem parameter_difference (u v : ℂ) :
    parameter u - parameter v = (u - v) * (2 - u - v) / 4 := by
  unfold parameter
  ring

theorem parameter_dist (u v : ℂ) :
    dist (parameter u) (parameter v) = ‖u - v‖ * ‖2 - u - v‖ / 4 := by
  rw [dist_eq_norm, parameter_difference, norm_div, norm_mul]
  norm_num

/-- The coordinate differential sends a disk tangent to one of the same intrinsic speed. -/
theorem tangent_speed (z v : ℂ) (hz : ‖z‖ < 1) :
    ‖((1 - z) / 2) * v‖ / ruler z = 2 * ‖v‖ / (1 - ‖z‖ ^ 2) := by
  have hnorm : ‖1 - z‖ ≠ 0 := by
    intro h
    have heq := sub_eq_zero.mp (norm_eq_zero.mp h)
    rw [← heq, norm_one] at hz
    exact (lt_irrefl 1) hz
  have hdisk : 1 - ‖z‖ ^ 2 ≠ 0 := by
    nlinarith [norm_nonneg z]
  rw [ruler, norm_mul, norm_div]
  norm_num
  field_simp [hnorm, hdisk]
  ring

private theorem norm_sub_unit (u v : ℂ) (hv : ‖v‖ = 1) : 1 - ‖u‖ ≤ ‖u - v‖ := by
  have h := norm_add_le (v - u) u
  rw [sub_add_cancel, norm_sub_rev, hv] at h
  linarith

/-- Chord bounds for the distance to the cardioid, with a minimizing multiplier `a`
and the radial boundary multiplier `b`. -/
theorem boundaryDistance_bracket (r : ℝ) (a b : ℂ) (hr : 0 ≤ r) (hr' : r ≤ 1)
    (ha : ‖a‖ = 1) (hb : ‖b‖ = 1)
    (hclosest : dist (parameter ((r : ℂ) * b)) (parameter a) =
      Metric.infDist (parameter ((r : ℂ) * b)) boundary) :
    let u : ℂ := (r : ℂ) * b
    (1 - r) * ‖2 - u - a‖ / 4 ≤ Metric.infDist (parameter u) boundary ∧
      Metric.infDist (parameter u) boundary ≤ (1 - r) * ‖2 - u - b‖ / 4 := by
  have hu : ‖(r : ℂ) * b‖ = r := by
    simp [hb, abs_of_nonneg hr]
  have hradial : ‖(r : ℂ) * b - b‖ = 1 - r := by
    rw [show (r : ℂ) * b - b = ((r : ℂ) - 1) * b by ring, norm_mul, hb, mul_one,
      ← Complex.ofReal_one, ← Complex.ofReal_sub, Complex.norm_real, Real.norm_eq_abs,
      abs_of_nonpos (sub_nonpos.mpr hr')]
    ring
  constructor
  · rw [← hclosest, parameter_dist]
    have hlower := norm_sub_unit ((r : ℂ) * b) a ha
    rw [hu] at hlower
    exact div_le_div_of_nonneg_right
      (mul_le_mul_of_nonneg_right hlower (norm_nonneg _)) (by norm_num)
  · have hmem : parameter b ∈ boundary := ⟨b, hb, rfl⟩
    simpa only [parameter_dist, hradial] using
      Metric.infDist_le_dist_of_mem (x := parameter ((r : ℂ) * b)) hmem

/-- Among unit multipliers, the cusp is nearest to the image of `0 ≤ r ≤ 1`
on the real axis. -/
theorem cusp_axis_nearest (r : ℝ) (hr : 0 ≤ r) (hr' : r ≤ 1)
    (v : ℂ) (hv : ‖v‖ = 1) :
    (1 - r) ^ 2 / 4 ≤ dist (parameter (r : ℂ)) (parameter v) := by
  rw [parameter_dist]
  have hfirst : 1 - r ≤ ‖(r : ℂ) - v‖ := by
    simpa [abs_of_nonneg hr] using norm_sub_unit (r : ℂ) v hv
  have hsecond : 1 - r ≤ ‖2 - (r : ℂ) - v‖ := by
    have h := Complex.re_le_norm (2 - (r : ℂ) - v)
    have hvre := Complex.re_le_norm v
    rw [hv] at hvre
    simp at h
    linarith
  have h := mul_le_mul hfirst hsecond (sub_nonneg.mpr hr') (norm_nonneg _)
  nlinarith

theorem cusp_axis_distance (r : ℝ) (hr : 0 ≤ r) (hr' : r ≤ 1) :
    Metric.infDist (parameter (r : ℂ)) boundary = (1 - r) ^ 2 / 4 := by
  have hone : parameter 1 ∈ boundary := ⟨1, norm_one, rfl⟩
  apply le_antisymm
  · have h := Metric.infDist_le_dist_of_mem (x := parameter (r : ℂ)) hone
    rw [parameter_dist] at h
    have hfirst : ‖(r : ℂ) - 1‖ = 1 - r := by
      rw [← Complex.ofReal_one, ← Complex.ofReal_sub, Complex.norm_real, Real.norm_eq_abs,
        abs_of_nonpos (sub_nonpos.mpr hr')]
      ring
    have hsecond : ‖2 - (r : ℂ) - 1‖ = 1 - r := by
      rw [show (2 : ℂ) - (r : ℂ) - 1 = 1 - (r : ℂ) by ring,
        ← Complex.ofReal_one, ← Complex.ofReal_sub, Complex.norm_real, Real.norm_eq_abs,
        abs_of_nonneg (sub_nonneg.mpr hr')]
    rw [hfirst, hsecond, ← sq] at h
    exact h
  · apply (Metric.le_infDist ⟨_, hone⟩).mpr
    rintro _ ⟨v, hv, rfl⟩
    exact cusp_axis_nearest r hr hr' v hv

theorem cusp_axis_ruler (r : ℝ) (hr : 0 ≤ r) (hr' : r ≤ 1) :
    ruler (r : ℂ) = (1 + r) * Metric.infDist (parameter (r : ℂ)) boundary := by
  have hnorm : ‖1 - (r : ℂ)‖ = 1 - r := by
    rw [← Complex.ofReal_one, ← Complex.ofReal_sub, Complex.norm_real, Real.norm_eq_abs,
      abs_of_nonneg (sub_nonneg.mpr hr')]
  rw [ruler, Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg hr, hnorm,
    cusp_axis_distance r hr hr']
  ring

end FractalSync.MainCardioidRuler
