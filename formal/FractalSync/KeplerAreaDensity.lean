import Mathlib.Analysis.SpecialFunctions.Integrals.Basic
import Mathlib.MeasureTheory.Integral.DominatedConvergence

namespace FractalSync.KeplerAreaDensity

/-- The normalized radial average of a pullback area density along one ray. -/
noncomputable def radialDensity (J : ℝ × ℝ → ℝ) (w : ℝ × ℝ) : ℝ :=
  2 * ∫ s in (0 : ℝ)..1, s * J (s * w.1, s * w.2)

theorem continuous_radialDensity {J : ℝ × ℝ → ℝ} (hJ : Continuous J) :
    Continuous (radialDensity J) := by
  have h : Continuous (Function.uncurry
      (fun (w : ℝ × ℝ) (s : ℝ) => s * J (s * w.1, s * w.2))) := by
    unfold Function.uncurry
    fun_prop
  exact continuous_const.mul
    (intervalIntegral.continuous_parametric_intervalIntegral_of_continuous' h 0 1)

private theorem weightedConstant (k : ℝ) :
    2 * (∫ s in (0 : ℝ)..1, s * k) = k := by
  rw [intervalIntegral.integral_mul_const, integral_id]
  ring

/-- The normalized radial average lies within any uniform bounds along the ray. -/
theorem radialDensity_bounds {J : ℝ × ℝ → ℝ} (hJ : Continuous J)
    (w : ℝ × ℝ) (lo hi : ℝ)
    (h : ∀ s ∈ Set.Icc (0 : ℝ) 1, lo ≤ J (s * w.1, s * w.2) ∧
      J (s * w.1, s * w.2) ≤ hi) :
    lo ≤ radialDensity J w ∧ radialDensity J w ≤ hi := by
  have hIntegrand : Continuous (fun s : ℝ => s * J (s * w.1, s * w.2)) := by
    fun_prop
  have hLower : Continuous (fun s : ℝ => s * lo) := by fun_prop
  have hUpper : Continuous (fun s : ℝ => s * hi) := by fun_prop
  have h01 : (0 : ℝ) ≤ 1 := zero_le_one
  have hlo := intervalIntegral.integral_mono_on (μ := MeasureTheory.volume) h01
    (hLower.intervalIntegrable 0 1)
    (hIntegrand.intervalIntegrable 0 1) (fun s hs => mul_le_mul_of_nonneg_left (h s hs).1 hs.1)
  have hhi := intervalIntegral.integral_mono_on (μ := MeasureTheory.volume) h01
    (hIntegrand.intervalIntegrable 0 1)
    (hUpper.intervalIntegrable 0 1) (fun s hs => mul_le_mul_of_nonneg_left (h s hs).2 hs.1)
  unfold radialDensity
  constructor
  · simpa only [weightedConstant] using
      (mul_le_mul_of_nonneg_left hlo (by norm_num : (0 : ℝ) ≤ 2))
  · simpa only [weightedConstant] using
      (mul_le_mul_of_nonneg_left hhi (by norm_num : (0 : ℝ) ≤ 2))

theorem radialDensity_one (w : ℝ × ℝ) : radialDensity (fun _ => 1) w = 1 :=
  weightedConstant 1

/-- The radial fan has tangent vectors `w` and `s v` in its two parameter directions. -/
theorem fan_determinant (w v : ℝ × ℝ) (s : ℝ) :
    w.1 * (s * v.2) - w.2 * (s * v.1) = s * (w.1 * v.2 - w.2 * v.1) := by
  ring

/-- Integrating the signed coordinate area of a radial fan factors into the classical
area rate and the normalized radial average of the pullback density. -/
theorem integral_fan_density (J : ℝ × ℝ → ℝ) (w v : ℝ × ℝ) :
    (∫ s in (0 : ℝ)..1, J (s * w.1, s * w.2) * (w.1 * (s * v.2) - w.2 * (s * v.1))) =
      ((w.1 * v.2 - w.2 * v.1) / 2) * radialDensity J w := by
  simp only [fan_determinant, radialDensity]
  calc
    _ = ∫ s in (0 : ℝ)..1, (w.1 * v.2 - w.2 * v.1) * (s * J (s * w.1, s * w.2)) := by
      congr 1
      funext s
      ring
    _ = _ := by rw [intervalIntegral.integral_const_mul]; ring

theorem integral_fan_density_bounds {J : ℝ × ℝ → ℝ} (hJ : Continuous J)
    (w v : ℝ × ℝ) (lo hi : ℝ) (hDet : 0 ≤ w.1 * v.2 - w.2 * v.1)
    (h : ∀ s ∈ Set.Icc (0 : ℝ) 1, lo ≤ J (s * w.1, s * w.2) ∧
      J (s * w.1, s * w.2) ≤ hi) :
    ((w.1 * v.2 - w.2 * v.1) / 2) * lo ≤
        (∫ s in (0 : ℝ)..1, J (s * w.1, s * w.2) *
          (w.1 * (s * v.2) - w.2 * (s * v.1))) ∧
      (∫ s in (0 : ℝ)..1, J (s * w.1, s * w.2) *
        (w.1 * (s * v.2) - w.2 * (s * v.1))) ≤
        ((w.1 * v.2 - w.2 * v.1) / 2) * hi := by
  rw [integral_fan_density]
  obtain ⟨hlo, hhi⟩ := radialDensity_bounds hJ w lo hi h
  have hRate := div_nonneg hDet (by norm_num : (0 : ℝ) ≤ 2)
  exact ⟨mul_le_mul_of_nonneg_left hlo hRate, mul_le_mul_of_nonneg_left hhi hRate⟩

end FractalSync.KeplerAreaDensity
