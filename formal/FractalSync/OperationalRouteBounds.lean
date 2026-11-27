import FractalSync.AbsolutelyContinuousPaths

/-!
# Length enclosures from a bounded operational ruler

The ruler and its derivative are supplied by the selected field. The statement
applies to the straight-segment length computation once the table and chart
enclose these quantities.
-/

open MeasureTheory
open scoped Interval

namespace FractalSync.OperationalRouteBounds

/-- Pullback speed with a regularized ruler and its rate along the path. -/
noncomputable def speed (a v rho rate : ℝ) : ℝ :=
  Real.sqrt (v ^ 2 + a ^ 2 * rate ^ 2) / rho

/-- A whole segment with constant ambient speed has the integrated enclosure
when the ruler and its along-segment rate obey the supplied bounds. -/
theorem segmentLength_bounds (a v lo hi g : ℝ) (rho rate : ℝ → ℝ)
    (hv : 0 ≤ v) (hlo : 0 < lo) (hg : 0 ≤ g)
    (bounds : ∀ t ∈ Set.Icc (0 : ℝ) 1, lo ≤ rho t ∧ rho t ≤ hi ∧ |rate t| ≤ g * v)
    (integrable : IntervalIntegrable (fun t => speed a v (rho t) (rate t)) volume 0 1) :
    v / hi ≤ (∫ t in (0 : ℝ)..1, speed a v (rho t) (rate t)) ∧
      (∫ t in (0 : ℝ)..1, speed a v (rho t) (rate t)) ≤
        v * Real.sqrt (1 + a ^ 2 * g ^ 2) / lo := by
  have hpoint : ∀ t ∈ Set.Icc (0 : ℝ) 1,
      v / hi ≤ speed a v (rho t) (rate t) ∧
      speed a v (rho t) (rate t) ≤ v * Real.sqrt (1 + a ^ 2 * g ^ 2) / lo := by
    intro t ht
    obtain ⟨hl, hu, hr⟩ := bounds t ht
    have hpos : 0 < rho t := hlo.trans_le hl
    have hrsq : rate t ^ 2 ≤ (g * v) ^ 2 := by
      rw [← sq_abs (rate t)]
      exact (sq_le_sq₀ (abs_nonneg _) (mul_nonneg hg hv)).2 hr
    have hnumLower : v ≤ Real.sqrt (v ^ 2 + a ^ 2 * rate t ^ 2) := by
      calc
        v = Real.sqrt (v ^ 2) := (Real.sqrt_sq hv).symm
        _ ≤ _ := Real.sqrt_le_sqrt (le_add_of_nonneg_right (mul_nonneg (sq_nonneg _) (sq_nonneg _)))
    have hnumUpper : Real.sqrt (v ^ 2 + a ^ 2 * rate t ^ 2) ≤
        v * Real.sqrt (1 + a ^ 2 * g ^ 2) := by
      have squared : v ^ 2 + a ^ 2 * rate t ^ 2 ≤ v ^ 2 * (1 + a ^ 2 * g ^ 2) := calc
        v ^ 2 + a ^ 2 * rate t ^ 2 ≤ v ^ 2 + a ^ 2 * (g * v) ^ 2 :=
          add_le_add_left (mul_le_mul_of_nonneg_left hrsq (sq_nonneg _)) _
        _ = v ^ 2 * (1 + a ^ 2 * g ^ 2) := by ring
      calc
        _ ≤ Real.sqrt (v ^ 2 * (1 + a ^ 2 * g ^ 2)) := Real.sqrt_le_sqrt squared
        _ = _ := by rw [Real.sqrt_mul (sq_nonneg v), Real.sqrt_sq hv]
    constructor
    · exact (div_le_div_of_nonneg_left hv hpos hu).trans (div_le_div_of_nonneg_right hnumLower hpos.le)
    · exact (div_le_div_of_nonneg_right hnumUpper hpos.le).trans
        (div_le_div_of_nonneg_left (mul_nonneg hv (Real.sqrt_nonneg _)) hlo hl)
  constructor
  · simpa only [intervalIntegral.integral_const, sub_zero, one_smul] using
      intervalIntegral.integral_mono_on zero_le_one (intervalIntegrable_const (c := v / hi))
        integrable (fun t ht => (hpoint t ht).1)
  · simpa only [intervalIntegral.integral_const, sub_zero, one_smul] using
      intervalIntegral.integral_mono_on zero_le_one integrable
        (intervalIntegrable_const (c := v * Real.sqrt (1 + a ^ 2 * g ^ 2) / lo))
        (fun t ht => (hpoint t ht).2)

/-- The ambient target-ball lower bound follows from a regional bound on
the ruler along the admitted path. -/
theorem ambientTargetLower_le {X : Type*} [PseudoMetricSpace X]
    (start arrival center : X) (radius rulerUpper length : ℝ)
    (hRuler : 0 < rulerUpper) (hLength : 0 ≤ length)
    (hDisplacement : dist start arrival ≤ length * rulerUpper)
    (hArrival : dist arrival center ≤ radius) :
    max 0 (dist start center - radius) / rulerUpper ≤ length := by
  apply (div_le_iff₀ hRuler).2
  apply max_le (mul_nonneg hLength hRuler.le)
  have hTriangle := dist_triangle start arrival center
  linarith

/-- A vertex on a feasible route must fit both the prefix and a suffix to
the target ball into the budget. This bound can precede edge certification. -/
theorem throughVertexLower_le_budget {X : Type*} [PseudoMetricSpace X]
    (start vertex arrival center : X) (radius rulerUpper prefixLength suffixLength budget : ℝ)
    (hRuler : 0 < rulerUpper) (hSuffix : 0 ≤ suffixLength)
    (hPrefixDisplacement : dist start vertex ≤ prefixLength * rulerUpper)
    (hSuffixDisplacement : dist vertex arrival ≤ suffixLength * rulerUpper)
    (hArrival : dist arrival center ≤ radius) (hBudget : prefixLength + suffixLength ≤ budget) :
    dist start vertex / rulerUpper + max 0 (dist vertex center - radius) / rulerUpper ≤ budget := by
  have hPrefix := (div_le_iff₀ hRuler).2 hPrefixDisplacement
  have hTail := ambientTargetLower_le vertex arrival center radius rulerUpper suffixLength
    hRuler hSuffix hSuffixDisplacement hArrival
  exact (add_le_add hPrefix hTail).trans hBudget

end FractalSync.OperationalRouteBounds
