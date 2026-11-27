import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus

/-! A normalized swept-area clock on one orbital period. -/

namespace FractalSync.KeplerAreaClock

open MeasureTheory Set
open scoped Interval

noncomputable section

variable {h g : ℝ → ℝ} {T l U x y : ℝ}

/-- Accumulated swept area, measured from periapsis. -/
def area (h : ℝ → ℝ) (E : ℝ) : ℝ := ∫ u in 0..E, h u

/-- Rescale the accumulated area to the same range as the orbital anomaly. -/
def phase (h : ℝ → ℝ) (T E : ℝ) : ℝ := T / area h T * area h E

theorem area_sub (hc : Continuous h) (x y : ℝ) :
    area h y - area h x = ∫ u in x..y, h u := by
  have hadd := intervalIntegral.integral_add_adjacent_intervals (μ := volume)
    (hc.intervalIntegrable 0 x) (hc.intervalIntegrable x y)
  unfold area
  linarith

theorem phase_zero (h : ℝ → ℝ) (T : ℝ) : phase h T 0 = 0 := by
  simp [phase, area]

theorem phase_end (hT : area h T ≠ 0) : phase h T T = T := by
  unfold phase
  field_simp

/-- Two ordered anomalies sweep an area bounded by the interval width times
the lower and upper bounds on area density. -/
theorem area_sub_bounds (hc : Continuous h) (hxy : x ≤ y)
    (hbound : ∀ u ∈ Icc x y, l ≤ h u ∧ h u ≤ U) :
    l * (y - x) ≤ area h y - area h x ∧ area h y - area h x ≤ U * (y - x) := by
  rw [area_sub hc]
  have hlow := intervalIntegral.integral_mono_on (μ := volume) hxy
    (continuous_const.intervalIntegrable x y) (hc.intervalIntegrable x y)
    (fun u hu => (hbound u hu).1)
  have hupp := intervalIntegral.integral_mono_on (μ := volume) hxy
    (hc.intervalIntegrable x y) (continuous_const.intervalIntegrable x y)
    (fun u hu => (hbound u hu).2)
  simpa [intervalIntegral.integral_const, smul_eq_mul, mul_comm] using And.intro hlow hupp

theorem total_pos (hc : Continuous h) (hT : 0 < T) (hl : 0 < l)
    (hbound : ∀ u ∈ Icc 0 T, l ≤ h u ∧ h u ≤ U) : 0 < area h T := by
  have hlo := (area_sub_bounds hc hT.le hbound).1
  simp [area] at hlo
  exact (mul_pos hl hT).trans_le hlo

/-- Increment bounds for the forward phase map, in both directions. -/
theorem phase_sub_bounds (hc : Continuous h) (hT : 0 < T) (hl : 0 < l)
    (hb : ∀ u ∈ Icc 0 T, l ≤ h u ∧ h u ≤ U)
    (hx : x ∈ Icc 0 T) (hy : y ∈ Icc 0 T) (hxy : x ≤ y) :
    (T / area h T * l) * (y - x) ≤ phase h T y - phase h T x ∧
    phase h T y - phase h T x ≤ (T / area h T * U) * (y - x) := by
  have hp := div_pos hT (total_pos hc hT hl hb)
  have hi := area_sub_bounds hc hxy (fun u hu => hb u ⟨hx.1.trans hu.1, hu.2.trans hy.2⟩)
  simpa only [phase, ← mul_sub, mul_assoc] using
    And.intro (mul_le_mul_of_nonneg_left hi.1 hp.le) (mul_le_mul_of_nonneg_left hi.2 hp.le)

theorem phase_strictMonoOn (hc : Continuous h) (hT : 0 < T) (hl : 0 < l)
    (hb : ∀ u ∈ Icc 0 T, l ≤ h u ∧ h u ≤ U) : StrictMonoOn (phase h T) (Icc 0 T) := by
  intro x hx y hy hxy
  have hp := mul_pos (div_pos hT (total_pos hc hT hl hb)) hl
  have hpos := (mul_pos hp (sub_pos.mpr hxy)).trans_le
    (phase_sub_bounds hc hT hl hb hx hy hxy.le).1
  exact sub_pos.mp hpos

theorem phase_hasDerivAt (hc : Continuous h) (E : ℝ) :
    HasDerivAt (phase h T) (T / area h T * h E) E := by
  exact (intervalIntegral.integral_hasDerivAt_right (hc.intervalIntegrable 0 E)
    hc.stronglyMeasurable.stronglyMeasurableAtFilter hc.continuousAt).const_mul _

theorem phase_continuous (hc : Continuous h) : Continuous (phase h T) :=
  continuous_iff_continuousAt.mpr fun E => (phase_hasDerivAt (T := T) hc E).continuousAt

/-- A fixed nonzero area scale cancels out during normalization. -/
theorem phase_const_mul (c : ℝ) (hc : c ≠ 0) (ha : area h T ≠ 0) (E : ℝ) :
    phase (fun u => c * h u) T E = phase h T E := by
  simp only [phase, area, intervalIntegral.integral_const_mul]
  change T / (c * area h T) * (c * area h E) = T / area h T * area h E
  field_simp

/-- Every musical phase has exactly one anomaly on the closed orbital period. -/
theorem anomaly_existsUnique (hc : Continuous h) (hT : 0 < T) (hl : 0 < l)
    (hb : ∀ u ∈ Icc 0 T, l ≤ h u ∧ h u ≤ U) {M : ℝ} (hM : M ∈ Icc 0 T) :
    ∃! E, E ∈ Icc 0 T ∧ phase h T E = M := by
  have hp := total_pos hc hT hl hb
  have hsurj := intermediate_value_Icc hT.le (phase_continuous (T := T) hc).continuousOn
  rw [phase_zero, phase_end hp.ne'] at hsurj
  obtain ⟨E, hE, heq⟩ := hsurj hM
  exact ⟨E, ⟨hE, heq⟩, fun _ hF =>
    (phase_strictMonoOn hc hT hl hb).injOn hF.1 hE (hF.2.trans heq.symm)⟩

/-- Equal phase increments are exactly equal swept-area increments after rescaling. -/
theorem area_sub_eq_phase_sub (hT : T ≠ 0) (ha : area h T ≠ 0) (x y : ℝ) :
    area h y - area h x = area h T / T * (phase h T y - phase h T x) := by
  unfold phase
  field_simp

/-- A phase error controls the corresponding anomaly error, including near periapsis. -/
theorem anomaly_error_bound (hc : Continuous h) (hT : 0 < T) (hl : 0 < l)
    (hb : ∀ u ∈ Icc 0 T, l ≤ h u ∧ h u ≤ U)
    (hx : x ∈ Icc 0 T) (hy : y ∈ Icc 0 T) :
    |y - x| ≤ |phase h T y - phase h T x| / (T / area h T * l) := by
  have hp := mul_pos (div_pos hT (total_pos hc hT hl hb)) hl
  rcases le_total x y with hxy | hyx
  · have hm := (phase_strictMonoOn hc hT hl hb).monotoneOn hx hy hxy
    rw [abs_of_nonneg (sub_nonneg.mpr hxy), abs_of_nonneg (sub_nonneg.mpr hm)]
    exact (le_div_iff₀ hp).2 (by
      rw [mul_comm]
      exact (phase_sub_bounds hc hT hl hb hx hy hxy).1)
  · have hm := (phase_strictMonoOn hc hT hl hb).monotoneOn hy hx hyx
    rw [abs_sub_comm y x, abs_sub_comm (phase h T y) (phase h T x),
      abs_of_nonneg (sub_nonneg.mpr hyx), abs_of_nonneg (sub_nonneg.mpr hm)]
    exact (le_div_iff₀ hp).2 (by
      rw [mul_comm]
      exact (phase_sub_bounds hc hT hl hb hy hx hyx).1)

/-- Normalize a distorted area density against a reference clock. -/
theorem relative_phase_sub_bounds (hc : Continuous h) (hg : Continuous g)
    (hT : 0 < T) (hl : 0 < l) (hU : 0 < U) (hgT : 0 < area g T)
    (hb : ∀ u ∈ Icc 0 T, l * g u ≤ h u ∧ h u ≤ U * g u)
    (hx : x ∈ Icc 0 T) (hy : y ∈ Icc 0 T) (hxy : x ≤ y)
    (hgxy : 0 ≤ area g y - area g x) :
    (l / U) * (phase g T y - phase g T x) ≤ phase h T y - phase h T x ∧
    phase h T y - phase h T x ≤ (U / l) * (phase g T y - phase g T x) := by
  have cmp (a b : ℝ) (hab : a ≤ b) (ha : 0 ≤ a) (hbt : b ≤ T) :
      l * (area g b - area g a) ≤ area h b - area h a ∧
      area h b - area h a ≤ U * (area g b - area g a) := by
    rw [area_sub hg, area_sub hc]
    rw [← intervalIntegral.integral_const_mul, ← intervalIntegral.integral_const_mul]
    exact ⟨intervalIntegral.integral_mono_on (μ := volume) hab
      ((continuous_const (y := l)).mul hg |>.intervalIntegrable a b) (hc.intervalIntegrable a b)
      (fun u hu => (hb u ⟨ha.trans hu.1, hu.2.trans hbt⟩).1),
      intervalIntegral.integral_mono_on (μ := volume) hab (hc.intervalIntegrable a b)
      ((continuous_const (y := U)).mul hg |>.intervalIntegrable a b)
      (fun u hu => (hb u ⟨ha.trans hu.1, hu.2.trans hbt⟩).2)⟩
  have ht := cmp 0 T hT.le le_rfl le_rfl
  simp only [area, intervalIntegral.integral_same, sub_zero] at ht
  change l * area g T ≤ area h T ∧ area h T ≤ U * area g T at ht
  have hhT := (mul_pos hl hgT).trans_le ht.1
  have hs := cmp x y hxy hx.1 hy.2
  have hhs := (mul_nonneg hl.le hgxy).trans hs.1
  have lowDen : (T / (U * area g T)) ≤ T / area h T :=
    div_le_div_of_nonneg_left hT.le hhT ht.2
  have highDen : T / area h T ≤ T / (l * area g T) :=
    div_le_div_of_nonneg_left hT.le (mul_pos hl hgT) ht.1
  simp only [phase, ← mul_sub]
  constructor
  · calc
      _ = (T / (U * area g T)) * (l * (area g y - area g x)) := by field_simp [ne_of_gt hU, ne_of_gt hl, ne_of_gt hgT]
      _ ≤ _ := mul_le_mul lowDen hs.1 (mul_nonneg hl.le hgxy) (div_nonneg hT.le hhT.le)
  · calc
      _ ≤ (T / (l * area g T)) * (U * (area g y - area g x)) :=
        mul_le_mul highDen hs.2 hhs (div_nonneg hT.le (mul_pos hl hgT).le)
      _ = _ := by field_simp [ne_of_gt hU, ne_of_gt hl, ne_of_gt hgT]

end
end FractalSync.KeplerAreaClock
