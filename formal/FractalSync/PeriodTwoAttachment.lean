import FractalSync.MainCardioidRuler
import FractalSync.QuarticRulerFloor
import Mathlib.Analysis.SpecialFunctions.Pow.Deriv

/-! The period-two multiplier, circle distance, and quartic extension at a smooth seam. -/

namespace FractalSync.PeriodTwoAttachment

noncomputable def quadratic (c z : ℂ) : ℂ := z ^ 2 + c

noncomputable def multiplier (c : ℂ) : ℂ := 4 * (c + 1)

noncomputable def ruler (c : ℂ) : ℝ := 1 / 8 - 2 * ‖c + 1‖ ^ 2

def boundary : Set ℂ := {c : ℂ | ‖c + 1‖ = 1 / 4}

theorem secondIterate_sub (c z : ℂ) :
    quadratic c (quadratic c z) - z =
      (quadratic c z - z) * (z ^ 2 + z + c + 1) := by
  unfold quadratic
  ring

theorem periodTwo_iff (c z : ℂ) :
    quadratic c (quadratic c z) = z ∧ quadratic c z ≠ z ↔
      z ^ 2 + z + c + 1 = 0 ∧ c ≠ -(3 / 4) := by
  have hsecond : quadratic c (quadratic c z) = z ↔
      quadratic c z = z ∨ z ^ 2 + z + c + 1 = 0 := by
    rw [← sub_eq_zero, secondIterate_sub, mul_eq_zero, sub_eq_zero]
  constructor
  · rintro ⟨hiterate, hfixed⟩
    have hroot := (hsecond.mp hiterate).resolve_left hfixed
    refine ⟨hroot, ?_⟩
    intro hc
    have hsquare : (z + 1 / 2) * (z + 1 / 2) = 0 := by
      linear_combination hroot - hc
    have hz : z + 1 / 2 = 0 := (mul_eq_zero.mp hsquare).elim id id
    apply hfixed
    unfold quadratic
    linear_combination hroot - 2 * hz
  · rintro ⟨hroot, hc⟩
    refine ⟨hsecond.mpr (Or.inr hroot), ?_⟩
    intro hfixed
    unfold quadratic at hfixed
    have hz : z + 1 / 2 = 0 := by linear_combination (hroot - hfixed) / 2
    apply hc
    linear_combination hroot - (z + 1 / 2) * hz

theorem cycle_multiplier (c z : ℂ) (hz : z ^ 2 + z + c + 1 = 0) :
    (2 * quadratic c z) * (2 * z) = multiplier c := by
  have hpartner : quadratic c z = -1 - z := by
    unfold quadratic
    linear_combination hz
  rw [hpartner, multiplier]
  linear_combination -4 * hz

theorem attracting_iff_disk (c : ℂ) : ‖multiplier c‖ < 1 ↔ ‖c + 1‖ < 1 / 4 := by
  rw [multiplier, norm_mul]
  norm_num [show ‖(4 : ℂ)‖ = 4 by norm_num]
  constructor <;> intro h <;> linarith

theorem ruler_attracting_coordinate (c : ℂ) :
    ruler c = (1 - ‖multiplier c‖ ^ 2) / 8 := by
  rw [ruler, multiplier, norm_mul, show ‖(4 : ℂ)‖ = 4 by norm_num]
  ring

/-- The circle distance is literal Euclidean set distance, including at the bulb center. -/
theorem boundaryDistance (c : ℂ) (hc : ‖c + 1‖ ≤ 1 / 4) :
    Metric.infDist c boundary = 1 / 4 - ‖c + 1‖ := by
  have hpoint : -(3 / 4) ∈ boundary := by
    change ‖-(3 / 4 : ℂ) + 1‖ = 1 / 4
    rw [show -(3 / 4 : ℂ) + 1 = 1 / 4 by ring]
    norm_num
  apply le_antisymm
  · by_cases hzero : c + 1 = 0
    · have hcenter : c = -1 := by linear_combination hzero
      have hnorm : ‖(-1 : ℂ) + 3 / 4‖ = 1 / 4 := by
        rw [show (-1 : ℂ) + 3 / 4 = -(1 / 4) by ring]
        norm_num
      simpa [hcenter, dist_eq_norm, hnorm] using
        (Metric.infDist_le_dist_of_mem (x := c) hpoint)
    let t : ℝ := 1 / (4 * ‖c + 1‖)
    let p : ℂ := (t : ℂ) * (c + 1) - 1
    have hn : 0 < ‖c + 1‖ := norm_pos_iff.mpr hzero
    have ht : 1 ≤ t := by dsimp [t]; rw [le_div_iff₀ (by positivity)]; linarith
    have hprod : t * ‖c + 1‖ = 1 / 4 := by
      dsimp [t]
      field_simp
    have hp : p ∈ boundary := by
      change ‖(t : ℂ) * (c + 1) - 1 + 1‖ = _
      rw [sub_add_cancel, norm_mul, Complex.norm_real, Real.norm_eq_abs,
        abs_of_nonneg (by linarith : 0 ≤ t), hprod]
    calc
      Metric.infDist c boundary ≤ dist c p := Metric.infDist_le_dist_of_mem hp
      _ = (t - 1) * ‖c + 1‖ := by
        rw [dist_comm, dist_eq_norm, show p - c = ((t - 1 : ℝ) : ℂ) * (c + 1) by
          dsimp [p]; push_cast; ring, norm_mul, Complex.norm_real, Real.norm_eq_abs,
          abs_of_nonneg (sub_nonneg.mpr ht)]
      _ = 1 / 4 - ‖c + 1‖ := by linarith
  · apply (Metric.le_infDist ⟨_, hpoint⟩).mpr
    intro p hp
    have h := norm_add_le (p - c) (c + 1)
    rw [show p - c + (c + 1) = p + 1 by ring, show ‖p + 1‖ = 1 / 4 from hp,
      norm_sub_rev, ← dist_eq_norm] at h
    linarith

theorem ruler_in_circle_distance (c : ℂ) (hc : ‖c + 1‖ ≤ 1 / 4) :
    ruler c = Metric.infDist c boundary - 2 * Metric.infDist c boundary ^ 2 := by
  rw [ruler, boundaryDistance c hc]
  ring

theorem attachment_multipliers :
    MainCardioidRuler.parameter (-1) = -(3 / 4) ∧ multiplier (-(3 / 4)) = 1 := by
  constructor <;> simp only [MainCardioidRuler.parameter, multiplier] <;> ring

/-- The scalar curvature expression with the period-two Laplacian `-8` and
squared gradient `16 s²`, where `s` is radial distance from the bulb center. -/
theorem unfloored_curvature (s : ℝ) : (1 / 8 - 2 * s ^ 2) * (-8) - 16 * s ^ 2 = -1 := by
  ring

private theorem hasDerivAt_positivePart_pow (n : ℕ) (r : ℝ) :
    HasDerivAt (fun x : ℝ => (max x 0) ^ (n + 2))
      ((n + 2 : ℕ) * (max r 0) ^ (n + 1)) r := by
  rcases lt_trichotomy r 0 with hn | rfl | hp
  · rw [max_eq_right hn.le, zero_pow (Nat.succ_ne_zero n), mul_zero]
    apply (hasDerivAt_const r (c := (0 : ℝ))).congr_of_eventuallyEq
    filter_upwards [eventually_lt_nhds hn] with x hx
    simp [max_eq_right hx.le]
  · have hpoly : HasDerivAt (fun x : ℝ => x ^ (n + 2)) 0 0 := by
      simpa using (hasDerivAt_id (0 : ℝ)).pow (n + 2)
    have hleft : HasDerivWithinAt (fun x : ℝ => (max x 0) ^ (n + 2)) 0 (Set.Iic 0) 0 := by
      apply (hasDerivAt_const (0 : ℝ) (c := (0 : ℝ))).hasDerivWithinAt.congr_of_mem
      · intro x hx
        simp [max_eq_right (show x ≤ 0 from hx)]
      · exact Set.mem_Iic.mpr le_rfl
    have hright : HasDerivWithinAt (fun x : ℝ => (max x 0) ^ (n + 2)) 0 (Set.Ici 0) 0 := by
      apply hpoly.hasDerivWithinAt.congr_of_mem
      · intro x hx
        rw [max_eq_left (show 0 ≤ x from hx)]
      · exact Set.mem_Ici.mpr le_rfl
    simpa [Set.Iic_union_Ici, hasDerivWithinAt_univ] using hleft.union hright
  · rw [max_eq_left hp.le]
    have hpoly : HasDerivAt (fun x : ℝ => x ^ (n + 2))
        ((n + 2 : ℕ) * r ^ (n + 1)) r := by
      simpa using (hasDerivAt_id r).pow (n + 2)
    apply hpoly.congr_of_eventuallyEq
    filter_upwards [eventually_gt_nhds hp] with x hx
    simp [max_eq_left hx.le]

theorem contDiff_positivePart_four : ContDiff ℝ 2 (fun x : ℝ => (max x 0) ^ 4) := by
  rw [show (2 : WithTop ℕ∞) = 1 + 1 from rfl, contDiff_succ_iff_deriv]
  refine ⟨fun x => (hasDerivAt_positivePart_pow 2 x).differentiableAt, by simp, ?_⟩
  have heq : deriv (fun x : ℝ => (max x 0) ^ 4) = fun x => 4 * (max x 0) ^ 3 := by
    funext x
    simpa using (hasDerivAt_positivePart_pow 2 x).deriv
  rw [heq, contDiff_one_iff_deriv]
  have hderiv (x : ℝ) : HasDerivAt (fun y : ℝ => 4 * (max y 0) ^ 3)
      (12 * (max x 0) ^ 2) x := by
    convert (hasDerivAt_positivePart_pow 1 x).const_mul 4 using 1
    norm_num
    ring
  refine ⟨fun x => (hderiv x).differentiableAt, ?_⟩
  rw [show deriv (fun x : ℝ => 4 * (max x 0) ^ 3) = _ from funext fun x => (hderiv x).deriv]
  exact continuous_const.mul ((continuous_id.max continuous_const).pow 2)

/-- Local smooth signed branches suffice for a `C²` sum of zero-extended powers. -/
theorem contDiffAt_glued_quartic_sum {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (r₁ r₂ : E → ℝ) (x : E) (epsilon : ℝ)
    (h₁ : ContDiffAt ℝ 2 r₁ x) (h₂ : ContDiffAt ℝ 2 r₂ x) :
    ContDiffAt ℝ 2 (fun y => epsilon ^ 4 + (max (r₁ y) 0) ^ 4 + (max (r₂ y) 0) ^ 4) x :=
  (contDiffAt_const.add (contDiff_positivePart_four.contDiffAt.comp x h₁)).add
    (contDiff_positivePart_four.contDiffAt.comp x h₂)

/-- The coefficient of the corresponding conformal metric is also `C²` for a positive floor. -/
theorem contDiffAt_glued_metric_coefficient {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] (r₁ r₂ : E → ℝ) (x : E) (epsilon : ℝ)
    (he : 0 < epsilon) (h₁ : ContDiffAt ℝ 2 r₁ x) (h₂ : ContDiffAt ℝ 2 r₂ x) :
    ContDiffAt ℝ 2
      (fun y => (epsilon ^ 4 + (max (r₁ y) 0) ^ 4 + (max (r₂ y) 0) ^ 4) ^ (-(1 / 2) : ℝ)) x := by
  exact (contDiffAt_glued_quartic_sum r₁ r₂ x epsilon h₁ h₂).rpow_const_of_ne (by positivity)

end FractalSync.PeriodTwoAttachment
