import FractalSync.ExteriorRulerBounds
import Mathlib.Analysis.Complex.LocallyUniformLimit
import Mathlib.Analysis.Complex.TaylorSeries
import Mathlib.Analysis.SpecialFunctions.Complex.LogBounds

namespace FractalSync.ExteriorRulerJets

open ExteriorRulerBounds
open scoped NNReal Interval

/-- Critical orbit of the quadratic family, indexed from the critical point. -/
def criticalOrbit (c : ℂ) (j : ℕ) : ℂ := orbitTail c 0 j

/-- Radius bound for the image of the parameter disk under each critical-orbit polynomial. -/
noncomputable def orbitError (c₀ : ℂ) (h : ℝ) : ℕ → ℝ
  | 0 => 0
  | j + 1 => 2 * ‖criticalOrbit c₀ j‖ * orbitError c₀ h j + orbitError c₀ h j ^ 2 + h

theorem orbit_error_nonneg {c₀ : ℂ} {h : ℝ} (hh : 0 ≤ h) (j : ℕ) :
    0 ≤ orbitError c₀ h j := by
  induction j with
  | zero => simp [orbitError]
  | succ j ih =>
    rw [orbitError]
    positivity

theorem critical_orbit_perturbation {c c₀ : ℂ} {h : ℝ}
    (hh : 0 ≤ h) (hc : ‖c - c₀‖ ≤ h) (j : ℕ) :
    ‖criticalOrbit c j - criticalOrbit c₀ j‖ ≤ orbitError c₀ h j := by
  induction j with
  | zero => simp [criticalOrbit, orbitTail, orbitError]
  | succ j ih =>
    let δ := criticalOrbit c j - criticalOrbit c₀ j
    have hstep : criticalOrbit c (j + 1) - criticalOrbit c₀ (j + 1) =
        (2 * criticalOrbit c₀ j * δ + δ ^ 2) + (c - c₀) := by
      change (criticalOrbit c j ^ 2 + c) - (criticalOrbit c₀ j ^ 2 + c₀) = _
      dsimp [δ]
      ring
    rw [hstep, orbitError]
    calc
      _ ≤ ‖2 * criticalOrbit c₀ j * δ‖ + ‖δ ^ 2‖ + ‖c - c₀‖ :=
        (norm_add_le _ _).trans (add_le_add_right (norm_add_le _ _) _)
      _ = 2 * ‖criticalOrbit c₀ j‖ * ‖δ‖ + ‖δ‖ ^ 2 + ‖c - c₀‖ := by
        simp [norm_pow]
      _ ≤ _ := by
        have he := orbit_error_nonneg (c₀ := c₀) hh j
        gcongr

theorem closed_disk_escape_guard {c₀ : ℂ} {h R C : ℝ} {n : ℕ}
    (hh : 0 ≤ h) (hR : R ≤ ‖criticalOrbit c₀ n‖ - orbitError c₀ h n)
    (hC : ‖c₀‖ + h ≤ C) :
    ∀ c ∈ Metric.closedBall c₀ h, R ≤ ‖criticalOrbit c n‖ ∧ ‖c‖ ≤ C := by
  intro c hc
  have hch : ‖c - c₀‖ ≤ h := by rwa [← dist_eq_norm]
  have hdiff := critical_orbit_perturbation hh hch n
  have hcenter : ‖criticalOrbit c₀ n‖ ≤
      ‖criticalOrbit c n‖ + ‖criticalOrbit c n - criticalOrbit c₀ n‖ := by
    simpa [norm_sub_rev] using norm_sub_le (criticalOrbit c n)
      (criticalOrbit c n - criticalOrbit c₀ n)
  have hparam : ‖c‖ ≤ ‖c - c₀‖ + ‖c₀‖ := by simpa using norm_add_le (c - c₀) c₀
  exact ⟨by linarith, by linarith⟩

theorem critical_orbit_differentiable (j : ℕ) : Differentiable ℂ (criticalOrbit · j) := by
  induction j with
  | zero => exact differentiable_const 0
  | succ j ih => exact (ih.pow 2).add differentiable_id

theorem critical_orbit_add (c : ℂ) (n j : ℕ) :
    criticalOrbit c (n + j) = orbitTail c (criticalOrbit c n) j := by
  induction j with
  | zero => simp [orbitTail]
  | succ j ih =>
    rw [Nat.add_succ]
    change criticalOrbit c (n + j) ^ 2 + c = orbitTail c (criticalOrbit c n) j ^ 2 + c
    rw [ih]

theorem complex_log_near_one {z : ℂ} (hz : ‖z‖ ≤ 1 / 4) :
    ‖Complex.log (1 + z)‖ ≤ (4 / 3 : ℝ) * ‖z‖ := by
  have hz' : ‖z‖ < 1 := by linarith
  have hi : (1 - ‖z‖)⁻¹ ≤ 4 / 3 := by
    rw [inv_eq_one_div, div_le_iff₀ (sub_pos.mpr hz')]
    linarith
  refine (Complex.norm_log_one_add_le hz').trans ?_
  calc
    _ ≤ ‖z‖ ^ 2 * (4 / 3) / 2 + ‖z‖ := by gcongr
    _ ≤ _ := by nlinarith [mul_nonneg (norm_nonneg z) (sub_nonneg.mpr hz)]

/-- The holomorphic truncation tail with first weight `a` (normally `2 ^ (-n)`). -/
noncomputable def logTailTerm (a : ℝ) (n j : ℕ) (c : ℂ) : ℂ :=
  (a : ℂ) * (1 / 2) ^ j * Complex.log (1 + c / criticalOrbit c (n + j) ^ 2)

theorem tail_term_bound {c : ℂ} {n : ℕ} {a C R : ℝ}
    (ha : 0 ≤ a) (hR : 4 ≤ R) (hCR : C ≤ R ^ 2 / 4)
    (hz : R ≤ ‖criticalOrbit c n‖) (hc : ‖c‖ ≤ C) (j : ℕ) :
    ‖logTailTerm a n j c‖ ≤ (4 / 3 : ℝ) * (1 / 18) ^ j * (a * C / R ^ 2) := by
  have hsz : 4 ≤ ‖criticalOrbit c n‖ := hR.trans hz
  have hsc : ‖c‖ ≤ ‖criticalOrbit c n‖ ^ 2 / 4 := by
    apply hc.trans (hCR.trans _)
    gcongr
  obtain ⟨h4, hguard, hg⟩ := guarded_growth hsz hsc j
  rw [← critical_orbit_add] at h4 hguard hg
  let z := criticalOrbit c (n + j)
  have hzpos : 0 < ‖z‖ := by dsimp [z]; linarith
  have hRpos : 0 < R := by linarith
  have hCnonneg : 0 ≤ C := (norm_nonneg c).trans hc
  have hb : (3 : ℝ) ^ j * R ≤ ‖z‖ :=
    (mul_le_mul_of_nonneg_left hz (by positivity)).trans hg
  have hquarter : ‖c / z ^ 2‖ ≤ 1 / 4 := by
    rw [norm_div, norm_pow, div_le_iff₀ (by positivity : 0 < ‖z‖ ^ 2)]
    linarith
  have hscale : ‖c / z ^ 2‖ ≤ (1 / 9 : ℝ) ^ j * (C / R ^ 2) := by
    rw [norm_div, norm_pow]
    calc
      _ ≤ C / ((3 : ℝ) ^ j * R) ^ 2 := by
        gcongr
      _ = _ := by
        rw [mul_pow, ← pow_mul, mul_comm j, pow_mul, div_eq_mul_inv, mul_inv_rev]
        rw [one_div, inv_pow]
        norm_num
        ring
  calc
    ‖logTailTerm a n j c‖ = a * (1 / 2 : ℝ) ^ j * ‖Complex.log (1 + c / z ^ 2)‖ := by
      simp only [logTailTerm, z, norm_mul, norm_pow, norm_div, norm_one, Complex.norm_of_nonneg ha]
      norm_num
    _ ≤ a * (1 / 2 : ℝ) ^ j * ((4 / 3) * ((1 / 9 : ℝ) ^ j * (C / R ^ 2))) := by
      gcongr
      exact (complex_log_near_one hquarter).trans (by gcongr)
    _ = _ := by
      rw [← show (1 / 2 : ℝ) * (1 / 9) = 1 / 18 by norm_num, mul_pow]
      ring

theorem tail_term_differentiable {c : ℂ} {a : ℝ} {n j : ℕ}
    (hz : 4 ≤ ‖criticalOrbit c n‖) (hc : ‖c‖ ≤ ‖criticalOrbit c n‖ ^ 2 / 4) :
    DifferentiableAt ℂ (logTailTerm a n j) c := by
  obtain ⟨h4, hguard, _⟩ := guarded_growth hz hc j
  rw [← critical_orbit_add] at h4 hguard
  have hzpos : 0 < ‖criticalOrbit c (n + j)‖ := by linarith
  have hzne : criticalOrbit c (n + j) ≠ 0 := norm_ne_zero_iff.mp hzpos.ne'
  have hquot : ‖c / criticalOrbit c (n + j) ^ 2‖ < 1 := by
    rw [norm_div, norm_pow, div_lt_iff₀ (by positivity)]
    nlinarith
  have hd := (differentiableAt_id.div ((critical_orbit_differentiable (n + j) c).pow 2)
    (pow_ne_zero 2 hzne)).const_add 1
  exact (hd.clog (Complex.mem_slitPlane_of_norm_lt_one hquot)).const_mul _

/-- Uniformly convergent tail on a closed disk, holomorphic on its interior. -/
theorem log_tail_on_disk {c₀ : ℂ} {n : ℕ} {a C R h : ℝ}
    (ha : 0 ≤ a) (hR : 4 ≤ R) (hCR : C ≤ R ^ 2 / 4)
    (hguard : ∀ c ∈ Metric.closedBall c₀ h, R ≤ ‖criticalOrbit c n‖ ∧ ‖c‖ ≤ C) :
    DiffContOnCl ℂ (fun c => ∑' j, logTailTerm a n j c) (Metric.ball c₀ h) ∧
      ∀ c ∈ Metric.closedBall c₀ h, ‖∑' j, logTailTerm a n j c‖ ≤ (24 / 17 : ℝ) * (a * C / R ^ 2) := by
  let majorant := fun j : ℕ => (4 / 3 : ℝ) * (1 / 18) ^ j * (a * C / R ^ 2)
  have hsum : HasSum majorant ((24 / 17 : ℝ) * (a * C / R ^ 2)) :=
    geometric_sums.2.mul_right _
  have hterm (j : ℕ) (c : ℂ) (hc : c ∈ Metric.closedBall c₀ h) :
      ‖logTailTerm a n j c‖ ≤ majorant j :=
    tail_term_bound ha hR hCR (hguard c hc).1 (hguard c hc).2 j
  have hdiff (j : ℕ) : DifferentiableOn ℂ (logTailTerm a n j) (Metric.closedBall c₀ h) := by
    intro c hc
    obtain ⟨hz, hzC⟩ := hguard c hc
    apply DifferentiableAt.differentiableWithinAt
    apply tail_term_differentiable (hR.trans hz)
    apply hzC.trans (hCR.trans _)
    gcongr
  exact ⟨⟨Complex.differentiableOn_tsum_of_summable_norm hsum.summable
      (fun j => (hdiff j).mono Metric.ball_subset_closedBall) Metric.isOpen_ball
      (fun j c hc => hterm j c (Metric.ball_subset_closedBall hc)),
    (continuousOn_tsum (fun j => (hdiff j).continuousOn) hsum.summable hterm).mono
      Metric.closure_ball_subset_closedBall⟩,
    fun c hc => tsum_of_norm_bounded hsum (fun j => hterm j c hc)⟩

theorem cauchy_jet_bound {f : ℂ → ℂ} {c₀ : ℂ} {h : ℝ≥0} {E : ℝ}
    (hh : 0 < h) (hf : DiffContOnCl ℂ f (Metric.ball c₀ h))
    (hbound : ∀ c ∈ Metric.sphere c₀ h, ‖f c‖ ≤ E) (k : ℕ) :
    ‖iteratedDeriv k f c₀‖ ≤ (k.factorial : ℝ) * E / (h : ℝ) ^ k := by
  have hp := (hf.hasFPowerSeriesOnBall hh).hasFPowerSeriesAt
  have hs := hp.eq_formalMultilinearSeries hp.analyticAt.hasFPowerSeriesAt
  have hfact : 0 < (k.factorial : ℝ) := by positivity
  have hnorm := norm_cauchyPowerSeries_le f c₀ (h : ℝ) k
  have hmean :
      (2 * Real.pi)⁻¹ * (∫ θ : ℝ in 0..2 * Real.pi, ‖f (circleMap c₀ h θ)‖) ≤ E := by
    have hcircle (θ : ℝ) : circleMap c₀ (h : ℝ) θ ∈ Metric.sphere c₀ h :=
      circleMap_mem_sphere c₀ h.coe_nonneg θ
    have hcont : Continuous fun θ : ℝ => ‖f (circleMap c₀ h θ)‖ :=
      (hf.continuousOn_ball.comp_continuous (continuous_circleMap c₀ h)
        (fun θ => Metric.sphere_subset_closedBall (hcircle θ))).norm
    rw [inv_mul_le_iff₀ Real.two_pi_pos]
    calc
      _ ≤ ∫ _ : ℝ in 0..2 * Real.pi, E :=
        intervalIntegral.integral_mono_on Real.two_pi_pos.le
          (hcont.intervalIntegrable _ _) (continuous_const.intervalIntegrable _ _)
          (fun θ _ => hbound _ (hcircle θ))
      _ = _ := by simp [mul_comm]
  rw [hs] at hnorm
  have hb := hnorm.trans (mul_le_mul_of_nonneg_right hmean (by positivity))
  rw [FormalMultilinearSeries.ofScalars_norm, norm_div, Complex.norm_natCast,
    abs_of_nonneg h.coe_nonneg, inv_pow, ← div_eq_mul_inv] at hb
  calc
    ‖iteratedDeriv k f c₀‖ ≤ E / (h : ℝ) ^ k * (k.factorial : ℝ) := (div_le_iff₀ hfact).mp hb
    _ = _ := by ring

/-- The center derivative bound takes only center-orbit and disk radius certificates. -/
theorem escape_disk_tail_jet {c₀ : ℂ} {h : ℝ≥0} {R C a : ℝ} {n : ℕ}
    (hh : 0 < h) (ha : 0 ≤ a) (hR : 4 ≤ R) (hCR : C ≤ R ^ 2 / 4)
    (hradius : R ≤ ‖criticalOrbit c₀ n‖ - orbitError c₀ h n) (hC : ‖c₀‖ + h ≤ C)
    (k : ℕ) :
    ‖iteratedDeriv k (fun c => ∑' j, logTailTerm a n j c) c₀‖ ≤
      (k.factorial : ℝ) * ((24 / 17 : ℝ) * (a * C / R ^ 2)) / (h : ℝ) ^ k := by
  obtain ⟨hdiff, hbound⟩ :=
    log_tail_on_disk ha hR hCR (closed_disk_escape_guard h.coe_nonneg hradius hC)
  exact cauchy_jet_bound hh hdiff (fun c hc => hbound c (Metric.sphere_subset_closedBall hc)) k

end FractalSync.ExteriorRulerJets
