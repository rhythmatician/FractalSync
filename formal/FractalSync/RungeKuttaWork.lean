import FractalSync.RungeKuttaStages
import FractalSync.NormalizedMapDifferential
import FractalSync.AccumulatedDynamics

/-!
# Energy and actual Runge--Kutta work quadrature

The power samples use the same mechanical stage states as the RK update.
An actual differential identity and its regional Lipschitz bound control the
energy/work discrepancy, without a distance field or an IEEE arithmetic claim.
-/

open Set
open scoped BigOperators

noncomputable section

namespace FractalSync.RungeKutta

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The actual four power samples with the classical RK weights. -/
def meanPower (f : E → E) (P : E → ℝ) (h : ℝ) (x : E) : ℝ :=
  (P x + 2 * P (stage2 f h x) + 2 * P (stage3 f h x) + P (stage4 f h x)) / 6

/-- Signed work quadrature during one mechanical RK substep. -/
def stepWork (f : E → E) (P : E → ℝ) (h : ℝ) (x : E) : ℝ :=
  h * meanPower f P h x

/-- Actual work samples accumulated over equal substeps of one outer interval. -/
def substepsWork (f : E → E) (P : E → ℝ) (h : ℝ) (n : ℕ) (x : E) : ℝ :=
  ∑ i ∈ Finset.range n, stepWork f P (h / n) (substeps f (h / n) i x)

/-- Applying one fixed covector commutes with the RK weighted average. -/
theorem meanPower_sub_linear (f : E → E) (P : E → ℝ) (h : ℝ) (x : E)
    (l : E →L[ℝ] ℝ) :
    meanPower f P h x - l (meanSlope f h x) =
      ((P x - l (f x)) + 2 * (P (stage2 f h x) - l (f (stage2 f h x))) +
        2 * (P (stage3 f h x) - l (f (stage3 f h x))) +
        (P (stage4 f h x) - l (f (stage4 f h x)))) / 6 := by
  simp only [meanPower, meanSlope, map_smul, map_add, smul_eq_mul]
  ring

/-- Regional actual differential variation gives the endpoint Taylor remainder. -/
theorem energy_taylor_remainder_on_ball (energy : E → ℝ) (c x y : E) (R K : ℝ)
    (hx : x ∈ Metric.closedBall c R) (hy : y ∈ Metric.closedBall c R)
    (hD : ∀ z ∈ Metric.closedBall c R, DifferentiableAt ℝ energy z)
    (hLip : ∀ z ∈ Metric.closedBall c R, ∀ w ∈ Metric.closedBall c R,
      ‖fderiv ℝ energy z - fderiv ℝ energy w‖ ≤ K * ‖z - w‖) :
    |energy y - energy x - fderiv ℝ energy x (y - x)| ≤ K / 2 * ‖y - x‖ ^ 2 := by
  have hmem : ∀ t ∈ Icc (0 : ℝ) 1, x + t • (y - x) ∈ Metric.closedBall c R :=
    fun t ht => (convex_closedBall c R).add_smul_mem hx (by simpa using hy) ht
  simpa only [add_sub_cancel] using
    NormalizedDistanceFields.field_quadratic_remainder energy x (y - x) K
      (fun t ht => hD _ (hmem t ht)) (fun t ht => by
        simpa only [add_sub_cancel_left] using hLip _ (hmem t ht) x hx)

private theorem covector_eval_sub_bound (l₁ l₀ : E →L[ℝ] ℝ) (v : E) (A B : ℝ)
    (hl : ‖l₁ - l₀‖ ≤ A) (hv : ‖v‖ ≤ B) : |l₁ v - l₀ v| ≤ A * B := by
  simpa only [Real.norm_eq_abs, ContinuousLinearMap.sub_apply] using
    ((l₁ - l₀).le_opNorm v).trans
      (mul_le_mul hl hv (norm_nonneg _) ((norm_nonneg _).trans hl))

/-- The RK weights retain the half-step factor in differential/power variation. -/
theorem StageBounds.meanPower_sub_le {f : E → E} {energy P : E → ℝ}
    {h M K R : ℝ} {c x : E} (hb : StageBounds f h x M)
    (hr : StageRegion f h x c R) (hK : 0 ≤ K)
    (hLip : ∀ z ∈ Metric.closedBall c R, ∀ w ∈ Metric.closedBall c R,
      ‖fderiv ℝ energy z - fderiv ℝ energy w‖ ≤ K * ‖z - w‖)
    (hP : ∀ z ∈ Metric.closedBall c R, P z = fderiv ℝ energy z (f z)) :
    |meanPower f P h x - fderiv ℝ energy x (meanSlope f h x)| ≤ K * M ^ 2 * h / 2 := by
  have heval : ∀ y ∈ Metric.closedBall c R, ∀ d : ℝ,
      ‖y - x‖ ≤ d → ‖f y‖ ≤ M →
      |P y - fderiv ℝ energy x (f y)| ≤ K * d * M := by
    intro y hy d hd hv
    rw [hP y hy]
    exact covector_eval_sub_bound _ _ _ _ _
      ((hLip y hy x hr.start_mem).trans (mul_le_mul_of_nonneg_left hd hK)) hv
  have havg := weighted_norm_le_of_bounds
    (show ‖P x - fderiv ℝ energy x (f x)‖ ≤ (0 : ℝ) by rw [hP x hr.start_mem]; simp)
    (heval _ hr.stage2_mem _ hb.stage2_displacement hb.slope2_norm)
    (heval _ hr.stage3_mem _ hb.stage3_displacement hb.slope3_norm)
    (heval _ hr.stage4_mem _ hb.stage4_displacement hb.slope4_norm)
  rw [meanPower_sub_linear]
  convert havg using 1
  all_goals simp only [Real.norm_eq_abs, smul_eq_mul, div_eq_mul_inv, mul_comm, one_mul]
  ring

/-- One actual RK step has a quadratic energy/work defect under a Lipschitz differential. -/
theorem step_energy_work_defect {f : E → E} {energy P : E → ℝ}
    {h M K R : ℝ} {c x : E} (hb : StageBounds f h x M)
    (hr : StageRegion f h x c R) (hh : 0 ≤ h) (hK : 0 ≤ K)
    (hD : ∀ z ∈ Metric.closedBall c R, DifferentiableAt ℝ energy z)
    (hLip : ∀ z ∈ Metric.closedBall c R, ∀ w ∈ Metric.closedBall c R,
      ‖fderiv ℝ energy z - fderiv ℝ energy w‖ ≤ K * ‖z - w‖)
    (hP : ∀ z ∈ Metric.closedBall c R, P z = fderiv ℝ energy z (f z)) :
    |energy (step f h x) - energy x - stepWork f P h x| ≤ K * M ^ 2 * h ^ 2 := by
  have ht := energy_taylor_remainder_on_ball energy c x (step f h x) R K
    hr.start_mem hr.step_mem hD hLip
  have ht' : |energy (step f h x) - energy x - fderiv ℝ energy x (step f h x - x)| ≤
      K / 2 * (h * M) ^ 2 :=
    ht.trans (mul_le_mul_of_nonneg_left
      (pow_le_pow_left₀ (norm_nonneg _) hb.step_displacement 2) (by positivity))
  have hlin : fderiv ℝ energy x (step f h x - x) - stepWork f P h x =
      h * (fderiv ℝ energy x (meanSlope f h x) - meanPower f P h x) := by
    simp only [step, stepWork, add_sub_cancel_left, map_smul, smul_eq_mul, mul_sub]
  have hp : |fderiv ℝ energy x (step f h x - x) - stepWork f P h x| ≤
      K * M ^ 2 * h ^ 2 / 2 := by
    rw [hlin, abs_mul, abs_of_nonneg hh, abs_sub_comm]
    exact (mul_le_mul_of_nonneg_left (hb.meanPower_sub_le hr hK hLip hP) hh).trans_eq (by ring)
  calc
    _ ≤ |energy (step f h x) - energy x - fderiv ℝ energy x (step f h x - x)| +
        |fderiv ℝ energy x (step f h x - x) - stepWork f P h x| := abs_sub_le _ _ _
    _ ≤ K / 2 * (h * M) ^ 2 + K * M ^ 2 * h ^ 2 / 2 := add_le_add ht' hp
    _ = K * M ^ 2 * h ^ 2 := by ring

/-- Signed work defects telescope over the actual equal-substep recurrence. -/
theorem substeps_energy_work_defect {f : E → E} {energy P : E → ℝ}
    {h M K R : ℝ} {c x : E} {n : ℕ} (hh : 0 ≤ h) (hK : 0 ≤ K) (hn : 0 < n)
    (hb : ∀ i < n, StageBounds f (h / n) (substeps f (h / n) i x) M)
    (hr : ∀ i < n, StageRegion f (h / n) (substeps f (h / n) i x) c R)
    (hD : ∀ z ∈ Metric.closedBall c R, DifferentiableAt ℝ energy z)
    (hLip : ∀ z ∈ Metric.closedBall c R, ∀ w ∈ Metric.closedBall c R,
      ‖fderiv ℝ energy z - fderiv ℝ energy w‖ ≤ K * ‖z - w‖)
    (hP : ∀ z ∈ Metric.closedBall c R, P z = fderiv ℝ energy z (f z)) :
    |energy (substeps f (h / n) n x) - energy x - substepsWork f P h n x| ≤
      K * M ^ 2 * h ^ 2 / n := by
  have hs := ScaleMetric.accumulated_energy_defect n
    (fun i => energy (substeps f (h / n) i x))
    (fun i => stepWork f P (h / n) (substeps f (h / n) i x))
    (fun _ => K * M ^ 2 * (h / n) ^ 2) (fun i => by
      simpa only [substeps] using step_energy_work_defect (hb i i.isLt) (hr i i.isLt)
        (div_nonneg hh (Nat.cast_nonneg n)) hK hD hLip hP)
  have hs' : |energy (substeps f (h / n) n x) - energy x - substepsWork f P h n x| ≤
      (n : ℝ) * (K * M ^ 2 * (h / n) ^ 2) := by
    simpa only [substepsWork, substeps,
      Fin.sum_univ_eq_sum_range (fun i => stepWork f P (h / n) (substeps f (h / n) i x)) n,
      Finset.sum_const,
      Finset.card_univ, Fintype.card_fin, Finset.card_range, nsmul_eq_mul] using hs
  exact hs'.trans_eq (by
    field_simp [show (n : ℝ) ≠ 0 from Nat.cast_ne_zero.mpr hn.ne'])

/-- The same actual stages inherit a larger containing ball. -/
theorem StageRegion.mono {f : E → E} {h R R' : ℝ} {x c c' : E}
    (hr : StageRegion f h x c R)
    (hsub : Metric.closedBall c R ⊆ Metric.closedBall c' R') : StageRegion f h x c' R' := by
  exact ⟨hsub hr.start_mem, hsub hr.stage2_mem, hsub hr.stage3_mem,
    hsub hr.stage4_mem, hsub hr.step_mem⟩

/-- Regional field and energy certificates supply the entire outer work ledger. -/
theorem outer_energy_work_defect {f : E → E} {energy P : E → ℝ}
    {h M K R : ℝ} {c x : E} {n : ℕ}
    (hh : 0 ≤ h) (hM : 0 ≤ M) (hK : 0 ≤ K) (hn : 0 < n)
    (hf : ∀ y ∈ Metric.closedBall x (h * M), ‖f y‖ ≤ M)
    (hsub : Metric.closedBall x (h * M) ⊆ Metric.closedBall c R)
    (hD : ∀ z ∈ Metric.closedBall c R, DifferentiableAt ℝ energy z)
    (hLip : ∀ z ∈ Metric.closedBall c R, ∀ w ∈ Metric.closedBall c R,
      ‖fderiv ℝ energy z - fderiv ℝ energy w‖ ≤ K * ‖z - w‖)
    (hP : ∀ z ∈ Metric.closedBall c R, P z = fderiv ℝ energy z (f z)) :
    |energy (substeps f (h / n) n x) - energy x - substepsWork f P h n x| ≤
      K * M ^ 2 * h ^ 2 / n := by
  exact substeps_energy_work_defect hh hK hn
    (fun i hi => outer_substeps_stageBounds hh hM hn hf i hi)
    (fun i hi => (outer_substeps_stageRegion hh hM hn hf i hi).mono hsub) hD hLip hP

end FractalSync.RungeKutta
