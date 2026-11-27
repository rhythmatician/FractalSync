import FractalSync.PerturbedRungeKutta
import FractalSync.RungeKuttaWork

/-!
# Signed work from observed Runge--Kutta stages

The work samples use the recorded mechanical stages. Separate residuals
describe power evaluation and scalar accumulation, while the mechanical
residuals describe RHS, stage construction, and endpoint arithmetic. The
result is conditional on these bounds, not on a floating-point model.
-/

open Set
open scoped BigOperators

noncomputable section

namespace FractalSync.RungeKutta.Perturbed

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The four signed power evaluations recorded at the mechanical stage states. -/
structure Powers where
  source : ℝ
  stage2 : ℝ
  stage3 : ℝ
  stage4 : ℝ

/-- The same classical weights as the recorded mechanical slopes. -/
def Powers.mean (p : Powers) : ℝ :=
  (p.source + 2 * p.stage2 + 2 * p.stage3 + p.stage4) / 6

/-- The weighted signed work before the separate scalar arithmetic residual. -/
def Powers.work (p : Powers) (h : ℝ) : ℝ := h * p.mean

/-- Evaluation errors refer to actual power at the recorded states. -/
structure PowerResiduals (P : E → ℝ) (x : E) (s : Stages E)
    (p : Powers) (ε : ℝ) : Prop where
  source_error : |p.source - P x| ≤ ε
  stage2_error : |p.stage2 - P s.stage2| ≤ ε
  stage3_error : |p.stage3 - P s.stage3| ≤ ε
  stage4_error : |p.stage4 - P s.stage4| ≤ ε

/-- One sample separates power evaluation, differential variation, and RHS error. -/
theorem power_sample_error (energy P : E → ℝ) (f : E → E)
    (x y k : E) (p A K M η ε d : ℝ)
    (hP : P y = fderiv ℝ energy y (f y))
    (heval : |p - P y| ≤ ε) (hf : ‖f y‖ ≤ M) (hres : ‖k - f y‖ ≤ η)
    (hA : ‖fderiv ℝ energy x‖ ≤ A)
    (hK : ‖fderiv ℝ energy y - fderiv ℝ energy x‖ ≤ K * d) :
    |p - fderiv ℝ energy x k| ≤ ε + K * d * M + A * η := by
  have hvar : |P y - fderiv ℝ energy x (f y)| ≤ K * d * M := by
    simpa only [hP, ContinuousLinearMap.sub_apply, Real.norm_eq_abs] using
      ((fderiv ℝ energy y - fderiv ℝ energy x).le_opNorm (f y)).trans
        (mul_le_mul hK hf (norm_nonneg _) ((norm_nonneg _).trans hK))
  have hrhs : |fderiv ℝ energy x (f y) - fderiv ℝ energy x k| ≤ A * η := by
    simpa only [map_sub, Real.norm_eq_abs] using
      ((fderiv ℝ energy x).le_opNorm (f y - k)).trans
        (mul_le_mul hA (by simpa only [norm_sub_rev] using hres)
          (norm_nonneg _) ((norm_nonneg _).trans hA))
  exact ((abs_sub_le p (P y) (fderiv ℝ energy x k)).trans
    (add_le_add heval ((abs_sub_le (P y) (fderiv ℝ energy x (f y))
      (fderiv ℝ energy x k)).trans (add_le_add hvar hrhs)))).trans_eq (by ring)

/-- Applying a fixed covector commutes with the recorded classical RK weights. -/
theorem Powers.mean_sub_linear (p : Powers) (s : Stages E) (l : E →L[ℝ] ℝ) :
    p.mean - l s.meanSlope =
      ((p.source - l s.slope0) + 2 * (p.stage2 - l s.slope2) +
        2 * (p.stage3 - l s.slope3) + (p.stage4 - l s.slope4)) / 6 := by
  simp only [Powers.mean, Stages.meanSlope, map_smul, map_add, smul_eq_mul]
  ring

/-- Positive RK weights preserve a common signed-power error budget. -/
theorem Powers.mean_error_le (p : Powers) (s : Stages E) (l : E →L[ℝ] ℝ)
    (B : ℝ) (h₀ : |p.source - l s.slope0| ≤ B)
    (h₂ : |p.stage2 - l s.slope2| ≤ B)
    (h₃ : |p.stage3 - l s.slope3| ≤ B)
    (h₄ : |p.stage4 - l s.slope4| ≤ B) :
    |p.mean - l s.meanSlope| ≤ B := by
  rw [p.mean_sub_linear]
  simpa only [Real.norm_eq_abs, smul_eq_mul, div_eq_mul_inv, mul_comm, one_mul] using
    weighted_norm_le (v₀ := p.source - l s.slope0) h₀ h₂ h₃ h₄

/-- Local work error includes observed stages, RHS evaluation, endpoint and ledger arithmetic. -/
theorem energy_work_defect {f : E → E} {energy P : E → ℝ}
    {h M A K R η δ τ ε ω work : ℝ} {c x : E} {s : Stages E} {p : Powers}
    (hb : StageBounds s h x M η δ τ) (hr : StageRegion s x c R)
    (he : Residuals f h x s η δ τ) (hp : PowerResiduals P x s p ε)
    (hh : 0 ≤ h) (hM : 0 ≤ M) (hη : 0 ≤ η) (hδ : 0 ≤ δ) (hK : 0 ≤ K)
    (hf : ∀ y ∈ Metric.closedBall c R, ‖f y‖ ≤ M)
    (hD : ∀ y ∈ Metric.closedBall c R, DifferentiableAt ℝ energy y)
    (hA : ‖fderiv ℝ energy x‖ ≤ A)
    (hLip : ∀ y ∈ Metric.closedBall c R, ∀ z ∈ Metric.closedBall c R,
      ‖fderiv ℝ energy y - fderiv ℝ energy z‖ ≤ K * ‖y - z‖)
    (hP : ∀ y ∈ Metric.closedBall c R, P y = fderiv ℝ energy y (f y))
    (hwork : |work - p.work h| ≤ ω) :
    |energy s.endpoint - energy x - work| ≤
      K / 2 * (h * (M + η) + τ) ^ 2 +
        h * (K * (h * (M + η) + δ) * M + A * η + ε) + A * τ + ω := by
  have hpoint (y k : E) (q : ℝ) (hy : y ∈ Metric.closedBall c R)
      (hd : ‖y - x‖ ≤ h * (M + η) + δ) (hq : |q - P y| ≤ ε)
      (hk : ‖k - f y‖ ≤ η) :
      |q - fderiv ℝ energy x k| ≤ ε + K * (h * (M + η) + δ) * M + A * η :=
    power_sample_error energy P f x y k q A K M η ε (h * (M + η) + δ)
      (hP y hy) hq (hf y hy) hk hA
      ((hLip y hy x hr.start_mem).trans (mul_le_mul_of_nonneg_left hd hK))
  have hmean := p.mean_error_le s (fderiv ℝ energy x)
    (ε + K * (h * (M + η) + δ) * M + A * η)
    (hpoint x s.slope0 p.source hr.start_mem
      (by simpa using add_nonneg (mul_nonneg hh (add_nonneg hM hη)) hδ)
      hp.source_error he.slope0_error)
    (hpoint s.stage2 s.slope2 p.stage2 hr.stage2_mem
      (by nlinarith [hb.stage2_displacement, mul_nonneg hh (add_nonneg hM hη)])
      hp.stage2_error he.slope2_error)
    (hpoint s.stage3 s.slope3 p.stage3 hr.stage3_mem
      (by nlinarith [hb.stage3_displacement, mul_nonneg hh (add_nonneg hM hη)])
      hp.stage3_error he.slope3_error)
    (hpoint s.stage4 s.slope4 p.stage4 hr.stage4_mem hb.stage4_displacement
      hp.stage4_error he.slope4_error)
  have ht : |energy s.endpoint - energy x - fderiv ℝ energy x (s.endpoint - x)| ≤
      K / 2 * (h * (M + η) + τ) ^ 2 :=
    (energy_taylor_remainder_on_ball energy c x s.endpoint R K
      hr.start_mem hr.endpoint_mem hD hLip).trans
      (mul_le_mul_of_nonneg_left
        (pow_le_pow_left₀ (norm_nonneg _) hb.endpoint_displacement 2) (by positivity))
  have hlin : fderiv ℝ energy x (s.endpoint - x) - h * fderiv ℝ energy x s.meanSlope =
      fderiv ℝ energy x (s.endpoint - (x + h • s.meanSlope)) := by
    simp [map_sub, map_add, map_smul, smul_eq_mul, sub_sub]
  have hend : |fderiv ℝ energy x (s.endpoint - x) -
      h * fderiv ℝ energy x s.meanSlope| ≤ A * τ := by
    simpa only [hlin, Real.norm_eq_abs] using
      ((fderiv ℝ energy x).le_opNorm (s.endpoint - (x + h • s.meanSlope))).trans
        (mul_le_mul hA he.endpoint_error (norm_nonneg _) ((norm_nonneg _).trans hA))
  have hpower : |h * fderiv ℝ energy x s.meanSlope - p.work h| ≤
      h * (ε + K * (h * (M + η) + δ) * M + A * η) := by
    simpa only [Powers.work, ← mul_sub, abs_mul, abs_of_nonneg hh, abs_sub_comm] using
      mul_le_mul_of_nonneg_left hmean hh
  exact ((abs_sub_le (energy s.endpoint - energy x)
    (fderiv ℝ energy x (s.endpoint - x)) work).trans
    (add_le_add ht ((abs_sub_le (fderiv ℝ energy x (s.endpoint - x))
      (h * fderiv ℝ energy x s.meanSlope) work).trans
      (add_le_add hend ((abs_sub_le (h * fderiv ℝ energy x s.meanSlope)
        (p.work h) work).trans (add_le_add hpower
        (by simpa only [abs_sub_comm] using hwork))))))).trans_eq (by ring)

/-- Energy and stored-work defects telescope with independent accumulator errors. -/
theorem accumulated_stored_work_defect (N : ℕ) (energy work : ℕ → ℝ)
    (increment defect arithmetic : Fin N → ℝ)
    (henergy : ∀ i : Fin N,
      |energy (i.val + 1) - energy i.val - increment i| ≤ defect i)
    (hwork : ∀ i : Fin N,
      |work (i.val + 1) - work i.val - increment i| ≤ arithmetic i) :
    |energy N - energy 0 - (work N - work 0)| ≤
      ∑ i : Fin N, (defect i + arithmetic i) := by
  have hlocal (i : Fin N) :
      |energy (i.val + 1) - energy i.val - (work (i.val + 1) - work i.val)| ≤
        defect i + arithmetic i :=
    (abs_sub_le (energy (i.val + 1) - energy i.val) (increment i)
      (work (i.val + 1) - work i.val)).trans
      (add_le_add (henergy i) (by simpa only [abs_sub_comm] using hwork i))
  have hsum := ScaleMetric.accumulated_energy_defect N energy
    (fun i => work (i.val + 1) - work i.val) (fun i => defect i + arithmetic i) hlocal
  simpa only [Fin.sum_univ_eq_sum_range (fun i => work (i + 1) - work i) N,
    Finset.sum_range_sub] using hsum

end FractalSync.RungeKutta.Perturbed
