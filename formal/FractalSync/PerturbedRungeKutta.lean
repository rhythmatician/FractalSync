import FractalSync.RungeKuttaPhase

/-!
# Local error contracts for observed Runge--Kutta stages

An observed step supplies its four slope evaluations, three constructed stage
states, and weighted endpoint. Separate hypotheses bound errors in RHS
evaluation, stage construction, and endpoint arithmetic. Regional field bounds
then imply stage containment and a whole-step consistency budget. These are
conditional statements over real values, not certificates of IEEE execution.
-/

noncomputable section

namespace FractalSync.RungeKutta.Perturbed

/-- Recorded values from one four-stage update, without an arithmetic model. -/
structure Stages (E : Type*) where
  stage2 : E
  stage3 : E
  stage4 : E
  slope0 : E
  slope2 : E
  slope3 : E
  slope4 : E
  endpoint : E

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The classical positive weights applied to the recorded slopes. -/
def Stages.meanSlope (s : Stages E) : E :=
  (1 / 6 : ℝ) • (s.slope0 + (2 : ℝ) • s.slope2 +
    (2 : ℝ) • s.slope3 + s.slope4)

/-- Local error obligations, with independent RHS, stage, and endpoint budgets. -/
structure Residuals (f : E → E) (h : ℝ) (x : E) (s : Stages E)
    (η δ τ : ℝ) : Prop where
  slope0_error : ‖s.slope0 - f x‖ ≤ η
  slope2_error : ‖s.slope2 - f s.stage2‖ ≤ η
  slope3_error : ‖s.slope3 - f s.stage3‖ ≤ η
  slope4_error : ‖s.slope4 - f s.stage4‖ ≤ η
  stage2_error : ‖s.stage2 - (x + (h / 2) • s.slope0)‖ ≤ δ
  stage3_error : ‖s.stage3 - (x + (h / 2) • s.slope2)‖ ≤ δ
  stage4_error : ‖s.stage4 - (x + h • s.slope3)‖ ≤ δ
  endpoint_error : ‖s.endpoint - (x + h • s.meanSlope)‖ ≤ τ

/-- Bounds on recorded slopes and their actual displacements. -/
structure StageBounds (s : Stages E) (h : ℝ) (x : E) (M η δ τ : ℝ) : Prop where
  slope0_norm : ‖s.slope0‖ ≤ M + η
  slope2_norm : ‖s.slope2‖ ≤ M + η
  slope3_norm : ‖s.slope3‖ ≤ M + η
  slope4_norm : ‖s.slope4‖ ≤ M + η
  stage2_displacement : ‖s.stage2 - x‖ ≤ h * (M + η) / 2 + δ
  stage3_displacement : ‖s.stage3 - x‖ ≤ h * (M + η) / 2 + δ
  stage4_displacement : ‖s.stage4 - x‖ ≤ h * (M + η) + δ
  meanSlope_norm : ‖s.meanSlope‖ ≤ M + η
  endpoint_displacement : ‖s.endpoint - x‖ ≤ h * (M + η) + τ

/-- Includes the source, every evaluated stage, and the weighted endpoint. -/
structure StageRegion (s : Stages E) (x c : E) (R : ℝ) : Prop where
  start_mem : x ∈ Metric.closedBall c R
  stage2_mem : s.stage2 ∈ Metric.closedBall c R
  stage3_mem : s.stage3 ∈ Metric.closedBall c R
  stage4_mem : s.stage4 ∈ Metric.closedBall c R
  endpoint_mem : s.endpoint ∈ Metric.closedBall c R

omit [NormedSpace ℝ E] in
private theorem norm_of_residual {v w : E} {M η : ℝ}
    (he : ‖v - w‖ ≤ η) (hw : ‖w‖ ≤ M) : ‖v‖ ≤ M + η := by
  simpa only [sub_add_cancel] using
    ((norm_add_le (v - w) w).trans (add_le_add he hw)).trans_eq (add_comm η M)

private theorem affine_error_displacement {x y v : E} {t M δ : ℝ}
    (ht : 0 ≤ t) (hv : ‖v‖ ≤ M) (he : ‖y - (x + t • v)‖ ≤ δ) :
    ‖y - x‖ ≤ t * M + δ := by
  have hdisp : ‖x + t • v - x‖ ≤ t * M := by
    simpa only [add_sub_cancel_left, norm_smul, Real.norm_eq_abs, abs_of_nonneg ht]
      using mul_le_mul_of_nonneg_left hv ht
  exact ((norm_sub_le_norm_sub_add_norm_sub y (x + t • v) x).trans
    (add_le_add he hdisp)).trans_eq (add_comm δ (t * M))

omit [NormedSpace ℝ E] in
private theorem mem_of_displacement {c x y : E} {R r : ℝ}
    (hy : ‖y - x‖ ≤ r) (hR : ‖x - c‖ + r ≤ R) :
    y ∈ Metric.closedBall c R := by
  exact mem_closedBall_iff_norm.mpr
    ((norm_sub_le_norm_sub_add_norm_sub y x c).trans
      ((add_le_add_right hy _).trans (by linarith)))

/-- Constructed stages stay in the containing region in evaluation order. -/
theorem stageBounds {f : E → E} {h M η δ τ R : ℝ} {c x : E} {s : Stages E}
    (he : Residuals f h x s η δ τ)
    (hh : 0 ≤ h) (hM : 0 ≤ M) (hη : 0 ≤ η) (hδ : 0 ≤ δ) (hτ : 0 ≤ τ)
    (hf : ∀ y ∈ Metric.closedBall c R, ‖f y‖ ≤ M)
    (hroom : ‖x - c‖ + h * (M + η) + δ + τ ≤ R) :
    StageBounds s h x M η δ τ := by
  have hB : 0 ≤ M + η := add_nonneg hM hη
  have hx : x ∈ Metric.closedBall c R :=
    mem_closedBall_iff_norm.mpr (by nlinarith [mul_nonneg hh hB])
  have hk₀ := norm_of_residual he.slope0_error (hf x hx)
  have h₂ : ‖s.stage2 - x‖ ≤ h * (M + η) / 2 + δ := by
    simpa only [div_mul_eq_mul_div] using
      affine_error_displacement (div_nonneg hh (by norm_num : (0 : ℝ) ≤ 2))
        hk₀ he.stage2_error
  have hx₂ : s.stage2 ∈ Metric.closedBall c R :=
    mem_of_displacement h₂ (by nlinarith [mul_nonneg hh hB])
  have hk₂ := norm_of_residual he.slope2_error (hf _ hx₂)
  have h₃ : ‖s.stage3 - x‖ ≤ h * (M + η) / 2 + δ := by
    simpa only [div_mul_eq_mul_div] using
      affine_error_displacement (div_nonneg hh (by norm_num : (0 : ℝ) ≤ 2))
        hk₂ he.stage3_error
  have hx₃ : s.stage3 ∈ Metric.closedBall c R :=
    mem_of_displacement h₃ (by nlinarith [mul_nonneg hh hB])
  have hk₃ := norm_of_residual he.slope3_error (hf _ hx₃)
  have h₄ := affine_error_displacement hh hk₃ he.stage4_error
  have hx₄ : s.stage4 ∈ Metric.closedBall c R :=
    mem_of_displacement h₄ (by linarith)
  have hk₄ := norm_of_residual he.slope4_error (hf _ hx₄)
  have havg : ‖s.meanSlope‖ ≤ M + η := weighted_norm_le hk₀ hk₂ hk₃ hk₄
  exact ⟨hk₀, hk₂, hk₃, hk₄, h₂, h₃, h₄, havg,
    affine_error_displacement hh havg he.endpoint_error⟩

/-- The same displacement bounds include all evaluated points and the endpoint. -/
theorem StageBounds.region {s : Stages E} {h M η δ τ R : ℝ} {x c : E}
    (hb : StageBounds s h x M η δ τ)
    (hh : 0 ≤ h) (hM : 0 ≤ M) (hη : 0 ≤ η) (hδ : 0 ≤ δ) (hτ : 0 ≤ τ)
    (hroom : ‖x - c‖ + h * (M + η) + δ + τ ≤ R) :
    StageRegion s x c R := by
  exact ⟨mem_closedBall_iff_norm.mpr
      (by nlinarith [mul_nonneg hh (add_nonneg hM hη)]),
    mem_of_displacement hb.stage2_displacement
      (by nlinarith [mul_nonneg hh (add_nonneg hM hη)]),
    mem_of_displacement hb.stage3_displacement
      (by nlinarith [mul_nonneg hh (add_nonneg hM hη)]),
    mem_of_displacement hb.stage4_displacement (by linarith),
    mem_of_displacement hb.endpoint_displacement (by linarith)⟩

omit [NormedSpace ℝ E] in
/-- A certified stage region can be embedded in a larger energy domain. -/
theorem StageRegion.mono {s : Stages E} {x c c' : E} {R R' : ℝ}
    (hr : StageRegion s x c R)
    (hsub : Metric.closedBall c R ⊆ Metric.closedBall c' R') :
    StageRegion s x c' R' := by
  exact ⟨hsub hr.start_mem, hsub hr.stage2_mem, hsub hr.stage3_mem,
    hsub hr.stage4_mem, hsub hr.endpoint_mem⟩

/-- Anchor variation and RHS residuals control the recorded weighted slope. -/
theorem StageBounds.meanSlope_sub_le {f : E → E} {s : Stages E}
    {h M η δ τ L R : ℝ} {x c : E}
    (hb : StageBounds s h x M η δ τ) (hr : StageRegion s x c R)
    (he : Residuals f h x s η δ τ)
    (hh : 0 ≤ h) (hM : 0 ≤ M) (hη : 0 ≤ η) (hδ : 0 ≤ δ) (hL : 0 ≤ L)
    (hf : ∀ y ∈ Metric.closedBall c R, ‖f y - f x‖ ≤ L * ‖y - x‖) :
    ‖s.meanSlope - f x‖ ≤ L * (h * (M + η) + δ) + η := by
  have hstep : 0 ≤ h * (M + η) := mul_nonneg hh (add_nonneg hM hη)
  have hbound : 0 ≤ L * (h * (M + η) + δ) := mul_nonneg hL (add_nonneg hstep hδ)
  have hvar (y : E) (hy : y ∈ Metric.closedBall c R) (k : E)
      (hd : ‖y - x‖ ≤ h * (M + η) + δ) (hk : ‖k - f y‖ ≤ η) :
      ‖k - f x‖ ≤ L * (h * (M + η) + δ) + η :=
    ((norm_sub_le_norm_sub_add_norm_sub k (f y) (f x)).trans
      (add_le_add hk ((hf y hy).trans (mul_le_mul_of_nonneg_left hd hL)))).trans_eq
        (add_comm _ _)
  exact weighted_sub_norm_le
    (he.slope0_error.trans (le_add_of_nonneg_left hbound))
    (hvar _ hr.stage2_mem _ (by linarith [hb.stage2_displacement]) he.slope2_error)
    (hvar _ hr.stage3_mem _ (by linarith [hb.stage3_displacement]) he.slope3_error)
    (hvar _ hr.stage4_mem _ hb.stage4_displacement he.slope4_error)

/-- Whole-step consistency retains the separate endpoint arithmetic allowance. -/
theorem StageBounds.euler_defect_le {f : E → E} {s : Stages E}
    {h M η δ τ L R : ℝ} {x c : E}
    (hb : StageBounds s h x M η δ τ) (hr : StageRegion s x c R)
    (he : Residuals f h x s η δ τ)
    (hh : 0 ≤ h) (hM : 0 ≤ M) (hη : 0 ≤ η) (hδ : 0 ≤ δ) (hL : 0 ≤ L)
    (hf : ∀ y ∈ Metric.closedBall c R, ‖f y - f x‖ ≤ L * ‖y - x‖) :
    ‖s.endpoint - (x + h • f x)‖ ≤
      h * (L * (h * (M + η) + δ) + η) + τ := by
  have hmean := hb.meanSlope_sub_le hr he hh hM hη hδ hL hf
  have hinc : ‖(x + h • s.meanSlope) - (x + h • f x)‖ ≤
      h * (L * (h * (M + η) + δ) + η) := by
    simpa only [add_sub_add_left_eq_sub, ← smul_sub, norm_smul,
      Real.norm_eq_abs, abs_of_nonneg hh] using mul_le_mul_of_nonneg_left hmean hh
  exact ((norm_sub_le_norm_sub_add_norm_sub s.endpoint
    (x + h • s.meanSlope) (x + h • f x)).trans
      (add_le_add he.endpoint_error hinc)).trans_eq (add_comm _ _)

end FractalSync.RungeKutta.Perturbed
