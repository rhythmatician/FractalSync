import FractalSync.BoundedLipschitz

/-!
# Classical Runge--Kutta stages in a bounded region

The definitions are the exact real classical autonomous four-stage update.
The estimates use only regional norm and Lipschitz bounds; they do not assume
any particular distance provider or claim a fourth-order convergence rate.
-/

noncomputable section

namespace FractalSync.RungeKutta

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

def stage2 (f : E → E) (h : ℝ) (x : E) : E := x + (h / 2) • f x

def stage3 (f : E → E) (h : ℝ) (x : E) : E := x + (h / 2) • f (stage2 f h x)

def stage4 (f : E → E) (h : ℝ) (x : E) : E := x + h • f (stage3 f h x)

def meanSlope (f : E → E) (h : ℝ) (x : E) : E :=
  (1 / 6 : ℝ) • (f x + (2 : ℝ) • f (stage2 f h x) +
    (2 : ℝ) • f (stage3 f h x) + f (stage4 f h x))

def step (f : E → E) (h : ℝ) (x : E) : E := x + h • meanSlope f h x

def substeps (f : E → E) (h : ℝ) : ℕ → E → E
  | 0, x => x
  | n + 1, x => step f h (substeps f h n x)

/-- Bounds concern the actual stages, including the full RK endpoint. -/
structure StageBounds (f : E → E) (h : ℝ) (x : E) (M : ℝ) : Prop where
  slope0_norm : ‖f x‖ ≤ M
  slope2_norm : ‖f (stage2 f h x)‖ ≤ M
  slope3_norm : ‖f (stage3 f h x)‖ ≤ M
  slope4_norm : ‖f (stage4 f h x)‖ ≤ M
  stage2_displacement : ‖stage2 f h x - x‖ ≤ h * M / 2
  stage3_displacement : ‖stage3 f h x - x‖ ≤ h * M / 2
  stage4_displacement : ‖stage4 f h x - x‖ ≤ h * M
  meanSlope_norm : ‖meanSlope f h x‖ ≤ M
  step_displacement : ‖step f h x - x‖ ≤ h * M

/-- Stage membership includes both the source and the weighted RK endpoint. -/
structure StageRegion (f : E → E) (h : ℝ) (x c : E) (R : ℝ) : Prop where
  start_mem : x ∈ Metric.closedBall c R
  stage2_mem : stage2 f h x ∈ Metric.closedBall c R
  stage3_mem : stage3 f h x ∈ Metric.closedBall c R
  stage4_mem : stage4 f h x ∈ Metric.closedBall c R
  step_mem : step f h x ∈ Metric.closedBall c R

private theorem affine_displacement {x v : E} {t M : ℝ}
    (ht : 0 ≤ t) (hv : ‖v‖ ≤ M) : ‖x + t • v - x‖ ≤ t * M := by
  simpa only [add_sub_cancel_left, norm_smul, Real.norm_eq_abs, abs_of_nonneg ht]
    using mul_le_mul_of_nonneg_left hv ht

/-- Positive RK weights combine the four individual norm bounds. -/
theorem weighted_norm_le_of_bounds {v₀ v₂ v₃ v₄ : E} {M₀ M₂ M₃ M₄ : ℝ}
    (h₀ : ‖v₀‖ ≤ M₀) (h₂ : ‖v₂‖ ≤ M₂) (h₃ : ‖v₃‖ ≤ M₃) (h₄ : ‖v₄‖ ≤ M₄) :
    ‖(1 / 6 : ℝ) • (v₀ + (2 : ℝ) • v₂ + (2 : ℝ) • v₃ + v₄)‖ ≤
      (M₀ + 2 * M₂ + 2 * M₃ + M₄) / 6 := by
  have ht := (norm_add_le (v₀ + (2 : ℝ) • v₂ + (2 : ℝ) • v₃) v₄).trans
    (add_le_add_right ((norm_add_le (v₀ + (2 : ℝ) • v₂) ((2 : ℝ) • v₃)).trans
      (add_le_add_right (norm_add_le v₀ ((2 : ℝ) • v₂)) _)) _)
  norm_num only [norm_smul, Real.norm_eq_abs, one_div, abs_inv] at ht ⊢
  linarith

/-- The classical positive RK weights preserve a common norm bound. -/
theorem weighted_norm_le {v₀ v₂ v₃ v₄ : E} {M : ℝ}
    (h₀ : ‖v₀‖ ≤ M) (h₂ : ‖v₂‖ ≤ M) (h₃ : ‖v₃‖ ≤ M) (h₄ : ‖v₄‖ ≤ M) :
    ‖(1 / 6 : ℝ) • (v₀ + (2 : ℝ) • v₂ + (2 : ℝ) • v₃ + v₄)‖ ≤ M := by
  linarith [weighted_norm_le_of_bounds h₀ h₂ h₃ h₄]

omit [NormedSpace ℝ E] in
private theorem mem_of_displacement {c x y : E} {R r : ℝ}
    (hy : ‖y - x‖ ≤ r) (hR : ‖x - c‖ + r ≤ R) : y ∈ Metric.closedBall c R := by
  exact mem_closedBall_iff_norm.mpr
    ((norm_sub_le_norm_sub_add_norm_sub y x c).trans
      ((add_le_add_right hy _).trans (by linarith)))

theorem StageBounds.region {f : E → E} {h M R : ℝ} {c x : E}
    (hb : StageBounds f h x M) (hh : 0 ≤ h) (hM : 0 ≤ M)
    (hroom : ‖x - c‖ + h * M ≤ R) : StageRegion f h x c R := by
  exact ⟨mem_closedBall_iff_norm.mpr
      ((le_add_of_nonneg_right (mul_nonneg hh hM)).trans hroom),
    mem_of_displacement hb.stage2_displacement (by nlinarith),
    mem_of_displacement hb.stage3_displacement (by nlinarith),
    mem_of_displacement hb.stage4_displacement hroom,
    mem_of_displacement hb.step_displacement hroom⟩

/-- A whole RK step stays in any ball with enough room for its length budget. -/
theorem stageBounds {f : E → E} {h M R : ℝ} {c x : E}
    (hh : 0 ≤ h) (hM : 0 ≤ M)
    (hf : ∀ y ∈ Metric.closedBall c R, ‖f y‖ ≤ M)
    (hroom : ‖x - c‖ + h * M ≤ R) : StageBounds f h x M := by
  have hx : x ∈ Metric.closedBall c R :=
    mem_closedBall_iff_norm.mpr
      ((le_add_of_nonneg_right (mul_nonneg hh hM)).trans hroom)
  have h₂ : ‖stage2 f h x - x‖ ≤ h * M / 2 := by
    simpa only [stage2, div_mul_eq_mul_div] using
      affine_displacement (x := x) (div_nonneg hh (by norm_num : (0 : ℝ) ≤ 2)) (hf x hx)
  have hx₂ := mem_of_displacement h₂
    (show ‖x - c‖ + h * M / 2 ≤ R by nlinarith)
  have h₃ : ‖stage3 f h x - x‖ ≤ h * M / 2 := by
    simpa only [stage3, div_mul_eq_mul_div] using
      affine_displacement (x := x) (div_nonneg hh (by norm_num : (0 : ℝ) ≤ 2))
        (hf _ hx₂)
  have hx₃ := mem_of_displacement h₃
    (show ‖x - c‖ + h * M / 2 ≤ R by nlinarith)
  have h₄ : ‖stage4 f h x - x‖ ≤ h * M :=
    affine_displacement hh (hf _ hx₃)
  have hx₄ := mem_of_displacement h₄ hroom
  have havg : ‖meanSlope f h x‖ ≤ M :=
    weighted_norm_le (hf _ hx) (hf _ hx₂) (hf _ hx₃) (hf _ hx₄)
  exact ⟨hf _ hx, hf _ hx₂, hf _ hx₃, hf _ hx₄, h₂, h₃, h₄, havg,
    affine_displacement hh havg⟩

/-- Every endpoint of repeated equal substeps satisfies its cumulative length budget. -/
theorem substeps_displacement {f : E → E} {h M R : ℝ} {c x : E}
    (hh : 0 ≤ h) (hM : 0 ≤ M)
    (hf : ∀ y ∈ Metric.closedBall c R, ‖f y‖ ≤ M) (n : ℕ) :
    ‖x - c‖ + (n : ℝ) * h * M ≤ R →
      ‖substeps f h n x - x‖ ≤ (n : ℝ) * h * M := by
  induction n with
  | zero => simp [substeps]
  | succ n ih =>
    intro hroom
    have hroom' : ‖x - c‖ + (n : ℝ) * h * M + h * M ≤ R := by
      simpa only [Nat.cast_add, Nat.cast_one, add_mul, one_mul, add_assoc] using hroom
    have hn := ih (by nlinarith [mul_nonneg hh hM])
    have hst := stageBounds hh hM hf
      (show ‖substeps f h n x - c‖ + h * M ≤ R by
        linarith [norm_sub_le_norm_sub_add_norm_sub (substeps f h n x) x c])
    simpa only [substeps, Nat.cast_add, Nat.cast_one, add_mul, one_mul, add_comm]
      using (norm_sub_le_norm_sub_add_norm_sub
        (step f h (substeps f h n x)) (substeps f h n x) x).trans
        (add_le_add hst.step_displacement hn)

/-- The next stage certificate uses the same original containing ball. -/
theorem substeps_stageBounds {f : E → E} {h M R : ℝ} {c x : E}
    (hh : 0 ≤ h) (hM : 0 ≤ M)
    (hf : ∀ y ∈ Metric.closedBall c R, ‖f y‖ ≤ M) (n : ℕ)
    (hroom : ‖x - c‖ + ((n + 1 : ℕ) : ℝ) * h * M ≤ R) :
    StageBounds f h (substeps f h n x) M := by
  have hroom' : ‖x - c‖ + (n : ℝ) * h * M + h * M ≤ R := by
    simpa only [Nat.cast_add, Nat.cast_one, add_mul, one_mul, add_assoc] using hroom
  have hn := substeps_displacement hh hM hf n
    (by nlinarith [mul_nonneg hh hM] : ‖x - c‖ + (n : ℝ) * h * M ≤ R)
  exact stageBounds hh hM hf (by
    linarith [norm_sub_le_norm_sub_add_norm_sub (substeps f h n x) x c])

/-- All stages of an intermediate substep stay in the original containing ball. -/
theorem substeps_stageRegion {f : E → E} {h M R : ℝ} {c x : E}
    (hh : 0 ≤ h) (hM : 0 ≤ M)
    (hf : ∀ y ∈ Metric.closedBall c R, ‖f y‖ ≤ M) (n : ℕ)
    (hroom : ‖x - c‖ + ((n + 1 : ℕ) : ℝ) * h * M ≤ R) :
    StageRegion f h (substeps f h n x) c R := by
  have hroom' : ‖x - c‖ + (n : ℝ) * h * M + h * M ≤ R := by
    simpa only [Nat.cast_add, Nat.cast_one, add_mul, one_mul, add_assoc] using hroom
  have hn := substeps_displacement hh hM hf n
    (by nlinarith [mul_nonneg hh hM] : ‖x - c‖ + (n : ℝ) * h * M ≤ R)
  exact (substeps_stageBounds hh hM hf n hroom).region hh hM (by
    linarith [norm_sub_le_norm_sub_add_norm_sub (substeps f h n x) x c])

theorem meanSlope_sub (f : E → E) (h : ℝ) (x v : E) :
    meanSlope f h x - v = (1 / 6 : ℝ) •
      ((f x - v) + (2 : ℝ) • (f (stage2 f h x) - v) +
        (2 : ℝ) • (f (stage3 f h x) - v) + (f (stage4 f h x) - v)) := by
  simp only [meanSlope, smul_add, smul_sub, smul_smul]
  module

/-- The averaged slope has a first-order defect from the source evaluation. -/
theorem StageBounds.meanSlope_sub_le {f : E → E} {h M L : ℝ} {x : E}
    (hb : StageBounds f h x M) (hh : 0 ≤ h) (hM : 0 ≤ M) (hL : 0 ≤ L)
    (hlip : ∀ y ∈ Metric.closedBall x (h * M),
      ‖f y - f x‖ ≤ L * ‖y - x‖) :
    ‖meanSlope f h x - f x‖ ≤ L * M * h / 2 := by
  have hr := hb.region hh hM (show ‖x - x‖ + h * M ≤ h * M by simp)
  have h₂ := (hlip _ hr.stage2_mem).trans
    (mul_le_mul_of_nonneg_left hb.stage2_displacement hL)
  have h₃ := (hlip _ hr.stage3_mem).trans
    (mul_le_mul_of_nonneg_left hb.stage3_displacement hL)
  have h₄ := (hlip _ hr.stage4_mem).trans
    (mul_le_mul_of_nonneg_left hb.stage4_displacement hL)
  have havg := weighted_norm_le_of_bounds
    (show ‖f x - f x‖ ≤ (0 : ℝ) by simp) h₂ h₃ h₄
  rw [← meanSlope_sub] at havg
  nlinarith

/-- An intermediate equal-substep budget never exceeds the outer-step budget. -/
theorem substep_budget {h M : ℝ} {n i : ℕ}
    (hh : 0 ≤ h) (hM : 0 ≤ M) (hn : 0 < n) (hi : i ≤ n) :
    (i : ℝ) * (h / n) * M ≤ h * M := by
  have hi' : (i : ℝ) ≤ (n : ℝ) := by exact_mod_cast hi
  have hbound := mul_le_mul_of_nonneg_right
    (mul_le_mul_of_nonneg_right hi' (div_nonneg hh (Nat.cast_nonneg n))) hM
  have heq : (n : ℝ) * (h / n) = h := by
    field_simp [show (n : ℝ) ≠ 0 from Nat.cast_ne_zero.mpr hn.ne']
  simpa only [heq] using hbound

/-- Every substep endpoint stays within the budget of the original outer step. -/
theorem outer_substeps_displacement {f : E → E} {h M : ℝ} {x : E} {n : ℕ}
    (hh : 0 ≤ h) (hM : 0 ≤ M) (hn : 0 < n)
    (hf : ∀ y ∈ Metric.closedBall x (h * M), ‖f y‖ ≤ M)
    (i : ℕ) (hi : i ≤ n) : ‖substeps f (h / n) i x - x‖ ≤ h * M := by
  exact (substeps_displacement (div_nonneg hh (Nat.cast_nonneg n)) hM hf i
    (by simpa using substep_budget hh hM hn hi)).trans (substep_budget hh hM hn hi)

/-- All four stage slopes retain the outer-step bound after any earlier substeps. -/
theorem outer_substeps_stageBounds {f : E → E} {h M : ℝ} {x : E} {n : ℕ}
    (hh : 0 ≤ h) (hM : 0 ≤ M) (hn : 0 < n)
    (hf : ∀ y ∈ Metric.closedBall x (h * M), ‖f y‖ ≤ M)
    (i : ℕ) (hi : i < n) : StageBounds f (h / n) (substeps f (h / n) i x) M := by
  exact substeps_stageBounds (div_nonneg hh (Nat.cast_nonneg n)) hM hf i
    (by simpa using substep_budget hh hM hn (Nat.succ_le_iff.mpr hi))

/-- Intermediate stages and weighted endpoints stay in the original outer-step ball. -/
theorem outer_substeps_stageRegion {f : E → E} {h M : ℝ} {x : E} {n : ℕ}
    (hh : 0 ≤ h) (hM : 0 ≤ M) (hn : 0 < n)
    (hf : ∀ y ∈ Metric.closedBall x (h * M), ‖f y‖ ≤ M)
    (i : ℕ) (hi : i < n) :
    StageRegion f (h / n) (substeps f (h / n) i x) x (h * M) := by
  exact substeps_stageRegion (div_nonneg hh (Nat.cast_nonneg n)) hM hf i
    (by simpa using substep_budget hh hM hn (Nat.succ_le_iff.mpr hi))

theorem step_sub_euler (f : E → E) (h : ℝ) (x : E) :
    step f h x - (x + h • f x) = h • (meanSlope f h x - f x) := by
  simp [step, smul_sub]

/-- A Lipschitz first-order consistency estimate, with the RK weights retained. -/
theorem StageBounds.euler_defect_le {f : E → E} {h M L : ℝ} {x : E}
    (hb : StageBounds f h x M) (hh : 0 ≤ h) (hM : 0 ≤ M) (hL : 0 ≤ L)
    (hlip : ∀ y ∈ Metric.closedBall x (h * M),
      ‖f y - f x‖ ≤ L * ‖y - x‖) :
    ‖step f h x - (x + h • f x)‖ ≤ L * M * h ^ 2 / 2 := by
  rw [step_sub_euler, norm_smul, Real.norm_eq_abs, abs_of_nonneg hh]
  calc
    _ ≤ h * (L * M * h / 2) :=
      mul_le_mul_of_nonneg_left (hb.meanSlope_sub_le hh hM hL hlip) hh
    _ = L * M * h ^ 2 / 2 := by ring

/-- Regional hypotheses suffice for the actual RK4 update's Euler defect. -/
theorem euler_defect_le {f : E → E} {h M L : ℝ} {x : E}
    (hh : 0 ≤ h) (hM : 0 ≤ M) (hL : 0 ≤ L)
    (hf : ∀ y ∈ Metric.closedBall x (h * M), ‖f y‖ ≤ M)
    (hlip : ∀ y ∈ Metric.closedBall x (h * M),
      ‖f y - f x‖ ≤ L * ‖y - x‖) :
    ‖step f h x - (x + h • f x)‖ ≤ L * M * h ^ 2 / 2 := by
  exact (stageBounds hh hM hf (by simp)).euler_defect_le hh hM hL hlip

end FractalSync.RungeKutta
