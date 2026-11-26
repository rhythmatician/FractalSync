import FractalSync.RungeKuttaStages
import FractalSync.TrajectoryError

/-!
# Phase-space residuals of actual Runge--Kutta substeps

The field is held fixed across every internal substep. Its first component is
the current velocity. The estimates compare the exact-real RK endpoint with
the existing velocity-first update, using averages of actual stage slopes.
They require no distance-to-set identity or high-order field derivatives.
-/

open scoped BigOperators

namespace FractalSync.RungeKutta

noncomputable section

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Average of all actual RK slopes over equal substeps of one outer step. -/
def averageSlope (f : E → E) (h : ℝ) (n : ℕ) (x : E) : E :=
  (n : ℝ)⁻¹ • ∑ i ∈ Finset.range n,
    meanSlope f (h / n) (substeps f (h / n) i x)

/-- The exact increment telescopes over the stored substep endpoints. -/
theorem substeps_eq_sum (f : E → E) (h : ℝ) (n : ℕ) (x : E) :
    substeps f h n x = x + h • ∑ i ∈ Finset.range n,
      meanSlope f h (substeps f h i x) := by
  induction n with
  | zero => simp [substeps]
  | succ n ih =>
    simpa only [substeps, step, Finset.sum_range_succ, smul_add, add_assoc] using
      congrArg (fun z => z + h • meanSlope f h (substeps f h n x)) ih

/-- Exact outer-step form, including the empty-substep convention. -/
theorem substeps_eq_averageSlope (f : E → E) (h : ℝ) (n : ℕ) (x : E) :
    substeps f (h / n) n x = x + h • averageSlope f h n x := by
  rw [substeps_eq_sum, averageSlope, smul_smul, div_eq_mul_inv]

/-- A nonempty finite average preserves a common vector norm bound. -/
theorem finite_average_norm_le (g : ℕ → E) (n : ℕ) (M : ℝ)
    (hn : 0 < n) (hg : ∀ i < n, ‖g i‖ ≤ M) :
    ‖(n : ℝ)⁻¹ • ∑ i ∈ Finset.range n, g i‖ ≤ M := by
  have hs : ‖∑ i ∈ Finset.range n, g i‖ ≤ (n : ℝ) * M :=
    (norm_sum_le _ _).trans (by
      simpa using (Finset.sum_le_sum (fun i hi => hg i (Finset.mem_range.mp hi))))
  simpa only [norm_smul, Real.norm_eq_abs, abs_of_nonneg
    (show 0 ≤ (n : ℝ)⁻¹ from inv_nonneg.mpr (Nat.cast_nonneg n)), ← mul_assoc,
    inv_mul_cancel₀ (show (n : ℝ) ≠ 0 from Nat.cast_ne_zero.mpr (Nat.ne_of_gt hn)), one_mul] using
      mul_le_mul_of_nonneg_left hs (inv_nonneg.mpr (Nat.cast_nonneg n))

/-- A bound for each actual substep mean bounds the full outer-step mean. -/
theorem averageSlope_norm_le (f : E → E) (h : ℝ) (n : ℕ) (x : E) (M : ℝ)
    (hn : 0 < n)
    (hg : ∀ i < n, ‖meanSlope f (h / n) (substeps f (h / n) i x)‖ ≤ M) :
    ‖averageSlope f h n x‖ ≤ M := by
  exact finite_average_norm_le _ n M hn hg

/-- Subtracting a fixed vector commutes with a nonempty finite average. -/
theorem finite_average_sub (g : ℕ → E) (n : ℕ) (v : E) (hn : 0 < n) :
    (n : ℝ)⁻¹ • ∑ i ∈ Finset.range n, (g i - v) =
      (n : ℝ)⁻¹ • ∑ i ∈ Finset.range n, g i - v := by
  simp only [Finset.sum_sub_distrib, Finset.sum_const, Finset.card_range,
    ← Nat.cast_smul_eq_nsmul ℝ, smul_sub, smul_smul,
    inv_mul_cancel₀ (show (n : ℝ) ≠ 0 from Nat.cast_ne_zero.mpr (Nat.ne_of_gt hn)), one_smul]

/-- A common deviation bound is preserved by the four positive RK weights. -/
theorem weighted_sub_norm_le {v₀ v₂ v₃ v₄ v : E} {M : ℝ}
    (h₀ : ‖v₀ - v‖ ≤ M) (h₂ : ‖v₂ - v‖ ≤ M)
    (h₃ : ‖v₃ - v‖ ≤ M) (h₄ : ‖v₄ - v‖ ≤ M) :
    ‖(1 / 6 : ℝ) • (v₀ + (2 : ℝ) • v₂ + (2 : ℝ) • v₃ + v₄) - v‖ ≤ M := by
  have heq : (1 / 6 : ℝ) • (v₀ + (2 : ℝ) • v₂ + (2 : ℝ) • v₃ + v₄) - v =
      (1 / 6 : ℝ) • ((v₀ - v) + (2 : ℝ) • (v₂ - v) +
        (2 : ℝ) • (v₃ - v) + (v₄ - v)) := by module
  simpa only [heq] using weighted_norm_le h₀ h₂ h₃ h₄

section Phase

open ScaleMetric

variable {F : Type*} [NormedAddCommGroup F] [InnerProductSpace ℝ F]

noncomputable local instance : NormedSpace ℝ (F × F) := Prod.normedSpace

/-- Actual acceleration average, retaining one field across all substeps. -/
def averageAcceleration (acc : F × F → F) (h : ℝ) (n : ℕ) (x : F × F) : F :=
  (averageSlope (phaseField acc) h n x).2

/-- The first RK mean component averages the actual stage velocities. -/
theorem meanSlope_fst_sub_le (acc : F × F → F) (h : ℝ) (x : F × F) (v : F) (R : ℝ)
    (h₀ : ‖x.2 - v‖ ≤ R)
    (h₂ : ‖(stage2 (phaseField acc) h x).2 - v‖ ≤ R)
    (h₃ : ‖(stage3 (phaseField acc) h x).2 - v‖ ≤ R)
    (h₄ : ‖(stage4 (phaseField acc) h x).2 - v‖ ≤ R) :
    ‖(meanSlope (phaseField acc) h x).1 - v‖ ≤ R := by
  simpa only [meanSlope, Prod.smul_fst, Prod.fst_add, phaseField] using
    weighted_sub_norm_le h₀ h₂ h₃ h₄

/-- The second RK mean component averages the actual stage accelerations. -/
theorem meanSlope_snd_sub_le (acc : F × F → F) (h : ℝ) (x : F × F) (a : F) (R : ℝ)
    (h₀ : ‖acc x - a‖ ≤ R)
    (h₂ : ‖acc (stage2 (phaseField acc) h x) - a‖ ≤ R)
    (h₃ : ‖acc (stage3 (phaseField acc) h x) - a‖ ≤ R)
    (h₄ : ‖acc (stage4 (phaseField acc) h x) - a‖ ≤ R) :
    ‖(meanSlope (phaseField acc) h x).2 - a‖ ≤ R := by
  simpa only [meanSlope, Prod.smul_snd, Prod.snd_add, phaseField] using
    weighted_sub_norm_le h₀ h₂ h₃ h₄

/-- Substep averaging preserves a common deviation of the velocity means. -/
theorem averageSlope_fst_sub_le (acc : F × F → F) (h : ℝ) (n : ℕ)
    (x : F × F) (v : F) (R : ℝ) (hn : 0 < n)
    (hg : ∀ i < n,
      ‖(meanSlope (phaseField acc) (h / n)
        (substeps (phaseField acc) (h / n) i x)).1 - v‖ ≤ R) :
    ‖(averageSlope (phaseField acc) h n x).1 - v‖ ≤ R := by
  simpa only [finite_average_sub _ n v hn, averageSlope, Prod.smul_fst, Prod.fst_sum] using
    finite_average_norm_le
      (fun i => (meanSlope (phaseField acc) (h / n)
        (substeps (phaseField acc) (h / n) i x)).1 - v) n R hn hg

/-- Substep averaging preserves a common deviation of the acceleration means. -/
theorem averageAcceleration_sub_le (acc : F × F → F) (h : ℝ) (n : ℕ)
    (x : F × F) (a : F) (R : ℝ) (hn : 0 < n)
    (hg : ∀ i < n,
      ‖(meanSlope (phaseField acc) (h / n)
        (substeps (phaseField acc) (h / n) i x)).2 - a‖ ≤ R) :
    ‖averageAcceleration acc h n x - a‖ ≤ R := by
  simpa only [finite_average_sub _ n a hn, averageAcceleration, averageSlope,
    Prod.smul_snd, Prod.snd_sum] using finite_average_norm_le
      (fun i => (meanSlope (phaseField acc) (h / n)
        (substeps (phaseField acc) (h / n) i x)).2 - a) n R hn hg

/-- Velocity agrees exactly with the ordered update using the actual RK
acceleration average. Only the position component contributes a residual. -/
theorem substeps_ordered_residual_eq (acc : F × F → F) (h : ℝ)
    (n : ℕ) (x : F × F) :
    substeps (phaseField acc) (h / n) n x -
      positionVelocityStep h x.1 x.2 (averageAcceleration acc h n x) =
    (h • ((averageSlope (phaseField acc) h n x).1 - x.2 -
      h • averageAcceleration acc h n x), 0) := by
  rw [substeps_eq_averageSlope]
  apply Prod.ext
  all_goals dsimp [positionVelocityStep, velocityEulerStep, averageAcceleration]
  all_goals module

/-- A first-order stage budget gives the quadratic stored-step residual used
by the existing trajectory certificate. This is not an IEEE rounding claim. -/
theorem substeps_ordered_residual_le (acc : F × F → F) (h M : ℝ)
    (n : ℕ) (x : F × F) (hh : 0 ≤ h)
    (hv : ‖(averageSlope (phaseField acc) h n x).1 - x.2‖ ≤ h * M)
    (ha : ‖averageAcceleration acc h n x‖ ≤ M) :
    ‖substeps (phaseField acc) (h / n) n x -
      positionVelocityStep h x.1 x.2 (averageAcceleration acc h n x)‖ ≤
        2 * M * h ^ 2 := by
  have ha' : ‖h • averageAcceleration acc h n x‖ ≤ h * M := by
    simpa only [norm_smul, Real.norm_eq_abs, abs_of_nonneg hh] using
      mul_le_mul_of_nonneg_left ha hh
  have ht := (norm_sub_le ((averageSlope (phaseField acc) h n x).1 - x.2)
    (h • averageAcceleration acc h n x)).trans (add_le_add hv ha')
  simp only [substeps_ordered_residual_eq, Prod.norm_mk, norm_zero,
    norm_smul, Real.norm_eq_abs, abs_of_nonneg hh]
  rw [max_eq_left (mul_nonneg hh (norm_nonneg _))]
  nlinarith [mul_le_mul_of_nonneg_left ht hh]

/-- The phase adapter uses the actual substep certificates and compares all
accelerations with the same outer-step target. -/
theorem phase_substeps_residuals_of_stages (acc : F × F → F) (target : F)
    (h M L : ℝ) (n : ℕ) (x : F × F) (hh : 0 ≤ h) (hL : 0 ≤ L) (hn : 0 < n)
    (hb : ∀ i < n, StageBounds (phaseField acc) (h / n)
      (substeps (phaseField acc) (h / n) i x) M)
    (hr : ∀ i < n, StageRegion (phaseField acc) (h / n)
      (substeps (phaseField acc) (h / n) i x) x (h * M))
    (ha : ∀ y ∈ Metric.closedBall x (h * M), ‖acc y - target‖ ≤ L * ‖y - x‖) :
    ‖averageAcceleration acc h n x - target‖ ≤ L * M * h ∧
      ‖substeps (phaseField acc) (h / n) n x -
        positionVelocityStep h x.1 x.2 (averageAcceleration acc h n x)‖ ≤ 2 * M * h ^ 2 := by
  have hv : ∀ y ∈ Metric.closedBall x (h * M), ‖y.2 - x.2‖ ≤ h * M :=
    fun y hy => (norm_snd_le (y - x)).trans (mem_closedBall_iff_norm.mp hy)
  have hvavg : ‖(averageSlope (phaseField acc) h n x).1 - x.2‖ ≤ h * M :=
    averageSlope_fst_sub_le acc h n x x.2 (h * M) hn (fun i hi =>
      meanSlope_fst_sub_le acc (h / n) _ x.2 (h * M)
        (hv _ (hr i hi).start_mem) (hv _ (hr i hi).stage2_mem)
        (hv _ (hr i hi).stage3_mem) (hv _ (hr i hi).stage4_mem))
  have havar : ∀ y ∈ Metric.closedBall x (h * M), ‖acc y - target‖ ≤ L * M * h :=
    fun y hy => (ha y hy).trans (by
      nlinarith [mul_le_mul_of_nonneg_left (mem_closedBall_iff_norm.mp hy) hL])
  have haavg : ‖averageAcceleration acc h n x - target‖ ≤ L * M * h :=
    averageAcceleration_sub_le acc h n x target (L * M * h) hn (fun i hi =>
      meanSlope_snd_sub_le acc (h / n) _ target (L * M * h)
        (havar _ (hr i hi).start_mem) (havar _ (hr i hi).stage2_mem)
        (havar _ (hr i hi).stage3_mem) (havar _ (hr i hi).stage4_mem))
  have hanorm : ‖averageAcceleration acc h n x‖ ≤ M :=
    (norm_snd_le (averageSlope (phaseField acc) h n x)).trans
      (averageSlope_norm_le _ h n x M hn (fun i hi => (hb i hi).meanSlope_norm))
  exact ⟨haavg, substeps_ordered_residual_le acc h M n x hh hvavg hanorm⟩

/-- Actual RK substeps supply both residuals needed by the existing trajectory
theorem. All stage containment and slope bounds follow from the regional
field bound. The same acceleration field and outer anchor are retained for
every internal substep, including the runtime's one- and four-substep cases. -/
theorem phase_substeps_residuals (acc : F × F → F) (target : F)
    (h M L : ℝ) (n : ℕ) (x : F × F)
    (hh : 0 ≤ h) (hM : 0 ≤ M) (hL : 0 ≤ L) (hn : 0 < n)
    (hf : ∀ y ∈ Metric.closedBall x (h * M), ‖phaseField acc y‖ ≤ M)
    (ha : ∀ y ∈ Metric.closedBall x (h * M), ‖acc y - target‖ ≤ L * ‖y - x‖) :
    ‖averageAcceleration acc h n x - target‖ ≤ L * M * h ∧
      ‖substeps (phaseField acc) (h / n) n x -
        positionVelocityStep h x.1 x.2 (averageAcceleration acc h n x)‖ ≤ 2 * M * h ^ 2 := by
  exact phase_substeps_residuals_of_stages acc target h M L n x hh hL hn
    (fun i hi => outer_substeps_stageBounds hh hM hn hf i hi)
    (fun i hi => outer_substeps_stageRegion hh hM hn hf i hi) ha

end Phase

end

end FractalSync.RungeKutta
