import FractalSync.ControlForceBounds

/-!
# Cancellation-preserving bounds for the rank-one inverse

The inverse of the unscaled positive graph operator is a Euclidean contraction.
Its resolvent identity gives regional variation estimates without expanding the
Sherman--Morrison denominator into independent positive terms.
-/

noncomputable section

namespace FractalSync.ScaleMetric

open Set
open scoped InnerProductSpace

variable {E X : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [PseudoMetricSpace X]

/-- The positive rank-one graph operator cannot decrease Euclidean norm. -/
theorem norm_le_norm_stretch (a : ℝ) (u v : E) : ‖v‖ ≤ ‖stretch a u v‖ := by
  have h : ⟪v, stretch a u v⟫_ℝ = ‖v‖ ^ 2 + a ^ 2 * ⟪u, v⟫_ℝ ^ 2 := by
    simp [stretch, inner_add_right, real_inner_smul_right, real_inner_self_eq_norm_sq,
      real_inner_comm v u, pow_two, mul_assoc]
  nlinarith [real_inner_le_norm v (stretch a u v), norm_nonneg v,
    norm_nonneg (stretch a u v), mul_nonneg (sq_nonneg a) (sq_nonneg ⟪u, v⟫_ℝ)]

/-- The inverse unscaled graph operator is a contraction, including zero gradients. -/
theorem norm_unstretch_le (a : ℝ) (u v : E) : ‖unstretch a u v‖ ≤ ‖v‖ := by
  simpa only [stretch_unstretch] using norm_le_norm_stretch a u (unstretch a u v)

/-- Sharp scale factor for the inverse metric; no anisotropy factor is lost. -/
theorem norm_inverseAction_le (rho a : ℝ) (u v : E) :
    ‖inverseAction rho a u v‖ ≤ rho ^ 2 * ‖v‖ := by
  simpa only [inverseAction, norm_smul, Real.norm_of_nonneg (sq_nonneg rho)] using
    mul_le_mul_of_nonneg_left (norm_unstretch_le a u v) (sq_nonneg rho)

/-- Subtraction is preserved by the inverse-metric action. -/
theorem inverseAction_sub (rho a : ℝ) (u v w : E) :
    inverseAction rho a u (v - w) = inverseAction rho a u v - inverseAction rho a u w := by
  simp only [inverseAction, unstretch, inner_sub_right, mul_sub, sub_div,
    sub_smul, smul_sub]
  abel

/-- Subtraction is preserved by the unscaled inverse action. -/
theorem unstretch_sub (a : ℝ) (u v w : E) :
    unstretch a u (v - w) = unstretch a u v - unstretch a u w := by
  simp only [unstretch, inner_sub_right, mul_sub, sub_div, sub_smul]
  abel

/-- Resolvent identity for changing the rank-one direction. -/
theorem unstretch_resolvent (a : ℝ) (u w v : E) :
    unstretch a u v - unstretch a w v =
      unstretch a u ((a ^ 2 * ⟪w, unstretch a w v⟫_ℝ) • w -
        (a ^ 2 * ⟪u, unstretch a w v⟫_ℝ) • u) := by
  have h : (a ^ 2 * ⟪w, unstretch a w v⟫_ℝ) • w -
      (a ^ 2 * ⟪u, unstretch a w v⟫_ℝ) • u =
      stretch a w (unstretch a w v) - stretch a u (unstretch a w v) := by
    simp only [stretch, add_sub_add_left_eq_sub]
  rw [h, stretch_unstretch, unstretch_sub, unstretch_stretch]

/-- Variation of a rank-one outer product in its defining vector. -/
theorem norm_rankOne_sub_le (u w v : E) :
    ‖⟪w, v⟫_ℝ • w - ⟪u, v⟫_ℝ • u‖ ≤
      (‖u‖ + ‖w‖) * ‖u - w‖ * ‖v‖ := by
  have h : ⟪w, v⟫_ℝ • w - ⟪u, v⟫_ℝ • u =
      ⟪w, v⟫_ℝ • (w - u) + ⟪w - u, v⟫_ℝ • u := by
    simp only [inner_sub_left, smul_sub, sub_smul]
    abel
  calc
    _ = ‖⟪w, v⟫_ℝ • (w - u) + ⟪w - u, v⟫_ℝ • u‖ := congrArg norm h
    _ ≤ ‖⟪w, v⟫_ℝ • (w - u)‖ + ‖⟪w - u, v⟫_ℝ • u‖ := norm_add_le _ _
    _ ≤ (‖w‖ * ‖v‖) * ‖w - u‖ + (‖w - u‖ * ‖v‖) * ‖u‖ := by
      simpa only [norm_smul] using add_le_add
        (mul_le_mul_of_nonneg_right (norm_inner_le_norm w v) (norm_nonneg (w - u)))
        (mul_le_mul_of_nonneg_right (norm_inner_le_norm (w - u) v) (norm_nonneg u))
    _ = (‖u‖ + ‖w‖) * ‖u - w‖ * ‖v‖ := by
      rw [norm_sub_rev w u]
      ring

/-- Resolvent variation uses contraction at both ends. -/
theorem norm_unstretch_parameter_sub_le (a : ℝ) (u w v : E) :
    ‖unstretch a u v - unstretch a w v‖ ≤
      a ^ 2 * (‖u‖ + ‖w‖) * ‖u - w‖ * ‖v‖ := by
  rw [unstretch_resolvent]
  refine (norm_unstretch_le a u _).trans ?_
  rw [mul_smul, mul_smul, ← smul_sub, norm_smul, Real.norm_of_nonneg (sq_nonneg a)]
  calc
    _ ≤ a ^ 2 * ((‖u‖ + ‖w‖) * ‖u - w‖ * ‖unstretch a w v‖) :=
      mul_le_mul_of_nonneg_left (norm_rankOne_sub_le u w _) (sq_nonneg a)
    _ ≤ a ^ 2 * ((‖u‖ + ‖w‖) * ‖u - w‖ * ‖v‖) :=
      mul_le_mul_of_nonneg_left (mul_le_mul_of_nonneg_left (norm_unstretch_le a w v)
        (mul_nonneg (add_nonneg (norm_nonneg _) (norm_nonneg _)) (norm_nonneg _))) (sq_nonneg a)
    _ = _ := by ring

/-- Regional inverse-action constants preserve the contraction and resolvent cancellations. -/
theorem unstretch_boundedLipschitz (a : ℝ) (s : Set X) (u v : X → E)
    (U Lu V Lv : ℝ) (hu : BoundedLipschitzOn u s U Lu)
    (hv : BoundedLipschitzOn v s V Lv) :
    BoundedLipschitzOn (fun x => unstretch a (u x) (v x)) s V
      (Lv + 2 * a ^ 2 * U * Lu * V) := by
  refine ⟨hv.bound_nonneg, add_nonneg hv.lip_nonneg (mul_nonneg
    (mul_nonneg (mul_nonneg (mul_nonneg (by norm_num) (sq_nonneg a)) hu.bound_nonneg)
      hu.lip_nonneg) hv.bound_nonneg), fun x hx =>
    (norm_unstretch_le a (u x) (v x)).trans (hv.norm_le x hx), ?_⟩
  intro x hx y hy
  calc
    _ ≤ ‖unstretch a (u x) (v x) - unstretch a (u x) (v y)‖ +
        ‖unstretch a (u x) (v y) - unstretch a (u y) (v y)‖ :=
      norm_sub_le_norm_sub_add_norm_sub _ _ _
    _ ≤ ‖v x - v y‖ + a ^ 2 * (‖u x‖ + ‖u y‖) * ‖u x - u y‖ * ‖v y‖ :=
      add_le_add (by rw [← unstretch_sub]; exact norm_unstretch_le a (u x) _)
        (norm_unstretch_parameter_sub_le a (u x) (u y) (v y))
    _ ≤ Lv * dist x y + a ^ 2 * (U + U) * (Lu * dist x y) * V := by
      gcongr
      all_goals first
        | exact hv.norm_sub_le x hx y hy
        | exact hu.norm_le x hx
        | exact hu.norm_le y hy
        | exact hu.norm_sub_le x hx y hy
        | exact hv.norm_le y hy
        | exact mul_nonneg (mul_nonneg (sq_nonneg a) (add_nonneg hu.bound_nonneg hu.bound_nonneg))
            (mul_nonneg hu.lip_nonneg dist_nonneg)
        | exact mul_nonneg (sq_nonneg a) (add_nonneg hu.bound_nonneg hu.bound_nonneg)
    _ = _ := by ring

/-- Sharp inverse norm and resolvent Lipschitz factor, applied to a varying covector. -/
theorem inverseAction_boundedLipschitz (a : ℝ) (s : Set X) (rho : X → ℝ) (u v : X → E)
    (R Lr U Lu V Lv : ℝ) (hr : BoundedLipschitzOn rho s R Lr)
    (hu : BoundedLipschitzOn u s U Lu) (hv : BoundedLipschitzOn v s V Lv) :
    BoundedLipschitzOn (fun x => inverseAction (rho x) a (u x) (v x)) s
      (R ^ 2 * V) (R ^ 2 * Lv + (2 * R * Lr + 2 * R ^ 2 * a ^ 2 * U * Lu) * V) := by
  convert (hr.mul hr).smul (unstretch_boundedLipschitz a s u v U Lu V Lv hu hv) using 1
  all_goals simp only [inverseAction, pow_two]
  all_goals ring

end FractalSync.ScaleMetric
