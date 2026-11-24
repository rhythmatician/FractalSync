/-
Copyright (c) 2025 Yizheng Zhu. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Yizheng Zhu

Adapted for FractalSync's Lean 4.24.0 pin from mathlib4 commit
 aad247f928b1eb7c58478742119fdd15a37f0ffd.
-/

import Mathlib.MeasureTheory.Function.AbsolutelyContinuous
import Mathlib.Analysis.BoundedVariation
import Mathlib.Analysis.Calculus.Deriv.Slope
import Mathlib.Analysis.SpecificLimits.Basic
import FractalSync.Analysis.Fatou

import Mathlib.Algebra.Order.Interval.Set.Group
import FractalSync.Analysis.IntervalSlope

/-!
# `f'` is interval integrable for certain classes of functions `f`

This file proves that:
* `MonotoneOn.intervalIntegrable_deriv`: If `f` is monotone on `a..b`, then `f'` is interval
  integrable on `a..b`.
* `BoundedVariationOn.intervalIntegrable_deriv`: If `f` has bounded variation on `a..b`,
  then `f'` is interval integrable on `a..b`.
* `AbsolutelyContinuousOnInterval.intervalIntegrable_deriv`: If `f` is absolutely continuous on
  `a..b`, then `f'` is interval integrable on `a..b`.

## Tags

interval integrable, monotone, bounded variation, absolutely continuous
-/


open MeasureTheory Set Filter
open FractalSync.Analysis

open scoped Topology

/-- If `f` is monotone on `[a, b]`, then `f'` is the limit of `G n` a.e. on `[a, b]`, where each
`G n` is `AEStronglyMeasurable` and the liminf of the lower Lebesgue integral of `‖G n ·‖ₑ` is at
most `f b - f a`. -/
lemma MonotoneOn.exists_tendsto_deriv_liminf_lintegral_enorm_le
    {f : ℝ → ℝ} {a b : ℝ} (hab : a ≤ b) (hf : MonotoneOn f (Icc a b)) :
    ∃ G : (ℕ → ℝ → ℝ), (∀ᵐ x ∂volume.restrict (Icc a b),
      Filter.Tendsto (fun (n : ℕ) ↦ G n x) Filter.atTop (𝓝 (deriv f x))) ∧
      (∀ (n : ℕ), AEStronglyMeasurable (G n) (volume.restrict (Icc a b))) ∧
      liminf (fun (n : ℕ) ↦ ∫⁻ (x : ℝ) in Icc a b, ‖G n x‖ₑ) atTop ≤
        ENNReal.ofReal (f b - f a) := by
  /- Proof Sketch: Extend `f` on `[a, b]` to a function `g` on `ℝ` by defining `g x = f a` for
  `x < a` and `g x = f b` for `x > b`. `g` is globally monotone and `g'` agrees with `f'` on
  `(a, b)`. We let `G c x = slope g x (x + c)` for `c > 0`. Then `G c x` is nonnegative,
  `∫⁻ (x : ℝ) in Icc a b, ‖G c x‖ₑ ≤ f b - f a`, and `G c x` tends to `f' x` as `c` tends to `0`
  from the right. The function `fun n x ↦ G (n : ℝ)⁻¹ x` is a witness to the conclusion of the
  lemma. -/
  let g (x : ℝ) : ℝ := f (max a (min x b))
  have hg : Monotone g := fun x y hxy =>
    hf ⟨le_max_left _ _, max_le hab (min_le_right _ _)⟩
      ⟨le_max_left _ _, max_le hab (min_le_right _ _)⟩
      (max_le_max le_rfl (min_le_min hxy le_rfl))
  have hfg : EqOn f g (Ioo a b) := fun x ⟨hxa, hxb⟩ => congrArg f <|
    min_eq_left hxb.le |>.symm ▸ (max_eq_right hxa.le).symm
  replace hfg := hfg.deriv isOpen_Ioo
  let G (c x : ℝ) := slope g x (x + c)
  have G_integrable (n : ℕ) : Integrable (G (↑n)⁻¹) (volume.restrict (Icc a b)) := by
    have := hg.monotoneOn (Icc a (b + (n : ℝ)⁻¹)) |>.intervalIntegrable_slope hab (by simp)
    exact intervalIntegrable_iff_integrableOn_Icc_of_le hab |>.mp this
  refine ⟨fun n x ↦ G (n : ℝ)⁻¹ x, ?_, fun n ↦ G_integrable n |>.aestronglyMeasurable, ?_⟩
  · rw [← restrict_Ioo_eq_restrict_Icc, MeasureTheory.ae_restrict_iff' measurableSet_Ioo]
    filter_upwards [hg.ae_differentiableAt] with x hx₁ hx₂
    rw [hfg hx₂]
    exact (hasDerivAt_iff_tendsto_slope.mp hx₁.hasDerivAt).comp <|
      tendsto_nhdsWithin_of_tendsto_nhds_of_eventually_within _
      (by simpa using tendsto_const_nhds.add tendsto_inverse_atTop_nhds_zero_nat)
      (by simp [eventually_ne_atTop 0])
  · calc
      _ = liminf (fun (n : ℕ) ↦ ENNReal.ofReal (∫ (x : ℝ) in Icc a b, (G (n : ℝ)⁻¹) x)) atTop := by
        apply Filter.liminf_congr
        filter_upwards with n
        rw [← MeasureTheory.ofReal_integral_norm_eq_lintegral_enorm (G_integrable n)]
        congr with y
        apply abs_eq_self.mpr
        simp only [G, slope, add_sub_cancel_left, vsub_eq_sub, smul_eq_mul]
        exact mul_nonneg (by positivity)
          (sub_nonneg.mpr (hg (le_add_of_nonneg_right (by positivity))))
      _ ≤ ENNReal.ofReal (g b - g a) := by
        refine Filter.liminf_le_of_frequently_le'
          (Filter.Frequently.of_forall fun n ↦ ENNReal.ofReal_le_ofReal ?_)
        rw [integral_Icc_eq_integral_Ioc, ← intervalIntegral.integral_of_le hab]
        convert
          hg.monotoneOn (Icc a (b + (n : ℝ)⁻¹)) |>.intervalIntegral_slope_le hab (by simp) using 2
        simp [g]
      _ = ENNReal.ofReal (f b - f a) := by grind

/-- If `f` is monotone on `a..b`, then `f'` is interval integrable on `a..b`. -/
theorem MonotoneOn.intervalIntegrable_deriv {f : ℝ → ℝ} {a b : ℝ}
    (hf : MonotoneOn f (uIcc a b)) :
    IntervalIntegrable (deriv f) volume a b := by
  wlog hab : a ≤ b generalizing a b with h
  · exact h (uIcc_comm a b ▸ hf) (by linarith) |>.symm
  rw [uIcc_of_le hab] at hf
  obtain ⟨G, hGf, hG, hG'⟩ := hf.exists_tendsto_deriv_liminf_lintegral_enorm_le hab
  have hG'₀ : liminf (fun (n : ℕ) ↦ ∫⁻ (x : ℝ) in Icc a b, ‖G n x‖ₑ) atTop ≠ ⊤ :=
    lt_of_le_of_lt hG' ENNReal.ofReal_lt_top |>.ne_top
  have integrable_f_deriv := integrable_of_tendsto hGf hG hG'₀
  exact (intervalIntegrable_iff_integrableOn_Icc_of_le hab).mpr integrable_f_deriv

/-- If `f` has bounded variation on `uIcc a b`, then `f'` is interval integrable on `a..b`. -/
theorem BoundedVariationOn.intervalIntegrable_deriv {f : ℝ → ℝ} {a b : ℝ}
    (hf : BoundedVariationOn f (uIcc a b)) :
    IntervalIntegrable (deriv f) volume a b := by
  obtain ⟨p, q, hp, hq, rfl⟩ := hf.locallyBoundedVariationOn.exists_monotoneOn_sub_monotoneOn
  have h₂ : ∀ᵐ x, x ≠ max a b := by simp [ae_iff, measure_singleton]
  apply (hp.intervalIntegrable_deriv.sub hq.intervalIntegrable_deriv).congr_ae
  rw [Filter.EventuallyEq, MeasureTheory.ae_restrict_iff' (by simp [uIoc])]
  filter_upwards [hp.ae_differentiableWithinAt_of_mem, hq.ae_differentiableWithinAt_of_mem, h₂]
    with x hx₁ hx₂ hx₃ hx₄
  have hx₅ : x ∈ uIcc a b := Ioc_subset_Icc_self hx₄
  rw [uIoc, mem_Ioc] at hx₄
  have hx₆ : uIcc a b ∈ 𝓝 x := Icc_mem_nhds hx₄.left (lt_of_le_of_ne hx₄.right hx₃)
  replace hx₁ := (hx₁ hx₅).differentiableAt hx₆ |>.hasDerivAt
  replace hx₂ := (hx₂ hx₅).differentiableAt hx₆ |>.hasDerivAt
  exact (hx₁.sub hx₂).deriv.symm

/-- If `f` is absolutely continuous on `uIcc a b`, then `f'` is interval integrable on `a..b`. -/
theorem AbsolutelyContinuousOnInterval.intervalIntegrable_deriv {f : ℝ → ℝ} {a b : ℝ}
    (hf : AbsolutelyContinuousOnInterval f a b) :
    IntervalIntegrable (deriv f) volume a b :=
  hf.boundedVariationOn.intervalIntegrable_deriv
