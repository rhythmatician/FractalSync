/-
Copyright (c) 2025 Yizheng Zhu. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Yizheng Zhu
-/
import Mathlib.MeasureTheory.Function.AbsolutelyContinuous
import Mathlib.Analysis.BoundedVariation
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic
import Mathlib.Tactic.GRewrite
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.FieldSimp

/-!
# Selected upstream absolute-continuity lemmas

Backported from Mathlib commit aad247f928b1eb7c58478742119fdd15a37f0ffd
for FractalSync's pinned Lean 4.24.0 environment. These extend, rather than
replace, Mathlib's actual `AbsolutelyContinuousOnInterval` definition.
The finite-dimensional almost-everywhere derivative lemma is proved here from
the pinned bounded-variation theorem.
-/

open Set Filter Function MeasureTheory
open scoped Topology NNReal

namespace AbsolutelyContinuousOnInterval

variable {X : Type*} [PseudoMetricSpace X]

lemma uIoc_subset_of_mem_disjWithin {a b : ℝ} {n : ℕ} {I : ℕ → ℝ × ℝ}
    (hnI : (n, I) ∈ disjWithin a b) {i : ℕ} (hi : i < n) : uIoc (I i).1 (I i).2 ⊆ uIoc a b := by
  simp only [disjWithin, Finset.mem_range, mem_setOf_eq, uIcc, mem_Icc] at hnI
  grind [uIoc]

lemma biUnion_uIoc_subset_of_mem_disjWithin {a b : ℝ} {n : ℕ} {I : ℕ → ℝ × ℝ}
    (hnI : (n, I) ∈ disjWithin a b) :
    (⋃ i ∈ Finset.range n, uIoc (I i).1 (I i).2) ⊆ uIoc a b := by
  simp only [iUnion_subset_iff, Finset.mem_range]
  exact fun i hi ↦ uIoc_subset_of_mem_disjWithin hnI hi

lemma tendsto_volume_totalLengthFilter_nhds_zero :
    Tendsto (fun E : ℕ × (ℕ → ℝ × ℝ) ↦ volume (⋃ i ∈ Finset.range E.1, uIoc (E.2 i).1 (E.2 i).2))
    totalLengthFilter (𝓝 0) := by
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds
    (h := fun E ↦ ENNReal.ofReal (∑ i ∈ Finset.range E.1, (dist (E.2 i).1 (E.2 i).2)))
  · convert ENNReal.tendsto_ofReal (Filter.tendsto_comap)
    simp
  · intro; simp
  · intro E
    simp only
    grw [measure_biUnion_finset_le]
    rw [ENNReal.ofReal_sum_of_nonneg (fun _ _ ↦ dist_nonneg)]
    apply Eq.le
    apply Finset.sum_congr rfl
    simp [uIoc, Real.dist_eq, max_sub_min_eq_abs']

lemma tendsto_volume_restrict_totalLengthFilter_disjWithin_nhds_zero (a b : ℝ) :
    Tendsto (fun E : ℕ × (ℕ → ℝ × ℝ) ↦ volume.restrict (uIoc a b)
        (⋃ i ∈ Finset.range E.1, uIoc (E.2 i).1 (E.2 i).2))
      (totalLengthFilter ⊓ 𝓟 (disjWithin a b))
      (𝓝 0) := by
  apply tendsto_of_tendsto_of_tendsto_of_le_of_le tendsto_const_nhds
    (h := fun E : ℕ × (ℕ → ℝ × ℝ) ↦ volume (⋃ i ∈ Finset.range E.1, uIoc (E.2 i).1 (E.2 i).2))
  · apply tendsto_volume_totalLengthFilter_nhds_zero.mono_left
    simp
  · intro; simp
  · intro E
    simp only [Finset.mem_range]
    apply Measure.restrict_le_self

theorem _root_.LipschitzOnWith.comp_absolutelyContinuousOnInterval
    {Y : Type*} [PseudoMetricSpace Y] {f : X → Y} {K : ℝ≥0} {t : Set X}
    (hf : LipschitzOnWith K f t) {g : ℝ → X} {a b : ℝ} (hg : MapsTo g (uIcc a b) t)
    (h : AbsolutelyContinuousOnInterval g a b) :
    AbsolutelyContinuousOnInterval (f ∘ g) a b := by
  apply squeeze_zero' ?_ ?_ (by simpa using Tendsto.const_mul (K : ℝ) h)
  · exact .of_forall fun _ ↦ by positivity
  rw [eventually_inf_principal]
  filter_upwards with (n, I) hnI
  rw [Finset.mul_sum]
  exact Finset.sum_le_sum fun i hi ↦
    hf.dist_le_mul _ (hg (hnI.left i hi).left) _ (hg (hnI.left i hi).right)

/-- If `f` is Lipschitz and `g` is absolutely continuous on `uIcc a b`, then `f ∘ g` is absolutely
continuous on `uIcc a b`. -/
theorem _root_.LipschitzWith.comp_absolutelyContinuousOnInterval
    {Y : Type*} [PseudoMetricSpace Y] {f : X → Y} {K : ℝ≥0} (hf : LipschitzWith K f)
    {g : ℝ → X} {a b : ℝ} (h : AbsolutelyContinuousOnInterval g a b) :
    AbsolutelyContinuousOnInterval (f ∘ g) a b :=
  hf.lipschitzOnWith.comp_absolutelyContinuousOnInterval (mapsTo_univ _ _) h


/-- Absolutely continuous finite-dimensional curves have derivatives almost
 everywhere on their closed parameter interval. Endpoints are a null set. -/
theorem ae_differentiableAt {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [FiniteDimensional ℝ E] {f : ℝ → E} {a b : ℝ}
    (hf : AbsolutelyContinuousOnInterval f a b) :
    ∀ᵐ (x : ℝ), x ∈ uIcc a b → DifferentiableAt ℝ f x := by
  have hdiff := hf.boundedVariationOn.locallyBoundedVariationOn.ae_differentiableWithinAt_of_mem
  have ha : ∀ᵐ x : ℝ, x ≠ min a b := by simp [ae_iff, measure_singleton]
  have hb : ∀ᵐ x : ℝ, x ≠ max a b := by simp [ae_iff, measure_singleton]
  filter_upwards [hdiff, ha, hb] with x hx hxa hxb hmem
  apply (hx hmem).differentiableAt
  exact Icc_mem_nhds (lt_of_le_of_ne hmem.1 hxa.symm) (lt_of_le_of_ne hmem.2 hxb)

theorem _root_.IntervalIntegrable.absolutelyContinuousOnInterval_intervalIntegral {f : ℝ → ℝ}
    {a b c : ℝ} (h : IntervalIntegrable f volume a b) (hc : c ∈ uIcc a b) :
    AbsolutelyContinuousOnInterval (fun x ↦ ∫ v in c..x, f v) a b := by
  -- Step 1: Use `MeasureTheory.tendsto_setLIntegral_zero` to conclude that the function sending
  -- `E` to `∫⁻ (x : ℝ) in s E, ‖f x‖ₑ ∂volume.restrict (uIoc a b)` tends to `0` along
  -- `totalLengthFilter ⊓ 𝓟 (disjWithin a b)`.
  let s := fun E : ℕ × (ℕ → ℝ × ℝ) ↦ ⋃ i ∈ Finset.range E.1, uIoc (E.2 i).1 (E.2 i).2
  have : Tendsto (fun i ↦ ∫⁻ (x : ℝ) in s i, ‖f x‖ₑ ∂volume.restrict (uIoc a b))
      (totalLengthFilter ⊓ 𝓟 (disjWithin a b)) (𝓝 0) :=
    tendsto_setLIntegral_zero
    (ne_of_lt <| intervalIntegrable_iff.mp h |>.hasFiniteIntegral)
    (tendsto_volume_restrict_totalLengthFilter_disjWithin_nhds_zero _ _)
  -- Step 2: Use the lintegral in Step 1 to bound the sum of the distances between
  -- `∫ v in c..(E.2 i).2, f v` and `∫ v in c..(E.2 i).2, f v` that occurs in the definition
  -- of absolutely continuous.
  have := ENNReal.toReal_zero ▸ (ENNReal.continuousAt_toReal (by simp)).tendsto.comp this
  refine squeeze_zero' ?_ ?_ this
  · filter_upwards with (n, I)
    exact Finset.sum_nonneg (fun _ _ ↦ dist_nonneg)
  simp only [comp_apply, s]
  have : ∀ᶠ (E : ℕ × (ℕ → ℝ × ℝ)) in totalLengthFilter ⊓ 𝓟 (disjWithin a b),
      E ∈ disjWithin a b :=
    eventually_inf_principal.mpr (by simp)
  filter_upwards [this] with (n, I) hnI
  obtain ⟨hnI1, hnI2⟩ := mem_setOf_eq ▸ hnI
  simp only
  rw [← integral_norm_eq_lintegral_enorm (h.def'.aestronglyMeasurable.restrict),
      integral_biUnion_finset _ (by simp +contextual [uIoc]) hnI2]
  · refine Finset.sum_le_sum (fun i hi ↦ ?_)
    rw [Real.dist_eq,
        intervalIntegral.integral_interval_sub_left
          (by apply IntervalIntegrable.mono_set' h; grind [uIoc, uIcc])
          (by apply IntervalIntegrable.mono_set' h; grind [uIoc, uIcc]),
        Measure.restrict_restrict_of_subset
          (uIoc_subset_of_mem_disjWithin hnI (Finset.mem_range.mp hi)),
        intervalIntegral.integral_symm, abs_neg,
        intervalIntegral.abs_intervalIntegral_eq]
    exact abs_integral_le_integral_abs
  · intro i hi
    unfold IntegrableOn
    have h_subset := uIoc_subset_of_mem_disjWithin hnI (Finset.mem_range.mp hi)
    rw [Measure.restrict_restrict_of_subset h_subset]
    exact IntegrableOn.mono_set h.def'.norm h_subset |>.integrable

end AbsolutelyContinuousOnInterval
