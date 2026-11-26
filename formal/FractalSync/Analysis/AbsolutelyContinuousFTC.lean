/-
Copyright (c) 2025 Yizheng Zhu. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Yizheng Zhu

Adapted for FractalSync's Lean 4.24.0 pin from mathlib4 commit
 aad247f928b1eb7c58478742119fdd15a37f0ffd. The original Apache 2.0
copyright and authorship are retained. Compatibility changes preserve the
mathematical definitions and conclusions needed by the path theorem.
-/

import FractalSync.Analysis.DerivIntegrable
import FractalSync.Analysis.AbsolutelyContinuous

import FractalSync.Analysis.FinsetGaps
import Mathlib.MeasureTheory.Integral.IntervalIntegral.LebesgueDifferentiationThm
import Mathlib.MeasureTheory.Covering.Vitali
import Mathlib.Analysis.Calculus.Deriv.Mul

import Mathlib.Tactic.GRewrite
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.FieldSimp

/-!
# Fundamental theorem of calculus and integration by parts for absolutely continuous functions

This file proves that:
* `AbsolutelyContinuousOnInterval.integral_deriv_eq_sub`: If `f` is absolutely continuous on
  `uIcc a b`, then *Fundamental Theorem of Calculus* holds for `f'` on `a..b`, i.e.
  `∫ (x : ℝ) in a..b, deriv f x = f b - f a`.
* `AbsolutelyContinuousOnInterval.integral_mul_deriv_eq_deriv_mul`:
  *Integration by Parts* holds for absolutely continuous functions, i.e. if `f` and `g` are
  absolutely continuous on `uIcc a b`, then
  `∫ x in a..b, f x * deriv g x = f b * g b - f a * g a - ∫ x in a..b, deriv f x * g x`.

## Tags

absolutely continuous, fundamental theorem of calculus, integration by parts
-/

open scoped Finset



variable {X F : Type*} [PseudoMetricSpace X] [NormedAddCommGroup F] [NormedSpace ℝ F]

open Filter Fin.NatCast Function MeasureTheory Set

open scoped Topology

/-- If `f` has derivative `f'` a.e. on `[d, b]` and `η` is positive, then there is a collection of
pairwise disjoint closed subintervals of `[a, b]` of total length `b - a` where the slope of `f`
on each subinterval `[x, y]` differs from `f' x` by at most `η`. -/
lemma exists_dist_slope_lt_pairwiseDisjoint_hasSum {f f' : ℝ → F} {d b η : ℝ}
    (hdb : d ≤ b) (hf : ∀ᵐ x, x ∈ Ioo d b → HasDerivAt f (f' x) x) (hη : 0 < η) :
    ∃ u : Set (ℝ × ℝ),
      (∀ z ∈ u, (d < z.1 ∧ z.1 < z.2 ∧ z.2 < b) ∧ dist (slope f z.1 z.2) (f' z.1) < η) ∧
      u.PairwiseDisjoint (fun z ↦ Icc z.1 z.2) ∧
      HasSum (fun (z : u) ↦ z.val.2 - z.val.1) (b - d) := by
  -- Proof idea: Use `Vitali.exists_disjoint_covering_ae` to get a Vitali cover of `[a, b]`
  -- consisting of closed subintervals `[x, y]` on which the slope of `f` differs from `f' x` by
  -- at most `η`.
  rcases hdb.eq_or_lt with rfl | hdb
  · refine ⟨∅, by simp, by simp, ?_⟩
    simp only [sub_self]
    convert (hasSum_zero : HasSum (fun _ : (∅ : Set (ℝ × ℝ)) => (0 : ℝ)) 0) using 1
    funext z
    exact False.elim z.property
  let t := {z : ℝ × ℝ | (d < z.1 ∧ z.1 < z.2 ∧ z.2 < b) ∧ dist (slope f z.1 z.2) (f' z.1) < η}
  let s := {x ∈ Ioo d b | HasDerivAt f (f' x) x}
  obtain ⟨u, ⟨hu₁, hu₂, hu₃, hu₄⟩⟩ : ∃ u ⊆ t, u.Countable ∧
      u.PairwiseDisjoint (fun z ↦ Icc z.1 z.2) ∧ volume (s \ ⋃ z ∈ u, Icc z.1 z.2) = 0 := by
    apply Vitali.exists_disjoint_covering_ae volume s t 6 (Prod.snd - Prod.fst) Prod.fst
      (fun z ↦ Icc z.1 z.2)
    · grind [Metric.closedBall, Real.dist_eq, Pi.sub_apply, abs_le']
    · intro A hA
      simp only [Pi.sub_apply, Real.volume_closedBall, ENNReal.coe_ofNat, Real.volume_Icc]
      rw [show 6 = ENNReal.ofReal 6 by simp, ← ENNReal.ofReal_mul (by simp),
          ENNReal.ofReal_le_ofReal_iff (by grind)]
      linarith
    · simp +contextual [t]
    · simp [isClosed_Icc]
    · intro x hx ε hε
      obtain ⟨δ, hδ, hδbound⟩ := (Metric.tendsto_nhdsWithin_nhds).mp
        (hasDerivAt_iff_tendsto_slope.mp hx.right) η hη
      let e := min ε (min δ (b - x)) / 2
      have hex : 0 < b - x := sub_pos.mpr hx.1.2
      have he : 0 < e := div_pos (lt_min hε (lt_min hδ hex)) (by norm_num)
      have heε : e ≤ ε := by
        have := min_le_left ε (min δ (b - x))
        dsimp [e]
        linarith
      have heδ : e < δ := by
        have := (min_le_right ε (min δ (b - x))).trans (min_le_left δ (b - x))
        dsimp [e]
        linarith
      have heb : e < b - x := by
        have := (min_le_right ε (min δ (b - x))).trans (min_le_right δ (b - x))
        dsimp [e]
        linarith
      refine ⟨(x, x + e), ⟨⟨hx.1.1, by linarith, by linarith⟩, ?_⟩,
        by simpa [Pi.sub_apply] using heε, rfl⟩
      exact hδbound (by simpa using ne_of_gt he) (by simpa [Real.dist_eq, abs_of_pos he] using heδ)
  simp only [t, subset_def, mem_setOf_eq] at hu₁
  refine ⟨u, ⟨hu₁, hu₃, ?_⟩⟩
  have : Countable u := by simp [hu₂]
  have : Pairwise (Disjoint on fun (z : u) ↦ Icc z.val.1 z.val.2) :=
    fun z₁ z₂ hz₁z₂ ↦ hu₃ z₁.prop z₂.prop (Subtype.coe_ne_coe.mpr hz₁z₂)
  replace hu₄ : volume (Ioo d b \ ⋃ z ∈ u, Icc z.1 z.2) = 0 := by
    rw [measure_eq_zero_iff_ae_notMem] at hu₄ ⊢
    filter_upwards [hf, hu₄] with x hx₁ hx₂
    grind
  have hsub : (⋃ z ∈ u, Icc z.1 z.2) ⊆ Ioo d b := by
    intro x hx
    obtain ⟨z, hz, hxz⟩ := mem_iUnion₂.mp hx
    exact ⟨(hu₁ z hz).1.1.trans_le hxz.1, hxz.2.trans_lt (hu₁ z hz).1.2.2⟩
  have hae : (⋃ z ∈ u, Icc z.1 z.2) =ᵐ[volume] Ioo d b := by
    rw [measure_eq_zero_iff_ae_notMem] at hu₄
    filter_upwards [hu₄] with x hx
    exact propext ⟨fun h => hsub h, fun h => by by_contra hn; exact hx ⟨h, hn⟩⟩
  have vol_sum : volume (⋃ z : u, Icc z.val.1 z.val.2) = ENNReal.ofReal (b - d) := by
    calc
      _ = volume (⋃ z ∈ u, Icc z.1 z.2) := by simp
      _ = volume (Ioo d b) := measure_congr hae
      _ = ENNReal.ofReal (b - d) := Real.volume_Ioo
  rw [measure_iUnion this (by simp)] at vol_sum
  simp_rw [Real.volume_Icc] at vol_sum
  apply_fun fun x ↦ x.toReal at vol_sum
  rw [ENNReal.tsum_toReal_eq (by simp), ENNReal.toReal_ofReal (by linarith),
      ← Summable.hasSum_iff (by grind [tsum_def])] at vol_sum
  grind [ENNReal.toReal_ofReal]

/-- If `f` is absolutely continuous on `[d, b]` and there is a collection of pairwise disjoint
closed subintervals of `(d, b)` of total length `b - d` such that the sum of `dist (f x) (f y)` for
`[x, y]` in the collection is equal to `y`, then `dist (f b) (f d) ≤ y`. -/
lemma AbsolutelyContinuousOnInterval.dist_le_of_pairwiseDisjoint_hasSum {f : ℝ → X} {d b y : ℝ}
    (hdb : d ≤ b) (hf : AbsolutelyContinuousOnInterval f d b) {u : Set (ℝ × ℝ)}
    (hu₁ : ∀ z ∈ u, d < z.1 ∧ z.1 < z.2 ∧ z.2 < b)
    (hu₂ : u.PairwiseDisjoint (fun z ↦ Icc z.1 z.2))
    (hu₃ : HasSum (fun (z : u) ↦ z.val.2 - z.val.1) (b - d))
    (hu₄ : HasSum (fun (z : u) ↦ dist (f z.val.1) (f z.val.2)) y) :
    dist (f d) (f b) ≤ y := by
  -- Proof idea: The complement of the collection of subintervals of `[d, b]` encoded in `u` can
  -- be approached by the complement of subcollections encoded by finite subsets `s ⊆ u`. These
  -- complements are encoded using `Finset.intervalGapsWithin`.
  -- Their total length tends to `0` as `s` tends to `u` and by absolute continuity of `f`, the sum
  -- of `dist (f x) (f y)` for `[x, y]` in the complement tends to `0` as `s` tends to `u`.
  -- Finally we use the triangle inequality of `dist` to obtain the result.
  let u_coe (s : Finset u) : Finset (ℝ × ℝ) := s.image Subtype.val
  replace hu₁ (s : Finset u) : ∀ ⦃z : ℝ × ℝ⦄, z ∈ u_coe s → d ≤ z.1 ∧ z.1 ≤ z.2 ∧ z.2 ≤ b := by
    grind
  replace hu₂ (s : Finset u) : (↑(u_coe s) : Set (ℝ × ℝ)).PairwiseDisjoint fun z ↦ Icc z.1 z.2 :=
    hu₂.subset (by grind)
  let T (s : Finset u) :=
    (#(u_coe s) + 1, fun (i : ℕ) ↦ (u_coe s).intervalGapsWithin rfl d b i)
  have hT₁ (s : Finset u) (i : ℕ) := (u_coe s).intervalGapsWithin_le_fst rfl i (hu₁ s)
  have hT₂ (s : Finset u) (i : ℕ) :=
    (u_coe s).intervalGapsWithin_fst_le_snd rfl i hdb (hu₁ s) (hu₂ s)
  have hT₃ (s : Finset u) (i : ℕ) := (u_coe s).intervalGapsWithin_snd_le rfl i (hu₁ s)
  have hT₄ (s : Finset u) := (u_coe s).intervalGapsWithin_pairwiseDisjoint_Ioc rfl (hu₁ s)
  have hT : univ.MapsTo T (disjWithin d b) := by
    intro s _
    constructor
    · intro i hi
      rw [uIcc_of_le hdb]
      exact ⟨⟨hT₁ s i, (hT₂ s i).trans (hT₃ s i)⟩,
        ⟨(hT₁ s i).trans (hT₂ s i), hT₃ s i⟩⟩
    · simpa only [T, Finset.coe_range, uIoc_of_le (hT₂ s _)] using hT₄ s
  have u_coe_sum (s : Finset u) (g : ℝ → ℝ → ℝ) :
      ∑ b ∈ s, (g b.val.1 b.val.2) = ∑ z ∈ u_coe s, (g z.1 z.2) :=
    Finset.sum_nbij Subtype.val (by simp [u_coe]) (by simp)
      (by simp only [Finset.coe_image, u_coe]; tauto) (by simp)
  replace hu₃ : Tendsto T atTop (totalLengthFilter ⊓ 𝓟 (disjWithin d b)) := by
    refine tendsto_inf.mpr ⟨?_, hT.tendsto.mono_left (by simp)⟩
    simp only [totalLengthFilter, tendsto_comap_iff]
    convert hu₃.const_sub (b - d) with s
    · simp only [comp_apply]
      rw [Finset.sum_congr rfl (g := fun i ↦ ((T s).2 i).2 - ((T s).2 i).1)
            (fun i hi ↦ by rw [dist_comm, Real.dist_eq, abs_of_nonneg (by grind)])]
      convert (u_coe s).sum_intervalGapsWithin_eq_sub_sub_sum rfl id
      exact u_coe_sum s fun x y ↦ y - x
    · abel
  rw [HasSum] at hu₄
  simp_rw [u_coe_sum _ fun x y ↦ dist (f x) (f y)] at hu₄
  have sum_tendsto := hf.comp hu₃ |>.add hu₄
  simp only [comp_apply, zero_add] at sum_tendsto
  have dist_le_sum (s : Finset u) :
      dist (f d) (f b) ≤
      ∑ i ∈ Finset.range (T s).1, dist (f ((T s).2 i).1) (f ((T s).2 i).2) +
      (∑ b ∈ u_coe s, dist (f b.1) (f b.2)) := by
    rw [Finset.sum_eq_sum_range_intervalGapsWithin _ rfl (fun x y ↦ dist (f x) (f y)),
        Finset.sum_range_succ, add_right_comm, ← Finset.sum_add_distrib]
    grw [← Finset.sum_le_sum (fun _ _ ↦ dist_triangle _ _ _),
        ← dist_le_range_sum_dist,
        ← dist_triangle]
    simp [T]
  exact le_of_tendsto_of_tendsto' (by simp) sum_tendsto dist_le_sum

/-- If `f` is absolutely continuous on `uIcc a b` and `f' x = 0` for a.e. `x ∈ uIcc a b`, then `f`
is constant on `uIcc a b`. -/
theorem AbsolutelyContinuousOnInterval.const_of_ae_hasDerivAt_zero {f : ℝ → F} {a b : ℝ}
    (hf : AbsolutelyContinuousOnInterval f a b) (hf₀ : ∀ᵐ x, x ∈ uIcc a b → HasDerivAt f 0 x) :
    ∃ C, ∀ x ∈ uIcc a b, f x = C := by
  -- Proof idea : Assume wlog `a < b`. We need to show that `f d = f b` for any `d ∈ [a, b]`.
  -- Fix `d`. It suffices to show that `dist (f d) (f b) ≤ r` for any `r > 0`. Fix `r`.
  -- Use `exists_dist_slope_lt_pairwiseDisjoint_hasSum` with `η = r / (b - d)` to
  -- get a cover of `[d, b]` consisting of closed subintervals with total length `b - d` such that
  -- the slope of `f` on each subinterval has absolute value `≤ η`. The sum of `dist (f x) (f y)`
  -- for `[x, y]` in the cover must therefore be `≤ (b - d) * η = r`. Use
  -- `AbsolutelyContinuousOnInterval.dist_le_of_pairwiseDisjoint_hasSum` to conclude that
  -- `dist (f d) (f b) ≤ r`.
  wlog hab : a ≤ b
  · exact uIcc_comm b a ▸ this (a := b) (b := a) hf.symm (uIcc_comm a b ▸ hf₀) (by linarith)
  suffices ∀ x ∈ uIcc a b, f x = f b by use f b
  rw [uIcc_of_le hab] at hf₀ ⊢
  intro d hd
  suffices ∀ r > 0, dist (f d) (f b) ≤ r by
    contrapose! this
    exact exists_between (dist_pos.mpr this)
  intro r hr
  rw [mem_Icc] at hd
  have had : a ≤ d := by linarith
  rcases hd.right.eq_or_lt with rfl | hdb
  · simp [hr.le]
  replace hf₀ : ∀ᵐ x, x ∈ Ioo d b → HasDerivAt f 0 x := by
    filter_upwards [hf₀] with x _ _ using by grind
  have hfdb' : 0 < r / (b - d) := by apply div_pos <;> linarith
  have ⟨u, hu₁, hu₂, hu₃⟩ :=
    exists_dist_slope_lt_pairwiseDisjoint_hasSum hd.right hf₀ hfdb'
  let g := fun (z : u) ↦ dist (f z.val.1) (f z.val.2)
  have g_nonneg : 0 ≤ g := by intro; simp [g]
  have g_finsum_bound (s : Finset u) : ∑ z ∈ s, g z ≤ r := by
    have (z : u) (hz : z ∈ s) : g z ≤ r / (b - d) * (z.val.2 - z.val.1) := by
      have slope_bound := hu₁ z (by simp) |>.right |>.le
      have : 0 < z.val.2 - z.val.1 := by linarith [hu₁ z (by simp)]
      grw [← slope_bound]
      simp only [dist_eq_norm, slope, vsub_eq_sub, sub_zero, g, mul_comm]
      nth_rw 1 [← Real.norm_of_nonneg this.le]
      simp only [norm_smul, Real.norm_eq_abs, norm_inv]
      field_simp
      rw [norm_sub_rev]
    grw [Finset.sum_le_sum this]
    rw [← Finset.mul_sum]
    have : ∑ z ∈ s, (z.val.2 - z.val.1) ≤ b - d :=
      hu₃.tsum_eq ▸ Summable.sum_le_tsum _ (by grind) hu₃.summable
    grw [this]
    field_simp
    grind
  have hu₄ := summable_of_sum_le g_nonneg g_finsum_bound |>.hasSum
  have g_sum_bound := hu₄.summable.tsum_le_of_sum_le g_finsum_bound
  have := (hf.mono (by grind [uIcc_of_le])).dist_le_of_pairwiseDisjoint_hasSum hd.right
    (fun s hs ↦ hu₁ s hs |>.left) hu₂ hu₃ hu₄
  linarith

/-- *Fundamental Theorem of Calculus* for absolutely continuous functions: if `f` is absolutely
continuous on `uIcc a b`, then `∫ (x : ℝ) in a..b, deriv f x = f b - f a`. -/
theorem AbsolutelyContinuousOnInterval.integral_deriv_eq_sub {f : ℝ → ℝ} {a b : ℝ}
    (hf : AbsolutelyContinuousOnInterval f a b) :
    ∫ (x : ℝ) in a..b, deriv f x = f b - f a := by
  have f_deriv_integral_ac :=
    hf.intervalIntegrable_deriv.absolutelyContinuousOnInterval_intervalIntegral
    (c := a) (by simp)
  let g (x : ℝ) := f x - ∫ (t : ℝ) in a..x, deriv f t
  have g_ac : AbsolutelyContinuousOnInterval g a b := hf.sub (f_deriv_integral_ac)
  have g_ae_deriv_zero : ∀ᵐ x, x ∈ uIcc a b → HasDerivAt g 0 x := by
    filter_upwards [hf.ae_differentiableAt, hf.intervalIntegrable_deriv.ae_hasDerivAt_integral]
      with x hx₁ hx₂ hx₃
    convert (hx₁ hx₃).hasDerivAt.sub (hx₂ hx₃ a (by simp))
    abel
  obtain ⟨C, hC⟩ := g_ac.const_of_ae_hasDerivAt_zero g_ae_deriv_zero
  have : f a = g a := by simp [g]
  have := hC a (by simp)
  have := hC b (by simp)
  grind

/-- The integral of the derivative of a product of two absolutely continuous functions. -/
theorem AbsolutelyContinuousOnInterval.integral_deriv_mul_eq_sub
    {f g : ℝ → ℝ} {a b : ℝ}
    (hf : AbsolutelyContinuousOnInterval f a b) (hg : AbsolutelyContinuousOnInterval g a b) :
    ∫ x in a..b, deriv f x * g x + f x * deriv g x = f b * g b - f a * g a := by
  rw [← (hf.fun_mul hg).integral_deriv_eq_sub]
  apply intervalIntegral.integral_congr_ae
  filter_upwards [hf.ae_differentiableAt, hg.ae_differentiableAt] with x hx₁ hx₂ hx₃
  have hx₄ : x ∈ uIcc a b := uIoc_subset_uIcc hx₃
  have hx₅ := (hx₁ hx₄).hasDerivAt.mul (hx₂ hx₄).hasDerivAt
  exact hx₅.deriv.symm

/-- *Integration by parts* for absolutely continuous functions. -/
theorem AbsolutelyContinuousOnInterval.integral_mul_deriv_eq_deriv_mul {f g : ℝ → ℝ} {a b : ℝ}
    (hf : AbsolutelyContinuousOnInterval f a b) (hg : AbsolutelyContinuousOnInterval g a b) :
    ∫ x in a..b, f x * deriv g x = f b * g b - f a * g a - ∫ x in a..b, deriv f x * g x := by
  rw [← AbsolutelyContinuousOnInterval.integral_deriv_mul_eq_sub hf hg,
      ← intervalIntegral.integral_sub]
  · simp_rw [add_sub_cancel_left]
  · exact (hf.intervalIntegrable_deriv.mul_continuousOn hg.continuousOn).add
      (hg.intervalIntegrable_deriv.continuousOn_mul hf.continuousOn)
  · exact hf.intervalIntegrable_deriv.mul_continuousOn hg.continuousOn
