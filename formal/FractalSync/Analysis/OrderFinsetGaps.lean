/-
Copyright (c) 2025 Yizheng Zhu. All rights reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Yizheng Zhu

Adapted for FractalSync's Lean 4.24.0 pin from mathlib4 commit
 aad247f928b1eb7c58478742119fdd15a37f0ffd. The original Apache 2.0
copyright and authorship are retained. Compatibility changes preserve the
mathematical definitions and conclusions needed by the path theorem.
-/

import Mathlib.Data.Finset.Sort
import Mathlib.Data.Prod.Lex

import Mathlib.Tactic.GRewrite
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.FieldSimp

/-!
# Gaps of disjoint closed intervals

This file defines `Finset.intervalGapsWithin` that computes the complement of the union of a
collection of pairwise disjoint subintervals of `[a, b]`.

If `LinearOrder α`, `F` is a finite subset of `α × α` such that for any `(x, y) ∈ F`,
`a ≤ x ≤ y ≤ b` and all such `[x, y]`'s are pairwise disjoint, `h` is a proof of `#F = k`,
`i` is in `Fin (k + 1)`, we order `F` from left to right as
`(x 0, y 0), ..., (x (k - 1), y (k - 1))`, then `F.intervalGapsWithin h a b i` is
- `(a, b)` if `0 = i = k`;
- `(a, x 0)` if `0 = i < k`;
- `(y (i - 1), x i)` if `0 < i < k`;
- `(y (i - 1), b)` if `0 < i = k`.

Technically, the definition `F.intervalGapsWithin a b` does not require `F` to be pairwise disjoint
or endpoints to be within `[a, b]` or even require that `a ≤ b`, but it makes the most sense if
they are actually satisfied. If they are actually satisfied, then we show that
* `Finset.intervalGapsWithin_mapsTo`, `Finset.intervalGapsWithin_injective`,
  `Finset.intervalGapsWithin_surjOn`:
  `(fun j ↦ ((F.intervalGapsWithin h a b j.castSucc).2, (F.intervalGapsWithin h a b j.succ).1))` is
  a bijection between `Set.Iio k` and `F`.
* `Finset.intervalGapsWithin_le_fst`, `Finset.intervalGapsWithin_snd_le`,
  `Finset.intervalGapsWithin_fst_le_snd`:
  `[(F.intervalGapsWithin h a b j).1, (F.intervalGapsWithin h a b j).2]` is indeed a subinterval of
  `[a, b]` when `j < k`.
* `Finset.intervalGapsWithin_pairwiseDisjoint_Ioc`: the half-closed intervals
  `[(F.intervalGapsWithin h a b j).1, (F.intervalGapsWithin h a b j).2)` are pairwise disjoint
  for `j < k + 1`.
-/

open scoped Finset



open Fin Fin.NatCast Set

section IntervalGapsWithin

namespace Finset

variable {α : Type*} [LinearOrder α] (F : Finset (α × α)) {k : ℕ} (h : #F = k) (a b : α)
  (j : ℕ)

/-- We order `F` in the lexicographic order as `(x 0, y 0), ..., (x (k - 1), y (k - 1))`.
Then `F.intervalGapsWithin h a b i` is
- `(a, b)` if `0 = i = k`;
- `(a, x 0)` if `0 = i < k`;
- `(y (i - 1), x i)` if `0 < i < k`;
- `(y (i - 1), b)` if `0 < i = k`.
-/
noncomputable def intervalGapsWithin (i : Fin (k + 1)) : α × α := (fst, snd) where
  /-- The first coordinate of `F.intervalGapsWithin h a b i` is `a` if `i = 0`,
  `y (i - 1)` otherwise. -/
  fst := if hi : i = 0 then a else
    F.orderEmbOfFin (α := α ×ₗ α) h (i.pred hi) |>.2
  /-- The second coordinate of `F.intervalGapsWithin h a b i` is `b` if `i = k`,
  `x i` otherwise. -/
  snd := if hi : i = last k then b else
    F.orderEmbOfFin (α := α ×ₗ α) h (i.castPred hi) |>.1

@[simp]
theorem intervalGapsWithin_zero_fst : (F.intervalGapsWithin h a b 0).1 = a := by
  simp [intervalGapsWithin, intervalGapsWithin.fst]

theorem intervalGapsWithin_succ_fst_of_lt (hj : j < k) :
    (F.intervalGapsWithin h a b (j.succ)).1 = (F.orderEmbOfFin (α := α ×ₗ α) h ⟨j, hj⟩).2 := by
  have : (j.succ : Fin (k + 1)) = (⟨j, hj⟩ : Fin k).succ := by ext; simp [hj]
  rw [this]
  simp only [intervalGapsWithin, intervalGapsWithin.fst, Fin.succ_ne_zero, ↓reduceDIte, Fin.pred_succ]

theorem intervalGapsWithin_fst_of_lt_lt (hj₁ : 0 < j) (hj₂ : j - 1 < k) :
    (F.intervalGapsWithin h a b j).1 = (F.orderEmbOfFin (α := α ×ₗ α) h ⟨j - 1, hj₂⟩).2 := by
  convert F.intervalGapsWithin_succ_fst_of_lt h a b (j - 1) hj₂
  omega

@[simp]
theorem intervalGapsWithin_last_snd : (F.intervalGapsWithin h a b (last k)).2 = b := by
  simp [intervalGapsWithin, intervalGapsWithin.snd]

theorem intervalGapsWithin_snd_of_lt (hj : j < k) :
    (F.intervalGapsWithin h a b j).2 = (F.orderEmbOfFin (α := α ×ₗ α) h ⟨j, hj⟩).1 := by
  have : (j : Fin (k + 1)) ≠ last k := by
    intro heq
    have hv := congrArg Fin.val heq
    simp only [val_cast_of_lt (hj.trans (Nat.lt_succ_self k)), val_last] at hv
    omega
  simp only [intervalGapsWithin, intervalGapsWithin.snd, this, ↓reduceDIte]
  congr
  ext
  simp only [coe_castPred, val_natCast, Nat.mod_succ_eq_iff_lt]
  omega

theorem intervalGapsWithin_mapsTo : (Set.Iio k).MapsTo
    (fun (j : ℕ) ↦ ((F.intervalGapsWithin h a b j).2, (F.intervalGapsWithin h a b j.succ).1))
    F := by
  intro j hj
  rw [mem_Iio] at hj
  simp only [intervalGapsWithin_snd_of_lt, intervalGapsWithin_succ_fst_of_lt,
    Finset.mem_coe, hj]
  convert F.orderEmbOfFin_mem h ⟨j, hj⟩ using 1

theorem intervalGapsWithin_injOn : (Set.Iio k).InjOn
    (fun (j : ℕ) ↦ ((F.intervalGapsWithin h a b j).2, (F.intervalGapsWithin h a b j.succ).1)) := by
  intro j hj j' hj' hjj'
  rw [mem_Iio] at hj hj'
  simp only [hj, hj', intervalGapsWithin_snd_of_lt, intervalGapsWithin_succ_fst_of_lt] at hjj'
  have heq := (F.orderEmbOfFin (α := α ×ₗ α) h).injective hjj'
  exact congrArg Fin.val heq

theorem intervalGapsWithin_surjOn : (Set.Iio k).SurjOn
    (fun (j : ℕ) ↦ ((F.intervalGapsWithin h a b j).2, (F.intervalGapsWithin h a b j.succ).1))
    F := by
  intro z hz
  rw [← F.range_orderEmbOfFin h (α := α ×ₗ α)] at hz
  obtain ⟨j, hj⟩ := hz
  use j.val, j.prop
  simp [intervalGapsWithin_snd_of_lt, intervalGapsWithin_succ_fst_of_lt, j.prop, hj,
    -coe_eq_castSucc]

theorem intervalGapsWithin_le_fst {a b : α} (hFab : ∀ ⦃z⦄, z ∈ F → a ≤ z.1 ∧ z.1 ≤ z.2 ∧ z.2 ≤ b) :
    a ≤ (F.intervalGapsWithin h a b j).1 := by
  unfold intervalGapsWithin intervalGapsWithin.fst
  dsimp only
  split_ifs with hj
  · exact le_rfl
  · have hm := F.orderEmbOfFin_mem (α := α ×ₗ α) h ((j : Fin (k + 1)).pred hj)
    exact (hFab hm).1.trans (hFab hm).2.1

theorem intervalGapsWithin_snd_le {a b : α} (hFab : ∀ ⦃z⦄, z ∈ F → a ≤ z.1 ∧ z.1 ≤ z.2 ∧ z.2 ≤ b) :
    (F.intervalGapsWithin h a b j).2 ≤ b := by
  unfold intervalGapsWithin intervalGapsWithin.snd
  dsimp only
  split_ifs with hj
  · exact le_rfl
  · have hm := F.orderEmbOfFin_mem (α := α ×ₗ α) h ((j : Fin (k + 1)).castPred hj)
    exact (hFab hm).2.1.trans (hFab hm).2.2

theorem intervalGapsWithin_fst_le_snd {a b : α} (hab : a ≤ b)
    (hFab : ∀ ⦃z⦄, z ∈ F → a ≤ z.1 ∧ z.1 ≤ z.2 ∧ z.2 ≤ b)
    (hF : (↑F : Set (α × α)).PairwiseDisjoint (fun z ↦ Set.Icc z.1 z.2)) :
    (F.intervalGapsWithin h a b j).1 ≤ (F.intervalGapsWithin h a b j).2 := by
  wlog hj : j < k + 1 generalizing j
  · convert this (j : Fin (k + 1)) (by grind) using 3 <;> grind [cast_val_eq_self]
  by_cases hj₁ : j = 0
  · simp only [hj₁]
    by_cases hk : 0 = k
    · simp only [show ((0 : ℕ) : Fin (k + 1)) = 0 from rfl, intervalGapsWithin_zero_fst]
      simp [show 0 = last k by grind, hab]
    · exact hFab (F.intervalGapsWithin_mapsTo h a b (x := 0) (by grind)) |>.left
  have hk : k - 1 + 1 = k := by omega
  by_cases hj₂ : j = k
  · simp only [hj₂, natCast_eq_last, intervalGapsWithin_last_snd, ge_iff_le]
    convert hFab (F.intervalGapsWithin_mapsTo h a b (x := j - 1) (by grind)) |>.right.right using 1
    simp [hj₂, hk]
  rw [intervalGapsWithin_fst_of_lt_lt (hj₁ := by omega) (hj₂ := by omega),
      intervalGapsWithin_snd_of_lt (hj := by omega)]
  have hj₃ : (⟨j - 1, by omega⟩ : Fin k) ≠ ⟨j, by omega⟩ := by grind
  set G := F.orderEmbOfFin (α := α ×ₗ α) h
  have := hF (by simp [G, F.orderEmbOfFin_mem (α := α ×ₗ α)])
    (by simp [G, F.orderEmbOfFin_mem (α := α ×ₗ α)]) (G.injective.ne (hj₃))
  contrapose! this
  simp only [Set.not_disjoint_iff, Set.mem_Icc]
  use (G ⟨j, by omega⟩).1
  have hG : (G ⟨j - 1, by omega⟩).1 ≤ (G ⟨j, by omega⟩).1 :=
    Prod.Lex.le_iff'.mp (G.monotone (by simp [le_iff_val_le_val])) |>.left
  have hFabi := hFab (z := G ⟨j, by omega⟩) (by simp [G, F.orderEmbOfFin_mem (α := α ×ₗ α)])
  simp [hFabi, this.le, hG]

theorem intervalGapsWithin_pairwiseDisjoint_Ioc {a b : α}
    (hFab : ∀ ⦃z⦄, z ∈ F → a ≤ z.1 ∧ z.1 ≤ z.2 ∧ z.2 ≤ b) :
    (Set.Iio (k + 1)).PairwiseDisjoint (fun (j : ℕ) ↦
      Set.Ioc (F.intervalGapsWithin h a b j).1 (F.intervalGapsWithin h a b j).2) := by
  intro j hj j' hj' hjj'
  rw [mem_Iio] at hj hj'
  wlog hij' : j < j' generalizing j j'
  · exact (this hj' hj hjj'.symm (by omega)).symm
  suffices (F.intervalGapsWithin h a b j).2 ≤ (F.intervalGapsWithin h a b j').1 by
    exact Set.disjoint_left.2 fun z hz hz' => (not_lt_of_ge (hz.2.trans this)) hz'.1
  have hj'₀ : j' - 1 + 1 = j' := by omega
  have := hFab (F.intervalGapsWithin_mapsTo h a b (x := j' - 1) (by grind)) |>.right.left
  simp only [Nat.succ_eq_add_one, hj'₀] at this
  grw [← this]
  rw [intervalGapsWithin_snd_of_lt (hj := by omega),
      intervalGapsWithin_snd_of_lt (hj := by omega)]
  exact Prod.Lex.le_iff'.mp (F.orderEmbOfFin (α := α ×ₗ α) h |>.monotone (by change j ≤ j' - 1; omega)) |>.left

end Finset

end IntervalGapsWithin
