import Mathlib.MeasureTheory.Function.AbsolutelyContinuous
import Mathlib.Order.Interval.Set.Disjoint
import Mathlib.Topology.MetricSpace.Lipschitz
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.FieldSimp

/-!
# Domain transformations and gluing for absolutely continuous paths

These results use the pinned Mathlib definition of absolute continuity.
Monotone Lipschitz domain maps preserve the disjoint-interval condition;
antitone maps do so after reversing the interval orientation. Gluing follows
by constant extension using the one-Lipschitz `min` and `max` domain maps.
-/

namespace FractalSync.ACComposition

open Set
open scoped NNReal

lemma disjoint_uIoc_map_monotone {f : ℝ → ℝ} (hf : Monotone f)
    {a b c d : ℝ} (h : Disjoint (uIoc a b) (uIoc c d)) :
    Disjoint (uIoc (f a) (f b)) (uIoc (f c) (f d)) := by
  simp only [uIoc, Ioc_disjoint_Ioc] at h ⊢
  simp only [← hf.map_min, ← hf.map_max]
  exact hf h

lemma disjoint_uIoc_map_antitone {f : ℝ → ℝ} (hf : Antitone f)
    {a b c d : ℝ} (h : Disjoint (uIoc a b) (uIoc c d)) :
    Disjoint (uIoc (f a) (f b)) (uIoc (f c) (f d)) := by
  simp only [uIoc, Ioc_disjoint_Ioc] at h ⊢
  simp only [← hf.map_min, ← hf.map_max]
  exact hf h

/-- Monotone or antitone Lipschitz changes of parameter preserve the actual
absolute-continuity condition. The interval image is an explicit hypothesis. -/
theorem comp_lipschitz_mono {X : Type*} [PseudoMetricSpace X]
    {f : ℝ → X} {g : ℝ → ℝ} {a b lo hi : ℝ} {K : ℝ≥0}
    (hf : AbsolutelyContinuousOnInterval f a b) (hg : LipschitzWith K g)
    (hmono : Monotone g ∨ Antitone g)
    (hmap : MapsTo g (uIcc lo hi) (uIcc a b)) :
    AbsolutelyContinuousOnInterval (fun t => f (g t)) lo hi := by
  rw [absolutelyContinuousOnInterval_iff] at hf ⊢
  intro ε hε
  obtain ⟨δ, hδ, hδf⟩ := hf ε hε
  refine ⟨δ / (K + 1), by positivity, ?_⟩
  rintro ⟨n, I⟩ hI hlen
  apply hδf (n, fun i => (g (I i).1, g (I i).2))
  · constructor
    · intro i hi
      exact ⟨hmap (hI.1 i hi).1, hmap (hI.1 i hi).2⟩
    · intro i hi j hj hij
      rcases hmono with hm | hm
      · exact disjoint_uIoc_map_monotone hm (hI.2 hi hj hij)
      · exact disjoint_uIoc_map_antitone hm (hI.2 hi hj hij)
  · calc
      _ ≤ ∑ i ∈ Finset.range n, (K : ℝ) * dist (I i).1 (I i).2 :=
        Finset.sum_le_sum fun i _ => hg.dist_le_mul _ _
      _ = (K : ℝ) * ∑ i ∈ Finset.range n, dist (I i).1 (I i).2 :=
        (Finset.mul_sum _ _ _).symm
      _ ≤ (K : ℝ) * (δ / (K + 1)) :=
        mul_le_mul_of_nonneg_left hlen.le K.coe_nonneg
      _ < ((K : ℝ) + 1) * (δ / (K + 1)) := by
        apply mul_lt_mul_of_pos_right (by linarith)
        positivity
      _ = δ := by field_simp

/-- Two paths glued at a specified time, choosing the first value at the join. -/
noncomputable def gluePath {X : Type*} (f g : ℝ → X) (c t : ℝ) : X :=
  if t ≤ c then f t else g t

/-- Matching absolutely continuous pieces glue to an absolutely continuous
path. No derivative is required at the joining time. -/
theorem absolutelyContinuous_gluePath {E : Type*} [SeminormedAddCommGroup E]
    {f g : ℝ → E} {lo c hi : ℝ} (hlc : lo ≤ c) (hchi : c ≤ hi)
    (hf : AbsolutelyContinuousOnInterval f lo c)
    (hg : AbsolutelyContinuousOnInterval g c hi) (hmatch : f c = g c) :
    AbsolutelyContinuousOnInterval (gluePath f g c) lo hi := by
  have hl : AbsolutelyContinuousOnInterval (fun t => f (min t c)) lo hi := by
    apply comp_lipschitz_mono hf (LipschitzWith.id.min_const c)
      (Or.inl fun _ _ h => min_le_min h le_rfl)
    intro t ht
    simp only [uIcc_of_le (hlc.trans hchi), uIcc_of_le hlc, mem_Icc] at ht ⊢
    exact ⟨le_min ht.1 hlc, min_le_right _ _⟩
  have hr : AbsolutelyContinuousOnInterval (fun t => g (max t c)) lo hi := by
    apply comp_lipschitz_mono hg (LipschitzWith.id.max_const c)
      (Or.inl fun _ _ h => max_le_max h le_rfl)
    intro t ht
    simp only [uIcc_of_le (hlc.trans hchi), uIcc_of_le hchi, mem_Icc] at ht ⊢
    exact ⟨le_max_right _ _, max_le ht.2 hchi⟩
  have hc : AbsolutelyContinuousOnInterval (fun _ : ℝ => f c) lo hi :=
    (LipschitzWith.const (f c)).lipschitzOnWith.absolutelyContinuousOnInterval
  have hid : (fun t => f (min t c) + g (max t c) - f c) = gluePath f g c := by
    funext t
    by_cases ht : t ≤ c
    · simp [gluePath, ht, ← hmatch]
    · simp [gluePath, ht, min_eq_right (le_of_not_ge ht),
        max_eq_left (le_of_not_ge ht), add_sub_cancel_left]
  rw [← hid]
  exact (hl.fun_add hr).fun_sub hc

end FractalSync.ACComposition
