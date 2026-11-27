import FractalSync.OperationalRouteBounds
import Mathlib.Topology.Order.IntermediateValue
import Mathlib.Topology.Order.Compact
import Mathlib.Topology.MetricSpace.Pseudo.Constructions

/-!
# Budget bounds from a local metric envelope

A path leaving a certified window must first spend enough length to reach
a certified boundary radius. Only the prefix to that radius needs the bound.
-/

namespace FractalSync.OperationalWindowExit

open Set

/-- A continuous nonnegative displacement history must reach a smaller
radius for the first time through a prefix that stays within that radius. -/
theorem first_radius {f : ℝ → ℝ} {r : ℝ} (hr : 0 ≤ r)
    (hf : ContinuousOn f (Icc 0 1)) (h0 : f 0 = 0)
    (hBeyond : ∃ t ∈ Icc 0 1, r < f t) :
    ∃ t ∈ Icc 0 1, f t = r ∧ ∀ u ∈ Icc 0 t, f u ≤ r := by
  have findHit {t : ℝ} (ht : t ∈ Icc 0 1) (hAt : r ≤ f t) :
      ∃ u ∈ Icc 0 t, f u = r := by
    exact intermediate_value_Icc ht.1 (hf.mono (Icc_subset_Icc le_rfl ht.2))
      ⟨by simpa only [h0] using hr, hAt⟩
  let hits : Set ℝ := f ⁻¹' ({r} : Set ℝ) ∩ Icc 0 1
  obtain ⟨s, hs⟩ := hBeyond
  obtain ⟨u, hu, hru⟩ := findHit hs.1 hs.2.le
  have nonempty : hits.Nonempty := ⟨u, hru, ⟨hu.1, hu.2.trans hs.1.2⟩⟩
  obtain ⟨closedSet, hclosed, heq⟩ :=
    continuousOn_iff_isClosed.mp hf ({r} : Set ℝ) isClosed_singleton
  have compact : IsCompact hits := by
    change IsCompact (f ⁻¹' ({r} : Set ℝ) ∩ Icc 0 1)
    rw [heq, inter_comm]
    exact isCompact_Icc.inter_right hclosed
  obtain ⟨t, least⟩ := compact.exists_isLeast nonempty
  refine ⟨t, least.1.2, least.1.1, ?_⟩
  intro v hv
  by_contra! hvr
  obtain ⟨w, hw, hrw⟩ := findHit ⟨hv.1, hv.2.trans least.1.2.2⟩ hvr.le
  have hwHit : w ∈ hits := ⟨hrw, ⟨hw.1, hw.2.trans (hv.2.trans least.1.2.2)⟩⟩
  have htw : t ≤ w := least.2 hwHit
  have htv : t = v := le_antisymm (htw.trans hw.2) hv.2
  have hAt : f t = r := least.1.1
  rw [htv] at hAt
  exact (ne_of_gt hvr) hAt

/-- Use a contained-arrival bound or the cost already incurred at the first
exit radius. Prefix lengths are bounded by the total length. -/
theorem targetLower_le_of_prefix_envelope {X : Type*} [PseudoMetricSpace X]
    (path : ℝ → X) (start center : X) (arrivalRadius exitRadius rulerUpper total : ℝ)
    (prefixLength : ℝ → ℝ) (hPath : ContinuousOn path (Icc 0 1)) (hStart : path 0 = start)
    (hArrival : dist (path 1) center ≤ arrivalRadius) (hExit : 0 ≤ exitRadius)
    (hRuler : 0 < rulerUpper) (hTotal : 0 ≤ total)
    (hPrefix : ∀ t ∈ Icc 0 1, prefixLength t ≤ total)
    (hLocal : ∀ t ∈ Icc 0 1, (∀ u ∈ Icc 0 t, dist (path u) start ≤ exitRadius) →
      dist (path t) start ≤ prefixLength t * rulerUpper) :
    min (max 0 (dist start center - arrivalRadius)) exitRadius / rulerUpper ≤ total := by
  by_cases contained : ∀ t ∈ Icc 0 1, dist (path t) start ≤ exitRadius
  · have last : (1 : ℝ) ∈ Icc 0 1 := ⟨zero_le_one, le_rfl⟩
    have hEnd := (hLocal 1 last contained).trans (mul_le_mul_of_nonneg_right (hPrefix 1 last) hRuler.le)
    rw [dist_comm] at hEnd
    exact (div_le_div_of_nonneg_right (min_le_left _ _) hRuler.le).trans
      (OperationalRouteBounds.ambientTargetLower_le start (path 1) center
        arrivalRadius rulerUpper total hRuler hTotal hEnd hArrival)
  · push Not at contained
    have hDist : ContinuousOn (fun t => dist (path t) start) (Icc 0 1) :=
      (continuous_dist.comp (continuous_id.prodMk continuous_const) :
        Continuous (fun x : X => dist x start)).comp_continuousOn hPath
    obtain ⟨t, ht, hBoundary, hBefore⟩ := first_radius hExit hDist
      (by simp [hStart]) contained
    have hRad := (hLocal t ht hBefore).trans (mul_le_mul_of_nonneg_right (hPrefix t ht) hRuler.le)
    rw [hBoundary] at hRad
    exact (div_le_div_of_nonneg_right (min_le_right _ _) hRuler.le).trans ((div_le_iff₀ hRuler).2 hRad)

end FractalSync.OperationalWindowExit
