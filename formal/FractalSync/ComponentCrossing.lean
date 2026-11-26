import FractalSync.AbsolutelyContinuousPaths
import Mathlib.Topology.Order.IntermediateValue

/-!
# Shore crossings forced by connected-component separation

A continuous interval image that avoids `S` lies in one connected component of
`Sᶜ`. Consequently, endpoints separated by those components force a crossing.
The topological statement requires neither a closed nor a nonempty Shore.
The absolutely continuous specialization supplies the crossing time required
by the existing calibrated length bound.
-/

namespace FractalSync.DistanceGraph

open MeasureTheory Set

/-- A continuous path whose endpoint does not lie in the starting point's
connected component of the Shore complement must meet Shore. This holds in
an arbitrary topological target, without closedness or nonemptiness of `S`.
The interval may be degenerate. -/
theorem continuousOn_exists_shore_crossing {E : Type*} [TopologicalSpace E]
    (S : Set E) (gamma : ℝ → E) (lo hi : ℝ) (horder : lo ≤ hi)
    (hgamma : ContinuousOn gamma (Icc lo hi))
    (hsep : gamma hi ∉ connectedComponentIn Sᶜ (gamma lo)) :
    ∃ c ∈ Icc lo hi, gamma c ∈ S := by
  by_contra! havoid
  have hsub : gamma '' Icc lo hi ⊆ Sᶜ :=
    fun _ ⟨t, ht, htg⟩ => htg ▸ havoid t ht
  have hpre : IsPreconnected (gamma '' Icc lo hi) :=
    isPreconnected_Icc.image gamma hgamma
  exact hsep ((hpre.subset_connectedComponentIn
    ⟨lo, ⟨le_rfl, horder⟩, rfl⟩ hsub) ⟨hi, ⟨horder, le_rfl⟩, rfl⟩)

/-- Absolute continuity supplies the interval continuity needed for the
connected-component crossing argument. -/
theorem ac_exists_shore_crossing {E : Type*} [SeminormedAddCommGroup E]
    (S : Set E) (gamma : ℝ → E) (lo hi : ℝ) (horder : lo ≤ hi)
    (hgamma : AbsolutelyContinuousOnInterval gamma lo hi)
    (hsep : gamma hi ∉ connectedComponentIn Sᶜ (gamma lo)) :
    ∃ c ∈ Icc lo hi, gamma c ∈ S := by
  exact continuousOn_exists_shore_crossing S gamma lo hi horder
    (by simpa only [uIcc_of_le horder] using hgamma.continuousOn) hsep

/-- Endpoints in different connected components of the Shore complement pay
the sum of the calibrated endpoint distances. The crossing time is deduced
from topology, with no hypothesis selecting it in advance. -/
theorem ac_distance_trace_component_bound {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] [FiniteDimensional ℝ E]
    (S : Set E) (gamma : ℝ → E) (epsilon a lo hi : ℝ)
    (he : 0 < epsilon) (horder : lo ≤ hi)
    (hgamma : AbsolutelyContinuousOnInterval gamma lo hi)
    (hsep : gamma hi ∉ connectedComponentIn Sᶜ (gamma lo)) :
    radialProfile epsilon a (Metric.infDist (gamma lo) S) +
      radialProfile epsilon a (Metric.infDist (gamma hi) S) ≤
      ∫ t in lo..hi, graphSpeed a epsilon (Metric.infDist (gamma t) S)
        ‖deriv gamma t‖ (deriv (fun s => Metric.infDist (gamma s) S) t) := by
  obtain ⟨c, hc, hshore⟩ :=
    ac_exists_shore_crossing S gamma lo hi horder hgamma hsep
  exact ac_distance_trace_crossing_bound S gamma epsilon a lo c hi
    he hc.1 hc.2 hgamma hshore

end FractalSync.DistanceGraph
