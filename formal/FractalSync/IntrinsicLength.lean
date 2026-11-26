import FractalSync.ACSubpaths
import FractalSync.IntrinsicTopology

/-!
# Cumulative graph length controls the actual intrinsic path distance
-/

namespace FractalSync.DistanceGraph

open MeasureTheory Set Filter
open scoped Topology NNReal

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

noncomputable def acCumulativeLength (S : Set E) (epsilon a : ℝ)
    (gamma : ℝ → E) (t : ℝ) : ℝ :=
  ∫ s in (0 : ℝ)..t, graphSpeed a epsilon (Metric.infDist (gamma s) S)
    ‖deriv gamma s‖ (deriv (fun u => Metric.infDist (gamma u) S) s)

@[simp] theorem acCumulativeLength_zero (S : Set E) (epsilon a : ℝ) (gamma : ℝ → E) :
    acCumulativeLength S epsilon a gamma 0 = 0 := by simp [acCumulativeLength]

@[simp] theorem acCumulativeLength_one (S : Set E) (epsilon a : ℝ) (gamma : ℝ → E) :
    acCumulativeLength S epsilon a gamma 1 = acGraphLength S epsilon a gamma := rfl

/-- The cumulative actual graph integral is continuous on the path interval. -/
theorem continuousOn_acCumulativeLength [FiniteDimensional ℝ E]
    (S : Set E) (epsilon a : ℝ) (gamma : ℝ → E) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) :
    ContinuousOn (acCumulativeLength S epsilon a gamma) (uIcc (0 : ℝ) 1) := by
  exact intervalIntegral.continuousOn_primitive_interval
    ((intervalIntegrable_iff').mp (ac_graphSpeed_intervalIntegrable S gamma epsilon a 0 1 he hgamma))

theorem acCumulativeLength_sub [FiniteDimensional ℝ E]
    (S : Set E) (epsilon a : ℝ) (gamma : ℝ → E) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) (u v : ℝ)
    (hu : u ∈ uIcc (0 : ℝ) 1) (hv : v ∈ uIcc (0 : ℝ) 1) :
    acCumulativeLength S epsilon a gamma v - acCumulativeLength S epsilon a gamma u =
      ∫ t in u..v, graphSpeed a epsilon (Metric.infDist (gamma t) S)
        ‖deriv gamma t‖ (deriv (fun s => Metric.infDist (gamma s) S) t) := by
  exact intervalIntegral.integral_interval_sub_left
    ((ac_graphSpeed_intervalIntegrable S gamma epsilon a 0 1 he hgamma).mono_set
      (uIcc_subset_uIcc_left hv))
    ((ac_graphSpeed_intervalIntegrable S gamma epsilon a 0 1 he hgamma).mono_set
      (uIcc_subset_uIcc_left hu))

theorem intrinsicDistance_le_acCumulativeLength_sub [FiniteDimensional ℝ E]
    (S : Set E) (epsilon a : ℝ) (gamma : ℝ → E) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) (u v : ℝ)
    (hu : u ∈ uIcc (0 : ℝ) 1) (hv : v ∈ uIcc (0 : ℝ) 1) (huv : u ≤ v) :
    intrinsicDistance S epsilon a (gamma u) (gamma v) ≤
      acCumulativeLength S epsilon a gamma v - acCumulativeLength S epsilon a gamma u := by
  exact (intrinsicDistance_le_subpath_integral S gamma epsilon a u v huv
    (hgamma.mono (uIcc_subset_uIcc hu hv))).trans_eq
      (acCumulativeLength_sub S epsilon a gamma he hgamma u v hu hv).symm

theorem intrinsicDistance_le_abs_acCumulativeLength_sub [FiniteDimensional ℝ E]
    (S : Set E) (epsilon a : ℝ) (gamma : ℝ → E) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) (u v : ℝ)
    (hu : u ∈ uIcc (0 : ℝ) 1) (hv : v ∈ uIcc (0 : ℝ) 1) :
    intrinsicDistance S epsilon a (gamma u) (gamma v) ≤
      |acCumulativeLength S epsilon a gamma u - acCumulativeLength S epsilon a gamma v| := by
  rcases le_total u v with huv | hvu
  · simpa only [abs_sub_comm] using
      (intrinsicDistance_le_acCumulativeLength_sub S epsilon a gamma he hgamma u v hu hv huv).trans
        (le_abs_self _)
  · simpa only [intrinsicDistance_symm S epsilon a (gamma v) (gamma u)] using
      (intrinsicDistance_le_acCumulativeLength_sub S epsilon a gamma he hgamma v u hv hu hvu).trans
        (le_abs_self _)

end FractalSync.DistanceGraph
