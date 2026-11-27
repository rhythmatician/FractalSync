import FractalSync.GraphChord
import FractalSync.IntrinsicVariation
import FractalSync.IntrinsicGeodesics

/-!
# Intrinsic metric curve length and attainment of the AC graph-length infimum

The metric variation below is Mathlib's actual partition-supremum variation
with the intrinsic metric instance supplied explicitly. The graph-chord estimate
recovers the full graph-speed density from its absolutely continuous variation
trace. This identifies the two lengths and supplies actual minimizing AC paths.
-/

namespace FractalSync.DistanceGraph

open Set MeasureTheory
open scoped Topology

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [FiniteDimensional ℝ E]

/-- Every ordinary AC path is also AC in the intrinsic metric. The metric
instance is explicit, so the ambient normed structure is not overwritten. -/
theorem absolutelyContinuousOnInterval_intrinsic (S : Set E) (epsilon a : ℝ)
    (gamma : ℝ → E) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) :
    @AbsolutelyContinuousOnInterval E
      (intrinsicMetricSpace S epsilon a he).toPseudoMetricSpace gamma 0 1 := by
  exact @absolutelyContinuousOnInterval_of_dist_le E ℝ
    (intrinsicMetricSpace S epsilon a he).toPseudoMetricSpace inferInstance gamma
    (acCumulativeLength S epsilon a gamma) 0 1
    (absolutelyContinuousOnInterval_acCumulativeLength S epsilon a gamma he hgamma)
    (fun s hs t ht => by
      simpa only [Real.dist_eq] using
        intrinsicDistance_le_abs_acCumulativeLength_sub S epsilon a gamma he hgamma s t hs ht)

/-- The derivative of the actual intrinsic variation controls graph speed a.e. -/
theorem ac_graphSpeed_le_deriv_intrinsicVariation (S : Set E) (epsilon a : ℝ)
    (gamma : ℝ → E) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) :
    ∀ᵐ t, t ∈ uIcc (0 : ℝ) 1 →
      graphSpeed a epsilon (Metric.infDist (gamma t) S) ‖deriv gamma t‖
        (deriv (fun s => Metric.infDist (gamma s) S) t) ≤
          deriv (intrinsicVariation S epsilon a gamma he) t := by
  have hbound := ac_graphSpeed_le_control_deriv S epsilon a gamma
    (intrinsicVariation S epsilon a gamma he) he hgamma
    (absolutelyContinuousOnInterval_intrinsicVariation S epsilon a gamma he hgamma)
    (fun s hs t ht => intrinsicDistance_le_intrinsicVariation_abs_sub
      S epsilon a gamma he hgamma s t (by simpa using hs) (by simpa using ht))
  filter_upwards [hbound, show ∀ᵐ t : ℝ, t ≠ 0 from by simp [ae_iff],
    show ∀ᵐ t : ℝ, t ≠ 1 from by simp [ae_iff]] with t hb h0 h1 ht
  have ht' : t ∈ Icc (0 : ℝ) 1 := by simpa using ht
  have hnonneg : 0 ≤ deriv (intrinsicVariation S epsilon a gamma he) t := by
    simpa only [derivWithin_of_mem_nhds
      (Icc_mem_nhds (lt_of_le_of_ne ht'.1 h0.symm) (lt_of_le_of_ne ht'.2 h1))] using
        (monotoneOn_intrinsicVariation S epsilon a gamma he hgamma).derivWithin_nonneg (x := t)
  simpa only [abs_of_nonneg hnonneg] using hb ht

/-- The integral graph length is exactly the terminal intrinsic variation. -/
theorem acGraphLength_eq_intrinsicVariation (S : Set E) (epsilon a : ℝ)
    (gamma : ℝ → E) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) :
    acGraphLength S epsilon a gamma = intrinsicVariation S epsilon a gamma he 1 := by
  apply le_antisymm ?_ (intrinsicVariation_one_le_acGraphLength S epsilon a gamma he hgamma)
  have hV := absolutelyContinuousOnInterval_intrinsicVariation S epsilon a gamma he hgamma
  have hint := intervalIntegral.integral_mono_ae_restrict zero_le_one
    (ac_graphSpeed_intervalIntegrable S gamma epsilon a 0 1 he hgamma)
    hV.intervalIntegrable_deriv
    ((ae_restrict_iff' measurableSet_Icc).mpr (by
      simpa only [uIcc_of_le zero_le_one] using
        ac_graphSpeed_le_deriv_intrinsicVariation S epsilon a gamma he hgamma))
  simpa only [hV.integral_deriv_eq_sub, intrinsicVariation_zero, sub_zero,
    acGraphLength] using hint

/-- The AC graph-length functional equals the actual metric curve length,
expressed as the supremum of partition sums in the intrinsic metric. -/
theorem eVariationOn_intrinsic_eq_acGraphLength (S : Set E) (epsilon a : ℝ)
    (gamma : ℝ → E) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) :
    @eVariationOn ℝ inferInstance E
      (intrinsicMetricSpace S epsilon a he).toPseudoMetricSpace.toPseudoEMetricSpace
        gamma (Icc (0 : ℝ) 1) = ENNReal.ofReal (acGraphLength S epsilon a gamma) := by
  rw [acGraphLength_eq_intrinsicVariation S epsilon a gamma he hgamma,
    intrinsicVariation_one, ENNReal.ofReal_toReal]
  exact boundedVariationOn_intrinsic S epsilon a gamma he hgamma

/-- Arbitrarily short excess metric curve length exists between each pair:
the actual intrinsic metric is a length metric. -/
theorem exists_metric_length_lt_intrinsicDistance_add (S : Set E) (epsilon a : ℝ)
    (x y : E) (he : 0 < epsilon) (delta : ℝ) (hdelta : 0 < delta) :
    ∃ gamma : ℝ → E, AbsolutelyContinuousOnInterval gamma 0 1 ∧
      gamma 0 = x ∧ gamma 1 = y ∧
      (@eVariationOn ℝ inferInstance E
        (intrinsicMetricSpace S epsilon a he).toPseudoMetricSpace.toPseudoEMetricSpace
          gamma (Icc (0 : ℝ) 1)).toReal < intrinsicDistance S epsilon a x y + delta := by
  obtain ⟨gamma, hgamma, h0, h1, hlength⟩ :=
    exists_path_length_lt_intrinsicDistance_add S epsilon a x y delta hdelta
  exact ⟨gamma, hgamma, h0, h1, by
    simpa only [eVariationOn_intrinsic_eq_acGraphLength S epsilon a gamma he hgamma,
      ENNReal.toReal_ofReal (acGraphLength_nonneg S epsilon a gamma)] using hlength⟩

/-- An AC path with bounded intrinsic speed has at most that graph length. -/
theorem acGraphLength_le_of_intrinsic_boundedSpeed (S : Set E) (epsilon a : ℝ)
    (gamma : ℝ → E) (K : ℝ) (he : 0 < epsilon) (hK : 0 ≤ K)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1)
    (hspeed : ∀ s ∈ uIcc (0 : ℝ) 1, ∀ t ∈ uIcc (0 : ℝ) 1,
      intrinsicDistance S epsilon a (gamma s) (gamma t) ≤ K * |s - t|) :
    acGraphLength S epsilon a gamma ≤ K := by
  have hA : AbsolutelyContinuousOnInterval (fun t : ℝ => K * t) 0 1 :=
    (show AbsolutelyContinuousOnInterval (fun t : ℝ => t) 0 1 from
      LipschitzWith.id.lipschitzOnWith.absolutelyContinuousOnInterval).const_mul K
  have hbound := ac_graphSpeed_le_control_deriv S epsilon a gamma (fun t => K * t)
    he hgamma hA (fun s hs t ht => by
      simpa only [← mul_sub, abs_mul, abs_of_nonneg hK] using hspeed s hs t ht)
  have hb : ∀ᵐ t, t ∈ uIcc (0 : ℝ) 1 →
      graphSpeed a epsilon (Metric.infDist (gamma t) S) ‖deriv gamma t‖
        (deriv (fun s => Metric.infDist (gamma s) S) t) ≤ K := by
    have hd : ∀ t : ℝ, deriv (fun s => K * s) t = K := fun t => by
      simpa only [mul_one] using ((hasDerivAt_id t).const_mul K).deriv
    simpa only [hd, abs_of_nonneg hK] using hbound
  have hint := intervalIntegral.integral_mono_ae_restrict zero_le_one
    (ac_graphSpeed_intervalIntegrable S gamma epsilon a 0 1 he hgamma)
    (intervalIntegrable_const (c := K))
    ((ae_restrict_iff' measurableSet_Icc).mpr (by
      simpa only [uIcc_of_le zero_le_one] using hb))
  simpa only [acGraphLength, intervalIntegral.integral_const, sub_zero,
    one_smul] using hint

/-- Every pair has an actual AC minimizer of the original graph-length
functional, with constant intrinsic speed on all subintervals. -/
theorem exists_acGraphLength_minimizer (S : Set E) (epsilon a : ℝ) (x y : E)
    (he : 0 < epsilon) :
    ∃ gamma : ℝ → E, AbsolutelyContinuousOnInterval gamma 0 1 ∧
      gamma 0 = x ∧ gamma 1 = y ∧
      acGraphLength S epsilon a gamma = intrinsicDistance S epsilon a x y ∧
      ∀ s ∈ uIcc (0 : ℝ) 1, ∀ t ∈ uIcc (0 : ℝ) 1,
        intrinsicDistance S epsilon a (gamma s) (gamma t) =
          intrinsicDistance S epsilon a x y * |s - t| := by
  obtain ⟨gamma, hgamma, h0, h1, hspeed⟩ :=
    exists_intrinsic_ac_metricGeodesic S epsilon a x y he
  exact ⟨gamma, hgamma, h0, h1,
    le_antisymm
      (acGraphLength_le_of_intrinsic_boundedSpeed S epsilon a gamma
        (intrinsicDistance S epsilon a x y) he (intrinsicDistance_nonneg S epsilon a x y)
        hgamma (fun s hs t ht => (hspeed s hs t ht).le))
      (intrinsicDistance_le_length S epsilon a x y gamma hgamma h0 h1), hspeed⟩

/-- The path infimum is a minimum in the original admissible length set. -/
theorem intrinsicDistance_isLeast_pairwisePathLengths (S : Set E) (epsilon a : ℝ)
    (x y : E) (he : 0 < epsilon) :
    IsLeast (pairwisePathLengths S epsilon a x y) (intrinsicDistance S epsilon a x y) := by
  obtain ⟨gamma, hgamma, h0, h1, hlength, _⟩ :=
    exists_acGraphLength_minimizer S epsilon a x y he
  exact ⟨⟨gamma, hgamma, h0, h1, hlength⟩,
    fun _ hL => csInf_le (pairwisePathLengths_bddBelow S epsilon a x y) hL⟩

end FractalSync.DistanceGraph
