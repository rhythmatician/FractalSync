import FractalSync.IntrinsicLength

/-!
# Intrinsic variation of absolutely continuous graph paths

The metric used for variation is the explicit positive-floor intrinsic metric.
Its real variation trace is controlled by the actual cumulative graph integral.
-/

namespace FractalSync.DistanceGraph

open Set Filter MeasureTheory
open scoped Topology NNReal ENNReal

/-- Pairwise distance domination compares every finite partition sum, hence
the extended variations, without a differentiability assumption. -/
theorem eVariationOn_le_of_dist_le {X Y : Type*} [PseudoMetricSpace X]
    [PseudoMetricSpace Y] (f : ℝ → X) (g : ℝ → Y) (s : Set ℝ)
    (h : ∀ x ∈ s, ∀ y ∈ s, dist (f x) (f y) ≤ dist (g x) (g y)) :
    eVariationOn f s ≤ eVariationOn g s := by
  refine iSup_le (fun ⟨n, ⟨u, hu, hs⟩⟩ => ?_)
  exact (Finset.sum_le_sum (fun i _ => by
    simpa only [edist_dist] using ENNReal.ofReal_le_ofReal
      (h (u (i + 1)) (hs (i + 1)) (u i) (hs i)))).trans
        (eVariationOn.sum_le g n hu hs)

/-- Absolute continuity transfers through pairwise distance domination on
the parameter interval. -/
theorem absolutelyContinuousOnInterval_of_dist_le {X Y : Type*}
    [PseudoMetricSpace X] [PseudoMetricSpace Y] (f : ℝ → X) (g : ℝ → Y) (a b : ℝ)
    (hg : AbsolutelyContinuousOnInterval g a b)
    (h : ∀ x ∈ uIcc a b, ∀ y ∈ uIcc a b, dist (f x) (f y) ≤ dist (g x) (g y)) :
    AbsolutelyContinuousOnInterval f a b := by
  refine squeeze_zero' ?_ ?_ hg
  · exact .of_forall (fun _ => Finset.sum_nonneg (fun _ _ => dist_nonneg))
  · rw [eventually_inf_principal]
    exact .of_forall (fun ⟨n, I⟩ hI => Finset.sum_le_sum (fun i hi =>
      h _ (hI.left i hi).left _ (hI.left i hi).right))

/-- A monotone real control bounds variation on every closed subinterval. -/
theorem eVariationOn_inter_Icc_le_of_control {X : Type*} [PseudoMetricSpace X]
    (f : ℝ → X) (A : ℝ → ℝ) (s : Set ℝ) (hA : MonotoneOn A s)
    (h : ∀ x ∈ s, ∀ y ∈ s, dist (f x) (f y) ≤ |A x - A y|)
    (x y : ℝ) (hx : x ∈ s) (hy : y ∈ s) :
    eVariationOn f (s ∩ Icc x y) ≤ ENNReal.ofReal (A y - A x) := by
  exact (eVariationOn_le_of_dist_le f A (s ∩ Icc x y)
    (fun u hu v hv => by simpa only [Real.dist_eq] using h u hu.1 v hv.1)).trans
      (hA.eVariationOn_le hx hy)

/-- Finite real control gives locally bounded metric variation. -/
theorem locallyBoundedVariationOn_of_control {X : Type*} [PseudoMetricSpace X]
    (f : ℝ → X) (A : ℝ → ℝ) (s : Set ℝ) (hA : MonotoneOn A s)
    (h : ∀ x ∈ s, ∀ y ∈ s, dist (f x) (f y) ≤ |A x - A y|) :
    LocallyBoundedVariationOn f s := by
  exact fun x y hx hy => ne_top_of_le_ne_top ENNReal.ofReal_ne_top
    (eVariationOn_inter_Icc_le_of_control f A s hA h x y hx hy)

/-- The variation trace has ordered increments bounded by its monotone control. -/
theorem variationOnFromTo_sub_le_of_control {X : Type*} [PseudoMetricSpace X]
    (f : ℝ → X) (A : ℝ → ℝ) (s : Set ℝ) (hA : MonotoneOn A s)
    (h : ∀ x ∈ s, ∀ y ∈ s, dist (f x) (f y) ≤ |A x - A y|)
    (z x y : ℝ) (hz : z ∈ s) (hx : x ∈ s) (hy : y ∈ s) (hxy : x ≤ y) :
    variationOnFromTo f s z y - variationOnFromTo f s z x ≤ A y - A x := by
  rw [← variationOnFromTo.add (locallyBoundedVariationOn_of_control f A s hA h) hz hx hy,
    add_sub_cancel_left, variationOnFromTo.eq_of_le f s hxy]
  simpa only [ENNReal.toReal_ofReal (sub_nonneg.mpr (hA hx hy hxy))] using
    ENNReal.toReal_mono ENNReal.ofReal_ne_top
      (eVariationOn_inter_Icc_le_of_control f A s hA h x y hx hy)

/-- The variation trace inherits absolute increment domination by its control. -/
theorem variationOnFromTo_abs_sub_le_of_control {X : Type*} [PseudoMetricSpace X]
    (f : ℝ → X) (A : ℝ → ℝ) (s : Set ℝ) (hA : MonotoneOn A s)
    (h : ∀ x ∈ s, ∀ y ∈ s, dist (f x) (f y) ≤ |A x - A y|)
    (z x y : ℝ) (hz : z ∈ s) (hx : x ∈ s) (hy : y ∈ s) :
    |variationOnFromTo f s z x - variationOnFromTo f s z y| ≤ |A x - A y| := by
  have hV := variationOnFromTo.monotoneOn (locallyBoundedVariationOn_of_control f A s hA h) hz
  rcases le_total x y with hxy | hyx
  · simpa only [abs_of_nonpos (sub_nonpos.mpr (hV hx hy hxy)),
      abs_of_nonpos (sub_nonpos.mpr (hA hx hy hxy)), neg_sub] using
        variationOnFromTo_sub_le_of_control f A s hA h z x y hz hx hy hxy
  · simpa only [abs_of_nonneg (sub_nonneg.mpr (hV hy hx hyx)),
      abs_of_nonneg (sub_nonneg.mpr (hA hy hx hyx))] using
        variationOnFromTo_sub_le_of_control f A s hA h z y x hz hy hx hyx

/-- Metric endpoint distance is bounded by an ordered variation-trace increment. -/
theorem dist_le_variationOnFromTo_sub {X : Type*} [PseudoMetricSpace X]
    (f : ℝ → X) (s : Set ℝ) (hf : LocallyBoundedVariationOn f s)
    (z x y : ℝ) (hz : z ∈ s) (hx : x ∈ s) (hy : y ∈ s) (hxy : x ≤ y) :
    dist (f x) (f y) ≤ variationOnFromTo f s z y - variationOnFromTo f s z x := by
  have hb := (hf x y hx hy).dist_le
    (show x ∈ s ∩ Icc x y from ⟨hx, le_rfl, hxy⟩)
    (show y ∈ s ∩ Icc x y from ⟨hy, hxy, le_rfl⟩)
  simpa only [← variationOnFromTo.add hf hz hx hy, add_sub_cancel_left,
    variationOnFromTo.eq_of_le f s hxy] using hb

/-- Metric endpoint distance is bounded by the absolute variation-trace increment. -/
theorem dist_le_variationOnFromTo_abs_sub {X : Type*} [PseudoMetricSpace X]
    (f : ℝ → X) (s : Set ℝ) (hf : LocallyBoundedVariationOn f s)
    (z x y : ℝ) (hz : z ∈ s) (hx : x ∈ s) (hy : y ∈ s) :
    dist (f x) (f y) ≤ |variationOnFromTo f s z x - variationOnFromTo f s z y| := by
  rcases le_total x y with hxy | hyx
  · simpa only [abs_sub_comm] using
      (dist_le_variationOnFromTo_sub f s hf z x y hz hx hy hxy).trans (le_abs_self _)
  · simpa only [dist_comm] using
      (dist_le_variationOnFromTo_sub f s hf z y x hz hy hx hyx).trans (le_abs_self _)

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [FiniteDimensional ℝ E]

/-- Variation trace in the actual intrinsic metric, with no global instance replacement. -/
noncomputable def intrinsicVariation (S : Set E) (epsilon a : ℝ) (gamma : ℝ → E)
    (he : 0 < epsilon) (t : ℝ) : ℝ :=
  @variationOnFromTo ℝ inferInstance E
    (intrinsicMetricSpace S epsilon a he).toPseudoMetricSpace.toPseudoEMetricSpace
      gamma (Icc (0 : ℝ) 1) 0 t

/-- The cumulative actual graph integral is absolutely continuous. -/
theorem absolutelyContinuousOnInterval_acCumulativeLength
    (S : Set E) (epsilon a : ℝ) (gamma : ℝ → E) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) :
    AbsolutelyContinuousOnInterval (acCumulativeLength S epsilon a gamma) 0 1 := by
  exact (ac_graphSpeed_intervalIntegrable S gamma epsilon a 0 1 he hgamma).absolutelyContinuousOnInterval_intervalIntegral
    (left_mem_uIcc : (0 : ℝ) ∈ uIcc 0 1)

/-- The cumulative graph length is monotone on the path interval. -/
theorem monotoneOn_acCumulativeLength (S : Set E) (epsilon a : ℝ) (gamma : ℝ → E)
    (he : 0 < epsilon) (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) :
    MonotoneOn (acCumulativeLength S epsilon a gamma) (Icc (0 : ℝ) 1) := by
  exact fun u hu v hv huv => sub_nonneg.mp
    ((intrinsicDistance_nonneg S epsilon a (gamma u) (gamma v)).trans
      (intrinsicDistance_le_acCumulativeLength_sub S epsilon a gamma he hgamma u v
        (by simpa only [uIcc_of_le zero_le_one] using hu)
        (by simpa only [uIcc_of_le zero_le_one] using hv) huv))

/-- The actual intrinsic metric variation is at most the graph length integral. -/
theorem eVariationOn_intrinsic_le_acGraphLength (S : Set E) (epsilon a : ℝ)
    (gamma : ℝ → E) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) :
    @eVariationOn ℝ inferInstance E
      (intrinsicMetricSpace S epsilon a he).toPseudoMetricSpace.toPseudoEMetricSpace
        gamma (Icc (0 : ℝ) 1) ≤ ENNReal.ofReal (acGraphLength S epsilon a gamma) := by
  have hb := @eVariationOn_inter_Icc_le_of_control E
    (intrinsicMetricSpace S epsilon a he).toPseudoMetricSpace gamma
    (acCumulativeLength S epsilon a gamma) (Icc (0 : ℝ) 1)
    (monotoneOn_acCumulativeLength S epsilon a gamma he hgamma)
    (fun u hu v hv => intrinsicDistance_le_abs_acCumulativeLength_sub
      S epsilon a gamma he hgamma u v
      (by simpa only [uIcc_of_le zero_le_one] using hu)
      (by simpa only [uIcc_of_le zero_le_one] using hv))
    0 1 (by simp) (by simp)
  simpa only [inter_self, acCumulativeLength_one, acCumulativeLength_zero, sub_zero] using hb

/-- An ordinary absolutely continuous path has finite variation in the intrinsic metric. -/
theorem boundedVariationOn_intrinsic (S : Set E) (epsilon a : ℝ)
    (gamma : ℝ → E) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) :
    @BoundedVariationOn ℝ inferInstance E
      (intrinsicMetricSpace S epsilon a he).toPseudoMetricSpace.toPseudoEMetricSpace
        gamma (Icc (0 : ℝ) 1) := by
  exact ne_top_of_le_ne_top ENNReal.ofReal_ne_top
    (eVariationOn_intrinsic_le_acGraphLength S epsilon a gamma he hgamma)

/-- Local bounded variation with the intrinsic metric instance made explicit. -/
theorem locallyBoundedVariationOn_intrinsic (S : Set E) (epsilon a : ℝ)
    (gamma : ℝ → E) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) :
    @LocallyBoundedVariationOn ℝ inferInstance E
      (intrinsicMetricSpace S epsilon a he).toPseudoMetricSpace.toPseudoEMetricSpace
        gamma (Icc (0 : ℝ) 1) := by
  exact @BoundedVariationOn.locallyBoundedVariationOn ℝ inferInstance E
    (intrinsicMetricSpace S epsilon a he).toPseudoMetricSpace.toPseudoEMetricSpace
      gamma (Icc (0 : ℝ) 1) (boundedVariationOn_intrinsic S epsilon a gamma he hgamma)

@[simp] theorem intrinsicVariation_zero (S : Set E) (epsilon a : ℝ)
    (gamma : ℝ → E) (he : 0 < epsilon) : intrinsicVariation S epsilon a gamma he 0 = 0 := by
  exact @variationOnFromTo.self ℝ inferInstance E
    (intrinsicMetricSpace S epsilon a he).toPseudoMetricSpace.toPseudoEMetricSpace
      gamma (Icc (0 : ℝ) 1) 0

/-- The terminal variation trace is the actual intrinsic extended variation converted to real. -/
theorem intrinsicVariation_one (S : Set E) (epsilon a : ℝ)
    (gamma : ℝ → E) (he : 0 < epsilon) :
    intrinsicVariation S epsilon a gamma he 1 =
      (@eVariationOn ℝ inferInstance E
        (intrinsicMetricSpace S epsilon a he).toPseudoMetricSpace.toPseudoEMetricSpace
          gamma (Icc (0 : ℝ) 1)).toReal := by
  simp [intrinsicVariation, variationOnFromTo]

/-- The terminal real variation is bounded by the graph length integral. -/
theorem intrinsicVariation_one_le_acGraphLength (S : Set E) (epsilon a : ℝ)
    (gamma : ℝ → E) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) :
    intrinsicVariation S epsilon a gamma he 1 ≤ acGraphLength S epsilon a gamma := by
  simpa only [intrinsicVariation_one,
    ENNReal.toReal_ofReal (acGraphLength_nonneg S epsilon a gamma)] using
      ENNReal.toReal_mono ENNReal.ofReal_ne_top
        (eVariationOn_intrinsic_le_acGraphLength S epsilon a gamma he hgamma)

/-- The intrinsic variation trace is monotone on the path interval. -/
theorem monotoneOn_intrinsicVariation (S : Set E) (epsilon a : ℝ)
    (gamma : ℝ → E) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) :
    MonotoneOn (intrinsicVariation S epsilon a gamma he) (Icc (0 : ℝ) 1) := by
  exact @variationOnFromTo.monotoneOn ℝ inferInstance E
    (intrinsicMetricSpace S epsilon a he).toPseudoMetricSpace.toPseudoEMetricSpace
    gamma (Icc (0 : ℝ) 1)
    (locallyBoundedVariationOn_intrinsic S epsilon a gamma he hgamma)
    0 (by simp)

/-- Intrinsic variation increments are controlled by actual cumulative graph-length increments. -/
theorem intrinsicVariation_abs_sub_le_acCumulativeLength_sub (S : Set E)
    (epsilon a : ℝ) (gamma : ℝ → E) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) (u v : ℝ)
    (hu : u ∈ Icc (0 : ℝ) 1) (hv : v ∈ Icc (0 : ℝ) 1) :
    |intrinsicVariation S epsilon a gamma he u - intrinsicVariation S epsilon a gamma he v| ≤
      |acCumulativeLength S epsilon a gamma u - acCumulativeLength S epsilon a gamma v| := by
  exact @variationOnFromTo_abs_sub_le_of_control E
    (intrinsicMetricSpace S epsilon a he).toPseudoMetricSpace gamma
    (acCumulativeLength S epsilon a gamma) (Icc (0 : ℝ) 1)
    (monotoneOn_acCumulativeLength S epsilon a gamma he hgamma)
    (fun x hx y hy => intrinsicDistance_le_abs_acCumulativeLength_sub
      S epsilon a gamma he hgamma x y
      (by simpa only [uIcc_of_le zero_le_one] using hx)
      (by simpa only [uIcc_of_le zero_le_one] using hy))
    0 u v (by simp) hu hv

/-- The actual intrinsic distance between path points is bounded by the variation trace. -/
theorem intrinsicDistance_le_intrinsicVariation_abs_sub (S : Set E)
    (epsilon a : ℝ) (gamma : ℝ → E) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) (u v : ℝ)
    (hu : u ∈ Icc (0 : ℝ) 1) (hv : v ∈ Icc (0 : ℝ) 1) :
    intrinsicDistance S epsilon a (gamma u) (gamma v) ≤
      |intrinsicVariation S epsilon a gamma he u - intrinsicVariation S epsilon a gamma he v| := by
  exact @dist_le_variationOnFromTo_abs_sub E
    (intrinsicMetricSpace S epsilon a he).toPseudoMetricSpace gamma (Icc (0 : ℝ) 1)
    (locallyBoundedVariationOn_intrinsic S epsilon a gamma he hgamma) 0 u v (by simp) hu hv

/-- The actual intrinsic variation trace is absolutely continuous as a real function. -/
theorem absolutelyContinuousOnInterval_intrinsicVariation (S : Set E)
    (epsilon a : ℝ) (gamma : ℝ → E) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) :
    AbsolutelyContinuousOnInterval (intrinsicVariation S epsilon a gamma he) 0 1 := by
  exact absolutelyContinuousOnInterval_of_dist_le
    (intrinsicVariation S epsilon a gamma he) (acCumulativeLength S epsilon a gamma) 0 1
    (absolutelyContinuousOnInterval_acCumulativeLength S epsilon a gamma he hgamma)
    (fun u hu v hv => by
      simpa only [Real.dist_eq] using
        intrinsicVariation_abs_sub_le_acCumulativeLength_sub S epsilon a gamma he hgamma u v
          (by simpa only [uIcc_of_le zero_le_one] using hu)
          (by simpa only [uIcc_of_le zero_le_one] using hv))

end FractalSync.DistanceGraph
