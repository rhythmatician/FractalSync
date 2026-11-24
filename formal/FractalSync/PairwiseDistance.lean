import FractalSync.IntrinsicShoreDistance
import FractalSync.ComponentCrossing

/-!
# The actual pairwise absolutely continuous path infimum

The admissible class uses the same graph length as the attained Shore-distance
theorem. Topological separation is converted to a uniform bound on every path,
then passed through the infimum. No minimizing-path existence is assumed.
-/

namespace FractalSync.DistanceGraph

open MeasureTheory Set Filter
open scoped Topology

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Lengths of all absolutely continuous paths with the specified endpoints. -/
def pairwisePathLengths (S : Set E) (epsilon a : ℝ) (x y : E) : Set ℝ :=
  {L | ∃ gamma : ℝ → E, AbsolutelyContinuousOnInterval gamma 0 1 ∧
    gamma 0 = x ∧ gamma 1 = y ∧ acGraphLength S epsilon a gamma = L}

/-- Pairwise infimum of the actual absolutely continuous graph-length functional.
The positive-floor theorems below establish its geometric properties. -/
noncomputable def intrinsicDistance (S : Set E) (epsilon a : ℝ) (x y : E) : ℝ :=
  sInf (pairwisePathLengths S epsilon a x y)

/-- An affine segment makes the actual pairwise length class nonempty. -/
theorem pairwisePathLengths_nonempty (S : Set E) (epsilon a : ℝ) (x y : E) :
    (pairwisePathLengths S epsilon a x y).Nonempty := by
  exact ⟨_, (fun t => AffineMap.lineMap x y t),
    absolutelyContinuousOnInterval_lineMap y x,
    by simp only [AffineMap.lineMap_apply_zero],
    by simp only [AffineMap.lineMap_apply_one], rfl⟩

/-- Nonnegativity supplies a common lower bound for every admissible class. -/
theorem pairwisePathLengths_bddBelow (S : Set E) (epsilon a : ℝ) (x y : E) :
    BddBelow (pairwisePathLengths S epsilon a x y) := by
  exact ⟨0, fun _ ⟨gamma, _, _, _, hlength⟩ =>
    hlength ▸ acGraphLength_nonneg S epsilon a gamma⟩

/-- The actual pairwise path infimum is nonnegative. -/
theorem intrinsicDistance_nonneg (S : Set E) (epsilon a : ℝ) (x y : E) :
    0 ≤ intrinsicDistance S epsilon a x y := by
  exact le_csInf (pairwisePathLengths_nonempty S epsilon a x y)
    (fun _ ⟨gamma, _, _, _, hlength⟩ =>
      hlength ▸ acGraphLength_nonneg S epsilon a gamma)

/-- Every admissible actual length bounds the pairwise infimum from above. -/
theorem intrinsicDistance_le_length (S : Set E) (epsilon a : ℝ) (x y : E)
    (gamma : ℝ → E) (hgamma : AbsolutelyContinuousOnInterval gamma 0 1)
    (hstart : gamma 0 = x) (hend : gamma 1 = y) :
    intrinsicDistance S epsilon a x y ≤ acGraphLength S epsilon a gamma := by
  exact csInf_le (pairwisePathLengths_bddBelow S epsilon a x y)
    ⟨gamma, hgamma, hstart, hend, rfl⟩

/-- The constant path gives zero pairwise self-distance. -/
theorem intrinsicDistance_self (S : Set E) (epsilon a : ℝ) (x : E) :
    intrinsicDistance S epsilon a x x = 0 := by
  apply le_antisymm ?_ (intrinsicDistance_nonneg S epsilon a x x)
  simpa [acGraphLength, AffineMap.lineMap_same_apply, graphSpeed, graphSpeedSq] using
    intrinsicDistance_le_length S epsilon a x x (fun t => AffineMap.lineMap x x t)
      (absolutelyContinuousOnInterval_lineMap x x) (by simp) (by simp)

/-- The unrestricted endpoint calibration passes to the actual pairwise
infimum. No minimizing path is assumed. -/
theorem radialProfile_difference_le_intrinsicDistance [FiniteDimensional ℝ E]
    (S : Set E) (epsilon a : ℝ) (x y : E) (he : 0 < epsilon) :
    |radialProfile epsilon a (Metric.infDist y S) -
      radialProfile epsilon a (Metric.infDist x S)| ≤
      intrinsicDistance S epsilon a x y := by
  exact le_csInf (pairwisePathLengths_nonempty S epsilon a x y)
    (fun _ ⟨gamma, hgamma, hstart, hend, hlength⟩ => hlength ▸ (by
      simpa only [hstart, hend] using
        ac_distance_trace_endpoint_bound S gamma epsilon a 0 1 he (by norm_num) hgamma))

/-- Component separation forces every admissible path to meet Shore, so the
sum of the endpoint radial profiles bounds the actual pairwise infimum. -/
theorem radialProfile_sum_le_intrinsicDistance_of_separated_components
    [FiniteDimensional ℝ E] (S : Set E) (epsilon a : ℝ) (x y : E)
    (he : 0 < epsilon) (hsep : y ∉ connectedComponentIn Sᶜ x) :
    radialProfile epsilon a (Metric.infDist x S) +
      radialProfile epsilon a (Metric.infDist y S) ≤ intrinsicDistance S epsilon a x y := by
  exact le_csInf (pairwisePathLengths_nonempty S epsilon a x y)
    (fun _ ⟨gamma, hgamma, hstart, hend, hlength⟩ => hlength ▸ (by
      simpa only [hstart, hend] using
        ac_distance_trace_component_bound S gamma epsilon a 0 1 he (by norm_num)
          hgamma (by simpa only [hstart, hend] using hsep)))

/-- The actual pairwise path infimum diverges as the positive floor vanishes
when `x` lies off a nonempty closed Shore and `y` is outside `x`'s connected
component of its complement. The endpoint `y` may itself lie on Shore. -/
theorem intrinsicDistance_tendsto_zero_floor_of_separated_components
    [FiniteDimensional ℝ E] (S : Set E) (hclosed : IsClosed S) (hne : S.Nonempty)
    (x y : E) (hx : x ∉ S) (hsep : y ∉ connectedComponentIn Sᶜ x) (a : ℝ) :
    Tendsto (fun epsilon => intrinsicDistance S epsilon a x y)
      (𝓝[>] (0 : ℝ)) atTop := by
  refine tendsto_atTop_mono' (𝓝[>] (0 : ℝ)) ?_
    (radialProfile_tendsto_zero_floor a (Metric.infDist x S)
      ((hclosed.notMem_iff_infDist_pos hne).1 hx))
  filter_upwards [self_mem_nhdsWithin] with epsilon he
  exact (le_add_of_nonneg_right (radialProfile_nonneg epsilon a
    (Metric.infDist y S) he Metric.infDist_nonneg)).trans
      (radialProfile_sum_le_intrinsicDistance_of_separated_components
        S epsilon a x y he hsep)

/-- Increasing the positive floor decreases graph speed for fixed scalar
distance and velocity data. No distance-trace inequality is needed here. -/
theorem graphSpeed_antitone_floor (epsilon₁ epsilon₂ a r speed w : ℝ)
    (he : 0 < epsilon₁) (hfloor : epsilon₁ ≤ epsilon₂) :
    graphSpeed a epsilon₂ r speed w ≤ graphSpeed a epsilon₁ r speed w := by
  apply Real.sqrt_le_sqrt
  have hden : 0 < r ^ 2 + epsilon₁ ^ 2 ∧
      r ^ 2 + epsilon₁ ^ 2 ≤ r ^ 2 + epsilon₂ ^ 2 :=
    ⟨by positivity, add_le_add_left ((sq_le_sq₀ he.le (he.trans_le hfloor).le).2 hfloor) _⟩
  exact add_le_add
    (div_le_div_of_nonneg_left (sq_nonneg speed) hden.1 hden.2)
    (div_le_div_of_nonneg_left (by positivity) (sq_pos_of_pos hden.1)
      ((sq_le_sq₀ hden.1.le (hden.1.trans_le hden.2).le).2 hden.2))

/-- The actual length of a fixed AC curve decreases as its positive floor
increases. Integrability at both floors follows from absolute continuity. -/
theorem acGraphLength_antitone_floor [FiniteDimensional ℝ E]
    (S : Set E) (epsilon₁ epsilon₂ a : ℝ) (gamma : ℝ → E)
    (he : 0 < epsilon₁) (hfloor : epsilon₁ ≤ epsilon₂)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) :
    acGraphLength S epsilon₂ a gamma ≤ acGraphLength S epsilon₁ a gamma := by
  exact intervalIntegral.integral_mono (by norm_num)
    (ac_graphSpeed_intervalIntegrable S gamma epsilon₂ a 0 1 (he.trans_le hfloor) hgamma)
    (ac_graphSpeed_intervalIntegrable S gamma epsilon₁ a 0 1 he hgamma)
    (fun t => graphSpeed_antitone_floor epsilon₁ epsilon₂ a
      (Metric.infDist (gamma t) S) ‖deriv gamma t‖
      (deriv (fun s => Metric.infDist (gamma s) S) t) he hfloor)

/-- Increasing the positive floor decreases the actual pairwise path infimum.
This comparison does not require an attained infimum. -/
theorem intrinsicDistance_antitone_floor [FiniteDimensional ℝ E]
    (S : Set E) (epsilon₁ epsilon₂ a : ℝ) (x y : E)
    (he : 0 < epsilon₁) (hfloor : epsilon₁ ≤ epsilon₂) :
    intrinsicDistance S epsilon₂ a x y ≤ intrinsicDistance S epsilon₁ a x y := by
  exact le_csInf (pairwisePathLengths_nonempty S epsilon₁ a x y)
    (fun _ ⟨gamma, hgamma, hstart, hend, hlength⟩ => hlength ▸
      (intrinsicDistance_le_length S epsilon₂ a x y gamma hgamma hstart hend).trans
        (acGraphLength_antitone_floor S epsilon₁ epsilon₂ a gamma he hfloor hgamma))

/-- The graph length is bounded by a global constant times the ordinary AC
length. All velocity and graph-speed integrability follows from AC. -/
theorem acGraphLength_le_global [FiniteDimensional ℝ E]
    (S : Set E) (epsilon a : ℝ) (gamma : ℝ → E) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) :
    acGraphLength S epsilon a gamma ≤
      Real.sqrt (1 + a ^ 2) / epsilon * ∫ t in (0 : ℝ)..1, ‖deriv gamma t‖ := by
  rw [← intervalIntegral.integral_const_mul]
  apply intervalIntegral.integral_mono_ae_restrict (by norm_num)
    (ac_graphSpeed_intervalIntegrable S gamma epsilon a 0 1 he hgamma)
    (hgamma.intervalIntegrable_deriv_vector.norm.const_mul _)
  rw [EventuallyLE, ae_restrict_iff' measurableSet_Icc]
  filter_upwards [ac_distance_trace_deriv_bound S gamma 0 1 hgamma] with t ht hmem
  apply graphSpeed_le_global epsilon a _ _ _ he (norm_nonneg _)
  simpa only [sq_abs] using (sq_le_sq₀ (abs_nonneg _) (norm_nonneg _)).2
    (ht (by simpa only [uIcc_of_le (by norm_num : (0 : ℝ) ≤ 1)] using hmem))

/-- The straight segment gives a uniform Euclidean upper bound for the actual
pairwise path infimum, for every set `S` and every real `a`. -/
theorem intrinsicDistance_le_global [FiniteDimensional ℝ E]
    (S : Set E) (epsilon a : ℝ) (x y : E) (he : 0 < epsilon) :
    intrinsicDistance S epsilon a x y ≤
      Real.sqrt (1 + a ^ 2) / epsilon * dist x y := by
  have hderiv : deriv (fun t : ℝ => AffineMap.lineMap x y t) = fun _ => y - x :=
    funext (fun t => (hasDerivAt_nearest_segment y x t).deriv)
  simpa only [hderiv, intervalIntegral.integral_const, sub_zero, one_smul,
    dist_eq_norm, norm_sub_rev y x] using
      (intrinsicDistance_le_length S epsilon a x y (fun t => AffineMap.lineMap x y t)
        (absolutelyContinuousOnInterval_lineMap y x) (by simp) (by simp)).trans
          (acGraphLength_le_global S epsilon a (fun t => AffineMap.lineMap x y t) he
            (absolutelyContinuousOnInterval_lineMap y x))

end FractalSync.DistanceGraph
