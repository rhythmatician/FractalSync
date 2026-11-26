import FractalSync.PairwiseDistance

/-!
# Uniform comparison with the scale-only intrinsic metric

The original scalar comparison is integrated along every actual AC path and
then passed through the actual path infima. The common admissible path class
does not depend on the graph parameter.
-/

namespace FractalSync.DistanceGraph

open MeasureTheory Set Filter

/-- The scale-only density is bounded by every graph density. -/
theorem graphSpeed_zero_le (epsilon a r speed w : ℝ) :
    graphSpeed 0 epsilon r speed w ≤ graphSpeed a epsilon r speed w := by
  apply Real.sqrt_le_sqrt
  simpa only [graphSpeedSq, zero_pow (by norm_num : (2 : ℕ) ≠ 0), zero_mul,
    zero_div, add_zero, le_add_iff_nonneg_right] using
      (div_nonneg (mul_nonneg (mul_nonneg (sq_nonneg a) (sq_nonneg r)) (sq_nonneg w))
        (sq_nonneg (r ^ 2 + epsilon ^ 2)))

/-- The graph density is at most sqrt(1+a²) times the scale-only density. -/
theorem graphSpeed_le_sqrt_mul_zero (epsilon a r speed w : ℝ)
    (he : 0 < epsilon) (hw : w ^ 2 ≤ speed ^ 2) :
    graphSpeed a epsilon r speed w ≤
      Real.sqrt (1 + a ^ 2) * graphSpeed 0 epsilon r speed w := by
  simpa only [graphSpeed, graphSpeedSq, zero_pow (by norm_num : (2 : ℕ) ≠ 0),
    zero_mul, zero_div, add_zero, Real.sqrt_mul (by positivity : 0 ≤ 1 + a ^ 2)] using
      Real.sqrt_le_sqrt (graph_scale_comparison a epsilon r speed w he hw).2

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [FiniteDimensional ℝ E]

/-- Every actual AC graph length dominates its scale-only length. -/
theorem acGraphLength_zero_le (S : Set E) (epsilon a : ℝ) (gamma : ℝ → E)
    (he : 0 < epsilon) (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) :
    acGraphLength S epsilon 0 gamma ≤ acGraphLength S epsilon a gamma := by
  exact intervalIntegral.integral_mono (by norm_num)
    (ac_graphSpeed_intervalIntegrable S gamma epsilon 0 0 1 he hgamma)
    (ac_graphSpeed_intervalIntegrable S gamma epsilon a 0 1 he hgamma)
    (fun t => graphSpeed_zero_le epsilon a (Metric.infDist (gamma t) S)
      ‖deriv gamma t‖ (deriv (fun s => Metric.infDist (gamma s) S) t))

/-- The uniform pointwise comparison integrates for every actual AC path. -/
theorem acGraphLength_le_sqrt_mul_zero (S : Set E) (epsilon a : ℝ)
    (gamma : ℝ → E) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) :
    acGraphLength S epsilon a gamma ≤
      Real.sqrt (1 + a ^ 2) * acGraphLength S epsilon 0 gamma := by
  unfold acGraphLength
  rw [← intervalIntegral.integral_const_mul]
  apply intervalIntegral.integral_mono_ae_restrict (by norm_num)
    (ac_graphSpeed_intervalIntegrable S gamma epsilon a 0 1 he hgamma)
    ((ac_graphSpeed_intervalIntegrable S gamma epsilon 0 0 1 he hgamma).const_mul _)
  exact (ae_restrict_iff' measurableSet_Icc).2
    ((ac_distance_trace_deriv_bound S gamma 0 1 hgamma).mono fun t ht hmem =>
      graphSpeed_le_sqrt_mul_zero epsilon a _ _ _ he (by
        simpa only [sq_abs] using (sq_le_sq₀ (abs_nonneg _) (norm_nonneg _)).2
          (ht (by simpa only [uIcc_of_le (by norm_num : (0 : ℝ) ≤ 1)] using hmem))))

/-- The scale-only intrinsic metric bounds the graph metric from below. -/
theorem intrinsicDistance_zero_le (S : Set E) (epsilon a : ℝ) (x y : E)
    (he : 0 < epsilon) :
    intrinsicDistance S epsilon 0 x y ≤ intrinsicDistance S epsilon a x y := by
  exact le_csInf (pairwisePathLengths_nonempty S epsilon a x y)
    (fun _ ⟨gamma, hgamma, hstart, hend, hlength⟩ => hlength ▸
      (intrinsicDistance_le_length S epsilon 0 x y gamma hgamma hstart hend).trans
        (acGraphLength_zero_le S epsilon a gamma he hgamma))

/-- The graph parameter changes pairwise distances by at most sqrt(1+a²). -/
theorem intrinsicDistance_le_sqrt_mul_zero (S : Set E) (epsilon a : ℝ) (x y : E)
    (he : 0 < epsilon) :
    intrinsicDistance S epsilon a x y ≤
      Real.sqrt (1 + a ^ 2) * intrinsicDistance S epsilon 0 x y := by
  have hb : 0 < Real.sqrt (1 + a ^ 2) := by positivity
  suffices h : intrinsicDistance S epsilon a x y / Real.sqrt (1 + a ^ 2) ≤
      intrinsicDistance S epsilon 0 x y by
    simpa only [mul_comm] using (div_le_iff₀ hb).mp h
  exact le_csInf (pairwisePathLengths_nonempty S epsilon 0 x y)
    (fun _ ⟨gamma, hgamma, hstart, hend, hlength⟩ => hlength ▸
      (div_le_iff₀ hb).mpr (by
        simpa only [mul_comm] using
          (intrinsicDistance_le_length S epsilon a x y gamma hgamma hstart hend).trans
            (acGraphLength_le_sqrt_mul_zero S epsilon a gamma he hgamma)))

end FractalSync.DistanceGraph
