import FractalSync.IntrinsicMetric

/-!
# Affine subpaths and the actual graph-length integral

Restricting an AC path to an interval and moving that interval to `[0,1]`
preserves the actual graph integral. This supplies endpoint-distance bounds
on every subinterval, without assuming metric curve-length identification.
-/

namespace FractalSync.DistanceGraph

open MeasureTheory Set Filter
open scoped Topology NNReal

def affineSubpath {E : Type*} (gamma : ℝ → E) (u v t : ℝ) : E :=
  gamma ((v - u) * t + u)

@[simp] theorem affineSubpath_zero {E : Type*} (gamma : ℝ → E) (u v : ℝ) :
    affineSubpath gamma u v 0 = gamma u := by simp [affineSubpath]

@[simp] theorem affineSubpath_one {E : Type*} (gamma : ℝ → E) (u v : ℝ) :
    affineSubpath gamma u v 1 = gamma v := by simp [affineSubpath]

theorem lipschitz_affine_parameter (u v : ℝ) (huv : u ≤ v) :
    LipschitzWith ⟨v - u, sub_nonneg.mpr huv⟩ (fun t : ℝ => (v - u) * t + u) := by
  refine LipschitzWith.of_dist_le_mul fun x y => ?_
  simp only [NNReal.coe_mk, Real.dist_eq, add_sub_add_right_eq_sub,
    ← mul_sub, abs_mul, abs_of_nonneg (sub_nonneg.mpr huv), le_refl]

/-- Every oriented affine subpath is AC on the unit interval. -/
theorem ac_affineSubpath {E : Type*} [PseudoMetricSpace E]
    (gamma : ℝ → E) (u v : ℝ) (huv : u ≤ v)
    (hgamma : AbsolutelyContinuousOnInterval gamma u v) :
    AbsolutelyContinuousOnInterval (affineSubpath gamma u v) 0 1 := by
  apply ACComposition.comp_lipschitz_mono hgamma (lipschitz_affine_parameter u v huv)
    (Or.inl fun _ _ h => add_le_add_right (mul_le_mul_of_nonneg_left h
      (sub_nonneg.mpr huv)) u)
  intro t ht
  simp only [uIcc_of_le huv, uIcc_of_le (by norm_num : (0 : ℝ) ≤ 1), mem_Icc] at ht ⊢
  constructor <;> nlinarith [mul_nonneg (sub_nonneg.mpr huv) ht.1,
    mul_nonneg (sub_nonneg.mpr huv) (sub_nonneg.mpr ht.2)]

theorem deriv_affineSubpath {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (gamma : ℝ → E) (u v t : ℝ) :
    deriv (affineSubpath gamma u v) t = (v - u) • deriv gamma ((v - u) * t + u) := by
  simpa only [affineSubpath, deriv_comp_add_const] using
    (deriv_comp_mul_left (v - u) (fun s => gamma (s + u)) t)

/-- The affine derivative factor scales the actual density, even at points
where the total derivative is defined by Lean to be zero. -/
theorem graphSpeed_affineSubpath {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (S : Set E) (gamma : ℝ → E) (epsilon a u v t : ℝ) (huv : u ≤ v) :
    graphSpeed a epsilon (Metric.infDist (affineSubpath gamma u v t) S)
      ‖deriv (affineSubpath gamma u v) t‖
      (deriv (fun s => Metric.infDist (affineSubpath gamma u v s) S) t) =
    (v - u) * graphSpeed a epsilon (Metric.infDist (gamma ((v - u) * t + u)) S)
      ‖deriv gamma ((v - u) * t + u)‖
      (deriv (fun s => Metric.infDist (gamma s) S) ((v - u) * t + u)) := by
  change graphSpeed a epsilon _ ‖deriv (affineSubpath gamma u v) t‖
    (deriv (affineSubpath (fun s => Metric.infDist (gamma s) S) u v) t) = _
  rw [deriv_affineSubpath, deriv_affineSubpath, norm_smul, Real.norm_eq_abs,
    abs_of_nonneg (sub_nonneg.mpr huv), smul_eq_mul,
    graphSpeed_mul_nonneg a epsilon _ _ _ (v - u) (sub_nonneg.mpr huv)]
  rfl

/-- The actual length of an affine subpath is exactly the original density
integrated on its interval, including a zero-length interval. -/
theorem acGraphLength_affineSubpath {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (S : Set E) (gamma : ℝ → E) (epsilon a u v : ℝ) (huv : u ≤ v) :
    acGraphLength S epsilon a (affineSubpath gamma u v) =
      ∫ t in u..v, graphSpeed a epsilon (Metric.infDist (gamma t) S)
        ‖deriv gamma t‖ (deriv (fun s => Metric.infDist (gamma s) S) t) := by
  simp_rw [acGraphLength, graphSpeed_affineSubpath S gamma epsilon a u v _ huv]
  rw [intervalIntegral.integral_const_mul]
  simpa only [smul_eq_mul, mul_zero, zero_add, mul_one, sub_add_cancel] using
    (intervalIntegral.smul_integral_comp_mul_add
      (fun t => graphSpeed a epsilon (Metric.infDist (gamma t) S)
        ‖deriv gamma t‖ (deriv (fun s => Metric.infDist (gamma s) S) t))
      (a := (0 : ℝ)) (b := 1) (v - u) u)

/-- Every actual AC subinterval bounds the endpoint path-infimum distance. -/
theorem intrinsicDistance_le_subpath_integral {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] (S : Set E) (gamma : ℝ → E) (epsilon a u v : ℝ)
    (huv : u ≤ v) (hgamma : AbsolutelyContinuousOnInterval gamma u v) :
    intrinsicDistance S epsilon a (gamma u) (gamma v) ≤
      ∫ t in u..v, graphSpeed a epsilon (Metric.infDist (gamma t) S)
        ‖deriv gamma t‖ (deriv (fun s => Metric.infDist (gamma s) S) t) := by
  exact (intrinsicDistance_le_length S epsilon a (gamma u) (gamma v)
    (affineSubpath gamma u v) (ac_affineSubpath gamma u v huv hgamma)
    (affineSubpath_zero gamma u v) (affineSubpath_one gamma u v)).trans_eq
      (acGraphLength_affineSubpath S gamma epsilon a u v huv)

end FractalSync.DistanceGraph
