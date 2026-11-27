import FractalSync.AbsolutelyContinuousPaths

/-!
# Duration and kinetic-action obstructions to a Shore crossing

These results integrate the length density of an arbitrary absolutely
continuous finite-dimensional curve. The quadratic functional is kinetic
action. It is not mechanical work or a sufficient crossing criterion.
-/

open MeasureTheory Set Filter

namespace FractalSync.DistanceGraph

/-- An exact integral square expansion, with both required integrability
hypotheses explicit. -/
theorem integral_affine_sq (f : ℝ → ℝ) (lo hi scale offset : ℝ)
    (hf : IntervalIntegrable f volume lo hi)
    (hsq : IntervalIntegrable (fun t => f t ^ 2) volume lo hi) :
    (∫ t in lo..hi, (scale * f t - offset) ^ 2) =
      scale ^ 2 * (∫ t in lo..hi, f t ^ 2) -
        2 * scale * offset * (∫ t in lo..hi, f t) + (hi - lo) * offset ^ 2 := by
  calc
    _ = ∫ t in lo..hi, (scale ^ 2 * f t ^ 2 - (2 * scale * offset) * f t) +
        offset ^ 2 := intervalIntegral.integral_congr (fun _ _ => by ring)
    _ = _ := by
      rw [intervalIntegral.integral_add ((hsq.const_mul _).sub (hf.const_mul _))
        intervalIntegrable_const, intervalIntegral.integral_sub (hsq.const_mul _) (hf.const_mul _),
        intervalIntegral.integral_const_mul, intervalIntegral.integral_const_mul,
        intervalIntegral.integral_const]
      rfl

/-- Interval Cauchy-Schwarz for a scalar integrable speed with an integrable
square. The proof is the nonnegativity of an actual integral of a square. -/
theorem sq_integral_le_duration_mul_integral_sq (f : ℝ → ℝ) (lo hi : ℝ)
    (horder : lo < hi) (hf : IntervalIntegrable f volume lo hi)
    (hsq : IntervalIntegrable (fun t => f t ^ 2) volume lo hi) :
    (∫ t in lo..hi, f t) ^ 2 ≤ (hi - lo) * ∫ t in lo..hi, f t ^ 2 := by
  have hnonneg := intervalIntegral.integral_nonneg_of_forall (μ := volume) horder.le
    (fun t => sq_nonneg ((hi - lo) * f t - ∫ s in lo..hi, f s))
  rw [integral_affine_sq f lo hi (hi - lo) (∫ s in lo..hi, f s) hf hsq] at hnonneg
  have hprod : 0 ≤ (hi - lo) *
      ((hi - lo) * (∫ t in lo..hi, f t ^ 2) - (∫ t in lo..hi, f t) ^ 2) := by
    nlinarith only [hnonneg]
  exact sub_nonneg.mp (nonneg_of_mul_nonneg_right hprod (sub_pos.mpr horder))

/-- A nonnegative lower bound on integrated speed forces a quadratic kinetic
action cost over a positive duration. -/
theorem kinetic_action_lower_bound (speed : ℝ → ℝ) (lo hi bound : ℝ)
    (horder : lo < hi) (hb : 0 ≤ bound)
    (hint : IntervalIntegrable speed volume lo hi)
    (hsq : IntervalIntegrable (fun t => speed t ^ 2) volume lo hi)
    (hlen : bound ≤ ∫ t in lo..hi, speed t) :
    bound ^ 2 / (2 * (hi - lo)) ≤ ∫ t in lo..hi, speed t ^ 2 / 2 := by
  have hcs := sq_integral_le_duration_mul_integral_sq speed lo hi horder hint hsq
  have hbound := (sq_le_sq₀ hb (hb.trans hlen)).2 hlen
  rw [intervalIntegral.integral_div]
  apply (div_le_iff₀ (mul_pos (by norm_num : (0 : ℝ) < 2) (sub_pos.mpr horder))).2
  nlinarith only [hbound, hcs]

/-- An a.e. intrinsic speed cap bounds how much calibrated radial distance can
be crossed in a specified duration, including zero-duration intervals. -/
theorem ac_crossing_duration_bound {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] [FiniteDimensional ℝ E]
    (S : Set E) (gamma : ℝ → E) (epsilon a lo c hi cap : ℝ)
    (he : 0 < epsilon) (hlc : lo ≤ c) (hchi : c ≤ hi)
    (hgamma : AbsolutelyContinuousOnInterval gamma lo hi) (hshore : gamma c ∈ S)
    (hcap : ∀ᵐ t ∂volume.restrict (Icc lo hi),
      graphSpeed a epsilon (Metric.infDist (gamma t) S)
        ‖deriv gamma t‖ (deriv (fun s => Metric.infDist (gamma s) S) t) ≤ cap) :
    radialProfile epsilon a (Metric.infDist (gamma lo) S) +
      radialProfile epsilon a (Metric.infDist (gamma hi) S) ≤ cap * (hi - lo) := by
  apply (ac_distance_trace_crossing_bound S gamma epsilon a lo c hi
    he hlc hchi hgamma hshore).trans
  simpa only [intervalIntegral.integral_const, smul_eq_mul, mul_comm] using
    intervalIntegral.integral_mono_ae_restrict (hlc.trans hchi)
      (ac_graphSpeed_intervalIntegrable S gamma epsilon a lo hi he hgamma)
      (intervalIntegrable_const (c := cap)) hcap

/-- Every absolutely continuous Shore crossing has this kinetic-action lower
bound. Absolute continuity supplies speed integrability, while integrability
of its square is the additional finite-action hypothesis. -/
theorem ac_crossing_kinetic_action_bound {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] [FiniteDimensional ℝ E]
    (S : Set E) (gamma : ℝ → E) (epsilon a lo c hi : ℝ)
    (he : 0 < epsilon) (hlc : lo ≤ c) (hchi : c ≤ hi) (horder : lo < hi)
    (hgamma : AbsolutelyContinuousOnInterval gamma lo hi) (hshore : gamma c ∈ S)
    (hsq : IntervalIntegrable (fun t =>
      (graphSpeed a epsilon (Metric.infDist (gamma t) S)
        ‖deriv gamma t‖ (deriv (fun s => Metric.infDist (gamma s) S) t)) ^ 2) volume lo hi) :
    (radialProfile epsilon a (Metric.infDist (gamma lo) S) +
      radialProfile epsilon a (Metric.infDist (gamma hi) S)) ^ 2 / (2 * (hi - lo)) ≤
      ∫ t in lo..hi, (graphSpeed a epsilon (Metric.infDist (gamma t) S)
        ‖deriv gamma t‖ (deriv (fun s => Metric.infDist (gamma s) S) t)) ^ 2 / 2 := by
  apply kinetic_action_lower_bound _ lo hi _ horder
    (add_nonneg (radialProfile_nonneg epsilon a _ he Metric.infDist_nonneg)
      (radialProfile_nonneg epsilon a _ he Metric.infDist_nonneg))
    (ac_graphSpeed_intervalIntegrable S gamma epsilon a lo hi he hgamma) hsq
  exact ac_distance_trace_crossing_bound S gamma epsilon a lo c hi he hlc hchi hgamma hshore

end FractalSync.DistanceGraph
