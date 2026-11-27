import Mathlib.Analysis.Calculus.ParametricIntegral
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic

/-!
# Actual differentiation of a parameter integral over a compact interval

Joint continuity of the integrand and its actual parameter derivative gives
the uniform local domination needed by Mathlib's differentiation theorem.
-/

namespace FractalSync.CompactParametricIntegral

open Set Filter MeasureTheory
open scoped Topology

/-- A jointly continuous actual parameter derivative may be integrated on a
fixed compact interval. The domination constant is derived by compactness. -/
theorem hasDerivAt_intervalIntegral (F F' : ℝ × ℝ → ℝ)
    (hF : Continuous F) (hF' : Continuous F')
    (hd : ∀ A t, HasDerivAt (fun B => F (B, t)) (F' (A, t)) A)
    (lo hi A : ℝ) (hle : lo ≤ hi) :
    HasDerivAt (fun B => ∫ t in lo..hi, F (B, t))
      (∫ t in lo..hi, F' (A, t)) A := by
  obtain ⟨C, hC⟩ := ((isCompact_closedBall A 1).prod (isCompact_Icc (a := lo) (b := hi))).exists_bound_of_continuousOn
    (f := F') hF'.continuousOn
  have hm : ∀ᶠ B in 𝓝 A, AEStronglyMeasurable (fun t => F (B, t))
      (volume.restrict (Ioc lo hi)) :=
    Filter.Eventually.of_forall (fun B =>
      (hF.comp (continuous_const.prodMk continuous_id)).aestronglyMeasurable)
  have hbound : ∀ᵐ t ∂volume.restrict (Ioc lo hi),
      ∀ B ∈ Metric.ball A 1, ‖F' (B, t)‖ ≤ C := by
    filter_upwards [ae_restrict_mem measurableSet_Ioc] with t ht
    exact fun B hB => hC (B, t) ⟨Metric.ball_subset_closedBall hB, ht.1.le, ht.2⟩
  have hactual := hasDerivAt_integral_of_dominated_loc_of_deriv_le
    (F := fun B t => F (B, t)) (F' := fun B t => F' (B, t)) (bound := fun _ => C)
    (by norm_num : (0 : ℝ) < 1) hm
    ((hF.comp (continuous_const.prodMk continuous_id)).intervalIntegrable lo hi).1
    ((hF'.comp (continuous_const.prodMk continuous_id)).aestronglyMeasurable)
    hbound (integrable_const C) (Filter.Eventually.of_forall (fun t B _ => hd B t))
  simpa only [intervalIntegral.integral_of_le hle] using hactual.2

end FractalSync.CompactParametricIntegral
