import Mathlib.Analysis.Normed.Affine.AddTorsor
import Mathlib.Analysis.Normed.Module.FiniteDimension
import Mathlib.Topology.MetricSpace.HausdorffDistance
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.Calculus.Deriv.Slope
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Tactic.Linarith

/-!
# Distance along a nearest-point segment

This is the geometric equality case needed by issue #160. It uses Mathlib's
actual distance to a set, not an abstract radial coordinate. No differentiable
distance field, convexity of the set, or uniqueness of the nearest point is
assumed. The interval includes both endpoints, including a zero-length segment.
-/

namespace FractalSync.DistanceGraph

open scoped Topology NNReal

/-- A Lipschitz outer map bounds the derivative of a curve composition wherever
both derivatives exist. The outer map itself need not be differentiable. -/
theorem norm_deriv_comp_lipschitz_le {E F : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E]
    [NormedAddCommGroup F] [NormedSpace ℝ F]
    (f : E → F) (gamma : ℝ → E) (C : ℝ≥0) (hf : LipschitzWith C f)
    (t : ℝ) (v : E) (w : F) (hgamma : HasDerivAt gamma v t)
    (hcomp : HasDerivAt (fun s => f (gamma s)) w t) :
    ‖w‖ ≤ (C : ℝ) * ‖v‖ := by
  apply le_of_tendsto_of_tendsto hcomp.tendsto_slope_zero.norm
    (hgamma.tendsto_slope_zero.norm.const_mul (C : ℝ))
  apply Filter.Eventually.of_forall
  intro s
  simp only [norm_smul]
  calc
    ‖s⁻¹‖ * ‖f (gamma (t + s)) - f (gamma t)‖ ≤
        ‖s⁻¹‖ * ((C : ℝ) * ‖gamma (t + s) - gamma t‖) :=
      mul_le_mul_of_nonneg_left (hf.norm_sub_le _ _) (norm_nonneg _)
    _ = (C : ℝ) * (‖s⁻¹‖ * ‖gamma (t + s) - gamma t‖) := by ring

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The radial speed bound uses differentiability of the two traces only,
including cases where the distance field has no ambient derivative. -/
theorem infDist_deriv_bound (S : Set E) (gamma : ℝ → E) (t w : ℝ) (v : E)
    (hgamma : HasDerivAt gamma v t)
    (hradial : HasDerivAt (fun s => Metric.infDist (gamma s) S) w t) :
    |w| ≤ ‖v‖ := by
  simpa using norm_deriv_comp_lipschitz_le (fun x => Metric.infDist x S)
    gamma 1 (Metric.lipschitz_infDist_pt S) t v w hgamma hradial

/-- On every segment joining a nearest point to `x`, distance to the set grows
exactly linearly. This holds in any real normed space once a nearest point is given. -/
theorem infDist_nearest_segment (S : Set E) (x p : E)
    (hp : p ∈ S) (hnearest : Metric.infDist x S = dist x p)
    (t : ℝ) (ht : t ∈ Set.Icc (0 : ℝ) 1) :
    Metric.infDist (AffineMap.lineMap p x t) S = t * Metric.infDist x S := by
  have hleft : dist (AffineMap.lineMap p x t) p = t * Metric.infDist x S := by
    rw [dist_lineMap_left, Real.norm_eq_abs, abs_of_nonneg ht.1,
      dist_comm p x, ← hnearest]
  have hright : dist x (AffineMap.lineMap p x t) = (1 - t) * Metric.infDist x S := by
    rw [dist_right_lineMap, Real.norm_eq_abs,
      abs_of_nonneg (sub_nonneg.mpr ht.2), dist_comm p x, ← hnearest]
  have hu := Metric.infDist_le_dist_of_mem (x := AffineMap.lineMap p x t) hp
  have hl := Metric.infDist_le_infDist_add_dist (x := x)
    (y := AffineMap.lineMap p x t) (s := S)
  rw [hleft] at hu
  rw [hright] at hl
  linarith

/-- The distance trace on a nearest-point segment has the expected derivative
at every interior point. Endpoint differentiability is deliberately not claimed. -/
theorem hasDerivAt_infDist_nearest_segment (S : Set E) (x p : E)
    (hp : p ∈ S) (hnearest : Metric.infDist x S = dist x p)
    (t : ℝ) (ht : t ∈ Set.Ioo (0 : ℝ) 1) :
    HasDerivAt (fun s => Metric.infDist (AffineMap.lineMap p x s) S)
      (Metric.infDist x S) t := by
  have hlinear : HasDerivAt (fun s : ℝ => s * Metric.infDist x S)
      (Metric.infDist x S) t := by
    simpa using (hasDerivAt_id t).mul_const (Metric.infDist x S)
  apply hlinear.congr_of_eventuallyEq
  filter_upwards [Ioo_mem_nhds ht.1 ht.2] with s hs
  exact infDist_nearest_segment S x p hp hnearest s ⟨hs.1.le, hs.2.le⟩

theorem hasDerivAt_nearest_segment (x p : E) (t : ℝ) :
    HasDerivAt (fun s : ℝ => AffineMap.lineMap p x s) (x - p) t := by
  simpa only [AffineMap.lineMap_apply, vsub_eq_sub, vadd_eq_add, one_smul] using
    ((hasDerivAt_id t).smul_const (x - p)).add_const p

/-- Closed nonempty sets in proper real normed spaces admit an entire calibrated
nearest-point segment. Finite-dimensional Euclidean spaces are instances. -/
theorem exists_nearest_segment [ProperSpace E] (S : Set E)
    (hclosed : IsClosed S) (hne : S.Nonempty) (x : E) :
    ∃ p ∈ S, Metric.infDist x S = dist x p ∧
      ∀ t ∈ Set.Icc (0 : ℝ) 1,
        Metric.infDist (AffineMap.lineMap p x t) S = t * Metric.infDist x S := by
  obtain ⟨p, hp, hnearest⟩ := hclosed.exists_infDist_eq_dist hne x
  exact ⟨p, hp, hnearest, fun t ht => infDist_nearest_segment S x p hp hnearest t ht⟩

end FractalSync.DistanceGraph
