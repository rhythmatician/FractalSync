import FractalSync.NormalizedDistanceFields
import Mathlib.Topology.MetricSpace.ProperSpace
import Mathlib.Topology.MetricSpace.HausdorffDistance

/-!
# Distance-function limits and local affine reconstruction

An anchored limit of unsigned distance functions retains its zero set.
For a different, smooth regime, a finite stencil estimates the affine
part of an anchored normalized field from certified Taylor and sample errors.
-/

namespace FractalSync.NormalizedMapLimits

open Filter
open scoped Topology

/-- Pointwise convergence suffices here. Properness prevents nearest points
from escaping, since every member of the family contains a fixed anchor. -/
theorem pointwise_infDist_limit {E : Type*} [MetricSpace E] [ProperSpace E]
    (S : ℕ → Set E) (anchor : E) (hclosed : ∀ n, IsClosed (S n))
    (hanchor : ∀ n, anchor ∈ S n) (f : E → ℝ)
    (hf : ∀ x, Tendsto (fun n => Metric.infDist x (S n)) atTop (𝓝 (f x))) (x : E) :
    f x = Metric.infDist x {y | f y = 0} := by
  classical
  choose p hp hdist using fun n => (hclosed n).exists_infDist_eq_dist ⟨anchor, hanchor n⟩ x
  have hpBall (n : ℕ) : p n ∈ Metric.closedBall anchor (2 * dist x anchor) := by
    have hx := Metric.infDist_le_dist_of_mem (x := x) (hanchor n)
    have htri := dist_triangle (p n) x anchor
    rw [dist_comm (p n) x, ← hdist n] at htri
    exact Metric.mem_closedBall.mpr (by linarith)
  obtain ⟨y, _, subseq, hsub, hsubLim⟩ :=
    (isCompact_closedBall anchor (2 * dist x anchor)).tendsto_subseq hpBall
  have hfxDist : f x = dist x y := by
    apply tendsto_nhds_unique ((hf x).comp hsub.tendsto_atTop)
    simpa only [Function.comp_apply, hdist] using tendsto_const_nhds.dist hsubLim
  have hyZero : f y = 0 := by
    have hleft : Tendsto (fun n => Metric.infDist y (S (subseq n))) atTop (𝓝 (f y)) :=
      (hf y).comp hsub.tendsto_atTop
    have hright : Tendsto (fun n => dist y (p (subseq n))) atTop (𝓝 0) := by
      simpa using (tendsto_const_nhds.dist hsubLim :
        Tendsto (fun n => dist y (p (subseq n))) atTop (𝓝 (dist y y)))
    apply le_antisymm
    · exact le_of_tendsto_of_tendsto' hleft hright
        (fun n => Metric.infDist_le_dist_of_mem (hp (subseq n)))
    · exact ge_of_tendsto' (hf y) (fun _ => Metric.infDist_nonneg)
  refine le_antisymm ((Metric.le_infDist ⟨y, hyZero⟩).mpr ?_) ?_
  · intro z hz
    have hle : f x ≤ f z + dist x z :=
      le_of_tendsto_of_tendsto' (hf x) ((hf z).add tendsto_const_nhds)
        (fun _ => Metric.infDist_le_infDist_add_dist)
    simpa [show f z = 0 from hz] using hle
  · rw [hfxDist]
    exact Metric.infDist_le_dist_of_mem (show y ∈ {z | f z = 0} from hyZero)

/-- If the two measured samples each agree with their linear target up
to `bound`, their symmetric slope has error at most `bound / step`. -/
theorem central_slope_error {step slope plus minus bound : ℝ} (hstep : 0 < step)
    (hp : |plus - step * slope| ≤ bound) (hm : |minus + step * slope| ≤ bound) :
    |(plus - minus) / (2 * step) - slope| ≤ bound / step := by
  rw [show (plus - minus) / (2 * step) - slope =
      ((plus - step * slope) - (minus + step * slope)) / (2 * step) by
    field_simp
    ring]
  rw [abs_div, abs_of_pos (mul_pos (by norm_num) hstep)]
  apply (div_le_iff₀ (mul_pos (by norm_num) hstep)).mpr
  calc
    _ ≤ |plus - step * slope| + |minus + step * slope| := by
      simpa only [sub_eq_add_neg, abs_neg] using
        abs_add_le (plus - step * slope) (-(minus + step * slope))
    _ ≤ bound + bound := add_le_add hp hm
    _ = _ := by field_simp; ring

/-- An actual Taylor remainder and coordinate-slope errors control the
anchored affine estimate at every supported query. -/
theorem reconstruct_affine {ι : Type*} [Fintype ι] (u g estimate error : ι → ℝ)
    (actual remainder : ℝ)
    (hTaylor : |actual - ∑ i, g i * u i| ≤ remainder)
    (hSlope : ∀ i, |g i - estimate i| ≤ error i) :
    |actual - ∑ i, estimate i * u i| ≤ remainder + ∑ i, error i * |u i| := by
  have hsplit : actual - ∑ i, estimate i * u i =
      (actual - ∑ i, g i * u i) + ∑ i, (g i - estimate i) * u i := by
    simp only [sub_mul, Finset.sum_sub_distrib]
    ring
  rw [hsplit]
  calc
    _ ≤ |actual - ∑ i, g i * u i| + |∑ i, (g i - estimate i) * u i| := abs_add_le _ _
    _ ≤ _ := add_le_add hTaylor <|
      (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum fun i _ => by
        rw [abs_mul]
        exact mul_le_mul_of_nonneg_right (hSlope i) (abs_nonneg _))

/-- A symmetric sample pair on each axis supplies the affine estimator. -/
theorem reconstruct_from_symmetric_samples {ι : Type*} [Fintype ι]
    (u g plus minus bound : ι → ℝ) {step : ℝ} (hstep : 0 < step)
    (actual remainder : ℝ)
    (hTaylor : |actual - ∑ i, g i * u i| ≤ remainder)
    (hp : ∀ i, |plus i - step * g i| ≤ bound i)
    (hm : ∀ i, |minus i + step * g i| ≤ bound i) :
    |actual - ∑ i, ((plus i - minus i) / (2 * step)) * u i| ≤
      remainder + ∑ i, (bound i / step) * |u i| := by
  apply reconstruct_affine u g _ _ actual remainder hTaylor
  intro i
  rw [abs_sub_comm]
  exact central_slope_error hstep (hp i) (hm i)

end FractalSync.NormalizedMapLimits
