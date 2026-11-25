import FractalSync.DisplacementBounds
import FractalSync.IntrinsicMetric
import Mathlib.Topology.MetricSpace.ProperSpace

/-!
# Topology and compact balls of the positive-floor intrinsic metric

The logarithmic displacement lower bound controls intrinsic balls in the
original normed-space topology. The global upper bound and triangle inequality
identify that topology with the intrinsic metric topology. Uniformities are not
identified: the local ruler need not be bounded at infinity.
-/

namespace FractalSync.DistanceGraph

open Set
open scoped NNReal Topology

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [FiniteDimensional ℝ E]

/-- The actual path metric satisfies the reverse triangle inequality. -/
theorem intrinsicDistance_abs_sub_le (S : Set E) (epsilon a : ℝ)
    (x y z : E) (he : 0 < epsilon) :
    |intrinsicDistance S epsilon a x y - intrinsicDistance S epsilon a x z| ≤
      intrinsicDistance S epsilon a y z := by
  have h₁ := intrinsicDistance_triangle S epsilon a x z y he
  have h₂ := intrinsicDistance_triangle S epsilon a x y z he
  rw [intrinsicDistance_symm S epsilon a z y] at h₁
  exact abs_le.mpr ⟨by linarith, by linarith⟩

/-- Distance to a fixed point is globally Lipschitz in the original normed
space topology, with the same explicit upper comparison constant. -/
theorem intrinsicDistance_lipschitz_right (S : Set E) (epsilon a : ℝ)
    (x : E) (he : 0 < epsilon) :
    LipschitzWith ⟨Real.sqrt (1 + a ^ 2) / epsilon,
      div_nonneg (Real.sqrt_nonneg _) he.le⟩ (intrinsicDistance S epsilon a x) := by
  refine LipschitzWith.of_dist_le_mul (fun y z => ?_)
  simpa only [Real.dist_eq, NNReal.coe_mk] using
    (intrinsicDistance_abs_sub_le S epsilon a x y z he).trans
      (intrinsicDistance_le_global S epsilon a y z he)

/-- Intrinsic distance controls ordinary displacement by an explicit
anchor-dependent exponential modulus. -/
theorem dist_le_ruler_mul_exp_intrinsicDistance (S : Set E) (epsilon a : ℝ)
    (x y : E) (he : 0 < epsilon) :
    dist x y ≤ ruler epsilon (Metric.infDist x S) *
      (Real.exp (intrinsicDistance S epsilon a x y) - 1) := by
  have hr := ruler_pos epsilon (Metric.infDist x S) he
  have hb := (Real.log_le_iff_le_exp (show 0 < 1 +
    dist x y / ruler epsilon (Metric.infDist x S) by positivity)).mp
      (log_displacement_le_intrinsicDistance S epsilon a x y he)
  have hd : dist x y / ruler epsilon (Metric.infDist x S) ≤
      Real.exp (intrinsicDistance S epsilon a x y) - 1 := by linarith
  simpa only [mul_comm] using (div_le_iff₀ hr).mp hd

/-- A closed intrinsic sublevel ball has an explicit ordinary-radius bound.
The statement includes negative radii, for which the sublevel is empty. -/
theorem intrinsicSublevel_subset_closedBall (S : Set E) (epsilon a : ℝ)
    (x : E) (R : ℝ) (he : 0 < epsilon) :
    {y | intrinsicDistance S epsilon a x y ≤ R} ⊆
      Metric.closedBall x (ruler epsilon (Metric.infDist x S) * (Real.exp R - 1)) := by
  intro y hy
  rw [Metric.mem_closedBall, dist_comm]
  exact (dist_le_ruler_mul_exp_intrinsicDistance S epsilon a x y he).trans
    (mul_le_mul_of_nonneg_left (sub_le_sub_right (Real.exp_le_exp.mpr hy) 1)
      (ruler_pos epsilon _ he).le)

/-- Intrinsic closed balls are closed in the original normed-space topology. -/
theorem isClosed_intrinsicSublevel (S : Set E) (epsilon a : ℝ)
    (x : E) (R : ℝ) (he : 0 < epsilon) :
    IsClosed {y | intrinsicDistance S epsilon a x y ≤ R} := by
  exact isClosed_le (intrinsicDistance_lipschitz_right S epsilon a x he).continuous
    continuous_const

/-- Finite-dimensional closed intrinsic balls are compact in the original
topology, by the explicit displacement bound and ordinary properness. -/
theorem isCompact_intrinsicSublevel (S : Set E) (epsilon a : ℝ)
    (x : E) (R : ℝ) (he : 0 < epsilon) :
    IsCompact {y | intrinsicDistance S epsilon a x y ≤ R} := by
  exact (isCompact_closedBall x
    (ruler epsilon (Metric.infDist x S) * (Real.exp R - 1))).of_isClosed_subset
      (isClosed_intrinsicSublevel S epsilon a x R he)
      (intrinsicSublevel_subset_closedBall S epsilon a x R he)

/-- The intrinsic balls form exactly the original normed-space neighborhood
basis. This compares topologies, not uniformities at infinity. -/
theorem isOpen_iff_intrinsicBall (S : Set E) (epsilon a : ℝ) (U : Set E)
    (he : 0 < epsilon) :
    IsOpen U ↔ ∀ x ∈ U, ∃ R > 0, {y | intrinsicDistance S epsilon a x y < R} ⊆ U := by
  constructor
  · intro hU x hx
    obtain ⟨r, hr, hsub⟩ := Metric.isOpen_iff.mp hU x hx
    refine ⟨Real.log (1 + r / ruler epsilon (Metric.infDist x S)),
      Real.log_pos (lt_add_of_pos_right 1 (div_pos hr (ruler_pos epsilon _ he))), ?_⟩
    refine fun y hy => hsub ?_
    have hrho := ruler_pos epsilon (Metric.infDist x S) he
    have hd := (Real.log_lt_log_iff
      (show 0 < 1 + dist x y / ruler epsilon (Metric.infDist x S) by positivity)
      (show 0 < 1 + r / ruler epsilon (Metric.infDist x S) by positivity)).mp
        ((log_displacement_le_intrinsicDistance S epsilon a x y he).trans_lt hy)
    simpa only [Metric.mem_ball, dist_comm, add_lt_add_iff_left,
      div_lt_div_iff_of_pos_right hrho] using hd
  · intro hU
    refine Metric.isOpen_iff.mpr (fun x hx => ?_)
    obtain ⟨R, hR, hsub⟩ := hU x hx
    have hC : 0 < Real.sqrt (1 + a ^ 2) / epsilon :=
      div_pos (Real.sqrt_pos.mpr (by positivity)) he
    refine ⟨R / (Real.sqrt (1 + a ^ 2) / epsilon), div_pos hR hC, fun y hy => hsub ?_⟩
    refine (intrinsicDistance_le_global S epsilon a x y he).trans_lt ?_
    simpa only [dist_comm, mul_comm] using (lt_div_iff₀ hC).mp (Metric.mem_ball.mp hy)

/-- The explicit intrinsic metric induces the original topology. Its own
uniform structure is preserved rather than identified with the Euclidean one. -/
theorem intrinsicMetricSpace_topology (S : Set E) (epsilon a : ℝ) (he : 0 < epsilon) :
    (intrinsicMetricSpace S epsilon a he).toUniformSpace.toTopologicalSpace =
      (inferInstance : TopologicalSpace E) := by
  apply TopologicalSpace.ext
  funext U
  apply propext
  rw [@Metric.isOpen_iff E (intrinsicMetricSpace S epsilon a he).toPseudoMetricSpace U,
    isOpen_iff_intrinsicBall S epsilon a U he]
  simp only [Metric.ball, intrinsicMetricSpace_dist, intrinsicDistance_symm S epsilon a]

/-- The actual intrinsic metric is proper: all of its closed metric balls
are compact in its own topology. -/
theorem intrinsicMetricSpace_proper (S : Set E) (epsilon a : ℝ) (he : 0 < epsilon) :
    @ProperSpace E (intrinsicMetricSpace S epsilon a he).toPseudoMetricSpace := by
  refine @ProperSpace.mk E (intrinsicMetricSpace S epsilon a he).toPseudoMetricSpace
    (fun x R => ?_)
  change @IsCompact E (intrinsicMetricSpace S epsilon a he).toUniformSpace.toTopologicalSpace
    {y | intrinsicDistance S epsilon a y x ≤ R}
  rw [intrinsicMetricSpace_topology S epsilon a he]
  simpa only [intrinsicDistance_symm S epsilon a] using
    isCompact_intrinsicSublevel S epsilon a x R he

/-- Completeness is for the intrinsic uniform structure and follows from
the properness of the actual intrinsic metric. -/
theorem intrinsicMetricSpace_complete (S : Set E) (epsilon a : ℝ) (he : 0 < epsilon) :
    @CompleteSpace E (intrinsicMetricSpace S epsilon a he).toUniformSpace := by
  exact @complete_of_proper E
    (intrinsicMetricSpace S epsilon a he).toPseudoMetricSpace
    (intrinsicMetricSpace_proper S epsilon a he)

/-- The ruler grows at most exponentially in the actual intrinsic distance. -/
theorem ruler_le_exp_intrinsicDistance (S : Set E) (epsilon a : ℝ) (x y : E)
    (he : 0 < epsilon) :
    ruler epsilon (Metric.infDist y S) ≤
      ruler epsilon (Metric.infDist x S) * Real.exp (intrinsicDistance S epsilon a x y) := by
  have h := ruler_le_anchor_add_distance S epsilon x y he
  rw [Metric.infDist_singleton, dist_comm y x] at h
  nlinarith [dist_le_ruler_mul_exp_intrinsicDistance S epsilon a x y he]

end FractalSync.DistanceGraph
