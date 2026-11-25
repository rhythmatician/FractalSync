import FractalSync.IntrinsicLength
import FractalSync.Analysis.ProperGeodesics

/-!
# Constant-speed metric geodesics for the actual AC path distance

Cumulative graph length supplies bounded-speed reparameterized paths.
Properness then supplies an exact metric geodesic by compactness, without
assuming that the original graph-length infimum is attained.
-/

namespace FractalSync.DistanceGraph

open Set
open scoped Topology

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [FiniteDimensional ℝ E]

/-- Every AC graph path gives an endpoint path with intrinsic Lipschitz
constant at most its actual graph length. -/
theorem exists_intrinsic_boundedSpeedPath (S : Set E) (epsilon a : ℝ)
    (gamma : ℝ → E) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) :
    ∃ f : unitInterval → E, f 0 = gamma 0 ∧ f 1 = gamma 1 ∧
      ∀ s t, intrinsicDistance S epsilon a (f s) (f t) ≤
        acGraphLength S epsilon a gamma * dist s t := by
  refine @MetricGeometry.exists_boundedSpeedPath_of_control E
    (intrinsicMetricSpace S epsilon a he) (fun t => gamma t)
    (fun t => acCumulativeLength S epsilon a gamma t) (acGraphLength S epsilon a gamma)
    (acGraphLength_nonneg S epsilon a gamma) ?_
    (acCumulativeLength_zero S epsilon a gamma) (acCumulativeLength_one S epsilon a gamma) ?_
  case refine_1 =>
    exact (continuousOn_acCumulativeLength S epsilon a gamma he hgamma).comp_continuous
      continuous_subtype_val (fun t => by simpa only [uIcc_of_le zero_le_one] using t.2)
  exact fun s t => intrinsicDistance_le_abs_acCumulativeLength_sub S epsilon a gamma he hgamma
    s t (by simpa only [uIcc_of_le zero_le_one] using s.2)
      (by simpa only [uIcc_of_le zero_le_one] using t.2)

/-- The actual positive-floor intrinsic distance admits a constant-speed
metric geodesic between every pair, including coincident endpoints. -/
theorem exists_intrinsic_metricGeodesic (S : Set E) (epsilon a : ℝ) (x y : E)
    (he : 0 < epsilon) :
    ∃ f : unitInterval → E, f 0 = x ∧ f 1 = y ∧
      ∀ s t, intrinsicDistance S epsilon a (f s) (f t) =
        intrinsicDistance S epsilon a x y * dist s t := by
  refine @MetricGeometry.exists_metricGeodesic E (intrinsicMetricSpace S epsilon a he)
    (intrinsicMetricSpace_proper S epsilon a he) x y ?_
  intro delta hdelta
  obtain ⟨gamma, hgamma, hstart, hend, hlength⟩ :=
    exists_path_length_lt_intrinsicDistance_add S epsilon a x y delta hdelta
  obtain ⟨f, hf0, hf1, hfspeed⟩ := exists_intrinsic_boundedSpeedPath S epsilon a gamma he hgamma
  exact ⟨f, hf0.trans hstart, hf1.trans hend,
    fun s t => (hfspeed s t).trans (mul_le_mul_of_nonneg_right hlength.le dist_nonneg)⟩

/-- An elementary exponential estimate used to transfer intrinsic speed to
ordinary speed on bounded intrinsic paths. -/
theorem exp_sub_one_le_mul_exp (q : ℝ) : Real.exp q - 1 ≤ q * Real.exp q := by
  have h := mul_le_mul_of_nonneg_left (Real.one_sub_le_exp_neg q) (Real.exp_pos q).le
  rw [← Real.exp_add, add_neg_cancel, Real.exp_zero] at h
  nlinarith

/-- Intrinsic bounded speed on the unit interval gives an explicit ordinary
Lipschitz bound. This estimate is used only for AC admissibility. -/
theorem lipschitz_of_intrinsic_boundedSpeed (S : Set E) (epsilon a : ℝ)
    (f : unitInterval → E) (K : ℝ) (he : 0 < epsilon) (hK : 0 ≤ K)
    (hf : ∀ s t, intrinsicDistance S epsilon a (f s) (f t) ≤ K * dist s t) :
    LipschitzWith ⟨ruler epsilon (Metric.infDist (f 0) S) * (Real.exp K) ^ 2 * K,
      mul_nonneg (mul_nonneg (ruler_pos epsilon _ he).le (sq_nonneg _)) hK⟩ f := by
  have hdiam : ∀ s t : unitInterval, dist s t ≤ 1 := by
    intro s t
    simpa only [Subtype.dist_eq, sub_zero] using Real.dist_le_of_mem_Icc s.2 t.2
  have hD : ∀ s t, intrinsicDistance S epsilon a (f s) (f t) ≤ K :=
    fun s t => (hf s t).trans (mul_le_of_le_one_right hK (hdiam s t))
  have hrho : ∀ s, ruler epsilon (Metric.infDist (f s) S) ≤
      ruler epsilon (Metric.infDist (f 0) S) * Real.exp K :=
    fun s => (ruler_le_exp_intrinsicDistance S epsilon a (f 0) (f s) he).trans
      (mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr (hD 0 s)) (ruler_pos epsilon _ he).le)
  refine LipschitzWith.of_dist_le_mul fun s t => ?_
  have hb := (dist_le_ruler_mul_exp_intrinsicDistance S epsilon a (f s) (f t) he).trans
    (mul_le_mul_of_nonneg_left (exp_sub_one_le_mul_exp _) (ruler_pos epsilon _ he).le)
  have hc := mul_le_mul (hrho s)
    (mul_le_mul (hf s t) (Real.exp_le_exp.mpr (hD s t)) (Real.exp_pos _).le
      (mul_nonneg hK dist_nonneg))
    (mul_nonneg (intrinsicDistance_nonneg S epsilon a (f s) (f t)) (Real.exp_pos _).le)
    (mul_nonneg (ruler_pos epsilon _ he).le (Real.exp_pos K).le)
  simpa only [NNReal.coe_mk, pow_two, mul_assoc, mul_left_comm, mul_comm] using hb.trans hc

/-- A metric geodesic has a real-parameter representative that is absolutely
continuous in the original normed space and preserves exact intrinsic distance
on every unit-interval subpath. -/
theorem exists_intrinsic_ac_metricGeodesic (S : Set E) (epsilon a : ℝ) (x y : E)
    (he : 0 < epsilon) :
    ∃ gamma : ℝ → E, AbsolutelyContinuousOnInterval gamma 0 1 ∧
      gamma 0 = x ∧ gamma 1 = y ∧
      ∀ s ∈ uIcc (0 : ℝ) 1, ∀ t ∈ uIcc (0 : ℝ) 1,
        intrinsicDistance S epsilon a (gamma s) (gamma t) =
          intrinsicDistance S epsilon a x y * |s - t| := by
  obtain ⟨f, hf0, hf1, hf⟩ := exists_intrinsic_metricGeodesic S epsilon a x y he
  have hLip := lipschitz_of_intrinsic_boundedSpeed S epsilon a f
    (intrinsicDistance S epsilon a x y) he (intrinsicDistance_nonneg S epsilon a x y)
    (fun s t => (hf s t).le)
  refine ⟨f ∘ projIcc (0 : ℝ) 1 zero_le_one,
    (hLip.comp (LipschitzWith.projIcc zero_le_one)).lipschitzOnWith.absolutelyContinuousOnInterval,
    ?_, ?_, ?_⟩
  case refine_3 =>
    intro s hs t ht
    simpa only [Function.comp_apply, Subtype.dist_eq,
      projIcc_of_mem zero_le_one (show s ∈ Icc (0 : ℝ) 1 by
        simpa only [uIcc_of_le zero_le_one] using hs),
      projIcc_of_mem zero_le_one (show t ∈ Icc (0 : ℝ) 1 by
        simpa only [uIcc_of_le zero_le_one] using ht), Real.dist_eq] using
      hf (projIcc (0 : ℝ) 1 zero_le_one s) (projIcc (0 : ℝ) 1 zero_le_one t)
  case refine_1 =>
    simpa only [Function.comp_apply, projIcc_left] using hf0
  simpa only [Function.comp_apply, projIcc_right] using hf1

end FractalSync.DistanceGraph
