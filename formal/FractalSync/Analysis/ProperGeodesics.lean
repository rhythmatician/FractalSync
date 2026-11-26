import Mathlib.Topology.MetricSpace.ProperSpace
import Mathlib.Topology.UnitInterval
import Mathlib.Topology.Instances.ENNReal.Lemmas
import Mathlib.Topology.Order.IntermediateValue

/-!
# Metric geodesics from nearly optimal Lipschitz paths

Compactness of a pointwise product of closed metric balls turns arbitrarily
near-optimal Lipschitz endpoint paths into an exact constant-speed geodesic.
The construction does not assume any minimizing path.
-/

namespace FractalSync.MetricGeometry

open Set Filter
open scoped Topology

variable {X : Type*} [MetricSpace X]

/-- Unit-interval paths with prescribed endpoints and a real Lipschitz bound. -/
def boundedSpeedPaths (x y : X) (K : ℝ) : Set (unitInterval → X) :=
  {f | f 0 = x ∧ f 1 = y ∧ ∀ s t, dist (f s) (f t) ≤ K * dist s t}

/-- The endpoint and Lipschitz conditions are closed for pointwise convergence. -/
theorem isClosed_boundedSpeedPaths (x y : X) (K : ℝ) :
    IsClosed (boundedSpeedPaths x y K) := by
  simp only [boundedSpeedPaths, setOf_and, setOf_forall]
  exact (isClosed_eq (continuous_apply 0) continuous_const).inter
    ((isClosed_eq (continuous_apply 1) continuous_const).inter
      (isClosed_iInter fun s => isClosed_iInter fun t =>
        isClosed_le ((continuous_apply s).dist (continuous_apply t)) continuous_const))

/-- Properness makes each family of uniformly bounded-speed endpoint paths
compact in the pointwise topology. -/
theorem isCompact_boundedSpeedPaths [ProperSpace X] (x y : X) (K : ℝ) (hK : 0 ≤ K) :
    IsCompact (boundedSpeedPaths x y K) := by
  refine (isCompact_pi_infinite (fun _ : unitInterval => isCompact_closedBall x K)).of_isClosed_subset
    (isClosed_boundedSpeedPaths x y K) ?_
  rintro f ⟨hzero, _, hspeed⟩ t
  have hdist : dist t (0 : unitInterval) ≤ 1 := by
    simpa only [Subtype.dist_eq, Set.Icc.coe_zero, Real.dist_eq, sub_zero,
      abs_of_nonneg t.2.1] using t.2.2
  simpa only [Metric.mem_closedBall, hzero, mul_one] using
    (hspeed t 0).trans (mul_le_mul_of_nonneg_left hdist hK)

/-- Increasing the speed bound enlarges the path family. -/
theorem boundedSpeedPaths_mono (x y : X) {K L : ℝ} (hKL : K ≤ L) :
    boundedSpeedPaths x y K ⊆ boundedSpeedPaths x y L := by
  exact fun _ ⟨hzero, hone, hspeed⟩ => ⟨hzero, hone,
    fun s t => (hspeed s t).trans (mul_le_mul_of_nonneg_right hKL dist_nonneg)⟩

/-- A proper metric space contains a path with speed at most the endpoint
distance whenever it contains paths with every slightly larger speed bound. -/
theorem exists_optimal_boundedSpeedPath [ProperSpace X] (x y : X)
    (hpaths : ∀ delta > (0 : ℝ), (boundedSpeedPaths x y (dist x y + delta)).Nonempty) :
    (boundedSpeedPaths x y (dist x y)).Nonempty := by
  let J := {delta : ℝ // 0 < delta}
  haveI : Nonempty J := ⟨⟨1, zero_lt_one⟩⟩
  have hdir : Directed (· ⊇ ·) (fun delta : J => boundedSpeedPaths x y (dist x y + delta.1)) :=
    fun i j => ⟨⟨min i.1 j.1, lt_min i.2 j.2⟩,
      boundedSpeedPaths_mono x y (add_le_add_left (min_le_left _ _) _),
      boundedSpeedPaths_mono x y (add_le_add_left (min_le_right _ _) _)⟩
  obtain ⟨f, hf⟩ := IsCompact.nonempty_iInter_of_directed_nonempty_isCompact_isClosed
    (fun delta : J => boundedSpeedPaths x y (dist x y + delta.1)) hdir
    (fun delta => hpaths delta.1 delta.2)
    (fun delta => isCompact_boundedSpeedPaths x y _ (add_nonneg dist_nonneg delta.2.le))
    (fun delta => isClosed_boundedSpeedPaths x y _)
  have hf' : ∀ delta > (0 : ℝ), f ∈ boundedSpeedPaths x y (dist x y + delta) :=
    fun delta hdelta => (mem_iInter.mp hf) ⟨delta, hdelta⟩
  refine ⟨f, (hf' 1 zero_lt_one).1, (hf' 1 zero_lt_one).2.1, fun s t => ?_⟩
  have hlim : Tendsto (fun delta : ℝ => (dist x y + delta) * dist s t)
      (𝓝[>] (0 : ℝ)) (𝓝 (dist x y * dist s t)) := by
    simpa only [id_eq, add_zero] using
      (((continuous_const.add continuous_id).mul continuous_const).tendsto (0 : ℝ)).mono_left
        nhdsWithin_le_nhds
  exact ge_of_tendsto hlim
    (Filter.Eventually.mono self_mem_nhdsWithin fun delta hdelta => (hf' delta hdelta).2.2 s t)

/-- A path whose speed bound equals its endpoint distance is a metric geodesic:
every subinterval realizes the corresponding fraction of the total distance. -/
theorem dist_eq_mul_of_optimal_boundedSpeedPath (x y : X) (f : unitInterval → X)
    (hf : f ∈ boundedSpeedPaths x y (dist x y)) (s t : unitInterval) :
    dist (f s) (f t) = dist x y * dist s t := by
  wlog hst : s ≤ t generalizing s t
  · simpa only [dist_comm] using this t s (le_of_not_ge hst)
  refine le_antisymm (hf.2.2 s t) ?_
  have hb := (dist_triangle4 (f 0) (f s) (f t) (f 1)).trans
    (add_le_add (add_le_add_right (hf.2.2 0 s) _) (hf.2.2 t 1))
  simp only [hf.1, hf.2.1, Subtype.dist_eq, Set.Icc.coe_zero, Set.Icc.coe_one,
    Real.dist_eq, zero_sub, abs_neg, abs_of_nonneg s.2.1,
    abs_of_nonpos (sub_nonpos.mpr t.2.2),
    abs_of_nonpos (sub_nonpos.mpr (show (s : ℝ) ≤ t from hst))] at hb ⊢
  nlinarith

/-- Nearly optimal bounded-speed paths produce a genuine metric geodesic. -/
theorem exists_metricGeodesic [ProperSpace X] (x y : X)
    (hpaths : ∀ delta > (0 : ℝ), (boundedSpeedPaths x y (dist x y + delta)).Nonempty) :
    ∃ f : unitInterval → X, f 0 = x ∧ f 1 = y ∧
      ∀ s t, dist (f s) (f t) = dist x y * dist s t := by
  exact (exists_optimal_boundedSpeedPath x y hpaths).imp fun f hf =>
    ⟨hf.1, hf.2.1, dist_eq_mul_of_optimal_boundedSpeedPath x y f hf⟩

/-- A continuous scalar control with full endpoint range reparameterizes a
path to its total control as a Lipschitz bound. Constant stretches of the
control cause no difficulty because the metric bound identifies their points. -/
theorem exists_boundedSpeedPath_of_control (gamma : unitInterval → X)
    (A : unitInterval → ℝ) (L : ℝ) (hL : 0 ≤ L) (hA : Continuous A)
    (hA0 : A 0 = 0) (hA1 : A 1 = L)
    (hcontrol : ∀ s t, dist (gamma s) (gamma t) ≤ |A s - A t|) :
    (boundedSpeedPaths (gamma 0) (gamma 1) L).Nonempty := by
  have hlevels : ∀ t : unitInterval, ∃ s : unitInterval, A s = L * (t : ℝ) := by
    intro t
    obtain ⟨s, _, hs⟩ := intermediate_value_Icc (show (0 : unitInterval) ≤ 1 from zero_le_one)
      hA.continuousOn (show L * (t : ℝ) ∈ Icc (A 0) (A 1) by
        simpa only [hA0, hA1] using
          And.intro (mul_nonneg hL t.2.1) (mul_le_of_le_one_right hL t.2.2))
    exact ⟨s, hs⟩
  choose sigma hsigma using hlevels
  refine ⟨gamma ∘ sigma, ?_, ?_, fun s t => ?_⟩
  case refine_3 =>
    simpa only [Function.comp_apply, hsigma, ← mul_sub, abs_mul, abs_of_nonneg hL,
      Subtype.dist_eq, Real.dist_eq] using hcontrol (sigma s) (sigma t)
  case refine_1 =>
    exact dist_eq_zero.mp (le_antisymm (by
      simpa only [Function.comp_apply, hsigma, Set.Icc.coe_zero, mul_zero, hA0,
        sub_zero, abs_zero] using hcontrol (sigma 0) 0) dist_nonneg)
  exact dist_eq_zero.mp (le_antisymm (by
    simpa only [Function.comp_apply, hsigma, Set.Icc.coe_one, mul_one, hA1,
      sub_self, abs_zero] using hcontrol (sigma 1) 1) dist_nonneg)

end FractalSync.MetricGeometry
