import FractalSync.DisplacementBounds
import FractalSync.PairwiseDistance
import FractalSync.ACComponentPaths
import Mathlib.Topology.Order.Compact

/-!
# The zero-floor comparison on off-Shore path classes

The unregularized infimum is taken only over absolutely continuous curves
whose entire unit-interval image avoids the closed Shore. Compactness supplies
positive clearance and hence integrability. The quantitative comparison uses
an explicit finite reference path and does not assume attained infima.
-/

namespace FractalSync.DistanceGraph

open MeasureTheory Set Filter
open scoped Topology

/-- The scalar regularization comparison for the actual graph density. -/
theorem graphSpeed_zero_sandwich (epsilon a r speed w : ℝ) (hr : 0 < r) :
    r ^ 2 / (r ^ 2 + epsilon ^ 2) * graphSpeed a 0 r speed w ≤
      graphSpeed a epsilon r speed w ∧
    graphSpeed a epsilon r speed w ≤ graphSpeed a 0 r speed w := by
  let t := r ^ 2 / (r ^ 2 + epsilon ^ 2)
  let A := speed ^ 2 / r ^ 2
  let C := a ^ 2 * w ^ 2 / r ^ 2
  have hden : 0 < r ^ 2 + epsilon ^ 2 := by positivity
  have ht0 : 0 ≤ t := by positivity
  have ht1 : t ≤ 1 := (div_le_one hden).2 (by nlinarith [sq_nonneg epsilon])
  have hg0 : graphSpeedSq a 0 r speed w = A + C := by
    dsimp [graphSpeedSq, A, C]
    field_simp
    ring
  have hge : graphSpeedSq a epsilon r speed w = t * A + t ^ 2 * C := by
    dsimp [graphSpeedSq, t, A, C]
    field_simp
  have hs := regularization_sandwich t A C ht0 ht1 (by positivity) (by positivity)
  constructor
  · have hb := Real.sqrt_le_sqrt hs.1
    rw [Real.sqrt_mul (sq_nonneg t), Real.sqrt_sq ht0] at hb
    simpa only [graphSpeed, hg0, hge] using hb
  · exact Real.sqrt_le_sqrt (by simpa only [hg0, hge] using hs.2)

/-- Clearance makes the unregularized density uniformly comparable with a
positive-floor density, without a velocity or distance-slope premise. -/
theorem graphSpeed_zero_le_twice (delta a r speed w : ℝ)
    (hd : 0 < delta) (hr : delta ≤ r) :
    graphSpeed a 0 r speed w ≤ 2 * graphSpeed a delta r speed w := by
  have hrpos := hd.trans_le hr
  have hfac : (1 / 2 : ℝ) ≤ r ^ 2 / (r ^ 2 + delta ^ 2) := by
    apply (le_div_iff₀ (by positivity : 0 < r ^ 2 + delta ^ 2)).2
    nlinarith [(sq_le_sq₀ hd.le hrpos.le).2 hr]
  have h := (mul_le_mul_of_nonneg_right hfac (graphSpeed_nonneg a 0 r speed w)).trans
    (graphSpeed_zero_sandwich delta a r speed w hrpos).1
  linarith

/-- A compact AC path avoiding a nonempty closed Shore has positive clearance. -/
theorem ac_path_exists_positive_clearance {E : Type*} [PseudoMetricSpace E]
    (S : Set E) (hclosed : IsClosed S) (hne : S.Nonempty) (gamma : ℝ → E)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1)
    (havoid : ∀ t ∈ uIcc (0 : ℝ) 1, gamma t ∉ S) :
    ∃ delta > (0 : ℝ), ∀ t ∈ uIcc (0 : ℝ) 1, delta ≤ Metric.infDist (gamma t) S := by
  exact isCompact_Icc.exists_forall_le'
    (ac_distance_trace S gamma 0 1 hgamma).continuousOn
    (fun t ht => (hclosed.notMem_iff_infDist_pos hne).1 (havoid t ht))

/-- Positive clearance and AC alone imply integrability of the actual
zero-floor graph density. -/
theorem ac_graphSpeed_zero_intervalIntegrable_of_clearance {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] [FiniteDimensional ℝ E]
    (S : Set E) (gamma : ℝ → E) (a delta : ℝ) (hd : 0 < delta)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1)
    (hclear : ∀ t ∈ uIcc (0 : ℝ) 1, delta ≤ Metric.infDist (gamma t) S) :
    IntervalIntegrable (fun t => graphSpeed a 0 (Metric.infDist (gamma t) S)
      ‖deriv gamma t‖ (deriv (fun s => Metric.infDist (gamma s) S) t)) volume 0 1 := by
  apply ((ac_graphSpeed_intervalIntegrable S gamma delta a 0 1 hd hgamma).const_mul 2).mono_fun'
  · apply aestronglyMeasurable_graphSpeed
    · exact (ac_distance_trace S gamma 0 1 hgamma).continuousOn.intervalIntegrable.def'.aestronglyMeasurable
    · exact (aestronglyMeasurable_deriv gamma _).norm
    · exact aestronglyMeasurable_deriv _ _
  · rw [EventuallyLE, ae_restrict_iff' measurableSet_uIoc]
    filter_upwards with t ht
    rw [Real.norm_eq_abs, abs_of_nonneg (graphSpeed_nonneg _ _ _ _ _)]
    exact graphSpeed_zero_le_twice delta a _ _ _ hd (hclear t (uIoc_subset_uIcc ht))

/-- Every AC path avoiding a nonempty closed Shore has an integrable actual
unregularized length density. -/
theorem ac_graphSpeed_zero_intervalIntegrable {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] [FiniteDimensional ℝ E]
    (S : Set E) (hclosed : IsClosed S) (hne : S.Nonempty) (gamma : ℝ → E) (a : ℝ)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1)
    (havoid : ∀ t ∈ uIcc (0 : ℝ) 1, gamma t ∉ S) :
    IntervalIntegrable (fun t => graphSpeed a 0 (Metric.infDist (gamma t) S)
      ‖deriv gamma t‖ (deriv (fun s => Metric.infDist (gamma s) S) t)) volume 0 1 := by
  obtain ⟨delta, hd, hclear⟩ := ac_path_exists_positive_clearance S hclosed hne gamma hgamma havoid
  exact ac_graphSpeed_zero_intervalIntegrable_of_clearance S gamma a delta hd hgamma hclear

/-- The logarithm of the positive ruler trace is absolutely continuous. -/
theorem ac_log_ruler_trace {E : Type*} [PseudoMetricSpace E]
    (S : Set E) (gamma : ℝ → E) (epsilon lo hi : ℝ) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma lo hi) :
    AbsolutelyContinuousOnInterval
      (fun t => Real.log (ruler epsilon (Metric.infDist (gamma t) S))) lo hi := by
  apply ac_log_of_ge _ lo hi epsilon he (ac_ruler_trace S gamma epsilon lo hi he hgamma)
  exact fun t _ => epsilon_le_ruler epsilon _ he.le

/-- Actual a.e. logarithmic ruler speed is bounded by the graph speed. -/
theorem ac_log_ruler_deriv_bound {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] [FiniteDimensional ℝ E]
    (S : Set E) (gamma : ℝ → E) (epsilon a lo hi : ℝ) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma lo hi) :
    ∀ᵐ t, t ∈ uIcc lo hi →
      |deriv (fun s => Real.log (ruler epsilon (Metric.infDist (gamma s) S))) t| ≤
      graphSpeed a epsilon (Metric.infDist (gamma t) S)
        ‖deriv gamma t‖ (deriv (fun s => Metric.infDist (gamma s) S) t) := by
  filter_upwards [(ac_distance_trace S gamma lo hi hgamma).ae_differentiableAt,
    ac_distance_trace_deriv_bound S gamma lo hi hgamma] with t ht hb hmem
  have hpos := ruler_pos epsilon (Metric.infDist (gamma t) S) he
  have hdr : HasDerivAt (fun s => ruler epsilon (Metric.infDist (gamma s) S))
      (Metric.infDist (gamma t) S / ruler epsilon (Metric.infDist (gamma t) S) *
        deriv (fun s => Metric.infDist (gamma s) S) t) t := by
    simpa only [Function.comp_def] using
      (hasDerivAt_ruler epsilon (Metric.infDist (gamma t) S) he).comp t (ht hmem).hasDerivAt
  have hd := (hdr.log hpos.ne').deriv
  have hprod : |Metric.infDist (gamma t) S / ruler epsilon (Metric.infDist (gamma t) S) *
      deriv (fun s => Metric.infDist (gamma s) S) t| ≤ ‖deriv gamma t‖ := by
    rw [abs_mul, abs_div, abs_of_pos hpos]
    calc
      _ ≤ 1 * ‖deriv gamma t‖ := mul_le_mul
        ((div_le_one hpos).2 (abs_le_ruler epsilon _)) (hb hmem) (abs_nonneg _) (by norm_num)
      _ = _ := one_mul _
  rw [hd, abs_div, abs_of_pos hpos]
  exact (div_le_div_of_nonneg_right hprod hpos.le).trans
    (speed_div_ruler_le_graphSpeed epsilon a _ _ _ he)

/-- Integrated logarithmic ruler calibration on every AC subpath. -/
theorem ac_log_ruler_increment_bound {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] [FiniteDimensional ℝ E]
    (S : Set E) (gamma : ℝ → E) (epsilon a lo hi : ℝ)
    (he : 0 < epsilon) (horder : lo ≤ hi)
    (hgamma : AbsolutelyContinuousOnInterval gamma lo hi) :
    |Real.log (ruler epsilon (Metric.infDist (gamma hi) S)) -
      Real.log (ruler epsilon (Metric.infDist (gamma lo) S))| ≤
      ∫ t in lo..hi, graphSpeed a epsilon (Metric.infDist (gamma t) S)
        ‖deriv gamma t‖ (deriv (fun s => Metric.infDist (gamma s) S) t) := by
  rw [← (ac_log_ruler_trace S gamma epsilon lo hi he hgamma).integral_deriv_eq_sub]
  apply (intervalIntegral.abs_integral_le_integral_abs horder).trans
  apply intervalIntegral.integral_mono_ae_restrict horder
    (ac_log_ruler_trace S gamma epsilon lo hi he hgamma).intervalIntegrable_deriv.abs
    (ac_graphSpeed_intervalIntegrable S gamma epsilon a lo hi he hgamma)
  exact (ae_restrict_iff' measurableSet_Icc).2
    ((ac_log_ruler_deriv_bound S gamma epsilon a lo hi he hgamma).mono
      fun t ht hmem => ht (by simpa only [uIcc_of_le horder] using hmem))

/-- A bounded-length path cannot make its ruler arbitrarily small. -/
theorem ac_ruler_lower_of_length_le {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] [FiniteDimensional ℝ E]
    (S : Set E) (gamma : ℝ → E) (epsilon a B t : ℝ) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1)
    (hB : acGraphLength S epsilon a gamma ≤ B) (ht : t ∈ uIcc (0 : ℝ) 1) :
    Metric.infDist (gamma 0) S * Real.exp (-B) ≤
      ruler epsilon (Metric.infDist (gamma t) S) := by
  have ht' : t ∈ Icc (0 : ℝ) 1 := by
    simpa only [uIcc_of_le (by norm_num : (0 : ℝ) ≤ 1)] using ht
  have hsub : uIcc (0 : ℝ) t ⊆ uIcc (0 : ℝ) 1 := by
    simpa only [uIcc_of_le ht'.1, uIcc_of_le (by norm_num : (0 : ℝ) ≤ 1)] using
      (Icc_subset_Icc le_rfl ht'.2)
  have hb := (ac_log_ruler_increment_bound S gamma epsilon a 0 t he ht'.1
    (hgamma.mono hsub)).trans
      (intervalIntegral.integral_mono_interval le_rfl ht'.1 ht'.2
        (Filter.Eventually.of_forall fun s => graphSpeed_nonneg a epsilon
          (Metric.infDist (gamma s) S) ‖deriv gamma s‖
          (deriv (fun u => Metric.infDist (gamma u) S) s))
        (ac_graphSpeed_intervalIntegrable S gamma epsilon a 0 1 he hgamma))
  have hlog : Real.log (ruler epsilon (Metric.infDist (gamma 0) S)) - B ≤
      Real.log (ruler epsilon (Metric.infDist (gamma t) S)) := by
    have hn := neg_le_abs (Real.log (ruler epsilon (Metric.infDist (gamma t) S)) -
      Real.log (ruler epsilon (Metric.infDist (gamma 0) S)))
    change _ ≤ acGraphLength S epsilon a gamma at hb
    linarith
  have hexp := Real.exp_le_exp.mpr hlog
  rw [sub_eq_add_neg, Real.exp_add, Real.exp_log (ruler_pos epsilon _ he),
    Real.exp_log (ruler_pos epsilon _ he)] at hexp
  exact (mul_le_mul_of_nonneg_right
    ((le_abs_self _).trans (abs_le_ruler epsilon _)) (Real.exp_pos _).le).trans hexp

/-- A length-bounded path avoids Shore once the floor is below its certified
ruler confinement scale. -/
theorem ac_path_avoids_of_length_le {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] [FiniteDimensional ℝ E]
    (S : Set E) (gamma : ℝ → E) (epsilon a B : ℝ) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1)
    (hB : acGraphLength S epsilon a gamma ≤ B)
    (hsmall : epsilon < Metric.infDist (gamma 0) S * Real.exp (-B)) :
    ∀ t ∈ uIcc (0 : ℝ) 1, gamma t ∉ S := by
  intro t ht hshore
  have hb := ac_ruler_lower_of_length_le S gamma epsilon a B t he hgamma hB ht
  have hz : ruler epsilon 0 = epsilon := by simp [ruler, Real.sqrt_sq he.le]
  rw [Metric.infDist_zero_of_mem hshore, hz] at hb
  exact not_lt_of_ge hb hsmall

/-- A uniform lower bound on the ruler gives the quantitative density factor. -/
theorem graphSpeed_zero_mul_le_of_ruler_ge (epsilon a r speed w m : ℝ)
    (he : 0 < epsilon) (hr : 0 < r) (hm : 0 < m) (hbound : m ≤ ruler epsilon r) :
    (1 - epsilon ^ 2 / m ^ 2) * graphSpeed a 0 r speed w ≤
      graphSpeed a epsilon r speed w := by
  have hden : 0 < r ^ 2 + epsilon ^ 2 := by positivity
  have hmb : m ^ 2 ≤ r ^ 2 + epsilon ^ 2 := by
    simpa only [ruler_sq] using (sq_le_sq₀ hm.le (ruler_pos epsilon r he).le).2 hbound
  have hdiv := div_le_div_of_nonneg_left (sq_nonneg epsilon) (sq_pos_of_pos hm) hmb
  have ht : r ^ 2 / (r ^ 2 + epsilon ^ 2) =
      1 - epsilon ^ 2 / (r ^ 2 + epsilon ^ 2) := by field_simp; ring
  have hfac : 1 - epsilon ^ 2 / m ^ 2 ≤ r ^ 2 / (r ^ 2 + epsilon ^ 2) := by
    rw [ht]
    linarith
  exact (mul_le_mul_of_nonneg_right hfac (graphSpeed_nonneg a 0 r speed w)).trans
    (graphSpeed_zero_sandwich epsilon a r speed w hr).1

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Actual unregularized lengths of AC paths whose entire interval avoids Shore. -/
def zeroFloorPathLengths (S : Set E) (a : ℝ) (x y : E) : Set ℝ :=
  {L | ∃ gamma : ℝ → E, AbsolutelyContinuousOnInterval gamma 0 1 ∧
    gamma 0 = x ∧ gamma 1 = y ∧
    (∀ t ∈ uIcc (0 : ℝ) 1, gamma t ∉ S) ∧ acGraphLength S 0 a gamma = L}

/-- The zero-floor path infimum. Its geometric theorems require a nonempty
admissible class, as supplied by the same-component theorem below. -/
noncomputable def zeroFloorDistance (S : Set E) (a : ℝ) (x y : E) : ℝ :=
  sInf (zeroFloorPathLengths S a x y)

theorem zeroFloorPathLengths_bddBelow (S : Set E) (a : ℝ) (x y : E) :
    BddBelow (zeroFloorPathLengths S a x y) := by
  exact ⟨0, fun _ ⟨gamma, _, _, _, _, hL⟩ =>
    hL ▸ acGraphLength_nonneg S 0 a gamma⟩

theorem zeroFloorDistance_nonneg (S : Set E) (a : ℝ) (x y : E)
    (hpaths : (zeroFloorPathLengths S a x y).Nonempty) :
    0 ≤ zeroFloorDistance S a x y := by
  exact le_csInf hpaths (fun _ ⟨gamma, _, _, _, _, hL⟩ =>
    hL ▸ acGraphLength_nonneg S 0 a gamma)

theorem zeroFloorDistance_le_length (S : Set E) (a : ℝ) (x y : E) (gamma : ℝ → E)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) (hstart : gamma 0 = x)
    (hend : gamma 1 = y) (havoid : ∀ t ∈ uIcc (0 : ℝ) 1, gamma t ∉ S) :
    zeroFloorDistance S a x y ≤ acGraphLength S 0 a gamma := by
  exact csInf_le (zeroFloorPathLengths_bddBelow S a x y)
    ⟨gamma, hgamma, hstart, hend, havoid, rfl⟩

/-- Regularization decreases the actual length of every off-Shore AC path. -/
theorem acGraphLength_le_zero_floor [FiniteDimensional ℝ E]
    (S : Set E) (hclosed : IsClosed S) (hne : S.Nonempty)
    (epsilon a : ℝ) (gamma : ℝ → E) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1)
    (havoid : ∀ t ∈ uIcc (0 : ℝ) 1, gamma t ∉ S) :
    acGraphLength S epsilon a gamma ≤ acGraphLength S 0 a gamma := by
  apply intervalIntegral.integral_mono_on (by norm_num)
    (ac_graphSpeed_intervalIntegrable S gamma epsilon a 0 1 he hgamma)
    (ac_graphSpeed_zero_intervalIntegrable S hclosed hne gamma a hgamma havoid)
  intro t ht
  exact (graphSpeed_zero_sandwich epsilon a _ _ _
    ((hclosed.notMem_iff_infDist_pos hne).1
      (havoid t (by simpa only [uIcc_of_le (by norm_num : (0 : ℝ) ≤ 1)] using ht)))).2

/-- The regularized pairwise infimum is bounded by the actual zero-floor infimum. -/
theorem intrinsicDistance_le_zeroFloorDistance [FiniteDimensional ℝ E]
    (S : Set E) (hclosed : IsClosed S) (hne : S.Nonempty)
    (epsilon a : ℝ) (x y : E) (he : 0 < epsilon)
    (hpaths : (zeroFloorPathLengths S a x y).Nonempty) :
    intrinsicDistance S epsilon a x y ≤ zeroFloorDistance S a x y := by
  apply le_csInf hpaths
  intro L ⟨gamma, hgamma, hstart, hend, havoid, hL⟩
  rw [← hL]
  exact (intrinsicDistance_le_length S epsilon a x y gamma hgamma hstart hend).trans
    (acGraphLength_le_zero_floor S hclosed hne epsilon a gamma he hgamma havoid)

/-- Bounded regularized length gives a multiplicative lower comparison with
the actual unregularized length of that same path. -/
theorem zeroFloorLength_mul_le_of_length_le [FiniteDimensional ℝ E]
    (S : Set E) (hclosed : IsClosed S) (hne : S.Nonempty)
    (epsilon a B : ℝ) (gamma : ℝ → E) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1)
    (hB : acGraphLength S epsilon a gamma ≤ B)
    (hsmall : epsilon < Metric.infDist (gamma 0) S * Real.exp (-B)) :
    (1 - epsilon ^ 2 / (Metric.infDist (gamma 0) S * Real.exp (-B)) ^ 2) *
      acGraphLength S 0 a gamma ≤ acGraphLength S epsilon a gamma := by
  have havoid := ac_path_avoids_of_length_le S gamma epsilon a B he hgamma hB hsmall
  unfold acGraphLength
  rw [← intervalIntegral.integral_const_mul]
  apply intervalIntegral.integral_mono_on (by norm_num)
    ((ac_graphSpeed_zero_intervalIntegrable S hclosed hne gamma a hgamma havoid).const_mul _)
    (ac_graphSpeed_intervalIntegrable S gamma epsilon a 0 1 he hgamma)
  intro t ht
  have ht' : t ∈ uIcc (0 : ℝ) 1 := by
    simpa only [uIcc_of_le (by norm_num : (0 : ℝ) ≤ 1)] using ht
  exact graphSpeed_zero_mul_le_of_ruler_ge epsilon a _ _ _ _ he
    ((hclosed.notMem_iff_infDist_pos hne).1 (havoid t ht')) (he.trans hsmall)
    (ac_ruler_lower_of_length_le S gamma epsilon a B t he hgamma hB ht')

/-- An actual AC reference path of zero-floor length at most `B` supplies a
quantitative comparison of the two path infima. The proof divides all
regularized candidates at length `B`; it does not assume a minimizing path. -/
theorem intrinsicDistance_zero_floor_sandwich_of_reference [FiniteDimensional ℝ E]
    (S : Set E) (hclosed : IsClosed S) (hne : S.Nonempty)
    (epsilon a B : ℝ) (x y : E) (he : 0 < epsilon)
    (hreference : ∃ L ∈ zeroFloorPathLengths S a x y, L ≤ B)
    (hsmall : epsilon < Metric.infDist x S * Real.exp (-B)) :
    (1 - epsilon ^ 2 / (Metric.infDist x S * Real.exp (-B)) ^ 2) *
        zeroFloorDistance S a x y ≤ intrinsicDistance S epsilon a x y ∧
      intrinsicDistance S epsilon a x y ≤ zeroFloorDistance S a x y := by
  obtain ⟨Lref, hLref, hrefB⟩ := hreference
  have hpaths : (zeroFloorPathLengths S a x y).Nonempty := ⟨Lref, hLref⟩
  have hD0B : zeroFloorDistance S a x y ≤ B :=
    (csInf_le (zeroFloorPathLengths_bddBelow S a x y) hLref).trans hrefB
  have hm : 0 < Metric.infDist x S * Real.exp (-B) := he.trans hsmall
  have hc0 : 0 ≤ 1 - epsilon ^ 2 / (Metric.infDist x S * Real.exp (-B)) ^ 2 := by
    apply sub_nonneg.mpr
    exact (div_le_one (sq_pos_of_pos hm)).2 ((sq_le_sq₀ he.le hm.le).2 hsmall.le)
  have hc1 : 1 - epsilon ^ 2 / (Metric.infDist x S * Real.exp (-B)) ^ 2 ≤ 1 :=
    sub_le_self _ (div_nonneg (sq_nonneg _) (sq_nonneg _))
  refine ⟨?_, intrinsicDistance_le_zeroFloorDistance S hclosed hne epsilon a x y he hpaths⟩
  apply le_csInf (pairwisePathLengths_nonempty S epsilon a x y)
  intro L ⟨gamma, hgamma, hstart, hend, hL⟩
  rw [← hL]
  by_cases hB : acGraphLength S epsilon a gamma ≤ B
  · have hsmall' : epsilon < Metric.infDist (gamma 0) S * Real.exp (-B) := by
      simpa only [hstart] using hsmall
    have havoid := ac_path_avoids_of_length_le S gamma epsilon a B he hgamma hB hsmall'
    have hcomp := zeroFloorLength_mul_le_of_length_le
      S hclosed hne epsilon a B gamma he hgamma hB hsmall'
    rw [hstart] at hcomp
    exact (mul_le_mul_of_nonneg_left
      (zeroFloorDistance_le_length S a x y gamma hgamma hstart hend havoid) hc0).trans hcomp
  · exact (mul_le_of_le_one_left (zeroFloorDistance_nonneg S a x y hpaths) hc1).trans
      (hD0B.trans (le_of_not_ge hB))

/-- The off-Shore AC length class is nonempty for points in one connected
component of the complement of a closed Shore. -/
theorem zeroFloorPathLengths_nonempty_of_mem_connectedComponentIn
    (S : Set E) (hclosed : IsClosed S) (a : ℝ) (x y : E) (hx : x ∉ S)
    (hy : y ∈ connectedComponentIn Sᶜ x) :
    (zeroFloorPathLengths S a x y).Nonempty := by
  obtain ⟨gamma, hgamma, hstart, hend, havoid⟩ :=
    exists_ac_path_avoiding_of_mem_connectedComponentIn S hclosed x y hx hy
  exact ⟨_, gamma, hgamma, hstart, hend, havoid, rfl⟩

/-- The actual positive-floor path infima converge to the actual off-Shore
zero-floor infimum whenever that admissible class is nonempty. -/
theorem intrinsicDistance_tendsto_zero_floor_of_path_nonempty [FiniteDimensional ℝ E]
    (S : Set E) (hclosed : IsClosed S) (hne : S.Nonempty) (a : ℝ) (x y : E)
    (hpaths : (zeroFloorPathLengths S a x y).Nonempty) :
    Tendsto (fun epsilon => intrinsicDistance S epsilon a x y)
      (𝓝[>] (0 : ℝ)) (𝓝 (zeroFloorDistance S a x y)) := by
  obtain ⟨B, hB⟩ := hpaths
  have hx : x ∉ S := by
    obtain ⟨gamma, _, hstart, _, havoid, _⟩ := hB
    simpa only [hstart] using havoid 0 left_mem_uIcc
  have hm : 0 < Metric.infDist x S * Real.exp (-B) :=
    mul_pos ((hclosed.notMem_iff_infDist_pos hne).1 hx) (Real.exp_pos _)
  have hlim : Tendsto
      (fun epsilon : ℝ =>
        (1 - epsilon ^ 2 / (Metric.infDist x S * Real.exp (-B)) ^ 2) *
          zeroFloorDistance S a x y)
      (𝓝[>] (0 : ℝ)) (𝓝 (zeroFloorDistance S a x y)) := by
    have hcont : Continuous (fun epsilon : ℝ =>
        (1 - epsilon ^ 2 / (Metric.infDist x S * Real.exp (-B)) ^ 2) *
          zeroFloorDistance S a x y) := by fun_prop
    simpa using (hcont.tendsto 0).mono_left nhdsWithin_le_nhds
  have hb : ∀ᶠ epsilon in 𝓝[>] (0 : ℝ),
      (1 - epsilon ^ 2 / (Metric.infDist x S * Real.exp (-B)) ^ 2) *
          zeroFloorDistance S a x y ≤ intrinsicDistance S epsilon a x y ∧
        intrinsicDistance S epsilon a x y ≤ zeroFloorDistance S a x y := by
    filter_upwards [self_mem_nhdsWithin,
      (eventually_lt_nhds hm).filter_mono nhdsWithin_le_nhds] with epsilon he hsmall
    exact intrinsicDistance_zero_floor_sandwich_of_reference S hclosed hne
      epsilon a B x y he ⟨B, hB, le_rfl⟩ hsmall
  exact tendsto_of_tendsto_of_tendsto_of_le_of_le' hlim tendsto_const_nhds
    (hb.mono fun _ h => h.1) (hb.mono fun _ h => h.2)

/-- The same-component branch of the pairwise zero-floor limit, using the
actual infimum over all AC paths and the actual off-Shore zero-floor infimum.
Connectedness supplies a reference path; no minimizing path is assumed. -/
theorem intrinsicDistance_tendsto_zero_floor_of_mem_connectedComponentIn
    [FiniteDimensional ℝ E] (S : Set E) (hclosed : IsClosed S) (hne : S.Nonempty)
    (x y : E) (hx : x ∉ S) (hy : y ∈ connectedComponentIn Sᶜ x) (a : ℝ) :
    Tendsto (fun epsilon => intrinsicDistance S epsilon a x y)
      (𝓝[>] (0 : ℝ)) (𝓝 (zeroFloorDistance S a x y)) := by
  exact intrinsicDistance_tendsto_zero_floor_of_path_nonempty S hclosed hne a x y
    (zeroFloorPathLengths_nonempty_of_mem_connectedComponentIn S hclosed a x y hx hy)

end FractalSync.DistanceGraph
