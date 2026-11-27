import FractalSync.IntrinsicLength

/-!
# Displacement from both endpoint rulers

A continuous path must pass a point where the two endpoint ruler-plus-distance
bounds agree. Apply logarithmic calibration to the subpaths on either side.
-/

namespace FractalSync.DistanceGraph

open Set

noncomputable def twoEndedDisplacement (r s d : ℝ) : ℝ :=
  2 * Real.log ((d + r + s) / 2) - Real.log r - Real.log s

/-- The two-ended estimate dominates the one-ended estimate from either endpoint
by symmetry. -/
theorem log_oneEnded_le_twoEndedDisplacement (r s d : ℝ)
    (hr : 0 < r) (hs : 0 < s) (hd : 0 ≤ d) :
    Real.log (1 + d / r) ≤ twoEndedDisplacement r s d := by
  let m := (d + r + s) / 2
  have hm : 0 < m := by dsimp [m]; positivity
  have hdr : 0 < r + d := add_pos_of_pos_of_nonneg hr hd
  have hprod : (r + d) * s ≤ m * m := by
    dsimp [m]
    nlinarith only [sq_nonneg (d + r - s)]
  have hlog := Real.log_le_log (mul_pos hdr hs) hprod
  rw [Real.log_mul hdr.ne' hs.ne', Real.log_mul hm.ne' hm.ne'] at hlog
  have hratio : (r + d) / r = 1 + d / r := by rw [add_div, div_self hr.ne']
  rw [← hratio, Real.log_div hdr.ne' hr.ne']
  change _ ≤ 2 * Real.log m - Real.log r - Real.log s
  linarith

theorem twoEndedDisplacement_swap (r s d : ℝ) :
    twoEndedDisplacement r s d = twoEndedDisplacement s r d := by
  unfold twoEndedDisplacement
  rw [show d + r + s = d + s + r by ring]
  ring

/-- Triangle law for the two-ended estimate. The shared ruler may grow
by at most the displacement from either neighboring endpoint. -/
theorem twoEndedDisplacement_triangle (r s u d e f : ℝ)
    (hr : 0 < r) (hs : 0 < s) (hu : 0 < u)
    (hd : 0 ≤ d) (he : 0 ≤ e) (hf : 0 ≤ f)
    (hds : s ≤ r + d) (hes : s ≤ u + e) (htri : f ≤ d + e) :
    twoEndedDisplacement r u f ≤
      twoEndedDisplacement r s d + twoEndedDisplacement s u e := by
  let m := (d + r + s) / 2
  let n := (e + s + u) / 2
  let p := (f + r + u) / 2
  have hm : 0 < m := by dsimp [m]; positivity
  have hn : 0 < n := by dsimp [n]; positivity
  have hp : 0 < p := by dsimp [p]; positivity
  have hprod : s * p ≤ m * n := by
    dsimp [p, m, n]
    nlinarith only [mul_nonneg (sub_nonneg.mpr hds) (sub_nonneg.mpr hes),
      mul_le_mul_of_nonneg_left htri hs.le]
  have hlog := Real.log_le_log (mul_pos hs hp) hprod
  rw [Real.log_mul hs.ne' hp.ne', Real.log_mul hm.ne' hn.ne'] at hlog
  change 2 * Real.log p - Real.log r - Real.log u ≤
    (2 * Real.log m - Real.log r - Real.log s) + (2 * Real.log n - Real.log s - Real.log u)
  linarith

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [FiniteDimensional ℝ E]

private theorem log_displacement_with_upper_ruler (S : Set E) (epsilon a : ℝ)
    (x y : E) (r : ℝ) (he : 0 < epsilon)
    (hr : ruler epsilon (Metric.infDist x S) ≤ r) :
    Real.log (1 + dist x y / r) ≤ intrinsicDistance S epsilon a x y := by
  have hpos : 0 < r := (ruler_pos epsilon _ he).trans_le hr
  exact (Real.log_le_log (by positivity)
    (add_le_add_left (div_le_div_of_nonneg_left dist_nonneg (ruler_pos epsilon _ he) hr) 1)).trans
      (log_displacement_le_intrinsicDistance S epsilon a x y he)

/-- Compatible upper rulers and lower displacement still give a lower bound.
Compatibility guarantees the balancing point even with interval uncertainty. -/
theorem twoEndedDisplacement_le_acGraphLength_of_bounds (S : Set E) (epsilon a : ℝ)
    (gamma : ℝ → E) (r s d : ℝ) (he : 0 < epsilon) (hd₀ : 0 ≤ d)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1)
    (hrUpper : ruler epsilon (Metric.infDist (gamma 0) S) ≤ r)
    (hsUpper : ruler epsilon (Metric.infDist (gamma 1) S) ≤ s)
    (hdLower : d ≤ dist (gamma 0) (gamma 1))
    (hrs : r ≤ s + d) (hsr : s ≤ r + d) :
    twoEndedDisplacement r s d ≤ acGraphLength S epsilon a gamma := by
  let f (t : ℝ) := r + dist (gamma t) (gamma 0) - (s + dist (gamma t) (gamma 1))
  have hc (p : E) : ContinuousOn (fun t => dist (gamma t) p) (Icc (0 : ℝ) 1) := by
    simpa only [uIcc_of_le zero_le_one, Metric.infDist_singleton] using
      (ac_distance_trace {p} gamma 0 1 hgamma).continuousOn
  have hf : ContinuousOn f (Icc 0 1) :=
    (continuousOn_const.add (hc (gamma 0))).sub
      (continuousOn_const.add (hc (gamma 1)))
  have hleft : r ≤ s + dist (gamma 0) (gamma 1) := hrs.trans (add_le_add_left hdLower s)
  have hright : s ≤ r + dist (gamma 1) (gamma 0) := by
    rw [dist_comm]
    exact hsr.trans (add_le_add_left hdLower r)
  have hzero : (0 : ℝ) ∈ Icc (f 0) (f 1) := by
    constructor <;> dsimp [f] <;> simp only [dist_self, add_zero] <;> linarith
  obtain ⟨t, ht, hbalance⟩ := intermediate_value_Icc zero_le_one hf hzero
  have ht' : t ∈ uIcc (0 : ℝ) 1 := by rwa [uIcc_of_le zero_le_one]
  have hp := intrinsicDistance_le_acCumulativeLength_sub S epsilon a gamma he hgamma
    0 t left_mem_uIcc ht' ht.1
  have hq := intrinsicDistance_le_acCumulativeLength_sub S epsilon a gamma he hgamma
    t 1 ht' right_mem_uIcc ht.2
  rw [acCumulativeLength_zero, sub_zero] at hp
  rw [acCumulativeLength_one, intrinsicDistance_symm S epsilon a (gamma t) (gamma 1)] at hq
  have hp' := (log_displacement_with_upper_ruler S epsilon a (gamma 0) (gamma t)
    r he hrUpper).trans hp
  have hq' := (log_displacement_with_upper_ruler S epsilon a (gamma 1) (gamma t)
    s he hsUpper).trans hq
  let m := r + dist (gamma t) (gamma 0)
  have hm : m = s + dist (gamma t) (gamma 1) := sub_eq_zero.mp hbalance
  have hr : 0 < r := (ruler_pos epsilon _ he).trans_le hrUpper
  have hs : 0 < s := (ruler_pos epsilon _ he).trans_le hsUpper
  have hmpos : 0 < m := add_pos_of_pos_of_nonneg hr dist_nonneg
  have hup : Real.log m - Real.log r ≤ acCumulativeLength S epsilon a gamma t := by
    rw [← Real.log_div hmpos.ne' hr.ne']
    simpa only [m, add_div, div_self hr.ne', dist_comm (gamma t) (gamma 0)] using hp'
  have hdown : Real.log m - Real.log s ≤
      acGraphLength S epsilon a gamma - acCumulativeLength S epsilon a gamma t := by
    rw [← Real.log_div hmpos.ne' hs.ne', hm]
    simpa only [add_div, div_self hs.ne', dist_comm (gamma t) (gamma 1)] using hq'
  have hd : (d + r + s) / 2 ≤ m := by
    have htri : dist (gamma 0) (gamma 1) ≤
        dist (gamma t) (gamma 0) + dist (gamma t) (gamma 1) := by
      simpa only [dist_comm (gamma t) (gamma 0)] using
        dist_triangle (gamma 0) (gamma t) (gamma 1)
    dsimp [m] at hm ⊢
    linarith
  have hlog := Real.log_le_log (by positivity) hd
  unfold twoEndedDisplacement
  linarith

/-- The bound applies to the unrestricted path infimum without attainment. -/
theorem twoEndedDisplacement_le_intrinsicDistance_of_bounds (S : Set E) (epsilon a : ℝ)
    (x y : E) (r s d : ℝ) (he : 0 < epsilon) (hd₀ : 0 ≤ d)
    (hr : ruler epsilon (Metric.infDist x S) ≤ r)
    (hs : ruler epsilon (Metric.infDist y S) ≤ s)
    (hd : d ≤ dist x y) (hrs : r ≤ s + d) (hsr : s ≤ r + d) :
    twoEndedDisplacement r s d ≤ intrinsicDistance S epsilon a x y := by
  apply le_csInf (pairwisePathLengths_nonempty S epsilon a x y)
  rintro L ⟨gamma, hgamma, hstart, hend, rfl⟩
  exact twoEndedDisplacement_le_acGraphLength_of_bounds S epsilon a gamma r s d he hd₀ hgamma
    (by simpa only [hstart] using hr) (by simpa only [hend] using hs)
    (by simpa only [hstart, hend] using hd) hrs hsr

theorem twoEndedDisplacement_le_intrinsicDistance (S : Set E) (epsilon a : ℝ)
    (x y : E) (he : 0 < epsilon) :
    twoEndedDisplacement (ruler epsilon (Metric.infDist x S))
        (ruler epsilon (Metric.infDist y S)) (dist x y) ≤ intrinsicDistance S epsilon a x y := by
  apply twoEndedDisplacement_le_intrinsicDistance_of_bounds S epsilon a x y _ _ _ he
    dist_nonneg le_rfl le_rfl le_rfl
  · simpa only [Metric.infDist_singleton] using ruler_le_anchor_add_distance S epsilon y x he
  · simpa only [Metric.infDist_singleton, dist_comm y x] using
      ruler_le_anchor_add_distance S epsilon x y he

/-- A fixed goal yields consistency whenever the edge cost bounds the
actual pairwise path distance from above. -/
theorem twoEndedDisplacement_consistent (S : Set E) (epsilon a : ℝ)
    (x y z : E) (cost : ℝ) (he : 0 < epsilon) (hc : intrinsicDistance S epsilon a x y ≤ cost) :
    twoEndedDisplacement (ruler epsilon (Metric.infDist x S))
        (ruler epsilon (Metric.infDist z S)) (dist x z) ≤
      cost + twoEndedDisplacement (ruler epsilon (Metric.infDist y S))
        (ruler epsilon (Metric.infDist z S)) (dist y z) := by
  have hxy := ruler_le_anchor_add_distance S epsilon x y he
  have hzy := ruler_le_anchor_add_distance S epsilon z y he
  rw [Metric.infDist_singleton, dist_comm y x] at hxy
  rw [Metric.infDist_singleton] at hzy
  refine (twoEndedDisplacement_triangle _ _ _ _ _ _ (ruler_pos epsilon _ he)
    (ruler_pos epsilon _ he) (ruler_pos epsilon _ he)
    dist_nonneg dist_nonneg dist_nonneg hxy hzy (dist_triangle x y z)).trans ?_
  exact add_le_add_right ((twoEndedDisplacement_le_intrinsicDistance S epsilon a x y he).trans hc) _

end FractalSync.DistanceGraph
