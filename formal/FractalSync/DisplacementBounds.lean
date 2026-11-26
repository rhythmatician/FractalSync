import FractalSync.IntrinsicShoreDistance
import FractalSync.PairwiseDistance
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.SpecialFunctions.Log.Deriv

/-!
# Logarithmic displacement bounds for arbitrary AC graph paths

The positive ruler is one-Lipschitz. A logarithmic Euclidean-distance trace
therefore calibrates every graph path, without reparameterizing by arclength
or assuming an ambient derivative of the distance field.
-/

namespace FractalSync.DistanceGraph

open MeasureTheory Set Filter
open scoped Topology NNReal

/-- Logarithm is Lipschitz on every half-line bounded away from zero. -/
theorem log_lipschitzOn_Ici (c : ℝ) (hc : 0 < c) :
    LipschitzOnWith ⟨c⁻¹, inv_nonneg.mpr hc.le⟩ Real.log (Ici c) := by
  apply (convex_Ici c).lipschitzOnWith_of_nnnorm_deriv_le
  · exact fun x hx => Real.differentiableAt_log (hc.trans_le hx).ne'
  · intro x hx
    simpa only [← NNReal.coe_le_coe, coe_nnnorm, NNReal.coe_mk, Real.deriv_log,
      Real.norm_eq_abs, abs_of_pos (inv_pos.mpr (hc.trans_le hx))] using inv_anti₀ hc hx

/-- A positive lower bound suffices for the logarithm of an AC scalar trace. -/
theorem ac_log_of_ge (f : ℝ → ℝ) (lo hi c : ℝ) (hc : 0 < c)
    (hf : AbsolutelyContinuousOnInterval f lo hi)
    (hbound : ∀ t ∈ uIcc lo hi, c ≤ f t) :
    AbsolutelyContinuousOnInterval (fun t => Real.log (f t)) lo hi := by
  exact (log_lipschitzOn_Ici c hc).comp_absolutelyContinuousOnInterval hbound hf

variable {E : Type*} [NormedAddCommGroup E]

/-- The logarithmic displacement trace remains AC through returns to its anchor. -/
theorem ac_log_displacement_trace (gamma : ℝ → E) (p : E) (lo hi c : ℝ)
    (hc : 0 < c) (hgamma : AbsolutelyContinuousOnInterval gamma lo hi) :
    AbsolutelyContinuousOnInterval
      (fun t => Real.log (c + Metric.infDist (gamma t) {p})) lo hi := by
  apply ac_log_of_ge _ lo hi c hc
  · exact ((LipschitzWith.const c).lipschitzOnWith.absolutelyContinuousOnInterval).fun_add
      (ac_distance_trace {p} gamma lo hi hgamma)
  · exact fun _ _ => le_add_of_nonneg_right Metric.infDist_nonneg

/-- The graph speed dominates the scale-only speed, without a trace-slope premise. -/
theorem speed_div_ruler_le_graphSpeed (epsilon a r speed w : ℝ) (he : 0 < epsilon) :
    speed / ruler epsilon r ≤ graphSpeed a epsilon r speed w := by
  rw [graphSpeed_eq_rulerSpeed epsilon a r speed w he]
  exact div_le_div_of_nonneg_right
    (Real.le_sqrt_of_sq_le (le_add_of_nonneg_right
      (mul_nonneg (sq_nonneg a) (sq_nonneg _)))) (ruler_pos epsilon r he).le

/-- The one-Lipschitz ruler grows by at most Euclidean displacement from an anchor. -/
theorem ruler_le_anchor_add_distance (S : Set E) (epsilon : ℝ) (p x : E)
    (he : 0 < epsilon) :
    ruler epsilon (Metric.infDist x S) ≤
      ruler epsilon (Metric.infDist p S) + Metric.infDist x {p} := by
  have h := (regularizedDistance_lipschitz epsilon he S).dist_le_mul x p
  simpa only [Metric.infDist_singleton, add_comm] using
    (sub_le_iff_le_add.mp ((le_abs_self _).trans
      (by simpa only [Real.dist_eq, NNReal.coe_one, one_mul] using h)))

variable [NormedSpace ℝ E] [FiniteDimensional ℝ E]

/-- The actual derivative of the logarithmic displacement is bounded by graph
speed almost everywhere, including at nonsmooth Shore and distance-trace points. -/
theorem ac_log_displacement_deriv_le_graphSpeed (S : Set E) (gamma : ℝ → E)
    (p : E) (epsilon a lo hi : ℝ) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma lo hi) :
    ∀ᵐ t, t ∈ uIcc lo hi →
      deriv (fun s => Real.log (ruler epsilon (Metric.infDist p S) +
        Metric.infDist (gamma s) {p})) t ≤
      graphSpeed a epsilon (Metric.infDist (gamma t) S)
        ‖deriv gamma t‖ (deriv (fun s => Metric.infDist (gamma s) S) t) := by
  filter_upwards [(ac_distance_trace {p} gamma lo hi hgamma).ae_differentiableAt,
    ac_distance_trace_deriv_bound {p} gamma lo hi hgamma] with t ht hb hmem
  have hpos : 0 < ruler epsilon (Metric.infDist p S) + Metric.infDist (gamma t) {p} :=
    add_pos_of_pos_of_nonneg (ruler_pos epsilon _ he) Metric.infDist_nonneg
  rw [(((ht hmem).hasDerivAt.const_add (ruler epsilon (Metric.infDist p S))).log
    hpos.ne').deriv]
  exact (div_le_div_of_nonneg_right ((le_abs_self _).trans (hb hmem)) hpos.le).trans
    ((div_le_div_of_nonneg_left (norm_nonneg _) (ruler_pos epsilon _ he)
      (ruler_le_anchor_add_distance S epsilon p (gamma t) he)).trans
        (speed_div_ruler_le_graphSpeed epsilon a _ _ _ he))

/-- Integrated logarithmic calibration with an arbitrary fixed anchor. -/
theorem ac_log_distance_increment_bound (S : Set E) (gamma : ℝ → E) (p : E)
    (epsilon a lo hi : ℝ) (he : 0 < epsilon) (horder : lo ≤ hi)
    (hgamma : AbsolutelyContinuousOnInterval gamma lo hi) :
    Real.log (ruler epsilon (Metric.infDist p S) + Metric.infDist (gamma hi) {p}) -
      Real.log (ruler epsilon (Metric.infDist p S) + Metric.infDist (gamma lo) {p}) ≤
      ∫ t in lo..hi, graphSpeed a epsilon (Metric.infDist (gamma t) S)
        ‖deriv gamma t‖ (deriv (fun s => Metric.infDist (gamma s) S) t) := by
  rw [← (ac_log_displacement_trace gamma p lo hi _ (ruler_pos epsilon _ he)
    hgamma).integral_deriv_eq_sub]
  apply intervalIntegral.integral_mono_ae_restrict horder
    (ac_log_displacement_trace gamma p lo hi _ (ruler_pos epsilon _ he)
      hgamma).intervalIntegrable_deriv
    (ac_graphSpeed_intervalIntegrable S gamma epsilon a lo hi he hgamma)
  exact (ae_restrict_iff' measurableSet_Icc).2
    ((ac_log_displacement_deriv_le_graphSpeed S gamma p epsilon a lo hi he hgamma).mono
      fun t ht hmem => ht (by simpa only [uIcc_of_le horder] using hmem))

/-- Every AC path pays at least a logarithmic cost for its Euclidean displacement. -/
theorem ac_log_displacement_bound (S : Set E) (gamma : ℝ → E)
    (epsilon a lo hi : ℝ) (he : 0 < epsilon) (horder : lo ≤ hi)
    (hgamma : AbsolutelyContinuousOnInterval gamma lo hi) :
    Real.log (1 + dist (gamma hi) (gamma lo) /
      ruler epsilon (Metric.infDist (gamma lo) S)) ≤
      ∫ t in lo..hi, graphSpeed a epsilon (Metric.infDist (gamma t) S)
        ‖deriv gamma t‖ (deriv (fun s => Metric.infDist (gamma s) S) t) := by
  have hb := ac_log_distance_increment_bound S gamma (gamma lo) epsilon a lo hi
    he horder hgamma
  simpa only [Metric.infDist_singleton, dist_self, add_zero,
    ← Real.log_div (add_pos_of_pos_of_nonneg
      (ruler_pos epsilon (Metric.infDist (gamma lo) S) he) dist_nonneg).ne'
      (ruler_pos epsilon (Metric.infDist (gamma lo) S) he).ne',
    add_div, div_self (ruler_pos epsilon (Metric.infDist (gamma lo) S) he).ne'] using hb

/-- The actual pairwise path infimum dominates the logarithmic displacement cost. -/
theorem log_displacement_le_intrinsicDistance (S : Set E) (epsilon a : ℝ)
    (x y : E) (he : 0 < epsilon) :
    Real.log (1 + dist x y / ruler epsilon (Metric.infDist x S)) ≤
      intrinsicDistance S epsilon a x y := by
  apply le_csInf (pairwisePathLengths_nonempty S epsilon a x y)
  rintro L ⟨gamma, hgamma, hstart, hend, rfl⟩
  simpa only [hstart, hend, dist_comm, acGraphLength] using
    ac_log_displacement_bound S gamma epsilon a 0 1 he (by norm_num) hgamma

/-- Distinct endpoints have strictly positive intrinsic distance. -/
theorem intrinsicDistance_pos (S : Set E) (epsilon a : ℝ) (x y : E)
    (he : 0 < epsilon) (hxy : x ≠ y) :
    0 < intrinsicDistance S epsilon a x y := by
  exact (Real.log_pos (lt_add_of_pos_right 1
    (div_pos (dist_pos.mpr hxy) (ruler_pos epsilon _ he)))).trans_le
      (log_displacement_le_intrinsicDistance S epsilon a x y he)

/-- The actual positive-floor path infimum separates points. -/
theorem intrinsicDistance_eq_zero_iff (S : Set E) (epsilon a : ℝ) (x y : E)
    (he : 0 < epsilon) : intrinsicDistance S epsilon a x y = 0 ↔ x = y := by
  exact ⟨fun hz => not_ne_iff.mp
    (fun hxy => (ne_of_gt (intrinsicDistance_pos S epsilon a x y he hxy)) hz),
    fun hxy => by simpa only [hxy] using intrinsicDistance_self S epsilon a y⟩

/-- The logarithmic displacement cost diverges at every fixed positive displacement. -/
theorem log_displacement_tendsto_zero_floor (r : ℝ) (hr : 0 < r) :
    Tendsto (fun epsilon : ℝ => Real.log (1 + r / epsilon))
      (𝓝[>] (0 : ℝ)) atTop := by
  have hq : Tendsto (fun epsilon : ℝ => r / epsilon) (𝓝[>] (0 : ℝ)) atTop := by
    simpa only [div_eq_mul_inv] using tendsto_inv_nhdsGT_zero.const_mul_atTop hr
  exact Real.tendsto_log_atTop.comp
    (tendsto_atTop_mono (fun _ => le_add_of_nonneg_left zero_le_one) hq)

/-- Distinct endpoints with the starting point on Shore become infinitely far
apart as the floor vanishes, including when both endpoints lie on Shore. -/
theorem intrinsicDistance_tendsto_zero_floor_of_mem_left (S : Set E) (x y : E)
    (hx : x ∈ S) (hxy : x ≠ y) (a : ℝ) :
    Tendsto (fun epsilon => intrinsicDistance S epsilon a x y)
      (𝓝[>] (0 : ℝ)) atTop := by
  refine tendsto_atTop_mono' _ ?_
    (log_displacement_tendsto_zero_floor (dist x y) (dist_pos.mpr hxy))
  filter_upwards [self_mem_nhdsWithin] with epsilon he
  simpa only [Metric.infDist_zero_of_mem hx, ruler, zero_pow (by norm_num : (2 : ℕ) ≠ 0),
    zero_add, Real.sqrt_sq he.le] using
      log_displacement_le_intrinsicDistance S epsilon a x y he

end FractalSync.DistanceGraph
