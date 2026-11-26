import FractalSync.Analysis.AbsolutelyContinuousFTC
import FractalSync.Analysis.FiniteDimensionalAC
import Mathlib.MeasureTheory.Function.SpecialFunctions.Basic
import FractalSync.PathCalibration
import FractalSync.Ruler
import FractalSync.RadialLipschitz

/-!
# Absolutely continuous paths in regularized distance-graph geometry

This module uses Mathlib's epsilon-delta definition of absolute continuity and
actual almost-everywhere derivatives. Distance-to-set and ruler traces need no
ambient differentiability. The scalar fundamental theorem is supplied by the
attributed, kernel-checked backport in `Analysis/AbsolutelyContinuousFTC`.
-/

namespace FractalSync.DistanceGraph

open MeasureTheory Set Filter
open scoped Topology

/-- The graph speed is nonnegative for all scalar inputs. -/
theorem graphSpeed_nonneg (a epsilon r speed w : ℝ) :
    0 ≤ graphSpeed a epsilon r speed w := Real.sqrt_nonneg _

/-- The expanded graph density equals the original ruler-graph expression when
its scalar derivative is supplied by the chain rule. -/
theorem graphSpeed_eq_rulerSpeed (epsilon a r speed w : ℝ) (he : 0 < epsilon) :
    graphSpeed a epsilon r speed w =
      Real.sqrt (speed ^ 2 + a ^ 2 * (r / ruler epsilon r * w) ^ 2) /
        ruler epsilon r := by
  have hrho := ruler_pos epsilon r he
  have heq : graphSpeedSq a epsilon r speed w =
      (speed ^ 2 + a ^ 2 * (r / ruler epsilon r * w) ^ 2) / (ruler epsilon r) ^ 2 := by
    unfold graphSpeedSq
    rw [← ruler_sq epsilon r]
    field_simp
  rw [graphSpeed, heq, Real.sqrt_div' _ (sq_nonneg _), Real.sqrt_sq hrho.le]

/-- A global upper speed bound, uniform in the distance coordinate. -/
theorem graphSpeed_le_global (epsilon a r speed w : ℝ) (he : 0 < epsilon)
    (hspeed : 0 ≤ speed) (hw : w ^ 2 ≤ speed ^ 2) :
    graphSpeed a epsilon r speed w ≤ Real.sqrt (1 + a ^ 2) / epsilon * speed := by
  apply (Real.sqrt_le_left (by positivity : 0 ≤ Real.sqrt (1 + a ^ 2) / epsilon * speed)).2
  have hb := (graph_scale_comparison a epsilon r speed w he hw).2
  have hden : 0 < r ^ 2 + epsilon ^ 2 := by positivity
  have he2 : 0 < epsilon ^ 2 := sq_pos_of_pos he
  have hdiv : speed ^ 2 / (r ^ 2 + epsilon ^ 2) ≤ speed ^ 2 / epsilon ^ 2 :=
    div_le_div_of_nonneg_left (sq_nonneg speed) he2 (by nlinarith [sq_nonneg r])
  calc
    graphSpeedSq a epsilon r speed w ≤
        (1 + a ^ 2) * (speed ^ 2 / epsilon ^ 2) :=
      hb.trans (mul_le_mul_of_nonneg_left hdiv (by positivity))
    _ = (Real.sqrt (1 + a ^ 2) / epsilon * speed) ^ 2 := by
      rw [mul_pow, div_pow, Real.sq_sqrt (by positivity)]
      ring

/-- Distance to any set preserves actual absolute continuity under composition.
Closedness and nonemptiness are needed only for nearest-point existence. -/
theorem ac_distance_trace {E : Type*} [PseudoMetricSpace E]
    (S : Set E) (gamma : ℝ → E) (lo hi : ℝ)
    (hgamma : AbsolutelyContinuousOnInterval gamma lo hi) :
    AbsolutelyContinuousOnInterval (fun t => Metric.infDist (gamma t) S) lo hi :=
  (Metric.lipschitz_infDist_pt S).comp_absolutelyContinuousOnInterval hgamma

/-- Positive regularization preserves absolute continuity of the distance trace. -/
theorem ac_ruler_trace {E : Type*} [PseudoMetricSpace E]
    (S : Set E) (gamma : ℝ → E) (epsilon lo hi : ℝ) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma lo hi) :
    AbsolutelyContinuousOnInterval
      (fun t => ruler epsilon (Metric.infDist (gamma t) S)) lo hi :=
  (regularizedDistance_lipschitz epsilon he S).comp_absolutelyContinuousOnInterval hgamma

/-- The distance-trace speed bound now follows almost everywhere from absolute
continuity alone, including curves through nonsmooth distance-field points. -/
theorem ac_distance_trace_deriv_bound {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] [FiniteDimensional ℝ E]
    (S : Set E) (gamma : ℝ → E) (lo hi : ℝ)
    (hgamma : AbsolutelyContinuousOnInterval gamma lo hi) :
    ∀ᵐ t, t ∈ uIcc lo hi →
      |deriv (fun s => Metric.infDist (gamma s) S) t| ≤ ‖deriv gamma t‖ := by
  filter_upwards [hgamma.ae_differentiableAt,
    (ac_distance_trace S gamma lo hi hgamma).ae_differentiableAt] with t hg hr ht
  exact infDist_deriv_bound S gamma t _ _ (hg ht).hasDerivAt (hr ht).hasDerivAt

/-- Composing the radial primitive with an absolutely continuous scalar trace
preserves absolute continuity. -/
theorem ac_radialProfile_trace (epsilon a lo hi : ℝ) (r : ℝ → ℝ)
    (he : 0 < epsilon) (hr : AbsolutelyContinuousOnInterval r lo hi) :
    AbsolutelyContinuousOnInterval (fun t => radialProfile epsilon a (r t)) lo hi :=
  (radialProfile_lipschitz epsilon a he).comp_absolutelyContinuousOnInterval hr

/-- Actual almost-everywhere chain rule for the radial calibration. -/
theorem ac_radialProfile_deriv (epsilon a lo hi : ℝ) (r : ℝ → ℝ)
    (he : 0 < epsilon) (hr : AbsolutelyContinuousOnInterval r lo hi) :
    ∀ᵐ t, t ∈ uIcc lo hi →
      deriv (fun s => radialProfile epsilon a (r s)) t =
        radialDensity epsilon a (r t) * deriv r t := by
  filter_upwards [hr.ae_differentiableAt] with t ht hmem
  exact ((hasDerivAt_radialProfile epsilon a (r t) he).comp t (ht hmem).hasDerivAt).deriv

/-- Absolute continuity supplies integrability of the calibrated radial speed. -/
theorem ac_radialDensity_intervalIntegrable (epsilon a lo hi : ℝ) (r : ℝ → ℝ)
    (he : 0 < epsilon) (hr : AbsolutelyContinuousOnInterval r lo hi) :
    IntervalIntegrable (fun t => radialDensity epsilon a (r t) * deriv r t)
      volume lo hi := by
  apply (ac_radialProfile_trace epsilon a lo hi r he hr).intervalIntegrable_deriv.congr_ae
  rw [EventuallyEq, ae_restrict_iff' measurableSet_uIoc]
  filter_upwards [ac_radialProfile_deriv epsilon a lo hi r he hr] with t ht hmem
  exact ht (uIoc_subset_uIcc hmem)

/-- The radial FTC identity for arbitrary absolutely continuous scalar traces. -/
theorem ac_radialProfile_integral (epsilon a lo hi : ℝ) (r : ℝ → ℝ)
    (he : 0 < epsilon) (hr : AbsolutelyContinuousOnInterval r lo hi) :
    (∫ t in lo..hi, radialDensity epsilon a (r t) * deriv r t) =
      radialProfile epsilon a (r hi) - radialProfile epsilon a (r lo) := by
  rw [← (ac_radialProfile_trace epsilon a lo hi r he hr).integral_deriv_eq_sub]
  apply intervalIntegral.integral_congr_ae
  filter_upwards [ac_radialProfile_deriv epsilon a lo hi r he hr] with t ht hmem
  exact (ht (uIoc_subset_uIcc hmem)).symm

/-- The speed functional is measurable whenever its three scalar inputs are. -/
theorem aestronglyMeasurable_graphSpeed (epsilon a : ℝ) (r speed w : ℝ → ℝ)
    (mu : Measure ℝ) (hr : AEStronglyMeasurable r mu)
    (hs : AEStronglyMeasurable speed mu) (hw : AEStronglyMeasurable w mu) :
    AEStronglyMeasurable (fun t => graphSpeed a epsilon (r t) (speed t) (w t)) mu := by
  unfold graphSpeed graphSpeedSq
  exact (((hs.aemeasurable.pow_const 2).div
    ((hr.aemeasurable.pow_const 2).add_const (epsilon ^ 2))).add
    ((((hr.aemeasurable.pow_const 2).const_mul (a ^ 2)).mul (hw.aemeasurable.pow_const 2)).div
      (((hr.aemeasurable.pow_const 2).add_const (epsilon ^ 2)).pow_const 2))).sqrt.aestronglyMeasurable

/-- Every finite-dimensional absolutely continuous path has an integrable graph
speed. No separate regularity or integrability hypothesis is imposed on its
actual velocity or distance-trace derivative. -/
theorem ac_graphSpeed_intervalIntegrable {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] [FiniteDimensional ℝ E]
    (S : Set E) (gamma : ℝ → E) (epsilon a lo hi : ℝ) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma lo hi) :
    IntervalIntegrable (fun t => graphSpeed a epsilon (Metric.infDist (gamma t) S)
      ‖deriv gamma t‖ (deriv (fun s => Metric.infDist (gamma s) S) t)) volume lo hi := by
  have hr := ac_distance_trace S gamma lo hi hgamma
  apply (hgamma.intervalIntegrable_deriv_vector.norm.const_mul
    (Real.sqrt (1 + a ^ 2) / epsilon)).mono_fun'
  · apply aestronglyMeasurable_graphSpeed
    · exact hr.continuousOn.intervalIntegrable.def'.aestronglyMeasurable
    · exact (aestronglyMeasurable_deriv gamma _).norm
    · exact aestronglyMeasurable_deriv _ _
  · rw [EventuallyLE, ae_restrict_iff' measurableSet_uIoc]
    filter_upwards [ac_distance_trace_deriv_bound S gamma lo hi hgamma] with t ht hmem
    rw [Real.norm_eq_abs, abs_of_nonneg (graphSpeed_nonneg _ _ _ _ _)]
    apply graphSpeed_le_global epsilon a _ _ _ he (norm_nonneg _)
    simpa only [sq_abs] using
      (sq_le_sq₀ (abs_nonneg _) (norm_nonneg _)).2 (ht (uIoc_subset_uIcc hmem))

/-- The exact integrated endpoint calibration for arbitrary absolutely
continuous finite-dimensional curves. All a.e. derivative and integrability
obligations are proved from absolute continuity. -/
theorem ac_distance_trace_endpoint_bound {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] [FiniteDimensional ℝ E]
    (S : Set E) (gamma : ℝ → E) (epsilon a lo hi : ℝ)
    (he : 0 < epsilon) (horder : lo ≤ hi)
    (hgamma : AbsolutelyContinuousOnInterval gamma lo hi) :
    |radialProfile epsilon a (Metric.infDist (gamma hi) S) -
      radialProfile epsilon a (Metric.infDist (gamma lo) S)| ≤
      ∫ t in lo..hi, graphSpeed a epsilon (Metric.infDist (gamma t) S)
        ‖deriv gamma t‖ (deriv (fun s => Metric.infDist (gamma s) S) t) := by
  have hr := ac_distance_trace S gamma lo hi hgamma
  rw [← ac_radialProfile_integral epsilon a lo hi _ he hr]
  apply (intervalIntegral.abs_integral_le_integral_abs horder).trans
  apply intervalIntegral.integral_mono_ae_restrict horder
    (ac_radialDensity_intervalIntegrable epsilon a lo hi _ he hr).abs
    (ac_graphSpeed_intervalIntegrable S gamma epsilon a lo hi he hgamma)
  rw [EventuallyLE, ae_restrict_iff' measurableSet_Icc]
  filter_upwards [ac_distance_trace_deriv_bound S gamma lo hi hgamma] with t ht hmem
  apply radial_speed_le_graphSpeed epsilon a _ _ _ he
  simpa only [sq_abs] using
    (sq_le_sq₀ (abs_nonneg _) (norm_nonneg _)).2 (ht (by simpa [uIcc_of_le horder] using hmem))

/-- An arbitrary absolutely continuous Shore-crossing path pays the sum of its
two calibrated endpoint distances. No differentiability is required at the
crossing time or at any specified parameter value. -/
theorem ac_distance_trace_crossing_bound {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] [FiniteDimensional ℝ E]
    (S : Set E) (gamma : ℝ → E) (epsilon a lo c hi : ℝ)
    (he : 0 < epsilon) (hlc : lo ≤ c) (hchi : c ≤ hi)
    (hgamma : AbsolutelyContinuousOnInterval gamma lo hi) (hshore : gamma c ∈ S) :
    radialProfile epsilon a (Metric.infDist (gamma lo) S) +
      radialProfile epsilon a (Metric.infDist (gamma hi) S) ≤
      ∫ t in lo..hi, graphSpeed a epsilon (Metric.infDist (gamma t) S)
        ‖deriv gamma t‖ (deriv (fun s => Metric.infDist (gamma s) S) t) := by
  have hleft_sub : uIcc lo c ⊆ uIcc lo hi := by
    simpa only [uIcc_of_le hlc, uIcc_of_le (hlc.trans hchi)] using
      (Icc_subset_Icc le_rfl hchi)
  have hright_sub : uIcc c hi ⊆ uIcc lo hi := by
    simpa only [uIcc_of_le hchi, uIcc_of_le (hlc.trans hchi)] using
      (Icc_subset_Icc hlc le_rfl)
  have hleft := ac_distance_trace_endpoint_bound S gamma epsilon a lo c he hlc
    (hgamma.mono hleft_sub)
  have hright := ac_distance_trace_endpoint_bound S gamma epsilon a c hi he hchi
    (hgamma.mono hright_sub)
  have hgraph := ac_graphSpeed_intervalIntegrable S gamma epsilon a lo hi he hgamma
  simp only [Metric.infDist_zero_of_mem hshore, radialProfile_zero,
    zero_sub, sub_zero, abs_neg] at hleft hright
  exact (add_le_add ((le_abs_self _).trans hleft) ((le_abs_self _).trans hright)).trans_eq
    (intervalIntegral.integral_add_adjacent_intervals
      (hgraph.mono_set hleft_sub) (hgraph.mono_set hright_sub))

end FractalSync.DistanceGraph
