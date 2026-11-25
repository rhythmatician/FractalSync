import FractalSync.IntrinsicLength
import FractalSync.Analysis.ACVectorFTC
import Mathlib.Analysis.Convex.StrictConvexBetween

/-!
# Equality cases for the nearest-Shore calibration

An attained radial calibration forces nonnegative radial velocity equal to
ordinary speed almost everywhere. Strict convexity then identifies the
geometric path with an ambient-norm nearest-point segment.
-/

namespace FractalSync.DistanceGraph

open Set Filter MeasureTheory
open scoped Topology

/-- Equality in the signed radial calibration forces pure outward radial
speed. The premise uses the actual square-root graph density. -/
theorem speed_eq_radial_of_calibration_eq (epsilon a r speed w : ℝ)
    (he : 0 < epsilon) (hspeed : 0 ≤ speed)
    (hcal : graphSpeed a epsilon r speed w = radialDensity epsilon a r * w) :
    speed = w := by
  have hw : 0 ≤ w := (mul_nonneg_iff_of_pos_left (radialDensity_pos epsilon a r he)).1
    (hcal ▸ graphSpeed_nonneg a epsilon r speed w)
  have hsq := congrArg (fun z : ℝ => z ^ 2) hcal
  change (Real.sqrt (graphSpeedSq a epsilon r speed w)) ^ 2 =
    (radialDensity epsilon a r * w) ^ 2 at hsq
  rw [Real.sq_sqrt (graphSpeedSq_nonneg a epsilon r speed w), mul_pow,
    radialDensity_sq epsilon a r w he] at hsq
  have hden : r ^ 2 + epsilon ^ 2 ≠ 0 := by positivity
  have hrem := radial_remainder a epsilon r speed w hden
  rw [hsq, sub_self, eq_div_iff hden, zero_mul] at hrem
  nlinarith

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [FiniteDimensional ℝ E]

/-- An AC path attaining the radial Shore bound moves purely outward almost
everywhere: ordinary speed equals the signed distance-trace derivative. -/
theorem ae_speed_eq_radial_of_minimizing (S : Set E) (epsilon a : ℝ) (gamma : ℝ → E)
    (he : 0 < epsilon) (hgamma : AbsolutelyContinuousOnInterval gamma 0 1)
    (hshore : gamma 0 ∈ S)
    (hmin : acGraphLength S epsilon a gamma =
      radialProfile epsilon a (Metric.infDist (gamma 1) S)) :
    ∀ᵐ t, t ∈ uIcc (0 : ℝ) 1 →
      ‖deriv gamma t‖ = deriv (fun s => Metric.infDist (gamma s) S) t := by
  let r := fun t => Metric.infDist (gamma t) S
  let c := fun t => radialDensity epsilon a (r t) * deriv r t
  let ell := fun t => graphSpeed a epsilon (r t) ‖deriv gamma t‖ (deriv r t)
  have hr : AbsolutelyContinuousOnInterval r 0 1 := ac_distance_trace S gamma 0 1 hgamma
  have hc : IntervalIntegrable c volume 0 1 := ac_radialDensity_intervalIntegrable epsilon a 0 1 r he hr
  have hell : IntervalIntegrable ell volume 0 1 := ac_graphSpeed_intervalIntegrable S gamma epsilon a 0 1 he hgamma
  have hle : c ≤ᵐ[volume.restrict (Ioc (0 : ℝ) 1)] ell := by
    rw [EventuallyLE, ae_restrict_iff' measurableSet_Ioc]
    filter_upwards [ac_distance_trace_deriv_bound S gamma 0 1 hgamma] with t ht hmem
    have hb : |deriv r t| ≤ ‖deriv gamma t‖ := ht (by
      simpa only [uIcc_of_le zero_le_one] using Ioc_subset_Icc_self hmem)
    exact (le_abs_self (c t)).trans (radial_speed_le_graphSpeed epsilon a (r t)
      ‖deriv gamma t‖ (deriv r t) he (by
        simpa only [sq_abs] using (sq_le_sq₀ (abs_nonneg _) (norm_nonneg _)).2 hb))
  have hceq : (∫ t in (0 : ℝ)..1, c t) = ∫ t in (0 : ℝ)..1, ell t := by
    simpa only [r, Metric.infDist_zero_of_mem hshore, radialProfile_zero, sub_zero, ← hmin]
      using ac_radialProfile_integral epsilon a 0 1 r he hr
  have heq : c =ᵐ[volume.restrict (Ioc (0 : ℝ) 1)] ell :=
    (integral_eq_iff_of_ae_le hc.1 hell.1 hle).1 (by
      simpa only [intervalIntegral.integral_of_le zero_le_one] using hceq)
  rw [Measure.restrict_congr_set Ioc_ae_eq_Icc, EventuallyEq,
    ae_restrict_iff' measurableSet_Icc] at heq
  filter_upwards [heq] with t ht hmem
  exact speed_eq_radial_of_calibration_eq epsilon a (r t) _ _ he (norm_nonneg _)
    (ht (by simpa only [uIcc_of_le zero_le_one] using hmem)).symm

omit [FiniteDimensional ℝ E] in
/-- The outward-speed equality integrates on every subinterval. -/
theorem integral_speed_eq_distance_trace_sub (S : Set E) (gamma : ℝ → E)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1)
    (hcal : ∀ᵐ t, t ∈ uIcc (0 : ℝ) 1 →
      ‖deriv gamma t‖ = deriv (fun s => Metric.infDist (gamma s) S) t)
    (u v : ℝ) (hu : u ∈ uIcc (0 : ℝ) 1) (hv : v ∈ uIcc (0 : ℝ) 1) :
    (∫ t in u..v, ‖deriv gamma t‖) = Metric.infDist (gamma v) S - Metric.infDist (gamma u) S := by
  rw [← ((ac_distance_trace S gamma 0 1 hgamma).mono
    (uIcc_subset_uIcc hu hv)).integral_deriv_eq_sub]
  apply intervalIntegral.integral_congr_ae
  filter_upwards [hcal] with t ht hmem
  exact ht ((uIcc_subset_uIcc hu hv) (uIoc_subset_uIcc hmem))

/-- Every ordered subpath has ordinary displacement at most its outward
distance-trace increment once radial calibration is attained. -/
theorem dist_le_distance_trace_sub (S : Set E) (gamma : ℝ → E)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1)
    (hcal : ∀ᵐ t, t ∈ uIcc (0 : ℝ) 1 →
      ‖deriv gamma t‖ = deriv (fun s => Metric.infDist (gamma s) S) t)
    (u v : ℝ) (hu : u ∈ uIcc (0 : ℝ) 1) (hv : v ∈ uIcc (0 : ℝ) 1) (huv : u ≤ v) :
    dist (gamma u) (gamma v) ≤ Metric.infDist (gamma v) S - Metric.infDist (gamma u) S := by
  rw [dist_eq_norm', ← (hgamma.mono (uIcc_subset_uIcc hu hv)).integral_deriv_eq_sub_vector]
  exact (intervalIntegral.norm_integral_le_integral_norm huv).trans_eq
    (integral_speed_eq_distance_trace_sub S gamma hgamma hcal u v hu hv)

/-- The one-Lipschitz distance field supplies the matching lower bound. -/
theorem dist_eq_distance_trace_sub (S : Set E) (gamma : ℝ → E)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1)
    (hcal : ∀ᵐ t, t ∈ uIcc (0 : ℝ) 1 →
      ‖deriv gamma t‖ = deriv (fun s => Metric.infDist (gamma s) S) t)
    (u v : ℝ) (hu : u ∈ uIcc (0 : ℝ) 1) (hv : v ∈ uIcc (0 : ℝ) 1) (huv : u ≤ v) :
    dist (gamma u) (gamma v) = Metric.infDist (gamma v) S - Metric.infDist (gamma u) S := by
  refine le_antisymm (dist_le_distance_trace_sub S gamma hgamma hcal u v hu hv huv) ?_
  have h := Metric.infDist_le_infDist_add_dist (x := gamma v) (y := gamma u) (s := S)
  simpa only [dist_comm] using sub_le_iff_le_add'.mpr h

/-- A calibrated path has a nondecreasing distance trace. -/
theorem monotoneOn_distance_trace_of_calibrated (S : Set E) (gamma : ℝ → E)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1)
    (hcal : ∀ᵐ t, t ∈ uIcc (0 : ℝ) 1 →
      ‖deriv gamma t‖ = deriv (fun s => Metric.infDist (gamma s) S) t) :
    MonotoneOn (fun t => Metric.infDist (gamma t) S) (uIcc (0 : ℝ) 1) := by
  exact fun u hu v hv huv => sub_nonneg.mp (dist_nonneg.trans
    (dist_le_distance_trace_sub S gamma hgamma hcal u v hu hv huv))

/-- The Shore endpoint of every path attaining the radial calibration is an
actual nearest point in the ambient norm. -/
theorem minimizing_shore_endpoint_is_nearest (S : Set E) (epsilon a : ℝ) (gamma : ℝ → E)
    (he : 0 < epsilon) (hgamma : AbsolutelyContinuousOnInterval gamma 0 1)
    (hshore : gamma 0 ∈ S)
    (hmin : acGraphLength S epsilon a gamma =
      radialProfile epsilon a (Metric.infDist (gamma 1) S)) :
    Metric.infDist (gamma 1) S = dist (gamma 1) (gamma 0) := by
  simpa only [Metric.infDist_zero_of_mem hshore, sub_zero, dist_comm] using
    (dist_eq_distance_trace_sub S gamma hgamma
      (ae_speed_eq_radial_of_minimizing S epsilon a gamma he hgamma hshore hmin)
      0 1 left_mem_uIcc right_mem_uIcc zero_le_one).symm

/-- In a strictly convex norm, calibration forces every point onto the
ordinary segment joining the endpoints. This includes real inner-product spaces. -/
theorem mem_segment_of_calibrated [StrictConvexSpace ℝ E] (S : Set E) (gamma : ℝ → E)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1)
    (hcal : ∀ᵐ t, t ∈ uIcc (0 : ℝ) 1 →
      ‖deriv gamma t‖ = deriv (fun s => Metric.infDist (gamma s) S) t)
    (t : ℝ) (ht : t ∈ uIcc (0 : ℝ) 1) :
    gamma t ∈ segment ℝ (gamma 0) (gamma 1) := by
  have ht' : t ∈ Icc (0 : ℝ) 1 := by simpa only [uIcc_of_le zero_le_one] using ht
  apply (mem_segment_iff_wbtw (R := ℝ)).2
  apply dist_add_dist_eq_iff.mp
  rw [dist_eq_distance_trace_sub S gamma hgamma hcal 0 t left_mem_uIcc ht ht'.1,
    dist_eq_distance_trace_sub S gamma hgamma hcal t 1 ht right_mem_uIcc ht'.2,
    dist_eq_distance_trace_sub S gamma hgamma hcal 0 1 left_mem_uIcc right_mem_uIcc zero_le_one]
  ring

/-- The parameter on every nondegenerate minimizing Shore segment is exactly
the normalized distance trace. -/
theorem minimizing_path_eq_lineMap [StrictConvexSpace ℝ E]
    (S : Set E) (epsilon a : ℝ) (gamma : ℝ → E)
    (he : 0 < epsilon) (hgamma : AbsolutelyContinuousOnInterval gamma 0 1)
    (hshore : gamma 0 ∈ S)
    (hmin : acGraphLength S epsilon a gamma = radialProfile epsilon a (Metric.infDist (gamma 1) S))
    (hr : 0 < Metric.infDist (gamma 1) S) (t : ℝ) (ht : t ∈ uIcc (0 : ℝ) 1) :
    gamma t = AffineMap.lineMap (gamma 0) (gamma 1)
      (Metric.infDist (gamma t) S / Metric.infDist (gamma 1) S) := by
  have hm := mem_segment_of_calibrated S gamma hgamma
    (ae_speed_eq_radial_of_minimizing S epsilon a gamma he hgamma hshore hmin) t ht
  rw [segment_eq_image_lineMap] at hm
  obtain ⟨q, hq, hline⟩ := hm
  have hd := infDist_nearest_segment S (gamma 1) (gamma 0) hshore
    (minimizing_shore_endpoint_is_nearest S epsilon a gamma he hgamma hshore hmin) q hq
  rw [hline] at hd
  simpa only [(div_eq_iff hr.ne').2 hd] using hline.symm

/-- Every nondegenerate minimizing Shore path is a monotone AC traversal of
the segment from its nearest Shore endpoint in the ambient norm. -/
theorem minimizing_shore_path_classification [StrictConvexSpace ℝ E]
    (S : Set E) (epsilon a : ℝ) (gamma : ℝ → E)
    (he : 0 < epsilon) (hgamma : AbsolutelyContinuousOnInterval gamma 0 1)
    (hshore : gamma 0 ∈ S)
    (hmin : acGraphLength S epsilon a gamma = radialProfile epsilon a (Metric.infDist (gamma 1) S))
    (hr : 0 < Metric.infDist (gamma 1) S) :
    Metric.infDist (gamma 1) S = dist (gamma 1) (gamma 0) ∧
      ∃ theta : ℝ → ℝ, AbsolutelyContinuousOnInterval theta 0 1 ∧
        MonotoneOn theta (uIcc (0 : ℝ) 1) ∧ theta 0 = 0 ∧ theta 1 = 1 ∧
        ∀ t ∈ uIcc (0 : ℝ) 1, gamma t = AffineMap.lineMap (gamma 0) (gamma 1) (theta t) := by
  refine ⟨minimizing_shore_endpoint_is_nearest S epsilon a gamma he hgamma hshore hmin,
    (fun t => Metric.infDist (gamma t) S / Metric.infDist (gamma 1) S), ?_, ?_, ?_, ?_, ?_⟩
  case refine_1 =>
    simpa only [div_eq_mul_inv, mul_comm] using
      (ac_distance_trace S gamma 0 1 hgamma).const_mul (Metric.infDist (gamma 1) S)⁻¹
  case refine_2 =>
    exact fun u hu v hv huv => div_le_div_of_nonneg_right
      (monotoneOn_distance_trace_of_calibrated S gamma hgamma
        (ae_speed_eq_radial_of_minimizing S epsilon a gamma he hgamma hshore hmin) hu hv huv) hr.le
  case refine_3 => simp only [Metric.infDist_zero_of_mem hshore, zero_div]
  case refine_4 => exact div_self hr.ne'
  exact minimizing_path_eq_lineMap S epsilon a gamma he hgamma hshore hmin hr

/-- A minimizing Shore path with zero endpoint distance is constant on its
whole parameter interval. No strict convexity is needed in this case. -/
theorem minimizing_shore_path_constant_of_zero (S : Set E) (epsilon a : ℝ) (gamma : ℝ → E)
    (he : 0 < epsilon) (hgamma : AbsolutelyContinuousOnInterval gamma 0 1)
    (hshore : gamma 0 ∈ S)
    (hmin : acGraphLength S epsilon a gamma = radialProfile epsilon a (Metric.infDist (gamma 1) S))
    (hzero : Metric.infDist (gamma 1) S = 0) :
    ∀ t ∈ uIcc (0 : ℝ) 1, gamma t = gamma 1 := by
  intro t ht
  have ht' : t ∈ Icc (0 : ℝ) 1 := by simpa only [uIcc_of_le zero_le_one] using ht
  have hb := dist_le_distance_trace_sub S gamma hgamma
    (ae_speed_eq_radial_of_minimizing S epsilon a gamma he hgamma hshore hmin)
    t 1 ht right_mem_uIcc ht'.2
  have hz : dist (gamma t) (gamma 1) ≤ 0 := hb.trans (by
    simpa only [hzero, zero_sub] using neg_nonpos.mpr (Metric.infDist_nonneg (x := gamma t) (s := S)))
  exact dist_eq_zero.mp (le_antisymm hz dist_nonneg)

omit [FiniteDimensional ℝ E] in
/-- Pure outward radial speed almost everywhere suffices to attain the
actual radial Shore length. -/
theorem acGraphLength_eq_radial_of_calibrated (S : Set E) (epsilon a : ℝ)
    (gamma : ℝ → E) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) (hshore : gamma 0 ∈ S)
    (hcal : ∀ᵐ t, t ∈ uIcc (0 : ℝ) 1 →
      ‖deriv gamma t‖ = deriv (fun s => Metric.infDist (gamma s) S) t) :
    acGraphLength S epsilon a gamma = radialProfile epsilon a (Metric.infDist (gamma 1) S) := by
  have hi := ac_radialProfile_integral epsilon a 0 1
    (fun t => Metric.infDist (gamma t) S) he (ac_distance_trace S gamma 0 1 hgamma)
  simp only [Metric.infDist_zero_of_mem hshore, radialProfile_zero, sub_zero] at hi
  rw [← hi]
  apply intervalIntegral.integral_congr_ae
  filter_upwards [hcal] with t ht hmem
  have ht' := ht (uIoc_subset_uIcc hmem)
  rw [← ht', graphSpeed_radial epsilon a _ _ he, abs_of_nonneg (norm_nonneg _)]

omit [FiniteDimensional ℝ E] in
/-- An AC parameter gives an actual AC traversal of an affine segment. -/
theorem ac_lineMap_comp (p x : E) (theta : ℝ → ℝ)
    (htheta : AbsolutelyContinuousOnInterval theta 0 1) :
    AbsolutelyContinuousOnInterval (fun t => AffineMap.lineMap p x (theta t)) 0 1 := by
  exact (lipschitzWith_lineMap p x).comp_absolutelyContinuousOnInterval htheta

omit [FiniteDimensional ℝ E] in
/-- Every monotone AC traversal of a nearest segment has pure outward radial
speed almost everywhere, including intervals on which its parameter is constant. -/
theorem ae_speed_eq_radial_of_monotone_nearest_segment (S : Set E)
    (gamma : ℝ → E) (theta : ℝ → ℝ)
    (hshore : gamma 0 ∈ S)
    (hnearest : Metric.infDist (gamma 1) S = dist (gamma 1) (gamma 0))
    (htheta : AbsolutelyContinuousOnInterval theta 0 1)
    (hmono : MonotoneOn theta (uIcc (0 : ℝ) 1))
    (hzero : theta 0 = 0) (hone : theta 1 = 1)
    (hsegment : ∀ t ∈ uIcc (0 : ℝ) 1,
      gamma t = AffineMap.lineMap (gamma 0) (gamma 1) (theta t)) :
    ∀ᵐ t, t ∈ uIcc (0 : ℝ) 1 →
      ‖deriv gamma t‖ = deriv (fun s => Metric.infDist (gamma s) S) t := by
  have hrange : ∀ t ∈ uIcc (0 : ℝ) 1, theta t ∈ Icc (0 : ℝ) 1 := by
    intro t ht
    have ht' : t ∈ Icc (0 : ℝ) 1 := by simpa only [uIcc_of_le zero_le_one] using ht
    exact ⟨hzero ▸ hmono left_mem_uIcc ht ht'.1, hone ▸ hmono ht right_mem_uIcc ht'.2⟩
  filter_upwards [htheta.ae_differentiableAt,
    show ∀ᵐ t : ℝ, t ≠ 0 from by simp [ae_iff],
    show ∀ᵐ t : ℝ, t ≠ 1 from by simp [ae_iff]] with t hdt hne0 hne1 hmem
  have ht : t ∈ Ioo (0 : ℝ) 1 := by
    have hm : t ∈ Icc (0 : ℝ) 1 := by simpa only [uIcc_of_le zero_le_one] using hmem
    exact ⟨lt_of_le_of_ne hm.1 hne0.symm, lt_of_le_of_ne hm.2 hne1⟩
  have hnonneg : 0 ≤ deriv theta t := by
    simpa only [uIcc_of_le zero_le_one,
      derivWithin_of_mem_nhds (Icc_mem_nhds ht.1 ht.2)] using hmono.derivWithin_nonneg (x := t)
  have hloc : gamma =ᶠ[𝓝 t] (fun s => AffineMap.lineMap (gamma 0) (gamma 1) (theta s)) := by
    filter_upwards [Icc_mem_nhds ht.1 ht.2] with s hs
    exact hsegment s (by simpa only [uIcc_of_le zero_le_one] using hs)
  have hdgamma : HasDerivAt gamma (deriv theta t • (gamma 1 - gamma 0)) t := by
    apply HasDerivAt.congr_of_eventuallyEq _ hloc
    simpa only [AffineMap.lineMap_apply_module'] using
      ((hdt hmem).hasDerivAt.smul_const (gamma 1 - gamma 0)).add_const (gamma 0)
  have hdtrace : HasDerivAt (fun s => Metric.infDist (gamma s) S)
      (deriv theta t * Metric.infDist (gamma 1) S) t := by
    apply ((hdt hmem).hasDerivAt.mul_const (Metric.infDist (gamma 1) S)).congr_of_eventuallyEq
    filter_upwards [Icc_mem_nhds ht.1 ht.2] with s hs
    have hs' : s ∈ uIcc (0 : ℝ) 1 := by simpa only [uIcc_of_le zero_le_one] using hs
    rw [hsegment s hs']
    exact infDist_nearest_segment S (gamma 1) (gamma 0) hshore hnearest (theta s) (hrange s hs')
  rw [hdgamma.deriv, hdtrace.deriv, norm_smul, Real.norm_eq_abs, abs_of_nonneg hnonneg,
    hnearest, dist_eq_norm]

omit [FiniteDimensional ℝ E] in
/-- Every monotone AC traversal of a nearest Shore segment attains the actual
radial length. This converse requires no strict convexity or nonzero length. -/
theorem acGraphLength_eq_radial_of_monotone_nearest_segment (S : Set E) (epsilon a : ℝ)
    (gamma : ℝ → E) (theta : ℝ → ℝ) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) (hshore : gamma 0 ∈ S)
    (hnearest : Metric.infDist (gamma 1) S = dist (gamma 1) (gamma 0))
    (htheta : AbsolutelyContinuousOnInterval theta 0 1)
    (hmono : MonotoneOn theta (uIcc (0 : ℝ) 1))
    (hzero : theta 0 = 0) (hone : theta 1 = 1)
    (hsegment : ∀ t ∈ uIcc (0 : ℝ) 1,
      gamma t = AffineMap.lineMap (gamma 0) (gamma 1) (theta t)) :
    acGraphLength S epsilon a gamma = radialProfile epsilon a (Metric.infDist (gamma 1) S) := by
  exact acGraphLength_eq_radial_of_calibrated S epsilon a gamma he hgamma hshore
    (ae_speed_eq_radial_of_monotone_nearest_segment S gamma theta hshore hnearest
      htheta hmono hzero hone hsegment)

/-- Complete classification of AC paths from Shore attaining its radial
distance. The degenerate branch is constant; every other minimizer traverses
a nearest-point segment with an AC nondecreasing parameter, and conversely. -/
theorem minimizing_shore_path_iff [StrictConvexSpace ℝ E]
    (S : Set E) (epsilon a : ℝ) (gamma : ℝ → E)
    (he : 0 < epsilon) (hgamma : AbsolutelyContinuousOnInterval gamma 0 1)
    (hshore : gamma 0 ∈ S) :
    acGraphLength S epsilon a gamma = radialProfile epsilon a (Metric.infDist (gamma 1) S) ↔
      (Metric.infDist (gamma 1) S = 0 ∧ ∀ t ∈ uIcc (0 : ℝ) 1, gamma t = gamma 1) ∨
      (0 < Metric.infDist (gamma 1) S ∧
        Metric.infDist (gamma 1) S = dist (gamma 1) (gamma 0) ∧
        ∃ theta : ℝ → ℝ, AbsolutelyContinuousOnInterval theta 0 1 ∧
          MonotoneOn theta (uIcc (0 : ℝ) 1) ∧ theta 0 = 0 ∧ theta 1 = 1 ∧
          ∀ t ∈ uIcc (0 : ℝ) 1,
            gamma t = AffineMap.lineMap (gamma 0) (gamma 1) (theta t)) := by
  constructor
  · intro hmin
    by_cases hzero : Metric.infDist (gamma 1) S = 0
    · exact Or.inl ⟨hzero, minimizing_shore_path_constant_of_zero S epsilon a gamma
        he hgamma hshore hmin hzero⟩
    · have hr : 0 < Metric.infDist (gamma 1) S :=
        lt_of_le_of_ne Metric.infDist_nonneg (Ne.symm hzero)
      exact Or.inr ⟨hr, minimizing_shore_path_classification S epsilon a gamma
        he hgamma hshore hmin hr⟩
  · rintro (⟨hzero, hconstant⟩ | ⟨_, hnearest, theta, htheta, hmono, h0, h1, hsegment⟩)
    · have hend : gamma 0 = gamma 1 := hconstant 0 left_mem_uIcc
      apply acGraphLength_eq_radial_of_monotone_nearest_segment S epsilon a gamma id
        he hgamma hshore (by rw [hend, dist_self, hzero])
        LipschitzWith.id.lipschitzOnWith.absolutelyContinuousOnInterval
        (monotone_id.monotoneOn _) rfl rfl
      intro t ht
      simpa only [hend, AffineMap.lineMap_same_apply] using hconstant t ht
    · exact acGraphLength_eq_radial_of_monotone_nearest_segment S epsilon a gamma theta
        he hgamma hshore hnearest htheta hmono h0 h1 hsegment

/-- The reversed orientation gives an AC nonincreasing nearest-segment
parameter for every nondegenerate path minimizing its distance to Shore. -/
theorem minimizing_path_to_shore_classification [StrictConvexSpace ℝ E]
    (S : Set E) (epsilon a : ℝ) (gamma : ℝ → E)
    (he : 0 < epsilon) (hgamma : AbsolutelyContinuousOnInterval gamma 0 1)
    (hshore : gamma 1 ∈ S)
    (hmin : acGraphLength S epsilon a gamma = radialProfile epsilon a (Metric.infDist (gamma 0) S))
    (hr : 0 < Metric.infDist (gamma 0) S) :
    Metric.infDist (gamma 0) S = dist (gamma 0) (gamma 1) ∧
      ∃ theta : ℝ → ℝ, AbsolutelyContinuousOnInterval theta 0 1 ∧
        AntitoneOn theta (uIcc (0 : ℝ) 1) ∧ theta 0 = 1 ∧ theta 1 = 0 ∧
        ∀ t ∈ uIcc (0 : ℝ) 1,
          gamma t = AffineMap.lineMap (gamma 1) (gamma 0) (theta t) := by
  have hrev := minimizing_shore_path_classification S epsilon a (reversePath gamma)
    he (ac_reversePath gamma hgamma) (by simpa only [reversePath_zero] using hshore)
    (by simpa only [acGraphLength_reversePath, reversePath_one] using hmin)
    (by simpa only [reversePath_one] using hr)
  rcases hrev with ⟨hnearest, theta, htheta, hmono, h0, h1, hsegment⟩
  refine ⟨by simpa only [reversePath_one, reversePath_zero] using hnearest,
    reversePath theta, ac_reversePath theta htheta, ?_, ?_, ?_, ?_⟩
  case refine_1 =>
    intro u hu v hv huv
    apply hmono
    · simp only [uIcc_of_le zero_le_one, mem_Icc] at hv ⊢
      constructor <;> linarith [hv.1, hv.2]
    · simp only [uIcc_of_le zero_le_one, mem_Icc] at hu ⊢
      constructor <;> linarith [hu.1, hu.2]
    · linarith
  case refine_2 => simpa only [reversePath_zero] using h1
  case refine_3 => simpa only [reversePath_one] using h0
  intro t ht
  have ht' : 1 - t ∈ uIcc (0 : ℝ) 1 := by
    simp only [uIcc_of_le zero_le_one, mem_Icc] at ht ⊢
    constructor <;> linarith [ht.1, ht.2]
  simpa only [reversePath, sub_sub_cancel, sub_zero, sub_self] using hsegment (1 - t) ht'

end FractalSync.DistanceGraph
