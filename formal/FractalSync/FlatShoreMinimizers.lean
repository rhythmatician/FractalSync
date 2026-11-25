import FractalSync.FlatShoreCalibration
import FractalSync.ACScalarVariation

/-!
# Global flat-Shore minimality and equality at the threshold

The spherical calibration proves the global lower bound on unrestricted AC
paths. Its strict radial speed deficit supplies equality rigidity.
-/

namespace FractalSync.FlatShoreMinimizers

open Set MeasureTheory Filter FlatShoreIntrinsic FlatShoreCalibration DistanceGraph
open scoped Topology

/-- The actual AC path distance equals the Shore length up to and including pi. -/
theorem intrinsicDistance_eq_shore_span (ell a : ℝ) (hell : ell ∈ Icc (0 : ℝ) Real.pi) :
    intrinsicDistance shore 1 a (chart.symm (0, 0)) (chart.symm (0, ell)) = ell := by
  refine le_antisymm ?_ (le_csInf (pairwisePathLengths_nonempty shore 1 a _ _)
    (fun L ⟨gamma, hg, hstart, hend, hlength⟩ =>
      hlength ▸ shore_span_le_acGraphLength ell a hell gamma hg hstart hend))
  by_cases hzero : ell = 0
  · simpa [hzero] using (intrinsicDistance_self shore 1 a (chart.symm (0, 0))).le
  · exact (FlatShoreShortcuts.intrinsicDistance_le_oneModeLength ell a 0
      (lt_of_le_of_ne hell.1 (Ne.symm hzero))).trans_eq
        (FlatShoreSecondVariation.oneModeLength_zero ell a)

theorem normal_mul_velocity_eq_zero_of_speed_eq (a r : ℝ) (v : FlatShoreCalibration.Coord)
    (heq : sphericalSpeed r v = Real.sqrt (FlatShoreGeometry.metric 1 a r v v)) :
    r * v.1 = 0 := by
  have hmetric : 0 ≤ FlatShoreGeometry.metric 1 a r v v := by
    simpa only [FlatShoreGeometry.metric, pow_two, mul_assoc] using
      add_nonneg
        (mul_nonneg (FlatShoreGeometry.radialMetric_pos 1 a r (by norm_num)).le (sq_nonneg v.1))
        (mul_nonneg (FlatShoreGeometry.tangentialMetric_pos 1 r (by norm_num)).le (sq_nonneg v.2))
  have hsq := congrArg (fun z : ℝ => z ^ 2) heq
  simp only [sphericalSpeed, Real.sq_sqrt (by positivity :
    0 ≤ v.1 ^ 2 / (1 + r ^ 2) ^ 2 + v.2 ^ 2 / (1 + r ^ 2)),
    Real.sq_sqrt hmetric] at hsq
  simp only [FlatShoreGeometry.metric, FlatShoreGeometry.radialMetric,
    FlatShoreGeometry.tangentialMetric, one_pow] at hsq
  field_simp at hsq
  exact mul_self_eq_zero.mp (show (r * v.1) * (r * v.1) = 0 by
    nlinarith [sq_nonneg (a * r * v.1), sq_nonneg (r * v.1)])

theorem ae_normal_mul_velocity_eq_zero (ell a : ℝ) (hell : ell ∈ Icc (0 : ℝ) Real.pi)
    (gamma : ℝ → Plane) (hg : AbsolutelyContinuousOnInterval gamma 0 1)
    (hstart : gamma 0 = chart.symm (0, 0)) (hend : gamma 1 = chart.symm (0, ell))
    (hmin : acGraphLength shore 1 a gamma = ell) :
    ∀ᵐ t, t ∈ uIcc (0 : ℝ) 1 → (gamma t).1 * (deriv gamma t).1 = 0 := by
  have heq : sphericalLength gamma = coordinateLength 1 a gamma :=
    le_antisymm
      ((sphericalLength_le_acGraphLength a gamma hg).trans_eq
        (coordinateLength_eq_acGraphLength 1 a (by norm_num) gamma hg).symm)
      ((coordinateLength_eq_acGraphLength 1 a (by norm_num) gamma hg).trans hmin ▸
        shore_span_le_sphericalLength ell hell gamma hg hstart hend)
  have hae := (integral_eq_iff_of_ae_le
    (sphericalSpeed_intervalIntegrable gamma hg).1
    (coordinateSpeed_intervalIntegrable 1 a 0 1 (by norm_num) gamma hg).1
      (Eventually.of_forall (fun t =>
        sphericalSpeed_le_coordinateSpeed a (gamma t).1 (chart (deriv gamma t))))).1
      (by simpa only [sphericalLength, coordinateLength,
        intervalIntegral.integral_of_le zero_le_one] using heq)
  rw [Measure.restrict_congr_set Ioc_ae_eq_Icc, EventuallyEq,
    ae_restrict_iff' measurableSet_Icc] at hae
  exact hae.mono (fun t ht hmem =>
    normal_mul_velocity_eq_zero_of_speed_eq a (gamma t).1 (chart (deriv gamma t))
      (ht (by simpa only [uIcc_of_le zero_le_one] using hmem)))

/-- Every global minimizer in the short-span branch stays on Shore, including
at the conjugate endpoint. Pauses and arbitrary AC parameterizations are allowed. -/
theorem minimizing_path_on_shore (ell a : ℝ) (hell : ell ∈ Icc (0 : ℝ) Real.pi)
    (gamma : ℝ → Plane) (hg : AbsolutelyContinuousOnInterval gamma 0 1)
    (hstart : gamma 0 = chart.symm (0, 0)) (hend : gamma 1 = chart.symm (0, ell))
    (hmin : acGraphLength shore 1 a gamma = ell) :
    ∀ t ∈ uIcc (0 : ℝ) 1, (gamma t).1 = 0 := by
  have hr : AbsolutelyContinuousOnInterval (fun t => (gamma t).1) 0 1 :=
    ((ContinuousLinearMap.fst ℝ ℝ ℝ).comp chart.toContinuousLinearMap).lipschitz.comp_absolutelyContinuousOnInterval hg
  have hzero : ∀ᵐ t, t ∈ uIcc (0 : ℝ) 1 →
      HasDerivAt (fun s => (gamma s).1 * (gamma s).1) 0 t := by
    filter_upwards [hg.ae_differentiableAt,
      ae_normal_mul_velocity_eq_zero ell a hell gamma hg hstart hend hmin] with t ht hnorm hmem
    have hd := hasDerivAt_normal gamma t (deriv gamma t) (ht hmem).hasDerivAt
    simpa only [Pi.mul_apply, mul_comm (deriv gamma t).1, hnorm hmem, zero_add] using hd.mul hd
  obtain ⟨C, hC⟩ := (hr.fun_mul hr).const_of_ae_hasDerivAt_zero hzero
  have hC0 : C = 0 := by simpa [hstart, chart] using (hC 0 left_mem_uIcc).symm
  exact fun t ht => mul_self_eq_zero.mp ((hC t ht).trans hC0)

/-- On Shore, the actual graph length is the total variation integral of the
tangential coordinate. -/
theorem acGraphLength_on_shore (a : ℝ) (gamma : ℝ → Plane)
    (hg : AbsolutelyContinuousOnInterval gamma 0 1)
    (hshore : ∀ t ∈ uIcc (0 : ℝ) 1, (gamma t).1 = 0) :
    acGraphLength shore 1 a gamma = ∫ t in (0 : ℝ)..1,
      |deriv (fun s => (gamma s).2) t| := by
  rw [← coordinateLength_eq_acGraphLength 1 a (by norm_num) gamma hg]
  have hnormal : EqOn (fun t => (gamma t).1) (fun _ => 0) (Ioo (0 : ℝ) 1) :=
    fun t ht => hshore t (by simpa only [uIcc_of_le zero_le_one] using Ioo_subset_Icc_self ht)
  apply intervalIntegral.integral_congr_ae
  filter_upwards [hg.ae_differentiableAt, show ∀ᵐ t : ℝ, t ≠ 1 from by simp [ae_iff]]
    with t ht hne hmem
  have hIoo : t ∈ Ioo (0 : ℝ) 1 := by
    simp only [uIoc_of_le zero_le_one] at hmem
    exact ⟨hmem.1, lt_of_le_of_ne hmem.2 hne⟩
  have hrzero : (deriv gamma t).1 = 0 := by
    simpa only [(hasDerivAt_normal gamma t _ (ht (uIoc_subset_uIcc hmem)).hasDerivAt).deriv,
      deriv_const] using hnormal.deriv isOpen_Ioo hIoo
  have hyderiv : HasDerivAt (fun s => (gamma s).2) (deriv gamma t).2 t :=
    ((ContinuousLinearMap.snd ℝ ℝ ℝ).comp chart.toContinuousLinearMap).hasFDerivAt.comp_hasDerivAt t
      (ht (uIoc_subset_uIcc hmem)).hasDerivAt
  simp [FlatShoreGeometry.metric, FlatShoreGeometry.radialMetric,
    FlatShoreGeometry.tangentialMetric, hshore t (uIoc_subset_uIcc hmem),
    chart, hrzero, hyderiv.deriv, ← pow_two, Real.sqrt_sq_eq_abs]

/-- Complete AC equality classification below and at the flat threshold.
The unique minimizing image is the Shore segment; its parameter may pause. -/
theorem minimizing_path_iff (ell a : ℝ) (hell : ell ∈ Icc (0 : ℝ) Real.pi)
    (gamma : ℝ → Plane) (hg : AbsolutelyContinuousOnInterval gamma 0 1)
    (hstart : gamma 0 = chart.symm (0, 0)) (hend : gamma 1 = chart.symm (0, ell)) :
    acGraphLength shore 1 a gamma = ell ↔
      (∀ t ∈ uIcc (0 : ℝ) 1, (gamma t).1 = 0) ∧
        MonotoneOn (fun t => (gamma t).2) (uIcc (0 : ℝ) 1) := by
  have hy : AbsolutelyContinuousOnInterval (fun t => (gamma t).2) 0 1 :=
    ((ContinuousLinearMap.snd ℝ ℝ ℝ).comp chart.toContinuousLinearMap).lipschitz.comp_absolutelyContinuousOnInterval hg
  constructor
  · intro hmin
    have hshore := minimizing_path_on_shore ell a hell gamma hg hstart hend hmin
    refine ⟨hshore, (ACScalarVariation.integral_abs_deriv_eq_sub_iff_monotoneOn
      _ 0 1 zero_le_one hy).mp ?_⟩
    simp [← acGraphLength_on_shore a gamma hg hshore, hmin, hstart, hend, chart]
  · rintro ⟨hshore, hmono⟩
    simpa only [← acGraphLength_on_shore a gamma hg hshore, hstart, hend, chart,
      WithLp.prodContinuousLinearEquiv_symm_apply, WithLp.toLp_snd, sub_zero] using
        (ACScalarVariation.integral_abs_deriv_eq_sub_iff_monotoneOn _ 0 1 zero_le_one hy).mpr hmono

end FractalSync.FlatShoreMinimizers
