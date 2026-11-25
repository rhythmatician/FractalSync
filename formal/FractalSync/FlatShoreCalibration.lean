import FractalSync.FlatShoreShortcuts
import Mathlib.Analysis.SpecialFunctions.Trigonometric.InverseDeriv

/-!
# A global spherical calibration for flat-Shore paths

Smooth contractions of an angular coordinate supply endpoint estimates on
the full absolutely continuous path class. The limiting calibration includes
the antipodal threshold without differentiating arccos at its endpoints.
-/

namespace FractalSync.FlatShoreCalibration

open Set MeasureTheory Filter FlatShoreIntrinsic DistanceGraph
open scoped Topology

abbrev Coord := ℝ × ℝ

/-- Smooth angular calibration when `0 < c < 1`. -/
noncomputable def calibration (c : ℝ) (x : Coord) : ℝ :=
  Real.arccos (c * Real.cos x.2 / ruler 1 x.1)

/-- The literal differential of the angular calibration. -/
noncomputable def calibrationDifferential (c : ℝ) (x : Coord) : Coord →L[ℝ] ℝ :=
  (c / ((1 + x.1 ^ 2) * Real.sqrt (1 + x.1 ^ 2 - c ^ 2 * Real.cos x.2 ^ 2))) •
    ((x.1 * Real.cos x.2) • ContinuousLinearMap.fst ℝ ℝ ℝ +
      ((1 + x.1 ^ 2) * Real.sin x.2) • ContinuousLinearMap.snd ℝ ℝ ℝ)

/-- Speed in the comparison sphere's latitude-longitude pullback. -/
noncomputable def sphericalSpeed (r : ℝ) (v : Coord) : ℝ :=
  Real.sqrt (v.1 ^ 2 / (1 + r ^ 2) ^ 2 + v.2 ^ 2 / (1 + r ^ 2))

/-- Its actual integral on an ordinary Euclidean absolutely continuous path. -/
noncomputable def sphericalLength (gamma : ℝ → Plane) : ℝ :=
  ∫ t in (0 : ℝ)..1, sphericalSpeed (gamma t).1 (chart (deriv gamma t))

theorem calibration_argument_mem (c : ℝ) (hc : c ∈ Ioo (0 : ℝ) 1) (x : Coord) :
    c * Real.cos x.2 / ruler 1 x.1 ∈ Ioo (-1 : ℝ) 1 := by
  have hbound : |c * Real.cos x.2 / ruler 1 x.1| ≤ c := by
    rw [abs_div, abs_mul, abs_of_pos hc.1, abs_of_pos (ruler_pos 1 x.1 (by norm_num))]
    exact (div_le_iff₀ (ruler_pos 1 x.1 (by norm_num))).2
      (mul_le_mul_of_nonneg_left
        ((Real.abs_cos_le_one _).trans (epsilon_le_ruler 1 x.1 (by norm_num))) hc.1.le)
  exact abs_lt.mp (hbound.trans_lt hc.2)

theorem calibration_sqrt_denominator (c r y : ℝ) :
    Real.sqrt (1 - (c * Real.cos y / ruler 1 r) ^ 2) =
      Real.sqrt (1 + r ^ 2 - c ^ 2 * Real.cos y ^ 2) / ruler 1 r := by
  have heq : 1 - (c * Real.cos y / ruler 1 r) ^ 2 =
      (1 + r ^ 2 - c ^ 2 * Real.cos y ^ 2) / (ruler 1 r) ^ 2 := by
    field_simp [(ruler_pos 1 r (by norm_num)).ne']
    simp [ruler_sq, add_comm]
  rw [heq, Real.sqrt_div' _ (sq_nonneg _), Real.sqrt_sq (ruler_pos 1 r (by norm_num)).le]

theorem calibration_denominator_pos (c : ℝ) (hc : c ∈ Ioo (0 : ℝ) 1) (x : Coord) :
    0 < 1 + x.1 ^ 2 - c ^ 2 * Real.cos x.2 ^ 2 := by
  exact sub_pos.mpr ((mul_le_of_le_one_right (sq_nonneg c)
    (by nlinarith [Real.sin_sq_add_cos_sq x.2, sq_nonneg (Real.sin x.2)])).trans_lt
      ((show c ^ 2 < 1 from by nlinarith [hc.1, hc.2]).trans_le
        (by nlinarith [sq_nonneg x.1])))

theorem hasFDerivAt_calibration (c : ℝ) (hc : c ∈ Ioo (0 : ℝ) 1) (x : Coord) :
    HasFDerivAt (calibration c) (calibrationDifferential c x) x := by
  have hrho : 0 < ruler 1 x.1 := ruler_pos 1 x.1 (by norm_num)
  have harg := calibration_argument_mem c hc x
  have hdiff := (Real.hasDerivAt_arccos harg.1.ne' harg.2.ne).comp_hasFDerivAt x
    ((((ContinuousLinearMap.snd ℝ ℝ ℝ).hasFDerivAt.cos).const_mul c).mul
      (((hasDerivAt_ruler 1 x.1 (by norm_num)).inv hrho.ne').comp_hasFDerivAt x
        (ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt))
  convert hdiff using 1
  refine ContinuousLinearMap.ext (fun v => ?_)
  simp only [calibrationDifferential, ContinuousLinearMap.smul_apply,
    ContinuousLinearMap.add_apply,
    smul_eq_mul, Function.comp_apply, Pi.inv_apply, calibration_sqrt_denominator]
  change c / ((1 + x.1 ^ 2) * Real.sqrt (1 + x.1 ^ 2 - c ^ 2 * Real.cos x.2 ^ 2)) *
      (x.1 * Real.cos x.2 * v.1 + (1 + x.1 ^ 2) * Real.sin x.2 * v.2) =
    -(1 / (Real.sqrt (1 + x.1 ^ 2 - c ^ 2 * Real.cos x.2 ^ 2) / ruler 1 x.1)) *
      (c * Real.cos x.2 * (-(x.1 / ruler 1 x.1) / ruler 1 x.1 ^ 2 * v.1) +
        (ruler 1 x.1)⁻¹ * (c * (-Real.sin x.2 * v.2)))
  field_simp [hrho.ne', (Real.sqrt_pos.mpr (calibration_denominator_pos c hc x)).ne',
    show 1 + x.1 ^ 2 ≠ 0 by positivity]
  rw [ruler_sq]
  ring

theorem calibrationDifferential_le_sphericalSpeed (c : ℝ)
    (hc : c ∈ Ioo (0 : ℝ) 1) (x v : Coord) :
    |calibrationDifferential c x v| ≤ sphericalSpeed x.1 v := by
  apply Real.abs_le_sqrt
  change (c / ((1 + x.1 ^ 2) * Real.sqrt (1 + x.1 ^ 2 - c ^ 2 * Real.cos x.2 ^ 2)) *
      (x.1 * Real.cos x.2 * v.1 + (1 + x.1 ^ 2) * Real.sin x.2 * v.2)) ^ 2 ≤ _
  rw [mul_pow, div_pow, mul_pow, Real.sq_sqrt (calibration_denominator_pos c hc x).le]
  have hD := calibration_denominator_pos c hc x
  field_simp
  have hidentity :
      (1 + x.1 ^ 2 - c ^ 2 * Real.cos x.2 ^ 2) * (v.1 ^ 2 + (1 + x.1 ^ 2) * v.2 ^ 2) -
        c ^ 2 * (x.1 * Real.cos x.2 * v.1 + (1 + x.1 ^ 2) * Real.sin x.2 * v.2) ^ 2 =
      (1 + x.1 ^ 2) * (1 - c ^ 2) * (v.1 ^ 2 + (1 + x.1 ^ 2) * v.2 ^ 2) +
        c ^ 2 * (1 + x.1 ^ 2) * (x.1 * Real.cos x.2 * v.2 - Real.sin x.2 * v.1) ^ 2 := by
    linear_combination -(c ^ 2 * (1 + x.1 ^ 2) *
      (v.1 ^ 2 + (1 + x.1 ^ 2) * v.2 ^ 2)) * Real.sin_sq_add_cos_sq x.2
  have hc2 : 0 ≤ 1 - c ^ 2 := by nlinarith [hc.1, hc.2]
  exact sub_nonneg.mp (hidentity.symm ▸ (by positivity))

theorem sphericalSpeed_le_norm (r : ℝ) (v : Coord) :
    sphericalSpeed r v ≤ 2 * ‖v‖ := by
  apply (Real.sqrt_le_left (by positivity : 0 ≤ 2 * ‖v‖)).2
  refine (add_le_add (div_le_self (sq_nonneg v.1) (by nlinarith [sq_nonneg r]))
    (div_le_self (sq_nonneg v.2) (by nlinarith [sq_nonneg r]))).trans ?_
  nlinarith [show v.1 ^ 2 ≤ ‖v‖ ^ 2 from by
    simpa only [Real.norm_eq_abs, sq_abs] using
      (sq_le_sq₀ (norm_nonneg v.1) (norm_nonneg v)).2 (norm_fst_le v),
    show v.2 ^ 2 ≤ ‖v‖ ^ 2 from by
    simpa only [Real.norm_eq_abs, sq_abs] using
      (sq_le_sq₀ (norm_nonneg v.2) (norm_nonneg v)).2 (norm_snd_le v)]

theorem calibration_lipschitz (c : ℝ) (hc : c ∈ Ioo (0 : ℝ) 1) :
    LipschitzWith 2 (calibration c) := by
  apply lipschitzWith_of_nnnorm_fderiv_le
    (fun x => (hasFDerivAt_calibration c hc x).differentiableAt)
  intro x
  change ‖fderiv ℝ (calibration c) x‖ ≤ 2
  rw [(hasFDerivAt_calibration c hc x).fderiv]
  exact ContinuousLinearMap.opNorm_le_bound _ (by norm_num) (fun v => by
    simpa only [Real.norm_eq_abs] using
      (calibrationDifferential_le_sphericalSpeed c hc x v).trans (sphericalSpeed_le_norm x.1 v))

theorem sphericalSpeed_le_coordinateSpeed (a r : ℝ) (v : Coord) :
    sphericalSpeed r v ≤ Real.sqrt (FlatShoreGeometry.metric 1 a r v v) := by
  apply Real.sqrt_le_sqrt
  simp only [FlatShoreGeometry.metric, FlatShoreGeometry.radialMetric,
    FlatShoreGeometry.tangentialMetric, one_pow]
  field_simp
  nlinarith [sq_nonneg (r * v.1), sq_nonneg (a * r * v.1)]

theorem sphericalSpeed_intervalIntegrable (gamma : ℝ → Plane)
    (hg : AbsolutelyContinuousOnInterval gamma 0 1) :
    IntervalIntegrable (fun t => sphericalSpeed (gamma t).1 (chart (deriv gamma t)))
      volume 0 1 := by
  have hcont : Continuous (fun p : Coord × Coord => sphericalSpeed p.1.1 p.2) := by
    unfold sphericalSpeed
    fun_prop (disch := intro x; positivity)
  apply (coordinateSpeed_intervalIntegrable 1 0 0 1 (by norm_num) gamma hg).mono_fun'
  · have hgmeas : AEStronglyMeasurable gamma (volume.restrict (uIoc (0 : ℝ) 1)) :=
      hg.continuousOn.intervalIntegrable.def'.aestronglyMeasurable
    simpa only [chart, WithLp.prodContinuousLinearEquiv_apply, WithLp.ofLp_fst] using
      hcont.comp_aestronglyMeasurable
      ((chart.continuous.comp_aestronglyMeasurable hgmeas).prodMk
        (chart.continuous.comp_aestronglyMeasurable (aestronglyMeasurable_deriv gamma _)))
  · exact Eventually.of_forall (fun t => by
      simpa only [Real.norm_eq_abs, sphericalSpeed, abs_of_nonneg (Real.sqrt_nonneg _)] using
        sphericalSpeed_le_coordinateSpeed 0 (gamma t).1 (chart (deriv gamma t)))

theorem calibration_endpoint_bound (c : ℝ) (hc : c ∈ Ioo (0 : ℝ) 1)
    (gamma : ℝ → Plane) (hg : AbsolutelyContinuousOnInterval gamma 0 1) :
    calibration c (chart (gamma 1)) - calibration c (chart (gamma 0)) ≤
      sphericalLength gamma := by
  have hcal := (calibration_lipschitz c hc).comp_absolutelyContinuousOnInterval
    (chart.lipschitz.comp_absolutelyContinuousOnInterval hg)
  simp only [Function.comp_def] at hcal
  rw [← hcal.integral_deriv_eq_sub]
  apply intervalIntegral.integral_mono_ae_restrict zero_le_one hcal.intervalIntegrable_deriv
    (sphericalSpeed_intervalIntegrable gamma hg)
  rw [EventuallyLE, ae_restrict_iff' measurableSet_Icc]
  filter_upwards [hg.ae_differentiableAt] with t ht hmem
  have hd := (hasFDerivAt_calibration c hc (chart (gamma t))).comp_hasDerivAt t
    (chart.hasFDerivAt.comp_hasDerivAt t
      (ht (by simpa only [uIcc_of_le zero_le_one] using hmem)).hasDerivAt)
  rw [show deriv (fun x => calibration c (chart (gamma x))) t =
    calibrationDifferential c (chart (gamma t)) (chart (deriv gamma t)) from hd.deriv]
  exact (le_abs_self _).trans
    (calibrationDifferential_le_sphericalSpeed c hc (chart (gamma t)) (chart (deriv gamma t)))

/-- Every AC path between Shore points separated by at most pi has at least
that much comparison length, including the antipodal endpoint. -/
theorem shore_span_le_sphericalLength (ell : ℝ) (hell : ell ∈ Icc (0 : ℝ) Real.pi)
    (gamma : ℝ → Plane) (hg : AbsolutelyContinuousOnInterval gamma 0 1)
    (hstart : gamma 0 = chart.symm (0, 0)) (hend : gamma 1 = chart.symm (0, ell)) :
    ell ≤ sphericalLength gamma := by
  have hclosed : IsClosed {c : ℝ | calibration c (chart (gamma 1)) -
      calibration c (chart (gamma 0)) ≤ sphericalLength gamma} := by
    exact isClosed_le
      (((continuous_id.mul continuous_const).div_const _).arccos.sub
        ((continuous_id.mul continuous_const).div_const _).arccos) continuous_const
  have hlimit := closure_minimal (fun c (hc : c ∈ Ioo (0 : ℝ) 1) =>
    calibration_endpoint_bound c hc gamma hg) hclosed
      (show (1 : ℝ) ∈ closure (Ioo (0 : ℝ) 1) by
        simp [closure_Ioo (by norm_num : (0 : ℝ) ≠ 1)])
  change calibration 1 (chart (gamma 1)) - calibration 1 (chart (gamma 0)) ≤
    sphericalLength gamma at hlimit
  simpa [hstart, hend, calibration, ruler, Real.arccos_cos hell.1 hell.2] using hlimit

/-- The comparison contraction bounds its actual length by graph length. -/
theorem sphericalLength_le_acGraphLength (a : ℝ) (gamma : ℝ → Plane)
    (hg : AbsolutelyContinuousOnInterval gamma 0 1) :
    sphericalLength gamma ≤ acGraphLength shore 1 a gamma := by
  rw [← coordinateLength_eq_acGraphLength 1 a (by norm_num) gamma hg]
  exact intervalIntegral.integral_mono_on zero_le_one (sphericalSpeed_intervalIntegrable gamma hg)
    (coordinateSpeed_intervalIntegrable 1 a 0 1 (by norm_num) gamma hg)
      (fun t _ => sphericalSpeed_le_coordinateSpeed a (gamma t).1 (chart (deriv gamma t)))

/-- The Shore span is a global lower bound on every admissible AC length. -/
theorem shore_span_le_acGraphLength (ell a : ℝ) (hell : ell ∈ Icc (0 : ℝ) Real.pi)
    (gamma : ℝ → Plane) (hg : AbsolutelyContinuousOnInterval gamma 0 1)
    (hstart : gamma 0 = chart.symm (0, 0)) (hend : gamma 1 = chart.symm (0, ell)) :
    ell ≤ acGraphLength shore 1 a gamma := by
  exact (shore_span_le_sphericalLength ell hell gamma hg hstart hend).trans
    (sphericalLength_le_acGraphLength a gamma hg)

end FractalSync.FlatShoreCalibration
