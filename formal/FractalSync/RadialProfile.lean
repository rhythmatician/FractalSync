import FractalSync.DistanceGraph
import Mathlib.Analysis.SpecialFunctions.Arsinh
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus
import Mathlib.Topology.Order.MonotoneConvergence

/-!
# Analysis of the regularized radial profile

The profile is the actual interval integral appearing in Issue 160. These
results concern this scalar function. `IntrinsicShoreDistance` proves its
identification with the actual absolutely continuous Shore-path infimum;
general pairwise metric properties remain separate. The definition extends
to signed radii by the oriented interval integral.
-/

namespace FractalSync.DistanceGraph

open Filter
open scoped Topology

noncomputable def radialDensity (epsilon a r : ℝ) : ℝ :=
  Real.sqrt (epsilon ^ 2 + (1 + a ^ 2) * r ^ 2) / (epsilon ^ 2 + r ^ 2)

noncomputable def radialProfile (epsilon a r : ℝ) : ℝ :=
  ∫ t in (0 : ℝ)..r, radialDensity epsilon a t

/-- A real-valued closed primitive using only `arsinh` and logarithms.
The logarithm arguments are positive for every real `a` and `q`. -/
noncomputable def normalizedRadialPrimitive (a q : ℝ) : ℝ :=
  Real.sqrt (1 + a ^ 2) * Real.arsinh (Real.sqrt (1 + a ^ 2) * q) -
    a * Real.log (Real.sqrt (1 + (1 + a ^ 2) * q ^ 2) + a * q) +
    (a / 2) * Real.log (1 + q ^ 2)

theorem radial_denominator_pos (epsilon r : ℝ) (he : 0 < epsilon) :
    0 < epsilon ^ 2 + r ^ 2 := by
  positivity

theorem radial_radicand_pos (epsilon a r : ℝ) (he : 0 < epsilon) :
    0 < epsilon ^ 2 + (1 + a ^ 2) * r ^ 2 := by
  positivity

theorem radialDensity_pos (epsilon a r : ℝ) (he : 0 < epsilon) :
    0 < radialDensity epsilon a r := by
  exact div_pos (Real.sqrt_pos.2 (radial_radicand_pos epsilon a r he))
    (radial_denominator_pos epsilon r he)

theorem continuous_radialDensity (epsilon a : ℝ) (he : 0 < epsilon) :
    Continuous (radialDensity epsilon a) := by
  exact ((continuous_const.add
    (continuous_const.mul (continuous_id.pow 2))).sqrt).div
    (continuous_const.add (continuous_id.pow 2))
    (fun r => (radial_denominator_pos epsilon r he).ne')

theorem radialProfile_zero (epsilon a : ℝ) : radialProfile epsilon a 0 = 0 := by
  simp [radialProfile]

/-- The fundamental theorem of calculus supplies the actual derivative of F. -/
theorem hasDerivAt_radialProfile (epsilon a r : ℝ) (he : 0 < epsilon) :
    HasDerivAt (radialProfile epsilon a) (radialDensity epsilon a r) r := by
  exact ((continuous_radialDensity epsilon a he).integral_hasStrictDerivAt 0 r).hasDerivAt

theorem radialProfile_strictMono (epsilon a : ℝ) (he : 0 < epsilon) :
    StrictMono (radialProfile epsilon a) := by
  exact strictMono_of_hasDerivAt_pos
    (fun r => hasDerivAt_radialProfile epsilon a r he)
    (fun r => radialDensity_pos epsilon a r he)

theorem radialProfile_nonneg (epsilon a r : ℝ) (he : 0 < epsilon) (hr : 0 ≤ r) :
    0 ≤ radialProfile epsilon a r := by
  simpa only [radialProfile_zero] using (radialProfile_strictMono epsilon a he).monotone hr

theorem radialProfile_pos (epsilon a r : ℝ) (he : 0 < epsilon) (hr : 0 < r) :
    0 < radialProfile epsilon a r := by
  simpa only [radialProfile_zero] using radialProfile_strictMono epsilon a he hr

/-- Link the analytic radial density to the previously checked squared speed. -/
theorem radialDensity_sq (epsilon a r w : ℝ) (he : 0 < epsilon) :
    (radialDensity epsilon a r) ^ 2 * w ^ 2 = radialSpeedSq a epsilon r w := by
  simp only [radialDensity, radialSpeedSq, div_pow,
    Real.sq_sqrt (radial_radicand_pos epsilon a r he).le]
  ring

theorem radialDensity_scale (epsilon a r c : ℝ) (he : 0 < epsilon) (hc : 0 < c) :
    c * radialDensity (c * epsilon) a (c * r) = radialDensity epsilon a r := by
  unfold radialDensity
  rw [show (c * epsilon) ^ 2 + (1 + a ^ 2) * (c * r) ^ 2 =
      c ^ 2 * (epsilon ^ 2 + (1 + a ^ 2) * r ^ 2) by ring,
    show (c * epsilon) ^ 2 + (c * r) ^ 2 = c ^ 2 * (epsilon ^ 2 + r ^ 2) by ring,
    Real.sqrt_mul (sq_nonneg c), Real.sqrt_sq hc.le]
  field_simp

/-- Exact covariance under simultaneous dilation of radius and regularization. -/
theorem radialProfile_scale (epsilon a r c : ℝ) (he : 0 < epsilon) (hc : 0 < c) :
    radialProfile (c * epsilon) a (c * r) = radialProfile epsilon a r := by
  calc
    radialProfile (c * epsilon) a (c * r) =
        c * ∫ t in (0 : ℝ)..r, radialDensity (c * epsilon) a (c * t) := by
      simpa only [radialProfile, smul_eq_mul, mul_zero] using
        (intervalIntegral.smul_integral_comp_mul_left
          (a := (0 : ℝ)) (b := r) (radialDensity (c * epsilon) a) c).symm
    _ = ∫ t in (0 : ℝ)..r, c * radialDensity (c * epsilon) a (c * t) :=
      (intervalIntegral.integral_const_mul c _).symm
    _ = radialProfile epsilon a r := by
      apply intervalIntegral.integral_congr
      intro t _
      exact radialDensity_scale epsilon a t c he hc

theorem radialProfile_normalize (epsilon a r : ℝ) (he : 0 < epsilon) :
    radialProfile epsilon a r = radialProfile 1 a (r / epsilon) := by
  simpa only [mul_one, mul_div_cancel₀ r he.ne'] using
    radialProfile_scale 1 a (r / epsilon) epsilon zero_lt_one he

theorem radial_log_argument_pos (a q : ℝ) :
    0 < Real.sqrt (1 + (1 + a ^ 2) * q ^ 2) + a * q := by
  have h : -(a * q) < Real.sqrt (1 + (1 + a ^ 2) * q ^ 2) :=
    Real.lt_sqrt_of_sq_lt (by nlinarith [sq_nonneg q])
  linarith

theorem normalizedRadialPrimitive_zero (a : ℝ) : normalizedRadialPrimitive a 0 = 0 := by
  simp [normalizedRadialPrimitive]

/-- A checked derivative of the full closed primitive, for arbitrary `a`. -/
theorem hasDerivAt_normalizedRadialPrimitive (a q : ℝ) :
    HasDerivAt (normalizedRadialPrimitive a) (radialDensity 1 a q) q := by
  let b := Real.sqrt (1 + a ^ 2)
  let s := Real.sqrt (1 + (1 + a ^ 2) * q ^ 2)
  have hb : b ^ 2 = 1 + a ^ 2 := Real.sq_sqrt (by positivity)
  have hs : s ^ 2 = 1 + (1 + a ^ 2) * q ^ 2 := Real.sq_sqrt (by positivity)
  have hs_pos : 0 < s := Real.sqrt_pos.2 (by positivity)
  have ht_pos : 0 < 1 + q ^ 2 := by positivity
  have hlog_pos := radial_log_argument_pos a q
  have hroot : HasDerivAt (fun x : ℝ => Real.sqrt (1 + (1 + a ^ 2) * x ^ 2))
      ((1 + a ^ 2) * q / s) q := by
    convert ((((hasDerivAt_id q).pow 2).const_mul (1 + a ^ 2)).const_add 1).sqrt
      (ne_of_gt (show 0 < 1 + (1 + a ^ 2) * q ^ 2 by positivity)) using 1
    simp only [id_eq, Pi.pow_apply]
    dsimp [s]
    field_simp
  have harsinh : HasDerivAt (fun x : ℝ => b * Real.arsinh (b * x)) (b ^ 2 / s) q := by
    convert (((hasDerivAt_id q).const_mul b).arsinh.const_mul b) using 1
    simp only [id_eq, smul_eq_mul, mul_pow, hb, s]
    ring_nf
    rw [hb]
    ring
  have hlog : HasDerivAt
      (fun x : ℝ => Real.log (Real.sqrt (1 + (1 + a ^ 2) * x ^ 2) + a * x))
      (((1 + a ^ 2) * q / s + a) / (s + a * q)) q := by
    simpa only [id_eq, mul_one] using
      (hroot.add ((hasDerivAt_id q).const_mul a)).log hlog_pos.ne'
  have hlogT : HasDerivAt (fun x : ℝ => Real.log (1 + x ^ 2))
      (2 * q / (1 + q ^ 2)) q := by
    simpa using (((hasDerivAt_id q).pow 2).const_add 1).log ht_pos.ne'
  convert (harsinh.sub (hlog.const_mul a)).add (hlogT.const_mul (a / 2)) using 1
  simp only [radialDensity, one_pow]
  change s / (1 + q ^ 2) =
    b ^ 2 / s - a * (((1 + a ^ 2) * q / s + a) / (s + a * q)) +
      (a / 2) * (2 * q / (1 + q ^ 2))
  have hlog_ne : s + q * a ≠ 0 := by simpa only [s, mul_comm q a] using hlog_pos.ne'
  field_simp [hlog_ne]
  rw [hb]
  nlinarith [congrArg (fun z : ℝ => s * z) hs]

theorem radialProfile_one_eq_primitive (a r : ℝ) :
    radialProfile 1 a r = normalizedRadialPrimitive a r := by
  simpa only [normalizedRadialPrimitive_zero, sub_zero] using
    intervalIntegral.integral_eq_sub_of_hasDerivAt
      (fun x _ => hasDerivAt_normalizedRadialPrimitive a x)
      ((continuous_radialDensity 1 a zero_lt_one).intervalIntegrable 0 r)

/-- The closed primitive equals the interval integral for every signed radius.
This proves the scalar formula, not yet its interpretation as a graph distance. -/
theorem radialProfile_closed_form (epsilon a r : ℝ) (he : 0 < epsilon) :
    radialProfile epsilon a r = normalizedRadialPrimitive a (r / epsilon) := by
  rw [radialProfile_normalize epsilon a r he, radialProfile_one_eq_primitive]

theorem radialProfile_scale_only (epsilon r : ℝ) (he : 0 < epsilon) :
    radialProfile epsilon 0 r = Real.arsinh (r / epsilon) := by
  simpa [normalizedRadialPrimitive] using radialProfile_closed_form epsilon 0 r he

theorem radialDensity_comparison (epsilon a r : ℝ) (he : 0 < epsilon) :
    radialDensity epsilon 0 r ≤ radialDensity epsilon a r ∧
      radialDensity epsilon a r ≤ Real.sqrt (1 + a ^ 2) * radialDensity epsilon 0 r := by
  have hden := radial_denominator_pos epsilon r he
  simp only [radialDensity, zero_pow (by decide : (2 : ℕ) ≠ 0), add_zero, one_mul]
  constructor
  · apply (div_le_div_iff_of_pos_right hden).2
    exact Real.sqrt_le_sqrt (by nlinarith [mul_nonneg (sq_nonneg a) (sq_nonneg r)])
  · rw [← mul_div_assoc]
    apply (div_le_div_iff_of_pos_right hden).2
    rw [← Real.sqrt_mul (show 0 ≤ 1 + a ^ 2 by positivity)]
    exact Real.sqrt_le_sqrt (by nlinarith [mul_nonneg (sq_nonneg a) (sq_nonneg epsilon)])

/-- The pointwise comparison is integrated, giving an actual profile bound. -/
theorem radialProfile_asinh_bounds (epsilon a r : ℝ) (he : 0 < epsilon) (hr : 0 ≤ r) :
    Real.arsinh (r / epsilon) ≤ radialProfile epsilon a r ∧
      radialProfile epsilon a r ≤ Real.sqrt (1 + a ^ 2) * Real.arsinh (r / epsilon) := by
  rw [← radialProfile_scale_only epsilon r he]
  constructor
  · exact intervalIntegral.integral_mono hr
      ((continuous_radialDensity epsilon 0 he).intervalIntegrable 0 r)
      ((continuous_radialDensity epsilon a he).intervalIntegrable 0 r)
      (fun t => (radialDensity_comparison epsilon a t he).1)
  · rw [radialProfile, radialProfile, ← intervalIntegral.integral_const_mul]
    exact intervalIntegral.integral_mono hr
      ((continuous_radialDensity epsilon a he).intervalIntegrable 0 r)
      ((continuous_const.mul (continuous_radialDensity epsilon 0 he)).intervalIntegrable 0 r)
      (fun t => (radialDensity_comparison epsilon a t he).2)

/-- For `a² ≤ 1`, the normalized profile has maximal slope one at the Shore. -/
theorem radialDensity_le_one (a q : ℝ) (ha : a ^ 2 ≤ 1) :
    radialDensity 1 a q ≤ 1 := by
  unfold radialDensity
  simp only [one_pow]
  apply (div_le_iff₀ (show 0 < 1 + q ^ 2 by positivity)).2
  simp only [one_mul]
  apply Real.sqrt_le_iff.2
  constructor
  · positivity
  · nlinarith [sq_nonneg (q ^ 2), mul_nonneg (sub_nonneg.mpr ha) (sq_nonneg q)]

/-- The completed-square certificate for the other sharp slope bound. -/
theorem radial_peak_remainder (a q : ℝ) :
    (1 + a ^ 2) ^ 2 * (1 + q ^ 2) ^ 2 -
      4 * a ^ 2 * (1 + (1 + a ^ 2) * q ^ 2) =
      ((1 + a ^ 2) * q ^ 2 - (a ^ 2 - 1)) ^ 2 := by
  ring

theorem radialDensity_le_peak (a q : ℝ) (ha : 0 < a) :
    radialDensity 1 a q ≤ (1 + a ^ 2) / (2 * a) := by
  let s := Real.sqrt (1 + (1 + a ^ 2) * q ^ 2)
  have hs : 0 ≤ s := Real.sqrt_nonneg _
  have hs2 : s ^ 2 = 1 + (1 + a ^ 2) * q ^ 2 := Real.sq_sqrt (by positivity)
  have hprod : 0 ≤ (1 + a ^ 2) * (1 + q ^ 2) := by positivity
  have hbound : 2 * a * s ≤ (1 + a ^ 2) * (1 + q ^ 2) := by
    nlinarith [sq_nonneg ((1 + a ^ 2) * q ^ 2 - (a ^ 2 - 1)),
      radial_peak_remainder a q]
  simp only [radialDensity, one_pow]
  change s / (1 + q ^ 2) ≤ (1 + a ^ 2) / (2 * a)
  apply (div_le_div_iff₀ (by positivity) (by positivity)).2
  nlinarith

theorem radialDensity_one_at_zero (a : ℝ) : radialDensity 1 a 0 = 1 := by
  simp [radialDensity]

/-- The bound `(1+a²)/(2a)` is attained when `a ≥ 1`. -/
theorem radialDensity_peak_attained (a : ℝ) (ha : 1 ≤ a) :
    radialDensity 1 a (Real.sqrt ((a ^ 2 - 1) / (1 + a ^ 2))) =
      (1 + a ^ 2) / (2 * a) := by
  have ha_pos : 0 < a := lt_of_lt_of_le zero_lt_one ha
  have hC_pos : 0 < 1 + a ^ 2 := by positivity
  have hq_nonneg : 0 ≤ (a ^ 2 - 1) / (1 + a ^ 2) :=
    div_nonneg (by nlinarith) hC_pos.le
  simp only [radialDensity, one_pow, Real.sq_sqrt hq_nonneg]
  rw [show 1 + (1 + a ^ 2) * ((a ^ 2 - 1) / (1 + a ^ 2)) = a ^ 2 by
    field_simp
    ring]
  rw [Real.sqrt_sq ha_pos.le]
  field_simp [ha_pos.ne', hC_pos.ne']
  rw [show 1 + a ^ 2 + (a ^ 2 - 1) = a ^ 2 * 2 by ring, div_self (by positivity)]

/-- The scalar profile is unbounded for each positive regularization floor. -/
theorem radialProfile_tendsto_atTop (epsilon a : ℝ) (he : 0 < epsilon) :
    Tendsto (radialProfile epsilon a) atTop atTop := by
  have h_arsinh : Tendsto Real.arsinh atTop atTop := Real.sinhOrderIso.symm.tendsto_atTop
  refine tendsto_atTop_mono' atTop ?_ (h_arsinh.comp (tendsto_id.atTop_div_const he))
  filter_upwards [eventually_ge_atTop (0 : ℝ)] with r hr
  exact (radialProfile_asinh_bounds epsilon a r he hr).1

/-- The scalar distance-to-Shore candidate diverges as the floor vanishes,
at every fixed positive Euclidean distance. -/
theorem radialProfile_tendsto_zero_floor (a r : ℝ) (hr : 0 < r) :
    Tendsto (fun epsilon : ℝ => radialProfile epsilon a r) (𝓝[>] (0 : ℝ)) atTop := by
  have hq : Tendsto (fun epsilon : ℝ => r / epsilon) (𝓝[>] (0 : ℝ)) atTop := by
    simpa only [div_eq_mul_inv] using tendsto_inv_nhdsGT_zero.const_mul_atTop hr
  refine ((radialProfile_tendsto_atTop 1 a zero_lt_one).comp hq).congr' ?_
  filter_upwards [self_mem_nhdsWithin] with epsilon he
  exact (radialProfile_normalize epsilon a r he).symm

noncomputable def radialDefect (a q : ℝ) : ℝ :=
  Real.sqrt (1 + a ^ 2) * Real.arsinh q - radialProfile 1 a q

noncomputable def radialDefectLimit (a : ℝ) : ℝ :=
  a * Real.arsinh a - Real.sqrt (1 + a ^ 2) * Real.log (Real.sqrt (1 + a ^ 2))

/-- Replacing `q` by `1/u` removes the cancelling divergent logarithms. -/
noncomputable def compactRadialDefect (a u : ℝ) : ℝ :=
  Real.sqrt (1 + a ^ 2) * Real.log (1 + Real.sqrt (1 + u ^ 2)) -
    Real.sqrt (1 + a ^ 2) *
      Real.log (Real.sqrt (1 + a ^ 2) + Real.sqrt (1 + a ^ 2 + u ^ 2)) +
    a * Real.log (a + Real.sqrt (1 + a ^ 2 + u ^ 2)) -
    (a / 2) * Real.log (1 + u ^ 2)

theorem sqrt_one_add_mul_sq_scale (C q : ℝ) (hq : 0 < q) :
    Real.sqrt (1 + C * q ^ 2) = q * Real.sqrt (C + q⁻¹ ^ 2) := by
  calc
    Real.sqrt (1 + C * q ^ 2) = Real.sqrt (q ^ 2 * (C + q⁻¹ ^ 2)) := by
      congr 1
      field_simp
      ring
    _ = Real.sqrt (q ^ 2) * Real.sqrt (C + q⁻¹ ^ 2) := Real.sqrt_mul (sq_nonneg q) _
    _ = q * Real.sqrt (C + q⁻¹ ^ 2) := by rw [Real.sqrt_sq hq.le]

theorem radialDefect_compact (a q : ℝ) (hq : 0 < q) :
    radialDefect a q = compactRadialDefect a q⁻¹ := by
  simp only [radialDefect, radialProfile_one_eq_primitive, normalizedRadialPrimitive,
    Real.arsinh, mul_pow, Real.sq_sqrt (show 0 ≤ 1 + a ^ 2 by positivity)]
  have hT : Real.sqrt (1 + q ^ 2) = q * Real.sqrt (1 + q⁻¹ ^ 2) := by
    simpa only [one_mul] using sqrt_one_add_mul_sq_scale 1 q hq
  have hS := sqrt_one_add_mul_sq_scale (1 + a ^ 2) q hq
  have hden : 1 + q ^ 2 = q ^ 2 * (1 + q⁻¹ ^ 2) := by
    field_simp
    ring
  have hP : 0 < 1 + Real.sqrt (1 + q⁻¹ ^ 2) := by positivity
  have hQ : 0 < Real.sqrt (1 + a ^ 2) + Real.sqrt (1 + a ^ 2 + q⁻¹ ^ 2) := by positivity
  have hA : 0 < a + Real.sqrt (1 + a ^ 2 + q⁻¹ ^ 2) := by
    have h : -a < Real.sqrt (1 + a ^ 2 + q⁻¹ ^ 2) :=
      Real.lt_sqrt_of_sq_lt (by nlinarith [sq_nonneg q⁻¹])
    linarith
  have hU : 0 < 1 + q⁻¹ ^ 2 := by positivity
  rw [hT, hS,
    show q + q * Real.sqrt (1 + q⁻¹ ^ 2) =
      q * (1 + Real.sqrt (1 + q⁻¹ ^ 2)) by ring,
    show Real.sqrt (1 + a ^ 2) * q + q * Real.sqrt (1 + a ^ 2 + q⁻¹ ^ 2) =
      q * (Real.sqrt (1 + a ^ 2) + Real.sqrt (1 + a ^ 2 + q⁻¹ ^ 2)) by ring,
    show q * Real.sqrt (1 + a ^ 2 + q⁻¹ ^ 2) + a * q =
      q * (a + Real.sqrt (1 + a ^ 2 + q⁻¹ ^ 2)) by ring,
    hden, Real.log_mul hq.ne' hP.ne', Real.log_mul hq.ne' hQ.ne',
    Real.log_mul hq.ne' hA.ne', Real.log_mul (pow_ne_zero 2 hq.ne') hU.ne', Real.log_pow]
  dsimp [compactRadialDefect]
  ring

theorem continuous_compactRadialDefect (a : ℝ) : Continuous (compactRadialDefect a) := by
  have hS : Continuous (fun u : ℝ => Real.sqrt (1 + a ^ 2 + u ^ 2)) :=
    (continuous_const.add (continuous_id.pow 2)).sqrt
  have hP : Continuous (fun u : ℝ => Real.sqrt (1 + u ^ 2)) :=
    (continuous_const.add (continuous_id.pow 2)).sqrt
  have h1 : Continuous (fun u : ℝ => Real.log (1 + Real.sqrt (1 + u ^ 2))) :=
    (continuous_const.add hP).log (fun _ => ne_of_gt (by positivity))
  have h2 : Continuous (fun u : ℝ =>
      Real.log (Real.sqrt (1 + a ^ 2) + Real.sqrt (1 + a ^ 2 + u ^ 2))) :=
    (continuous_const.add hS).log (fun _ => ne_of_gt (by positivity))
  have h3 : Continuous (fun u : ℝ => Real.log (a + Real.sqrt (1 + a ^ 2 + u ^ 2))) := by
    apply (continuous_const.add hS).log
    intro u
    have h : -a < Real.sqrt (1 + a ^ 2 + u ^ 2) :=
      Real.lt_sqrt_of_sq_lt (by nlinarith [sq_nonneg u])
    linarith
  have h4 : Continuous (fun u : ℝ => Real.log (1 + u ^ 2)) :=
    (continuous_const.add (continuous_id.pow 2)).log (fun _ => ne_of_gt (by positivity))
  exact (((continuous_const.mul h1).sub (continuous_const.mul h2)).add
    (continuous_const.mul h3)).sub (continuous_const.mul h4)

theorem compactRadialDefect_zero (a : ℝ) : compactRadialDefect a 0 = radialDefectLimit a := by
  simp only [compactRadialDefect, radialDefectLimit, zero_pow (by decide : (2 : ℕ) ≠ 0),
    add_zero, Real.sqrt_one, Real.log_one, mul_zero, sub_zero, Real.arsinh]
  rw [show Real.sqrt (1 + a ^ 2) + Real.sqrt (1 + a ^ 2) =
      2 * Real.sqrt (1 + a ^ 2) by ring,
    Real.log_mul (by norm_num : (2 : ℝ) ≠ 0)
      (ne_of_gt (Real.sqrt_pos.2 (show 0 < 1 + a ^ 2 by positivity)))]
  norm_num
  ring

/-- The additive error tends to the claimed sharp constant. -/
theorem radialDefect_tendsto (a : ℝ) :
    Tendsto (radialDefect a) atTop (𝓝 (radialDefectLimit a)) := by
  have h := (continuous_compactRadialDefect a).continuousAt.tendsto.comp
    (tendsto_inv_atTop_zero : Tendsto (fun q : ℝ => q⁻¹) atTop (𝓝 0))
  rw [compactRadialDefect_zero] at h
  refine h.congr' ?_
  filter_upwards [eventually_gt_atTop (0 : ℝ)] with q hq
  exact (radialDefect_compact a q hq).symm

theorem hasDerivAt_radialDefect (a q : ℝ) :
    HasDerivAt (radialDefect a)
      (Real.sqrt (1 + a ^ 2) * radialDensity 1 0 q - radialDensity 1 a q) q := by
  have h0 : HasDerivAt Real.arsinh (radialDensity 1 0 q) q := by
    have hfun : normalizedRadialPrimitive 0 = Real.arsinh := by
      funext x
      simp [normalizedRadialPrimitive]
    simpa only [hfun] using hasDerivAt_normalizedRadialPrimitive 0 q
  exact (h0.const_mul (Real.sqrt (1 + a ^ 2))).sub
    (hasDerivAt_radialProfile 1 a q zero_lt_one)

theorem radialDefect_monotone (a : ℝ) : Monotone (radialDefect a) := by
  exact monotone_of_hasDerivAt_nonneg (hasDerivAt_radialDefect a)
    (fun q => sub_nonneg.mpr (radialDensity_comparison 1 a q zero_lt_one).2)

theorem radialDefect_zero (a : ℝ) : radialDefect a 0 = 0 := by
  simp [radialDefect, radialProfile_zero]

theorem radialDefect_bounds (a q : ℝ) (hq : 0 ≤ q) :
    0 ≤ radialDefect a q ∧ radialDefect a q ≤ radialDefectLimit a := by
  constructor
  · simpa only [radialDefect_zero] using radialDefect_monotone a hq
  · exact (radialDefect_monotone a).ge_of_tendsto (radialDefect_tendsto a) q

theorem radialDefectLimit_nonneg (a : ℝ) : 0 ≤ radialDefectLimit a := by
  simpa only [radialDefect_zero] using (radialDefect_bounds a 0 le_rfl).2

/-- The complete sharp additive estimate for the original dimensional profile. -/
theorem radialProfile_sharp_bounds (epsilon a r : ℝ) (he : 0 < epsilon) (hr : 0 ≤ r) :
    0 ≤ Real.sqrt (1 + a ^ 2) * Real.arsinh (r / epsilon) - radialProfile epsilon a r ∧
      Real.sqrt (1 + a ^ 2) * Real.arsinh (r / epsilon) - radialProfile epsilon a r ≤
        radialDefectLimit a := by
  rw [radialProfile_normalize epsilon a r he]
  exact radialDefect_bounds a (r / epsilon) (div_nonneg hr he.le)

/-- No smaller uniform additive bound works even for the normalized profile. -/
theorem radialDefectLimit_le_of_uniform_bound (a C : ℝ)
    (hC : ∀ q : ℝ, 0 ≤ q → radialDefect a q ≤ C) : radialDefectLimit a ≤ C := by
  apply le_of_tendsto (radialDefect_tendsto a)
  filter_upwards [eventually_ge_atTop (0 : ℝ)] with q hq
  exact hC q hq

theorem arsinh_sub_log_tendsto :
    Tendsto (fun q : ℝ => Real.arsinh q - Real.log q) atTop (𝓝 (Real.log 2)) := by
  have hcont : Continuous (fun u : ℝ => Real.log (1 + Real.sqrt (1 + u ^ 2))) :=
    (continuous_const.add ((continuous_const.add (continuous_id.pow 2)).sqrt)).log
      (fun _ => ne_of_gt (by positivity))
  have h : Tendsto (fun q : ℝ => Real.log (1 + Real.sqrt (1 + q⁻¹ ^ 2)))
      atTop (𝓝 (Real.log 2)) := by
    convert hcont.continuousAt.tendsto.comp
      (tendsto_inv_atTop_zero : Tendsto (fun q : ℝ => q⁻¹) atTop (𝓝 0)) using 1
    norm_num [Function.comp_def]
  refine h.congr' ?_
  filter_upwards [eventually_gt_atTop (0 : ℝ)] with q hq
  have hT : Real.sqrt (1 + q ^ 2) = q * Real.sqrt (1 + q⁻¹ ^ 2) := by
    simpa only [one_mul] using sqrt_one_add_mul_sq_scale 1 q hq
  rw [Real.arsinh, hT, show q + q * Real.sqrt (1 + q⁻¹ ^ 2) =
      q * (1 + Real.sqrt (1 + q⁻¹ ^ 2)) by ring,
    Real.log_mul hq.ne' (ne_of_gt (show 0 < 1 + Real.sqrt (1 + q⁻¹ ^ 2) by positivity))]
  ring

/-- Exact constant in the logarithmic asymptotic, expressed using the sharp defect. -/
theorem radialProfile_one_log_asymptotic (a : ℝ) :
    Tendsto (fun q : ℝ => radialProfile 1 a q - Real.sqrt (1 + a ^ 2) * Real.log q)
      atTop (𝓝 (Real.sqrt (1 + a ^ 2) * Real.log 2 - radialDefectLimit a)) := by
  have h := (arsinh_sub_log_tendsto.const_mul (Real.sqrt (1 + a ^ 2))).sub
    (radialDefect_tendsto a)
  convert h using 1
  funext q
  simp only [radialDefect]
  ring

theorem radialProfile_log_asymptotic (epsilon a : ℝ) (he : 0 < epsilon) :
    Tendsto (fun r : ℝ =>
      radialProfile epsilon a r - Real.sqrt (1 + a ^ 2) * Real.log (r / epsilon))
      atTop (𝓝 (Real.sqrt (1 + a ^ 2) * Real.log 2 - radialDefectLimit a)) := by
  refine ((radialProfile_one_log_asymptotic a).comp (tendsto_id.atTop_div_const he)).congr ?_
  intro r
  rw [radialProfile_normalize epsilon a r he]
  rfl

theorem radialProfile_zero_floor_log_asymptotic (a r : ℝ) (hr : 0 < r) :
    Tendsto (fun epsilon : ℝ =>
      radialProfile epsilon a r - Real.sqrt (1 + a ^ 2) * Real.log (r / epsilon))
      (𝓝[>] (0 : ℝ)) (𝓝 (Real.sqrt (1 + a ^ 2) * Real.log 2 - radialDefectLimit a)) := by
  have hq : Tendsto (fun epsilon : ℝ => r / epsilon) (𝓝[>] (0 : ℝ)) atTop := by
    simpa only [div_eq_mul_inv] using tendsto_inv_nhdsGT_zero.const_mul_atTop hr
  refine ((radialProfile_one_log_asymptotic a).comp hq).congr' ?_
  filter_upwards [self_mem_nhdsWithin] with epsilon he
  rw [radialProfile_normalize epsilon a r he]
  rfl

end FractalSync.DistanceGraph
