import FractalSync.Analysis.CompactParametricIntegral
import FractalSync.FlatShoreGeodesics

/-!
# Actual sine-graph second variation in the flat Shore metric

The functional is an interval integral of the actual flat graph density.
Parameter derivatives are derived from that density and integrated with a
compact domination argument; no functional variation formula is assumed.
-/

namespace FractalSync.FlatShoreSecondVariation

open Set Filter
open scoped Topology

noncomputable def modeRadicand (a k A t : ℝ) : ℝ :=
  1 + A ^ 2 * (Real.sin t ^ 2 + k ^ 2 * Real.cos t ^ 2) +
    (1 + a ^ 2) * A ^ 4 * Real.sin t ^ 2 * (k ^ 2 * Real.cos t ^ 2)

noncomputable def modeDenominator (A t : ℝ) : ℝ := 1 + A ^ 2 * Real.sin t ^ 2

noncomputable def modeDensity (a k A t : ℝ) : ℝ :=
  Real.sqrt (modeRadicand a k A t) / modeDenominator A t

noncomputable def modeRateFactor (a k A t : ℝ) : ℝ :=
  (Real.sin t ^ 2 + k ^ 2 * Real.cos t ^ 2 +
      2 * (1 + a ^ 2) * A ^ 2 * Real.sin t ^ 2 * (k ^ 2 * Real.cos t ^ 2)) /
        (Real.sqrt (modeRadicand a k A t) * modeDenominator A t) -
    2 * Real.sin t ^ 2 * Real.sqrt (modeRadicand a k A t) / modeDenominator A t ^ 2

noncomputable def modeRate (a k A t : ℝ) : ℝ := A * modeRateFactor a k A t

theorem modeRadicand_pos (a k A t : ℝ) : 0 < modeRadicand a k A t := by
  unfold modeRadicand
  positivity

theorem modeDenominator_pos (A t : ℝ) : 0 < modeDenominator A t := by
  unfold modeDenominator
  positivity

theorem contDiff_modeRadicand (a k : ℝ) :
    ContDiff ℝ 1 (fun p : ℝ × ℝ => modeRadicand a k p.1 p.2) := by
  unfold modeRadicand
  fun_prop

theorem contDiff_modeDenominator :
    ContDiff ℝ 1 (fun p : ℝ × ℝ => modeDenominator p.1 p.2) := by
  unfold modeDenominator
  fun_prop

theorem contDiff_modeDensity (a k : ℝ) :
    ContDiff ℝ 1 (fun p : ℝ × ℝ => modeDensity a k p.1 p.2) := by
  exact ((contDiff_modeRadicand a k).sqrt
    (fun p => (modeRadicand_pos a k p.1 p.2).ne')).div
      contDiff_modeDenominator (fun p => (modeDenominator_pos p.1 p.2).ne')

theorem contDiff_modeRateFactor (a k : ℝ) :
    ContDiff ℝ 1 (fun p : ℝ × ℝ => modeRateFactor a k p.1 p.2) := by
  have hs := (contDiff_modeRadicand a k).sqrt
    (fun p => (modeRadicand_pos a k p.1 p.2).ne')
  have hn : ContDiff ℝ 1 (fun p : ℝ × ℝ =>
      Real.sin p.2 ^ 2 + k ^ 2 * Real.cos p.2 ^ 2 +
      2 * (1 + a ^ 2) * p.1 ^ 2 * Real.sin p.2 ^ 2 * (k ^ 2 * Real.cos p.2 ^ 2)) := by
    fun_prop
  have hleft := hn.div (hs.mul contDiff_modeDenominator) (fun p =>
    mul_ne_zero (Real.sqrt_pos.2 (modeRadicand_pos a k p.1 p.2)).ne'
      (modeDenominator_pos p.1 p.2).ne')
  have hp : ContDiff ℝ 1 (fun p : ℝ × ℝ => 2 * Real.sin p.2 ^ 2) := by fun_prop
  exact hleft.sub ((hp.mul hs).div (contDiff_modeDenominator.pow 2)
    (fun p => pow_ne_zero 2 (modeDenominator_pos p.1 p.2).ne'))

theorem contDiff_modeRate (a k : ℝ) :
    ContDiff ℝ 1 (fun p : ℝ × ℝ => modeRate a k p.1 p.2) := by
  exact contDiff_fst.mul (contDiff_modeRateFactor a k)

/-- The actual first amplitude derivative of the literal graph density. -/
theorem hasDerivAt_modeDensity (a k A t : ℝ) :
    HasDerivAt (fun B => modeDensity a k B t) (modeRate a k A t) A := by
  have hN : HasDerivAt (fun B => modeRadicand a k B t)
      (2 * A * (Real.sin t ^ 2 + k ^ 2 * Real.cos t ^ 2) +
        4 * (1 + a ^ 2) * A ^ 3 * Real.sin t ^ 2 * (k ^ 2 * Real.cos t ^ 2)) A := by
    convert ((((hasDerivAt_id A).pow 2).mul_const
      (Real.sin t ^ 2 + k ^ 2 * Real.cos t ^ 2)).const_add 1).add
      (((((hasDerivAt_id A).pow 4).const_mul (1 + a ^ 2)).mul_const
        (Real.sin t ^ 2)).mul_const (k ^ 2 * Real.cos t ^ 2)) using 1
    dsimp [modeRadicand]
    ring
  have hD : HasDerivAt (fun B => modeDenominator B t) (2 * A * Real.sin t ^ 2) A := by
    convert (((hasDerivAt_id A).pow 2).mul_const (Real.sin t ^ 2)).const_add 1 using 1
    dsimp [modeDenominator]
    ring
  convert (hN.sqrt (modeRadicand_pos a k A t).ne').div hD
    (modeDenominator_pos A t).ne' using 1
  unfold modeRate modeRateFactor
  field_simp [(modeDenominator_pos A t).ne',
    (Real.sqrt_pos.2 (modeRadicand_pos a k A t)).ne']
  ring

/-- Actual second amplitude derivative of the density at the Shore graph. -/
theorem hasDerivAt_modeRate_zero (a k t : ℝ) :
    HasDerivAt (fun A => modeRate a k A t)
      (k ^ 2 * Real.cos t ^ 2 - Real.sin t ^ 2) 0 := by
  have hf := (((contDiff_modeRateFactor a k).comp
    (contDiff_id.prodMk (contDiff_const (c := t)))).differentiable (by norm_num) 0).hasDerivAt
  convert (hasDerivAt_id (0 : ℝ)).mul hf using 1
  simp [modeRateFactor, modeRadicand, modeDenominator]
  ring

noncomputable def modeIntegral (a k A : ℝ) : ℝ :=
  ∫ t in (0 : ℝ)..Real.pi, modeDensity a k A t

/-- Differentiation under the actual graph-density integral. -/
theorem hasDerivAt_modeIntegral (a k A : ℝ) :
    HasDerivAt (modeIntegral a k)
      (∫ t in (0 : ℝ)..Real.pi, modeRate a k A t) A := by
  exact CompactParametricIntegral.hasDerivAt_intervalIntegral
    (fun p => modeDensity a k p.1 p.2) (fun p => modeRate a k p.1 p.2)
    (contDiff_modeDensity a k).continuous (contDiff_modeRate a k).continuous
    (hasDerivAt_modeDensity a k) 0 Real.pi A Real.pi_pos.le

/-- The actual first derivative vanishes at zero amplitude. -/
theorem hasDerivAt_modeIntegral_zero (a k : ℝ) :
    HasDerivAt (modeIntegral a k) 0 0 := by
  simpa [modeRate] using hasDerivAt_modeIntegral a k 0

/-- The actual derivative of the actual first integral derivative at zero. -/
theorem hasDerivAt_deriv_modeIntegral_zero (a k : ℝ) :
    HasDerivAt (deriv (modeIntegral a k)) (Real.pi / 2 * (k ^ 2 - 1)) 0 := by
  let F : ℝ × ℝ → ℝ := fun p => modeRate a k p.1 p.2
  let F' : ℝ × ℝ → ℝ := fun p => fderiv ℝ F p (1, 0)
  have hc : Continuous F' :=
    ((contDiff_modeRate a k).continuous_fderiv (by norm_num)).clm_apply continuous_const
  have hd : ∀ A t, HasDerivAt (fun B => F (B, t)) (F' (A, t)) A := by
    intro A t
    exact ((contDiff_modeRate a k).differentiable (by norm_num) (A, t)).hasFDerivAt.comp_hasDerivAt A
      ((hasDerivAt_id A).prodMk (hasDerivAt_const A t))
  have hz : ∀ t, F' (0, t) = k ^ 2 * Real.cos t ^ 2 - Real.sin t ^ 2 :=
    fun t => (hd 0 t).unique (hasDerivAt_modeRate_zero a k t)
  have hi : (∫ t in (0 : ℝ)..Real.pi, F' (0, t)) = Real.pi / 2 * (k ^ 2 - 1) := by
    simp_rw [hz]
    rw [intervalIntegral.integral_sub
      ((continuous_const.mul (Real.continuous_cos.pow 2)).intervalIntegrable _ _)
      ((Real.continuous_sin.pow 2).intervalIntegrable _ _),
      intervalIntegral.integral_const_mul, integral_cos_sq, integral_sin_sq]
    simp
    ring
  have hderiv : deriv (modeIntegral a k) =
      fun A => ∫ t in (0 : ℝ)..Real.pi, F (A, t) :=
    funext fun A => (hasDerivAt_modeIntegral a k A).deriv
  rw [hderiv, ← hi]
  exact CompactParametricIntegral.hasDerivAt_intervalIntegral F F'
    (contDiff_modeRate a k).continuous hc hd 0 Real.pi 0 Real.pi_pos.le

/-- For positive ell, the sine-graph length after the parameterization y=(ell/pi)*t.
Its equality to the actual coordinate metric integral is proved below.
For nonpositive ell this remains a signed algebraic functional. -/
noncomputable def oneModeLength (ell a A : ℝ) : ℝ :=
  ell / Real.pi * modeIntegral a (Real.pi / ell) A

theorem oneModeLength_zero (ell a : ℝ) : oneModeLength ell a 0 = ell := by
  simp [oneModeLength, modeIntegral, modeDensity, modeRadicand, modeDenominator,
    Real.pi_ne_zero]

theorem hasDerivAt_oneModeLength_zero (ell a : ℝ) :
    HasDerivAt (oneModeLength ell a) 0 0 := by
  simpa [oneModeLength] using
    (hasDerivAt_modeIntegral_zero a (Real.pi / ell)).const_mul (ell / Real.pi)

/-- The actual second derivative of the literal sine-graph length. -/
theorem hasDerivAt_deriv_oneModeLength_zero (ell a : ℝ) :
    HasDerivAt (deriv (oneModeLength ell a))
      (ell / 2 * ((Real.pi / ell) ^ 2 - 1)) 0 := by
  have hderiv : deriv (oneModeLength ell a) =
      fun A => (ell / Real.pi) * deriv (modeIntegral a (Real.pi / ell)) A := by
    funext A
    rw [(show HasDerivAt (oneModeLength ell a)
      ((ell / Real.pi) * (∫ t in (0 : ℝ)..Real.pi, modeRate a (Real.pi / ell) A t)) A from
        (hasDerivAt_modeIntegral a (Real.pi / ell) A).const_mul (ell / Real.pi)).deriv,
      (hasDerivAt_modeIntegral a (Real.pi / ell) A).deriv]
  rw [hderiv]
  convert (hasDerivAt_deriv_modeIntegral_zero a (Real.pi / ell)).const_mul (ell / Real.pi) using 1
  field_simp

noncomputable def modeCurve (ell A t : ℝ) : FlatShoreGeometry.Coord :=
  (A * Real.sin t, (ell / Real.pi) * t)

noncomputable def modeVelocity (ell A t : ℝ) : FlatShoreGeometry.Coord :=
  (A * Real.cos t, ell / Real.pi)

/-- Actual tangent of the sine-graph parameterization. -/
theorem hasDerivAt_modeCurve (ell A t : ℝ) :
    HasDerivAt (modeCurve ell A) (modeVelocity ell A t) t := by
  simpa only [modeCurve, modeVelocity, mul_one] using
    ((Real.hasDerivAt_sin t).const_mul A).prodMk
      ((hasDerivAt_id t).const_mul (ell / Real.pi))

theorem modeCurve_endpoints (ell A : ℝ) :
    modeCurve ell A 0 = (0, 0) ∧ modeCurve ell A Real.pi = (0, ell) := by
  simp [modeCurve, Real.pi_ne_zero]

/-- This parameterized curve is exactly the graph q(y)=A*sin(pi*y/ell). -/
theorem modeCurve_graph (ell A t : ℝ) (hell : ell ≠ 0) :
    (modeCurve ell A t).1 =
      A * Real.sin (Real.pi * (modeCurve ell A t).2 / ell) := by
  have harg : Real.pi * ((ell / Real.pi) * t) / ell = t := by
    field_simp
  simp only [modeCurve, harg]

/-- The actual competitor is absolutely continuous on every finite interval. -/
theorem absolutelyContinuousOnInterval_modeCurve (ell A lo hi : ℝ) :
    AbsolutelyContinuousOnInterval (modeCurve ell A) lo hi := by
  have hLip : LipschitzWith (max ‖A‖₊ ‖ell / Real.pi‖₊) (modeCurve ell A) := by
    apply lipschitzWith_of_nnnorm_deriv_le (fun t => (hasDerivAt_modeCurve ell A t).differentiableAt)
    intro t
    rw [(hasDerivAt_modeCurve ell A t).deriv]
    apply NNReal.coe_le_coe.mp
    change max |A * Real.cos t| |ell / Real.pi| ≤ max |A| |ell / Real.pi|
    exact max_le_max (by rw [abs_mul]; nlinarith [Real.abs_cos_le_one t, abs_nonneg A]) le_rfl
  exact hLip.lipschitzOnWith.absolutelyContinuousOnInterval

/-- The density radicand comes from the actual existing flat metric and tangent. -/
theorem metric_modeCurve (ell a A t : ℝ) (hell : ell ≠ 0) :
    FlatShoreGeometry.metric 1 a (modeCurve ell A t).1
      (modeVelocity ell A t) (modeVelocity ell A t) =
      (ell / Real.pi) ^ 2 * modeRadicand a (Real.pi / ell) A t /
        modeDenominator A t ^ 2 := by
  have hD : 1 + (A * Real.sin t) ^ 2 = modeDenominator A t := by
    unfold modeDenominator
    ring
  dsimp [FlatShoreGeometry.metric, FlatShoreGeometry.radialMetric,
    FlatShoreGeometry.tangentialMetric, modeCurve, modeVelocity]
  rw [one_pow, hD]
  unfold modeRadicand
  field_simp [hell, Real.pi_ne_zero, (modeDenominator_pos A t).ne']
  unfold modeDenominator
  ring

/-- For positive span the actual metric speed equals the normalized graph density. -/
theorem metric_speed_modeCurve (ell a A t : ℝ) (hell : 0 < ell) :
    Real.sqrt (FlatShoreGeometry.metric 1 a (modeCurve ell A t).1
      (modeVelocity ell A t) (modeVelocity ell A t)) =
      (ell / Real.pi) * modeDensity a (Real.pi / ell) A t := by
  rw [metric_modeCurve ell a A t hell.ne',
    Real.sqrt_div (mul_nonneg (sq_nonneg _) (modeRadicand_pos a (Real.pi / ell) A t).le),
    Real.sqrt_mul (sq_nonneg _), Real.sqrt_sq_eq_abs, Real.sqrt_sq_eq_abs,
    abs_of_pos (div_pos hell Real.pi_pos), abs_of_pos (modeDenominator_pos A t)]
  exact mul_div_assoc _ _ _

/-- The amplitude functional is exactly the existing flat coordinate metric
length integral of the actual sine-curve tangent, with fixed endpoints. -/
theorem oneModeLength_eq_coordinateIntegral (ell a A : ℝ) (hell : 0 < ell) :
    oneModeLength ell a A =
      ∫ t in (0 : ℝ)..Real.pi,
        Real.sqrt (FlatShoreGeometry.metric 1 a (modeCurve ell A t).1
          (deriv (modeCurve ell A) t) (deriv (modeCurve ell A) t)) := by
  simp_rw [(hasDerivAt_modeCurve ell A _).deriv, metric_speed_modeCurve ell a A _ hell]
  exact (intervalIntegral.integral_const_mul (ell / Real.pi)
    (modeDensity a (Real.pi / ell) A)).symm

/-- Beyond pi the actual sine-family second variation is strictly negative. -/
theorem deriv_deriv_oneModeLength_neg (ell a : ℝ) (hell : Real.pi < ell) :
    deriv (deriv (oneModeLength ell a)) 0 < 0 := by
  rw [(hasDerivAt_deriv_oneModeLength_zero ell a).deriv]
  have hlpos := Real.pi_pos.trans hell
  have hkpos : 0 < Real.pi / ell := div_pos Real.pi_pos hlpos
  have hklt : Real.pi / ell < 1 := (div_lt_one hlpos).2 hell
  exact mul_neg_of_pos_of_neg (div_pos hlpos (by norm_num)) (by nlinarith)

/-- Every sufficiently small positive sine amplitude strictly shortens the
actual coordinate length when the Shore span is greater than pi. -/
theorem exists_shortening_amplitude_interval (ell a : ℝ) (hell : Real.pi < ell) :
    ∃ delta > 0, ∀ A ∈ Ioo (0 : ℝ) delta, oneModeLength ell a A < ell := by
  have h0 := hasDerivAt_oneModeLength_zero ell a
  have h2 := hasDerivAt_deriv_oneModeLength_zero ell a
  have hc : ell / 2 * ((Real.pi / ell) ^ 2 - 1) < 0 := by
    rw [← h2.deriv]
    exact deriv_deriv_oneModeLength_neg ell a hell
  have hf : Continuous (oneModeLength ell a) := continuous_iff_continuousAt.mpr
    (fun A => ((hasDerivAt_modeIntegral a (Real.pi / ell) A).const_mul
      (ell / Real.pi)).continuousAt)
  have hs : ∀ᶠ A in 𝓝[>] (0 : ℝ), deriv (oneModeLength ell a) A < 0 := by
    have hsl := h2.tendsto_slope_zero_right.eventually (eventually_lt_nhds hc)
    filter_upwards [hsl, self_mem_nhdsWithin] with A hA (hApos : 0 < A)
    simp only [zero_add, h0.deriv, sub_zero, smul_eq_mul] at hA
    exact neg_of_mul_neg_right hA (inv_nonneg.mpr hApos.le)
  obtain ⟨delta, hdelta, hbound⟩ := (nhdsGT_basis (0 : ℝ)).eventually_iff.mp hs
  refine ⟨delta, hdelta, ?_⟩
  intro A hA
  have hanti : StrictAntiOn (oneModeLength ell a) (Icc (0 : ℝ) delta) :=
    strictAntiOn_of_deriv_neg (convex_Icc _ _) hf.continuousOn
      (by simpa only [interior_Icc] using hbound)
  simpa only [oneModeLength_zero] using
    hanti (left_mem_Icc.mpr hdelta.le) (Ioo_subset_Icc_self hA) hA.1

end FractalSync.FlatShoreSecondVariation
