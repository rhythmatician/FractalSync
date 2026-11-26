import FractalSync.ConformalFloorCurvature

/-!
# A quartic floor on a planar conformal ruler

The checked statements cover the log-floor derivatives, the weighted trace
identity, and the main-cardioid scalar curvature bounds.
-/

namespace FractalSync.QuarticRulerFloor

open Filter
open scoped Topology

noncomputable def quarticFloor (epsilon r : ℝ) : ℝ :=
  Real.sqrt (Real.sqrt (r ^ 4 + epsilon ^ 4))

private theorem quarticSum_pos (epsilon r : ℝ) (he : 0 < epsilon) :
    0 < r ^ 4 + epsilon ^ 4 := by positivity

theorem hasDerivAt_log_quarticFloor (epsilon r : ℝ) (he : 0 < epsilon) :
    HasDerivAt (fun s => Real.log (quarticFloor epsilon s))
      (r ^ 3 / (r ^ 4 + epsilon ^ 4)) r := by
  have hlog (s : ℝ) : Real.log (quarticFloor epsilon s) =
      Real.log (s ^ 4 + epsilon ^ 4) / 4 := by
    rw [quarticFloor, Real.log_sqrt (Real.sqrt_nonneg _),
      Real.log_sqrt (quarticSum_pos epsilon s he).le]
    ring
  simp only [hlog]
  have h := (Real.hasDerivAt_log (quarticSum_pos epsilon r he).ne').comp r
    (((hasDerivAt_id r).pow 4).add_const (epsilon ^ 4))
  convert h.div_const 4 using 1
  simp only [id_eq, Nat.cast_ofNat, mul_one]
  ring

theorem hasDerivAt_log_quarticFloor_slope (epsilon r : ℝ) (he : 0 < epsilon) :
    HasDerivAt (fun s => s ^ 3 / (s ^ 4 + epsilon ^ 4))
      (r ^ 2 * (3 * epsilon ^ 4 - r ^ 4) / (r ^ 4 + epsilon ^ 4) ^ 2) r := by
  convert ((hasDerivAt_id r).pow 3).div
    (((hasDerivAt_id r).pow 4).add_const (epsilon ^ 4))
    (quarticSum_pos epsilon r he).ne' using 1
  simp only [id_eq, Nat.cast_ofNat, mul_one, Pi.pow_apply]
  congr 1
  ring

theorem hasFDerivAt_fderiv_log_quarticFloor {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E] (epsilon : ℝ) (he : 0 < epsilon)
    (rho : E → ℝ) (x : E) (H : E →L[ℝ] E →L[ℝ] ℝ)
    (hD : ∀ᶠ y in 𝓝 x, DifferentiableAt ℝ rho y)
    (hH : HasFDerivAt (fderiv ℝ rho) H x) :
    HasFDerivAt (fderiv ℝ (fun y => Real.log (quarticFloor epsilon (rho y))))
      ((rho x ^ 3 / (rho x ^ 4 + epsilon ^ 4)) • H +
        (rho x ^ 2 * (3 * epsilon ^ 4 - rho x ^ 4) / (rho x ^ 4 + epsilon ^ 4) ^ 2) •
          (fderiv ℝ rho x).smulRight (fderiv ℝ rho x)) x := by
  have hx := (Filter.Eventually.self_of_nhds hD).hasFDerivAt
  have hd :=
    ((hasDerivAt_log_quarticFloor_slope epsilon (rho x) he).comp_hasFDerivAt x hx).smul hH
  have heq : fderiv ℝ (fun y => Real.log (quarticFloor epsilon (rho y))) =ᶠ[𝓝 x]
      (fun y => (rho y ^ 3 / (rho y ^ 4 + epsilon ^ 4)) • fderiv ℝ rho y) := by
    filter_upwards [hD] with y hy
    exact ((hasDerivAt_log_quarticFloor epsilon (rho y) he).comp_hasFDerivAt y
      hy.hasFDerivAt).fderiv
  convert hd.congr_of_eventuallyEq heq using 1
  ext u v
  simp
  ring

/-- Weighted trace identity, with the Euclidean Hessian trace `lap` and
squared gradient norm `g2` of the unfloored ruler. -/
theorem quartic_floor_curvature_eq (epsilon r lap g2 : ℝ) (he : 0 < epsilon) :
    quarticFloor epsilon r ^ 2 *
      (r ^ 3 / (r ^ 4 + epsilon ^ 4) * lap +
        r ^ 2 * (3 * epsilon ^ 4 - r ^ 4) / (r ^ 4 + epsilon ^ 4) ^ 2 * g2) =
      (r / quarticFloor epsilon r) ^ 2 *
        ((r * lap - g2) + (4 * epsilon ^ 4 / (r ^ 4 + epsilon ^ 4)) * g2) := by
  have hS := quarticSum_pos epsilon r he
  have hR : 0 < quarticFloor epsilon r := by
    exact Real.sqrt_pos.mpr (Real.sqrt_pos.mpr hS)
  have hR2 : quarticFloor epsilon r ^ 2 = Real.sqrt (r ^ 4 + epsilon ^ 4) :=
    Real.sq_sqrt (Real.sqrt_nonneg _)
  have hS_eq : (quarticFloor epsilon r ^ 2) ^ 2 = r ^ 4 + epsilon ^ 4 := by
    rw [hR2, Real.sq_sqrt hS.le]
  have hweight : (r / quarticFloor epsilon r) ^ 2 =
      quarticFloor epsilon r ^ 2 * (r ^ 2 / (r ^ 4 + epsilon ^ 4)) := by
    rw [div_pow, ← hS_eq, pow_two (quarticFloor epsilon r ^ 2)]
    field_simp [hR.ne']
  rw [hweight]
  field_simp [hS.ne']
  ring

/-- On the cardioid `y = (ρ/R)²` and `g2 = |∇ρ|²`. The same estimate
gives vanishing curvature as `y` tends to zero. -/
theorem cardioid_quartic_curvature_bounds (y g2 : ℝ) (hy : 0 ≤ y) (hy1 : y ≤ 1)
    (hg : 0 ≤ g2) (hg4 : g2 ≤ 4) :
    -y ≤ y * (-1 + 4 * (1 - y ^ 2) * g2) ∧
      y * (-1 + 4 * (1 - y ^ 2) * g2) ≤ 15 * y - 16 * y ^ 3 := by
  have hfactor : 0 ≤ 1 - y ^ 2 := by
    nlinarith [mul_nonneg (sub_nonneg.mpr hy1) (show 0 ≤ 1 + y by linarith)]
  constructor
  · have hcorrection : 0 ≤ 4 * (1 - y ^ 2) * g2 := by positivity
    calc
      -y = y * (-1) := by ring
      _ ≤ y * (-1 + 4 * (1 - y ^ 2) * g2) :=
        mul_le_mul_of_nonneg_left (by linarith) hy
  · have hcorrection := mul_le_mul_of_nonneg_left hg4
      (show 0 ≤ 4 * (1 - y ^ 2) by positivity)
    calc
      y * (-1 + 4 * (1 - y ^ 2) * g2) ≤ y * (-1 + 4 * (1 - y ^ 2) * 4) :=
        mul_le_mul_of_nonneg_left (by linarith) hy
      _ = 15 * y - 16 * y ^ 3 := by ring

/-- The upper envelope is sharp at `y = √5 / 4`. -/
theorem quartic_curvature_envelope_le (y : ℝ) (hy : 0 ≤ y) :
    15 * y - 16 * y ^ 3 ≤ 5 * Real.sqrt 5 / 2 := by
  set a := Real.sqrt 5 / 4
  have ha : 0 ≤ a := by positivity
  have ha2 : a ^ 2 = 5 / 16 := by
    dsimp [a]
    rw [div_pow, Real.sq_sqrt (by norm_num : (0 : ℝ) ≤ 5)]
    norm_num
  have hfactor : 10 * a - (15 * y - 16 * y ^ 3) = 16 * (y - a) ^ 2 * (y + 2 * a) := by
    linear_combination (48 * y - 32 * a) * ha2
  have hnonneg : 0 ≤ 16 * (y - a) ^ 2 * (y + 2 * a) := by positivity
  rw [← hfactor] at hnonneg
  dsimp [a] at hnonneg
  linarith

end FractalSync.QuarticRulerFloor
