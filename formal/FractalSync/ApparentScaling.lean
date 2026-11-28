import Mathlib.Analysis.Calculus.FDeriv.Mul
import Mathlib.Analysis.Complex.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Deriv

namespace FractalSync.ApparentScaling

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
variable {w : E → ℂ} {W : E →L[ℝ] ℂ} {c : E}

private theorem hasFDerivAt_normSq (hw : HasFDerivAt w W c) :
    HasFDerivAt (fun x => Complex.normSq (w x))
      ((2 * (w c).re) • (Complex.reCLM.comp W) +
       (2 * (w c).im) • (Complex.imCLM.comp W)) c := by
  have hr := Complex.reCLM.hasFDerivAt.comp c hw
  have hi := Complex.imCLM.hasFDerivAt.comp c hw
  convert (hr.mul hr).add (hi.mul hi) using 1
  ext v
  simp [two_mul, add_mul]

/-- The fractional radial rate in the plane, with the center already subtracted. -/
noncomputable def radialRate (z : ℂ) : ℂ →L[ℝ] ℝ :=
  (z.re / Complex.normSq z) • Complex.reCLM + (z.im / Complex.normSq z) • Complex.imCLM

theorem radialRate_apply (z v : ℂ) : radialRate z v = (v / z).re := by
  simp only [radialRate, Complex.div_re, ContinuousLinearMap.add_apply,
    ContinuousLinearMap.smul_apply, Complex.reCLM_apply, Complex.imCLM_apply, smul_eq_mul]
  ring

/-- A resolved, differentiable landmark has this real differential of log radius. -/
theorem hasFDerivAt_log_norm (hw : HasFDerivAt w W c) (hz : w c ≠ 0) :
    HasFDerivAt (fun x => Real.log ‖w x‖) ((radialRate (w c)).comp W) c := by
  have hn : Complex.normSq (w c) ≠ 0 := mt Complex.normSq_eq_zero.mp hz
  convert ((hasFDerivAt_normSq hw).log hn).const_mul (1 / 2) using 1
  · funext x
    rw [← Complex.sq_norm, Real.log_pow]
    push_cast
    ring
  · ext h
    simp only [radialRate, ContinuousLinearMap.comp_apply, ContinuousLinearMap.add_apply,
      ContinuousLinearMap.smul_apply, Complex.reCLM_apply, Complex.imCLM_apply, smul_eq_mul,
      div_eq_mul_inv]
    ring

/-- The opening rate relative to an outer landmark cancels common scale. -/
theorem hasFDerivAt_log_radius_ratio {u : E → ℂ} {U : E →L[ℝ] ℂ}
    (hw : HasFDerivAt w W c) (hu : HasFDerivAt u U c)
    (hw₀ : w c ≠ 0) (hu₀ : u c ≠ 0) :
    HasFDerivAt (fun x => Real.log (‖w x‖ / ‖u x‖))
      ((radialRate (w c)).comp W - (radialRate (u c)).comp U) c := by
  refine ((hasFDerivAt_log_norm hw hw₀).sub
    (hasFDerivAt_log_norm hu hu₀)).congr_of_eventuallyEq ?_
  filter_upwards [hw.continuousAt.eventually_ne hw₀, hu.continuousAt.eventually_ne hu₀] with x hx hy
  exact Real.log_div (norm_ne_zero_iff.mpr hx) (norm_ne_zero_iff.mpr hy)

/-- Center and landmark velocities enter through their difference. -/
theorem hasFDerivAt_relative_log_norm {p q : E → ℂ} {P Q : E →L[ℝ] ℂ}
    (hp : HasFDerivAt p P c) (hq : HasFDerivAt q Q c) (hne : q c ≠ p c) :
    HasFDerivAt (fun x => Real.log ‖q x - p x‖)
      ((radialRate (q c - p c)).comp (Q - P)) c :=
  hasFDerivAt_log_norm (hq.sub hp) (sub_ne_zero.mpr hne)

/-- Synthetic fixture: a common translation cancels and a rotation has zero radial rate.
The remaining real coefficient is the uniform fractional scale rate. -/
theorem radialRate_similarity {z : ℂ} (hz : z ≠ 0) (v k : ℂ) :
    radialRate z ((v + k * z) - v) = k.re := by
  rw [add_sub_cancel_left, radialRate_apply, mul_div_cancel_right₀ _ hz]

/-- Synthetic aperture fixture: inner and outer landmark scale rates can differ. -/
theorem radialRate_aperture {inner outer : ℂ} (hi : inner ≠ 0) (ho : outer ≠ 0)
    (v innerRate outerRate : ℂ) :
    radialRate inner ((v + innerRate * inner) - v) -
      radialRate outer ((v + outerRate * outer) - v) = innerRate.re - outerRate.re := by
  rw [radialRate_similarity hi, radialRate_similarity ho]

end FractalSync.ApparentScaling
