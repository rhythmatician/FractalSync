import FractalSync.Ruler
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.Calculus.FDeriv.Mul

/-!
# A positive floor on a planar conformal ruler

We check the derivatives of the logarithm of the floored ruler, and the
curvature correction expressed in Euclidean field derivatives. The geometric
identification of the weighted Laplacian with Gaussian curvature is derived
in the accompanying research note.
-/

namespace FractalSync.ConformalFloorCurvature

open Filter
open scoped Topology

theorem hasDerivAt_log_floor (epsilon r : ℝ) (he : 0 < epsilon) :
    HasDerivAt (fun s => Real.log (ruler epsilon s)) (r / (r ^ 2 + epsilon ^ 2)) r := by
  convert (Real.hasDerivAt_log (ruler_pos epsilon r he).ne').comp r
    (hasDerivAt_ruler epsilon r he) using 1
  rw [← ruler_sq epsilon r]
  field_simp [(ruler_pos epsilon r he).ne']

theorem hasDerivAt_log_floor_slope (epsilon r : ℝ) (he : 0 < epsilon) :
    HasDerivAt (fun s => s / (s ^ 2 + epsilon ^ 2))
      ((epsilon ^ 2 - r ^ 2) / (r ^ 2 + epsilon ^ 2) ^ 2) r := by
  have hn : r ^ 2 + epsilon ^ 2 ≠ 0 := by positivity
  convert (hasDerivAt_id r).div (((hasDerivAt_id r).pow 2).add_const (epsilon ^ 2)) hn
    using 1
  simp only [id_eq, Nat.cast_ofNat, mul_one, one_mul, Pi.pow_apply]
  congr 1
  ring

/-- The actual second Fréchet derivative of the logarithm of the floored ruler. -/
theorem hasFDerivAt_fderiv_log_floor {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (epsilon : ℝ) (he : 0 < epsilon) (rho : E → ℝ) (x : E)
    (H : E →L[ℝ] E →L[ℝ] ℝ) (hD : ∀ᶠ y in 𝓝 x, DifferentiableAt ℝ rho y)
    (hH : HasFDerivAt (fderiv ℝ rho) H x) :
    HasFDerivAt (fderiv ℝ (fun y => Real.log (ruler epsilon (rho y))))
      ((rho x / (rho x ^ 2 + epsilon ^ 2)) • H +
        ((epsilon ^ 2 - rho x ^ 2) / (rho x ^ 2 + epsilon ^ 2) ^ 2) •
          (fderiv ℝ rho x).smulRight (fderiv ℝ rho x)) x := by
  have hx := (Filter.Eventually.self_of_nhds hD).hasFDerivAt
  have hd := ((hasDerivAt_log_floor_slope epsilon (rho x) he).comp_hasFDerivAt x hx).smul hH
  have heq : fderiv ℝ (fun y => Real.log (ruler epsilon (rho y))) =ᶠ[𝓝 x]
      (fun y => (rho y / (rho y ^ 2 + epsilon ^ 2)) • fderiv ℝ rho y) := by
    filter_upwards [hD] with y hy
    exact ((hasDerivAt_log_floor epsilon (rho y) he).comp_hasFDerivAt y hy.hasFDerivAt).fderiv
  convert hd.congr_of_eventuallyEq heq using 1
  ext u v
  simp
  ring

/-- Weighted trace of the regularized log Hessian, expressed using the trace
of the original Hessian and the squared Euclidean gradient norm. -/
theorem floor_curvature_eq (epsilon r lap g2 : ℝ) (he : 0 < epsilon) :
    (r ^ 2 + epsilon ^ 2) *
      (r / (r ^ 2 + epsilon ^ 2) * lap +
        ((epsilon ^ 2 - r ^ 2) / (r ^ 2 + epsilon ^ 2) ^ 2) * g2) =
      (r * lap - g2) + (2 * epsilon ^ 2 / (r ^ 2 + epsilon ^ 2)) * g2 := by
  have hn : r ^ 2 + epsilon ^ 2 ≠ 0 := by positivity
  field_simp
  ring

/-- The correction is nonnegative and at most twice the squared gradient norm. -/
theorem floor_correction_bounds (epsilon r g2 : ℝ) (he : 0 < epsilon) (hg : 0 ≤ g2) :
    0 ≤ (2 * epsilon ^ 2 / (r ^ 2 + epsilon ^ 2)) * g2 ∧
      (2 * epsilon ^ 2 / (r ^ 2 + epsilon ^ 2)) * g2 ≤ 2 * g2 := by
  refine ⟨by positivity, mul_le_mul_of_nonneg_right ?_ hg⟩
  rw [div_le_iff₀ (by positivity)]
  nlinarith [sq_nonneg r]

/-- Algebraic simplification of the main-cardioid squared gradient.
`a = 1 - |u|²` and `s2 = |1-u|²`. -/
theorem cardioid_gradient_sq (a s2 : ℝ) (hs : s2 ≠ 0) :
    (1 - a) - a * ((s2 - a) / 2) / s2 + a ^ 2 / (4 * s2) =
      1 - 3 * a / 2 + 3 * a ^ 2 / (4 * s2) := by
  field_simp
  ring

/-- This bound applies throughout the disk, since `0 ≤ a ≤ 2 * sqrt s2`. -/
theorem cardioid_gradient_sq_le (a s2 : ℝ) (ha : 0 ≤ a) (hs : 0 < s2)
    (ht : a ^ 2 ≤ 4 * s2) :
    1 - 3 * a / 2 + 3 * a ^ 2 / (4 * s2) ≤ 4 := by
  have h : 3 * a ^ 2 / (4 * s2) ≤ 3 := by
    rw [div_le_iff₀ (by positivity)]
    linarith
  linarith

end FractalSync.ConformalFloorCurvature
