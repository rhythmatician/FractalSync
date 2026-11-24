import Mathlib.Analysis.Normed.Group.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Ring

/-!
# Sensitivity transfer bounds

These statements formalize the norm and scalar algebra used to transfer a
two-sided sensitivity bound from an analytic model to a perturbed observable.
They do not formalize differentiability of the L2 affine action, its curvature,
or the behavior of FractalSync's renderer. See research issue #161.
-/

namespace FractalSync

/-- Derivative-error bounds transfer upper and lower norm bounds. Here `r` is
the input displacement norm, and `x`, `y` are model and actual output responses.
This first, unsquared statement needs no sign assumptions on the constants. -/
theorem sensitivity_norm_transfer {E : Type*} [SeminormedAddCommGroup E]
    (x y : E) (lower upper error radius : ℝ)
    (h_lower : lower * radius ≤ ‖x‖)
    (h_upper : ‖x‖ ≤ upper * radius)
    (h_error : ‖y - x‖ ≤ error * radius) :
    (lower - error) * radius ≤ ‖y‖ ∧
      ‖y‖ ≤ (upper + error) * radius := by
  constructor
  · nlinarith [norm_sub_norm_le x y, norm_sub_rev x y]
  · nlinarith [norm_sub_norm_le y x]

/-- Squaring the transferred norm bounds gives the robust metric constants.
The lower margin must be nonnegative before it can safely be squared. -/
theorem sensitivity_squared_transfer {E : Type*} [SeminormedAddCommGroup E]
    (x y : E) (lower upper error radius : ℝ)
    (h_lower : lower * radius ≤ ‖x‖)
    (h_upper : ‖x‖ ≤ upper * radius)
    (h_error : ‖y - x‖ ≤ error * radius)
    (h_radius : 0 ≤ radius) (h_margin : error ≤ lower) :
    (lower - error) ^ 2 * radius ^ 2 ≤ ‖y‖ ^ 2 ∧
      ‖y‖ ^ 2 ≤ (upper + error) ^ 2 * radius ^ 2 := by
  have h_bounds := sensitivity_norm_transfer x y lower upper error radius
    h_lower h_upper h_error
  constructor
  · simpa only [mul_pow] using
      (sq_le_sq₀ (mul_nonneg (sub_nonneg.mpr h_margin) h_radius)
        (norm_nonneg y)).mpr h_bounds.1
  · simpa only [mul_pow] using
      (sq_le_sq₀ (norm_nonneg y)
        ((norm_nonneg y).trans h_bounds.2)).mpr h_bounds.2

/-- The location-scale mixed term is removed by a shear. This is the scalar
translation case of the block Schur-complement identity in research issue #161. -/
theorem location_scale_complete_square (A b c u v : ℝ) (hA : A ≠ 0) :
    A * u ^ 2 + 2 * b * u * v + c * v ^ 2 =
      A * (u + (b / A) * v) ^ 2 + (c - b ^ 2 / A) * v ^ 2 := by
  field_simp
  ring

/-- Positive leading principal minors give a positive Schur complement.
Combined with the preceding identity, this is the scalar algebra behind a
nondegenerate location-scale coordinate shear. -/
theorem location_scale_schur_pos (A b c : ℝ) (hA : 0 < A)
    (h_det : 0 < A * c - b ^ 2) : 0 < c - b ^ 2 / A := by
  apply sub_pos.mpr
  apply (div_lt_iff₀ hA).2
  nlinarith

end FractalSync
