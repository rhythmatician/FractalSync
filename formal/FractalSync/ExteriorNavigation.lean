import FractalSync.ExteriorRulerBounds

/-!
# Exterior ray and equipotential velocities

The complex differential of the local log coordinate is multiplication
by `u`. These algebraic statements give the exact rates and the effect
of replacing that differential with a finite approximation.
-/

namespace FractalSync.ExteriorNavigation

noncomputable def ray (g : ℝ) (u : ℂ) : ℂ := (g : ℂ) / u

noncomputable def equipotential (g : ℝ) (u : ℂ) : ℂ := Complex.I * ray g u

theorem ray_rate (g : ℝ) {u : ℂ} (hu : u ≠ 0) :
    u * ray g u = (g : ℂ) := by
  unfold ray
  field_simp

theorem equipotential_rate (g : ℝ) {u : ℂ} (hu : u ≠ 0) :
    u * equipotential g u = Complex.I * (g : ℂ) := by
  rw [equipotential, ← mul_left_comm, ray_rate g hu]

theorem coordinate_speeds (g : ℝ) (hg : 0 ≤ g) (u : ℂ) :
    ‖ray g u‖ = g / ‖u‖ ∧ ‖equipotential g u‖ = g / ‖u‖ := by
  have hr : ‖ray g u‖ = g / ‖u‖ := by
    rw [ray, norm_div, Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg hg]
  exact ⟨hr, by rw [equipotential, norm_mul, Complex.norm_I, one_mul, hr]⟩

theorem approximated_ray_rate_error (g : ℝ) (hg : 0 ≤ g) (u : ℂ)
    {estimate : ℂ} (hne : estimate ≠ 0) {error : ℝ} (h : ‖u - estimate‖ ≤ error) :
    ‖u * ray g estimate - (g : ℂ)‖ ≤ g * error / ‖estimate‖ := by
  rw [show u * ray g estimate - (g : ℂ) = (u - estimate) * ray g estimate by
    rw [sub_mul, ray_rate g hne]]
  rw [norm_mul, (coordinate_speeds g hg estimate).1]
  calc
    _ = g * ‖u - estimate‖ / ‖estimate‖ := by ring
    _ ≤ _ := div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left h hg) (norm_nonneg _)

theorem approximated_equipotential_rate_error (g : ℝ) (hg : 0 ≤ g) (u : ℂ)
    {estimate : ℂ} (hne : estimate ≠ 0) {error : ℝ} (h : ‖u - estimate‖ ≤ error) :
    ‖u * equipotential g estimate - Complex.I * (g : ℂ)‖ ≤ g * error / ‖estimate‖ := by
  rw [show u * equipotential g estimate - Complex.I * (g : ℂ) =
      Complex.I * (u * ray g estimate - (g : ℂ)) by unfold equipotential; ring]
  rw [norm_mul, Complex.norm_I, one_mul]
  exact approximated_ray_rate_error g hg u hne h

end FractalSync.ExteriorNavigation
