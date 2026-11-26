import FractalSync.OrbitTrapContours

/-!
# Turning of regular angular trap lifts

The acceleration is the real-phase second derivative of the existing lift.
Its normalized imaginary component gives the signed Euclidean heading
derivative per unit phase.
-/

namespace FractalSync.OrbitTrapCurvature

open OrbitTrapContours

noncomputable section

theorem hasDerivAt_lift_angleVelocity (c : ℂ) (θ : ℝ) (hA : slope c ≠ 0)
    (hbranch : relativeDiscriminant c θ ∈ Complex.slitPlane) :
    HasDerivAt (fun t : ℝ => angleVelocity (lift c t))
      (angleVelocity (lift c θ) * (Complex.I *
        (1 - 2 * orbit (lift c θ) / slope (lift c θ) ^ 2))) θ := by
  have hc := hasDerivAt_lift c θ hA hbranch
  have hnum : HasDerivAt (fun t : ℝ => Complex.I * orbit (lift c t))
      (Complex.I * (slope (lift c θ) * angleVelocity (lift c θ))) θ := by
    convert (((hc.pow 2).add hc).const_mul Complex.I) using 1
    unfold OrbitTrapContours.slope
    norm_num
    ring
  have hden : HasDerivAt (fun t : ℝ => slope (lift c t))
      (2 * angleVelocity (lift c θ)) θ := (hc.const_mul 2).add_const 1
  have hs := slope_lift_ne_zero c θ hA hbranch
  convert hnum.fun_div hden hs using 1
  field_simp [hs]

theorem hasDerivAt_deriv_lift (c : ℂ) (θ : ℝ) (hA : slope c ≠ 0)
    (hbranch : relativeDiscriminant c θ ∈ Complex.slitPlane) :
    HasDerivAt (deriv (fun t : ℝ => lift c t))
      (angleVelocity (lift c θ) * (Complex.I *
        (1 - 2 * orbit (lift c θ) / slope (lift c θ) ^ 2))) θ := by
  apply (hasDerivAt_lift_angleVelocity c θ hA hbranch).congr_of_eventuallyEq
  have hcontinuous : Continuous (fun t : ℝ => relativeDiscriminant c t) := by
    unfold relativeDiscriminant rotation
    fun_prop
  filter_upwards [hcontinuous.continuousAt.preimage_mem_nhds
    (Complex.isOpen_slitPlane.mem_nhds hbranch)] with t ht
  exact (hasDerivAt_lift c t hA ht).deriv

theorem lift_acceleration_div_velocity (c : ℂ) (θ : ℝ) (hp : orbit c ≠ 0)
    (hA : slope c ≠ 0) (hbranch : relativeDiscriminant c θ ∈ Complex.slitPlane) :
    deriv (deriv (fun t : ℝ => lift c t)) θ / deriv (fun t : ℝ => lift c t) θ =
      Complex.I * (1 - 2 * orbit (lift c θ) / slope (lift c θ) ^ 2) := by
  rw [(hasDerivAt_lift c θ hA hbranch).deriv,
    (hasDerivAt_deriv_lift c θ hA hbranch).deriv]
  have hv : angleVelocity (lift c θ) ≠ 0 := by
    unfold angleVelocity
    rw [orbit_lift c θ hA]
    exact div_ne_zero (mul_ne_zero Complex.I_ne_zero
      (mul_ne_zero hp (Complex.exp_ne_zero _))) (slope_lift_ne_zero c θ hA hbranch)
  rw [mul_comm (angleVelocity (lift c θ)), mul_div_cancel_right₀ _ hv]

theorem lift_heading_rate (c : ℂ) (θ : ℝ) (hp : orbit c ≠ 0)
    (hA : slope c ≠ 0) (hbranch : relativeDiscriminant c θ ∈ Complex.slitPlane) :
    (deriv (deriv (fun t : ℝ => lift c t)) θ / deriv (fun t : ℝ => lift c t) θ).im =
      (1 + (1 / (1 + 4 * orbit (lift c θ))).re) / 2 := by
  rw [lift_acceleration_div_velocity c θ hp hA hbranch]
  have hdisc : slope (lift c θ) ^ 2 = 1 + 4 * orbit (lift c θ) := by
    unfold OrbitTrapContours.slope orbit
    ring
  have hreal : 1 - 2 * orbit (lift c θ) / slope (lift c θ) ^ 2 =
      (1 + 1 / slope (lift c θ) ^ 2) / 2 := by
    field_simp [slope_lift_ne_zero c θ hA hbranch]
    linear_combination hdisc
  rw [hreal, hdisc]
  simp

end

end FractalSync.OrbitTrapCurvature
