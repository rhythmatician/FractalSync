import FractalSync.OrbitTrapSteering
import Mathlib.Analysis.Complex.RealDeriv
import Mathlib.Analysis.SpecialFunctions.Pow.Deriv

/-!
# Angular lifts of the second-iterate point trap

The complex polynomial is identified with the existing coordinate trap.
A principal square root normalized at the starting point gives the actual
angle derivative whenever the normalized discriminant avoids its branch cut.
-/

namespace FractalSync.OrbitTrapContours

open OrbitTrapSteering

noncomputable section

def coordinates (z : ℂ) : Point := (z.re, z.im)

def orbit (z : ℂ) : ℂ := z ^ 2 + z

def slope (z : ℂ) : ℂ := 2 * z + 1

def rotation (θ : ℂ) : ℂ := Complex.exp (θ * Complex.I)

def relativeDiscriminant (c θ : ℂ) : ℂ :=
  1 + 4 * orbit c * (rotation θ - 1) / slope c ^ 2

def lift (c θ : ℂ) : ℂ :=
  (slope c * relativeDiscriminant c θ ^ (1 / 2 : ℂ) - 1) / 2

def angleVelocity (z : ℂ) : ℂ := Complex.I * orbit z / slope z

theorem trap_coordinates (z : ℂ) : trap (coordinates z) = ‖orbit z‖ ^ 2 := by
  rw [Complex.sq_norm, Complex.normSq_apply]
  simp [trap, coordinates, orbitRe, orbitIm, orbit, pow_two, Complex.mul_re, Complex.mul_im]
  ring

theorem lift_zero (c : ℂ) : lift c 0 = c := by
  simp [lift, slope, relativeDiscriminant, rotation, Complex.one_cpow]

private theorem slope_sq (z : ℂ) : slope z ^ 2 = 1 + 4 * orbit z := by
  unfold slope orbit
  ring

private theorem sq_half_cpow (z : ℂ) : (z ^ (1 / 2 : ℂ)) ^ (2 : ℕ) = z := by
  rw [← Complex.cpow_nat_mul]
  norm_num [Complex.cpow_one]

theorem slope_lift (c θ : ℂ) :
    slope (lift c θ) = slope c * relativeDiscriminant c θ ^ (1 / 2 : ℂ) := by
  simp only [slope, lift]
  ring

theorem orbit_lift (c θ : ℂ) (hA : slope c ≠ 0) :
    orbit (lift c θ) = orbit c * rotation θ := by
  have hs : slope (lift c θ) ^ 2 = 1 + 4 * (orbit c * rotation θ) := by
    rw [slope_lift, mul_pow, sq_half_cpow]
    unfold relativeDiscriminant
    field_simp
    linear_combination slope_sq c
  linear_combination (slope_sq (lift c θ) - hs) / (-4)

theorem trap_lift (c : ℂ) (θ : ℝ) (hA : slope c ≠ 0) :
    trap (coordinates (lift c θ)) = trap (coordinates c) := by
  rw [trap_coordinates, trap_coordinates, orbit_lift c θ hA, norm_mul]
  simp [rotation, Complex.norm_exp_ofReal_mul_I]

theorem slitPlane_of_step_guard (c θ : ℂ)
    (h : ‖4 * orbit c * (rotation θ - 1)‖ < ‖slope c‖ ^ 2) :
    relativeDiscriminant c θ ∈ Complex.slitPlane := by
  apply Complex.mem_slitPlane_of_norm_lt_one
  rw [norm_div, norm_pow]
  exact (div_lt_iff₀ (lt_of_le_of_lt (norm_nonneg _) h)).2 (by simpa using h)

theorem slope_lift_ne_zero (c θ : ℂ) (hA : slope c ≠ 0)
    (hbranch : relativeDiscriminant c θ ∈ Complex.slitPlane) :
    slope (lift c θ) ≠ 0 := by
  rw [slope_lift]
  exact mul_ne_zero hA (Complex.cpow_ne_zero_iff.2
    (Or.inl (Complex.slitPlane_ne_zero hbranch)))

private theorem hasDerivAt_rotation (θ : ℂ) :
    HasDerivAt rotation (Complex.I * rotation θ) θ := by
  simpa [rotation, mul_comm] using
    (Complex.hasDerivAt_exp (θ * Complex.I)).comp θ ((hasDerivAt_id θ).mul_const Complex.I)

private theorem differentiableAt_lift (c θ : ℂ)
    (hbranch : relativeDiscriminant c θ ∈ Complex.slitPlane) :
    DifferentiableAt ℂ (lift c) θ := by
  have hu : DifferentiableAt ℂ (relativeDiscriminant c) θ :=
    ((((hasDerivAt_rotation θ).differentiableAt.sub_const 1).const_mul
      (4 * orbit c)).div_const (slope c ^ 2)).const_add 1
  exact ((((hu.hasDerivAt.cpow_const hbranch).differentiableAt.const_mul
    (slope c)).sub_const 1).div_const 2)

theorem hasDerivAt_lift (c : ℂ) (θ : ℝ) (hA : slope c ≠ 0)
    (hbranch : relativeDiscriminant c θ ∈ Complex.slitPlane) :
    HasDerivAt (fun t : ℝ => lift c t) (angleVelocity (lift c θ)) θ := by
  have hc := ((differentiableAt_lift c θ hbranch).hasDerivAt).comp_ofReal
  have horbit := (hc.pow 2).add hc
  change HasDerivAt (fun t : ℝ => lift c t ^ (2 : ℕ) + lift c t) _ θ at horbit
  have hrot := (((hasDerivAt_rotation (θ : ℂ)).comp_ofReal).const_mul (orbit c))
  have heq : (fun t : ℝ => lift c t ^ (2 : ℕ) + lift c t) =
      (fun t : ℝ => orbit c * rotation t) := funext fun t => orbit_lift c t hA
  rw [heq] at horbit
  have hv := horbit.unique hrot
  have hne := slope_lift_ne_zero c θ hA hbranch
  convert hc using 1
  unfold angleVelocity
  rw [eq_comm, eq_div_iff hne, orbit_lift c θ hA]
  unfold slope
  linear_combination hv

theorem lift_tangent (c : ℂ) (θ : ℝ) (hA : slope c ≠ 0)
    (hbranch : relativeDiscriminant c θ ∈ Complex.slitPlane) :
    fderiv ℝ trap (coordinates (lift c θ)) (coordinates (angleVelocity (lift c θ))) = 0 := by
  let toCoordinates : ℂ →L[ℝ] Point := Complex.reCLM.prod Complex.imCLM
  have hpath : HasDerivAt (fun t : ℝ => coordinates (lift c t))
      (coordinates (angleVelocity (lift c θ))) θ := by
    have h := (toCoordinates.hasFDerivAt (x := lift c θ)).comp_hasDerivAt θ
      (hasDerivAt_lift c θ hA hbranch)
    simpa [toCoordinates, coordinates, Function.comp_def] using h
  have htrap := hasFDerivAt_trap (coordinates (lift c θ))
  have hderivative := htrap.comp_hasDerivAt θ hpath
  have heq : (trap ∘ fun t : ℝ => coordinates (lift c t)) =
      fun _ : ℝ => trap (coordinates c) := funext fun t => trap_lift c t hA
  rw [heq] at hderivative
  rw [htrap.fderiv]
  exact hderivative.unique (hasDerivAt_const θ (trap (coordinates c)))

theorem angleVelocity_metricSpeedSq (ρ : ℝ) (c : ℂ) :
    metricSpeedSq ρ (coordinates (angleVelocity c)) = (‖orbit c‖ / (ρ * ‖slope c‖)) ^ 2 := by
  unfold metricSpeedSq pairing coordinates
  rw [← Complex.normSq_apply, ← Complex.sq_norm]
  simp [angleVelocity, div_pow, mul_pow, div_div, mul_comm]

theorem lift_metricSpeedSq (ρ : ℝ) (c : ℂ) (θ : ℝ) (hA : slope c ≠ 0) :
    metricSpeedSq ρ (coordinates (angleVelocity (lift c θ))) =
      (‖orbit c‖ / (ρ * ‖slope (lift c θ)‖)) ^ 2 := by
  rw [angleVelocity_metricSpeedSq, orbit_lift c θ hA, norm_mul]
  simp [rotation, Complex.norm_exp_ofReal_mul_I]

end

end FractalSync.OrbitTrapContours
