import FractalSync.RadialProfile

/-!
# Same-sign closed form for the radial profile

The stable primitive is stated over the reals. Its two terms are nonnegative
for nonnegative anisotropy and radius. A machine implementation still needs
appropriate intermediate and final range handling.
-/

namespace FractalSync.DistanceGraph

/-- Difference identity obtained from the hyperbolic sine subtraction law. -/
theorem arsinh_sub_eq_arsinh (x y : ℝ) :
    Real.arsinh x - Real.arsinh y =
      Real.arsinh (x * Real.sqrt (1 + y ^ 2) - Real.sqrt (1 + x ^ 2) * y) := by
  apply Real.sinh_injective
  simp [Real.sinh_sub, Real.sinh_arsinh, Real.cosh_arsinh]

/-- Rationalizing the small coefficient avoids subtracting nearly equal values. -/
theorem radial_small_coefficient (a : ℝ) :
    Real.sqrt (1 + a ^ 2) - a = (Real.sqrt (1 + a ^ 2) + a)⁻¹ := by
  have rootSquare : Real.sqrt (1 + a ^ 2) ^ 2 = 1 + a ^ 2 :=
    Real.sq_sqrt (by positivity)
  have rootLower : -a < Real.sqrt (1 + a ^ 2) := Real.lt_sqrt_of_sq_lt (by nlinarith)
  have denominator : 0 < Real.sqrt (1 + a ^ 2) + a := by linarith
  rw [← one_div]
  apply (eq_div_iff denominator.ne').2
  nlinarith only [rootSquare]

/-- The two inverse-hyperbolic-sine arguments share this square-root factor. -/
theorem radial_sqrt_ratio (a q : ℝ) :
    Real.sqrt (1 + (a * q / Real.sqrt (1 + q ^ 2)) ^ 2) =
      Real.sqrt (1 + (1 + a ^ 2) * q ^ 2) / Real.sqrt (1 + q ^ 2) := by
  let d := Real.sqrt (1 + q ^ 2)
  let s := Real.sqrt (1 + (1 + a ^ 2) * q ^ 2)
  change Real.sqrt (1 + (a * q / d) ^ 2) = s / d
  have hd : 0 < d := by dsimp [d]; positivity
  have hs : 0 ≤ s := Real.sqrt_nonneg _
  have hdSq : d ^ 2 = 1 + q ^ 2 := Real.sq_sqrt (by positivity)
  have hsSq : s ^ 2 = 1 + (1 + a ^ 2) * q ^ 2 := Real.sq_sqrt (by positivity)
  apply (sq_eq_sq₀ (Real.sqrt_nonneg _) (div_nonneg hs hd.le)).1
  rw [Real.sq_sqrt (by positivity : 0 ≤ 1 + (a * q / d) ^ 2)]
  simp only [div_pow, hsSq]
  have hdNe : d ^ 2 ≠ 0 := pow_ne_zero 2 hd.ne'
  apply (eq_div_iff hdNe).2
  rw [add_mul, one_mul, div_mul_cancel₀ _ hdNe, hdSq]
  ring

/-- The logarithmic part of the checked primitive is one inverse hyperbolic sine. -/
theorem radial_arsinh_ratio_eq_logs (a q : ℝ) :
    Real.arsinh (a * q / Real.sqrt (1 + q ^ 2)) =
      Real.log (Real.sqrt (1 + (1 + a ^ 2) * q ^ 2) + a * q) -
        Real.log (1 + q ^ 2) / 2 := by
  unfold Real.arsinh
  rw [radial_sqrt_ratio, ← add_div, add_comm (a * q),
    Real.log_div (radial_log_argument_pos a q).ne' (by positivity), Real.log_sqrt (by positivity)]

/-- The normalized radial primitive with no subtraction between its two terms. -/
noncomputable def stableNormalizedRadialPrimitive (a q : ℝ) : ℝ :=
  let b := Real.sqrt (1 + a ^ 2)
  let k := (b + a)⁻¹
  k * Real.arsinh (b * q) +
    a * Real.arsinh (k * q * (Real.sqrt (1 + (b * q) ^ 2) / Real.sqrt (1 + q ^ 2)))

/-- The stable expression equals the existing logarithmic primitive exactly. -/
theorem normalizedRadialPrimitive_eq_stable (a q : ℝ) :
    normalizedRadialPrimitive a q = stableNormalizedRadialPrimitive a q := by
  let b := Real.sqrt (1 + a ^ 2)
  let k := (b + a)⁻¹
  let d := Real.sqrt (1 + q ^ 2)
  let s := Real.sqrt (1 + (1 + a ^ 2) * q ^ 2)
  have coefficient : b - a = k := radial_small_coefficient a
  have root₁ : Real.sqrt (1 + (b * q) ^ 2) = s := by
    dsimp [b, s]
    rw [mul_pow, Real.sq_sqrt (by positivity)]
  have root₂ : Real.sqrt (1 + (a * q / d) ^ 2) = s / d := radial_sqrt_ratio a q
  have differenceArgument : b * q * Real.sqrt (1 + (a * q / d) ^ 2) -
      Real.sqrt (1 + (b * q) ^ 2) * (a * q / d) = k * q * (s / d) := by
    rw [root₁, root₂, ← coefficient]
    ring
  calc
    normalizedRadialPrimitive a q = b * Real.arsinh (b * q) -
        a * Real.arsinh (a * q / d) := by
      have logPair : Real.arsinh (a * q / d) = Real.log (s + a * q) -
          Real.log (1 + q ^ 2) / 2 := radial_arsinh_ratio_eq_logs a q
      unfold normalizedRadialPrimitive
      rw [logPair]
      dsimp [b, s]
      ring
    _ = (b - a) * Real.arsinh (b * q) +
        a * (Real.arsinh (b * q) - Real.arsinh (a * q / d)) := by ring
    _ = stableNormalizedRadialPrimitive a q := by
      rw [coefficient, arsinh_sub_eq_arsinh, differenceArgument]
      unfold stableNormalizedRadialPrimitive
      rw [← root₁]

/-- The stable expression evaluates the original interval-integral radial profile. -/
theorem radialProfile_eq_stable (epsilon a r : ℝ) (he : 0 < epsilon) :
    radialProfile epsilon a r = stableNormalizedRadialPrimitive a (r / epsilon) := by
  rw [radialProfile_closed_form epsilon a r he, normalizedRadialPrimitive_eq_stable]

theorem stableNormalizedRadialPrimitive_zero_radius (a : ℝ) :
    stableNormalizedRadialPrimitive a 0 = 0 := by
  simp [stableNormalizedRadialPrimitive]

theorem stableNormalizedRadialPrimitive_zero_anisotropy (q : ℝ) :
    stableNormalizedRadialPrimitive 0 q = Real.arsinh q := by
  simp [stableNormalizedRadialPrimitive]

/-- On the physical domain, the stable formula adds two nonnegative terms. -/
theorem stableRadialTerms_nonneg (a q : ℝ) (ha : 0 ≤ a) (hq : 0 ≤ q) :
    let b := Real.sqrt (1 + a ^ 2)
    let k := (b + a)⁻¹
    0 ≤ k * Real.arsinh (b * q) ∧
      0 ≤ a * Real.arsinh (k * q * (Real.sqrt (1 + (b * q) ^ 2) / Real.sqrt (1 + q ^ 2))) := by
  dsimp only
  constructor
  · exact mul_nonneg (by positivity) (Real.arsinh_nonneg_iff.2 (by positivity))
  · exact mul_nonneg ha (Real.arsinh_nonneg_iff.2 (by positivity))

/-- A scaled form of the second argument avoids the product `b * q` under its square root. -/
theorem stableRadialSecondArgument_eq_scaled (a q : ℝ) :
    let b := Real.sqrt (1 + a ^ 2)
    (b + a)⁻¹ * q * (Real.sqrt (1 + (b * q) ^ 2) / Real.sqrt (1 + q ^ 2)) =
      q / (1 + a / b) * (Real.sqrt (b⁻¹ ^ 2 + q ^ 2) / Real.sqrt (1 + q ^ 2)) := by
  let b := Real.sqrt (1 + a ^ 2)
  change (b + a)⁻¹ * q * (Real.sqrt (1 + (b * q) ^ 2) / Real.sqrt (1 + q ^ 2)) =
    q / (1 + a / b) * (Real.sqrt (b⁻¹ ^ 2 + q ^ 2) / Real.sqrt (1 + q ^ 2))
  have hb : 0 < b := by dsimp [b]; positivity
  have rootFactor : b * Real.sqrt (b⁻¹ ^ 2 + q ^ 2) = Real.sqrt (1 + (b * q) ^ 2) := by
    apply (sq_eq_sq₀ (mul_nonneg hb.le (Real.sqrt_nonneg _)) (Real.sqrt_nonneg _)).1
    rw [mul_pow, Real.sq_sqrt (by positivity : 0 ≤ b⁻¹ ^ 2 + q ^ 2),
      Real.sq_sqrt (by positivity : 0 ≤ 1 + (b * q) ^ 2), mul_pow]
    rw [mul_add, ← mul_pow, mul_inv_cancel₀ hb.ne', one_pow]
  have denominator : 1 + a / b = (b + a) / b := by
    rw [add_div, div_self hb.ne']
  have coefficient : (b + a)⁻¹ * b = (1 + a / b)⁻¹ := by
    rw [denominator, inv_div, div_eq_mul_inv, mul_comm]
  rw [← rootFactor, div_eq_mul_inv q]
  calc
    _ = ((b + a)⁻¹ * b) * q * (Real.sqrt (b⁻¹ ^ 2 + q ^ 2) / Real.sqrt (1 + q ^ 2)) := by ring
    _ = _ := by rw [coefficient, mul_comm q]

end FractalSync.DistanceGraph
