import FractalSync.SensitivityBounds
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.InnerProductSpace.Calculus

/-!
# Residual Hessians and finite-horizon response

The loss calculation follows a real parameter line in a Hilbert observable.
The response bound uses displacement norms measured in the chosen state and
action metrics. Both results state the comparison premises explicitly.
-/

namespace FractalSync.SensitivityResponse

open Filter
open scoped InnerProductSpace Topology

variable {H : Type*} [NormedAddCommGroup H] [InnerProductSpace ℝ H]

noncomputable def loss (r : ℝ → H) (x : ℝ) : ℝ := ⟪r x, r x⟫_ℝ / 2

theorem hasDerivAt_loss {r : ℝ → H} {x : ℝ} {j : H} (h : HasDerivAt r j x) :
    HasDerivAt (loss r) ⟪r x, j⟫_ℝ x := by
  convert (h.inner ℝ h).div_const 2 using 1
  rw [real_inner_comm j (r x)]
  ring

theorem hasDerivAt_loss_deriv {r j : ℝ → H} {x : ℝ} {b : H}
    (hr : ∀ᶠ t in 𝓝 x, HasDerivAt r (j t) t) (hj : HasDerivAt j b x) :
    HasDerivAt (deriv (loss r)) (‖j x‖ ^ 2 + ⟪r x, b⟫_ℝ) x := by
  have ha : deriv (loss r) =ᶠ[𝓝 x] fun t => ⟪r t, j t⟫_ℝ := by
    filter_upwards [hr] with t ht using (hasDerivAt_loss ht).deriv
  have h := ((hr.self_of_nhds).inner ℝ hj).congr_of_eventuallyEq ha
  simpa [real_inner_self_eq_norm_sq, add_comm] using h

/-- Pointwise residual correction to the Gauss--Newton bounds along a line. -/
theorem residual_hessian_bounds (r j b : H) {l u m radius : ℝ}
    (hl : 0 ≤ l) (hradius : 0 ≤ radius)
    (hlo : l * radius ≤ ‖j‖) (hhi : ‖j‖ ≤ u * radius)
    (hsecond : ‖b‖ ≤ m * radius ^ 2) :
    (l ^ 2 - m * ‖r‖) * radius ^ 2 ≤ ‖j‖ ^ 2 + ⟪r, b⟫_ℝ ∧
    ‖j‖ ^ 2 + ⟪r, b⟫_ℝ ≤ (u ^ 2 + m * ‖r‖) * radius ^ 2 := by
  have hsqlo := pow_le_pow_left₀ (mul_nonneg hl hradius) hlo 2
  have hsqhi := pow_le_pow_left₀ (norm_nonneg j) hhi 2
  rw [mul_pow] at hsqlo hsqhi
  have hcorrection := (abs_real_inner_le_norm r b).trans
    (mul_le_mul_of_nonneg_left hsecond (norm_nonneg r))
  rw [abs_le] at hcorrection
  constructor
  · calc
      _ = l ^ 2 * radius ^ 2 - ‖r‖ * (m * radius ^ 2) := by ring
      _ ≤ _ := add_le_add hsqlo hcorrection.1
  · calc
      _ ≤ u ^ 2 * radius ^ 2 + ‖r‖ * (m * radius ^ 2) := add_le_add hsqhi hcorrection.2
      _ = _ := by ring

theorem loss_deriv_deriv_bounds {r j : ℝ → H} {x : ℝ} {b : H}
    (hr : ∀ᶠ t in 𝓝 x, HasDerivAt r (j t) t) (hj : HasDerivAt j b x)
    {l u m radius : ℝ} (hl : 0 ≤ l) (hradius : 0 ≤ radius)
    (hlo : l * radius ≤ ‖j x‖) (hhi : ‖j x‖ ≤ u * radius)
    (hsecond : ‖b‖ ≤ m * radius ^ 2) :
    (l ^ 2 - m * ‖r x‖) * radius ^ 2 ≤ deriv (deriv (loss r)) x ∧
    deriv (deriv (loss r)) x ≤ (u ^ 2 + m * ‖r x‖) * radius ^ 2 := by
  rw [(hasDerivAt_loss_deriv hr hj).deriv]
  exact residual_hessian_bounds (r x) (j x) b hl hradius hlo hhi hsecond

omit [InnerProductSpace ℝ H] in
/-- The observation and reachable-control margins multiply. The state norm
may already incorporate a block metric along the finite horizon. -/
theorem action_response_bounds {E : Type*} [SeminormedAddCommGroup E]
    (motion : E) (response : H) {betaLower betaUpper lambdaLower lambdaUpper radius : ℝ}
    (hbeta : 0 ≤ betaLower) (hr : 0 ≤ radius)
    (hlambdaLower : 0 ≤ lambdaLower) (hlambdaUpper : 0 ≤ lambdaUpper)
    (hmlo : betaLower * radius ≤ ‖motion‖) (hmhi : ‖motion‖ ≤ betaUpper * radius)
    (hvlo : lambdaLower * ‖motion‖ ^ 2 ≤ ‖response‖ ^ 2)
    (hvhi : ‖response‖ ^ 2 ≤ lambdaUpper * ‖motion‖ ^ 2) :
    (lambdaLower * betaLower ^ 2) * radius ^ 2 ≤ ‖response‖ ^ 2 ∧
    ‖response‖ ^ 2 ≤ (lambdaUpper * betaUpper ^ 2) * radius ^ 2 := by
  have hm := FractalSync.sensitivity_squared_transfer motion motion betaLower betaUpper 0 radius
    hmlo hmhi (by simp) hr hbeta
  simp only [sub_zero, add_zero] at hm
  constructor
  · simpa [mul_assoc] using (mul_le_mul_of_nonneg_left hm.1 hlambdaLower).trans hvlo
  · simpa [mul_assoc] using hvhi.trans (mul_le_mul_of_nonneg_left hm.2 hlambdaUpper)

end FractalSync.SensitivityResponse
