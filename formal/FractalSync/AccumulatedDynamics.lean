import FractalSync.MovingMetricMechanics
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# Residual-aware finite-step dynamics

The actual phase-energy differential identifies the power contributed by an
acceleration residual. Local Taylor bounds then accumulate along the actual
ordered recurrence, including impulses at the new position. Whole-segment
domain bounds concern the proposed polygonal path, not the exact ODE path.
-/

open Set
open scoped InnerProductSpace BigOperators NNReal

namespace FractalSync.ScaleMetric

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

noncomputable local instance : NormedSpace ℝ (E × E) := Prod.normedSpace

/-- The actual failure of the proposed acceleration to satisfy the ideal
connection/force equation. It includes provider and arithmetic errors only
when those have separately been related to these exact-real quantities. -/
noncomputable def dynamicsResidual (rho a : ℝ) (u : E) (H : E →L[ℝ] E)
    (v acc gradU Q : E) : E :=
  acc + connection rho a u H v v - inverseAction rho a u (Q - gradU)

/-- The actual first-order power error is exactly the metric pairing with
the acceleration residual. No exact equation-of-motion hypothesis is needed. -/
theorem fderiv_phaseEnergy_motion_residual (a : ℝ) (rho potential : E → ℝ)
    (gradient : E → E) (x v acc gradU Q : E) (H : E →L[ℝ] E)
    (hr : HasFDerivAt rho (innerSL ℝ (gradient x)) x)
    (hu : HasFDerivAt gradient H x)
    (hU : HasFDerivAt potential (innerSL ℝ gradU) x) (hrho : rho x ≠ 0) :
    fderiv ℝ (phaseEnergy a rho potential gradient) (x, v) (v, acc) =
      ⟪v, Q⟫_ℝ + metric (rho x) a (gradient x) v
        (dynamicsResidual (rho x) a (gradient x) H v acc gradU Q) := by
  rw [fderiv_phaseEnergy_apply a rho potential gradient x v v acc gradU H hr hu hU hrho,
    connection_metric_compatible (rho x) a (gradient x) H v v v hrho,
    metric_symm (rho x) a (gradient x) (connection (rho x) a (gradient x) H v v) v]
  have hm : acc + connection (rho x) a (gradient x) H v v =
      inverseAction (rho x) a (gradient x) (Q - gradU) +
        dynamicsResidual (rho x) a (gradient x) H v acc gradU Q := by
    simp [dynamicsResidual]
  have hp := congrArg (metric (rho x) a (gradient x) v) hm
  rw [metric_add_right, metric_add_right,
    metric_inverse_pairing (rho x) a (gradient x) v (Q - gradU) hrho,
    inner_sub_right, real_inner_comm gradU v] at hp
  linarith

/-- The ordered phase increment retains the residual power and its extra
position-only term. -/
theorem fderiv_phaseEnergy_step_increment_residual (a dt : ℝ)
    (rho potential : E → ℝ) (gradient : E → E)
    (x v acc gradU Q : E) (H : E →L[ℝ] E)
    (hr : HasFDerivAt rho (innerSL ℝ (gradient x)) x)
    (hu : HasFDerivAt gradient H x)
    (hU : HasFDerivAt potential (innerSL ℝ gradU) x) (hrho : rho x ≠ 0) :
    fderiv ℝ (phaseEnergy a rho potential gradient) (x, v)
      (dt • (velocityEulerStep dt v acc, acc)) =
      dt * (⟪v, Q⟫_ℝ + metric (rho x) a (gradient x) v
        (dynamicsResidual (rho x) a (gradient x) H v acc gradU Q)) +
      dt ^ 2 * positionEnergyRate (rho x) a (gradient x) H v gradU acc := by
  rw [positionVelocityStep_increment, map_add, map_smul, map_smul,
    fderiv_phaseEnergy_motion_residual a rho potential gradient x v acc gradU Q H hr hu hU hrho,
    fderiv_phaseEnergy_apply a rho potential gradient x v acc 0 gradU H hr hu hU hrho]
  simp [positionEnergyRate, metric, smul_eq_mul]

/-- Residual-aware local energy defect for the actual changing-position step.
The first-order residual contribution is explicit; only the analytic remainder
uses a differential variation certificate along the actual proposed segment. -/
theorem positionVelocityStep_energy_defect_residual (a dt L : ℝ)
    (rho potential : E → ℝ) (gradient : E → E)
    (x v acc gradU Q : E) (H : E →L[ℝ] E)
    (hr : HasFDerivAt rho (innerSL ℝ (gradient x)) x)
    (hu : HasFDerivAt gradient H x)
    (hU : HasFDerivAt potential (innerSL ℝ gradU) x) (hrho : rho x ≠ 0)
    (hD : ∀ t ∈ Icc (0 : ℝ) 1,
      DifferentiableAt ℝ (phaseEnergy a rho potential gradient)
        ((x, v) + t • (dt • (velocityEulerStep dt v acc, acc))))
    (hbound : ∀ t ∈ Icc (0 : ℝ) 1,
      ‖fderiv ℝ (phaseEnergy a rho potential gradient)
          ((x, v) + t • (dt • (velocityEulerStep dt v acc, acc))) -
        fderiv ℝ (phaseEnergy a rho potential gradient) (x, v)‖ ≤
          L * ‖t • (dt • (velocityEulerStep dt v acc, acc))‖) :
    |phaseEnergy a rho potential gradient (positionVelocityStep dt x v acc) -
      phaseEnergy a rho potential gradient (x, v) - dt * ⟪v, Q⟫_ℝ| ≤
      |dt| * |metric (rho x) a (gradient x) v
        (dynamicsResidual (rho x) a (gradient x) H v acc gradU Q)| +
      dt ^ 2 * (|positionEnergyRate (rho x) a (gradient x) H v gradU acc| +
        L / 2 * (max ‖velocityEulerStep dt v acc‖ ‖acc‖) ^ 2) := by
  have hR := NormalizedDistanceFields.field_quadratic_remainder
    (phaseEnergy a rho potential gradient) (x, v)
    (dt • (velocityEulerStep dt v acc, acc)) L hD hbound
  rw [← positionVelocityStep_eq_add,
    fderiv_phaseEnergy_step_increment_residual a dt rho potential gradient x v acc gradU Q H
      hr hu hU hrho] at hR
  have ht := abs_sub_le
    (phaseEnergy a rho potential gradient (positionVelocityStep dt x v acc) -
      phaseEnergy a rho potential gradient (x, v))
    (dt * (⟪v, Q⟫_ℝ + metric (rho x) a (gradient x) v
      (dynamicsResidual (rho x) a (gradient x) H v acc gradU Q)) +
      dt ^ 2 * positionEnergyRate (rho x) a (gradient x) H v gradU acc)
    (dt * ⟪v, Q⟫_ℝ)
  have hb := abs_add_le
    (dt * metric (rho x) a (gradient x) v
      (dynamicsResidual (rho x) a (gradient x) H v acc gradU Q))
    (dt ^ 2 * positionEnergyRate (rho x) a (gradient x) H v gradU acc)
  simp only [mul_add, add_assoc, add_sub_cancel_left] at ht hR
  simp only [abs_mul, abs_sq] at hb
  simp only [norm_smul, Real.norm_eq_abs, Prod.norm_def, mul_pow, sq_abs] at hR
  linarith

/-- Pointwise operator bound for the actual bilinear metric, including the
zero-gradient case. This turns an acceleration residual norm into power error. -/
theorem abs_metric_le_norm (rho a : ℝ) (u v w : E) :
    |metric rho a u v w| ≤ weight a u / rho ^ 2 * ‖v‖ * ‖w‖ := by
  refine abs_le_of_sq_le_sq ?_ (mul_nonneg
    (mul_nonneg (div_nonneg (weight_pos a u).le (sq_nonneg rho)) (norm_nonneg v))
    (norm_nonneg w))
  have hp := mul_le_mul (metric_norm_upper rho a u v) (metric_norm_upper rho a u w)
    (metric_self_nonneg rho a u w)
    (div_nonneg (mul_nonneg (weight_pos a u).le (sq_nonneg ‖v‖)) (sq_nonneg rho))
  convert (metric_cauchy_schwarz_sq rho a u v w).trans hp using 1
  ring

/-- A separately estimated force acceleration and connection contribution
give an explicit norm budget for the actual dynamics residual. -/
theorem dynamicsResidual_norm_le (rho a etaForce etaConnection : ℝ)
    (u v acc gradU Q forceEstimate connectionEstimate : E) (H : E →L[ℝ] E)
    (hacc : acc = forceEstimate - connectionEstimate)
    (hforce : ‖forceEstimate - inverseAction rho a u (Q - gradU)‖ ≤ etaForce)
    (hconnection : ‖connection rho a u H v v - connectionEstimate‖ ≤ etaConnection) :
    ‖dynamicsResidual rho a u H v acc gradU Q‖ ≤ etaForce + etaConnection := by
  rw [dynamicsResidual, hacc]
  convert (norm_add_le (forceEstimate - inverseAction rho a u (Q - gradU))
    (connection rho a u H v v - connectionEstimate)).trans
      (add_le_add hforce hconnection) using 1
  congr 1
  abel

/-- Actual local regularity data for one force step. Its fields are derivative
and variation statements, not an assumed energy-error estimate. -/
structure StepJetCertificate (a dt : ℝ) (rho potential : E → ℝ)
    (gradient : E → E) (x v acc : E) where
  H : E →L[ℝ] E
  gradU : E
  L : ℝ≥0
  ruler_ne : rho x ≠ 0
  ruler_deriv : HasFDerivAt rho (innerSL ℝ (gradient x)) x
  gradient_deriv : HasFDerivAt gradient H x
  potential_deriv : HasFDerivAt potential (innerSL ℝ gradU) x
  differentiable : ∀ t ∈ Icc (0 : ℝ) 1,
    DifferentiableAt ℝ (phaseEnergy a rho potential gradient)
      ((x, v) + t • (dt • (velocityEulerStep dt v acc, acc)))
  variation : ∀ t ∈ Icc (0 : ℝ) 1,
    ‖fderiv ℝ (phaseEnergy a rho potential gradient)
        ((x, v) + t • (dt • (velocityEulerStep dt v acc, acc))) -
      fderiv ℝ (phaseEnergy a rho potential gradient) (x, v)‖ ≤
        (L : ℝ) * ‖t • (dt • (velocityEulerStep dt v acc, acc))‖

/-- The explicit local budget determined by actual regularity and the measured
acceleration residual. -/
noncomputable def StepJetCertificate.errorBudget {a dt : ℝ} {rho potential : E → ℝ}
    {gradient : E → E} {x v acc : E}
    (c : StepJetCertificate a dt rho potential gradient x v acc) (Q : E) : ℝ :=
  |dt| * |metric (rho x) a (gradient x) v
    (dynamicsResidual (rho x) a (gradient x) c.H v acc c.gradU Q)| +
  dt ^ 2 * (|positionEnergyRate (rho x) a (gradient x) c.H v c.gradU acc| +
    (c.L : ℝ) / 2 * (max ‖velocityEulerStep dt v acc‖ ‖acc‖) ^ 2)

theorem StepJetCertificate.energy_defect {a dt : ℝ} {rho potential : E → ℝ}
    {gradient : E → E} {x v acc : E}
    (c : StepJetCertificate a dt rho potential gradient x v acc) (Q : E) :
    |phaseEnergy a rho potential gradient (positionVelocityStep dt x v acc) -
      phaseEnergy a rho potential gradient (x, v) - dt * ⟪v, Q⟫_ℝ| ≤ c.errorBudget Q := by
  exact positionVelocityStep_energy_defect_residual a dt c.L rho potential gradient
    x v acc c.gradU Q c.H c.ruler_deriv c.gradient_deriv c.potential_deriv c.ruler_ne
    c.differentiable c.variation

/-- Signed work-error accumulation over a finite history. The zero-step case
is included; local errors may have arbitrary signs. -/
theorem accumulated_energy_defect (N : ℕ) (energy : ℕ → ℝ) (work error : Fin N → ℝ)
    (hlocal : ∀ i : Fin N, |energy (i.val + 1) - energy i.val - work i| ≤ error i) :
    |energy N - energy 0 - ∑ i : Fin N, work i| ≤ ∑ i : Fin N, error i := by
  have htel : (∑ i : Fin N, (energy (i.val + 1) - energy i.val)) = energy N - energy 0 := by
    rw [Fin.sum_univ_eq_sum_range (fun i => energy (i + 1) - energy i) N,
      Finset.sum_range_sub]
  rw [← htel, ← Finset.sum_sub_distrib]
  exact (Finset.abs_sum_le_sum_abs _ _).trans (Finset.sum_le_sum (fun i _ => hlocal i))

/-- Adding the exact new-position impulse ledger leaves the force-step
residual budget unchanged. -/
theorem StepJetCertificate.impulse_energy_defect {a dt : ℝ}
    {rho potential : E → ℝ} {gradient : E → E} {x v acc : E}
    (c : StepJetCertificate a dt rho potential gradient x v acc) (Q J : E)
    (hrho : rho (positionVelocityStep dt x v acc).1 ≠ 0) :
    |phaseEnergy a rho potential gradient
        (impulseAtPhase a rho gradient (positionVelocityStep dt x v acc) J) -
      phaseEnergy a rho potential gradient (x, v) -
      (dt * ⟪v, Q⟫_ℝ + ⟪velocityEulerStep dt v acc, J⟫_ℝ +
        dualSq (rho (positionVelocityStep dt x v acc).1) a
          (gradient (positionVelocityStep dt x v acc).1) J / 2)| ≤ c.errorBudget Q := by
  rw [positionVelocityImpulse_energy_ledger a dt rho potential gradient x v acc J hrho]
  convert c.energy_defect Q using 1
  congr 1
  ring

/-- Finite-horizon energy/work control for the actual velocity-position-impulse
recurrence. Every local analytic certificate is required only at a step in the
finite history; no infinite trajectory or exact minimizing/ODE solution is assumed. -/
theorem positionVelocityImpulse_energy_defect_sum (N : ℕ) (a : ℝ)
    (rho potential : E → ℝ) (gradient : E → E) (z : ℕ → E × E)
    (dt : Fin N → ℝ) (acc Q J : Fin N → E)
    (c : ∀ i : Fin N, StepJetCertificate a (dt i) rho potential gradient
      (z i.val).1 (z i.val).2 (acc i))
    (hrho : ∀ i : Fin N,
      rho (positionVelocityStep (dt i) (z i.val).1 (z i.val).2 (acc i)).1 ≠ 0)
    (hstep : ∀ i : Fin N,
      z (i.val + 1) = impulseAtPhase a rho gradient
        (positionVelocityStep (dt i) (z i.val).1 (z i.val).2 (acc i)) (J i)) :
    |phaseEnergy a rho potential gradient (z N) -
      phaseEnergy a rho potential gradient (z 0) -
      ∑ i : Fin N, ((dt i) * ⟪(z i.val).2, Q i⟫_ℝ +
        ⟪velocityEulerStep (dt i) (z i.val).2 (acc i), J i⟫_ℝ +
        dualSq (rho (positionVelocityStep (dt i) (z i.val).1 (z i.val).2 (acc i)).1) a
          (gradient (positionVelocityStep (dt i) (z i.val).1 (z i.val).2 (acc i)).1)
          (J i) / 2)| ≤ ∑ i : Fin N, (c i).errorBudget (Q i) := by
  refine accumulated_energy_defect N (fun k => phaseEnergy a rho potential gradient (z k)) _ _ ?_
  intro i
  simpa only [hstep i] using (c i).impulse_energy_defect (Q i) (J i) (hrho i)

/-- A uniform residual-power and Taylor coefficient bound gives an accumulated
budget controlled by elapsed time and the maximum step, for nonnegative steps. -/
theorem sum_step_budget_le (N : ℕ) (dt : Fin N → ℝ) (eta C hmax : ℝ)
    (hdt : ∀ i, 0 ≤ dt i) (hstep : ∀ i, dt i ≤ hmax) (hC : 0 ≤ C) :
    (∑ i : Fin N, (dt i * eta + (dt i) ^ 2 * C)) ≤
      (eta + hmax * C) * ∑ i : Fin N, dt i := by
  rw [Finset.mul_sum]
  refine Finset.sum_le_sum (fun i _ => ?_)
  nlinarith [mul_nonneg (hdt i) (sub_nonneg.mpr (hstep i)),
    mul_nonneg hC (mul_nonneg (hdt i) (sub_nonneg.mpr (hstep i)))]

omit [InnerProductSpace ℝ E] in
/-- Actual finite nodal variation bounds every endpoint displacement. -/
theorem norm_position_le_initial_add_variation (x : ℕ → E) (n : ℕ) :
    ‖x n‖ ≤ ‖x 0‖ + ∑ k ∈ Finset.range n, ‖x (k + 1) - x k‖ := by
  have hs := norm_sum_le (Finset.range n) (fun k => x (k + 1) - x k)
  rw [Finset.sum_range_sub] at hs
  have hn := norm_add_le (x n - x 0) (x 0)
  simpa [add_comm] using hn.trans (add_le_add_right hs ‖x 0‖)

/-- A single accumulated displacement margin controls every point of every
proposed position segment in a finite history. -/
theorem position_interpolation_norm_lt_of_variation (x : ℕ → E) (N k : ℕ)
    (R t : ℝ) (hk : k < N) (ht : t ∈ Icc (0 : ℝ) 1)
    (hbudget : ‖x 0‖ + ∑ i ∈ Finset.range N, ‖x (i + 1) - x i‖ < R) :
    ‖x k + t • (x (k + 1) - x k)‖ < R := by
  refine affine_segment_norm_lt (x k) (x (k + 1) - x k) R t ?_ ht
  have hs : (∑ i ∈ Finset.range (k + 1), ‖x (i + 1) - x i‖) ≤
      ∑ i ∈ Finset.range N, ‖x (i + 1) - x i‖ :=
    Finset.sum_le_sum_of_subset_of_nonneg (Finset.range_mono hk)
      (fun i _ _ => norm_nonneg (x (i + 1) - x i))
  rw [Finset.sum_range_succ] at hs
  linarith [norm_position_le_initial_add_variation x k]

/-- The actual position recurrence rewrites the global variation certificate
as a sum of step duration times proposed-velocity norm. Impulses following each
position update do not alter this position-segment condition. -/
theorem position_steps_norm_lt (x proposedVelocity : ℕ → E) (dt : ℕ → ℝ)
    (N k : ℕ) (R t : ℝ) (hk : k < N) (ht : t ∈ Icc (0 : ℝ) 1)
    (hstep : ∀ i < N, x (i + 1) = x i + dt i • proposedVelocity i)
    (hbudget : ‖x 0‖ + ∑ i ∈ Finset.range N, |dt i| * ‖proposedVelocity i‖ < R) :
    ‖x k + t • (dt k • proposedVelocity k)‖ < R := by
  have hvar : (∑ i ∈ Finset.range N, ‖x (i + 1) - x i‖) =
      ∑ i ∈ Finset.range N, |dt i| * ‖proposedVelocity i‖ := by
    refine Finset.sum_congr rfl ?_
    intro i hi
    simp [hstep i (Finset.mem_range.mp hi), norm_smul]
  simpa only [hstep k hk, add_sub_cancel_left] using
    position_interpolation_norm_lt_of_variation x N k R t hk ht (by rwa [hvar])

/-- The actual metric turns a norm certificate on the acceleration residual
into a quantitative residual-power bound. -/
theorem dynamicsResidual_power_le (rho a eta : ℝ) (u v acc gradU Q : E)
    (H : E →L[ℝ] E) (hres : ‖dynamicsResidual rho a u H v acc gradU Q‖ ≤ eta) :
    |metric rho a u v (dynamicsResidual rho a u H v acc gradU Q)| ≤
      weight a u / rho ^ 2 * ‖v‖ * eta := by
  exact (abs_metric_le_norm rho a u v _).trans
    (mul_le_mul_of_nonneg_left hres
      (mul_nonneg (div_nonneg (weight_pos a u).le (sq_nonneg rho)) (norm_nonneg v)))

end FractalSync.ScaleMetric
