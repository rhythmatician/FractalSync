import FractalSync.MetricImpulse

/-!
# Exact frozen-metric velocity-step contracts

The velocity update matches the algebraic ordering in the runtime's Euler
step. The metric coefficients are held fixed in every energy comparison here.
The subsequent position update, changing metric and potential, provider error,
and floating-point arithmetic require separate correspondence arguments.
-/

open scoped InnerProductSpace

namespace FractalSync.ScaleMetric

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The explicit velocity update, before the runtime's position update. -/
def velocityEulerStep (dt : ℝ) (v acc : E) : E := v + dt • acc

/-- A force covector converted once by the frozen inverse metric. -/
noncomputable def frozenForceStep (rho a : ℝ) (u v Q : E) (dt : ℝ) : E :=
  velocityEulerStep dt v (inverseAction rho a u Q)

/-- Explicit metric-proportional drag after raising its force covector. -/
def explicitDragStep (beta dt : ℝ) (v : E) : E := velocityEulerStep dt v ((-beta) • v)

/-- Kinetic energy is exactly quadratic under scalar velocity multiplication. -/
theorem kinetic_smul (rho a c : ℝ) (u v : E) :
    kinetic rho a u (c • v) = c ^ 2 * kinetic rho a u v := by
  simp only [kinetic, metric_smul_left, metric_smul_right]
  ring

/-- Exact frozen-metric energy change for an arbitrary explicit velocity increment. -/
theorem velocityEulerStep_energy_change (rho a dt : ℝ) (u v acc : E) :
    kinetic rho a u (velocityEulerStep dt v acc) - kinetic rho a u v =
      dt * metric rho a u v acc + dt ^ 2 * kinetic rho a u acc := by
  rw [velocityEulerStep, kinetic_add, metric_smul_right, metric_smul_left,
    metric_smul_right, kinetic]
  ring

/-- A frozen force step has initial-velocity work plus an exact nonnegative
quadratic correction. No position update is included. -/
theorem frozenForceStep_energy_change (rho a dt : ℝ) (u v Q : E) (hrho : rho ≠ 0) :
    kinetic rho a u (frozenForceStep rho a u v Q dt) - kinetic rho a u v =
      dt * ⟪v, Q⟫_ℝ + dt ^ 2 / 2 * dualSq rho a u Q := by
  rw [frozenForceStep, velocityEulerStep_energy_change,
    metric_inverse_pairing rho a u v Q hrho, kinetic, ← dualSq_eq_metric rho a u Q hrho]
  ring

/-- Initial-velocity work underestimates the exact frozen-metric energy change. -/
theorem frozenForceStep_work_defect_nonneg (rho a dt : ℝ) (u v Q : E) (hrho : rho ≠ 0) :
    0 ≤ kinetic rho a u (frozenForceStep rho a u v Q dt) - kinetic rho a u v -
      dt * ⟪v, Q⟫_ℝ := by
  simpa only [frozenForceStep_energy_change rho a dt u v Q hrho, add_sub_cancel_left] using
    mul_nonneg (div_nonneg (sq_nonneg dt) (by norm_num : (0 : ℝ) ≤ 2))
      (dualSq_nonneg rho a u Q hrho)

/-- For a frozen force and metric, the exact energy change equals work computed
with the average of the old and new velocities. -/
theorem frozenForceStep_average_velocity_work (rho a dt : ℝ) (u v Q : E)
    (hrho : rho ≠ 0) :
    kinetic rho a u (frozenForceStep rho a u v Q dt) - kinetic rho a u v =
      dt / 2 * (⟪v, Q⟫_ℝ + ⟪frozenForceStep rho a u v Q dt, Q⟫_ℝ) := by
  rw [frozenForceStep_energy_change rho a dt u v Q hrho,
    frozenForceStep, velocityEulerStep, inner_add_left, real_inner_smul_left,
    real_inner_comm Q (inverseAction rho a u Q), dualSq]
  ring

/-- Raising the metric drag covector and then applying the frozen velocity step
is exactly explicit drag. This uses the checked inverse, not a flat-metric assumption. -/
theorem frozenForceStep_metric_drag (rho a beta dt : ℝ) (u v : E) (hrho : rho ≠ 0) :
    frozenForceStep rho a u v ((-beta) • metricAction rho a u v) dt =
      explicitDragStep beta dt v := by
  apply congrArg (velocityEulerStep dt v)
  have hlin : inverseAction rho a u ((-beta) • metricAction rho a u v) =
      (-beta) • inverseAction rho a u (metricAction rho a u v) := by
    simp only [inverseAction, unstretch_smul, smul_smul, mul_comm]
  rw [hlin, inverse_metricAction rho a u v hrho]

/-- The explicit drag update scales velocity by one minus the damping step. -/
theorem explicitDragStep_eq_smul (beta dt : ℝ) (v : E) :
    explicitDragStep beta dt v = (1 - beta * dt) • v := by
  simp [explicitDragStep, velocityEulerStep, sub_eq_add_neg, add_smul, smul_smul, mul_comm]

/-- Exact explicit-drag energy factor in a frozen metric. -/
theorem explicitDragStep_kinetic (rho a beta dt : ℝ) (u v : E) :
    kinetic rho a u (explicitDragStep beta dt v) =
      (1 - beta * dt) ^ 2 * kinetic rho a u v := by
  rw [explicitDragStep_eq_smul, kinetic_smul]

/-- The explicit drag step is dissipative throughout the closed stability interval. -/
theorem explicitDragStep_kinetic_le (rho a beta dt : ℝ) (u v : E)
    (h0 : 0 ≤ beta * dt) (h2 : beta * dt ≤ 2) :
    kinetic rho a u (explicitDragStep beta dt v) ≤ kinetic rho a u v := by
  rw [explicitDragStep_kinetic]
  nlinarith [mul_nonneg (mul_nonneg h0 (sub_nonneg.mpr h2)) (kinetic_nonneg rho a u v)]

/-- For positive metric scale and nonzero velocity, the explicit-drag
nonincrease condition is necessary as well as sufficient. -/
theorem explicitDragStep_kinetic_le_iff_of_ne_zero (rho a beta dt : ℝ) (u v : E)
    (hrho : 0 < rho) (hv : v ≠ 0) :
    kinetic rho a u (explicitDragStep beta dt v) ≤ kinetic rho a u v ↔
      0 ≤ beta * dt ∧ beta * dt ≤ 2 := by
  have hK : 0 < kinetic rho a u v :=
    div_pos (metric_self_pos rho a u v hrho hv) (by norm_num)
  constructor
  · intro h
    have hc : (1 - beta * dt) ^ 2 ≤ 1 := (mul_le_mul_iff_left₀ hK).mp
      (by simpa only [explicitDragStep_kinetic, one_mul] using h)
    exact ⟨by nlinarith [sq_nonneg (beta * dt)], by nlinarith [sq_nonneg (beta * dt - 2)]⟩
  · exact fun h => explicitDragStep_kinetic_le rho a beta dt u v h.1 h.2

/-- Complete nonincrease classification, including the stationary-velocity case. -/
theorem explicitDragStep_kinetic_le_iff (rho a beta dt : ℝ) (u v : E)
    (hrho : 0 < rho) :
    kinetic rho a u (explicitDragStep beta dt v) ≤ kinetic rho a u v ↔
      v = 0 ∨ (0 ≤ beta * dt ∧ beta * dt ≤ 2) := by
  by_cases hv : v = 0
  · simp [hv, explicitDragStep, velocityEulerStep]
  · simpa only [hv, false_or] using
      explicitDragStep_kinetic_le_iff_of_ne_zero rho a beta dt u v hrho hv

end FractalSync.ScaleMetric
