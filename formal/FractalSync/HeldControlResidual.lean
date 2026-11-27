import FractalSync.OperationalCatalogBounds
import FractalSync.RankOneInverseBounds

/-!
# Residual from holding the constructed control covector

The operational engine holds its supplied drive covector through its Runge–Kutta
stages, while reevaluating conservative forces, viscous drag and the metric.
This module bounds the exact-real acceleration discrepancy from reevaluating a
fixed normalized command. It does not identify the numerical Runge–Kutta map
or its floating-point implementation with an exact trajectory.
-/

noncomputable section

namespace FractalSync.HeldControlResidual

open Set ScaleMetric ControlMechanics OperationalDynamics
open ScaleMetric.MechanicalBounds PotentialForceBounds
open scoped NNReal

variable {E X : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [PseudoMetricSpace X]

/-- The local unit drive has explicit covector bounds before the metric is
raised. These constants apply to every fixed nonzero command direction. -/
theorem normalized_drive_local_bound (s : Set X) (rho : X → ℝ)
    (u : X → E) (e : E) (he : e ≠ 0)
    (hr : BoundedLipschitzOn rho s 2 1)
    (hu : BoundedLipschitzOn u s 1 20)
    (hfloor : ∀ z ∈ s, 1 ≤ rho z) :
    BoundedLipschitzOn
      (fun z => driveForce (rho z) 1 1 (u z) (metricNormalize (rho z) 1 (u z) e))
      s 4 170 := by
  have hOne : BoundedLipschitzOn (fun _ : X => (1 : ℝ)) s 1 0 :=
    by simpa only [abs_one] using const_real s (1 : ℝ)
  have hd := normalizedDriveForce_boundedLipschitz 1 1 s rho (fun _ => 1) u e
    (2, 1) (1, 20) (1, 0) (by norm_num) hr hu hOne hfloor he
  norm_num [normalizedDriveForceBudget, normalizedHeadingBudget, driveForceBudget,
    metricActionBudget, addBudget, productBudget, inverseBudget] at hd
  exact hd

/-- Conservative forces and drag are reevaluated; only the supplied control
covector is frozen. This is the holding policy of the operational RHS. -/
def frozenCatalogForce (rho a kappa mu beta : ℝ) (u x v q : E) : E :=
  PotentialForces.totalForce kappa mu rho u x + q + -(beta • metricAction rho a u v)

/-- Metric raising and the unchanged connection cancel exactly in the
difference between frozen and reevaluated catalog acceleration. -/
theorem frozen_catalog_acceleration_sub (rho a kappa mu F beta : ℝ)
    (u x v e q connectionTerm : E) :
    (inverseAction rho a u (frozenCatalogForce rho a kappa mu beta u x v q) -
      connectionTerm) -
    (inverseAction rho a u (catalogForce rho a kappa mu F beta u x v e) -
      connectionTerm) =
    inverseAction rho a u (q - driveForce rho a F u (metricNormalize rho a u e)) := by
  simp only [frozenCatalogForce, catalogForce, inverseAction_add, inverseAction_sub]
  abel

/-- A regional control-covector Lipschitz constant produces the acceleration
residual without separately bounding the rank-one inverse's summands. -/
theorem inverseAction_hold_error (a rho R : ℝ) (u : E) (Q : X → E)
    (s : Set X) (M L : ℝ) (hQ : BoundedLipschitzOn Q s M L)
    (x y : X) (hx : x ∈ s) (hy : y ∈ s) (hr : ‖rho‖ ≤ R) :
    ‖inverseAction rho a u (Q x - Q y)‖ ≤ R ^ 2 * L * dist x y := by
  have hs : rho ^ 2 ≤ R ^ 2 := by
    simpa only [Real.norm_eq_abs, sq_abs] using pow_le_pow_left₀ (norm_nonneg rho) hr 2
  simpa only [mul_assoc] using (norm_inverseAction_le rho a u (Q x - Q y)).trans
    (mul_le_mul hs (hQ.norm_sub_le x hx y hy) (norm_nonneg _) (sq_nonneg R))

/-- A separately certified control-query error is raised by at most the
square of the current ruler bound. -/
theorem inverseAction_sample_error (rho a R delta : ℝ) (u q qExact : E)
    (hr : ‖rho‖ ≤ R) (hq : ‖q - qExact‖ ≤ delta) :
    ‖inverseAction rho a u (q - qExact)‖ ≤ R ^ 2 * delta := by
  have hs : rho ^ 2 ≤ R ^ 2 := by
    simpa only [Real.norm_eq_abs, sq_abs] using pow_le_pow_left₀ (norm_nonneg rho) hr 2
  exact (norm_inverseAction_le rho a u (q - qExact)).trans
    (mul_le_mul hs hq (norm_nonneg _) (sq_nonneg R))

/-- Sample-construction error and the later holding error add. This theorem
exposes the numerical input premise; it supplies no floating-point certificate. -/
theorem inverseAction_hold_error_with_sample_error (a rho R : ℝ) (u : E)
    (Q : X → E) (s : Set X) (M L : ℝ)
    (hQ : BoundedLipschitzOn Q s M L)
    (x y : X) (hx : x ∈ s) (hy : y ∈ s) (hr : ‖rho‖ ≤ R)
    (q : E) (delta : ℝ) (hq : ‖q - Q x‖ ≤ delta) :
    ‖inverseAction rho a u (q - Q y)‖ ≤ R ^ 2 * delta + R ^ 2 * L * dist x y := by
  rw [show q - Q y = (q - Q x) + (Q x - Q y) by abel, inverseAction_add]
  exact (norm_add_le _ _).trans (add_le_add
    (inverseAction_sample_error rho a R delta u q (Q x) hr hq)
    (inverseAction_hold_error a rho R u Q s M L hQ x y hx hy hr))

/-- A stage-displacement estimate can be supplied without pretending that a
Runge–Kutta stage lies on an exact solution curve. -/
theorem inverseAction_hold_error_of_distance_le (a rho R : ℝ) (u : E)
    (Q : X → E) (s : Set X) (M L B tau : ℝ)
    (hQ : BoundedLipschitzOn Q s M L)
    (x y : X) (hx : x ∈ s) (hy : y ∈ s) (hr : ‖rho‖ ≤ R)
    (hxy : dist x y ≤ B * tau) :
    ‖inverseAction rho a u (Q x - Q y)‖ ≤ R ^ 2 * L * B * tau := by
  simpa only [mul_assoc] using (inverseAction_hold_error a rho R u Q s M L hQ x y hx hy hr).trans
    (mul_le_mul_of_nonneg_left hxy (mul_nonneg (sq_nonneg R) hQ.lip_nonneg))

/-- Along a certified Lipschitz phase curve, holding at the cell's sample
time creates a residual proportional to elapsed time. -/
theorem inverseAction_hold_error_along_curve (a rho R : ℝ) (u : E)
    (Q : X → E) (s : Set X) (M L : ℝ)
    (hQ : BoundedLipschitzOn Q s M L) (z : ℝ → X) (I : Set ℝ) (B : ℝ≥0)
    (hpath : LipschitzOnWith B z I) (hmap : MapsTo z I s)
    (c t : ℝ) (hc : c ∈ I) (ht : t ∈ I) (hr : ‖rho‖ ≤ R) :
    ‖inverseAction rho a u (Q (z c) - Q (z t))‖ ≤ R ^ 2 * L * B * |t - c| := by
  simpa only [Real.dist_eq, abs_sub_comm] using
    inverseAction_hold_error_of_distance_le a rho R u Q s M L B (dist c t) hQ
      (z c) (z t) (hmap hc) (hmap ht) hr (hpath.dist_le_mul c hc t ht)

/-- For the improved local ruler budget, the unit-command holding residual
has a completely numerical spatial coefficient. -/
theorem normalized_drive_hold_error (s : Set X) (rho : X → ℝ)
    (u : X → E) (e : E) (he : e ≠ 0)
    (hr : BoundedLipschitzOn rho s 2 1)
    (hu : BoundedLipschitzOn u s 1 20)
    (hfloor : ∀ z ∈ s, 1 ≤ rho z)
    (x y : X) (hx : x ∈ s) (hy : y ∈ s) :
    ‖inverseAction (rho y) 1 (u y)
      (driveForce (rho x) 1 1 (u x) (metricNormalize (rho x) 1 (u x) e) -
        driveForce (rho y) 1 1 (u y) (metricNormalize (rho y) 1 (u y) e))‖ ≤
      680 * dist x y := by
  have h := inverseAction_hold_error 1 (rho y) 2 (u y)
    (fun z => driveForce (rho z) 1 1 (u z) (metricNormalize (rho z) 1 (u z) e))
    s 4 170 (normalized_drive_local_bound s rho u e he hr hu hfloor)
    x y hx hy (hr.norm_le y hy)
  simpa only [show (2 : ℝ) ^ 2 * 170 = 680 by norm_num] using h

/-- This is the actual catalog-acceleration comparison: conservative force,
drag and connection are all evaluated at the current state. -/
theorem frozen_catalog_local_error (s : Set X) (rho : X → ℝ)
    (u position velocity : X → E) (e : E) (he : e ≠ 0)
    (hr : BoundedLipschitzOn rho s 2 1)
    (hu : BoundedLipschitzOn u s 1 20)
    (hfloor : ∀ z ∈ s, 1 ≤ rho z)
    (kappa mu beta : ℝ) (connectionTerm : E)
    (x y : X) (hx : x ∈ s) (hy : y ∈ s) :
    ‖(inverseAction (rho y) 1 (u y)
        (frozenCatalogForce (rho y) 1 kappa mu beta (u y) (position y) (velocity y)
          (driveForce (rho x) 1 1 (u x) (metricNormalize (rho x) 1 (u x) e))) -
        connectionTerm) -
      (inverseAction (rho y) 1 (u y)
        (catalogForce (rho y) 1 kappa mu 1 beta (u y) (position y) (velocity y) e) -
        connectionTerm)‖ ≤ 680 * dist x y := by
  rw [frozen_catalog_acceleration_sub]
  exact normalized_drive_hold_error s rho u e he hr hu hfloor x y hx hy

/-- A speed-eight displacement budget gives a first-order holding residual.
Its constant must be included in, rather than silently omitted from, the
numerical acceleration-residual budget. -/
theorem normalized_drive_hold_error_time (s : Set X) (rho : X → ℝ)
    (u : X → E) (e : E) (he : e ≠ 0)
    (hr : BoundedLipschitzOn rho s 2 1)
    (hu : BoundedLipschitzOn u s 1 20)
    (hfloor : ∀ z ∈ s, 1 ≤ rho z)
    (x y : X) (hx : x ∈ s) (hy : y ∈ s) (tau : ℝ)
    (hxy : dist x y ≤ 8 * tau) :
    ‖inverseAction (rho y) 1 (u y)
      (driveForce (rho x) 1 1 (u x) (metricNormalize (rho x) 1 (u x) e) -
        driveForce (rho y) 1 1 (u y) (metricNormalize (rho y) 1 (u y) e))‖ ≤
      5440 * tau := by
  exact (normalized_drive_hold_error s rho u e he hr hu hfloor x y hx hy).trans
    (by linarith)

end FractalSync.HeldControlResidual
