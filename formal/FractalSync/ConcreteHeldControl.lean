import FractalSync.ImprovedConcreteTrajectory
import FractalSync.HeldControlResidual

/-!
# Held-drive residual for the literal quadratic fixture

The local field certificate discharges every ruler and drive regularity premise.
Only region membership, sample construction error, and stage displacement remain
conditional. These are exact-real comparisons, not a Runge--Kutta implementation
or a certificate for floating-point control construction.
-/

noncomputable section

namespace FractalSync.ConcreteHeldControl

open Set Metric FlatShoreIntrinsic ScaleMetric OperationalDynamics ConcreteOperationalAsset
open ControlMechanics HeldControlResidual
open ConcreteOperationalDynamics (rho force acc rho_floor heading_ne_zero)

local instance concreteHeldPhaseNormedSpace : NormedSpace ℝ (Plane × Plane) :=
  Prod.normedSpace

/-- The actual unit normalized drive covector for the fixture's fixed command. -/
def drive (z : Plane × Plane) : Plane :=
  driveForce (rho z.1) 1 1 (gradient rho z.1)
    (metricNormalize (rho z.1) 1 (gradient rho z.1) velocity)

/-- The supplied drive covector is held; all other force and connection terms
are evaluated at the current phase state. -/
def frozenAcceleration (q : Plane) (z : Plane × Plane) : Plane :=
  inverseAction (rho z.1) 1 (gradient rho z.1)
    (frozenCatalogForce (rho z.1) 1 (Real.log 2) (1 / Real.pi) 1
      (gradient rho z.1) z.1 z.2 q) -
    connection (rho z.1) 1 (gradient rho z.1) (fderiv ℝ (gradient rho) z.1) z.2 z.2

/-- Only the held drive remains in the acceleration difference. -/
theorem frozenAcceleration_sub (q : Plane) (z : Plane × Plane) :
    frozenAcceleration q z - acc z =
      inverseAction (rho z.1) 1 (gradient rho z.1) (q - drive z) := by
  exact frozen_catalog_acceleration_sub (rho z.1) 1 (Real.log 2) (1 / Real.pi) 1 1
    (gradient rho z.1) z.1 z.2 velocity q
    (connection (rho z.1) 1 (gradient rho z.1) (fderiv ℝ (gradient rho) z.1) z.2 z.2)

/-- Holding the freshly constructed covector creates no discrepancy at its sample state. -/
theorem frozenAcceleration_at_sample (z : Plane × Plane) :
    frozenAcceleration (drive z) z = acc z := by
  have h := frozenAcceleration_sub (drive z) z
  simpa only [sub_self, inverseAction, unstretch, inner_zero_right, mul_zero, zero_div,
    zero_smul, sub_zero, smul_zero, sub_eq_zero] using h

/-- The literal asset and local jet proofs discharge the unit-drive bounds. -/
theorem drive_bound : BoundedLipschitzOn drive (closedBall phase radius) 4 170 := by
  exact normalized_drive_local_bound (closedBall phase radius)
    (fun z : Plane × Plane => rho z.1) (fun z => gradient rho z.1) velocity heading_ne_zero
    ImprovedConcreteDynamics.regional_jets.1 ImprovedConcreteDynamics.regional_jets.2.1
    (fun z _ => rho_floor z.1)

/-- A concrete holding residual with no remaining field-regularity premise. -/
theorem held_acceleration_sub_le (sample current : Plane × Plane)
    (hsample : sample ∈ closedBall phase radius) (hcurrent : current ∈ closedBall phase radius) :
    ‖frozenAcceleration (drive sample) current - acc current‖ ≤ 680 * dist sample current := by
  rw [frozenAcceleration_sub]
  simpa only [show (2 : ℝ) ^ 2 * 170 = 680 by norm_num] using
    inverseAction_hold_error 1 (rho current.1) 2 (gradient rho current.1) drive
      (closedBall phase radius) 4 170 drive_bound sample current hsample hcurrent
      (ImprovedConcreteDynamics.regional_jets.1.norm_le current hcurrent)

/-- Sample construction error adds four times its covector error budget. -/
theorem held_acceleration_sub_le_with_sample_error (sample current : Plane × Plane)
    (hsample : sample ∈ closedBall phase radius) (hcurrent : current ∈ closedBall phase radius)
    (q : Plane) (delta : ℝ) (hq : ‖q - drive sample‖ ≤ delta) :
    ‖frozenAcceleration q current - acc current‖ ≤ 4 * delta + 680 * dist sample current := by
  rw [frozenAcceleration_sub]
  simpa only [show (2 : ℝ) ^ 2 = 4 by norm_num, show (4 : ℝ) * 170 = 680 by norm_num] using
    inverseAction_hold_error_with_sample_error 1 (rho current.1) 2 (gradient rho current.1) drive
      (closedBall phase radius) 4 170 drive_bound sample current hsample hcurrent
      (ImprovedConcreteDynamics.regional_jets.1.norm_le current hcurrent) q delta hq

/-- A supplied speed-eight stage displacement gives the concrete time residual. -/
theorem held_acceleration_sub_le_time (sample current : Plane × Plane)
    (hsample : sample ∈ closedBall phase radius) (hcurrent : current ∈ closedBall phase radius)
    (tau : ℝ) (hmove : dist sample current ≤ 8 * tau) :
    ‖frozenAcceleration (drive sample) current - acc current‖ ≤ 5440 * tau := by
  exact (held_acceleration_sub_le sample current hsample hcurrent).trans (by linarith)

/-- Combined sample error and holding age, still conditional on actual region
membership and a proved displacement estimate for the compared states. -/
theorem held_acceleration_sub_le_time_with_sample_error (sample current : Plane × Plane)
    (hsample : sample ∈ closedBall phase radius) (hcurrent : current ∈ closedBall phase radius)
    (q : Plane) (delta : ℝ) (hq : ‖q - drive sample‖ ≤ delta)
    (tau : ℝ) (hmove : dist sample current ≤ 8 * tau) :
    ‖frozenAcceleration q current - acc current‖ ≤ 4 * delta + 5440 * tau := by
  exact (held_acceleration_sub_le_with_sample_error sample current hsample hcurrent q delta hq).trans
    (by linarith)

end FractalSync.ConcreteHeldControl
