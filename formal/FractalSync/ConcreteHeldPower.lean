import FractalSync.ConcreteHeldControl
import FractalSync.ImprovedConcreteEnergy

/-!
# Actual work-energy power for the concrete held-drive field

The supplied covector is arbitrary and fixed during a step. Conservative
forces and the changing metric cancel in the actual phase-energy derivative,
leaving exactly the drive-plus-drag power used by the operational RHS.
-/

noncomputable section

namespace FractalSync.ConcreteHeldPower

open Set Metric FlatShoreIntrinsic ScaleMetric OperationalDynamics ConcreteOperationalAsset
open ConcreteOperationalDynamics (rho acc potential potentialGradient rho_floor
  ruler_jet gradient_jet potential_jet position_bound phaseEnergy_differentiable)
open ConcreteHeldControl
open scoped InnerProductSpace

local instance heldPowerPhaseNormedSpace : NormedSpace ℝ (Plane × Plane) :=
  Prod.normedSpace

/-- The actual kinetic-plus-conservative-potential energy of the existing fixture. -/
def energy : Plane × Plane → ℝ := phaseEnergy 1 rho potential (gradient rho)

/-- The operational power for a supplied drive covector and reevaluated unit drag. -/
def heldPower (q : Plane) (z : Plane × Plane) : ℝ :=
  ⟪z.2, q + -(metricAction (rho z.1) 1 (gradient rho z.1) z.2)⟫_ℝ

/-- The continuously reevaluated fixed-command power of the reference ODE. -/
def targetPower (z : Plane × Plane) : ℝ := heldPower (drive z) z

/-- Actual energy differentiation gives the held-drive plus current-drag power. -/
theorem fderiv_energy_heldField (q : Plane) (z : Plane × Plane)
    (hz : z ∈ closedBall phase radius) :
    fderiv ℝ energy z (phaseField (frozenAcceleration q) z) = heldPower q z := by
  refine fderiv_phaseEnergy_motion 1 rho potential (gradient rho) z.1 z.2
    (frozenAcceleration q z) (potentialGradient z.1)
    (q + -(metricAction (rho z.1) 1 (gradient rho z.1) z.2))
    (fderiv ℝ (gradient rho) z.1) (ruler_jet z.1) (gradient_jet z.1)
    (potential_jet z.1 ((position_bound.norm_le z hz).trans_lt (by norm_num)))
    (zero_lt_one.trans_le (rho_floor z.1)).ne' ?_
  rw [frozenAcceleration, sub_add_cancel]
  simp only [HeldControlResidual.frozenCatalogForce, potentialGradient, one_smul,
    sub_neg_eq_add, add_comm, add_left_comm, add_assoc]

/-- The actual reference field uses the same energy derivative with its freshly constructed drive. -/
theorem fderiv_energy_targetField (z : Plane × Plane) (hz : z ∈ closedBall phase radius) :
    fderiv ℝ energy z (phaseField acc z) = targetPower z := by
  simpa only [phaseField, frozenAcceleration_at_sample] using
    fderiv_energy_heldField (drive z) z hz

/-- The energy differentiability premise is discharged by the actual fixture jets. -/
theorem energy_differentiable (z : Plane × Plane) (hz : z ∈ closedBall phase radius) :
    DifferentiableAt ℝ energy z := by
  exact phaseEnergy_differentiable z hz

/-- Actual differential norm and variation bounds on the same certified phase ball. -/
theorem energy_differential_bound :
    BoundedLipschitzOn (fderiv ℝ energy) (closedBall phase radius) 5 134 := by
  exact ImprovedConcreteDynamics.energy_differential_bound

/-- Separate speed and variation constants for the actual reference phase field. -/
theorem targetField_bound :
    BoundedLipschitzOn (phaseField acc) (closedBall phase radius) 8 400 := by
  exact ⟨by norm_num, by norm_num, ImprovedConcreteDynamics.phaseField_bound,
    fun z hz w hw => by
      simpa only [max_eq_right (by norm_num : (1 : ℝ) ≤ 400), dist_eq_norm] using
        phaseField_sub_le acc z w 400 (by
          simpa only [dist_eq_norm] using ImprovedConcreteDynamics.acc_bound.norm_sub_le z hz w hw)⟩

/-- Actual reference power is bounded and Lipschitz, derived from DE applied to its field. -/
theorem targetPower_bound :
    BoundedLipschitzOn targetPower (closedBall phase radius) 40 3072 := by
  have h := targetField_bound.bilinear energy_differential_bound
    (ContinuousLinearMap.apply ℝ ℝ) (fun v L => by
      simpa only [mul_comm] using L.le_opNorm v)
  norm_num only [ContinuousLinearMap.apply_apply] at h
  exact ⟨h.bound_nonneg, h.lip_nonneg,
    fun z hz => by simpa only [fderiv_energy_targetField z hz] using h.norm_le z hz,
    fun z hz w hw => by simpa only [fderiv_energy_targetField z hz,
      fderiv_energy_targetField w hw] using h.norm_sub_le z hz w hw⟩

/-- Continuity needed for the reference work integral follows from the actual power bound. -/
theorem targetPower_continuousOn : ContinuousOn targetPower (closedBall phase radius) := by
  exact targetPower_bound.lipschitzOnWith.continuousOn

end FractalSync.ConcreteHeldPower
