import FractalSync.ConcreteEnergyCertificate
import FractalSync.ImprovedConcreteTrajectory

/-!
# Local energy regularity for the concrete quadratic dynamics

The improved local ruler and velocity budgets yield a usable actual
phase-energy differential bound. All constants concern the same literal
asset and conservative potential as the earlier certificate.
-/

noncomputable section

namespace FractalSync.ImprovedConcreteDynamics

open Set Metric OperationalDynamics ConcreteOperationalAsset FlatShoreIntrinsic ScaleMetric
open MechanicalBounds PhaseEnergyBounds PotentialForces PotentialForceBounds
open ConcreteOperationalDynamics (rho force acc rho_floor position_bound
  potential potentialGradient ruler_jet gradient_jet potential_jet phaseEnergy_differentiable)
open scoped RealInnerProductSpace NNReal BigOperators

local instance improvedConcreteEnergyPhaseNormedSpace : NormedSpace ℝ (Plane × Plane) :=
  Prod.normedSpace

/-- Exact arithmetic for the existing bundled energy-covector budget. -/
theorem local_energy_budget :
    energyCovectorBudget 1 1 (2, 1) (1, 20) (20, 200) (1 / 3, 1) (17 / 16, 43 / 2) =
      (601 / 144, 267 / 2) := by
  norm_num [energyCovectorBudget, positionCovectorBudget,
    ControlMechanics.metricActionBudget, addBudget, productBudget, inverseBudget]

/-- The actual potential gradient inherits the locally sharpened ruler data. -/
theorem potentialGradient_bound_of_jets
    (hr : BoundedLipschitzOn (fun z : Plane × Plane => rho z.1)
      (closedBall phase radius) 2 1)
    (hu : BoundedLipschitzOn (fun z : Plane × Plane => gradient rho z.1)
      (closedBall phase radius) 1 20) :
    BoundedLipschitzOn (fun z : Plane × Plane => potentialGradient z.1)
      (closedBall phase radius) (17 / 16) (43 / 2) := by
  have h := totalForce_log_two_default_unit_bound (X := Plane × Plane) (E := Plane)
    1 (fun z => rho z.1) (fun z => gradient rho z.1) (fun z => z.1)
    (closedBall phase radius) (2, 1) (1, 20) 1 (by norm_num) hr hu
    position_bound (fun z _ => rho_floor z.1)
  norm_num at h
  exact ⟨h.bound_nonneg, h.lip_nonneg,
    fun z hz => by simpa only [potentialGradient, norm_neg, one_div] using h.norm_le z hz,
    fun z hz w hw => by
      simpa only [potentialGradient, neg_sub_neg, dist_comm w z, one_div] using
        h.norm_sub_le w hw z hz⟩

/-- The sharpened local field and velocity bounds propagate to the actual
Fréchet differential of the same concrete phase energy. -/
theorem energy_differential_bound_of_jets
    (hr : BoundedLipschitzOn (fun z : Plane × Plane => rho z.1)
      (closedBall phase radius) 2 1)
    (hu : BoundedLipschitzOn (fun z : Plane × Plane => gradient rho z.1)
      (closedBall phase radius) 1 20)
    (hH : BoundedLipschitzOn (fun z : Plane × Plane => fderiv ℝ (gradient rho) z.1)
      (closedBall phase radius) 20 200)
    (hv : BoundedLipschitzOn (Prod.snd : Plane × Plane → Plane)
      (closedBall phase radius) (1 / 3) 1) :
    BoundedLipschitzOn (fderiv ℝ (phaseEnergy 1 rho potential (gradient rho)))
      (closedBall phase radius) 5 134 := by
  have h := fderiv_phaseEnergy_bound (E := Plane) 1 1 (closedBall phase radius)
    rho potential (gradient rho) potentialGradient (fderiv ℝ (gradient rho))
    (2, 1) (1, 20) (20, 200) (1 / 3, 1) (17 / 16, 43 / 2) (by norm_num)
    hr hu hH hv (potentialGradient_bound_of_jets hr hu)
    (fun z _ => rho_floor z.1) (fun z _ => ruler_jet z.1) (fun z _ => gradient_jet z.1)
    (fun z hz => potential_jet z.1 ((position_bound.norm_le z hz).trans_lt (by norm_num)))
  exact h.enlarge (by norm_num [local_energy_budget]) (by norm_num [local_energy_budget])

/-- The full changing-position segment receives the improved differential
constant once its two endpoints lie in the convex certified region. -/
def energyStepCertificateOfBound
    (hE : BoundedLipschitzOn (fderiv ℝ (phaseEnergy 1 rho potential (gradient rho)))
      (closedBall phase radius) 5 134)
    (dt : ℝ) (z : Plane × Plane) (hz : z ∈ closedBall phase radius)
    (hend : positionVelocityStep dt z.1 z.2 (acc z) ∈ closedBall phase radius) :
    StepJetCertificate 1 dt rho potential (gradient rho) z.1 z.2 (acc z) := by
  exact StepJetCertificate.ofLipschitzRegion 1 dt rho potential (gradient rho)
    z.1 z.2 (acc z) (potentialGradient z.1) (fderiv ℝ (gradient rho) z.1)
    (closedBall phase radius) 134 (convex_closedBall phase radius) hz hend
    (zero_lt_one.trans_le (rho_floor z.1)).ne' (ruler_jet z.1) (gradient_jet z.1)
    (potential_jet z.1 ((position_bound.norm_le z hz).trans_lt (by norm_num)))
    phaseEnergy_differentiable hE.lipschitzOnWith

theorem energy_step_taylor_remainder_of_bound
    (hE : BoundedLipschitzOn (fderiv ℝ (phaseEnergy 1 rho potential (gradient rho)))
      (closedBall phase radius) 5 134)
    (dt : ℝ) (z : Plane × Plane) (hz : z ∈ closedBall phase radius)
    (hend : positionVelocityStep dt z.1 z.2 (acc z) ∈ closedBall phase radius) :
    |phaseEnergy 1 rho potential (gradient rho) (positionVelocityStep dt z.1 z.2 (acc z)) -
      phaseEnergy 1 rho potential (gradient rho) z -
      fderiv ℝ (phaseEnergy 1 rho potential (gradient rho)) z
        (dt • (velocityEulerStep dt z.2 (acc z), acc z))| ≤
      67 * dt ^ 2 * (max ‖velocityEulerStep dt z.2 (acc z)‖ ‖acc z‖) ^ 2 := by
  let c := energyStepCertificateOfBound hE dt z hz hend
  have h := NormalizedDistanceFields.field_quadratic_remainder
    (phaseEnergy 1 rho potential (gradient rho)) z
    (dt • (velocityEulerStep dt z.2 (acc z), acc z)) 134 c.differentiable c.variation
  norm_num only [positionVelocityStep_eq_add, norm_smul, Real.norm_eq_abs, Prod.norm_def,
    mul_pow, sq_abs, mul_assoc] at h ⊢
  exact h

/-- The actual energy value is five-Lipschitz on the same convex phase ball. -/
theorem energy_sub_le_of_bound
    (hE : BoundedLipschitzOn (fderiv ℝ (phaseEnergy 1 rho potential (gradient rho)))
      (closedBall phase radius) 5 134)
    (z w : Plane × Plane) (hz : z ∈ closedBall phase radius)
    (hw : w ∈ closedBall phase radius) :
    |phaseEnergy 1 rho potential (gradient rho) z -
      phaseEnergy 1 rho potential (gradient rho) w| ≤ 5 * dist z w := by
  simpa only [Real.norm_eq_abs, dist_eq_norm] using
    Convex.norm_image_sub_le_of_norm_fderiv_le phaseEnergy_differentiable hE.norm_le
      (convex_closedBall phase radius) hw hz

/-- The actual phase-energy differential applied to one stored recurrence
increment. This is a Taylor ledger, not an asserted physical work integral. -/
def exactEnergyIncrement (N i : ℕ) : ℝ :=
  fderiv ℝ (phaseEnergy 1 rho potential (gradient rho)) (exactTrajectory N i)
    (refinementStep horizon N •
      (velocityEulerStep (refinementStep horizon N) (exactTrajectory N i).2
        (acc (exactTrajectory N i)), acc (exactTrajectory N i)))

theorem exactTrajectory_energy_remainder_of_bounds
    (hE : BoundedLipschitzOn (fderiv ℝ (phaseEnergy 1 rho potential (gradient rho)))
      (closedBall phase radius) 5 134)
    (hacc : ∀ z ∈ closedBall phase radius, ‖acc z‖ ≤ 8)
    (N i : ℕ) (hz : exactTrajectory N i ∈ closedBall phase radius)
    (hend : exactTrajectory N (i + 1) ∈ closedBall phase radius) :
    |phaseEnergy 1 rho potential (gradient rho) (exactTrajectory N (i + 1)) -
      phaseEnergy 1 rho potential (gradient rho) (exactTrajectory N i) -
      exactEnergyIncrement N i| ≤ 4288 * (refinementStep horizon N) ^ 2 := by
  have hmax : max ‖(exactTrajectory N (i + 1)).2‖ ‖acc (exactTrajectory N i)‖ ≤ (8 : ℝ) :=
    max_le ((velocity_bound.norm_le _ hend).trans (by norm_num)) (hacc _ hz)
  have h := energy_step_taylor_remainder_of_bound hE (refinementStep horizon N)
    (exactTrajectory N i) hz (show positionVelocityStep (refinementStep horizon N)
      (exactTrajectory N i).1 (exactTrajectory N i).2 (acc (exactTrajectory N i)) ∈
        closedBall phase radius from hend)
  apply h.trans
  have hquad := mul_le_mul_of_nonneg_left
    (pow_le_pow_left₀ ((norm_nonneg _).trans (le_max_left _ _)) hmax 2)
    (show 0 ≤ 67 * (refinementStep horizon N) ^ 2 by positivity)
  convert hquad using 1
  ring

theorem exactTrajectory_energy_remainder_rate_of_bounds
    (hE : BoundedLipschitzOn (fderiv ℝ (phaseEnergy 1 rho potential (gradient rho)))
      (closedBall phase radius) 5 134)
    (hacc : ∀ z ∈ closedBall phase radius, ‖acc z‖ ≤ 8)
    (N i : ℕ) (hz : exactTrajectory N i ∈ closedBall phase radius)
    (hend : exactTrajectory N (i + 1) ∈ closedBall phase radius) :
    |phaseEnergy 1 rho potential (gradient rho) (exactTrajectory N (i + 1)) -
      phaseEnergy 1 rho potential (gradient rho) (exactTrajectory N i) -
      exactEnergyIncrement N i| ≤ 1 / (64 * ((N : ℝ) + 1) ^ 2) := by
  apply (exactTrajectory_energy_remainder_of_bounds hE hacc N i hz hend).trans
  simpa only [refinementStep, horizon_value, Nat.cast_add, Nat.cast_one,
    div_pow, mul_div_assoc, div_div, mul_pow] using
    div_le_div_of_nonneg_right
      (by norm_num : (4288 : ℝ) * (1 / 576) ^ 2 ≤ 1 / 64)
      (sq_nonneg ((N : ℝ) + 1))

/-- Finite telescoping accumulates the Taylor defects against the actual
sum of differential increments, without calling that sum physical work. -/
theorem exactTrajectory_energy_taylor_ledger_of_bounds
    (hE : BoundedLipschitzOn (fderiv ℝ (phaseEnergy 1 rho potential (gradient rho)))
      (closedBall phase radius) 5 134)
    (hacc : ∀ z ∈ closedBall phase radius, ‖acc z‖ ≤ 8)
    (N : ℕ) (hmem : ∀ i ≤ N + 1, exactTrajectory N i ∈ closedBall phase radius) :
    |phaseEnergy 1 rho potential (gradient rho) (exactTrajectory N (N + 1)) -
      phaseEnergy 1 rho potential (gradient rho) (exactTrajectory N 0) -
      ∑ i : Fin (N + 1), exactEnergyIncrement N i| ≤ 1 / (64 * ((N : ℝ) + 1)) := by
  have h := accumulated_energy_defect (N + 1)
    (fun i => phaseEnergy 1 rho potential (gradient rho) (exactTrajectory N i))
    (fun i => exactEnergyIncrement N i) (fun _ => 1 / (64 * ((N : ℝ) + 1) ^ 2))
    (fun i => exactTrajectory_energy_remainder_rate_of_bounds hE hacc N i
      (hmem i i.isLt.le) (hmem (i + 1) (Nat.succ_le_of_lt i.isLt)))
  apply h.trans_eq
  simp only [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul,
    Nat.cast_add, Nat.cast_one]
  field_simp

/-- The literal asset discharges both local potential-gradient premises. -/
theorem potentialGradient_bound :
    BoundedLipschitzOn (fun z : Plane × Plane => potentialGradient z.1)
      (closedBall phase radius) (17 / 16) (43 / 2) := by
  exact potentialGradient_bound_of_jets regional_jets.1 regional_jets.2.1

/-- Unconditional improved regularity of the concrete phase-energy differential. -/
theorem energy_differential_bound :
    BoundedLipschitzOn (fderiv ℝ (phaseEnergy 1 rho potential (gradient rho)))
      (closedBall phase radius) 5 134 := by
  exact energy_differential_bound_of_jets regional_jets.1 regional_jets.2.1
    regional_jets.2.2 velocity_bound

/-- Only actual endpoint inclusion remains for an arbitrary contained step. -/
def energyStepCertificate (dt : ℝ) (z : Plane × Plane) (hz : z ∈ closedBall phase radius)
    (hend : positionVelocityStep dt z.1 z.2 (acc z) ∈ closedBall phase radius) :
    StepJetCertificate 1 dt rho potential (gradient rho) z.1 z.2 (acc z) :=
  energyStepCertificateOfBound energy_differential_bound dt z hz hend

theorem energyStepCertificate_L (dt : ℝ) (z : Plane × Plane)
    (hz : z ∈ closedBall phase radius)
    (hend : positionVelocityStep dt z.1 z.2 (acc z) ∈ closedBall phase radius) :
    ((energyStepCertificate dt z hz hend).L : ℝ) = 134 := by
  rfl

theorem energy_step_taylor_remainder (dt : ℝ) (z : Plane × Plane)
    (hz : z ∈ closedBall phase radius)
    (hend : positionVelocityStep dt z.1 z.2 (acc z) ∈ closedBall phase radius) :
    |phaseEnergy 1 rho potential (gradient rho) (positionVelocityStep dt z.1 z.2 (acc z)) -
      phaseEnergy 1 rho potential (gradient rho) z -
      fderiv ℝ (phaseEnergy 1 rho potential (gradient rho)) z
        (dt • (velocityEulerStep dt z.2 (acc z), acc z))| ≤
      67 * dt ^ 2 * (max ‖velocityEulerStep dt z.2 (acc z)‖ ‖acc z‖) ^ 2 := by
  exact energy_step_taylor_remainder_of_bound energy_differential_bound dt z hz hend

theorem energy_sub_le (z w : Plane × Plane)
    (hz : z ∈ closedBall phase radius) (hw : w ∈ closedBall phase radius) :
    |phaseEnergy 1 rho potential (gradient rho) z -
      phaseEnergy 1 rho potential (gradient rho) w| ≤ 5 * dist z w := by
  exact energy_sub_le_of_bound energy_differential_bound z w hz hw

/-- Every step of the actual new recurrence satisfies the explicit quadratic
remainder, with all field and membership premises discharged. -/
theorem exactTrajectory_energy_remainder (N i : ℕ) (hi : i < N + 1) :
    |phaseEnergy 1 rho potential (gradient rho) (exactTrajectory N (i + 1)) -
      phaseEnergy 1 rho potential (gradient rho) (exactTrajectory N i) -
      exactEnergyIncrement N i| ≤ 4288 * (refinementStep horizon N) ^ 2 := by
  exact exactTrajectory_energy_remainder_of_bounds energy_differential_bound acc_bound.norm_le
    N i (ball_subset_closedBall (exactTrajectory_mem N i hi.le))
    (ball_subset_closedBall (exactTrajectory_mem N (i + 1) (Nat.succ_le_of_lt hi)))

theorem exactTrajectory_energy_remainder_rate (N i : ℕ) (hi : i < N + 1) :
    |phaseEnergy 1 rho potential (gradient rho) (exactTrajectory N (i + 1)) -
      phaseEnergy 1 rho potential (gradient rho) (exactTrajectory N i) -
      exactEnergyIncrement N i| ≤ 1 / (64 * ((N : ℝ) + 1) ^ 2) := by
  exact exactTrajectory_energy_remainder_rate_of_bounds energy_differential_bound acc_bound.norm_le
    N i (ball_subset_closedBall (exactTrajectory_mem N i hi.le))
    (ball_subset_closedBall (exactTrajectory_mem N (i + 1) (Nat.succ_le_of_lt hi)))

/-- The finite Taylor ledger for the full certified exact-real run. -/
theorem exactTrajectory_energy_taylor_ledger (N : ℕ) :
    |phaseEnergy 1 rho potential (gradient rho) (exactTrajectory N (N + 1)) -
      phaseEnergy 1 rho potential (gradient rho) (exactTrajectory N 0) -
      ∑ i : Fin (N + 1), exactEnergyIncrement N i| ≤ 1 / (64 * ((N : ℝ) + 1)) := by
  exact exactTrajectory_energy_taylor_ledger_of_bounds energy_differential_bound acc_bound.norm_le
    N (fun i hi => ball_subset_closedBall (exactTrajectory_mem N i hi))

/-- Per-step remainder on the same 2048-step mesh as the trajectory result. -/
theorem exactTrajectory_2048_energy_remainder (i : ℕ) (hi : i < 2048) :
    |phaseEnergy 1 rho potential (gradient rho) (exactTrajectory 2047 (i + 1)) -
      phaseEnergy 1 rho potential (gradient rho) (exactTrajectory 2047 i) -
      exactEnergyIncrement 2047 i| ≤ (1 / 268435456 : ℝ) := by
  simpa only [Nat.cast_ofNat, show (2047 : ℝ) + 1 = 2048 by norm_num,
    show (64 : ℝ) * 2048 ^ 2 = 268435456 by norm_num] using
      exactTrajectory_energy_remainder_rate 2047 i hi

/-- The complete 2048-step Taylor ledger has defect at most 2^-17. -/
theorem exactTrajectory_2048_energy_taylor_ledger :
    |phaseEnergy 1 rho potential (gradient rho) (exactTrajectory 2047 2048) -
      phaseEnergy 1 rho potential (gradient rho) (exactTrajectory 2047 0) -
      ∑ i : Fin 2048, exactEnergyIncrement 2047 i| ≤ (1 / 131072 : ℝ) := by
  simpa only [Nat.cast_ofNat, show (2047 : ℝ) + 1 = 2048 by norm_num,
    show (64 : ℝ) * 2048 = 131072 by norm_num] using
      exactTrajectory_energy_taylor_ledger 2047

end FractalSync.ImprovedConcreteDynamics
