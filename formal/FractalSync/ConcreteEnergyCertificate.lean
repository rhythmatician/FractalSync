import FractalSync.ConcreteTrajectoryCertificate
import FractalSync.PhaseEnergyBounds

/-!
# Concrete changing-metric energy regularity

The literal finite asset and actual conservative catalog instantiate the
regional phase-energy differential bound with rational constants. This
certificate applies to complete steps contained in the stated phase ball;
it does not assert exact conservation or floating-point correspondence.
-/

noncomputable section

namespace FractalSync.ConcreteOperationalDynamics

open Set Metric OperationalDynamics ConcreteOperationalAsset FlatShoreIntrinsic ScaleMetric
open MechanicalBounds PhaseEnergyBounds PotentialForces PotentialForceBounds
open scoped RealInnerProductSpace NNReal

local instance concreteEnergyPhaseNormedSpace : NormedSpace ℝ (Plane × Plane) :=
  Prod.normedSpace

def potential : Plane → ℝ := totalPotential (Real.log 2) (1 / Real.pi) 1 rho

def potentialGradient (x : Plane) : Plane :=
  -totalForce (Real.log 2) (1 / Real.pi) (rho x) (gradient rho x) x

/-- Actual Riesz gradient identity for the literal regularized spline. -/
theorem ruler_jet (x : Plane) : HasFDerivAt rho (innerSL ℝ (gradient rho x)) x := by
  exact ((contDiff_rulerField ConcreteOperationalAsset.coefficients origin spacing 1
    (by norm_num)).differentiable (by norm_num) x).hasGradientAt.hasFDerivAt

/-- Actual Hessian identity, with no externally supplied derivative matrix. -/
theorem gradient_jet (x : Plane) :
    HasFDerivAt (gradient rho) (fderiv ℝ (gradient rho) x) x := by
  exact ((actual_jet_regularity ConcreteOperationalAsset.coefficients origin spacing 1
    (by norm_num)).1.differentiable (by norm_num) x).hasFDerivAt

/-- The conservative catalog supplies the actual potential gradient. -/
theorem potential_jet (x : Plane) (hx : ‖x‖ < 2) :
    HasFDerivAt potential (innerSL ℝ (potentialGradient x)) x := by
  simpa only [potential, potentialGradient, map_neg] using
    hasFDerivAt_totalPotential (Real.log 2) (1 / Real.pi) 1 rho x (gradient rho x)
      (by norm_num) (zero_lt_one.trans_le (rho_floor x)) hx (ruler_jet x)

/-- Numerical bounds for the actual conservative potential gradient. -/
theorem potentialGradient_bound :
    BoundedLipschitzOn (fun z : Plane × Plane => potentialGradient z.1)
      (closedBall phase radius) 201 90001 := by
  have h := totalForce_log_two_default_unit_bound (X := Plane × Plane) (E := Plane) 1 (fun z => rho z.1)
    (fun z => gradient rho z.1) (fun z => z.1) (closedBall phase radius)
    (8, 200) (200, 50000) 1 (by norm_num) regional_jets.1 regional_jets.2.1
    position_bound (fun z _ => rho_floor z.1)
  have h := h.enlarge (N := 201) (K := 90001) (by norm_num) (by norm_num)
  exact ⟨h.bound_nonneg, h.lip_nonneg,
    fun z hz => by simpa only [potentialGradient, norm_neg] using h.norm_le z hz,
    fun z hz w hw => by
      simpa only [potentialGradient, neg_sub_neg, dist_comm w z] using h.norm_sub_le w hw z hz⟩

/-- A fully numerical norm and Lipschitz certificate for the actual energy
differential of this literal asset and conservative potential. -/
theorem energy_differential_bound :
    BoundedLipschitzOn (fderiv ℝ (phaseEnergy 1 rho potential (gradient rho)))
      (closedBall phase radius) 20000000 24000000000 := by
  have h := fderiv_phaseEnergy_bound (E := Plane) 1 1 (closedBall phase radius)
    rho potential (gradient rho) potentialGradient (fderiv ℝ (gradient rho))
    (8, 200) (200, 50000) (50000, 30000000) (1, 1) (201, 90001) (by norm_num)
    regional_jets.1 regional_jets.2.1 regional_jets.2.2 velocity_bound potentialGradient_bound
    (fun z _ => rho_floor z.1) (fun z _ => ruler_jet z.1) (fun z _ => gradient_jet z.1)
    (fun z hz => potential_jet z.1 ((position_bound.norm_le z hz).trans_lt (by norm_num)))
  apply h.enlarge
  all_goals norm_num [energyCovectorBudget, positionCovectorBudget,
    ControlMechanics.metricActionBudget, addBudget, productBudget, inverseBudget]

theorem phaseEnergy_differentiable (z : Plane × Plane) (hz : z ∈ closedBall phase radius) :
    DifferentiableAt ℝ (phaseEnergy 1 rho potential (gradient rho)) z := by
  exact differentiableAt_phaseEnergy 1 rho potential (gradient rho) z.1 z.2
    (ruler_jet z.1).differentiableAt (gradient_jet z.1).differentiableAt
    (potential_jet z.1 ((position_bound.norm_le z hz).trans_lt (by norm_num))).differentiableAt
    (zero_lt_one.trans_le (rho_floor z.1)).ne'

/-- Only actual endpoint inclusion remains for a force step: the asset, force,
and energy regularity constants and derivative identities have been discharged. -/
def energyStepCertificate (dt : ℝ) (z : Plane × Plane) (hz : z ∈ closedBall phase radius)
    (hend : positionVelocityStep dt z.1 z.2 (acc z) ∈ closedBall phase radius) :
    StepJetCertificate 1 dt rho potential (gradient rho) z.1 z.2 (acc z) := by
  exact StepJetCertificate.ofLipschitzRegion 1 dt rho potential (gradient rho)
    z.1 z.2 (acc z) (potentialGradient z.1) (fderiv ℝ (gradient rho) z.1)
    (closedBall phase radius) 24000000000 (convex_closedBall phase radius) hz hend
    (zero_lt_one.trans_le (rho_floor z.1)).ne' (ruler_jet z.1) (gradient_jet z.1)
    (potential_jet z.1 ((position_bound.norm_le z hz).trans_lt (by norm_num)))
    phaseEnergy_differentiable energy_differential_bound.lipschitzOnWith

/-- The certificate carries a literal regularity constant. -/
theorem energyStepCertificate_L (dt : ℝ) (z : Plane × Plane)
    (hz : z ∈ closedBall phase radius)
    (hend : positionVelocityStep dt z.1 z.2 (acc z) ∈ closedBall phase radius) :
    ((energyStepCertificate dt z hz hend).L : ℝ) = 24000000000 := by
  rfl

/-- A numerical quadratic remainder for the complete changing-configuration
step, measured against the actual phase-energy differential. -/
theorem energy_step_taylor_remainder (dt : ℝ) (z : Plane × Plane)
    (hz : z ∈ closedBall phase radius)
    (hend : positionVelocityStep dt z.1 z.2 (acc z) ∈ closedBall phase radius) :
    |phaseEnergy 1 rho potential (gradient rho) (positionVelocityStep dt z.1 z.2 (acc z)) -
      phaseEnergy 1 rho potential (gradient rho) z -
      fderiv ℝ (phaseEnergy 1 rho potential (gradient rho)) z
        (dt • (velocityEulerStep dt z.2 (acc z), acc z))| ≤
      12000000000 * dt ^ 2 * (max ‖velocityEulerStep dt z.2 (acc z)‖ ‖acc z‖) ^ 2 := by
  let c := energyStepCertificate dt z hz hend
  have h := NormalizedDistanceFields.field_quadratic_remainder
    (phaseEnergy 1 rho potential (gradient rho)) z
    (dt • (velocityEulerStep dt z.2 (acc z), acc z)) 24000000000
    c.differentiable c.variation
  norm_num only [positionVelocityStep_eq_add, norm_smul, Real.norm_eq_abs, Prod.norm_def,
    mul_pow, sq_abs, mul_assoc] at h ⊢
  exact h

/-- Every stored state of every certified exact-real refinement stays in the
phase region. This follows from the interpolant certificate at actual grid times. -/
theorem exactTrajectory_mem (N i : ℕ) (hi : i ≤ N + 1) :
    exactTrajectory N i ∈ ball phase radius := by
  obtain ⟨z, _, _, hmesh⟩ := exists_exact_scheme_certificate
  have hh : 0 < refinementStep horizon N := div_pos horizon_pos (by positivity)
  have ht : (i : ℝ) * refinementStep horizon N ∈ Icc (0 : ℝ) horizon :=
    ⟨mul_nonneg (Nat.cast_nonneg i) hh.le,
      (mul_le_mul_of_nonneg_right (Nat.cast_le.mpr hi) hh.le).trans_eq
        (refinementStep_properties horizon N horizon_pos.le).2.2⟩
  simpa only [exactInterpolant,
    trajectoryInterpolant_grid _ (exactTrajectory N) hh (N + 1) i hi] using
    ((hmesh N).2 _ ht).2.1

/-- The certified finite run discharges both endpoint hypotheses of the
changing-configuration energy remainder at every actual recurrence step. -/
theorem exactTrajectory_energy_remainder (N i : ℕ) (hi : i < N + 1) :
    |phaseEnergy 1 rho potential (gradient rho) (exactTrajectory N (i + 1)) -
      phaseEnergy 1 rho potential (gradient rho) (exactTrajectory N i) -
      fderiv ℝ (phaseEnergy 1 rho potential (gradient rho)) (exactTrajectory N i)
        (refinementStep horizon N •
          (velocityEulerStep (refinementStep horizon N) (exactTrajectory N i).2
            (acc (exactTrajectory N i)), acc (exactTrajectory N i)))| ≤
      12000000000 * (refinementStep horizon N) ^ 2 *
        (max ‖(exactTrajectory N (i + 1)).2‖ ‖acc (exactTrajectory N i)‖) ^ 2 := by
  have h := energy_step_taylor_remainder (refinementStep horizon N) (exactTrajectory N i)
    (ball_subset_closedBall (exactTrajectory_mem N i hi.le))
    (show positionVelocityStep (refinementStep horizon N) (exactTrajectory N i).1
      (exactTrajectory N i).2 (acc (exactTrajectory N i)) ∈ closedBall phase radius from
      ball_subset_closedBall (exactTrajectory_mem N (i + 1) (Nat.succ_le_of_lt hi)))
  exact h

/-- A fully numerical per-step energy remainder for every certified exact-real
run. Only the refinement index and a valid step index are supplied. -/
theorem exactTrajectory_energy_remainder_numeric (N i : ℕ) (hi : i < N + 1) :
    |phaseEnergy 1 rho potential (gradient rho) (exactTrajectory N (i + 1)) -
      phaseEnergy 1 rho potential (gradient rho) (exactTrajectory N i) -
      fderiv ℝ (phaseEnergy 1 rho potential (gradient rho)) (exactTrajectory N i)
        (refinementStep horizon N •
          (velocityEulerStep (refinementStep horizon N) (exactTrajectory N i).2
            (acc (exactTrajectory N i)), acc (exactTrajectory N i)))| ≤
      14520000000000000000000000000000000 * (refinementStep horizon N) ^ 2 := by
  have hmax : max ‖(exactTrajectory N (i + 1)).2‖ ‖acc (exactTrajectory N i)‖ ≤
      (1100000000000 : ℝ) :=
    max_le ((velocity_bound.norm_le _
      (ball_subset_closedBall (exactTrajectory_mem N (i + 1) (Nat.succ_le_of_lt hi)))).trans
        (by norm_num))
      (acc_bound.norm_le _ (ball_subset_closedBall (exactTrajectory_mem N i hi.le)))
  have hquad := mul_le_mul_of_nonneg_left
    (pow_le_pow_left₀ ((norm_nonneg _).trans (le_max_left _ _)) hmax 2)
    (show 0 ≤ 12000000000 * (refinementStep horizon N) ^ 2 by positivity)
  nlinarith [exactTrajectory_energy_remainder N i hi]

/-- The literal certified horizon turns the finite-run remainder into an
explicit quadratic refinement rate with no remaining trajectory-dependent factor. -/
theorem exactTrajectory_energy_remainder_rate (N i : ℕ) (hi : i < N + 1) :
    |phaseEnergy 1 rho potential (gradient rho) (exactTrajectory N (i + 1)) -
      phaseEnergy 1 rho potential (gradient rho) (exactTrajectory N i) -
      fderiv ℝ (phaseEnergy 1 rho potential (gradient rho)) (exactTrajectory N i)
        (refinementStep horizon N •
          (velocityEulerStep (refinementStep horizon N) (exactTrajectory N i).2
            (acc (exactTrajectory N i)), acc (exactTrajectory N i)))| ≤
      1 / (400000000 * ((N : ℝ) + 1) ^ 2) := by
  apply (exactTrajectory_energy_remainder_numeric N i hi).trans
  simpa only [refinementStep, horizon_value, Nat.cast_add, Nat.cast_one,
    div_pow, mul_div_assoc, div_div, mul_pow] using
    div_le_div_of_nonneg_right
      (by norm_num : (14520000000000000000000000000000000 : ℝ) *
        (1 / 2560000000000000000064) ^ 2 ≤ 1 / 400000000)
      (sq_nonneg ((N : ℝ) + 1))

end FractalSync.ConcreteOperationalDynamics
