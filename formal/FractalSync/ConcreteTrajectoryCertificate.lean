import FractalSync.ConcreteOperationalAsset
import FractalSync.OperationalCatalogBounds
import FractalSync.RationalTrajectoryMesh

/-!
# A fully instantiated finite-asset trajectory certificate

The literal nonconstant fixture, an actual conservative potential, fixed
normalized drive, and viscous drag supply every regional coefficient premise.
The resulting constants are conservative exact-real certificates, not measured
runtime errors or recommendations for a production timestep.
-/

noncomputable section

namespace FractalSync.ConcreteOperationalDynamics

open Set Metric OperationalDynamics ConcreteOperationalAsset FlatShoreIntrinsic ScaleMetric
open MechanicalBounds
open scoped NNReal

local instance concreteTrajectoryPhaseNormedSpace : NormedSpace ℝ (Plane × Plane) :=
  Prod.normedSpace

def rho : Plane → ℝ := rulerField ConcreteOperationalAsset.coefficients origin spacing 1

/-- Explicit research configuration: a=1, kappa=log 2, mu=1/pi,
unit drive amplitude and viscous damping, and fixed positive first-axis heading. -/
def force (z : Plane × Plane) : Plane :=
  catalogForce (rho z.1) 1 (Real.log 2) (1 / Real.pi) 1 1
    (gradient rho z.1) z.1 z.2 velocity

def acc : Plane × Plane → Plane :=
  acceleration ConcreteOperationalAsset.coefficients origin spacing 1 1 force

/-- The actual ruler, gradient, and Hessian on the specified phase ball. -/
theorem regional_jets :
    BoundedLipschitzOn (fun z : Plane × Plane => rho z.1) (closedBall phase radius) 8 200 ∧
      BoundedLipschitzOn (fun z : Plane × Plane => gradient rho z.1)
        (closedBall phase radius) 200 50000 ∧
      BoundedLipschitzOn (fun z : Plane × Plane => fderiv ℝ (gradient rho) z.1)
        (closedBall phase radius) 50000 30000000 := by
  obtain ⟨hr, hu, hH⟩ := rational_operational_jets ConcreteOperationalAsset.coefficients
    origin spacing spacing_lower coefficients_bound
  have hfst : LipschitzOnWith 1 (fun z : Plane × Plane => z.1) (closedBall phase radius) :=
    LipschitzOnWith.of_dist_le_mul (fun x _ y _ => by
      simpa only [NNReal.coe_one, one_mul, dist_eq_norm] using norm_fst_le (x - y))
  simpa only [rho, NNReal.coe_one, mul_one] using
    And.intro (hr.comp hfst (fun _ _ => mem_univ _))
      (And.intro (hu.comp hfst (fun _ _ => mem_univ _))
        (hH.comp hfst (fun _ _ => mem_univ _)))

theorem position_bound : BoundedLipschitzOn (fun z : Plane × Plane => z.1)
    (closedBall phase radius) 1 1 := by
  exact ⟨zero_le_one, zero_le_one, phaseBall_position_norm_le_one,
    fun x _ y _ => by simpa only [one_mul, dist_eq_norm] using norm_fst_le (x - y)⟩

theorem velocity_bound : BoundedLipschitzOn (fun z : Plane × Plane => z.2)
    (closedBall phase radius) 1 1 := by
  apply (velocity_boundedLipschitz phase radius radius_pos.le).enlarge ?_ le_rfl
  norm_num [phase, norm_velocity, radius]

theorem rho_floor (x : Plane) : 1 ≤ rho x := by
  exact (rulerField_bounds ConcreteOperationalAsset.coefficients origin spacing 1 7
    spacing_pos (by norm_num) (by norm_num) coefficients_bound x).1

theorem heading_ne_zero : velocity ≠ 0 := by
  exact norm_pos_iff.mp (by norm_num only [norm_velocity])

/-- No force or derivative-bound premise remains in this certificate. -/
theorem force_bound : BoundedLipschitzOn force (closedBall phase radius) 400000 4000000000000 := by
  exact catalogForce_rational_bound (closedBall phase radius)
    (fun z => rho z.1) (fun z => gradient rho z.1) (fun z => z.1) (fun z => z.2)
    velocity heading_ne_zero regional_jets.1 regional_jets.2.1 position_bound velocity_bound
    (fun z _ => rho_floor z.1)

/-- Numerical norm and Lipschitz bounds for the actual changing-metric
acceleration, including the actual connection term. -/
theorem acc_bound : BoundedLipschitzOn acc (closedBall phase radius)
    1100000000000 40000000000000000000 := by
  have hb := MechanicalBounds.acceleration_bound 1 1 (closedBall phase radius)
    (fun z => rho z.1) (fun z => gradient rho z.1)
    (fun z => fderiv ℝ (gradient rho) z.1) force (fun z => z.2)
    (8, 200) (200, 50000) (50000, 30000000) (400000, 4000000000000) (1, 1)
    (by norm_num) regional_jets.1 regional_jets.2.1 regional_jets.2.2 force_bound velocity_bound
    (fun z _ => rho_floor z.1)
  apply hb.enlarge
  all_goals norm_num [accelerationBudget, raisedForceBudget, connectionBudget,
    addBudget, productBudget, inverseBudget]

/-- A shared phase-speed and acceleration-Lipschitz bound is chosen to
keep the exponential in the mesh estimate uniformly controlled. -/
def dynamicsBound : ℝ≥0 := 40000000000000000000

def horizon : ℝ := quantitativeHorizon radius dynamicsBound

theorem horizon_value : horizon = (1 / 2560000000000000000064 : ℝ) := by
  norm_num [horizon, quantitativeHorizon, radius, dynamicsBound]

theorem horizon_pos : 0 < horizon := by
  exact (quantitativeHorizon_properties radius dynamicsBound radius_pos).1

theorem phaseField_bound (z : Plane × Plane) (hz : z ∈ closedBall phase radius) :
    ‖phaseField acc z‖ ≤ dynamicsBound := by
  change max ‖z.2‖ ‖acc z‖ ≤ (dynamicsBound : ℝ)
  exact max_le ((velocity_bound.norm_le z hz).trans (by norm_num [dynamicsBound]))
    ((acc_bound.norm_le z hz).trans (by norm_num [dynamicsBound]))

/-- An actual solution and interior tube are constructed from the literal
asset and catalog. No solution, force bound, or field-jet premise is supplied. -/
theorem exists_reference : ∃ z : ℝ → Plane × Plane, z 0 = phase ∧
    ∃ c : UniformODECertificate acc z horizon (ball phase radius),
      c.accelerationLip = dynamicsBound ∧ c.phaseSpeed = dynamicsBound ∧
      ∀ t ∈ Icc (0 : ℝ) horizon, ‖z t - phase‖ ≤ (dynamicsBound : ℝ) * t ∧
        ∀ w : Plane × Plane, ‖w - z t‖ ≤ radius / 2 → w ∈ ball phase radius := by
  exact exists_quantitative_ODE_certificate acc phase radius dynamicsBound dynamicsBound
    radius_pos acc_bound.lipschitzOnWith phaseField_bound

theorem horizon_eq_rational : horizon = rationalTrajectoryHorizon dynamicsBound := by
  exact (rationalTrajectoryHorizon_eq_quantitative dynamicsBound).symm

/-- A numerical error bound with no remaining symbolic regional constant. -/
theorem numerical_error_bound (N : ℕ) :
    refinementStep horizon N *
      (refinementCoefficient horizon dynamicsBound dynamicsBound 1 1 1 + (dynamicsBound : ℝ) / 2) ≤
        1 / (64 * ((N : ℝ) + 1)) := by
  simpa only [horizon_eq_rational, rationalTrajectoryCoefficient] using
    rationalTrajectoryMesh_error_le dynamicsBound (by norm_num [dynamicsBound]) N

theorem numerical_error_fits (N : ℕ) :
    refinementStep horizon N *
      (refinementCoefficient horizon dynamicsBound dynamicsBound 1 1 1 + (dynamicsBound : ℝ) / 2) ≤
        radius / 2 := by
  apply (numerical_error_bound N).trans
  exact (div_le_iff₀ (by positivity)).mpr (by
    dsimp only [radius]
    nlinarith [Nat.cast_nonneg (α := ℝ) N])

/-- The fixed finite fixture and actual catalog construct one reference
controlling every admitted numerical mesh, including all off-grid queries.
Only numerical initial, acceleration, and stored-step errors remain inputs. -/
theorem exists_certified_trajectory : ∃ z : ℝ → Plane × Plane, z 0 = phase ∧
    (∀ t ∈ Icc (0 : ℝ) horizon, HasDerivAt z (phaseField acc (z t)) t) ∧
    (∀ t ∈ Icc (0 : ℝ) horizon, z t ∈ ball phase radius) ∧
    (∀ w : ℝ → Plane × Plane,
      (∀ t ∈ Icc (0 : ℝ) horizon, HasDerivAt w (phaseField acc (w t)) t) →
      (∀ t ∈ Icc (0 : ℝ) horizon, w t ∈ ball phase radius) →
      w 0 = phase → EqOn z w (Icc (0 : ℝ) horizon)) ∧
    ∀ (N : ℕ) (y : ℕ → Plane × Plane) (Astep : ℕ → Plane),
      ‖y 0 - phase‖ ≤ refinementStep horizon N →
      (∀ i < N + 1, y i ∈ ball phase radius →
        ‖Astep i - acc (y i)‖ ≤ refinementStep horizon N) →
      (∀ i < N + 1, y i ∈ ball phase radius →
        ‖y (i + 1) - positionVelocityStep (refinementStep horizon N)
          (y i).1 (y i).2 (Astep i)‖ ≤ (refinementStep horizon N) ^ 2) →
      Continuous (trajectoryInterpolant (refinementStep horizon N) y (N + 1)) ∧
      ∀ t ∈ Icc (0 : ℝ) horizon,
        ‖trajectoryInterpolant (refinementStep horizon N) y (N + 1) t - z t‖ ≤
          1 / (64 * ((N : ℝ) + 1)) ∧
        trajectoryInterpolant (refinementStep horizon N) y (N + 1) t ∈ ball phase radius ∧
        ∀ q : OperationalField.Coords,
          ‖q - chart (trajectoryInterpolant (refinementStep horizon N) y (N + 1) t).1‖ ≤ queryError →
          SafeReportedQuery 16 16 origin spacing q := by
  obtain ⟨z, hz0, c, hcL, hcB, htube⟩ := exists_reference
  refine ⟨z, hz0, c.equation, c.reference_mem, ?_, ?_⟩
  case refine_2 =>
    intro N y Astep hinit hres hround
    have hfit : refinementStep horizon N *
        (refinementCoefficient horizon c.accelerationLip c.phaseSpeed 1 1 1 +
          (c.phaseSpeed : ℝ) / 2) ≤ radius / 2 := by
      simpa only [hcL, hcB] using numerical_error_fits N
    have hbound := c.interpolation_error_bound horizon_pos N y Astep 1 1 1 (radius / 2)
      (by simpa only [hz0, NNReal.coe_one, mul_one] using hinit)
      (by simpa only [NNReal.coe_one, mul_one] using hres)
      (by simpa only [NNReal.coe_one, mul_one] using hround)
      (fun t ht => (htube t ht).2) hfit
    simp only [hcL, hcB, NNReal.coe_one] at hbound
    exact ⟨continuous_trajectoryInterpolant _ y (div_pos horizon_pos (by positivity)) _,
      fun t ht => ⟨((hbound t ht).1.trans (numerical_error_bound N)), (hbound t ht).2,
        fun q hquery => safe_reported_query _ q (ball_subset_closedBall (hbound t ht).2) hquery⟩⟩
  case refine_1 =>
    exact fun w hw hmem hw0 => c.solution_unique w hw hmem (hz0.trans hw0.symm)

/-- The actual velocity-first position update, evaluated over exact reals
with the literal acceleration. Refinement index N means N+1 steps. -/
def exactTrajectory (N : ℕ) : ℕ → Plane × Plane
  | 0 => phase
  | i + 1 => positionVelocityStep (refinementStep horizon N)
      (exactTrajectory N i).1 (exactTrajectory N i).2 (acc (exactTrajectory N i))

def exactInterpolant (N : ℕ) : ℝ → Plane × Plane :=
  trajectoryInterpolant (refinementStep horizon N) (exactTrajectory N) (N + 1)

/-- An unconditional exact-real finite-asset discretization certificate.
No numerical residual or coefficient premise remains. -/
theorem exists_exact_scheme_certificate : ∃ z : ℝ → Plane × Plane, z 0 = phase ∧
    (∀ t ∈ Icc (0 : ℝ) horizon, HasDerivAt z (phaseField acc (z t)) t) ∧
    ∀ N : ℕ, Continuous (exactInterpolant N) ∧ ∀ t ∈ Icc (0 : ℝ) horizon,
      ‖exactInterpolant N t - z t‖ ≤ 1 / (64 * ((N : ℝ) + 1)) ∧
      exactInterpolant N t ∈ ball phase radius ∧
      ∀ q : OperationalField.Coords, ‖q - chart (exactInterpolant N t).1‖ ≤ queryError →
        SafeReportedQuery 16 16 origin spacing q := by
  obtain ⟨z, hz0, heq, _, _, hmesh⟩ := exists_certified_trajectory
  refine ⟨z, hz0, heq, fun N => hmesh N (exactTrajectory N)
    (fun i => acc (exactTrajectory N i)) ?_ ?_ ?_⟩
  all_goals simp only [exactTrajectory, sub_self, norm_zero]
  · exact div_nonneg horizon_pos.le (by positivity)
  · exact fun _ _ _ => div_nonneg horizon_pos.le (by positivity)
  · exact fun _ _ _ => sq_nonneg _

/-- The 1024-step exact-real run has a uniform phase error at most 2^-16,
and every reported query within 1/64 has all 36 valid stencil addresses. -/
theorem exists_1024_step_certificate : ∃ z : ℝ → Plane × Plane, z 0 = phase ∧
    (∀ t ∈ Icc (0 : ℝ) horizon, HasDerivAt z (phaseField acc (z t)) t) ∧
    Continuous (exactInterpolant 1023) ∧ ∀ t ∈ Icc (0 : ℝ) horizon,
      ‖exactInterpolant 1023 t - z t‖ ≤ (1 / 65536 : ℝ) ∧
      exactInterpolant 1023 t ∈ ball phase radius ∧
      ∀ q : OperationalField.Coords, ‖q - chart (exactInterpolant 1023 t).1‖ ≤ queryError →
        SafeReportedQuery 16 16 origin spacing q := by
  obtain ⟨z, hz0, heq, hmesh⟩ := exists_exact_scheme_certificate
  exact ⟨z, hz0, heq, by convert hmesh 1023 using 1; norm_num⟩

end FractalSync.ConcreteOperationalDynamics
