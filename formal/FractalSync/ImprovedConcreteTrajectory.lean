import FractalSync.ConcreteTrajectoryCertificate
import FractalSync.LocalRulerBounds
import FractalSync.LocalTrajectoryMesh
import FractalSync.CancelledCatalogBounds
import FractalSync.ConcreteLocalFieldBounds

/-!
# The concrete quadratic dynamics at a locally certified horizon

This uses the same actual field, catalog and acceleration as the preceding
fixture certificate. Only the regional estimates and certified time change.
-/

noncomputable section

namespace FractalSync.ImprovedConcreteDynamics

open Set Metric OperationalDynamics ConcreteOperationalAsset FlatShoreIntrinsic ScaleMetric
open ConcreteOperationalDynamics (rho force acc rho_floor heading_ne_zero position_bound)
open scoped NNReal

local instance improvedConcretePhaseNormedSpace : NormedSpace ℝ (Plane × Plane) :=
  Prod.normedSpace

private theorem raw_field_bound : BoundedLipschitzOn
    (fun x => OperationalField.physicalField ConcreteOperationalAsset.coefficients origin spacing (chart x))
    (closedBall position radius) (1 / 4) 4 := by
  refine ⟨by norm_num, by norm_num, ConcreteLocalFieldBounds.local_field_abs_le, ?_⟩
  exact fun x hx y hy => by
    simpa only [dist_eq_norm] using Convex.norm_image_sub_le_of_norm_hasFDerivWithin_le
      (fun z (_ : z ∈ closedBall position radius) =>
        ((OperationalField.hasFDerivAt_physicalField ConcreteOperationalAsset.coefficients
          origin (chart z) spacing).comp z chart.hasFDerivAt).hasFDerivWithinAt)
      ConcreteLocalFieldBounds.local_differential_norm_le (convex_closedBall position radius) hy hx

private theorem raw_differential_bound : BoundedLipschitzOn
    (euclideanDifferential ConcreteOperationalAsset.coefficients origin spacing)
    (closedBall position radius) 4 12 := by
  refine ⟨by norm_num, by norm_num, ConcreteLocalFieldBounds.local_differential_norm_le, ?_⟩
  exact fun x hx y hy => by
    simpa only [dist_eq_norm] using Convex.norm_image_sub_le_of_norm_hasFDerivWithin_le
      (fun z (_ : z ∈ closedBall position radius) =>
        (hasFDerivAt_euclideanDifferential ConcreteOperationalAsset.coefficients origin spacing z).hasFDerivWithinAt)
      ConcreteLocalFieldBounds.local_hessian_norm_le (convex_closedBall position radius) hy hx

private theorem raw_hessian_bound : BoundedLipschitzOn
    (euclideanHessianForm ConcreteOperationalAsset.coefficients origin spacing)
    (closedBall position radius) 12 1 := by
  exact ⟨by norm_num, by norm_num, ConcreteLocalFieldBounds.local_hessian_norm_le,
    fun x hx y hy => by simpa only [one_mul] using ConcreteLocalFieldBounds.local_hessian_sub_le x y hx hy⟩

private theorem regional_jets_of_raw_bounds
    (hf : BoundedLipschitzOn (fun x => OperationalField.physicalField
      ConcreteOperationalAsset.coefficients origin spacing (chart x)) (closedBall position radius) (1 / 4) 4)
    (hd : BoundedLipschitzOn (euclideanDifferential ConcreteOperationalAsset.coefficients origin spacing)
      (closedBall position radius) 4 12)
    (hH : BoundedLipschitzOn (euclideanHessianForm ConcreteOperationalAsset.coefficients origin spacing)
      (closedBall position radius) 12 1) :
    BoundedLipschitzOn (fun z : Plane × Plane => rho z.1) (closedBall phase radius) 2 1 ∧
      BoundedLipschitzOn (fun z : Plane × Plane => gradient rho z.1)
        (closedBall phase radius) 1 20 ∧
      BoundedLipschitzOn (fun z : Plane × Plane => fderiv ℝ (gradient rho) z.1)
        (closedBall phase radius) 20 200 := by
  have hfst : LipschitzOnWith 1 (fun z : Plane × Plane => z.1) (closedBall phase radius) :=
    LipschitzOnWith.of_dist_le_mul (fun x _ y _ => by
      simpa only [NNReal.coe_one, one_mul, dist_eq_norm] using norm_fst_le (x - y))
  have hmap : MapsTo (Prod.fst : Plane × Plane → Plane) (closedBall phase radius)
      (closedBall position radius) := fun z hz => mem_closedBall_iff_norm.mpr
        ((norm_fst_le (z - phase)).trans (mem_closedBall_iff_norm.mp hz))
  simpa only [rho, NNReal.coe_one, mul_one] using
    And.intro ((LocalRulerBounds.local_ruler_boundedLipschitz _ _ _ _
      (convex_closedBall position radius) hf hd).comp hfst hmap)
      (And.intro ((LocalRulerBounds.local_gradient_boundedLipschitz _ _ _ _
        (convex_closedBall position radius) hf hd hH).comp hfst hmap)
        ((LocalRulerBounds.local_hessian_boundedLipschitz _ _ _ _ hf hd hH).comp hfst hmap))

/-- Local bounds for the actual finite-asset ruler, Riesz gradient and Hessian. -/
theorem regional_jets :
    BoundedLipschitzOn (fun z : Plane × Plane => rho z.1) (closedBall phase radius) 2 1 ∧
      BoundedLipschitzOn (fun z : Plane × Plane => gradient rho z.1)
        (closedBall phase radius) 1 20 ∧
      BoundedLipschitzOn (fun z : Plane × Plane => fderiv ℝ (gradient rho) z.1)
        (closedBall phase radius) 20 200 := by
  exact regional_jets_of_raw_bounds raw_field_bound raw_differential_bound raw_hessian_bound

theorem velocity_bound : BoundedLipschitzOn (fun z : Plane × Plane => z.2)
    (closedBall phase radius) (1 / 3) 1 := by
  apply (velocity_boundedLipschitz phase radius radius_pos.le).enlarge ?_ le_rfl
  norm_num [phase, norm_velocity, radius]

theorem force_bound : BoundedLipschitzOn force (closedBall phase radius) 6 210 := by
  exact catalogForce_local_bound (closedBall phase radius)
    (fun z => rho z.1) (fun z => gradient rho z.1) (fun z => z.1) (fun z => z.2)
    velocity heading_ne_zero regional_jets.1 regional_jets.2.1 position_bound velocity_bound
    (fun z _ => rho_floor z.1)

theorem acc_bound : BoundedLipschitzOn acc (closedBall phase radius) 8 400 := by
  exact cancelled_catalog_acceleration_local_bound (closedBall phase radius)
    (fun z => rho z.1) (fun z => gradient rho z.1)
    (fun z => fderiv ℝ (gradient rho) z.1) (fun z => z.1) (fun z => z.2)
    velocity heading_ne_zero regional_jets.1 regional_jets.2.1 regional_jets.2.2
    position_bound velocity_bound (fun z _ => rho_floor z.1)

def horizon : ℝ := localTrajectoryHorizon

theorem horizon_value : horizon = (1 / 576 : ℝ) := by rfl

theorem horizon_pos : 0 < horizon := localTrajectoryHorizon_pos

theorem phaseField_bound (z : Plane × Plane) (hz : z ∈ closedBall phase radius) :
    ‖phaseField acc z‖ ≤ (8 : ℝ≥0) := by
  change max ‖z.2‖ ‖acc z‖ ≤ (8 : ℝ)
  exact max_le ((velocity_bound.norm_le z hz).trans (by norm_num)) (acc_bound.norm_le z hz)

/-- Actual reference with the new separate speed and Lipschitz constants. -/
theorem exists_reference : ∃ z : ℝ → Plane × Plane, z 0 = phase ∧
    ∃ c : UniformODECertificate acc z horizon (ball phase radius),
      c.accelerationLip = 400 ∧ c.phaseSpeed = 8 ∧
      ∀ t ∈ Icc (0 : ℝ) horizon, ‖z t - phase‖ ≤ 8 * t ∧
        ∀ w : Plane × Plane, ‖w - z t‖ ≤ radius / 2 → w ∈ ball phase radius := by
  rw [show horizon = quantitativeHorizon radius 8 from localTrajectoryHorizon_eq_quantitative]
  exact exists_quantitative_ODE_certificate acc phase radius 400 8
    radius_pos acc_bound.lipschitzOnWith phaseField_bound

theorem numerical_error_bound (N : ℕ) :
    refinementStep horizon N *
      (refinementCoefficient horizon 400 8 1 1 1 + (8 : ℝ) / 2) ≤
        1 / (36 * ((N : ℝ) + 1)) := by
  exact localTrajectoryMesh_error_le N

theorem numerical_error_fits (N : ℕ) :
    refinementStep horizon N *
      (refinementCoefficient horizon 400 8 1 1 1 + (8 : ℝ) / 2) ≤ radius / 2 := by
  exact localTrajectoryMesh_fits N

/-- The actual fixture controls every admitted numerical mesh on the longer interval. -/
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
          1 / (36 * ((N : ℝ) + 1)) ∧
        trajectoryInterpolant (refinementStep horizon N) y (N + 1) t ∈ ball phase radius ∧
        ∀ q : OperationalField.Coords,
          ‖q - chart (trajectoryInterpolant (refinementStep horizon N) y (N + 1) t).1‖ ≤ queryError →
          SafeReportedQuery 16 16 origin spacing q := by
  obtain ⟨z, hz0, c, hcL, hcB, htube⟩ := exists_reference
  refine ⟨z, hz0, c.equation, c.reference_mem,
    fun w hw hmem hw0 => c.solution_unique w hw hmem (hz0.trans hw0.symm), ?_⟩
  intro N y Astep hinit hres hround
  have hfit : refinementStep horizon N *
      (refinementCoefficient horizon c.accelerationLip c.phaseSpeed 1 1 1 +
        (c.phaseSpeed : ℝ) / 2) ≤ radius / 2 := by
    simpa only [hcL, hcB, NNReal.coe_ofNat] using numerical_error_fits N
  have hbound := c.interpolation_error_bound horizon_pos N y Astep 1 1 1 (radius / 2)
    (by simpa only [hz0, NNReal.coe_one, mul_one] using hinit)
    (by simpa only [NNReal.coe_one, mul_one] using hres)
    (by simpa only [NNReal.coe_one, mul_one] using hround)
    (fun t ht => (htube t ht).2) hfit
  simp only [hcL, hcB, NNReal.coe_one, NNReal.coe_ofNat] at hbound
  exact ⟨continuous_trajectoryInterpolant _ y (div_pos horizon_pos (by positivity)) _,
    fun t ht => ⟨((hbound t ht).1.trans (numerical_error_bound N)), (hbound t ht).2,
      fun q hquery => safe_reported_query _ q (ball_subset_closedBall (hbound t ht).2) hquery⟩⟩

/-- Exact-real velocity-first recursion at the new horizon, with N+1 steps. -/
def exactTrajectory (N : ℕ) : ℕ → Plane × Plane
  | 0 => phase
  | i + 1 => positionVelocityStep (refinementStep horizon N)
      (exactTrajectory N i).1 (exactTrajectory N i).2 (acc (exactTrajectory N i))

def exactInterpolant (N : ℕ) : ℝ → Plane × Plane :=
  trajectoryInterpolant (refinementStep horizon N) (exactTrajectory N) (N + 1)

/-- The literal recurrence discharges all numerical-residual premises. -/
theorem exists_exact_scheme_certificate : ∃ z : ℝ → Plane × Plane, z 0 = phase ∧
    (∀ t ∈ Icc (0 : ℝ) horizon, HasDerivAt z (phaseField acc (z t)) t) ∧
    ∀ N : ℕ, Continuous (exactInterpolant N) ∧ ∀ t ∈ Icc (0 : ℝ) horizon,
      ‖exactInterpolant N t - z t‖ ≤ 1 / (36 * ((N : ℝ) + 1)) ∧
      exactInterpolant N t ∈ ball phase radius ∧
      ∀ q : OperationalField.Coords, ‖q - chart (exactInterpolant N t).1‖ ≤ queryError →
        SafeReportedQuery 16 16 origin spacing q := by
  obtain ⟨z, hz0, heq, _, _, hmesh⟩ := exists_certified_trajectory
  refine ⟨z, hz0, heq, fun N => hmesh N (exactTrajectory N)
    (fun i => acc (exactTrajectory N i)) ?_ ?_ ?_⟩
  all_goals simp only [exactTrajectory, sub_self, norm_zero]
  case refine_1 => exact div_nonneg horizon_pos.le (by positivity)
  case refine_2 => exact fun _ _ _ => div_nonneg horizon_pos.le (by positivity)
  exact fun _ _ _ => sq_nonneg _

/-- Every stored state of the actual longer-horizon recurrence lies in the open phase ball. -/
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

/-- The exact-real 2048-step run meets the uniform phase-error target 2^-16. -/
theorem exists_2048_step_certificate : ∃ z : ℝ → Plane × Plane, z 0 = phase ∧
    (∀ t ∈ Icc (0 : ℝ) horizon, HasDerivAt z (phaseField acc (z t)) t) ∧
    Continuous (exactInterpolant 2047) ∧ ∀ t ∈ Icc (0 : ℝ) horizon,
      ‖exactInterpolant 2047 t - z t‖ ≤ (1 / 65536 : ℝ) ∧
      exactInterpolant 2047 t ∈ ball phase radius ∧
      ∀ q : OperationalField.Coords, ‖q - chart (exactInterpolant 2047 t).1‖ ≤ queryError →
        SafeReportedQuery 16 16 origin spacing q := by
  obtain ⟨z, hz0, heq, hmesh⟩ := exists_exact_scheme_certificate
  refine ⟨z, hz0, heq, (hmesh 2047).1, fun t ht => ?_⟩
  exact ⟨((hmesh 2047).2 t ht).1.trans (by norm_num), ((hmesh 2047).2 t ht).2⟩

end FractalSync.ImprovedConcreteDynamics
