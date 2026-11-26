import FractalSync.ConcreteRungeKuttaBounds
import FractalSync.RungeKuttaMesh
import FractalSync.RungeKuttaPhase

/-!
# Complete held-control RK histories for the concrete field

The enclosing-ball estimates discharge stage/domain premises at every outer
step. The reference ODE reevaluates its fixed-command drive continuously;
the numerical scheme reconstructs that drive only at outer-step anchors.
-/

noncomputable section

namespace FractalSync.ConcreteRungeKutta

open Set Metric FlatShoreIntrinsic ScaleMetric OperationalDynamics ConcreteOperationalAsset
open ConcreteOperationalDynamics (acc)
open ConcreteHeldControl
open ImprovedConcreteDynamics (horizon horizon_pos)

local instance concreteRKRunNormedSpace : NormedSpace ℝ (Plane × Plane) :=
  Prod.normedSpace

/-- One actual outer update, with a single held covector across all internal substeps. -/
def outerStep (h : ℝ) (n : ℕ) (sample : Plane × Plane) : Plane × Plane :=
  RungeKutta.substeps (heldField sample) (h / n) n sample

/-- The acceleration average is made from the actual stages of that outer update. -/
def stepAcceleration (h : ℝ) (n : ℕ) (sample : Plane × Plane) : Plane :=
  RungeKutta.averageAcceleration (frozenAcceleration (drive sample)) h n sample

/-- The complete exact-real recurrence samples the drive only at outer anchors. -/
def trajectory (n N : ℕ) : ℕ → Plane × Plane
  | 0 => phase
  | i + 1 => outerStep (refinementStep horizon N) n (trajectory n N i)

/-- Continuous affine interpolation through the actual stored outer states. -/
def interpolant (n N : ℕ) : ℝ → Plane × Plane :=
  trajectoryInterpolant (refinementStep horizon N) (trajectory n N) (N + 1)

/-- The existing interpolation theorem with the actual held-RK residual coefficients. -/
theorem exists_residual_certificate : ∃ z : ℝ → Plane × Plane, z 0 = phase ∧
    (∀ t ∈ Icc (0 : ℝ) horizon, HasDerivAt z (phaseField acc (z t)) t) ∧
    ∀ (N : ℕ), 15 ≤ N → ∀ (y : ℕ → Plane × Plane) (Astep : ℕ → Plane),
      y 0 = phase →
      (∀ i < N + 1, y i ∈ ball phase (radius / 2) →
        ‖Astep i - acc (y i)‖ ≤ refinementStep horizon N * 17280) →
      (∀ i < N + 1, y i ∈ ball phase (radius / 2) →
        ‖y (i + 1) - positionVelocityStep (refinementStep horizon N)
          (y i).1 (y i).2 (Astep i)‖ ≤ (refinementStep horizon N) ^ 2 * 32) →
      Continuous (trajectoryInterpolant (refinementStep horizon N) y (N + 1)) ∧
      ∀ t ∈ Icc (0 : ℝ) horizon,
        ‖trajectoryInterpolant (refinementStep horizon N) y (N + 1) t - z t‖ ≤
          3 / (16 * ((N : ℝ) + 1)) ∧
        trajectoryInterpolant (refinementStep horizon N) y (N + 1) t ∈ ball phase (radius / 2) ∧
        ∀ q : OperationalField.Coords,
          ‖q - chart (trajectoryInterpolant (refinementStep horizon N) y (N + 1) t).1‖ ≤ queryError →
          SafeReportedQuery 16 16 origin spacing q := by
  obtain ⟨z, hz0, c, hcL, hcB, htube⟩ := exists_inner_reference
  refine ⟨z, hz0, c.equation, ?_⟩
  intro N hN y Astep hinit hres hround
  have hfit : refinementStep horizon N *
      (refinementCoefficient horizon c.accelerationLip c.phaseSpeed 17280 32 0 +
        (c.phaseSpeed : ℝ) / 2) ≤ radius / 4 := by
    simpa only [hcL, hcB, NNReal.coe_ofNat] using heldRKMesh_fits N hN
  have hbound := c.interpolation_error_bound horizon_pos N y Astep 17280 32 0 (radius / 4)
    (by simp [hinit, hz0])
    (by simpa only [NNReal.coe_ofNat] using hres)
    (by simpa only [NNReal.coe_ofNat] using hround)
    (fun t ht => (htube t ht).2) hfit
  simp only [hcL, hcB, NNReal.coe_ofNat, NNReal.coe_zero] at hbound
  exact ⟨continuous_trajectoryInterpolant _ y (div_pos horizon_pos (by positivity)) _,
    fun t ht => ⟨((hbound t ht).1.trans (heldRKMesh_error_le N)), (hbound t ht).2,
      fun q hquery => safe_reported_query _ q
        (ball_subset_closedBall (ball_subset_ball (by norm_num [radius]) (hbound t ht).2)) hquery⟩⟩

/-- All local field premises for the complete RK outer step are discharged. -/
theorem mesh_stage_field_bounds (N : ℕ) (hN : 15 ≤ N) (sample : Plane × Plane)
    (hsample : sample ∈ ball phase (radius / 2)) :
    (∀ current ∈ closedBall sample (refinementStep horizon N * 16),
      ‖heldField sample current‖ ≤ 16) ∧
    (∀ current ∈ closedBall sample (refinementStep horizon N * 16),
      ‖frozenAcceleration (drive sample) current - acc sample‖ ≤ 1080 * ‖current - sample‖) := by
  have hfit : 16 * refinementStep horizon N ≤ radius / 2 := heldRKMesh_stage_fits N
  exact ⟨fun current hcurrent => heldField_stageBall_bound sample _ hsample hfit
      (heldRKMesh_small N hN) current (by simpa only [mul_comm] using hcurrent),
    fun current hcurrent => by
      simpa only [dist_eq_norm] using acceleration_from_anchor sample current
        (ball_subset_closedBall (ball_subset_ball (by norm_num [radius]) hsample))
        (ball_subset_closedBall (stageBall_subset sample _ hsample hfit
          (by simpa only [mul_comm] using hcurrent)))⟩

/-- Actual RK stage averages discharge both numerical residuals at an inner anchor. -/
theorem outerStep_residuals (n N : ℕ) (hn : 0 < n) (hN : 15 ≤ N)
    (sample : Plane × Plane) (hsample : sample ∈ ball phase (radius / 2)) :
    ‖stepAcceleration (refinementStep horizon N) n sample - acc sample‖ ≤
        refinementStep horizon N * 17280 ∧
      ‖outerStep (refinementStep horizon N) n sample -
        positionVelocityStep (refinementStep horizon N) sample.1 sample.2
          (stepAcceleration (refinementStep horizon N) n sample)‖ ≤
        (refinementStep horizon N) ^ 2 * 32 := by
  simpa only [outerStep, heldField, stepAcceleration,
    show (1080 : ℝ) * 16 = 17280 by norm_num,
    show (2 : ℝ) * 16 = 32 by norm_num, mul_comm] using
      RungeKutta.phase_substeps_residuals (frozenAcceleration (drive sample)) (acc sample)
        (refinementStep horizon N) 16 1080 n sample
        (div_pos horizon_pos (by positivity)).le (by norm_num) (by norm_num) hn
        (mesh_stage_field_bounds N hN sample hsample).1
        (mesh_stage_field_bounds N hN sample hsample).2

/-- The literal held-control RK recurrence has a complete finite-run certificate. -/
theorem exists_scheme_certificate : ∃ z : ℝ → Plane × Plane, z 0 = phase ∧
    (∀ t ∈ Icc (0 : ℝ) horizon, HasDerivAt z (phaseField acc (z t)) t) ∧
    ∀ n : ℕ, 0 < n → ∀ N : ℕ, 15 ≤ N →
      Continuous (interpolant n N) ∧ ∀ t ∈ Icc (0 : ℝ) horizon,
        ‖interpolant n N t - z t‖ ≤ 3 / (16 * ((N : ℝ) + 1)) ∧
        interpolant n N t ∈ ball phase (radius / 2) ∧
        ∀ q : OperationalField.Coords, ‖q - chart (interpolant n N t).1‖ ≤ queryError →
          SafeReportedQuery 16 16 origin spacing q := by
  obtain ⟨z, hz0, heq, hmesh⟩ := exists_residual_certificate
  exact ⟨z, hz0, heq, fun n hn N hN => hmesh N hN (trajectory n N)
    (fun i => stepAcceleration (refinementStep horizon N) n (trajectory n N i)) rfl
    (fun i _ hi => (outerStep_residuals n N hn hN _ hi).1)
    (fun i _ hi => (outerStep_residuals n N hn hN _ hi).2)⟩

/-- Every stored outer state lies in the inner phase region. -/
theorem trajectory_mem (n N i : ℕ) (hn : 0 < n) (hN : 15 ≤ N) (hi : i ≤ N + 1) :
    trajectory n N i ∈ ball phase (radius / 2) := by
  obtain ⟨z, _, _, hmesh⟩ := exists_scheme_certificate
  have hh : 0 < refinementStep horizon N := div_pos horizon_pos (by positivity)
  have ht : (i : ℝ) * refinementStep horizon N ∈ Icc (0 : ℝ) horizon :=
    ⟨mul_nonneg (Nat.cast_nonneg i) hh.le,
      (mul_le_mul_of_nonneg_right (Nat.cast_le.mpr hi) hh.le).trans_eq
        (refinementStep_properties horizon N horizon_pos.le).2.2⟩
  simpa only [interpolant,
    trajectoryInterpolant_grid _ (trajectory n N) hh (N + 1) i hi] using
      ((hmesh n hn N hN).2 _ ht).2.1

/-- One- and four-substep profiles, and every positive substep count, meet the same target. -/
theorem exists_16384_step_certificate : ∃ z : ℝ → Plane × Plane, z 0 = phase ∧
    (∀ t ∈ Icc (0 : ℝ) horizon, HasDerivAt z (phaseField acc (z t)) t) ∧
    ∀ n : ℕ, 0 < n → Continuous (interpolant n 16383) ∧ ∀ t ∈ Icc (0 : ℝ) horizon,
      ‖interpolant n 16383 t - z t‖ ≤ (1 / 65536 : ℝ) ∧
      interpolant n 16383 t ∈ ball phase (radius / 2) ∧
      ∀ q : OperationalField.Coords, ‖q - chart (interpolant n 16383 t).1‖ ≤ queryError →
        SafeReportedQuery 16 16 origin spacing q := by
  obtain ⟨z, hz0, heq, hmesh⟩ := exists_scheme_certificate
  exact ⟨z, hz0, heq, fun n hn => ⟨(hmesh n hn 16383 (by norm_num)).1, fun t ht =>
    ⟨((hmesh n hn 16383 (by norm_num)).2 t ht).1.trans (by norm_num),
      ((hmesh n hn 16383 (by norm_num)).2 t ht).2⟩⟩⟩

/-- Uniform convergence over the whole time interval and every positive internal count. -/
theorem exists_uniformly_convergent_scheme : ∃ z : ℝ → Plane × Plane, z 0 = phase ∧
    (∀ t ∈ Icc (0 : ℝ) horizon, HasDerivAt z (phaseField acc (z t)) t) ∧
    ∀ epsilon > 0, ∃ N₀ : ℕ, 15 ≤ N₀ ∧ ∀ N ≥ N₀, ∀ n : ℕ, 0 < n →
      ∀ t ∈ Icc (0 : ℝ) horizon,
        ‖interpolant n N t - z t‖ < epsilon ∧
          interpolant n N t ∈ ball phase (radius / 2) := by
  obtain ⟨z, hz0, heq, hmesh⟩ := exists_scheme_certificate
  refine ⟨z, hz0, heq, ?_⟩
  intro epsilon hepsilon
  obtain ⟨K, hK⟩ := exists_nat_gt ((3 : ℝ) / epsilon)
  refine ⟨max 15 K, le_max_left _ _, ?_⟩
  intro N hN n hn t ht
  have hKN : (K : ℝ) ≤ N := Nat.cast_le.mpr ((le_max_right 15 K).trans hN)
  have hnear := (hmesh n hn N ((le_max_left 15 K).trans hN)).2 t ht
  exact ⟨hnear.1.trans_lt ((div_lt_iff₀ (by positivity)).mpr (by
    have hK' := (div_lt_iff₀ hepsilon).mp hK
    have hmul := mul_le_mul_of_nonneg_right hKN hepsilon.le
    nlinarith [mul_nonneg (Nat.cast_nonneg (α := ℝ) N) hepsilon.le])), hnear.2.1⟩

end FractalSync.ConcreteRungeKutta
