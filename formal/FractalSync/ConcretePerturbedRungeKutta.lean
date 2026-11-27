import FractalSync.ConcreteRungeKuttaQueries
import FractalSync.PerturbedRungeKutta

/-!
# Concrete histories with local Runge--Kutta errors

One observed four-stage update is used per outer interval. Its RHS errors are
at most h, and its stage-construction and endpoint errors are at most h².
The existing exact reference and regional field bounds then certify the whole
history. These shrinking budgets are hypotheses, not IEEE error estimates.
-/

noncomputable section

namespace FractalSync.ConcretePerturbedRungeKutta

open Set Metric FlatShoreIntrinsic ScaleMetric OperationalDynamics ConcreteOperationalAsset
open ConcreteOperationalDynamics (acc)
open ConcreteRungeKutta (heldField exists_inner_reference)
open ImprovedConcreteDynamics (horizon horizon_pos)
open RungeKutta.Perturbed

local instance concretePerturbedPhaseNormedSpace : NormedSpace ℝ (Plane × Plane) :=
  Prod.normedSpace

/-- The complete observed-step defect is charged to the ordered-step error slot. -/
def coefficient : ℝ := refinementCoefficient horizon 400 8 0 18000 0 + 8 / 2

theorem coefficient_le : coefficient ≤ 108 := by
  have hexp : Real.exp ((1 / 576 : ℝ) * (400 + (1 / 576) * 400)) ≤ 3 :=
    (Real.exp_le_exp.mpr (by norm_num : (1 / 576 : ℝ) * (400 + (1 / 576) * 400) ≤ 1)).trans
      (Real.exp_one_lt_d9.le.trans (by norm_num))
  norm_num [coefficient, horizon, localTrajectoryHorizon, refinementCoefficient] at hexp ⊢
  linarith

theorem mesh_error_le (N : ℕ) :
    refinementStep horizon N * coefficient ≤ 3 / (16 * ((N : ℝ) + 1)) := by
  rw [refinementStep, Nat.cast_add, Nat.cast_one, div_mul_eq_mul_div, ← div_div]
  exact div_le_div_of_nonneg_right (by
    dsimp only [horizon, localTrajectoryHorizon]
    linarith [coefficient_le]) (by positivity)

theorem mesh_fits (N : ℕ) (hN : 15 ≤ N) :
    refinementStep horizon N * coefficient ≤ radius / 4 := by
  have hNr : (15 : ℝ) ≤ N := by exact_mod_cast hN
  exact (mesh_error_le N).trans ((div_le_iff₀ (by positivity)).mpr (by
    norm_num [radius]
    nlinarith))

theorem mesh_small (N : ℕ) (hN : 15 ≤ N) :
    refinementStep horizon N ≤ (1 / 9216 : ℝ) := by
  have hNr : (15 : ℝ) ≤ N := by exact_mod_cast hN
  exact (div_le_iff₀ (by positivity : (0 : ℝ) < ((N + 1 : ℕ) : ℝ))).mpr (by
    norm_num [horizon, localTrajectoryHorizon]
    linarith)

/-- The observed-stage ball stays inside the original physical phase region. -/
theorem local_ball_subset (N : ℕ) (hN : 15 ≤ N) (sample : Plane × Plane)
    (hsample : sample ∈ ball phase (radius / 2)) :
    closedBall sample (18 * refinementStep horizon N) ⊆ ball phase radius := by
  exact fun current hcurrent => mem_ball_iff_norm.mpr
    ((norm_sub_le_norm_sub_add_norm_sub current sample phase).trans_lt (by
      linarith [mem_closedBall_iff_norm.mp hcurrent, mem_ball_iff_norm.mp hsample,
        mesh_small N hN, show radius = (1 / 16 : ℝ) from rfl]))

/-- All regional speed premises are discharged by the actual held field. -/
theorem held_field_bound (N : ℕ) (hN : 15 ≤ N) (sample : Plane × Plane)
    (hsample : sample ∈ ball phase (radius / 2)) :
    ∀ current ∈ closedBall sample (18 * refinementStep horizon N),
      ‖heldField sample current‖ ≤ 16 := by
  exact fun current hcurrent =>
    (ConcreteRungeKutta.heldField_norm_le sample current
      (ball_subset_closedBall (ball_subset_ball (by norm_num [radius]) hsample))
      (ball_subset_closedBall (local_ball_subset N hN sample hsample hcurrent))).trans
        (by linarith [mem_closedBall.mp hcurrent, mesh_small N hN])

/-- Local errors imply bounds for the actually recorded stages and slopes. -/
theorem observed_stage_bounds (N : ℕ) (hN : 15 ≤ N) (sample : Plane × Plane)
    (hsample : sample ∈ ball phase (radius / 2)) (s : Stages (Plane × Plane))
    (he : Residuals (heldField sample) (refinementStep horizon N) sample s
      (refinementStep horizon N) ((refinementStep horizon N) ^ 2)
      ((refinementStep horizon N) ^ 2)) :
    StageBounds s (refinementStep horizon N) sample 16
      (refinementStep horizon N) ((refinementStep horizon N) ^ 2)
      ((refinementStep horizon N) ^ 2) := by
  have hh : 0 ≤ refinementStep horizon N := (div_pos horizon_pos (by positivity)).le
  have hsmall := mesh_small N hN
  exact RungeKutta.Perturbed.stageBounds he hh (by norm_num) hh (sq_nonneg _)
    (sq_nonneg _) (held_field_bound N hN sample hsample) (by
      simp only [sub_self, norm_zero, zero_add]
      nlinarith [mul_nonneg hh (sub_nonneg.mpr hsmall)])

/-- No separate stage-membership premise remains in the concrete contract. -/
theorem observed_stage_region (N : ℕ) (hN : 15 ≤ N) (sample : Plane × Plane)
    (hsample : sample ∈ ball phase (radius / 2)) (s : Stages (Plane × Plane))
    (he : Residuals (heldField sample) (refinementStep horizon N) sample s
      (refinementStep horizon N) ((refinementStep horizon N) ^ 2)
      ((refinementStep horizon N) ^ 2)) :
    StageRegion s sample sample (18 * refinementStep horizon N) := by
  have hh : 0 ≤ refinementStep horizon N := (div_pos horizon_pos (by positivity)).le
  exact (observed_stage_bounds N hN sample hsample s he).region hh (by norm_num) hh
    (sq_nonneg _) (sq_nonneg _) (by
      simp only [sub_self, norm_zero, zero_add]
      nlinarith [mul_nonneg hh (sub_nonneg.mpr (mesh_small N hN))])

/-- The local arithmetic contract supplies the whole-step residual used by the ODE theorem. -/
theorem ordered_defect (N : ℕ) (hN : 15 ≤ N) (sample : Plane × Plane)
    (hsample : sample ∈ ball phase (radius / 2)) (s : Stages (Plane × Plane))
    (he : Residuals (heldField sample) (refinementStep horizon N) sample s
      (refinementStep horizon N) ((refinementStep horizon N) ^ 2)
      ((refinementStep horizon N) ^ 2)) :
    ‖s.endpoint - positionVelocityStep (refinementStep horizon N)
      sample.1 sample.2 (acc sample)‖ ≤ (refinementStep horizon N) ^ 2 * 18000 := by
  have hh : 0 ≤ refinementStep horizon N := (div_pos horizon_pos (by positivity)).le
  have hb := observed_stage_bounds N hN sample hsample s he
  have hr := observed_stage_region N hN sample hsample s he
  have hs : sample ∈ closedBall phase radius :=
    ball_subset_closedBall (ball_subset_ball (by norm_num [radius]) hsample)
  have hsame : heldField sample sample = phaseField acc sample := by
    simp only [heldField, phaseField, ConcreteHeldControl.frozenAcceleration_at_sample]
  have hEuler := hb.euler_defect_le hr he hh (by norm_num) hh (sq_nonneg _)
    (by norm_num : (0 : ℝ) ≤ 1080) (fun current hc => by
      simpa only [hsame, dist_eq_norm] using ConcreteRungeKutta.heldField_from_anchor
        sample current hs (ball_subset_closedBall (local_ball_subset N hN sample hsample hc)))
  rw [hsame] at hEuler
  have hgap : ‖(sample + refinementStep horizon N • phaseField acc sample) -
      orderedStep acc (refinementStep horizon N) sample‖ =
      (refinementStep horizon N) ^ 2 * ‖acc sample‖ := by
    simp [orderedStep_eq, norm_smul, Real.norm_eq_abs]
    positivity
  have ht := norm_sub_le_norm_sub_add_norm_sub s.endpoint
    (sample + refinementStep horizon N • phaseField acc sample)
    (orderedStep acc (refinementStep horizon N) sample)
  rw [hgap] at ht
  change ‖s.endpoint - orderedStep acc (refinementStep horizon N) sample‖ ≤ _
  nlinarith [mul_le_mul_of_nonneg_left (ImprovedConcreteDynamics.acc_bound.norm_le sample hs)
    (sq_nonneg (refinementStep horizon N)),
    mul_nonneg (sq_nonneg (refinementStep horizon N)) (sub_nonneg.mpr (mesh_small N hN))]

/-- An observed finite history, with explicit local operation-error evidence. -/
structure History (N : ℕ) where
  states : ℕ → Plane × Plane
  stages : ℕ → Stages (Plane × Plane)
  initial : states 0 = phase
  next : ∀ i < N + 1, (stages i).endpoint = states (i + 1)
  errors : ∀ i < N + 1,
    Residuals (heldField (states i)) (refinementStep horizon N) (states i) (stages i)
      (refinementStep horizon N) ((refinementStep horizon N) ^ 2)
      ((refinementStep horizon N) ^ 2)

/-- Affine interpolation through the observed outer states. -/
def History.interpolant {N : ℕ} (r : History N) : ℝ → Plane × Plane :=
  trajectoryInterpolant (refinementStep horizon N) r.states (N + 1)

/-- One exact reference controls every history satisfying the local error budgets. -/
theorem exists_history_certificate : ∃ z : ℝ → Plane × Plane, z 0 = phase ∧
    (∀ t ∈ Icc (0 : ℝ) horizon, HasDerivAt z (phaseField acc (z t)) t) ∧
    (∀ t ∈ Icc (0 : ℝ) horizon, z t ∈ closedBall phase radius) ∧
    ∀ N : ℕ, 15 ≤ N → ∀ r : History N,
      Continuous r.interpolant ∧ ∀ t ∈ Icc (0 : ℝ) horizon,
        ‖r.interpolant t - z t‖ ≤ 3 / (16 * ((N : ℝ) + 1)) ∧
        r.interpolant t ∈ ball phase (radius / 2) := by
  obtain ⟨z, hz0, c, hcL, hcB, htube⟩ := exists_inner_reference
  refine ⟨z, hz0, c.equation, fun t ht =>
    ball_subset_closedBall (ball_subset_ball (by norm_num [radius]) (c.reference_mem t ht)), ?_⟩
  intro N hN r
  have hfit : refinementStep horizon N *
      (refinementCoefficient horizon c.accelerationLip c.phaseSpeed 0 18000 0 +
        (c.phaseSpeed : ℝ) / 2) ≤ radius / 4 := by
    simpa only [hcL, hcB, NNReal.coe_ofNat, coefficient] using mesh_fits N hN
  have hbound := c.interpolation_error_bound horizon_pos N r.states
    (fun i => acc (r.states i)) 0 18000 0 (radius / 4)
    (by simp [r.initial, hz0]) (fun i _ _ => by simp)
    (fun i hi hstate => by
      simpa only [r.next i hi, NNReal.coe_ofNat] using
        (ordered_defect N hN (r.states i) hstate (r.stages i) (r.errors i hi)))
    (fun t ht => (htube t ht).2) hfit
  simp only [hcL, hcB, NNReal.coe_ofNat, NNReal.coe_zero] at hbound
  exact ⟨continuous_trajectoryInterpolant _ r.states (div_pos horizon_pos (by positivity)) _,
    fun t ht => ⟨((hbound t ht).1.trans (mesh_error_le N)), (hbound t ht).2⟩⟩

/-- The whole-history certificate supplies every stored anchor's local premise. -/
theorem history_mem {N : ℕ} (r : History N) (hN : 15 ≤ N)
    (i : ℕ) (hi : i ≤ N + 1) : r.states i ∈ ball phase (radius / 2) := by
  obtain ⟨z, _, _, _, hmesh⟩ := exists_history_certificate
  have hh : 0 < refinementStep horizon N := div_pos horizon_pos (by positivity)
  have ht : (i : ℝ) * refinementStep horizon N ∈ Icc (0 : ℝ) horizon :=
    ⟨mul_nonneg (Nat.cast_nonneg i) hh.le,
      (mul_le_mul_of_nonneg_right (Nat.cast_le.mpr hi) hh.le).trans_eq
        (refinementStep_properties horizon N horizon_pos.le).2.2⟩
  simpa only [History.interpolant,
    trajectoryInterpolant_grid _ r.states hh (N + 1) i hi] using
      ((hmesh N hN r).2 _ ht).2

/-- Reported queries at all observed stages and endpoints inherit the existing halo guarantee. -/
theorem history_stage_safe_query {N : ℕ} (r : History N) (hN : 15 ≤ N)
    (i : ℕ) (hi : i < N + 1) (y : Plane × Plane)
    (hy : y = r.states i ∨ y = (r.stages i).stage2 ∨ y = (r.stages i).stage3 ∨
      y = (r.stages i).stage4 ∨ y = (r.stages i).endpoint)
    (q : OperationalField.Coords) (hq : ‖q - chart y.1‖ ≤ queryError) :
    SafeReportedQuery 16 16 origin spacing q := by
  have hs := history_mem r hN i hi.le
  have hr := observed_stage_region N hN (r.states i) hs (r.stages i) (r.errors i hi)
  have hym : y ∈ closedBall (r.states i) (18 * refinementStep horizon N) :=
    hy.elim (fun e => e.symm ▸ hr.start_mem) (fun h =>
      h.elim (fun e => e.symm ▸ hr.stage2_mem) (fun h =>
        h.elim (fun e => e.symm ▸ hr.stage3_mem) (fun h =>
          h.elim (fun e => e.symm ▸ hr.stage4_mem) (fun e => e.symm ▸ hr.endpoint_mem))))
  exact safe_reported_query y q
    (ball_subset_closedBall (local_ball_subset N hN (r.states i) hs hym)) hq

end FractalSync.ConcretePerturbedRungeKutta
