import FractalSync.ConcretePerturbedRungeKutta
import FractalSync.PerturbedRungeKuttaWork
import FractalSync.ConcreteWorkIntegral

/-!
# Concrete signed-work certificates with local numerical errors

The observed phase history uses one four-stage step per outer interval.
Power evaluation errors are at most h, and the residual in the complete
stored-work increment is at most h². All budgets shrink with refinement;
their satisfaction by an IEEE implementation remains separate evidence.
-/

open Set Metric
open scoped BigOperators

noncomputable section

namespace FractalSync.ConcretePerturbedRungeKutta

open FlatShoreIntrinsic ScaleMetric ConcreteOperationalAsset
open ConcreteOperationalDynamics (acc)
open ConcreteRungeKutta (heldField)
open ConcreteHeldControl ConcreteHeldPower ConcreteWorkIntegral
open ImprovedConcreteDynamics (horizon horizon_pos)
open RungeKutta.Perturbed

local instance concretePerturbedWorkNormedSpace : NormedSpace ℝ (Plane × Plane) :=
  Prod.normedSpace

/-- A scalar budget for the existing fixture's local mechanical and power errors. -/
theorem work_budget_le {h : ℝ} (hh : 0 ≤ h) (hsmall : h ≤ 1) :
    134 / 2 * (h * (16 + h) + h ^ 2) ^ 2 +
      h * (134 * (h * (16 + h) + h ^ 2) * 16 + 5 * h + h) + 5 * h ^ 2 + h ^ 2 ≤
        65536 * h ^ 2 := by
  have hD : h * (16 + h) + h ^ 2 ≤ 18 * h := by
    nlinarith [mul_nonneg hh (sub_nonneg.mpr hsmall)]
  nlinarith [pow_le_pow_left₀ (by positivity : 0 ≤ h * (16 + h) + h ^ 2) hD 2,
    mul_le_mul_of_nonneg_left hD hh]

/-- Every local energy/work premise is discharged except the explicit operation errors. -/
theorem observed_work_defect (N : ℕ) (hN : 15 ≤ N) (sample : Plane × Plane)
    (hsample : sample ∈ ball phase (radius / 2)) (s : Stages (Plane × Plane))
    (he : Residuals (heldField sample) (refinementStep horizon N) sample s
      (refinementStep horizon N) ((refinementStep horizon N) ^ 2)
      ((refinementStep horizon N) ^ 2)) (p : Powers)
    (hp : PowerResiduals (heldPower (drive sample)) sample s p (refinementStep horizon N))
    (work : ℝ) (hwork : |work - p.work (refinementStep horizon N)| ≤
      (refinementStep horizon N) ^ 2) :
    |energy s.endpoint - energy sample - work| ≤
      65536 * (refinementStep horizon N) ^ 2 := by
  have hsub : closedBall sample (18 * refinementStep horizon N) ⊆ closedBall phase radius :=
    fun y hy => ball_subset_closedBall (local_ball_subset N hN sample hsample hy)
  have hh : 0 ≤ refinementStep horizon N := (div_pos horizon_pos (by positivity)).le
  exact (energy_work_defect (observed_stage_bounds N hN sample hsample s he)
    (observed_stage_region N hN sample hsample s he) he hp hh (by norm_num) hh
    (sq_nonneg _) (by norm_num) (held_field_bound N hN sample hsample)
    (fun y hy => energy_differentiable y (hsub hy))
    (energy_differential_bound.norm_le sample
      (ball_subset_closedBall (ball_subset_ball (by norm_num [radius]) hsample)))
    (fun y hy z hz => by simpa only [dist_eq_norm] using
      energy_differential_bound.norm_sub_le y (hsub hy) z (hsub hz))
    (fun y hy => (fderiv_energy_heldField (drive sample) y (hsub hy)).symm) hwork).trans
      (work_budget_le hh ((mesh_small N hN).trans (by norm_num)))

/-- Recorded signed power and work for one observed mechanical history. -/
structure WorkHistory {N : ℕ} (r : History N) where
  powers : ℕ → Powers
  values : ℕ → ℝ
  initial : values 0 = 0
  evaluations : ∀ i < N + 1,
    PowerResiduals (heldPower (drive (r.states i))) (r.states i) (r.stages i)
      (powers i) (refinementStep horizon N)
  accumulation : ∀ i < N + 1,
    |values (i + 1) - values i - (powers i).work (refinementStep horizon N)| ≤
      (refinementStep horizon N) ^ 2

/-- Affine interpolation through the actual stored cumulative-work values. -/
def WorkHistory.interpolant {N : ℕ} {r : History N} (w : WorkHistory r) : ℝ → ℝ :=
  trajectoryInterpolant (refinementStep horizon N) w.values (N + 1)

/-- Signed defects telescope over every prefix of the stored work ledger. -/
theorem WorkHistory.energy_defect_prefix {N : ℕ} {r : History N} (w : WorkHistory r)
    (hN : 15 ≤ N) (i : ℕ) (hi : i ≤ N + 1) :
    |energy (r.states i) - energy phase - w.values i| ≤
      (i : ℝ) * (65536 * (refinementStep horizon N) ^ 2) := by
  have hlocal (j : Fin i) :
      |energy (r.states (j.val + 1)) - energy (r.states j.val) -
        (w.values (j.val + 1) - w.values j.val)| ≤
          65536 * (refinementStep horizon N) ^ 2 := by
    simpa only [r.next j.val (j.isLt.trans_le hi)] using
      observed_work_defect N hN (r.states j.val)
        (history_mem r hN j.val (j.isLt.le.trans hi)) (r.stages j.val)
        (r.errors j.val (j.isLt.trans_le hi)) (w.powers j.val)
        (w.evaluations j.val (j.isLt.trans_le hi))
        (w.values (j.val + 1) - w.values j.val) (w.accumulation j.val (j.isLt.trans_le hi))
  simpa only [Fin.sum_univ_eq_sum_range (fun j => w.values (j + 1) - w.values j) i,
    Finset.sum_range_sub, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
    nsmul_eq_mul, r.initial, w.initial, sub_zero] using
    ScaleMetric.accumulated_energy_defect i (fun j => energy (r.states j))
      (fun j => w.values (j.val + 1) - w.values j.val)
      (fun _ => 65536 * (refinementStep horizon N) ^ 2) hlocal

/-- A common full-horizon budget covers every prefix, including the endpoints. -/
theorem WorkHistory.energy_defect {N : ℕ} {r : History N} (w : WorkHistory r)
    (hN : 15 ≤ N) (i : ℕ) (hi : i ≤ N + 1) :
    |energy (r.states i) - energy phase - w.values i| ≤
      1 / (4 * ((N : ℝ) + 1)) := by
  refine (w.energy_defect_prefix hN i hi).trans
    ((mul_le_mul_of_nonneg_right (show (i : ℝ) ≤ (N : ℝ) + 1 by exact_mod_cast hi)
      (show 0 ≤ 65536 * (refinementStep horizon N) ^ 2 by positivity)).trans ?_)
  norm_num [refinementStep, horizon, localTrajectoryHorizon]
  field_simp
  norm_num

/-- Actual reference work and observed accumulated work agree up to phase and ledger error. -/
theorem WorkHistory.reference_error {N : ℕ} {r : History N} (w : WorkHistory r)
    (z : ℝ → Plane × Plane) (hz0 : z 0 = phase)
    (heq : ∀ t ∈ Icc (0 : ℝ) horizon, HasDerivAt z (phaseField acc (z t)) t)
    (hmem : ∀ t ∈ Icc (0 : ℝ) horizon, z t ∈ closedBall phase radius)
    (hN : 15 ≤ N) (i : ℕ) (hi : i ≤ N + 1)
    (hphase : ‖r.states i - z ((i : ℝ) * refinementStep horizon N)‖ ≤
      3 / (16 * ((N : ℝ) + 1))) :
    |w.values i - referenceWork z ((i : ℝ) * refinementStep horizon N)| ≤
      19 / (16 * ((N : ℝ) + 1)) := by
  have hh : 0 < refinementStep horizon N := div_pos horizon_pos (by positivity)
  have ht : (i : ℝ) * refinementStep horizon N ∈ Icc (0 : ℝ) horizon :=
    ⟨mul_nonneg (Nat.cast_nonneg i) hh.le,
      (mul_le_mul_of_nonneg_right (Nat.cast_le.mpr hi) hh.le).trans_eq
        (refinementStep_properties horizon N horizon_pos.le).2.2⟩
  have henergy : |energy (r.states i) - energy (z ((i : ℝ) * refinementStep horizon N))| ≤
      5 * ‖r.states i - z ((i : ℝ) * refinementStep horizon N)‖ := by
    simpa only [dist_eq_norm] using ImprovedConcreteDynamics.energy_sub_le _ _
      (ball_subset_closedBall (ball_subset_ball (by norm_num [radius]) (history_mem r hN i hi)))
      (hmem _ ht)
  have henergy' : |energy (r.states i) - energy (z ((i : ℝ) * refinementStep horizon N))| ≤
      15 / (16 * ((N : ℝ) + 1)) :=
    henergy.trans ((mul_le_mul_of_nonneg_left hphase (by norm_num : (0 : ℝ) ≤ 5)).trans_eq
      (by ring))
  rw [referenceWork_eq_energy_sub z horizon heq hmem _ ht, hz0]
  refine ((abs_sub_le (w.values i) (energy (r.states i) - energy phase)
    (energy (z ((i : ℝ) * refinementStep horizon N)) - energy phase)).trans
    (add_le_add (by simpa only [abs_sub_comm] using w.energy_defect hN i hi)
      (by simpa only [sub_sub_sub_cancel_right] using henergy'))).trans_eq ?_
  field_simp
  norm_num

/-- One reference controls every observed phase/work history with the shrinking local budgets. -/
theorem exists_phase_work_certificate : ∃ z : ℝ → Plane × Plane, z 0 = phase ∧
    (∀ t ∈ Icc (0 : ℝ) horizon, HasDerivAt z (phaseField acc (z t)) t) ∧
    (∀ t ∈ Icc (0 : ℝ) horizon, z t ∈ closedBall phase radius) ∧
    ∀ N : ℕ, 15 ≤ N → ∀ r : History N, ∀ w : WorkHistory r,
      Continuous r.interpolant ∧ Continuous w.interpolant ∧
      ∀ t ∈ Icc (0 : ℝ) horizon,
        ‖r.interpolant t - z t‖ ≤ 3 / (16 * ((N : ℝ) + 1)) ∧
        |w.interpolant t - referenceWork z t| ≤ 11 / (9 * ((N : ℝ) + 1)) ∧
        r.interpolant t ∈ ball phase (radius / 2) := by
  obtain ⟨z, hz0, heq, hmem, hmesh⟩ := exists_history_certificate
  refine ⟨z, hz0, heq, hmem, fun N hN r w => ?_⟩
  have hh : 0 < refinementStep horizon N := div_pos horizon_pos (by positivity)
  have hgrid (i : ℕ) (hi : i ≤ N + 1) :
      ‖w.values i - referenceWork z ((i : ℝ) * refinementStep horizon N)‖ ≤
        19 / (16 * ((N : ℝ) + 1)) := by
    have ht : (i : ℝ) * refinementStep horizon N ∈ Icc (0 : ℝ) horizon :=
      ⟨mul_nonneg (Nat.cast_nonneg i) hh.le,
        (mul_le_mul_of_nonneg_right (Nat.cast_le.mpr hi) hh.le).trans_eq
          (refinementStep_properties horizon N horizon_pos.le).2.2⟩
    have hphase : ‖r.states i - z ((i : ℝ) * refinementStep horizon N)‖ ≤
        3 / (16 * ((N : ℝ) + 1)) := by
      simpa only [History.interpolant,
        trajectoryInterpolant_grid _ r.states hh (N + 1) i hi] using
        ((hmesh N hN r).2 _ ht).1
    simpa only [Real.norm_eq_abs] using w.reference_error z hz0 heq hmem hN i hi hphase
  have hwork := trajectoryInterpolant_error (refinementStep horizon N)
    w.values (referenceWork z) (19 / (16 * ((N : ℝ) + 1))) 40 hh (N + 1) hgrid
    (by simpa only [(refinementStep_properties horizon N horizon_pos.le).2.2] using
      referenceWork_lipschitzOn z horizon heq hmem)
  refine ⟨(hmesh N hN r).1, continuous_trajectoryInterpolant _ w.values hh _,
    fun t ht => ⟨((hmesh N hN r).2 t ht).1, ?_, ((hmesh N hN r).2 t ht).2⟩⟩
  refine (hwork t (by simpa only [(refinementStep_properties horizon N horizon_pos.le).2.2]
    using ht)).trans_eq ?_
  norm_num [refinementStep, horizon, localTrajectoryHorizon, div_eq_mul_inv, mul_inv_rev]
  ring

/-- The stated local operation-error budgets imply uniform phase and actual-work convergence. -/
theorem exists_uniformly_convergent_phase_work : ∃ z : ℝ → Plane × Plane, z 0 = phase ∧
    (∀ t ∈ Icc (0 : ℝ) horizon, HasDerivAt z (phaseField acc (z t)) t) ∧
    (∀ t ∈ Icc (0 : ℝ) horizon,
      referenceWork z t = energy (z t) - energy (z 0)) ∧
    ∀ epsilon > 0, ∃ N₀ : ℕ, 15 ≤ N₀ ∧ ∀ N ≥ N₀,
      ∀ r : History N, ∀ w : WorkHistory r, ∀ t ∈ Icc (0 : ℝ) horizon,
        ‖r.interpolant t - z t‖ < epsilon ∧
        |w.interpolant t - referenceWork z t| < epsilon ∧
        r.interpolant t ∈ ball phase (radius / 2) := by
  obtain ⟨z, hz0, heq, hmem, hmesh⟩ := exists_phase_work_certificate
  refine ⟨z, hz0, heq, referenceWork_eq_energy_sub z horizon heq hmem,
    fun epsilon hepsilon => ?_⟩
  obtain ⟨K, hK⟩ := exists_nat_gt ((2 : ℝ) / epsilon)
  refine ⟨max 15 K, le_max_left _ _, fun N hN r w t ht => ?_⟩
  have hKN : (K : ℝ) ≤ N := Nat.cast_le.mpr ((le_max_right 15 K).trans hN)
  have hlarge : (2 : ℝ) < epsilon * ((N : ℝ) + 1) := by
    nlinarith [(div_lt_iff₀ hepsilon).mp hK,
      mul_le_mul_of_nonneg_right hKN hepsilon.le]
  have hnear := (hmesh N ((le_max_left 15 K).trans hN) r w).2.2 t ht
  exact ⟨hnear.1.trans_lt ((div_lt_iff₀ (by positivity)).mpr (by nlinarith)),
    hnear.2.1.trans_lt ((div_lt_iff₀ (by positivity)).mpr (by nlinarith)), hnear.2.2⟩

end FractalSync.ConcretePerturbedRungeKutta
