import FractalSync.ConcreteRungeKuttaWork
import FractalSync.ConcreteWorkIntegral

/-!
# Convergence of actual RK work to the actual reference power integral

The finite signed ledger and the phase approximation use one common reference
solution. Affine interpolation of the stored work values converges uniformly
to its integrated drive-plus-drag power, uniformly over positive internal RK
substep counts. No numerical state-membership premises remain in the final
certificate.
-/

noncomputable section

namespace FractalSync.ConcreteRungeKutta

open Set Metric FlatShoreIntrinsic ScaleMetric ConcreteOperationalAsset
open ConcreteOperationalDynamics (acc)
open ConcreteHeldPower ConcreteWorkIntegral
open ImprovedConcreteDynamics (horizon horizon_pos)

local instance concreteWorkConvergenceNormedSpace : NormedSpace ℝ (Plane × Plane) :=
  Prod.normedSpace

/-- State error and the actual finite ledger control work error at every stored time. -/
theorem workTrajectory_reference_error (z : ℝ → Plane × Plane) (hz0 : z 0 = phase)
    (heq : ∀ t ∈ Icc (0 : ℝ) horizon, HasDerivAt z (phaseField acc (z t)) t)
    (hmem : ∀ t ∈ Icc (0 : ℝ) horizon, z t ∈ closedBall phase radius)
    (n N i : ℕ) (hn : 0 < n) (hN : 15 ≤ N) (hi : i ≤ N + 1)
    (hphase : ‖trajectory n N i - z ((i : ℝ) * refinementStep horizon N)‖ ≤
      3 / (16 * ((N : ℝ) + 1))) :
    |workTrajectory n N i - referenceWork z ((i : ℝ) * refinementStep horizon N)| ≤
      heldRKWorkGridError N n := by
  have hh : 0 < refinementStep horizon N := div_pos horizon_pos (by positivity)
  have ht : (i : ℝ) * refinementStep horizon N ∈ Icc (0 : ℝ) horizon :=
    ⟨mul_nonneg (Nat.cast_nonneg i) hh.le,
      (mul_le_mul_of_nonneg_right (Nat.cast_le.mpr hi) hh.le).trans_eq
        (refinementStep_properties horizon N horizon_pos.le).2.2⟩
  have hledger := workTrajectory_energy_defect n N i hn hN hi
  have henergy : |energy (trajectory n N i) -
      energy (z ((i : ℝ) * refinementStep horizon N))| ≤
      15 / (16 * ((N : ℝ) + 1)) := by
    calc
      _ ≤ 5 * ‖trajectory n N i - z ((i : ℝ) * refinementStep horizon N)‖ := by
        simpa only [dist_eq_norm] using ImprovedConcreteDynamics.energy_sub_le _ _
          (ball_subset_closedBall (ball_subset_ball (by norm_num [radius])
            (trajectory_mem n N i hn hN hi))) (hmem _ ht)
      _ ≤ 5 * (3 / (16 * ((N : ℝ) + 1))) :=
        mul_le_mul_of_nonneg_left hphase (by norm_num)
      _ = 15 / (16 * ((N : ℝ) + 1)) := by ring
  rw [referenceWork_eq_energy_sub z horizon heq hmem _ ht, hz0]
  have hsplit : workTrajectory n N i -
      (energy (z ((i : ℝ) * refinementStep horizon N)) - energy phase) =
      -(energy (trajectory n N i) - energy phase - workTrajectory n N i) +
        (energy (trajectory n N i) - energy (z ((i : ℝ) * refinementStep horizon N))) := by
    ring
  rw [hsplit]
  exact (abs_add_le _ _).trans (add_le_add (by simpa only [abs_neg] using hledger) henergy)

/-- One reference solution controls both the actual phase and signed work histories. -/
theorem exists_phase_work_certificate : ∃ z : ℝ → Plane × Plane, z 0 = phase ∧
    (∀ t ∈ Icc (0 : ℝ) horizon, HasDerivAt z (phaseField acc (z t)) t) ∧
    (∀ t ∈ Icc (0 : ℝ) horizon, z t ∈ closedBall phase radius) ∧
    ∀ n : ℕ, 0 < n → ∀ N : ℕ, 15 ≤ N →
      Continuous (interpolant n N) ∧ Continuous (workInterpolant n N) ∧
      ∀ t ∈ Icc (0 : ℝ) horizon,
        ‖interpolant n N t - z t‖ ≤ 3 / (16 * ((N : ℝ) + 1)) ∧
        |workInterpolant n N t - referenceWork z t| ≤ heldRKWorkInterpolationError N n ∧
        interpolant n N t ∈ ball phase (radius / 2) := by
  obtain ⟨z, hz0, heq, hmesh⟩ := exists_scheme_certificate
  have hmem : ∀ t ∈ Icc (0 : ℝ) horizon, z t ∈ closedBall phase radius := by
    intro t ht
    have hnear := (hmesh 1 (by norm_num) 15 (by norm_num)).2 t ht
    rw [mem_closedBall, dist_eq_norm]
    calc
      _ ≤ ‖z t - interpolant 1 15 t‖ + ‖interpolant 1 15 t - phase‖ :=
        norm_sub_le_norm_sub_add_norm_sub _ _ _
      _ ≤ (3 / 256 : ℝ) + radius / 2 := add_le_add
        (by rw [norm_sub_rev]; convert hnear.1 using 1; norm_num)
        (le_of_lt (by simpa only [mem_ball, dist_eq_norm] using hnear.2.1))
      _ ≤ radius := by norm_num [radius]
  refine ⟨z, hz0, heq, hmem, ?_⟩
  intro n hn N hN
  have hh : 0 < refinementStep horizon N := div_pos horizon_pos (by positivity)
  have htime := (refinementStep_properties horizon N horizon_pos.le).2.2
  have hgrid : ∀ i : ℕ, i ≤ N + 1 →
      ‖workTrajectory n N i - referenceWork z ((i : ℝ) * refinementStep horizon N)‖ ≤
        heldRKWorkGridError N n := by
    intro i hi
    have ht : (i : ℝ) * refinementStep horizon N ∈ Icc (0 : ℝ) horizon :=
      ⟨mul_nonneg (Nat.cast_nonneg i) hh.le,
        (mul_le_mul_of_nonneg_right (Nat.cast_le.mpr hi) hh.le).trans_eq htime⟩
    have hp := ((hmesh n hn N hN).2 _ ht).1
    rw [interpolant, trajectoryInterpolant_grid _ _ hh (N + 1) i hi] at hp
    simpa only [Real.norm_eq_abs] using
      workTrajectory_reference_error z hz0 heq hmem n N i hn hN hi hp
  have hwork := trajectoryInterpolant_error (refinementStep horizon N)
    (workTrajectory n N) (referenceWork z) (heldRKWorkGridError N n) 40 hh (N + 1)
    hgrid (by rw [htime]; exact referenceWork_lipschitzOn z horizon heq hmem)
  refine ⟨(hmesh n hn N hN).1, workInterpolant_continuous n N, ?_⟩
  intro t ht
  refine ⟨((hmesh n hn N hN).2 t ht).1, ?_, ((hmesh n hn N hN).2 t ht).2.1⟩
  simpa only [workInterpolant, Real.norm_eq_abs, NNReal.coe_ofNat,
    ImprovedConcreteDynamics.horizon, heldRKWork_interpolation_rate] using
      hwork t (by simpa only [htime] using ht)

/-- Both interpolants converge uniformly to one ODE solution and its actual power integral. -/
theorem exists_uniformly_convergent_phase_work : ∃ z : ℝ → Plane × Plane, z 0 = phase ∧
    (∀ t ∈ Icc (0 : ℝ) horizon, HasDerivAt z (phaseField acc (z t)) t) ∧
    (∀ t ∈ Icc (0 : ℝ) horizon,
      referenceWork z t = energy (z t) - energy (z 0)) ∧
    ∀ epsilon > 0, ∃ N₀ : ℕ, 15 ≤ N₀ ∧ ∀ N ≥ N₀, ∀ n : ℕ, 0 < n →
      ∀ t ∈ Icc (0 : ℝ) horizon,
        ‖interpolant n N t - z t‖ < epsilon ∧
        |workInterpolant n N t - referenceWork z t| < epsilon ∧
        interpolant n N t ∈ ball phase (radius / 2) := by
  obtain ⟨z, hz0, heq, hmem, hmesh⟩ := exists_phase_work_certificate
  refine ⟨z, hz0, heq, referenceWork_eq_energy_sub z horizon heq hmem, ?_⟩
  intro epsilon hepsilon
  obtain ⟨K, hK⟩ := exists_nat_gt ((9 : ℝ) / epsilon)
  refine ⟨max 15 K, le_max_left _ _, ?_⟩
  intro N hN n hn t ht
  have hKN : (K : ℝ) ≤ N := Nat.cast_le.mpr ((le_max_right 15 K).trans hN)
  have hnear := (hmesh n hn N ((le_max_left 15 K).trans hN)).2.2 t ht
  have hsmall : (9 : ℝ) / (8 * ((N : ℝ) + 1)) < epsilon := by
    apply (div_lt_iff₀ (by positivity)).mpr
    have hK' := (div_lt_iff₀ hepsilon).mp hK
    have hmul := mul_le_mul_of_nonneg_right hKN hepsilon.le
    nlinarith [mul_nonneg (Nat.cast_nonneg (α := ℝ) N) hepsilon.le]
  have hphase_le : (3 : ℝ) / (16 * ((N : ℝ) + 1)) ≤ 9 / (8 * ((N : ℝ) + 1)) := by
    field_simp
    norm_num
  exact ⟨hnear.1.trans_lt (hphase_le.trans_lt hsmall),
    hnear.2.1.trans_lt ((heldRKWork_interpolation_le N n hn).trans_lt hsmall), hnear.2.2⟩

end FractalSync.ConcreteRungeKutta
