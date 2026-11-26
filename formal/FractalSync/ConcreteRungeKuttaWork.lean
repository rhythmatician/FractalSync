import FractalSync.ConcreteRungeKutta
import FractalSync.ConcreteHeldPower
import FractalSync.RungeKuttaWork
import FractalSync.RungeKuttaWorkMesh

/-!
# Actual work quadrature along the concrete held-control RK history

Every power sample uses the same stage state as the mechanical update and
the drive held at its outer-step anchor. The existing phase-region theorem
discharges all numerical domain premises throughout each finite history.
-/

noncomputable section

namespace FractalSync.ConcreteRungeKutta

open Set Metric FlatShoreIntrinsic ScaleMetric ConcreteOperationalAsset
open ConcreteHeldControl
open ConcreteHeldPower
open ImprovedConcreteDynamics (horizon horizon_pos)

local instance concreteRKWorkNormedSpace : NormedSpace ℝ (Plane × Plane) :=
  Prod.normedSpace

/-- Actual RK power quadrature, with one drive covector held for the whole outer step. -/
def outerWork (h : ℝ) (n : ℕ) (sample : Plane × Plane) : ℝ :=
  RungeKutta.substepsWork (heldField sample) (heldPower (drive sample)) h n sample

/-- The accumulated signed RK work along the actual stored phase history. -/
def workTrajectory (n N : ℕ) : ℕ → ℝ
  | 0 => 0
  | i + 1 => workTrajectory n N i +
      outerWork (refinementStep horizon N) n (trajectory n N i)

/-- Continuous affine interpolation of the actual accumulated RK work samples. -/
def workInterpolant (n N : ℕ) : ℝ → ℝ :=
  trajectoryInterpolant (refinementStep horizon N) (workTrajectory n N) (N + 1)

/-- All actual energy and stage premises give the outer-step quadrature defect. -/
theorem outerWork_energy_defect (n N : ℕ) (hn : 0 < n) (hN : 15 ≤ N)
    (sample : Plane × Plane) (hsample : sample ∈ ball phase (radius / 2)) :
    |energy (outerStep (refinementStep horizon N) n sample) - energy sample -
      outerWork (refinementStep horizon N) n sample| ≤
      34304 * (refinementStep horizon N) ^ 2 / n := by
  have hsub : closedBall sample (refinementStep horizon N * 16) ⊆
      closedBall phase radius := fun current hcurrent =>
    ball_subset_closedBall (stageBall_subset sample _ hsample (heldRKMesh_stage_fits N)
      (by simpa only [mul_comm] using hcurrent))
  simpa only [outerStep, outerWork, show (134 : ℝ) * 16 ^ 2 = 34304 by norm_num] using
    RungeKutta.outer_energy_work_defect
      (f := heldField sample) (energy := energy) (P := heldPower (drive sample))
      (h := refinementStep horizon N) (M := 16) (K := 134)
      (c := phase) (R := radius) (x := sample) (n := n)
      (div_pos horizon_pos (by positivity)).le (by norm_num) (by norm_num) hn
      (mesh_stage_field_bounds N hN sample hsample).1 hsub
      energy_differentiable
      (fun z hz w hw => by simpa only [dist_eq_norm] using
        energy_differential_bound.norm_sub_le z hz w hw)
      (fun z hz => (fderiv_energy_heldField (drive sample) z hz).symm)

/-- The actual work defect accumulates over every prefix, with no assumed state membership. -/
theorem workTrajectory_energy_defect_prefix (n N : ℕ) (hn : 0 < n) (hN : 15 ≤ N) :
    ∀ i : ℕ, i ≤ N + 1 →
      |energy (trajectory n N i) - energy phase - workTrajectory n N i| ≤
        (i : ℝ) * (34304 * (refinementStep horizon N) ^ 2 / n) := by
  intro i
  induction i with
  | zero =>
    intro _
    simp [trajectory, workTrajectory]
  | succ i ih =>
    intro hi
    have hi' : i ≤ N + 1 := by omega
    have hstep := outerWork_energy_defect n N hn hN (trajectory n N i)
      (trajectory_mem n N i hn hN hi')
    have hprev := ih hi'
    have heq : energy (trajectory n N (i + 1)) - energy phase -
        workTrajectory n N (i + 1) =
        (energy (trajectory n N (i + 1)) - energy (trajectory n N i) -
          outerWork (refinementStep horizon N) n (trajectory n N i)) +
        (energy (trajectory n N i) - energy phase - workTrajectory n N i) := by
      rw [workTrajectory]
      ring
    rw [heq]
    calc
      _ ≤ |energy (trajectory n N (i + 1)) - energy (trajectory n N i) -
            outerWork (refinementStep horizon N) n (trajectory n N i)| +
          |energy (trajectory n N i) - energy phase - workTrajectory n N i| :=
        abs_add_le _ _
      _ ≤ 34304 * (refinementStep horizon N) ^ 2 / n +
          (i : ℝ) * (34304 * (refinementStep horizon N) ^ 2 / n) :=
        add_le_add (by simpa only [trajectory] using hstep) hprev
      _ = ((i + 1 : ℕ) : ℝ) * (34304 * (refinementStep horizon N) ^ 2 / n) := by
        push_cast
        ring

/-- Every stored prefix satisfies the explicit full-horizon ledger budget. -/
theorem workTrajectory_energy_defect (n N i : ℕ) (hn : 0 < n) (hN : 15 ≤ N)
    (hi : i ≤ N + 1) :
    |energy (trajectory n N i) - energy phase - workTrajectory n N i| ≤
      1 / (8 * ((N : ℝ) + 1) * n) := by
  exact (workTrajectory_energy_defect_prefix n N hn hN i hi).trans
    (heldRKWork_prefix_rate N n i hn hi)

/-- The final ledger has the same bound, for every positive internal substep count. -/
theorem workTrajectory_final_energy_defect (n N : ℕ) (hn : 0 < n) (hN : 15 ≤ N) :
    |energy (trajectory n N (N + 1)) - energy phase - workTrajectory n N (N + 1)| ≤
      1 / (8 * ((N : ℝ) + 1) * n) := by
  exact workTrajectory_energy_defect n N (N + 1) hn hN le_rfl

/-- At 16384 outer steps, one and four internal substeps give budgets 1/131072 and 1/524288. -/
theorem workTrajectory_16384_energy_defect (n i : ℕ) (hn : 0 < n) (hi : i ≤ 16384) :
    |energy (trajectory n 16383 i) - energy phase - workTrajectory n 16383 i| ≤
      1 / (131072 * (n : ℝ)) := by
  simpa only [Nat.cast_ofNat, show (8 : ℝ) * (16383 + 1) = 131072 by norm_num] using
    workTrajectory_energy_defect n 16383 i hn (by norm_num) hi

/-- The accumulated work samples have a continuous affine interpolant. -/
theorem workInterpolant_continuous (n N : ℕ) : Continuous (workInterpolant n N) := by
  exact continuous_trajectoryInterpolant _ _ (div_pos horizon_pos (by positivity)) _

end FractalSync.ConcreteRungeKutta
