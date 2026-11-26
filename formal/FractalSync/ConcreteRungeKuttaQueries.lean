import FractalSync.ConcreteRungeKutta

/-!
# Query validity throughout actual held-control RK histories

Each inner substep contributes its source, three later RK stages, and its
weighted endpoint. The query allowance concerns reported real coordinates;
it does not certify any particular machine-rounding implementation.
-/

noncomputable section

namespace FractalSync.ConcreteRungeKutta

open Set Metric FlatShoreIntrinsic ScaleMetric OperationalDynamics ConcreteOperationalAsset
open ConcreteHeldControl
open ImprovedConcreteDynamics (horizon horizon_pos)

local instance concreteRKQueryNormedSpace : NormedSpace ℝ (Plane × Plane) :=
  Prod.normedSpace

/-- All phase states queried by a classical substep, including its checked endpoint. -/
def substepQueryStates (h : ℝ) (n : ℕ) (sample : Plane × Plane) (j : ℕ) :
    Set (Plane × Plane) :=
  let f := heldField sample
  let δ := h / n
  let x := RungeKutta.substeps f δ j sample
  {x, RungeKutta.stage2 f δ x, RungeKutta.stage3 f δ x,
    RungeKutta.stage4 f δ x, RungeKutta.step f δ x}

theorem substepQueryStates_finite (h : ℝ) (n : ℕ) (sample : Plane × Plane) (j : ℕ) :
    (substepQueryStates h n sample j).Finite := by
  simp [substepQueryStates]

/-- The completed inner endpoint is one of the explicitly covered query states. -/
theorem substep_endpoint_mem (h : ℝ) (n : ℕ) (sample : Plane × Plane) (j : ℕ) :
    RungeKutta.substeps (heldField sample) (h / n) (j + 1) sample ∈
      substepQueryStates h n sample j := by
  simp [substepQueryStates, RungeKutta.substeps]

theorem substepQueryStates_subset (h : ℝ) (n : ℕ) (sample : Plane × Plane) (j : ℕ)
    (c : Plane × Plane) (R : ℝ)
    (hr : RungeKutta.StageRegion (heldField sample) (h / n)
      (RungeKutta.substeps (heldField sample) (h / n) j sample) c R) :
    substepQueryStates h n sample j ⊆ closedBall c R := by
  simp only [substepQueryStates, insert_subset_iff, singleton_subset_iff]
  exact ⟨hr.start_mem, hr.stage2_mem, hr.stage3_mem, hr.stage4_mem, hr.step_mem⟩

/-- Every actual inner-stage query remains in the certified phase region. -/
theorem substepQueryStates_mem (h : ℝ) (n : ℕ) (sample : Plane × Plane) (j : ℕ)
    (hh : 0 ≤ h) (hn : 0 < n) (hj : j < n)
    (hsample : sample ∈ ball phase (radius / 2))
    (hfit : 16 * h ≤ radius / 2) (hsmall : h ≤ 1 / 2160) :
    substepQueryStates h n sample j ⊆ ball phase radius := by
  have hr := RungeKutta.outer_substeps_stageRegion hh (by norm_num : (0 : ℝ) ≤ 16) hn
    (f := heldField sample) (x := sample)
    (fun y hy => heldField_stageBall_bound sample h hsample hfit hsmall y
      (by simpa only [mul_comm h 16] using hy)) j hj
  exact fun y hy => stageBall_subset sample h hsample hfit
    (by simpa only [mul_comm h 16] using
      substepQueryStates_subset h n sample j sample (h * 16) hr hy)

/-- Domain validity holds at every inner stage of every outer step in the certified run. -/
theorem trajectory_stageQueries_mem (n N i j : ℕ)
    (hn : 0 < n) (hN : 15 ≤ N) (hi : i < N + 1) (hj : j < n) :
    substepQueryStates (refinementStep horizon N) n (trajectory n N i) j ⊆
      ball phase radius := by
  exact substepQueryStates_mem _ n _ j
    (div_pos horizon_pos (by positivity)).le hn hj
    (trajectory_mem n N i hn hN hi.le) (heldRKMesh_stage_fits N) (heldRKMesh_small N hN)

/-- All actual stage and endpoint queries tolerate a reported-coordinate error of `1/64`. -/
theorem trajectory_stage_safe_query (n N i j : ℕ)
    (hn : 0 < n) (hN : 15 ≤ N) (hi : i < N + 1) (hj : j < n)
    (y : Plane × Plane)
    (hy : y ∈ substepQueryStates (refinementStep horizon N) n (trajectory n N i) j)
    (q : OperationalField.Coords) (hquery : ‖q - chart y.1‖ ≤ (1 / 64 : ℝ)) :
    SafeReportedQuery 16 16 origin spacing q := by
  exact safe_reported_query y q
    (ball_subset_closedBall (trajectory_stageQueries_mem n N i j hn hN hi hj hy)) hquery

/-- The independently checked weighted endpoint of every internal substep is covered. -/
theorem trajectory_substep_endpoint_safe_query (n N i j : ℕ)
    (hn : 0 < n) (hN : 15 ≤ N) (hi : i < N + 1) (hj : j < n)
    (q : OperationalField.Coords)
    (hquery : ‖q - chart
      (RungeKutta.substeps (heldField (trajectory n N i)) (refinementStep horizon N / n)
        (j + 1) (trajectory n N i)).1‖ ≤ (1 / 64 : ℝ)) :
    SafeReportedQuery 16 16 origin spacing q := by
  exact trajectory_stage_safe_query n N i j hn hN hi hj _
    (substep_endpoint_mem _ n _ j) q hquery

/-- Every stored outer endpoint, including the final one, has the same query allowance. -/
theorem trajectory_endpoint_safe_query (n N i : ℕ)
    (hn : 0 < n) (hN : 15 ≤ N) (hi : i ≤ N + 1)
    (q : OperationalField.Coords)
    (hquery : ‖q - chart (trajectory n N i).1‖ ≤ (1 / 64 : ℝ)) :
    SafeReportedQuery 16 16 origin spacing q := by
  exact safe_reported_query _ q
    (ball_subset_closedBall (ball_subset_ball (by norm_num [radius])
      (trajectory_mem n N i hn hN hi))) hquery

end FractalSync.ConcreteRungeKutta
