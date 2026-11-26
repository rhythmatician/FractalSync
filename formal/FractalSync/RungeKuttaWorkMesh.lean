import FractalSync.RungeKuttaMesh

/-!
# Explicit work-ledger rates for the existing RK mesh

The energy differential has Lipschitz constant 134 and all held-field stage
slopes are bounded by 16. Their product gives a per-substep work defect of
34304 times the squared internal step size. These are exact-real bounds.
-/

noncomputable section

namespace FractalSync.ScaleMetric

def heldRKWorkGridError (N n : ℕ) : ℝ :=
  1 / (8 * ((N : ℝ) + 1) * n) + 15 / (16 * ((N : ℝ) + 1))

def heldRKWorkInterpolationError (N n : ℕ) : ℝ :=
  1 / (8 * ((N : ℝ) + 1) * n) + 35 / (36 * ((N : ℝ) + 1))

/-- Exact accumulated coefficient at the unchanged reference horizon. -/
theorem heldRKWork_coefficient :
    134 * (16 : ℝ) ^ 2 * localTrajectoryHorizon ^ 2 = 67 / 648 := by
  norm_num [localTrajectoryHorizon]

/-- One outer step has a quadratic mesh rate and an inverse internal-count factor. -/
theorem heldRKWork_step_rate (N n : ℕ) :
    34304 * (refinementStep localTrajectoryHorizon N) ^ 2 / n ≤
      1 / (8 * ((N : ℝ) + 1) ^ 2 * n) := by
  have h : (34304 : ℝ) * localTrajectoryHorizon ^ 2 ≤ 1 / 8 := by
    norm_num [localTrajectoryHorizon]
  simpa only [refinementStep, Nat.cast_add, Nat.cast_one, div_pow, mul_div_assoc,
    div_div, mul_assoc] using div_le_div_of_nonneg_right h
      (show 0 ≤ ((N : ℝ) + 1) ^ 2 * n by positivity)

/-- Every prefix of the finite history obeys the full-horizon work budget. -/
theorem heldRKWork_prefix_rate (N n i : ℕ) (hn : 0 < n) (hi : i ≤ N + 1) :
    (i : ℝ) * (34304 * (refinementStep localTrajectoryHorizon N) ^ 2 / n) ≤
      1 / (8 * ((N : ℝ) + 1) * n) := by
  calc
    _ ≤ ((N : ℝ) + 1) * (34304 * (refinementStep localTrajectoryHorizon N) ^ 2 / n) :=
      mul_le_mul_of_nonneg_right (by exact_mod_cast hi) (by positivity)
    _ ≤ ((N : ℝ) + 1) * (1 / (8 * ((N : ℝ) + 1) ^ 2 * n)) :=
      mul_le_mul_of_nonneg_left (heldRKWork_step_rate N n) (by positivity)
    _ = 1 / (8 * ((N : ℝ) + 1) * n) := by
      field_simp

/-- Affine interpolation adds half a cell of the actual reference-power bound. -/
theorem heldRKWork_interpolation_rate (N n : ℕ) :
    heldRKWorkGridError N n + 40 * refinementStep localTrajectoryHorizon N / 2 =
      heldRKWorkInterpolationError N n := by
  simp only [heldRKWorkGridError, heldRKWorkInterpolationError, refinementStep,
    localTrajectoryHorizon, Nat.cast_add, Nat.cast_one]
  field_simp
  ring

/-- One common convergence rate works for every positive internal count. -/
theorem heldRKWork_interpolation_le (N n : ℕ) (hn : 0 < n) :
    heldRKWorkInterpolationError N n ≤ 9 / (8 * ((N : ℝ) + 1)) := by
  have hn1 : (1 : ℝ) ≤ n := by exact_mod_cast (Nat.succ_le_iff.mpr hn)
  have hden : 8 * ((N : ℝ) + 1) ≤ 8 * ((N : ℝ) + 1) * n :=
    le_mul_of_one_le_right (by positivity) hn1
  have hfirst : (1 : ℝ) / (8 * ((N : ℝ) + 1) * n) ≤
      1 / (8 * ((N : ℝ) + 1)) :=
    div_le_div_of_nonneg_left (by norm_num) (by positivity) hden
  unfold heldRKWorkInterpolationError
  calc
    _ ≤ 1 / (8 * ((N : ℝ) + 1)) + 35 / (36 * ((N : ℝ) + 1)) :=
      add_le_add_right hfirst _
    _ ≤ 9 / (8 * ((N : ℝ) + 1)) := by
      field_simp
      norm_num

end FractalSync.ScaleMetric
