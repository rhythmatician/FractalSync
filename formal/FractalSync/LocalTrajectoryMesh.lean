import FractalSync.RationalTrajectoryMesh

/-!
# A local mesh with separate speed and Lipschitz constants

The phase speed is 8 while the acceleration Lipschitz constant is 400. Keeping
these constants separate gives horizon 1/576 on the phase ball of radius 1/16.
-/

noncomputable section

namespace FractalSync.ScaleMetric

def localTrajectoryHorizon : ℝ := 1 / 576

def localTrajectoryCoefficient : ℝ :=
  refinementCoefficient localTrajectoryHorizon 400 8 1 1 1 + 8 / 2

theorem localTrajectoryHorizon_eq_quantitative :
    localTrajectoryHorizon = quantitativeHorizon (1 / 16) 8 := by
  norm_num [localTrajectoryHorizon, quantitativeHorizon]

theorem localTrajectoryHorizon_pos : 0 < localTrajectoryHorizon := by
  norm_num [localTrajectoryHorizon]

theorem localTrajectoryCoefficient_le : localTrajectoryCoefficient ≤ 16 := by
  have hexp : Real.exp ((1 / 576 : ℝ) * (400 + (1 / 576) * 400)) ≤ 3 :=
    (Real.exp_le_exp.mpr (by norm_num : (1 / 576 : ℝ) * (400 + (1 / 576) * 400) ≤ 1)).trans
      (Real.exp_one_lt_d9.le.trans (by norm_num))
  norm_num [localTrajectoryCoefficient, localTrajectoryHorizon, refinementCoefficient] at hexp ⊢
  linarith

theorem localTrajectoryMesh_error_le (N : ℕ) :
    refinementStep localTrajectoryHorizon N * localTrajectoryCoefficient ≤
      1 / (36 * ((N : ℝ) + 1)) := by
  rw [refinementStep, Nat.cast_add, Nat.cast_one, div_mul_eq_mul_div, ← div_div]
  exact div_le_div_of_nonneg_right (by
    dsimp only [localTrajectoryHorizon]
    linarith [localTrajectoryCoefficient_le])
    (by positivity)

theorem localTrajectoryMesh_fits (N : ℕ) :
    refinementStep localTrajectoryHorizon N * localTrajectoryCoefficient ≤ (1 / 16 : ℝ) / 2 := by
  apply (localTrajectoryMesh_error_le N).trans
  exact (div_le_iff₀ (by positivity)).mpr (by nlinarith [Nat.cast_nonneg (α := ℝ) N])

theorem localTrajectoryMesh_error_le_65536 (N : ℕ) (hN : 2047 ≤ N) :
    refinementStep localTrajectoryHorizon N * localTrajectoryCoefficient ≤ (1 / 65536 : ℝ) := by
  have hNr : (2047 : ℝ) ≤ N := by exact_mod_cast hN
  exact (localTrajectoryMesh_error_le N).trans
    (div_le_div₀ (by norm_num) le_rfl (by norm_num) (by nlinarith))

end FractalSync.ScaleMetric
