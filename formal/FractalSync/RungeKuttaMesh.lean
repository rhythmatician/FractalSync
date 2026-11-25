import FractalSync.LocalTrajectoryMesh

/-!
# A mesh budget for exact-real held-control RK histories

The complete-stage bounds give acceleration residual 17280 h and a comparison
with the velocity-first scheme of 32 h². These are consistency estimates for
the held-control policy, not a fourth-order error claim for the target ODE.
-/

noncomputable section

namespace FractalSync.ScaleMetric

def heldRKCoefficient : ℝ :=
  refinementCoefficient localTrajectoryHorizon 400 8 17280 32 0 + 8 / 2

theorem heldRKCoefficient_le : heldRKCoefficient ≤ 108 := by
  have hexp : Real.exp ((1 / 576 : ℝ) * (400 + (1 / 576) * 400)) ≤ 3 :=
    (Real.exp_le_exp.mpr (by norm_num : (1 / 576 : ℝ) * (400 + (1 / 576) * 400) ≤ 1)).trans
      (Real.exp_one_lt_d9.le.trans (by norm_num))
  norm_num [heldRKCoefficient, localTrajectoryHorizon, refinementCoefficient] at hexp ⊢
  linarith

/-- The all-time phase error is first order in the complete outer mesh. -/
theorem heldRKMesh_error_le (N : ℕ) :
    refinementStep localTrajectoryHorizon N * heldRKCoefficient ≤
      3 / (16 * ((N : ℝ) + 1)) := by
  rw [refinementStep, Nat.cast_add, Nat.cast_one, div_mul_eq_mul_div, ← div_div]
  exact div_le_div_of_nonneg_right (by
    dsimp only [localTrajectoryHorizon]
    linarith [heldRKCoefficient_le]) (by positivity)

/-- Sixteen or more outer steps fit the smaller reference tube. -/
theorem heldRKMesh_fits (N : ℕ) (hN : 15 ≤ N) :
    refinementStep localTrajectoryHorizon N * heldRKCoefficient ≤ (1 / 16 : ℝ) / 4 := by
  apply (heldRKMesh_error_le N).trans
  have hNr : (15 : ℝ) ≤ N := by exact_mod_cast hN
  exact (div_le_iff₀ (by positivity)).mpr (by nlinarith)

/-- The stage-enclosing ball fits inside half the original phase radius. -/
theorem heldRKMesh_stage_fits (N : ℕ) :
    16 * refinementStep localTrajectoryHorizon N ≤ (1 / 16 : ℝ) / 2 := by
  rw [refinementStep, ← mul_div_assoc]
  exact (div_le_iff₀ (by positivity)).mpr (by
    norm_num [localTrajectoryHorizon]
    nlinarith [Nat.cast_nonneg (α := ℝ) N])

/-- The selected mesh closes the held-field speed bootstrap. -/
theorem heldRKMesh_small (N : ℕ) (hN : 15 ≤ N) :
    refinementStep localTrajectoryHorizon N ≤ (1 / 2160 : ℝ) := by
  rw [refinementStep]
  have hNr : (15 : ℝ) ≤ N := by exact_mod_cast hN
  exact (div_le_iff₀ (by positivity)).mpr (by
    norm_num [localTrajectoryHorizon]
    linarith)

/-- An explicit dyadic outer mesh meets the phase-error target 2^-16. -/
theorem heldRKMesh_error_le_65536 (N : ℕ) (hN : 16383 ≤ N) :
    refinementStep localTrajectoryHorizon N * heldRKCoefficient ≤ (1 / 65536 : ℝ) := by
  have hNr : (16383 : ℝ) ≤ N := by exact_mod_cast hN
  exact (heldRKMesh_error_le N).trans
    ((div_le_iff₀ (by positivity)).mpr (by nlinarith))

end FractalSync.ScaleMetric
