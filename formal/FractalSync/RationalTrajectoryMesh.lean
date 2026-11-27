import FractalSync.QuantitativeDynamics
import Mathlib.Analysis.Complex.ExponentialBounds

/-!
# An explicit rational horizon and mesh threshold

A shared upper bound on acceleration Lipschitz constant and phase speed gives
a rational horizon on the phase ball of radius 1/16. Unit residual, rounding,
and initial-error coefficients then give a mesh-independent tube criterion
and an explicit O(1/(N+1)) interpolant-error bound.
-/

noncomputable section

open scoped NNReal

namespace FractalSync.ScaleMetric

def rationalTrajectoryHorizon (B : ℝ) : ℝ := 1 / (64 * (B + 1))

def rationalTrajectoryCoefficient (B : ℝ) : ℝ :=
  refinementCoefficient (rationalTrajectoryHorizon B) B B 1 1 1 + B / 2

/-- A coarse exponential estimate suffices for a linear interpolation-error
coefficient whenever the horizon and speed-horizon product are small. -/
theorem refinementCoefficient_add_half_le (T B : ℝ)
    (hT : 0 ≤ T) (hT64 : T ≤ 1 / 64) (hTB : T * B ≤ 1 / 64) (hB : 8 ≤ B) :
    refinementCoefficient T B B 1 1 1 + B / 2 ≤ B := by
  have hB0 : 0 ≤ B := by linarith
  have hT2 : T ^ 2 ≤ 1 / 4096 := by
    nlinarith [mul_nonneg hT (sub_nonneg.mpr hT64)]
  have harg : T * (B + T * B) ≤ 1 := by
    nlinarith [mul_le_mul_of_nonneg_left hTB hT]
  have hexp : Real.exp (T * (B + T * B)) ≤ 3 :=
    (Real.exp_le_exp.mpr harg).trans (Real.exp_one_lt_d9.le.trans (by norm_num))
  have hbracket : 1 + T * ((1 + T) * 1 + 1 + B + B * B / 2) ≤ B / 6 := by
    nlinarith [mul_le_mul_of_nonneg_right hTB hB0]
  rw [refinementCoefficient, max_eq_right (show 1 ≤ B by linarith)]
  nlinarith [mul_le_mul hexp hbracket (by positivity) (by norm_num : (0 : ℝ) ≤ 3)]

theorem rationalTrajectoryHorizon_properties (B : ℝ) (hB : 0 ≤ B) :
    0 < rationalTrajectoryHorizon B ∧
      rationalTrajectoryHorizon B ≤ 1 / 64 ∧ rationalTrajectoryHorizon B * B ≤ 1 / 64 := by
  have hden : 0 < 64 * (B + 1) := by positivity
  refine ⟨div_pos (by norm_num) hden, (div_le_iff₀ hden).mpr (by nlinarith), ?_⟩
  rw [rationalTrajectoryHorizon, div_mul_eq_mul_div, div_le_iff₀ hden]
  linarith

theorem rationalTrajectoryHorizon_eq_quantitative (B : ℝ≥0) :
    rationalTrajectoryHorizon B = quantitativeHorizon (1 / 16) B := by
  unfold rationalTrajectoryHorizon quantitativeHorizon
  field_simp
  norm_num

theorem rationalTrajectoryCoefficient_le (B : ℝ) (hB : 8 ≤ B) :
    rationalTrajectoryCoefficient B ≤ B := by
  obtain ⟨hT, hT64, hTB⟩ := rationalTrajectoryHorizon_properties B (by linarith)
  exact refinementCoefficient_add_half_le (rationalTrajectoryHorizon B) B hT.le hT64 hTB hB

theorem rationalTrajectoryHorizon_mul_coefficient_le (B : ℝ) (hB : 8 ≤ B) :
    rationalTrajectoryHorizon B * rationalTrajectoryCoefficient B ≤ 1 / 64 := by
  exact (mul_le_mul_of_nonneg_left (rationalTrajectoryCoefficient_le B hB)
    (rationalTrajectoryHorizon_properties B (by linarith)).1.le).trans
    (rationalTrajectoryHorizon_properties B (by linarith)).2.2

/-- Every nonempty mesh satisfies the phase-ball tube inequality. -/
theorem rationalTrajectoryMesh_fits (B : ℝ) (hB : 8 ≤ B) (N : ℕ) :
    2 * rationalTrajectoryHorizon B * rationalTrajectoryCoefficient B ≤
      (1 / 16 : ℝ) * ((N : ℝ) + 1) := by
  nlinarith [rationalTrajectoryHorizon_mul_coefficient_le B hB,
    show (0 : ℝ) ≤ N by positivity]

/-- A fully rational upper bound on the existing interpolation error coefficient. -/
theorem rationalTrajectoryMesh_error_le (B : ℝ) (hB : 8 ≤ B) (N : ℕ) :
    refinementStep (rationalTrajectoryHorizon B) N * rationalTrajectoryCoefficient B ≤
      1 / (64 * ((N : ℝ) + 1)) := by
  rw [refinementStep, Nat.cast_add, Nat.cast_one, div_mul_eq_mul_div, ← div_div]
  exact div_le_div_of_nonneg_right (rationalTrajectoryHorizon_mul_coefficient_le B hB)
    (by positivity)

/-- At least 1024 steps make the certified uniform phase error at most 2^-16. -/
theorem rationalTrajectoryMesh_error_le_65536 (B : ℝ) (hB : 8 ≤ B)
    (N : ℕ) (hN : 1023 ≤ N) :
    refinementStep (rationalTrajectoryHorizon B) N * rationalTrajectoryCoefficient B ≤
      (1 / 65536 : ℝ) := by
  have hNr : (1023 : ℝ) ≤ N := by exact_mod_cast hN
  exact (rationalTrajectoryMesh_error_le B hB N).trans
    (div_le_div₀ (by norm_num) le_rfl (by norm_num) (by nlinarith))

end FractalSync.ScaleMetric
