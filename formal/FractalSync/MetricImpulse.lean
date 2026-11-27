import FractalSync.ScaleMetric
import Mathlib.LinearAlgebra.SesquilinearForm.Basic

/-!
# Fixed-configuration metric impulses

Exact-real consequences of the scale metric for #159. This module reuses the
single metric and inverse definitions from `ScaleMetric`. An impulse is a
covector, represented using the ambient real inner product, and changes velocity
by `inverseAction rho a u J`. No assertion about finite-step integration is made.
-/

open scoped InnerProductSpace

namespace FractalSync.ScaleMetric

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem metric_add_right (rho a : ℝ) (u v w z : E) :
    metric rho a u v (w + z) = metric rho a u v w + metric rho a u v z := by
  rw [metric_symm rho a u v (w + z), metric_add_left,
    metric_symm rho a u w v, metric_symm rho a u z v]

theorem metric_smul_right (rho a c : ℝ) (u v w : E) :
    metric rho a u v (c • w) = c * metric rho a u v w := by
  rw [metric_symm rho a u v (c • w), metric_smul_left, metric_symm rho a u w v]

/-- Bundling the already-defined metric allows reuse of Mathlib's general
positive-semidefinite-form inequalities. This is not a second metric definition. -/
noncomputable def metricBilinForm (rho a : ℝ) (u : E) : LinearMap.BilinForm ℝ E :=
  LinearMap.mk₂ ℝ (metric rho a u)
    (metric_add_left rho a u) (fun c => metric_smul_left rho a c u)
    (metric_add_right rho a u) (fun c => metric_smul_right rho a c u)

theorem metric_cauchy_schwarz_sq (rho a : ℝ) (u v w : E) :
    metric rho a u v w ^ 2 ≤ metric rho a u v v * metric rho a u w w := by
  exact (metricBilinForm rho a u).apply_sq_le_of_symm
    (fun x => metric_self_nonneg rho a u x) ⟨fun x y => metric_symm rho a u x y⟩ v w

/-- Pairing a velocity with the metric-raised impulse recovers ordinary covector work. -/
theorem metric_inverse_pairing (rho a : ℝ) (u v J : E) (hrho : rho ≠ 0) :
    metric rho a u v (inverseAction rho a u J) = ⟪v, J⟫_ℝ := by
  rw [← inner_metricAction, metric_inverseAction rho a u J hrho]

noncomputable def dualSq (rho a : ℝ) (u J : E) : ℝ :=
  ⟪J, inverseAction rho a u J⟫_ℝ

noncomputable def dualNorm (rho a : ℝ) (u J : E) : ℝ := Real.sqrt (dualSq rho a u J)

/-- The dual squared norm equals the primal squared norm of the raised impulse. -/
theorem dualSq_eq_metric (rho a : ℝ) (u J : E) (hrho : rho ≠ 0) :
    dualSq rho a u J =
      metric rho a u (inverseAction rho a u J) (inverseAction rho a u J) := by
  rw [dualSq, metric_inverse_pairing rho a u (inverseAction rho a u J) J hrho,
    real_inner_comm J (inverseAction rho a u J)]

theorem dualSq_nonneg (rho a : ℝ) (u J : E) (hrho : rho ≠ 0) :
    0 ≤ dualSq rho a u J := by
  rw [dualSq_eq_metric rho a u J hrho]
  exact metric_self_nonneg rho a u (inverseAction rho a u J)

/-- The complete quadratic expansion, before identifying a covector impulse. -/
theorem kinetic_add (rho a : ℝ) (u v w : E) :
    kinetic rho a u (v + w) - kinetic rho a u v =
      metric rho a u v w + metric rho a u w w / 2 := by
  simp only [kinetic, metric_add_left, metric_add_right, metric_symm rho a u w v]
  ring

/-- Exact work plus quadratic correction for a fixed-configuration covector impulse. -/
theorem impulse_energy_jump (rho a : ℝ) (u v J : E) (hrho : rho ≠ 0) :
    kinetic rho a u (v + inverseAction rho a u J) - kinetic rho a u v =
      ⟪v, J⟫_ℝ + dualSq rho a u J / 2 := by
  rw [kinetic_add, metric_inverse_pairing rho a u v J hrho,
    ← dualSq_eq_metric rho a u J hrho]

theorem two_kinetic (rho a : ℝ) (u v : E) :
    2 * kinetic rho a u v = metric rho a u v v := by
  dsimp [kinetic]
  ring

theorem kinetic_nonneg (rho a : ℝ) (u v : E) : 0 ≤ kinetic rho a u v := by
  exact div_nonneg (metric_self_nonneg rho a u v) (by norm_num)

theorem impulse_pairing_sq (rho a : ℝ) (u v J : E) (hrho : rho ≠ 0) :
    ⟪v, J⟫_ℝ ^ 2 ≤ (2 * kinetic rho a u v) * dualSq rho a u J := by
  rw [two_kinetic, dualSq_eq_metric rho a u J hrho, ← metric_inverse_pairing rho a u v J hrho]
  exact metric_cauchy_schwarz_sq rho a u v (inverseAction rho a u J)

theorem dualNorm_sq (rho a : ℝ) (u J : E) (hrho : rho ≠ 0) :
    dualNorm rho a u J ^ 2 = dualSq rho a u J := by
  exact Real.sq_sqrt (dualSq_nonneg rho a u J hrho)

/-- Primal/dual Cauchy--Schwarz, with the actual intrinsic speed. -/
theorem impulse_pairing_bound (rho a : ℝ) (u v J : E) (hrho : rho ≠ 0) :
    |⟪v, J⟫_ℝ| ≤ Real.sqrt (2 * kinetic rho a u v) * dualNorm rho a u J := by
  refine abs_le_of_sq_le_sq ?_ (mul_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _))
  simpa only [mul_pow, dualNorm_sq rho a u J hrho,
    Real.sq_sqrt (mul_nonneg (by norm_num : (0 : ℝ) ≤ 2) (kinetic_nonneg rho a u v))]
    using impulse_pairing_sq rho a u v J hrho

/-- A bounded dual-metric impulse has a scale-independent upper bound on its
kinetic-energy jump. The metric and position are fixed during the impulse. -/
theorem impulse_energy_bound (rho a Jmax : ℝ) (u v J : E) (hrho : rho ≠ 0)
    (hJ : dualNorm rho a u J ≤ Jmax) :
    kinetic rho a u (v + inverseAction rho a u J) - kinetic rho a u v ≤
      Real.sqrt (2 * kinetic rho a u v) * Jmax + Jmax ^ 2 / 2 := by
  rw [impulse_energy_jump rho a u v J hrho, ← dualNorm_sq rho a u J hrho]
  have hnonneg : 0 ≤ dualNorm rho a u J := Real.sqrt_nonneg _
  exact add_le_add
    ((le_abs_self _).trans ((impulse_pairing_bound rho a u v J hrho).trans
      (mul_le_mul_of_nonneg_left hJ (Real.sqrt_nonneg _))))
    (div_le_div_of_nonneg_right ((sq_le_sq₀ hnonneg (hnonneg.trans hJ)).mpr hJ)
      (by norm_num))

end FractalSync.ScaleMetric
