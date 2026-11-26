import FractalSync.Analysis.FiniteDimensionalAC
import FractalSync.Analysis.AbsolutelyContinuousFTC
import Mathlib.Analysis.Normed.Module.Dual

/-!
# The fundamental theorem for finite-dimensional AC curves

Scalar AC calculus and continuous linear functionals recover the actual
vector integral. This is compatibility support for the pinned Mathlib.
-/

open Set Filter MeasureTheory

namespace AbsolutelyContinuousOnInterval

/-- Pairing two AC curves preserves actual absolute continuity. -/
theorem prodMk {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [NormedAddCommGroup F] [NormedSpace ℝ F] {f : ℝ → E} {g : ℝ → F} {a b : ℝ}
    (hf : AbsolutelyContinuousOnInterval f a b) (hg : AbsolutelyContinuousOnInterval g a b) :
    AbsolutelyContinuousOnInterval (fun t => (f t, g t)) a b := by
  simpa using ((ContinuousLinearMap.inl ℝ E F).lipschitz.comp_absolutelyContinuousOnInterval hf).fun_add
    ((ContinuousLinearMap.inr ℝ E F).lipschitz.comp_absolutelyContinuousOnInterval hg)

theorem integral_deriv_eq_sub_vector {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] [FiniteDimensional ℝ E] {f : ℝ → E} {a b : ℝ}
    (hf : AbsolutelyContinuousOnInterval f a b) :
    (∫ t in a..b, deriv f t) = f b - f a := by
  apply (NormedSpace.eq_iff_forall_dual_eq (𝕜 := ℝ)).2
  intro g
  rw [map_sub, ← g.intervalIntegral_comp_comm hf.intervalIntegrable_deriv_vector]
  change _ = (g ∘ f) b - (g ∘ f) a
  rw [← (g.lipschitz.comp_absolutelyContinuousOnInterval hf).integral_deriv_eq_sub]
  apply intervalIntegral.integral_congr_ae
  filter_upwards [hf.ae_differentiableAt] with t ht hmem
  exact (g.hasFDerivAt.comp_hasDerivAt t (ht (uIoc_subset_uIcc hmem)).hasDerivAt).deriv.symm

end AbsolutelyContinuousOnInterval
