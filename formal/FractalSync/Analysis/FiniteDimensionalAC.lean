import FractalSync.Analysis.AbsolutelyContinuous
import FractalSync.Analysis.DerivIntegrable
import Mathlib.MeasureTheory.SpecificCodomains.Pi
import Mathlib.Analysis.Calculus.Deriv.Comp

/-!
# Integrable velocity of finite-dimensional absolutely continuous curves

This project lemma extends the scalar derivative-integrability theorem through
finite continuous linear coordinates. It uses Mathlib's actual derivative;
integrability of the curve's velocity is a conclusion, not an extra hypothesis.
-/

open Set Filter Function MeasureTheory

namespace AbsolutelyContinuousOnInterval

/-- An absolutely continuous curve in any finite-dimensional real normed space
has an interval-integrable actual vector derivative. -/
theorem intervalIntegrable_deriv_vector {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] [FiniteDimensional ℝ E] {f : ℝ → E} {a b : ℝ}
    (hf : AbsolutelyContinuousOnInterval f a b) :
    IntervalIntegrable (deriv f) volume a b := by
  let A := (Module.Basis.ofVectorSpace ℝ E).equivFun.toContinuousLinearEquiv
  rw [intervalIntegrable_iff]
  apply A.integrable_comp_iff.mp
  apply integrable_pi_iff.mpr
  intro i
  let coord : E →L[ℝ] ℝ := (ContinuousLinearMap.proj i).comp A.toContinuousLinearMap
  have hac := coord.lipschitz.comp_absolutelyContinuousOnInterval hf
  apply (intervalIntegrable_iff.mp hac.intervalIntegrable_deriv).congr
  rw [EventuallyEq, ae_restrict_iff' (by simp [uIoc])]
  filter_upwards [hf.ae_differentiableAt] with x hx hmem
  exact (coord.hasFDerivAt.comp_hasDerivAt x
    (hx (Ioc_subset_Icc_self hmem)).hasDerivAt).deriv

end AbsolutelyContinuousOnInterval
