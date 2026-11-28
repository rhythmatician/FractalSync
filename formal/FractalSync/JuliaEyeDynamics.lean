import FractalSync.PeriodTwoAttachment
import Mathlib.Analysis.SpecialFunctions.Complex.LogDeriv
import Mathlib.Dynamics.FixedPoints.Basic

/-! Quadratic orbit identities and parameter derivatives relevant to #258. -/

namespace FractalSync.JuliaEyeDynamics

open Filter Topology
open FractalSync.PeriodTwoAttachment (quadratic)

/-- At a simple fixed point, any differentiable local fixed-point branch has this derivative. -/
theorem fixedPoint_branch_derivative {p : ℂ → ℂ} {c d : ℂ}
    (hp : HasDerivAt p d c)
    (hfixed : (fun a => quadratic a (p a)) =ᶠ[𝓝 c] p)
    (hsimple : 1 - 2 * p c ≠ 0) :
    d = 1 / (1 - 2 * p c) := by
  have hpoly : HasDerivAt (fun a => quadratic a (p a)) (2 * p c * d + 1) c := by
    simpa [quadratic] using (hp.pow 2).add (hasDerivAt_id c)
  have hlinear := (hpoly.congr_of_eventuallyEq hfixed.symm).unique hp
  apply (eq_div_iff hsimple).2
  linear_combination -hlinear

/-- The two half-turn partners have exactly the same forward image. -/
theorem quadratic_neg (c z : ℂ) : quadratic c (-z) = quadratic c z := by
  simp [quadratic]

/-- The local displacement has an exact quadratic remainder. -/
theorem fixedPoint_displacement {c p : ℂ} (hp : quadratic c p = p) (w : ℂ) :
    quadratic c (p + w) - p = 2 * p * w + w ^ 2 := by
  unfold quadratic at hp ⊢
  linear_combination hp

/-- The logarithmic derivative of the fixed-point multiplier on the principal chart. -/
theorem fixedPoint_log_multiplier_derivative {p : ℂ → ℂ} {c d : ℂ}
    (hp : HasDerivAt p d c)
    (hfixed : (fun a => quadratic a (p a)) =ᶠ[𝓝 c] p)
    (hsimple : 1 - 2 * p c ≠ 0)
    (hchart : 2 * p c ∈ Complex.slitPlane) :
    HasDerivAt (fun a => Complex.log (2 * p a))
      ((2 * p c)⁻¹ * (2 * (1 / (1 - 2 * p c)))) c := by
  rw [← fixedPoint_branch_derivative hp hfixed hsimple]
  exact (Complex.hasDerivAt_log hchart).comp c (hp.const_mul 2)

/-- The negative of a fixed point is sent to the fixed point. -/
theorem fixedPoint_neg_preimage {c p : ℂ} (hp : quadratic c p = p) :
    quadratic c (-p) = p :=
  (quadratic_neg c p).trans hp

/-- Every positive iterate of the negative preimage lands at the fixed point. -/
theorem fixedPoint_neg_iterate {c p : ℂ} (hp : quadratic c p = p) (n : ℕ) :
    ((quadratic c)^[n + 1]) (-p) = p := by
  rw [Function.iterate_succ_apply, fixedPoint_neg_preimage hp]
  exact (show Function.IsFixedPt (quadratic c) p from hp).iterate n

end FractalSync.JuliaEyeDynamics
