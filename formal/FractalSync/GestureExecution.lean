import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Analysis.Calculus.FDeriv.Comp

/-! Local execution/development calculus, conditional on actual derivatives. -/

namespace FractalSync.GestureExecution

variable {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

/-- A changing linear shape and a changing curve both contribute to velocity. -/
theorem hasDerivAt_execution (B : ℝ → E →L[ℝ] F) (u : ℝ → E)
    (t : ℝ) (dB : E →L[ℝ] F) (du : E)
    (hB : HasDerivAt B dB t) (hu : HasDerivAt u du t) :
    HasDerivAt (fun s => B s (u s)) (dB (u t) + B t du) t := by
  exact hB.clm_apply hu

/-- Traversal through a fixed parameterized curve adds its phase-rate factor. -/
theorem hasDerivAt_traversal (B : ℝ → E →L[ℝ] F) (u : ℝ → E)
    (phase : ℝ → ℝ) (t : ℝ) (dB : E →L[ℝ] F) (du : E) (rate : ℝ)
    (hB : HasDerivAt B dB t) (hu : HasDerivAt u du (phase t))
    (hp : HasDerivAt phase rate t) :
    HasDerivAt (fun s => B s (u (phase s)))
      (dB (u (phase t)) + B t (rate • du)) t := by
  exact hasDerivAt_execution B (fun s => u (phase s)) t dB (rate • du) hB (hu.scomp t hp)

end FractalSync.GestureExecution
