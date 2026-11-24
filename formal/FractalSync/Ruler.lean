import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.Calculus.MeanValue
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Topology.MetricSpace.HausdorffDistance
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# The scalar regularized ruler and its actual differential

The scalar map is smooth when epsilon is positive. Composing it with distance
to a closed set preserves the Lipschitz bound, not global differentiability.
Differential composition theorems require the derivative of the underlying
field explicitly. This distinction is essential for issues #145, #159 and #160.
-/

namespace FractalSync

noncomputable def ruler (epsilon r : ℝ) : ℝ := Real.sqrt (r ^ 2 + epsilon ^ 2)

theorem ruler_pos (epsilon r : ℝ) (he : 0 < epsilon) : 0 < ruler epsilon r := by
  unfold ruler
  positivity

theorem ruler_sq (epsilon r : ℝ) : (ruler epsilon r) ^ 2 = r ^ 2 + epsilon ^ 2 := by
  exact Real.sq_sqrt (by positivity)

theorem abs_le_ruler (epsilon r : ℝ) : |r| ≤ ruler epsilon r := by
  exact Real.abs_le_sqrt (by nlinarith [sq_nonneg epsilon])

theorem epsilon_le_ruler (epsilon r : ℝ) (he : 0 ≤ epsilon) :
    epsilon ≤ ruler epsilon r := by
  have h := Real.abs_le_sqrt (x := epsilon) (y := r ^ 2 + epsilon ^ 2)
    (by nlinarith [sq_nonneg r])
  simpa [ruler, abs_of_nonneg he] using h

theorem hasDerivAt_ruler (epsilon r : ℝ) (he : 0 < epsilon) :
    HasDerivAt (ruler epsilon) (r / ruler epsilon r) r := by
  have hpos : 0 < r ^ 2 + epsilon ^ 2 := by positivity
  convert (((hasDerivAt_id r).pow 2).add_const (epsilon ^ 2)).sqrt hpos.ne' using 1
  simp only [ruler, Nat.cast_ofNat, Pi.pow_apply, id_eq, mul_one]
  ring

/-- Positive regularization preserves the global one-Lipschitz bound. -/
theorem ruler_lipschitz (epsilon : ℝ) (he : 0 < epsilon) :
    LipschitzWith 1 (ruler epsilon) := by
  apply lipschitzWith_of_nnnorm_deriv_le
    (fun r => (hasDerivAt_ruler epsilon r he).differentiableAt)
  intro r
  rw [(hasDerivAt_ruler epsilon r he).deriv]
  change |r / ruler epsilon r| ≤ 1
  rw [abs_div, abs_of_pos (ruler_pos epsilon r he)]
  exact (div_le_one (ruler_pos epsilon r he)).2 (abs_le_ruler epsilon r)

/-- The regularized distance to any set is one-Lipschitz. No smoothness of the
set or distance field is inferred from this statement. -/
theorem regularizedDistance_lipschitz {E : Type*} [PseudoMetricSpace E]
    (epsilon : ℝ) (he : 0 < epsilon) (S : Set E) :
    LipschitzWith 1 (fun x => ruler epsilon (Metric.infDist x S)) := by
  simpa using (ruler_lipschitz epsilon he).comp (Metric.lipschitz_infDist_pt S)

/-- Chain rule for a differentiable underlying field. This is a genuine
Fréchet derivative in an arbitrary real normed space. -/
theorem hasFDerivAt_ruler_comp {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (epsilon : ℝ) (he : 0 < epsilon) (D : E → ℝ) (D' : E →L[ℝ] ℝ) (x : E)
    (hD : HasFDerivAt D D' x) :
    HasFDerivAt (fun y => ruler epsilon (D y))
      ((D x / ruler epsilon (D x)) • D') x := by
  exact (hasDerivAt_ruler epsilon (D x) he).comp_hasFDerivAt x hD

/-- The scalar second derivative responsible for the positive rank-one term in
the GeometryJet Hessian formula. -/
theorem hasDerivAt_ruler_slope (epsilon r : ℝ) (he : 0 < epsilon) :
    HasDerivAt (fun t => t / ruler epsilon t)
      (epsilon ^ 2 / (ruler epsilon r) ^ 3) r := by
  have hne := (ruler_pos epsilon r he).ne'
  convert (hasDerivAt_id r).div (hasDerivAt_ruler epsilon r he) hne using 1
  simp only [id_eq, one_mul]
  field_simp
  nlinarith [ruler_sq epsilon r]

end FractalSync
