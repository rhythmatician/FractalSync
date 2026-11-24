import Mathlib.Data.Real.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Pointwise algebra for regularized distance graphs

These statements verify the local real-arithmetic step used by the conventional
proof in `docs/research/160-distance-geometry.md`. They do not formalize absolute
continuity, length integration, nearest-point existence, or the singular limit.
`speed` represents Euclidean speed; `radialSpeed` represents the derivative of
the distance-to-set function along a curve.
-/

namespace FractalSync.DistanceGraph

noncomputable def graphSpeedSq (a epsilon r speed radialSpeed : ℝ) : ℝ :=
  speed ^ 2 / (r ^ 2 + epsilon ^ 2) +
    a ^ 2 * r ^ 2 * radialSpeed ^ 2 / (r ^ 2 + epsilon ^ 2) ^ 2

noncomputable def radialSpeedSq (a epsilon r radialSpeed : ℝ) : ℝ :=
  (epsilon ^ 2 + (1 + a ^ 2) * r ^ 2) * radialSpeed ^ 2 /
    (r ^ 2 + epsilon ^ 2) ^ 2

/-- Exact remainder in the radial lower bound, before taking square roots. -/
theorem radial_remainder (a epsilon r speed w : ℝ)
    (hden : r ^ 2 + epsilon ^ 2 ≠ 0) :
    graphSpeedSq a epsilon r speed w - radialSpeedSq a epsilon r w =
      (speed ^ 2 - w ^ 2) / (r ^ 2 + epsilon ^ 2) := by
  unfold graphSpeedSq radialSpeedSq
  field_simp
  ring

/-- The curvewise radial density is a lower bound for graph speed squared
whenever the distance function's derivative is bounded by Euclidean speed. -/
theorem radial_lower_bound (a epsilon r speed w : ℝ)
    (h_epsilon : 0 < epsilon) (h_radial : w ^ 2 ≤ speed ^ 2) :
    radialSpeedSq a epsilon r w ≤ graphSpeedSq a epsilon r speed w := by
  have h_den : 0 < r ^ 2 + epsilon ^ 2 :=
    add_pos_of_nonneg_of_pos (sq_nonneg r) (sq_pos_of_pos h_epsilon)
  have h_nonneg : 0 ≤ (speed ^ 2 - w ^ 2) / (r ^ 2 + epsilon ^ 2) :=
    div_nonneg (sub_nonneg.mpr h_radial) h_den.le
  linarith [radial_remainder a epsilon r speed w h_den.ne']

/-- Pointwise comparison with the scale-only metric. Taking square roots and
integrating these inequalities is a separate analytic step. -/
theorem graph_scale_comparison (a epsilon r speed w : ℝ)
    (h_epsilon : 0 < epsilon) (h_radial : w ^ 2 ≤ speed ^ 2) :
    speed ^ 2 / (r ^ 2 + epsilon ^ 2) ≤ graphSpeedSq a epsilon r speed w ∧
      graphSpeedSq a epsilon r speed w ≤
        (1 + a ^ 2) * (speed ^ 2 / (r ^ 2 + epsilon ^ 2)) := by
  unfold graphSpeedSq
  constructor
  case right =>
    have h_den : 0 < r ^ 2 + epsilon ^ 2 :=
      add_pos_of_nonneg_of_pos (sq_nonneg r) (sq_pos_of_pos h_epsilon)
    field_simp
    nlinarith [mul_nonneg (sq_nonneg a)
      (mul_nonneg (sq_nonneg r) (sub_nonneg.mpr h_radial)),
      mul_nonneg (sq_nonneg a) (mul_nonneg (sq_nonneg epsilon) (sq_nonneg speed))]
  exact le_add_of_nonneg_right (div_nonneg
    (mul_nonneg (mul_nonneg (sq_nonneg a) (sq_nonneg r)) (sq_nonneg w))
    (sq_nonneg (r ^ 2 + epsilon ^ 2)))

/-- The normalized denominator factor gives both sides of the pointwise
regularization comparison used in the conventional zero-floor limit proof. -/
theorem regularization_sandwich (t A C : ℝ)
    (h_t0 : 0 ≤ t) (h_t1 : t ≤ 1) (h_A : 0 ≤ A) (h_C : 0 ≤ C) :
    t ^ 2 * (A + C) ≤ t * A + t ^ 2 * C ∧
      t * A + t ^ 2 * C ≤ A + C := by
  constructor
  case right =>
    nlinarith [mul_nonneg h_A (sub_nonneg.mpr h_t1),
      mul_nonneg h_C (mul_nonneg (sub_nonneg.mpr h_t1) (add_nonneg zero_le_one h_t0))]
  nlinarith [mul_nonneg h_A (mul_nonneg h_t0 (sub_nonneg.mpr h_t1))]

end FractalSync.DistanceGraph
