import FractalSync.OrbitTrapConfidence

/-!
# Normal acceleration of a prescribed planar path

For a Levi-Civita connection, the covariant acceleration at unit speed is
the metric-normal part of the phase acceleration divided by squared speed.
These statements check the pointwise linear algebra of that conversion.
-/

namespace FractalSync.TrapMetricComparison

open OrbitTrapSteering (Point)
open OrbitTrapConfidence (determinant)

def metric (a b d : ℝ) (v w : Point) : ℝ :=
  a * v.1 * w.1 + b * (v.1 * w.2 + v.2 * w.1) + d * v.2 * w.2

theorem gram (a b d : ℝ) (v w : Point) :
    metric a b d v v * metric a b d w w - metric a b d v w ^ 2 =
      (a * d - b ^ 2) * determinant v w ^ 2 := by
  unfold metric determinant
  ring

noncomputable def unitSpeedAcceleration (a b d : ℝ) (v w : Point) : Point :=
  let speedSq := metric a b d v v
  let projection := metric a b d v w / speedSq
  ((w.1 - projection * v.1) / speedSq, (w.2 - projection * v.2) / speedSq)

theorem unitSpeedAcceleration_orthogonal (a b d : ℝ) (v w : Point)
    (hv : metric a b d v v ≠ 0) :
    metric a b d v (unitSpeedAcceleration a b d v w) = 0 := by
  have scale_sub (q t : ℝ) :
      metric a b d v ((w.1 - q * v.1) * t, (w.2 - q * v.2) * t) =
        (metric a b d v w - q * metric a b d v v) * t := by
    unfold metric
    ring
  simp only [unitSpeedAcceleration, div_eq_mul_inv]
  rw [scale_sub]
  field_simp [hv]
  ring

theorem unitSpeedAcceleration_normSq (a b d : ℝ) (v w : Point)
    (hv : metric a b d v v ≠ 0) :
    metric a b d (unitSpeedAcceleration a b d v w) (unitSpeedAcceleration a b d v w) =
      (a * d - b ^ 2) * determinant v w ^ 2 / metric a b d v v ^ 3 := by
  have scale_sub_sq (q t : ℝ) :
      let x := ((w.1 - q * v.1) * t, (w.2 - q * v.2) * t)
      metric a b d x x =
        (metric a b d w w - 2 * q * metric a b d v w + q ^ 2 * metric a b d v v) * t ^ 2 := by
    dsimp only
    unfold metric
    ring
  rw [← gram]
  simp only [unitSpeedAcceleration, div_eq_mul_inv]
  rw [scale_sub_sq]
  field_simp [hv]
  ring

end FractalSync.TrapMetricComparison
