import FractalSync.OrbitTrapConfidence
import Mathlib.Algebra.BigOperators.Intervals

/-!
# Finite-step drift of the fixed second-iterate trap

The step `d` is the actual displacement in the Euclidean coordinate pair.
The certificate bounds the change in the squared trap value in exact arithmetic.
-/

namespace FractalSync.OrbitTrapDrift

open OrbitTrapSteering OrbitTrapConfidence
open scoped BigOperators

noncomputable section

private def orbitValue (c : Point) : Point := (orbitRe c, orbitIm c)

private def orbitSlope (c : Point) : Point := (2 * c.1 + 1, 2 * c.2)

private def complexProduct (u v : Point) : Point :=
  (u.1 * v.1 - u.2 * v.2, u.1 * v.2 + u.2 * v.1)

private theorem pairing_nonneg (u : Point) : 0 ≤ pairing u u := by
  unfold pairing
  nlinarith [sq_nonneg u.1, sq_nonneg u.2]

private theorem norm_nonneg (u : Point) : 0 ≤ euclideanNorm u := Real.sqrt_nonneg _

private theorem norm_sq (u : Point) : euclideanNorm u ^ 2 = pairing u u :=
  Real.sq_sqrt (pairing_nonneg u)

private theorem abs_pairing_le (u v : Point) :
    |pairing u v| ≤ euclideanNorm u * euclideanNorm v := by
  have h := dot_det_sq u v
  rw [← norm_sq u, ← norm_sq v, ← mul_pow, ← sq_abs (pairing u v)] at h
  have hs : |pairing u v| ^ 2 ≤ (euclideanNorm u * euclideanNorm v) ^ 2 := by
    linarith [sq_nonneg (determinant u v)]
  exact (sq_le_sq₀ (abs_nonneg _) (mul_nonneg (norm_nonneg u) (norm_nonneg v))).mp hs

private theorem norm_add_le (u v : Point) :
    euclideanNorm (u + v) ≤ euclideanNorm u + euclideanNorm v := by
  have h : pairing (u + v) (u + v) = pairing u u + 2 * pairing u v + pairing v v := by
    simp only [pairing, Prod.fst_add, Prod.snd_add]
    ring
  have hs : euclideanNorm (u + v) ^ 2 ≤ (euclideanNorm u + euclideanNorm v) ^ 2 := by
    rw [norm_sq, h, ← norm_sq u, ← norm_sq v]
    nlinarith only [(le_abs_self (pairing u v)).trans (abs_pairing_le u v)]
  exact (sq_le_sq₀ (norm_nonneg _) (add_nonneg (norm_nonneg u) (norm_nonneg v))).mp hs

private theorem norm_step_square (d : Point) :
    euclideanNorm (complexProduct d d) = euclideanNorm d ^ 2 := by
  have hs : euclideanNorm (complexProduct d d) ^ 2 = (euclideanNorm d ^ 2) ^ 2 := by
    rw [norm_sq, norm_sq]
    unfold pairing complexProduct
    dsimp only
    ring
  exact (sq_eq_sq₀ (norm_nonneg _) (sq_nonneg _)).mp hs

/-- The exact quartic increment, expressed as the linear term and two
contributions of order two or higher in the displacement. -/
theorem trap_step_increment (c d : Point) :
    trap (c + d) - trap c = pairing (trapGradient c) d +
      (2 * pairing (orbitValue c) (complexProduct d d) +
        pairing d d * pairing (orbitSlope c + d) (orbitSlope c + d)) := by
  simp only [trap, trapGradient, orbitValue, orbitSlope, complexProduct, pairing,
    orbitRe, orbitIm, Prod.fst_add, Prod.snd_add]
  ring

/-- The remainder coefficient uses only the start and the displacement norm. -/
def stepRemainderCoefficient (c : Point) (s : ℝ) : ℝ :=
  2 * Real.sqrt (trap c) + (euclideanNorm (orbitSlope c) + s) ^ 2

theorem trap_step_remainder_bound (c d : Point) :
    |trap (c + d) - trap c - pairing (trapGradient c) d| ≤
      stepRemainderCoefficient c (euclideanNorm d) * euclideanNorm d ^ 2 := by
  have hn : euclideanNorm (orbitValue c) = Real.sqrt (trap c) := by
    simp only [euclideanNorm, orbitValue, trap, pairing]
    ring_nf
  have hp := abs_pairing_le (orbitValue c) (complexProduct d d)
  rw [hn, norm_step_square] at hp
  have hsum := pow_le_pow_left₀ (norm_nonneg (orbitSlope c + d))
    (norm_add_le (orbitSlope c) d) 2
  have hremainder : trap (c + d) - trap c - pairing (trapGradient c) d =
      2 * pairing (orbitValue c) (complexProduct d d) +
        pairing d d * pairing (orbitSlope c + d) (orbitSlope c + d) := by
    rw [trap_step_increment]
    ring
  rw [hremainder]
  calc
    _ ≤ |2 * pairing (orbitValue c) (complexProduct d d)| +
        |pairing d d * pairing (orbitSlope c + d) (orbitSlope c + d)| := abs_add_le _ _
    _ ≤ 2 * (Real.sqrt (trap c) * euclideanNorm d ^ 2) +
        euclideanNorm d ^ 2 * (euclideanNorm (orbitSlope c) + euclideanNorm d) ^ 2 := by
      rw [abs_mul, abs_of_nonneg (by norm_num : 0 ≤ (2 : ℝ)),
        abs_of_nonneg (mul_nonneg (pairing_nonneg d) (pairing_nonneg _)),
        ← norm_sq d, ← norm_sq (orbitSlope c + d)]
      exact add_le_add (mul_le_mul_of_nonneg_left hp (by norm_num))
        (mul_le_mul_of_nonneg_left hsum (sq_nonneg _))
    _ = _ := by unfold stepRemainderCoefficient; ring

/-- The step budget includes a perpendicularity defect for an arbitrary displacement. -/
theorem estimated_step_bound (c d estimate : Point) (E : ℝ)
    (herr : errorSq estimate (trapGradient c) ≤ E) :
    |trap (c + d) - trap c| ≤ |pairing estimate d| + Real.sqrt E * euclideanNorm d +
      stepRemainderCoefficient c (euclideanNorm d) * euclideanNorm d ^ 2 := by
  let g := trapGradient c
  have hnorm : euclideanNorm (g - estimate) = Real.sqrt (errorSq estimate g) := by
    simp only [euclideanNorm, pairing, errorSq, Prod.fst_sub, Prod.snd_sub]
    ring_nf
  have hdot : pairing g d = pairing estimate d + pairing (g - estimate) d := by
    simp only [pairing, Prod.fst_sub, Prod.snd_sub]
    ring
  have herror := abs_pairing_le (g - estimate) d
  rw [hnorm] at herror
  have hlinear : |pairing g d| ≤ |pairing estimate d| + Real.sqrt E * euclideanNorm d := by
    rw [hdot]
    exact (abs_add_le _ _).trans (add_le_add_left
      (herror.trans (mul_le_mul_of_nonneg_right (Real.sqrt_le_sqrt herr) (norm_nonneg d))) _)
  have hstep : trap (c + d) - trap c =
      pairing g d + (trap (c + d) - trap c - pairing g d) := by ring
  calc
    _ ≤ |pairing g d| + |trap (c + d) - trap c - pairing g d| := by
      conv_lhs => rw [hstep]
      exact abs_add_le _ _
    _ ≤ _ := add_le_add hlinear (trap_step_remainder_bound c d)

/-- For a tangent displacement, the gradient error budget contributes one
power of step length and the remainder contributes two. -/
theorem estimated_tangent_step_bound (c d estimate : Point) (E : ℝ)
    (herr : errorSq estimate (trapGradient c) ≤ E) (htangent : pairing estimate d = 0) :
    |trap (c + d) - trap c| ≤ Real.sqrt E * euclideanNorm d +
      stepRemainderCoefficient c (euclideanNorm d) * euclideanNorm d ^ 2 := by
  simpa only [htangent, abs_zero, zero_add] using estimated_step_bound c d estimate E herr

/-- Per-step budgets telescope to an endpoint budget, including for every prefix. -/
theorem accumulated_trap_drift (path : ℕ → Point) (budget : ℕ → ℝ) (n : ℕ)
    (hstep : ∀ i < n, |trap (path (i + 1)) - trap (path i)| ≤ budget i) :
    |trap (path n) - trap (path 0)| ≤ ∑ i ∈ Finset.range n, budget i := by
  induction n with
  | zero => simp
  | succ n ih =>
    rw [Finset.sum_range_succ]
    have hsplit : trap (path (n + 1)) - trap (path 0) =
        (trap (path n) - trap (path 0)) + (trap (path (n + 1)) - trap (path n)) := by ring
    rw [hsplit]
    exact (abs_add_le _ _).trans
      (add_le_add (ih (fun i hi => hstep i (Nat.lt_succ_of_lt hi))) (hstep n (Nat.lt_succ_self n)))

end

end FractalSync.OrbitTrapDrift
