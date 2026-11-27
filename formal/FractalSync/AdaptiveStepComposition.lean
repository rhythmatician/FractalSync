import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Analysis.Complex.Exponential
import Mathlib.Topology.MetricSpace.Pseudo.Defs

/-!
# Error composition on accepted heterogeneous steps

The recurrence starts after each accepted transition has been compared with its
reference on the same interval. Rejected trials do not enter the accepted history.
-/

open scoped BigOperators

noncomputable section

namespace FractalSync.AdaptiveStepComposition

/-- The budget adds each defect at its own step and amplifies it only on later steps. -/
def budget (gain defect : ℕ → ℝ) (initial : ℝ) : ℕ → ℝ
  | 0 => initial
  | n + 1 => gain n * budget gain defect initial n + defect n

/-- The prefix budget uses each accepted step's own gain and defect. -/
theorem error_le_budget (n : ℕ) (error gain defect : ℕ → ℝ) (initial : ℝ)
    (hinit : error 0 ≤ initial) (hgain : ∀ j < n, 0 ≤ gain j)
    (hstep : ∀ j < n, error (j + 1) ≤ gain j * error j + defect j) :
    error n ≤ budget gain defect initial n := by
  induction n with
  | zero => exact hinit
  | succ n ih =>
    exact (hstep n n.lt_succ_self).trans (add_le_add_right
      (mul_le_mul_of_nonneg_left
        (ih (fun j hj => hgain j hj.step) (fun j hj => hstep j hj.step))
        (hgain n n.lt_succ_self)) _)

/-- A common exponential envelope depends on elapsed time and the sum of the
actual local defects, allowing different step lengths throughout the history. -/
theorem error_le_exp_mul (n : ℕ) (error dt defect : ℕ → ℝ) (initial rate : ℝ)
    (hinit : error 0 ≤ initial) (hinitial : 0 ≤ initial) (hrate : 0 ≤ rate)
    (hdt : ∀ j < n, 0 ≤ dt j) (hdefect : ∀ j < n, 0 ≤ defect j)
    (hstep : ∀ j < n, error (j + 1) ≤ (1 + rate * dt j) * error j + defect j) :
    error n ≤ Real.exp (rate * ∑ j ∈ Finset.range n, dt j) *
      (initial + ∑ j ∈ Finset.range n, defect j) := by
  induction n with
  | zero => simpa using hinit
  | succ n ih =>
    exact
      let T := ∑ j ∈ Finset.range n, dt j
      let D := initial + ∑ j ∈ Finset.range n, defect j
      let x := rate * dt n
      have hn := n.lt_succ_self
      have hT : 0 ≤ T := Finset.sum_nonneg (fun j hj => hdt j (Finset.mem_range.mp hj).step)
      have hD : 0 ≤ D := add_nonneg hinitial
        (Finset.sum_nonneg (fun j hj => hdefect j (Finset.mem_range.mp hj).step))
      have hx : 0 ≤ x := mul_nonneg hrate (hdt n hn)
      have hE : 1 ≤ Real.exp (rate * (T + dt n)) :=
        Real.one_le_exp (mul_nonneg hrate (add_nonneg hT (hdt n hn)))
      calc
        error (n + 1) ≤ (1 + x) * error n + defect n := hstep n hn
        _ ≤ (1 + x) * (Real.exp (rate * T) * D) + defect n :=
          add_le_add_right (mul_le_mul_of_nonneg_left
            (ih (fun j hj => hdt j hj.step) (fun j hj => hdefect j hj.step)
              (fun j hj => hstep j hj.step)) (add_nonneg zero_le_one hx)) _
        _ ≤ Real.exp x * (Real.exp (rate * T) * D) + defect n :=
          add_le_add_right (mul_le_mul_of_nonneg_right
            (by simpa only [add_comm] using Real.add_one_le_exp x)
            (mul_nonneg (Real.exp_nonneg _) hD)) _
        _ = Real.exp (rate * (T + dt n)) * D + defect n := by
          have hexponent : x + rate * T = rate * (T + dt n) := by
            rw [mul_add, add_comm]
          rw [← mul_assoc, ← Real.exp_add, hexponent]
        _ ≤ Real.exp (rate * (T + dt n)) * (D + defect n) := by
          rw [mul_add (Real.exp (rate * (T + dt n))) D (defect n)]
          exact add_le_add_left (le_mul_of_one_le_left (hdefect n hn) hE) _
        _ = _ := by rw [Finset.sum_range_succ, Finset.sum_range_succ, ← add_assoc]

/-- An accepted numerical history and its reference share the same ordered time
intervals. Each local defect is the sum of a method bound, a duration-scaled
field residual, and an additive evaluation bound. Only the two states compared
on each interval need satisfy the reference transition's stability bound. -/
theorem acceptedHistory_dist_le {E : Type*} [PseudoMetricSpace E] (n : ℕ)
    (reference actual : ℕ → E) (advance : ℕ → E → E)
    (dt method residual evaluation : ℕ → ℝ) (initial rate : ℝ)
    (hinit : dist (reference 0) (actual 0) ≤ initial) (hinitial : 0 ≤ initial)
    (hrate : 0 ≤ rate) (hdt : ∀ j < n, 0 ≤ dt j)
    (hmethod : ∀ j < n, 0 ≤ method j) (hresidual : ∀ j < n, 0 ≤ residual j)
    (hevaluation : ∀ j < n, 0 ≤ evaluation j)
    (href : ∀ j < n, reference (j + 1) = advance j (reference j))
    (hstable : ∀ j < n, dist (advance j (reference j)) (advance j (actual j)) ≤
      (1 + rate * dt j) * dist (reference j) (actual j))
    (hlocal : ∀ j < n, dist (advance j (actual j)) (actual (j + 1)) ≤
      method j + dt j * residual j + evaluation j) :
    dist (reference n) (actual n) ≤ Real.exp (rate * ∑ j ∈ Finset.range n, dt j) *
      (initial + ∑ j ∈ Finset.range n, (method j + dt j * residual j + evaluation j)) := by
  refine error_le_exp_mul n (fun j => dist (reference j) (actual j)) dt
    (fun j => method j + dt j * residual j + evaluation j) initial rate
    hinit hinitial hrate hdt ?_ ?_
  · exact fun j hj => add_nonneg
      (add_nonneg (hmethod j hj) (mul_nonneg (hdt j hj) (hresidual j hj))) (hevaluation j hj)
  · exact fun j hj => by
      simpa only [href j hj] using
        (dist_triangle (advance j (reference j)) (advance j (actual j)) (actual (j + 1))).trans
          (add_le_add (hstable j hj) (hlocal j hj))

end FractalSync.AdaptiveStepComposition
