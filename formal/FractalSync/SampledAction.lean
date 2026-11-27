import Mathlib.Algebra.Order.BigOperators.Ring.Finset
import Mathlib.Data.Real.Basic
import Mathlib.Tactic.Ring

/-! A kinetic-action obstruction for a finite accepted history. -/

namespace FractalSync

open scoped BigOperators

/-- A lower length bound implies a lower sampled action bound once each
accepted interval's length is bounded by its duration times its sampled speed.
The speed belongs to the same kinetic action that appears in the conclusion. -/
theorem sampledAction_lower_bound {ι : Type*} (steps : Finset ι)
    (dt speed length : ι → ℝ) (bound : ℝ)
    (hdt : ∀ i ∈ steps, 0 ≤ dt i) (hT : 0 < ∑ i ∈ steps, dt i)
    (hbound : 0 ≤ bound) (hpath : bound ≤ ∑ i ∈ steps, length i)
    (hcovered : ∀ i ∈ steps, length i ≤ dt i * speed i) :
    bound ^ 2 / (2 * ∑ i ∈ steps, dt i) ≤ ∑ i ∈ steps, dt i * speed i ^ 2 / 2 := by
  have hl := hpath.trans (Finset.sum_le_sum hcovered)
  have hs := (sq_le_sq₀ hbound (hbound.trans hl)).2 hl
  have hcs := Finset.sum_sq_le_sum_mul_sum_of_sq_eq_mul steps hdt
    (fun i hi => mul_nonneg (hdt i hi) (sq_nonneg (speed i)))
    (fun i _ => show (dt i * speed i) ^ 2 = dt i * (dt i * speed i ^ 2) by ring)
  have hsum : (∑ i ∈ steps, dt i * speed i ^ 2 / 2) =
      (∑ i ∈ steps, dt i * speed i ^ 2) / 2 := by
    simp only [div_eq_mul_inv, Finset.sum_mul]
  rw [hsum]
  apply (div_le_iff₀ (mul_pos zero_lt_two hT)).2
  calc
    bound ^ 2 ≤ (∑ i ∈ steps, dt i) * ∑ i ∈ steps, dt i * speed i ^ 2 :=
      hs.trans hcs
    _ = ((∑ i ∈ steps, dt i * speed i ^ 2) / 2) * (2 * ∑ i ∈ steps, dt i) := by ring

end FractalSync
