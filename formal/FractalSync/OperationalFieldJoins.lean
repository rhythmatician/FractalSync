import FractalSync.OperationalFieldGluing

/-!
# Concrete quintic joins through four derivatives

The derivative recurrence extends the existing exact Horner jets. All endpoint
matching identities are calculated from the runtime's six coefficient lists.
-/

open scoped BigOperators

noncomputable section

namespace FractalSync.OperationalField

def hornerDerivative : List ℝ → ℕ → ℝ → ℝ
  | [], _, _ => 0
  | a :: p, 0, t => a + t * hornerDerivative p 0 t
  | _ :: p, n+1, t => (n+1 : ℕ) * hornerDerivative p n t + t * hornerDerivative p (n+1) t

theorem hasDerivAt_hornerDerivative (p : List ℝ) (n : ℕ) (t : ℝ) :
    HasDerivAt (hornerDerivative p n) (hornerDerivative p (n+1) t) t := by
  induction p generalizing n with
  | nil => exact hasDerivAt_const _ _
  | cons a p ih =>
    cases n with
    | zero => simpa [hornerDerivative] using (((hasDerivAt_id t).mul (ih 0)).const_add a)
    | succ n =>
      convert ((ih n).const_mul ((n : ℝ)+1)).add ((hasDerivAt_id t).mul (ih (n+1))) using 1
      all_goals simp [hornerDerivative, Nat.cast_add, add_mul, add_assoc, funext_iff, two_mul]


theorem hornerDerivative_low (p : List ℝ) (t : ℝ) :
    hornerDerivative p 0 t = (hornerJet p t).1 ∧
      hornerDerivative p 1 t = (hornerJet p t).2.1 ∧
      hornerDerivative p 2 t = (hornerJet p t).2.2 := by
  induction p with
  | nil => simp [hornerDerivative, hornerJet]
  | cons a p ih => simp [hornerDerivative, hornerJet, ih.1, ih.2.1, ih.2.2]

def basisJet (n : ℕ) (t : ℝ) (i : Fin 6) : ℝ :=
  hornerDerivative (coefficients i) n t / 120

theorem hasDerivAt_basisJet (n : ℕ) (t : ℝ) (i : Fin 6) :
    HasDerivAt (fun x => basisJet n x i) (basisJet (n+1) t i) t :=
  (hasDerivAt_hornerDerivative (coefficients i) n t).div_const 120

@[simp] theorem basisJet_zero : basisJet 0 = basis := by
  funext t i
  exact congrArg (fun x : ℝ => x/120) (hornerDerivative_low (coefficients i) t).1

@[simp] theorem basisJet_one : basisJet 1 = basisFirst := by
  funext t i
  exact congrArg (fun x : ℝ => x/120) (hornerDerivative_low (coefficients i) t).2.1

@[simp] theorem basisJet_two : basisJet 2 = basisSecond := by
  funext t i
  exact congrArg (fun x : ℝ => x/120) (hornerDerivative_low (coefficients i) t).2.2

/-- One fixed unit cell, with the coherent six samples from an integer lattice. -/
def latticeLineJet (n : ℕ) (C : ℤ → ℝ) (k : ℤ) (t : ℝ) : ℝ :=
  ∑ i : Fin 6, C (k + i.val - 2) * basisJet n t i

/-- Actual endpoint jets match after the one-cell coefficient shift. -/
theorem latticeLineJet_join (n : Fin 5) (C : ℤ → ℝ) (k : ℤ) :
    latticeLineJet n C k 1 = latticeLineJet n C (k+1) 0 := by
  fin_cases n
  all_goals norm_num [latticeLineJet, basisJet, coefficients, hornerDerivative, Fin.sum_univ_succ]
  all_goals ring_nf

end FractalSync.OperationalField
