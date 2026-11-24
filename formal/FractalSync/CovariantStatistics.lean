import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Data.Complex.Basic
import Mathlib.Tactic.NormNum

/-!
# Algebraic contracts for covariant statistics (#163)

This file verifies the finite complex algebra of a common-reference transport
kernel and an exact counterexample to a different, inconsistent pairwise kernel.
It does not formalize a manifold connection, numerical transport, phase-harmonic
Lipschitz continuity, or audiovisual fit.
-/

open scoped BigOperators ComplexConjugate
open Complex

namespace FractalSync.CovariantStatistics

/-- A common-reference pairwise term is a product of transported coordinates. -/
theorem commonReferenceTerm (pi pj vi vj : ℂ) :
    conj vi * (conj pi * pj) * vj = conj (pi * vi) * (pj * vj) := by
  simp only [map_mul]
  ring

/-- A common-reference kernel has a Gram factorization for arbitrary finite data.
Unit-modulus hypotheses are unnecessary for this positivity identity. -/
theorem commonReferenceGram {ι : Type*} [Fintype ι] (p v : ι → ℂ) :
    (∑ i, ∑ j, conj (v i) * (conj (p i) * p j) * v j) =
      conj (∑ i, p i * v i) * (∑ j, p j * v j) := by
  simp_rw [commonReferenceTerm]
  simp only [map_sum, Finset.sum_mul_sum]

/-- The quadratic form of a common-reference kernel is nonnegative. -/
theorem commonReferenceNonnegative {ι : Type*} [Fintype ι] (p v : ι → ℂ) :
    0 ≤ (∑ i, ∑ j, conj (v i) * (conj (p i) * p j) * v j).re := by
  rw [commonReferenceGram, ← Complex.normSq_eq_conj_mul_self]
  exact Complex.normSq_nonneg _

/-- A coordinate gauge cancels against its matching transport gauge. -/
theorem transportGaugeCancel (g0 p g b : ℂ) (hg : conj g * g = 1) :
    (g0 * p * conj g) * (g * b) = g0 * (p * b) := by
  calc
    (g0 * p * conj g) * (g * b) = (g0 * (p * b)) * (conj g * g) := by ring
    _ = g0 * (p * b) := by rw [hg, mul_one]

/-- The same cancellation holds in every natural harmonic representation.
The complex value `b` is the already-computed harmonic coefficient. -/
theorem harmonicTransportGaugeCancel (g0 p g b : ℂ) (m : ℕ)
    (hg : conj g * g = 1) :
    (g0 * p * conj g) ^ m * (g ^ m * b) = g0 ^ m * (p ^ m * b) := by
  simpa only [mul_one, mul_pow, mul_assoc] using
    congrArg (fun z : ℂ => z ^ m * b) (transportGaugeCancel g0 p g 1 hg)

/-- A finite cross-moment of coefficients already expressed in a common fiber. -/
def moment {ι : Type*} [Fintype ι] (w : ι → ℝ) (a b : ι → ℂ) : ℂ :=
  ∑ i, (w i : ℂ) * a i * conj (b i)

/-- A common reference rotation changes the moment by its conjugate. -/
theorem momentGaugeCovariance {ι : Type*} [Fintype ι]
    (w : ι → ℝ) (a b : ι → ℂ) (g : ℂ) :
    moment w a (fun i => g * b i) = conj g * moment w a b := by
  simp [moment, map_mul, Finset.mul_sum, mul_comm, mul_left_comm, mul_assoc]

/-- A unit reference rotation leaves squared cross-moment magnitude unchanged. -/
theorem momentGaugeNormSq {ι : Type*} [Fintype ι]
    (w : ι → ℝ) (a b : ι → ℂ) (g : ℂ) (hg : Complex.normSq g = 1) :
    Complex.normSq (moment w a (fun i => g * b i)) =
      Complex.normSq (moment w a b) := by
  simp [momentGaugeCovariance, map_mul, Complex.normSq_conj, hg]

/-- Unit edge transports with half-turn holonomy around one triangle. -/
def triangleKernel (i j : Fin 3) : ℝ :=
  if (i = 0 ∧ j = 2) ∨ (i = 2 ∧ j = 0) then -1 else 1

/-- The exact negative-eigenvalue witness `(1, -1, 1)`. -/
def triangleWitness (i : Fin 3) : ℝ := if i = 1 then -1 else 1

/-- This is an exact arithmetic identity, not a sampled numerical eigenvalue. -/
theorem triangleNegative :
    (∑ i, ∑ j, triangleWitness i * triangleKernel i j * triangleWitness j) = -3 := by
  norm_num [triangleKernel, triangleWitness, Fin.sum_univ_three]
  norm_num [Fin.ext_iff]

/-- The counterexample has unit diagonal. -/
theorem triangleDiagonal : ∀ i : Fin 3, triangleKernel i i = 1 := by
  norm_num [Fin.forall_fin_succ, triangleKernel]
  norm_num [Fin.ext_iff]

/-- The counterexample is symmetric, hence Hermitian after embedding in `ℂ`. -/
theorem triangleSymmetric : ∀ i j : Fin 3, triangleKernel i j = triangleKernel j i := by
  simp [triangleKernel, and_comm, or_comm]

/-- Every entry has unit squared magnitude. -/
theorem triangleUnit : ∀ i j : Fin 3, triangleKernel i j ^ 2 = 1 := by
  norm_num [triangleKernel, ite_pow]

/-- Unit entries, symmetry, and unit diagonal do not imply positivity. -/
theorem triangleNotPositiveSemidefinite :
    ¬ ∀ v : Fin 3 → ℝ, 0 ≤ ∑ i, ∑ j, v i * triangleKernel i j * v j := by
  intro h
  have hv := h triangleWitness
  norm_num [triangleNegative] at hv

end FractalSync.CovariantStatistics
