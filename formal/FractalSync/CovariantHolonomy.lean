import Mathlib.Data.Complex.Basic
import Mathlib.LinearAlgebra.Matrix.PosDef
import Mathlib.LinearAlgebra.Matrix.Rank
import Mathlib.Tactic.Ring

/-!
# Holonomy of finite transport kernels

A Hermitian matrix of unit complex transports with unit diagonal is positive
semidefinite exactly when the triangle products are trivial. This criterion
applies to the integer harmonic representation of the transports.
-/

open scoped ComplexConjugate ComplexOrder

namespace FractalSync.CovariantHolonomy

/-- The determinant of a triangle of unit transports measures its holonomy defect. -/
theorem triangleDeterminant {ι : Type*} (K : Matrix ι ι ℂ)
    (diagonal : ∀ i, K i i = 1) (reverse : ∀ i j, K j i = conj (K i j))
    (unit : ∀ i j, Complex.normSq (K i j) = 1) (a b c : ι) :
    let cycle := K a b * K b c * K c a
    (K.submatrix ![a, b, c] ![a, b, c]).det = -(Complex.normSq (cycle - 1) : ℂ) := by
  rw [Matrix.det_fin_three]
  let cycle := K a b * K b c * K c a
  change K a a * K b b * K c c - K a a * K b c * K c b
    - K a b * K b a * K c c + cycle + K a c * K b a * K c b
    - K a c * K b b * K c a = -(Complex.normSq (cycle - 1) : ℂ)
  have cycleUnit : Complex.normSq cycle = 1 := by simp [cycle, map_mul, unit]
  have unitPair (i j : ι) : K i j * K j i = 1 := by
    rw [reverse i j, mul_comm, ← Complex.normSq_eq_conj_mul_self, unit i j, Complex.ofReal_one]
  have cycleReverse : K a c * K b a * K c b = conj cycle := by
    rw [reverse c a, reverse a b, reverse b c]
    simp only [cycle, map_mul]
    ring
  simp only [diagonal, one_mul, mul_one]
  rw [unitPair b c,
    unitPair a b, unitPair a c, cycleReverse, Complex.normSq_sub, cycleUnit]
  simp only [map_one, mul_one, Complex.ofReal_sub, Complex.ofReal_add,
    Complex.ofReal_one, ← Complex.add_conj]
  ring

/-- A nontrivial triangle product contradicts positive semidefiniteness. -/
theorem triangleOfPosSemidef {ι : Type*} [Fintype ι] (K : Matrix ι ι ℂ)
    (diagonal : ∀ i, K i i = 1) (reverse : ∀ i j, K j i = conj (K i j))
    (unit : ∀ i j, Complex.normSq (K i j) = 1) (positive : K.PosSemidef)
    (a b c : ι) : K a b * K b c * K c a = 1 := by
  have determinant := (positive.submatrix ![a, b, c]).det_nonneg
  rw [triangleDeterminant K diagonal reverse unit a b c] at determinant
  have nonpositive := (Complex.nonneg_iff.mp determinant).1
  simp only [Complex.neg_re, Complex.ofReal_re, neg_nonneg] at nonpositive
  have zeroDefect := le_antisymm nonpositive (Complex.normSq_nonneg _)
  exact sub_eq_zero.mp (Complex.normSq_eq_zero.mp zeroDefect)

/-- Trivial triangle products recover the unit factors from any chosen row. -/
theorem referenceFactorization {ι : Type*} (K : Matrix ι ι ℂ)
    (reverse : ∀ i j, K j i = conj (K i j))
    (unit : ∀ i j, Complex.normSq (K i j) = 1)
    (triangle : ∀ a b c, K a b * K b c * K c a = 1)
    (a i j : ι) : K i j = conj (K a i) * K a j := by
  have unitFactor (k : ι) : conj (K a k) * K a k = 1 := by
    rw [← Complex.normSq_eq_conj_mul_self, unit a k, Complex.ofReal_one]
  calc
    K i j = (conj (K a i) * K a i) * K i j * (conj (K a j) * K a j) := by
      rw [unitFactor i, unitFactor j, one_mul, mul_one]
    _ = conj (K a i) * (K a i * K i j * K j a) * K a j := by
      rw [reverse a j]
      ring
    _ = conj (K a i) * K a j := by rw [triangle a i j, mul_one]

/-- A Hermitian unit transport kernel is PSD exactly when its triangles close. -/
theorem posSemidef_iff_triangle {ι : Type*} [Fintype ι] [Nonempty ι] (K : Matrix ι ι ℂ)
    (diagonal : ∀ i, K i i = 1) (reverse : ∀ i j, K j i = conj (K i j))
    (unit : ∀ i j, Complex.normSq (K i j) = 1) :
    K.PosSemidef ↔ ∀ a b c, K a b * K b c * K c a = 1 := by
  refine ⟨triangleOfPosSemidef K diagonal reverse unit, ?_⟩
  intro triangle
  obtain ⟨a⟩ := ‹Nonempty ι›
  have factorization : K = Matrix.vecMulVec (star (K a)) (K a) := by
    funext i j
    exact referenceFactorization K reverse unit triangle a i j
  rw [factorization]
  exact Matrix.posSemidef_vecMulVec_star_self (K a)

/-- A PSD unit transport kernel on a nonempty sample set has rank exactly one. -/
theorem rank_eq_one_of_posSemidef {ι : Type*} [Fintype ι] [Nonempty ι] (K : Matrix ι ι ℂ)
    (diagonal : ∀ i, K i i = 1) (reverse : ∀ i j, K j i = conj (K i j))
    (unit : ∀ i j, Complex.normSq (K i j) = 1) (positive : K.PosSemidef) : K.rank = 1 := by
  classical
  obtain ⟨a⟩ := ‹Nonempty ι›
  have upper : K.rank ≤ 1 := by
    have factorization : K = Matrix.vecMulVec (star (K a)) (K a) := by
      funext i j
      exact referenceFactorization K reverse unit
        (triangleOfPosSemidef K diagonal reverse unit positive) a i j
    rw [factorization]
    exact Matrix.rank_vecMulVec_le _ _
  refine le_antisymm upper ?_
  change 1 ≤ Module.finrank ℂ (LinearMap.range K.mulVecLin)
  rw [Submodule.one_le_finrank_iff]
  intro emptyRange
  have colInRange : K.col a ∈ LinearMap.range K.mulVecLin := by
    refine ⟨Pi.single a 1, ?_⟩
    rw [Matrix.mulVecLin_apply, Matrix.mulVec_single_one]
  rw [emptyRange, Submodule.mem_bot] at colInRange
  have entry := congrFun colInRange a
  simp [Matrix.col_apply, diagonal] at entry

/-- Holonomy must vanish in the chosen harmonic representation, for any integer order. -/
theorem harmonic_posSemidef_iff_triangle {ι : Type*} [Fintype ι] [Nonempty ι]
    (U : Matrix ι ι ℂ) (diagonal : ∀ i, U i i = 1)
    (reverse : ∀ i j, U j i = conj (U i j))
    (unit : ∀ i j, Complex.normSq (U i j) = 1) (m : ℤ) :
    (Matrix.of (fun i j => U i j ^ m)).PosSemidef ↔
      ∀ a b c, (U a b * U b c * U c a) ^ m = 1 := by
  let K := Matrix.of (fun i j => U i j ^ m)
  have harmonicDiagonal (i : ι) : K i i = 1 := by simp [K, diagonal]
  have harmonicReverse (i j : ι) : K j i = conj (K i j) := by
    simp only [K, Matrix.of_apply, reverse i j, map_zpow₀]
  have harmonicUnit (i j : ι) : Complex.normSq (K i j) = 1 := by
    simp only [K, Matrix.of_apply, map_zpow₀, unit, one_zpow]
  rw [posSemidef_iff_triangle K harmonicDiagonal harmonicReverse harmonicUnit]
  simp only [K, Matrix.of_apply, mul_zpow]

end FractalSync.CovariantHolonomy
