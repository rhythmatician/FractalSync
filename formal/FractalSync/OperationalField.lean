import FractalSync.ProviderCertificates
import Mathlib.Data.Fin.VecNotation

/-!
# Exact-real jets of the operational quintic B-spline

The coefficient lists and simultaneous Horner recurrence are those in
`runtime-core/src/operational_field.rs`. These theorems identify the actual
real derivatives of a fixed cell's polynomial. They do not identify Rust
floating operations with real arithmetic or the operational field with Shore
signed distance.
-/

open scoped BigOperators

noncomputable section

namespace FractalSync.OperationalField

/-- The three simultaneous Horner accumulators, coefficients in ascending order. -/
def hornerJet : List ℝ → ℝ → ℝ × ℝ × ℝ
  | [], _ => (0, 0, 0)
  | a :: p, t =>
    let z := hornerJet p t
    (a + t * z.1, z.1 + t * z.2.1, 2 * z.2.1 + t * z.2.2)

def coefficients : Fin 6 → List ℝ :=
  ![[1, -5, 10, -10, 5, -1],
    [26, -50, 20, 20, -20, 5],
    [66, 0, -60, 0, 30, -10],
    [26, 50, 20, -20, -20, 10],
    [1, 5, 10, 10, 5, -5],
    [0, 0, 0, 0, 0, 1]]

def basis (t : ℝ) (i : Fin 6) : ℝ := (hornerJet (coefficients i) t).1 / 120

def basisFirst (t : ℝ) (i : Fin 6) : ℝ := (hornerJet (coefficients i) t).2.1 / 120

def basisSecond (t : ℝ) (i : Fin 6) : ℝ := (hornerJet (coefficients i) t).2.2 / 120

theorem hasDerivAt_hornerJet (p : List ℝ) (t : ℝ) :
    HasDerivAt (fun x => (hornerJet p x).1) (hornerJet p t).2.1 t ∧
    HasDerivAt (fun x => (hornerJet p x).2.1) (hornerJet p t).2.2 t := by
  induction p with
  | nil => simp only [hornerJet]; exact ⟨hasDerivAt_const _ _, hasDerivAt_const _ _⟩
  | cons a p ih =>
    simpa [hornerJet, two_mul, add_assoc] using
      And.intro (((hasDerivAt_id t).mul ih.1).const_add a)
        (ih.1.add ((hasDerivAt_id t).mul ih.2))

theorem hasDerivAt_basis (t : ℝ) (i : Fin 6) :
    HasDerivAt (fun x => basis x i) (basisFirst t i) t ∧
    HasDerivAt (fun x => basisFirst x i) (basisSecond t i) t := by
  exact ⟨((hasDerivAt_hornerJet (coefficients i) t).1).div_const 120,
    ((hasDerivAt_hornerJet (coefficients i) t).2).div_const 120⟩

abbrev Coords := ℝ × ℝ
abbrev Coefficients := Fin 6 → Fin 6 → ℝ

def tensorPatch (C : Coefficients) (f g : ℝ → Fin 6 → ℝ) (q : Coords) : ℝ :=
  ∑ j, ∑ i, C i j * f q.1 i * g q.2 j

def linearForm (x y : ℝ) : Coords →L[ℝ] ℝ :=
  x • ContinuousLinearMap.fst ℝ ℝ ℝ + y • ContinuousLinearMap.snd ℝ ℝ ℝ

theorem hasFDerivAt_tensorPatch (C : Coefficients) (f f' g g' : ℝ → Fin 6 → ℝ)
    (q : Coords)
    (hf : ∀ i, HasDerivAt (fun x => f x i) (f' q.1 i) q.1)
    (hg : ∀ i, HasDerivAt (fun x => g x i) (g' q.2 i) q.2) :
    HasFDerivAt (tensorPatch C f g)
      (linearForm (tensorPatch C f' g q) (tensorPatch C f g' q)) q := by
  convert HasFDerivAt.sum (fun j (_ : j ∈ (Finset.univ : Finset (Fin 6))) =>
    HasFDerivAt.sum (fun i (_ : i ∈ (Finset.univ : Finset (Fin 6))) =>
      (((hf i).comp_hasFDerivAt q (hasFDerivAt_fst (𝕜 := ℝ) (p := q))).mul
        ((hg j).comp_hasFDerivAt q (hasFDerivAt_snd (𝕜 := ℝ) (p := q)))).const_mul (C i j))) using 1
  all_goals ext <;> simp [tensorPatch, linearForm, mul_comm, mul_left_comm]

def tensorDifferential (C : Coefficients) (f f' g g' : ℝ → Fin 6 → ℝ)
    (q : Coords) : Coords →L[ℝ] ℝ :=
  linearForm (tensorPatch C f' g q) (tensorPatch C f g' q)

def quadraticForm (xx xy yy : ℝ) : Coords →L[ℝ] (Coords →L[ℝ] ℝ) :=
  (linearForm xx xy).smulRight (ContinuousLinearMap.fst ℝ ℝ ℝ) +
  (linearForm xy yy).smulRight (ContinuousLinearMap.snd ℝ ℝ ℝ)

theorem hasFDerivAt_linearForm (f g : Coords → ℝ) (f' g' : Coords →L[ℝ] ℝ)
    (q : Coords) (hf : HasFDerivAt f f' q) (hg : HasFDerivAt g g' q) :
    HasFDerivAt (fun z => linearForm (f z) (g z))
      (f'.smulRight (ContinuousLinearMap.fst ℝ ℝ ℝ) +
        g'.smulRight (ContinuousLinearMap.snd ℝ ℝ ℝ)) q := by
  exact (hf.smul_const (ContinuousLinearMap.fst ℝ ℝ ℝ)).add
    (hg.smul_const (ContinuousLinearMap.snd ℝ ℝ ℝ))

theorem hasFDerivAt_tensorDifferential (C : Coefficients)
    (f f' f'' g g' g'' : ℝ → Fin 6 → ℝ) (q : Coords)
    (hf : ∀ i, HasDerivAt (fun x => f x i) (f' q.1 i) q.1)
    (hf' : ∀ i, HasDerivAt (fun x => f' x i) (f'' q.1 i) q.1)
    (hg : ∀ i, HasDerivAt (fun x => g x i) (g' q.2 i) q.2)
    (hg' : ∀ i, HasDerivAt (fun x => g' x i) (g'' q.2 i) q.2) :
    HasFDerivAt (tensorDifferential C f f' g g')
      (quadraticForm (tensorPatch C f'' g q) (tensorPatch C f' g' q)
        (tensorPatch C f g'' q)) q := by
  exact hasFDerivAt_linearForm _ _ _ _ q
    (hasFDerivAt_tensorPatch C f' f'' g g' q hf' hg)
    (hasFDerivAt_tensorPatch C f f' g' g'' q hf hg')

/-- Physical-coordinate weights; the origin is the chosen cell's lattice origin. -/
def weights (o h x : ℝ) (i : Fin 6) : ℝ := basis ((x - o) / h) i

def weightsFirst (o h x : ℝ) (i : Fin 6) : ℝ := basisFirst ((x - o) / h) i / h

def weightsSecond (o h x : ℝ) (i : Fin 6) : ℝ := basisSecond ((x - o) / h) i / h ^ 2

theorem hasDerivAt_weights (o h x : ℝ) (i : Fin 6) :
    HasDerivAt (fun z => weights o h z i) (weightsFirst o h x i) x := by
  simpa [weights, weightsFirst, Function.comp_def, div_eq_mul_inv] using
    (hasDerivAt_basis ((x - o) / h) i).1.comp x
      (((hasDerivAt_id x).sub_const o).div_const h)

theorem hasDerivAt_weightsFirst (o h x : ℝ) (i : Fin 6) :
    HasDerivAt (fun z => weightsFirst o h z i) (weightsSecond o h x i) x := by
  simpa [weightsFirst, weightsSecond, Function.comp_def, div_eq_mul_inv, pow_two,
    mul_assoc] using ((hasDerivAt_basis ((x - o) / h) i).2.comp x
      (((hasDerivAt_id x).sub_const o).div_const h)).div_const h

def field (C : Coefficients) (o : Coords) (h : ℝ) : Coords → ℝ :=
  tensorPatch C (weights o.1 h) (weights o.2 h)

def fieldDifferential (C : Coefficients) (o : Coords) (h : ℝ)
    (q : Coords) : Coords →L[ℝ] ℝ :=
  tensorDifferential C (weights o.1 h) (weightsFirst o.1 h)
    (weights o.2 h) (weightsFirst o.2 h) q

def fieldHessian (C : Coefficients) (o : Coords) (h : ℝ)
    (q : Coords) : Coords →L[ℝ] (Coords →L[ℝ] ℝ) :=
  quadraticForm
    (tensorPatch C (weightsSecond o.1 h) (weights o.2 h) q)
    (tensorPatch C (weightsFirst o.1 h) (weightsFirst o.2 h) q)
    (tensorPatch C (weights o.1 h) (weightsSecond o.2 h) q)

theorem hasFDerivAt_field (C : Coefficients) (o : Coords) (h : ℝ) (q : Coords) :
    HasFDerivAt (field C o h) (fieldDifferential C o h q) q := by
  exact hasFDerivAt_tensorPatch C (weights o.1 h) (weightsFirst o.1 h)
    (weights o.2 h) (weightsFirst o.2 h) q
    (hasDerivAt_weights o.1 h q.1) (hasDerivAt_weights o.2 h q.2)

theorem hasFDerivAt_fieldDifferential (C : Coefficients) (o : Coords)
    (h : ℝ) (q : Coords) :
    HasFDerivAt (fieldDifferential C o h) (fieldHessian C o h q) q := by
  exact hasFDerivAt_tensorDifferential C
    (weights o.1 h) (weightsFirst o.1 h) (weightsSecond o.1 h)
    (weights o.2 h) (weightsFirst o.2 h) (weightsSecond o.2 h) q
    (hasDerivAt_weights o.1 h q.1) (hasDerivAt_weightsFirst o.1 h q.1)
    (hasDerivAt_weights o.2 h q.2) (hasDerivAt_weightsFirst o.2 h q.2)

theorem fderiv_field (C : Coefficients) (o : Coords) (h : ℝ) (q : Coords) :
    fderiv ℝ (field C o h) q = fieldDifferential C o h q :=
  (hasFDerivAt_field C o h q).fderiv

theorem fderiv_fderiv_field (C : Coefficients) (o : Coords) (h : ℝ) (q : Coords) :
    fderiv ℝ (fderiv ℝ (field C o h)) q = fieldHessian C o h q := by
  simpa only [show fderiv ℝ (field C o h) = fieldDifferential C o h from
    funext (fun z => fderiv_field C o h z)] using
    (hasFDerivAt_fieldDifferential C o h q).fderiv

theorem basis_partition (t : ℝ) : ∑ i : Fin 6, basis t i = 1 := by
  norm_num [Fin.sum_univ_succ, basis, coefficients, hornerJet]
  ring

theorem basis_first_moment (t : ℝ) :
    ∑ i : Fin 6, ((i.val : ℝ) - 2) * basis t i = t := by
  norm_num [Fin.sum_univ_succ, basis, coefficients, hornerJet]
  ring

theorem basis_second_moment (t : ℝ) :
    ∑ i : Fin 6, ((i.val : ℝ) - 2)^2 * basis t i = t^2 + 1/2 := by
  norm_num [Fin.sum_univ_succ, basis, coefficients, hornerJet]
  ring

def bernsteinControls : Fin 6 → Fin 6 → ℕ :=
  ![![1, 0, 0, 0, 0, 0], ![26, 16, 8, 4, 2, 1],
    ![66, 66, 60, 48, 36, 26], ![26, 36, 48, 60, 66, 66],
    ![1, 2, 4, 8, 16, 26], ![0, 0, 0, 0, 0, 1]]

def bernstein (t : ℝ) (k : Fin 6) : ℝ :=
  (Nat.choose 5 k.val : ℝ) * t ^ k.val * (1 - t) ^ (5 - k.val)

theorem basis_eq_bernstein (t : ℝ) (i : Fin 6) :
    basis t i = (∑ k : Fin 6, (bernsteinControls i k : ℝ) * bernstein t k) / 120 := by
  fin_cases i
  all_goals norm_num [Fin.sum_univ_succ, basis, coefficients, hornerJet,
    bernsteinControls, bernstein, Matrix.cons_val_two, Matrix.cons_val_three,
    Matrix.cons_val_four]
  all_goals norm_num [Nat.choose]
  all_goals ring

theorem bernstein_nonneg (t : ℝ) (ht : t ∈ Set.Icc (0 : ℝ) 1) (k : Fin 6) :
    0 ≤ bernstein t k := by
  exact mul_nonneg (mul_nonneg (Nat.cast_nonneg _) (pow_nonneg ht.1 _))
    (pow_nonneg (sub_nonneg.mpr ht.2) _)

theorem basis_nonneg (t : ℝ) (ht : t ∈ Set.Icc (0 : ℝ) 1) (i : Fin 6) :
    0 ≤ basis t i := by
  rw [basis_eq_bernstein]
  exact div_nonneg (Finset.sum_nonneg (fun k _ =>
    mul_nonneg (Nat.cast_nonneg _) (bernstein_nonneg t ht k))) (by norm_num)

def node (o h : ℝ) (i : Fin 6) : ℝ := o + h * ((i.val : ℝ) - 2)

theorem weights_partition (o h x : ℝ) : ∑ i : Fin 6, weights o h x i = 1 :=
  basis_partition ((x - o) / h)

theorem weights_first_moment (o h x : ℝ) (hh : h ≠ 0) :
    ∑ i : Fin 6, node o h i * weights o h x i = x := by
  simp only [node, weights, add_mul, mul_assoc, Finset.sum_add_distrib,
    ← Finset.mul_sum, basis_partition, basis_first_moment, mul_one]
  field_simp
  ring

theorem tensorPatch_div (C : Coefficients) (f g : ℝ → Fin 6 → ℝ)
    (s t : ℝ) (q : Coords) :
    tensorPatch C (fun x i => f x i / s) (fun y j => g y j / t) q =
      tensorPatch C f g q / (s * t) := by
  simp only [tensorPatch, div_eq_mul_inv, mul_inv_rev, Finset.sum_mul]
  exact Finset.sum_congr rfl (fun j _ => Finset.sum_congr rfl (fun i _ => by ring))

def normalizedCoordinates (o : Coords) (h : ℝ) (q : Coords) : Coords :=
  ((q.1 - o.1) / h, (q.2 - o.2) / h)

theorem fieldDifferential_eq_query (C : Coefficients) (o : Coords)
    (h : ℝ) (q : Coords) :
    fieldDifferential C o h q =
      linearForm
        (tensorPatch C basisFirst basis (normalizedCoordinates o h q) / h)
        (tensorPatch C basis basisFirst (normalizedCoordinates o h q) / h) := by
  unfold fieldDifferential tensorDifferential
  congr 1
  · simpa [weightsFirst, weights, tensorPatch, normalizedCoordinates] using
      tensorPatch_div C (fun x => basisFirst ((x - o.1) / h))
        (fun y => basis ((y - o.2) / h)) h 1 q
  · simpa [weightsFirst, weights, tensorPatch, normalizedCoordinates] using
      tensorPatch_div C (fun x => basis ((x - o.1) / h))
        (fun y => basisFirst ((y - o.2) / h)) 1 h q

theorem fieldHessian_eq_query (C : Coefficients) (o : Coords)
    (h : ℝ) (q : Coords) :
    fieldHessian C o h q =
      quadraticForm
        (tensorPatch C basisSecond basis (normalizedCoordinates o h q) / h^2)
        (tensorPatch C basisFirst basisFirst (normalizedCoordinates o h q) / h^2)
        (tensorPatch C basis basisSecond (normalizedCoordinates o h q) / h^2) := by
  unfold fieldHessian
  congr 1
  · simpa [weightsSecond, weights, tensorPatch, normalizedCoordinates] using
      tensorPatch_div C (fun x => basisSecond ((x - o.1) / h))
        (fun y => basis ((y - o.2) / h)) (h^2) 1 q
  · simpa [weightsFirst, tensorPatch, normalizedCoordinates, pow_two] using
      tensorPatch_div C (fun x => basisFirst ((x - o.1) / h))
        (fun y => basisFirst ((y - o.2) / h)) h h q
  · simpa [weightsSecond, weights, tensorPatch, normalizedCoordinates] using
      tensorPatch_div C (fun x => basis ((x - o.1) / h))
        (fun y => basisSecond ((y - o.2) / h)) 1 (h^2) q

end FractalSync.OperationalField
