import FractalSync.OperationalFieldBounds

/-!
# Concrete coefficient-to-jet error bounds for the operational field

The degree-four and degree-three Bernstein control tables below are exact
identities for the actual first and second derivatives of the six runtime
quintic weights. Their column sums give explicit coefficient perturbation
bounds. These are real-arithmetic sensitivity constants, not IEEE budgets.
-/

open scoped BigOperators

noncomputable section

namespace FractalSync.OperationalField

def bernsteinOfDegree (n : ℕ) (t : ℝ) (k : Fin (n+1)) : ℝ :=
  (Nat.choose n k.val : ℝ) * t^k.val * (1-t)^(n-k.val)

theorem bernsteinOfDegree_partition (n : ℕ) (t : ℝ) :
    ∑ k : Fin (n+1), bernsteinOfDegree n t k = 1 := by
  simpa [bernsteinOfDegree, Finset.sum_range, mul_comm, mul_left_comm, mul_assoc] using
    (add_pow t (1-t) n).symm

theorem bernsteinOfDegree_nonneg (n : ℕ) (t : ℝ)
    (ht : t ∈ Set.Icc (0 : ℝ) 1) (k : Fin (n+1)) : 0 ≤ bernsteinOfDegree n t k := by
  exact mul_nonneg (mul_nonneg (Nat.cast_nonneg _) (pow_nonneg ht.1 _))
    (pow_nonneg (sub_nonneg.mpr ht.2) _)

def firstControls : Fin 6 → Fin 5 → ℤ :=
  ![![-5, 0, 0, 0, 0], ![-50, -40, -20, -10, -5],
    ![0, -30, -60, -60, -50], ![50, 60, 60, 30, 0],
    ![5, 10, 20, 40, 50], ![0, 0, 0, 0, 5]]

def secondControls : Fin 6 → Fin 4 → ℤ :=
  ![![20, 0, 0, 0], ![40, 80, 40, 20], ![-120, -120, 0, 40],
    ![40, 0, -120, -120], ![20, 40, 80, 40], ![0, 0, 0, 20]]

theorem basisFirst_eq_bernstein (t : ℝ) (i : Fin 6) :
    basisFirst t i = (∑ k : Fin 5, (firstControls i k : ℝ) * bernsteinOfDegree 4 t k) / 120 := by
  fin_cases i
  all_goals norm_num [basisFirst, coefficients, hornerJet, firstControls,
    bernsteinOfDegree, Fin.sum_univ_succ, Nat.choose]
  all_goals ring

theorem basisSecond_eq_bernstein (t : ℝ) (i : Fin 6) :
    basisSecond t i = (∑ k : Fin 4, (secondControls i k : ℝ) * bernsteinOfDegree 3 t k) / 120 := by
  fin_cases i
  all_goals norm_num [basisSecond, coefficients, hornerJet, secondControls,
    bernsteinOfDegree, Fin.sum_univ_succ, Nat.choose]
  all_goals ring

theorem firstControls_column_bound (k : Fin 5) :
    (∑ i : Fin 6, |(firstControls i k : ℝ)|) ≤ 160 := by
  fin_cases k <;> norm_num [firstControls, Fin.sum_univ_succ]

theorem secondControls_column_bound (k : Fin 4) :
    (∑ i : Fin 6, |(secondControls i k : ℝ)|) ≤ 240 := by
  fin_cases k <;> norm_num [secondControls, Fin.sum_univ_succ]

theorem sum_abs_weighted_columns_bound {ι κ : Type*} [Fintype ι] [Fintype κ]
    (C : ι → κ → ℝ) (w : κ → ℝ) (M : ℝ)
    (hw : ∀ k, 0 ≤ w k) (hws : ∑ k, w k = 1)
    (hC : ∀ k, ∑ i, |C i k| ≤ M) :
    ∑ i, |∑ k, C i k * w k| ≤ M := by
  calc
    _ ≤ ∑ i, ∑ k, |C i k * w k| :=
      Finset.sum_le_sum (fun i _ => Finset.abs_sum_le_sum_abs _ _)
    _ = ∑ k, (∑ i, |C i k|) * w k := by
      simp_rw [abs_mul, abs_of_nonneg (hw _)]
      rw [Finset.sum_comm]
      simp only [Finset.sum_mul]
    _ ≤ ∑ k, M * w k :=
      Finset.sum_le_sum (fun k _ => mul_le_mul_of_nonneg_right (hC k) (hw k))
    _ = M := by rw [← Finset.mul_sum, hws, mul_one]

theorem basisFirst_abs_sum_le (t : ℝ) (ht : t ∈ Set.Icc (0 : ℝ) 1) :
    (∑ i : Fin 6, |basisFirst t i|) ≤ 4/3 := by
  calc
    _ = (∑ i : Fin 6, |∑ k : Fin 5,
        (firstControls i k : ℝ) * bernsteinOfDegree 4 t k|) / 120 := by
      simp only [basisFirst_eq_bernstein, abs_div, abs_of_pos (show (0 : ℝ) < 120 by norm_num)]
      rw [Finset.sum_div]
    _ ≤ 160/120 := div_le_div_of_nonneg_right
      (sum_abs_weighted_columns_bound (fun i k => (firstControls i k : ℝ))
        (bernsteinOfDegree 4 t) 160 (bernsteinOfDegree_nonneg 4 t ht)
        (bernsteinOfDegree_partition 4 t) firstControls_column_bound) (by norm_num)
    _ = 4/3 := by norm_num

theorem basisSecond_abs_sum_le (t : ℝ) (ht : t ∈ Set.Icc (0 : ℝ) 1) :
    (∑ i : Fin 6, |basisSecond t i|) ≤ 2 := by
  calc
    _ = (∑ i : Fin 6, |∑ k : Fin 4,
        (secondControls i k : ℝ) * bernsteinOfDegree 3 t k|) / 120 := by
      simp only [basisSecond_eq_bernstein, abs_div,
        abs_of_pos (show (0 : ℝ) < 120 by norm_num)]
      rw [Finset.sum_div]
    _ ≤ 240/120 := div_le_div_of_nonneg_right
      (sum_abs_weighted_columns_bound (fun i k => (secondControls i k : ℝ))
        (bernsteinOfDegree 3 t) 240 (bernsteinOfDegree_nonneg 3 t ht)
        (bernsteinOfDegree_partition 3 t) secondControls_column_bound) (by norm_num)
    _ = 2 := by norm_num

theorem tensorPatch_abs_bound (C : Coefficients) (f g : ℝ → Fin 6 → ℝ)
    (q : Coords) (eta F G : ℝ) (heta : 0 ≤ eta)
    (hC : ∀ i j, |C i j| ≤ eta)
    (hF : (∑ i : Fin 6, |f q.1 i|) ≤ F)
    (hG : (∑ j : Fin 6, |g q.2 j|) ≤ G) :
    |tensorPatch C f g q| ≤ eta * F * G := by
  have hF0 : 0 ≤ F := (Finset.sum_nonneg (fun i _ => abs_nonneg (f q.1 i))).trans hF
  calc
    _ ≤ ∑ j : Fin 6, |∑ i : Fin 6, C i j * f q.1 i * g q.2 j| :=
      Finset.abs_sum_le_sum_abs _ _
    _ ≤ ∑ j : Fin 6, ∑ i : Fin 6, |C i j * f q.1 i * g q.2 j| :=
      Finset.sum_le_sum (fun j _ => Finset.abs_sum_le_sum_abs _ _)
    _ ≤ ∑ j : Fin 6, ∑ i : Fin 6, eta * |f q.1 i| * |g q.2 j| := by
      refine Finset.sum_le_sum (fun j _ => Finset.sum_le_sum (fun i _ => ?_))
      rw [abs_mul, abs_mul]
      exact mul_le_mul_of_nonneg_right
        (mul_le_mul_of_nonneg_right (hC i j) (abs_nonneg _)) (abs_nonneg _)
    _ = eta * (∑ i : Fin 6, |f q.1 i|) * (∑ j : Fin 6, |g q.2 j|) := by
      simp [← Finset.sum_mul, ← Finset.mul_sum]
    _ ≤ eta * F * (∑ j : Fin 6, |g q.2 j|) :=
      mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hF heta)
        (Finset.sum_nonneg (fun j _ => abs_nonneg _))
    _ ≤ eta * F * G := mul_le_mul_of_nonneg_left hG (mul_nonneg heta hF0)


theorem basis_abs_sum (t : ℝ) (ht : t ∈ Set.Icc (0 : ℝ) 1) :
    (∑ i : Fin 6, |basis t i|) = 1 := by
  simp_rw [abs_of_nonneg (basis_nonneg t ht _)]
  exact basis_partition t

theorem tensorPatch_sub (C D : Coefficients) (f g : ℝ → Fin 6 → ℝ) (q : Coords) :
    tensorPatch (fun i j => C i j - D i j) f g q =
      tensorPatch C f g q - tensorPatch D f g q := by
  simp [tensorPatch, sub_mul, Finset.sum_sub_distrib]

theorem tensorPatch_coefficient_error_bound (C D : Coefficients)
    (f g : ℝ → Fin 6 → ℝ) (q : Coords) (eta F G : ℝ) (heta : 0 ≤ eta)
    (hC : ∀ i j, |C i j - D i j| ≤ eta)
    (hF : (∑ i : Fin 6, |f q.1 i|) ≤ F)
    (hG : (∑ j : Fin 6, |g q.2 j|) ≤ G) :
    |tensorPatch C f g q - tensorPatch D f g q| ≤ eta * F * G := by
  rw [← tensorPatch_sub]
  exact tensorPatch_abs_bound _ f g q eta F G heta hC hF hG

theorem tensorPatch_scaled_coefficient_error_bound (C D : Coefficients)
    (f g : ℝ → Fin 6 → ℝ) (q : Coords) (eta F G s : ℝ) (heta : 0 ≤ eta)
    (hC : ∀ i j, |C i j - D i j| ≤ eta)
    (hF : (∑ i : Fin 6, |f q.1 i|) ≤ F)
    (hG : (∑ j : Fin 6, |g q.2 j|) ≤ G) (hs : 0 ≤ s) :
    |tensorPatch C f g q / s - tensorPatch D f g q / s| ≤ eta * F * G / s := by
  rw [← sub_div, abs_div, abs_of_nonneg hs]
  exact div_le_div_of_nonneg_right
    (tensorPatch_coefficient_error_bound C D f g q eta F G heta hC hF hG) hs


/-- Each actual coordinate derivative is stable with explicit spacing dependence. -/
theorem fieldDifferential_coefficient_error_bound (C D : Coefficients)
    (o q : Coords) (h eta : ℝ) (hh : 0 < h) (heta : 0 ≤ eta)
    (hx : (q.1 - o.1) / h ∈ Set.Icc (0 : ℝ) 1)
    (hy : (q.2 - o.2) / h ∈ Set.Icc (0 : ℝ) 1)
    (hC : ∀ i j, |C i j - D i j| ≤ eta) :
    |fieldDifferential C o h q (1, 0) - fieldDifferential D o h q (1, 0)| ≤
        (4/3 : ℝ) * eta / h ∧
      |fieldDifferential C o h q (0, 1) - fieldDifferential D o h q (0, 1)| ≤
        (4/3 : ℝ) * eta / h := by
  constructor
  · simpa [fieldDifferential_eq_query, linearForm, mul_comm, mul_left_comm, mul_assoc] using
      tensorPatch_scaled_coefficient_error_bound C D basisFirst basis
        (normalizedCoordinates o h q) eta (4/3) 1 h heta hC
        (basisFirst_abs_sum_le _ hx) (le_of_eq (basis_abs_sum _ hy)) hh.le
  · simpa [fieldDifferential_eq_query, linearForm, mul_comm, mul_left_comm, mul_assoc] using
      tensorPatch_scaled_coefficient_error_bound C D basis basisFirst
        (normalizedCoordinates o h q) eta 1 (4/3) h heta hC
        (le_of_eq (basis_abs_sum _ hx)) (basisFirst_abs_sum_le _ hy) hh.le


/-- The actual pure and mixed Hessian components have certified sensitivity constants. -/
theorem fieldHessian_coefficient_error_bound (C D : Coefficients)
    (o q : Coords) (h eta : ℝ) (hh : 0 < h) (heta : 0 ≤ eta)
    (hx : (q.1 - o.1) / h ∈ Set.Icc (0 : ℝ) 1)
    (hy : (q.2 - o.2) / h ∈ Set.Icc (0 : ℝ) 1)
    (hC : ∀ i j, |C i j - D i j| ≤ eta) :
    |fieldHessian C o h q (1, 0) (1, 0) - fieldHessian D o h q (1, 0) (1, 0)| ≤
        2 * eta / h^2 ∧
      |fieldHessian C o h q (1, 0) (0, 1) - fieldHessian D o h q (1, 0) (0, 1)| ≤
        (16/9 : ℝ) * eta / h^2 ∧
      |fieldHessian C o h q (0, 1) (0, 1) - fieldHessian D o h q (0, 1) (0, 1)| ≤
        2 * eta / h^2 := by
  constructor
  · simpa [fieldHessian_eq_query, quadraticForm, linearForm, mul_comm, mul_left_comm, mul_assoc] using
      tensorPatch_scaled_coefficient_error_bound C D basisSecond basis
        (normalizedCoordinates o h q) eta 2 1 (h^2) heta hC
        (basisSecond_abs_sum_le _ hx) (le_of_eq (basis_abs_sum _ hy)) (pow_pos hh 2).le
  constructor
  · have hbound := tensorPatch_scaled_coefficient_error_bound C D basisFirst basisFirst
      (normalizedCoordinates o h q) eta (4/3) (4/3) (h^2) heta hC
      (basisFirst_abs_sum_le _ hx) (basisFirst_abs_sum_le _ hy) (pow_pos hh 2).le
    rw [show eta * (4/3) * (4/3) = (16/9 : ℝ) * eta by ring] at hbound
    simpa [fieldHessian_eq_query, quadraticForm, linearForm] using hbound
  · simpa [fieldHessian_eq_query, quadraticForm, linearForm, mul_comm, mul_left_comm, mul_assoc] using
      tensorPatch_scaled_coefficient_error_bound C D basis basisSecond
        (normalizedCoordinates o h q) eta 1 2 (h^2) heta hC
        (le_of_eq (basis_abs_sum _ hx)) (basisSecond_abs_sum_le _ hy) (pow_pos hh 2).le

end FractalSync.OperationalField
