import FractalSync.OperationalField

/-!
# Concrete operational-field approximation and stability

The nonnegative quintic weights are applied to actual coefficient arrays.
Polynomial reproduction and coefficient stability are proved for the real
field represented by those arrays. Rust rounding and fidelity to signed Shore
distance remain separate obligations.
-/

open scoped BigOperators

noncomputable section

namespace FractalSync.OperationalField

theorem weights_second_moment (o h x : ℝ) (hh : h ≠ 0) :
    ∑ i : Fin 6, (node o h i)^2 * weights o h x i = x^2 + h^2/2 := by
  simp only [node, weights, add_sq, mul_pow, add_mul, mul_assoc,
    Finset.sum_add_distrib, ← Finset.mul_sum, basis_partition,
    basis_first_moment, basis_second_moment, mul_one]
  field_simp
  ring

theorem weights_quadratic (o h x a b c : ℝ) (hh : h ≠ 0) :
    ∑ i : Fin 6, (a * (node o h i)^2 + b * node o h i + c) * weights o h x i =
      a * (x^2 + h^2/2) + b*x + c := by
  simp only [add_mul, mul_assoc, Finset.sum_add_distrib, ← Finset.mul_sum,
    weights_second_moment o h x hh, weights_first_moment o h x hh,
    weights_partition, mul_one]

/-- A general quadratic; the mixed coefficient is not doubled. -/
def quadratic (a b c d e f : ℝ) (q : Coords) : ℝ :=
  a*q.1^2 + b*q.1*q.2 + c*q.2^2 + d*q.1 + e*q.2 + f

def sampledCoefficients (f : Coords → ℝ) (o : Coords) (h : ℝ) : Coefficients :=
  fun i j => f (node o.1 h i, node o.2 h j)

/-- Raw quadratic samples produce the exact quadratic plus its known bias. -/
theorem field_quadratic (a b c d e f : ℝ) (o q : Coords) (h : ℝ) (hh : h ≠ 0) :
    field (sampledCoefficients (quadratic a b c d e f) o h) o h q =
      quadratic a b c d e f q + h^2/2 * (a+c) := by
  unfold field tensorPatch sampledCoefficients
  simp_rw [← Finset.sum_mul]
  have hrow (j : Fin 6) :
      (∑ i : Fin 6, quadratic a b c d e f (node o.1 h i, node o.2 h j) * weights o.1 h q.1 i) =
      a*(q.1^2+h^2/2) + (b*node o.2 h j+d)*q.1 +
        (c*(node o.2 h j)^2+e*node o.2 h j+f) := by
    convert weights_quadratic o.1 h q.1 a (b*node o.2 h j+d)
      (c*(node o.2 h j)^2+e*node o.2 h j+f) hh using 1
    refine Finset.sum_congr rfl (fun i _ => ?_)
    dsimp [quadratic]
    ring
  simp_rw [hrow]
  calc
    _ = ∑ j : Fin 6,
        (c*(node o.2 h j)^2 + (b*q.1+e)*node o.2 h j +
          (a*(q.1^2+h^2/2)+d*q.1+f)) * weights o.2 h q.2 j :=
      Finset.sum_congr rfl (fun j _ => by ring)
    _ = c*(q.2^2+h^2/2) + (b*q.1+e)*q.2 +
        (a*(q.1^2+h^2/2)+d*q.1+f) :=
      weights_quadratic o.2 h q.2 c (b*q.1+e) (a*(q.1^2+h^2/2)+d*q.1+f) hh
    _ = _ := by
      dsimp [quadratic]
      ring

theorem fieldDifferential_quadratic (a b c d e f : ℝ) (o q : Coords)
    (h : ℝ) (hh : h ≠ 0) :
    fieldDifferential (sampledCoefficients (quadratic a b c d e f) o h) o h q =
      fderiv ℝ (quadratic a b c d e f) q := by
  rw [← fderiv_field, show field (sampledCoefficients (quadratic a b c d e f) o h) o h =
    (fun z => quadratic a b c d e f z + h^2/2 * (a+c)) from
      funext (fun z => field_quadratic a b c d e f o z h hh), fderiv_add_const]

theorem fieldHessian_quadratic (a b c d e f : ℝ) (o q : Coords)
    (h : ℝ) (hh : h ≠ 0) :
    fieldHessian (sampledCoefficients (quadratic a b c d e f) o h) o h q =
      fderiv ℝ (fderiv ℝ (quadratic a b c d e f)) q := by
  rw [← (hasFDerivAt_fieldDifferential _ o h q).fderiv,
    show fieldDifferential (sampledCoefficients (quadratic a b c d e f) o h) o h =
      fderiv ℝ (quadratic a b c d e f) from
        funext (fun z => fieldDifferential_quadratic a b c d e f o z h hh)]

theorem field_sub (C D : Coefficients) (o q : Coords) (h : ℝ) :
    field (fun i j => C i j - D i j) o h q = field C o h q - field D o h q := by
  simp [field, tensorPatch, sub_mul, Finset.sum_sub_distrib]

theorem field_const (v : ℝ) (o q : Coords) (h : ℝ) :
    field (fun _ _ => v) o h q = v := by
  simp [field, tensorPatch, ← Finset.mul_sum, ← Finset.sum_mul, weights_partition]

theorem field_mono (C D : Coefficients) (o q : Coords) (h : ℝ)
    (hx : (q.1-o.1)/h ∈ Set.Icc (0 : ℝ) 1)
    (hy : (q.2-o.2)/h ∈ Set.Icc (0 : ℝ) 1)
    (hCD : ∀ i j, C i j ≤ D i j) : field C o h q ≤ field D o h q := by
  unfold field tensorPatch weights
  exact Finset.sum_le_sum (fun j _ => Finset.sum_le_sum (fun i _ =>
    mul_le_mul_of_nonneg_right
      (mul_le_mul_of_nonneg_right (hCD i j) (basis_nonneg _ hx i))
      (basis_nonneg _ hy j)))

/-- A uniform error in the 36 stored real coefficients is not amplified in value. -/
theorem field_coefficient_error_bound (C D : Coefficients) (o q : Coords) (h eta : ℝ)
    (hx : (q.1-o.1)/h ∈ Set.Icc (0 : ℝ) 1)
    (hy : (q.2-o.2)/h ∈ Set.Icc (0 : ℝ) 1)
    (hCD : ∀ i j, |C i j - D i j| ≤ eta) :
    |field C o h q - field D o h q| ≤ eta := by
  rw [← field_sub]
  refine abs_le.mpr ⟨?_, ?_⟩
  · simpa only [field_const] using field_mono (fun _ _ => -eta)
      (fun i j => C i j - D i j) o q h hx hy (fun i j => (abs_le.mp (hCD i j)).1)
  · simpa only [field_const] using field_mono (fun i j => C i j - D i j)
      (fun _ _ => eta) o q h hx hy (fun i j => (abs_le.mp (hCD i j)).2)

end FractalSync.OperationalField
