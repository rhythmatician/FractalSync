import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.LinearAlgebra.Matrix.SchurComplement
import Mathlib.LinearAlgebra.Matrix.Notation
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Module
import Mathlib.Tactic.Ring

/-!
# Positive local-ruler metrics and metric-plane rotations

Development-time foundations for #157 and #159. The rank-one statements hold
in an arbitrary real inner-product space; neither Mandelbrot membership nor a
signed-distance hypothesis is needed. In the application `u = grad rho` and
`a = lambda / log 2`. The plane identities apply to every symmetric positive
definite 2-by-2 metric, not only the rank-one family.

These are exact-real identities, not certificates for floating-point evaluation
or the GeometryProvider's differentiability and error contracts.
-/

open scoped InnerProductSpace

namespace FractalSync.ScaleMetric

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The dimensionless normal-to-tangent eigenvalue ratio. -/
noncomputable def weight (a : ℝ) (u : E) : ℝ := 1 + a ^ 2 * ⟪u, u⟫_ℝ

/-- The unscaled rank-one graph operator. -/
noncomputable def stretch (a : ℝ) (u v : E) : E := v + (a ^ 2 * ⟪u, v⟫_ℝ) • u

/-- The Sherman--Morrison inverse of the graph operator. -/
noncomputable def unstretch (a : ℝ) (u v : E) : E :=
  v - (a ^ 2 * ⟪u, v⟫_ℝ / weight a u) • u

/-- Applying the scale-relative metric to a coordinate vector. The ambient
inner product identifies vectors with covectors only for writing this formula. -/
noncomputable def metricAction (rho a : ℝ) (u v : E) : E :=
  (rho ^ 2)⁻¹ • stretch a u v

/-- Applying the inverse scale-relative metric to a coordinate covector. -/
noncomputable def inverseAction (rho a : ℝ) (u v : E) : E :=
  rho ^ 2 • unstretch a u v

/-- The induced bilinear metric. -/
noncomputable def metric (rho a : ℝ) (u v w : E) : ℝ :=
  (⟪v, w⟫_ℝ + a ^ 2 * ⟪u, v⟫_ℝ * ⟪u, w⟫_ℝ) / rho ^ 2

/-- Kinetic energy at a fixed configuration. -/
noncomputable def kinetic (rho a : ℝ) (u v : E) : ℝ := metric rho a u v v / 2

/-- The inverse denominator is strictly positive without a nonzero-gradient assumption. -/
theorem weight_pos (a : ℝ) (u : E) : 0 < weight a u := by
  exact add_pos_of_pos_of_nonneg zero_lt_one (mul_nonneg (sq_nonneg a) real_inner_self_nonneg)

/-- The rank-one inverse is a left inverse in every real inner-product space. -/
theorem unstretch_stretch (a : ℝ) (u v : E) : unstretch a u (stretch a u v) = v := by
  simp only [unstretch, stretch, inner_add_right, real_inner_smul_right]
  match_scalars
  all_goals field_simp [ne_of_gt (weight_pos a u)]
  simp [weight]

/-- The same closed form is also a right inverse. -/
theorem stretch_unstretch (a : ℝ) (u v : E) : stretch a u (unstretch a u v) = v := by
  simp only [unstretch, stretch, inner_sub_right, real_inner_smul_right]
  match_scalars <;> field_simp [ne_of_gt (weight_pos a u)]
  simp [weight]

theorem stretch_smul (a c : ℝ) (u v : E) : stretch a u (c • v) = c • stretch a u v := by
  simp only [stretch, real_inner_smul_right, smul_add, smul_smul]
  module

theorem unstretch_smul (a c : ℝ) (u v : E) : unstretch a u (c • v) = c • unstretch a u v := by
  simp only [unstretch, real_inner_smul_right, smul_sub, smul_smul]
  module

/-- The displayed closed inverse cancels the full scale-relative metric. -/
theorem inverse_metricAction (rho a : ℝ) (u v : E) (hrho : rho ≠ 0) :
    inverseAction rho a u (metricAction rho a u v) = v := by
  simp [inverseAction, metricAction, unstretch_smul, unstretch_stretch, smul_smul, hrho]

/-- Cancellation also holds in the opposite order. -/
theorem metric_inverseAction (rho a : ℝ) (u v : E) (hrho : rho ≠ 0) :
    metricAction rho a u (inverseAction rho a u v) = v := by
  simp [inverseAction, metricAction, stretch_smul, stretch_unstretch, smul_smul, hrho]

/-- The norm formula makes the positive-definiteness hypothesis transparent. -/
theorem metric_self_formula (rho a : ℝ) (u v : E) :
    metric rho a u v v = (‖v‖ ^ 2 + a ^ 2 * ⟪u, v⟫_ℝ ^ 2) / rho ^ 2 := by
  simp [metric, real_inner_self_eq_norm_sq, pow_two, mul_assoc]

/-- Every nonzero vector has strictly positive squared metric length. -/
theorem metric_self_pos (rho a : ℝ) (u v : E) (hrho : 0 < rho) (hv : v ≠ 0) :
    0 < metric rho a u v v := by
  rw [metric_self_formula]
  exact div_pos (add_pos_of_pos_of_nonneg (sq_pos_of_pos (norm_pos_iff.mpr hv))
    (mul_nonneg (sq_nonneg a) (sq_nonneg _))) (sq_pos_of_pos hrho)

/-- Squared metric length is nonnegative, including at zero velocity. -/
theorem metric_self_nonneg (rho a : ℝ) (u v : E) : 0 ≤ metric rho a u v v := by
  rw [metric_self_formula]
  exact div_nonneg (add_nonneg (sq_nonneg _) (mul_nonneg (sq_nonneg _) (sq_nonneg _)))
    (sq_nonneg _)

theorem metric_symm (rho a : ℝ) (u v w : E) : metric rho a u v w = metric rho a u w v := by
  simp only [metric, real_inner_comm v w]
  ring

theorem metric_add_left (rho a : ℝ) (u v w z : E) :
    metric rho a u (v + w) z = metric rho a u v z + metric rho a u w z := by
  simp only [metric, inner_add_left, inner_add_right]
  ring

theorem metric_smul_left (rho a c : ℝ) (u v w : E) :
    metric rho a u (c • v) w = c * metric rho a u v w := by
  simp only [metric, real_inner_smul_left, real_inner_smul_right]
  ring

/-- The bilinear form and the explicit metric operator agree. -/
theorem inner_metricAction (rho a : ℝ) (u v w : E) :
    ⟪v, metricAction rho a u w⟫_ℝ = metric rho a u v w := by
  simp only [metricAction, stretch, metric, real_inner_smul_right, inner_add_right,
    real_inner_comm u v]
  ring

theorem weight_norm (a : ℝ) (u : E) : weight a u = 1 + a ^ 2 * ‖u‖ ^ 2 := by
  simp [weight, real_inner_self_eq_norm_sq]

/-- Equivalent kinetic-energy formula in ordinary ambient coordinates. -/
theorem kinetic_formula (rho a : ℝ) (u v : E) :
    kinetic rho a u v = (‖v‖ ^ 2 + a ^ 2 * ⟪u, v⟫_ℝ ^ 2) / (2 * rho ^ 2) := by
  simp [kinetic, metric_self_formula, div_div, mul_comm]

/-- Pointwise comparison with the conformal metric, without requiring an ideal SDF. -/
theorem metric_norm_lower (rho a : ℝ) (u v : E) :
    ‖v‖ ^ 2 / rho ^ 2 ≤ metric rho a u v v := by
  rw [metric_self_formula]
  exact div_le_div_of_nonneg_right
    (le_add_of_nonneg_right (mul_nonneg (sq_nonneg a) (sq_nonneg _))) (sq_nonneg rho)

theorem metric_norm_upper (rho a : ℝ) (u v : E) :
    metric rho a u v v ≤ weight a u * ‖v‖ ^ 2 / rho ^ 2 := by
  rw [metric_self_formula, weight_norm]
  refine div_le_div_of_nonneg_right ?_ (sq_nonneg rho)
  have hcs : ⟪u, v⟫_ℝ ^ 2 ≤ ‖u‖ ^ 2 * ‖v‖ ^ 2 := by
    simpa only [← pow_two, real_inner_self_eq_norm_sq] using real_inner_mul_inner_self_le u v
  nlinarith [mul_le_mul_of_nonneg_left hcs (sq_nonneg a)]

/-- A true ideal-SDF derivative bound gives the claimed uniform anisotropy bound. -/
theorem weight_bounds (a : ℝ) (u : E) (hu : ‖u‖ ≤ 1) :
    1 ≤ weight a u ∧ weight a u ≤ 1 + a ^ 2 := by
  rw [weight_norm]
  exact ⟨le_add_of_nonneg_right (mul_nonneg (sq_nonneg a) (sq_nonneg _)),
    add_le_add_left (by simpa using (mul_le_mul_of_nonneg_left
      ((sq_le_sq₀ (norm_nonneg u) zero_le_one).mpr hu) (sq_nonneg a))) 1⟩

/-- Perpendicular vectors have the scale-only eigenvalue. -/
theorem metricAction_tangent (rho a : ℝ) (u v : E) (huv : ⟪u, v⟫_ℝ = 0) :
    metricAction rho a u v = (rho ^ 2)⁻¹ • v := by
  simp [metricAction, stretch, huv]

/-- The gradient direction satisfies the normal eigenvalue equation. If u=0,
this remains an identity, but does not assert that zero is an eigenvector. -/
theorem metricAction_normal (rho a : ℝ) (u : E) :
    metricAction rho a u u = (weight a u / rho ^ 2) • u := by
  simp only [metricAction, stretch, weight, div_eq_mul_inv]
  module

/-- The same rank-one metric in a finite orthonormal coordinate system. -/
noncomputable def metricMatrix {ι : Type*} [DecidableEq ι] (rho a : ℝ)
    (u : ι → ℝ) : Matrix ι ι ℝ :=
  (rho ^ 2)⁻¹ • (1 + Matrix.vecMulVec (fun i => a ^ 2 * u i) u)

/-- Determinant in every finite dimension, including dimension zero. -/
theorem det_metricMatrix {ι : Type*} [Fintype ι] [DecidableEq ι]
    (rho a : ℝ) (u : ι → ℝ) :
    (metricMatrix rho a u).det =
      (rho ^ 2)⁻¹ ^ Fintype.card ι * (1 + a ^ 2 * ∑ i, u i ^ 2) := by
  rw [metricMatrix, Matrix.det_smul, Matrix.vecMulVec_eq Unit,
    Matrix.det_one_add_replicateCol_mul_replicateRow]
  simp [dotProduct, Finset.mul_sum, pow_two, mul_left_comm, mul_assoc]

theorem det_metricMatrix_pos {ι : Type*} [Fintype ι] [DecidableEq ι]
    (rho a : ℝ) (u : ι → ℝ) (hrho : 0 < rho) :
    0 < (metricMatrix rho a u).det := by
  rw [det_metricMatrix]
  exact mul_pos (pow_pos (inv_pos.mpr (sq_pos_of_pos hrho)) _)
    (add_pos_of_pos_of_nonneg zero_lt_one
      (mul_nonneg (sq_nonneg a) (Finset.sum_nonneg fun i _ => sq_nonneg (u i))))

theorem metricMatrix_apply {ι : Type*} [DecidableEq ι] (rho a : ℝ)
    (u : ι → ℝ) (i j : ι) :
    metricMatrix rho a u i j = ((if i = j then 1 else 0) + a ^ 2 * u i * u j) / rho ^ 2 := by
  simp [metricMatrix, Matrix.vecMulVec, Matrix.one_apply, div_eq_mul_inv,
    mul_comm, mul_left_comm]

namespace Plane

/-- The symmetric coordinate form. Pairs here are coordinates, with no claim
that the default norm on a product type is the Euclidean norm. -/
noncomputable def form (A B C : ℝ) (v w : ℝ × ℝ) : ℝ :=
  A * v.1 * w.1 + B * (v.1 * w.2 + v.2 * w.1) + C * v.2 * w.2

noncomputable def delta (A B C : ℝ) : ℝ := A * C - B ^ 2

/-- The orientation-compatible metric quarter-turn. -/
noncomputable def quarter (A B C : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  ((-B * v.1 - C * v.2) / Real.sqrt (delta A B C),
    (A * v.1 + B * v.2) / Real.sqrt (delta A B C))

/-- The quarter-turn squares to minus the identity. -/
theorem quarter_square (A B C : ℝ) (v : ℝ × ℝ) (hd : 0 < delta A B C) :
    quarter A B C (quarter A B C v) = -v := by
  apply Prod.ext
  all_goals simp only [quarter, Prod.fst_neg, Prod.snd_neg]
  all_goals field_simp [ne_of_gt (Real.sqrt_pos.2 hd)]
  all_goals rw [Real.sq_sqrt hd.le]
  all_goals dsimp [delta]; ring

/-- Orthogonality is algebraic, even before imposing positive definiteness. -/
theorem quarter_orthogonal (A B C : ℝ) (v : ℝ × ℝ) :
    form A B C v (quarter A B C v) = 0 := by
  simp only [form, quarter]
  ring

/-- The matrix identity Jᵀ G J=G, expressed as preservation of every pairing. -/
theorem quarter_isometry (A B C : ℝ) (v w : ℝ × ℝ) (hd : 0 < delta A B C) :
    form A B C (quarter A B C v) (quarter A B C w) = form A B C v w := by
  simp only [form, quarter]
  field_simp [ne_of_gt (Real.sqrt_pos.2 hd)]
  rw [Real.sq_sqrt hd.le]
  dsimp [delta]
  ring

/-- Skew-adjointness provides the cross-term cancellation in every rotation. -/
theorem quarter_skew (A B C : ℝ) (v w : ℝ × ℝ) :
    form A B C v (quarter A B C w) + form A B C (quarter A B C v) w = 0 := by
  simp only [form, quarter]
  ring

theorem form_mix (A B C c s : ℝ) (v v' w w' : ℝ × ℝ) :
    form A B C (c • v + s • v') (c • w + s • w') =
      c ^ 2 * form A B C v w + c * s * (form A B C v w' + form A B C v' w) +
        s ^ 2 * form A B C v' w' := by
  simp [form]
  ring

/-- Every unit-circle combination of I and J preserves all metric pairings. -/
theorem circle_rotation_isometry (A B C c s : ℝ) (v w : ℝ × ℝ)
    (hd : 0 < delta A B C) (hcircle : c ^ 2 + s ^ 2 = 1) :
    form A B C (c • v + s • quarter A B C v) (c • w + s • quarter A B C w) =
      form A B C v w := by
  rw [form_mix, quarter_skew, quarter_isometry A B C v w hd]
  simp only [mul_zero, add_zero, ← add_mul, hcircle, one_mul]

noncomputable def rotation (A B C theta : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  Real.cos theta • v + Real.sin theta • quarter A B C v

/-- The canonical angular rotation R(theta)=cos(theta) I+sin(theta) J is an isometry. -/
theorem rotation_isometry (A B C theta : ℝ) (v w : ℝ × ℝ) (hd : 0 < delta A B C) :
    form A B C (rotation A B C theta v) (rotation A B C theta w) = form A B C v w := by
  exact circle_rotation_isometry A B C (Real.cos theta) (Real.sin theta) v w hd
    (Real.cos_sq_add_sin_sq theta)

end Plane

end FractalSync.ScaleMetric
