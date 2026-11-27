import FractalSync.OperationalFieldJetError

/-!
# Same-cell query displacement for the concrete operational field

The bounds follow from the stored coefficient magnitudes and the checked
quintic basis constants. Coordinates use the product (maximum) norm. They
compare real queries in one closed cell; an IEEE displacement certificate and
cross-cell regularity are separate obligations.
-/

noncomputable section

namespace FractalSync.OperationalField

/-- The physical closed cell for one fixed 36-coefficient patch. -/
def cell (o : Coords) (h : ℝ) : Set Coords :=
  Set.Icc o.1 (o.1 + h) ×ˢ Set.Icc o.2 (o.2 + h)

theorem normalized_mem_of_mem_cell (o q : Coords) (h : ℝ) (hh : 0 < h)
    (hq : q ∈ cell o h) :
    (q.1 - o.1) / h ∈ Set.Icc (0 : ℝ) 1 ∧
      (q.2 - o.2) / h ∈ Set.Icc (0 : ℝ) 1 := by
  exact ⟨⟨div_nonneg (sub_nonneg.mpr hq.1.1) hh.le,
      (div_le_one hh).mpr (sub_le_iff_le_add.mpr (by simpa [add_comm] using hq.1.2))⟩,
    ⟨div_nonneg (sub_nonneg.mpr hq.2.1) hh.le,
      (div_le_one hh).mpr (sub_le_iff_le_add.mpr (by simpa [add_comm] using hq.2.2))⟩⟩

theorem cell_convex (o : Coords) (h : ℝ) : Convex ℝ (cell o h) :=
  (convex_Icc _ _).prod (convex_Icc _ _)

theorem norm_linearForm_le (x y : ℝ) : ‖linearForm x y‖ ≤ |x| + |y| := by
  refine (linearForm x y).opNorm_le_bound (add_nonneg (abs_nonneg _) (abs_nonneg _)) ?_
  intro v
  calc
    _ = |x * v.1 + y * v.2| := by simp [linearForm, Real.norm_eq_abs]
    _ ≤ |x| * |v.1| + |y| * |v.2| := by simpa [abs_mul] using abs_add_le (x*v.1) (y*v.2)
    _ ≤ |x| * ‖v‖ + |y| * ‖v‖ := add_le_add
      (mul_le_mul_of_nonneg_left (norm_fst_le v) (abs_nonneg x))
      (mul_le_mul_of_nonneg_left (norm_snd_le v) (abs_nonneg y))
    _ = (|x| + |y|) * ‖v‖ := by ring

theorem norm_quadraticForm_le (xx xy yy : ℝ) :
    ‖quadraticForm xx xy yy‖ ≤ |xx| + 2 * |xy| + |yy| := by
  have hfst : ‖ContinuousLinearMap.fst ℝ ℝ ℝ‖ ≤ 1 :=
    (ContinuousLinearMap.fst ℝ ℝ ℝ).opNorm_le_bound zero_le_one
      (fun v => by simpa using norm_fst_le v)
  have hsnd : ‖ContinuousLinearMap.snd ℝ ℝ ℝ‖ ≤ 1 :=
    (ContinuousLinearMap.snd ℝ ℝ ℝ).opNorm_le_bound zero_le_one
      (fun v => by simpa using norm_snd_le v)
  calc
    _ ≤ ‖(linearForm xx xy).smulRight (ContinuousLinearMap.fst ℝ ℝ ℝ)‖ +
        ‖(linearForm xy yy).smulRight (ContinuousLinearMap.snd ℝ ℝ ℝ)‖ := by
      exact norm_add_le
        ((linearForm xx xy).smulRight (ContinuousLinearMap.fst ℝ ℝ ℝ))
        ((linearForm xy yy).smulRight (ContinuousLinearMap.snd ℝ ℝ ℝ))
    _ ≤ ‖linearForm xx xy‖ + ‖linearForm xy yy‖ := by
      simp only [ContinuousLinearMap.norm_smulRight_apply]
      simpa using add_le_add
        (mul_le_mul_of_nonneg_left hfst (norm_nonneg (linearForm xx xy)))
        (mul_le_mul_of_nonneg_left hsnd (norm_nonneg (linearForm xy yy)))
    _ ≤ (|xx| + |xy|) + (|xy| + |yy|) :=
      add_le_add (norm_linearForm_le xx xy) (norm_linearForm_le xy yy)
    _ = |xx| + 2 * |xy| + |yy| := by ring


@[simp] theorem fieldDifferential_zero (o q : Coords) (h : ℝ) :
    fieldDifferential (fun _ _ => 0) o h q = 0 := by
  simp [fieldDifferential, tensorDifferential, tensorPatch, linearForm]

@[simp] theorem fieldHessian_zero (o q : Coords) (h : ℝ) :
    fieldHessian (fun _ _ => 0) o h q = 0 := by
  apply ContinuousLinearMap.ext
  intro v
  apply ContinuousLinearMap.ext
  intro w
  simp [fieldHessian, tensorPatch, quadraticForm, linearForm]

theorem norm_fieldDifferential_le (C : Coefficients) (o q : Coords)
    (h B : ℝ) (hh : 0 < h) (hB : 0 ≤ B) (hq : q ∈ cell o h)
    (hC : ∀ i j, |C i j| ≤ B) :
    ‖fieldDifferential C o h q‖ ≤ (8/3 : ℝ) * B / h := by
  have hq' := normalized_mem_of_mem_cell o q h hh hq
  have hb := fieldDifferential_coefficient_error_bound C (fun _ _ => 0)
    o q h B hh hB hq'.1 hq'.2 (fun i j => by simpa using hC i j)
  simp only [fieldDifferential_zero, ContinuousLinearMap.zero_apply, sub_zero] at hb
  have hrep : fieldDifferential C o h q =
      linearForm (fieldDifferential C o h q (1, 0)) (fieldDifferential C o h q (0, 1)) := by
    simp [fieldDifferential_eq_query, linearForm]
  calc
    _ = ‖linearForm (fieldDifferential C o h q (1, 0))
        (fieldDifferential C o h q (0, 1))‖ := congrArg norm hrep
    _ ≤ |fieldDifferential C o h q (1, 0)| + |fieldDifferential C o h q (0, 1)| :=
      norm_linearForm_le _ _
    _ ≤ (4/3 : ℝ) * B / h + (4/3 : ℝ) * B / h := add_le_add hb.1 hb.2
    _ = (8/3 : ℝ) * B / h := by ring


theorem norm_fieldHessian_le (C : Coefficients) (o q : Coords)
    (h B : ℝ) (hh : 0 < h) (hB : 0 ≤ B) (hq : q ∈ cell o h)
    (hC : ∀ i j, |C i j| ≤ B) :
    ‖fieldHessian C o h q‖ ≤ (68/9 : ℝ) * B / h^2 := by
  have hq' := normalized_mem_of_mem_cell o q h hh hq
  have hb := fieldHessian_coefficient_error_bound C (fun _ _ => 0)
    o q h B hh hB hq'.1 hq'.2 (fun i j => by simpa using hC i j)
  simp only [fieldHessian_zero, ContinuousLinearMap.zero_apply, sub_zero] at hb
  have hrep : fieldHessian C o h q =
      quadraticForm (fieldHessian C o h q (1, 0) (1, 0))
        (fieldHessian C o h q (1, 0) (0, 1))
        (fieldHessian C o h q (0, 1) (0, 1)) := by
    simp [fieldHessian_eq_query, quadraticForm, linearForm]
  calc
    _ = ‖quadraticForm (fieldHessian C o h q (1, 0) (1, 0))
        (fieldHessian C o h q (1, 0) (0, 1))
        (fieldHessian C o h q (0, 1) (0, 1))‖ := congrArg norm hrep
    _ ≤ |fieldHessian C o h q (1, 0) (1, 0)| +
        2 * |fieldHessian C o h q (1, 0) (0, 1)| +
        |fieldHessian C o h q (0, 1) (0, 1)| := norm_quadraticForm_le _ _ _
    _ ≤ 2 * B / h^2 + 2 * ((16/9 : ℝ) * B / h^2) + 2 * B / h^2 :=
      add_le_add (add_le_add hb.1 (mul_le_mul_of_nonneg_left hb.2.1 (by norm_num))) hb.2.2
    _ = (68/9 : ℝ) * B / h^2 := by ring


/-- Actual value variation for two real queries in the same physical cell. -/
theorem field_query_error_bound (C : Coefficients) (o q q' : Coords)
    (h B : ℝ) (hh : 0 < h) (hB : 0 ≤ B)
    (hq : q ∈ cell o h) (hq' : q' ∈ cell o h)
    (hC : ∀ i j, |C i j| ≤ B) :
    |field C o h q' - field C o h q| ≤ (8/3 : ℝ) * B / h * ‖q' - q‖ := by
  simpa only [Real.norm_eq_abs] using
    Convex.norm_image_sub_le_of_norm_hasFDerivWithin_le
      (fun z (_ : z ∈ cell o h) => (hasFDerivAt_field C o h z).hasFDerivWithinAt)
      (fun z hz => norm_fieldDifferential_le C o z h B hh hB hz hC)
      (cell_convex o h) hq hq'

/-- Actual derivative-map variation, with a coefficient-derived Hessian bound. -/
theorem fieldDifferential_query_error_bound (C : Coefficients) (o q q' : Coords)
    (h B : ℝ) (hh : 0 < h) (hB : 0 ≤ B)
    (hq : q ∈ cell o h) (hq' : q' ∈ cell o h)
    (hC : ∀ i j, |C i j| ≤ B) :
    ‖fieldDifferential C o h q' - fieldDifferential C o h q‖ ≤
      (68/9 : ℝ) * B / h^2 * ‖q' - q‖ := by
  exact Convex.norm_image_sub_le_of_norm_hasFDerivWithin_le
    (fun z (_ : z ∈ cell o h) => (hasFDerivAt_fieldDifferential C o h z).hasFDerivWithinAt)
    (fun z hz => norm_fieldHessian_le C o z h B hh hB hz hC)
    (cell_convex o h) hq hq'

/-- Simultaneous coefficient and query perturbation, with an explicit gain. -/
theorem field_combined_error_bound (C D : Coefficients) (o q q' : Coords)
    (h eta B : ℝ) (hh : 0 < h) (hB : 0 ≤ B)
    (hq : q ∈ cell o h) (hq' : q' ∈ cell o h)
    (hCD : ∀ i j, |C i j - D i j| ≤ eta) (hD : ∀ i j, |D i j| ≤ B) :
    |field C o h q' - field D o h q| ≤ eta + (8/3 : ℝ) * B / h * ‖q' - q‖ := by
  have hqnorm := normalized_mem_of_mem_cell o q' h hh hq'
  calc
    _ ≤ |field C o h q' - field D o h q'| + |field D o h q' - field D o h q| :=
      abs_sub_le _ _ _
    _ ≤ eta + (8/3 : ℝ) * B / h * ‖q' - q‖ := add_le_add
      (field_coefficient_error_bound C D o q' h eta hqnorm.1 hqnorm.2 hCD)
      (field_query_error_bound D o q q' h B hh hB hq hq' hD)

end FractalSync.OperationalField
