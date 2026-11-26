import FractalSync.ConcreteAssetFidelity

/-!
# Local bounds for the literal quadratic operational fixture

The finite coefficient error is extended by zero. Its global cross-cell jet
bounds control local Hessian variation after exact quadratic cancellation.
All norms on `Plane` are Euclidean; no machine-evaluation claim is made.
-/

noncomputable section

namespace FractalSync.ConcreteLocalFieldBounds

open OperationalField OperationalHalo OperationalDynamics ConcreteOperationalAsset
  ConcreteAssetFidelity FlatShoreIntrinsic Set Metric
open scoped BigOperators

def idealCoefficients : Lattice :=
  fun i j => reference ((i : ℝ) * spacing, (j : ℝ) * spacing)

def errorCoefficients : Lattice :=
  zeroExtension (fun i j => ConcreteOperationalAsset.coefficients i j - idealCoefficients i j) 16 16

theorem errorCoefficients_bound : ∀ i j, |errorCoefficients i j| ≤ sampleError := by
  exact abs_zeroExtension_le _ 16 16 sampleError (by norm_num [sampleError]) finite_sample_error

theorem physicalJet_sub (n m : ℕ) (C D : Lattice) (o q : Coords) (h : ℝ) :
    physicalJet n m (fun i j => C i j - D i j) o h q =
      physicalJet n m C o h q - physicalJet n m D o h q := by
  simp [physicalJet, globalJet, latticePatchJet, latticeCoefficients, tensorPatch,
    sub_mul, Finset.sum_sub_distrib, sub_div]

theorem physicalHessian_sub (C D : Lattice) (o q : Coords) (h : ℝ) :
    physicalHessian (fun i j => C i j - D i j) o h q =
      physicalHessian C o h q - physicalHessian D o h q := by
  simp only [physicalHessian, physicalJet_sub, quadraticForm_sub]

theorem ideal_selectedCoefficients (q : Coords) :
    latticeCoefficients idealCoefficients ⌊(normalizedCoordinates origin spacing q).1⌋
      ⌊(normalizedCoordinates origin spacing q).2⌋ =
      sampledCoefficients reference (queryCellOrigin origin spacing q) spacing := by
  ext i j
  simp only [latticeCoefficients, idealCoefficients, sampledCoefficients, node,
    queryCellOrigin, origin, Int.cast_sub, Int.cast_add, Int.cast_natCast, Int.cast_ofNat,
    zero_add]
  congr 2 <;> ring

theorem hasFDerivAt_reference (q : Coords) :
    HasFDerivAt reference (linearForm (2 * q.1 + 2 * q.2) (2 * q.1 - 6 * q.2)) q := by
  convert ((linearForm 1 (-1)).hasFDerivAt.mul (linearForm 1 3).hasFDerivAt) using 1
  all_goals ext
  all_goals simp [reference, quadratic, linearForm, mul_comm, mul_left_comm]
  all_goals ring

theorem fderiv_reference (q : Coords) :
    fderiv ℝ reference q = linearForm (2 * q.1 + 2 * q.2) (2 * q.1 - 6 * q.2) :=
  (hasFDerivAt_reference q).fderiv

theorem hasFDerivAt_referenceDifferential (q : Coords) :
    HasFDerivAt (fderiv ℝ reference) (quadraticForm 2 2 (-6)) q := by
  rw [show fderiv ℝ reference =
    (fun z => linearForm (2 * z.1 + 2 * z.2) (2 * z.1 - 6 * z.2)) from funext fderiv_reference]
  convert hasFDerivAt_linearForm (linearForm 2 2) (linearForm 2 (-6))
    (linearForm 2 2) (linearForm 2 (-6)) q
    (linearForm 2 2).hasFDerivAt (linearForm 2 (-6)).hasFDerivAt using 1
  simp [linearForm, sub_eq_add_neg]

theorem physicalHessian_ideal (q : Coords) :
    physicalHessian idealCoefficients origin spacing q = quadraticForm 2 2 (-6) := by
  rw [physicalHessian_eq_query_patch _ _ _ _ spacing_pos.ne', ideal_selectedCoefficients,
    reference, fieldHessian_quadratic _ _ _ _ _ _ _ _ _ spacing_pos.ne',
    ← reference, (hasFDerivAt_referenceDifferential q).fderiv]

theorem physicalHessian_error (q : Coords) (hq : q ∈ physicalHalo 16 16 origin spacing) :
    physicalHessian errorCoefficients origin spacing q =
      physicalHessian ConcreteOperationalAsset.coefficients origin spacing q - quadraticForm 2 2 (-6) := by
  have hj (n m : ℕ) := physicalJet_eq_of_mem_halo n m errorCoefficients
    (fun i j => ConcreteOperationalAsset.coefficients i j - idealCoefficients i j)
    16 16 origin q spacing (zeroExtension_agrees _ 16 16) hq
  rw [← physicalHessian_ideal q, ← physicalHessian_sub, physicalHessian,
    hj 2 0, hj 1 1, hj 0 2, physicalHessian]

/-- Tiny finite sample noise controls Hessian variation across all selected cells. -/
theorem physicalHessian_sub_le (q q' : Coords)
    (hq : q ∈ physicalHalo 16 16 origin spacing)
    (hq' : q' ∈ physicalHalo 16 16 origin spacing) :
    ‖physicalHessian ConcreteOperationalAsset.coefficients origin spacing q' -
      physicalHessian ConcreteOperationalAsset.coefficients origin spacing q‖ ≤
      32 * sampleError / spacing ^ 3 * ‖q' - q‖ := by
  have he := physicalHessian_query_error_bound errorCoefficients
      origin q q' spacing sampleError spacing_pos (by norm_num [sampleError]) errorCoefficients_bound
  rw [physicalHessian_error q hq, physicalHessian_error q' hq'] at he
  convert he using 1
  congr 1
  abel

theorem closedBall_openHalo (x : Plane) (hx : x ∈ closedBall position radius) :
    chart x ∈ openPhysicalHalo 16 16 origin spacing := by
  exact euclidean_closedBall_subset_openPhysicalHalo 16 16 origin position spacing radius
    spacing_pos (lt_trans (by norm_num [radius]) halo_margin_gt_quarter) hx

theorem closedBall_halo (x : Plane) (hx : x ∈ closedBall position radius) :
    chart x ∈ physicalHalo 16 16 origin spacing :=
  ⟨⟨(closedBall_openHalo x hx).1.1.le, (closedBall_openHalo x hx).1.2⟩,
   ⟨(closedBall_openHalo x hx).2.1.le, (closedBall_openHalo x hx).2.2⟩⟩

theorem third_error_gain : 32 * sampleError / spacing ^ 3 ≤ 1 := by
  norm_num [sampleError, spacing]

theorem pull_linearForm_eq_dual (a b : ℝ) :
    pullDifferential (linearForm a b) =
      InnerProductSpace.toDual ℝ Plane (WithLp.toLp 2 (a, b)) := by
  apply ContinuousLinearMap.ext
  intro x
  simp [pullDifferential, linearForm, chart, InnerProductSpace.toDual_apply,
    WithLp.prod_inner_apply, mul_comm]

theorem norm_pull_linearForm_sq (a b : ℝ) :
    ‖pullDifferential (linearForm a b)‖ ^ 2 = a ^ 2 + b ^ 2 := by
  simp [pull_linearForm_eq_dual, WithLp.prod_norm_sq_eq_of_L2]

theorem norm_referenceHessian_le : ‖pullBilinear (quadraticForm 2 2 (-6))‖ ≤ 8 := by
  apply ContinuousLinearMap.opNorm_le_bound _ (by norm_num)
  intro x
  change ‖pullDifferential (linearForm (2 * (chart x).1 + 2 * (chart x).2)
    (2 * (chart x).1 + (-6) * (chart x).2))‖ ≤ 8 * ‖x‖
  have hx : ‖x‖ ^ 2 = (chart x).1 ^ 2 + (chart x).2 ^ 2 := by
    simpa [chart, Real.norm_eq_abs] using WithLp.prod_norm_sq_eq_of_L2 x
  nlinarith [norm_pull_linearForm_sq (2 * (chart x).1 + 2 * (chart x).2)
    (2 * (chart x).1 + (-6) * (chart x).2), norm_nonneg x,
    sq_nonneg ((chart x).1 + (chart x).2), sq_nonneg (chart x).1]

def referenceDifferential (x : Plane) : Plane →L[ℝ] ℝ :=
  pullDifferential (fderiv ℝ reference (chart x))

theorem hasFDerivAt_referenceEuclideanDifferential (x : Plane) :
    HasFDerivAt referenceDifferential (pullBilinear (quadraticForm 2 2 (-6))) x := by
  exact pullDifferential.hasFDerivAt.comp x
    ((hasFDerivAt_referenceDifferential (chart x)).comp x chart.hasFDerivAt)

theorem referenceDifferential_sub_le (x y : Plane) :
    ‖referenceDifferential x - referenceDifferential y‖ ≤ 8 * dist x y := by
  simpa only [dist_eq_norm] using Convex.norm_image_sub_le_of_norm_hasFDerivWithin_le
    (fun z (_ : z ∈ (univ : Set Plane)) =>
      (hasFDerivAt_referenceEuclideanDifferential z).hasFDerivWithinAt)
    (fun _ _ => norm_referenceHessian_le) convex_univ (mem_univ y) (mem_univ x)

theorem norm_referenceDifferential_center : ‖referenceDifferential position‖ ≤ 3 := by
  norm_num [referenceDifferential, fderiv_reference, position, chart]
  nlinarith [norm_pull_linearForm_sq 2 (-2)]

theorem norm_referenceDifferential_local (x : Plane)
    (hx : x ∈ closedBall position radius) : ‖referenceDifferential x‖ ≤ 7 / 2 := by
  have he := referenceDifferential_sub_le x position
  have hd : dist x position ≤ 1 / 16 := mem_closedBall.mp hx
  linarith [norm_le_norm_sub_add (referenceDifferential x) (referenceDifferential position),
    norm_referenceDifferential_center]

theorem local_hessian_norm_le (x : Plane) (hx : x ∈ closedBall position radius) :
    ‖euclideanHessianForm ConcreteOperationalAsset.coefficients origin spacing x‖ ≤ 12 := by
  have he := (norm_pullBilinear_le (physicalHessian ConcreteOperationalAsset.coefficients
    origin spacing (chart x) - fderiv ℝ (fderiv ℝ reference) (chart x))).trans
      ((physicalHessian_reference_error (chart x) (closedBall_halo x hx)).trans reference_gains.2)
  change ‖euclideanHessianForm ConcreteOperationalAsset.coefficients origin spacing x -
    pullBilinear (fderiv ℝ (fderiv ℝ reference) (chart x))‖ ≤ 1 / 274877906944 at he
  rw [(hasFDerivAt_referenceDifferential (chart x)).fderiv] at he
  linarith [norm_le_norm_sub_add
    (euclideanHessianForm ConcreteOperationalAsset.coefficients origin spacing x)
    (pullBilinear (quadraticForm 2 2 (-6))), norm_referenceHessian_le]

theorem local_differential_norm_le (x : Plane) (hx : x ∈ closedBall position radius) :
    ‖euclideanDifferential ConcreteOperationalAsset.coefficients origin spacing x‖ ≤ 4 := by
  have he := (norm_pullDifferential_le (physicalDifferential ConcreteOperationalAsset.coefficients
    origin spacing (chart x) - fderiv ℝ reference (chart x))).trans
      ((physicalDifferential_reference_error (chart x) (closedBall_halo x hx)).trans reference_gains.1)
  change ‖euclideanDifferential ConcreteOperationalAsset.coefficients origin spacing x -
    referenceDifferential x‖ ≤ 1 / 8796093022208 at he
  linarith [norm_le_norm_sub_add
    (euclideanDifferential ConcreteOperationalAsset.coefficients origin spacing x)
    (referenceDifferential x), norm_referenceDifferential_local x hx]

theorem reference_center : reference (chart position) = 0 := by
  norm_num [reference, quadratic, chart, position]

theorem hasFDerivAt_euclideanReference (x : Plane) :
    HasFDerivAt (reference ∘ chart) (referenceDifferential x) x := by
  simpa only [referenceDifferential, fderiv_reference] using
    (hasFDerivAt_reference (chart x)).comp x chart.hasFDerivAt

theorem local_reference_value (x : Plane) (hx : x ∈ closedBall position radius) :
    |reference (chart x)| ≤ 7 / 32 := by
  have he := Convex.norm_image_sub_le_of_norm_hasFDerivWithin_le
    (fun z (hz : z ∈ closedBall position radius) =>
      (hasFDerivAt_euclideanReference z).hasFDerivWithinAt)
    (fun z hz => norm_referenceDifferential_local z hz) (convex_closedBall position radius)
    (mem_closedBall_self radius_pos.le) hx
  simp only [Function.comp_apply, reference_center, sub_zero, Real.norm_eq_abs] at he
  have hd : ‖x - position‖ ≤ 1 / 16 := mem_closedBall_iff_norm.mp hx
  linarith

theorem local_field_abs_le (x : Plane) (hx : x ∈ closedBall position radius) :
    |physicalField ConcreteOperationalAsset.coefficients origin spacing (chart x)| ≤ 1 / 4 := by
  have he := physicalField_reference_error (chart x) (closedBall_halo x hx)
  have hs : sampleError + |spacing ^ 2| ≤ 1 / 32 := by norm_num [sampleError, spacing]
  have hr : |reconstructedReference (chart x)| ≤ |reference (chart x)| + |spacing ^ 2| := by
    simpa only [reconstructedReference, Real.norm_eq_abs] using
      norm_sub_le (reference (chart x)) (spacing ^ 2)
  have ht := abs_sub_le (physicalField ConcreteOperationalAsset.coefficients origin spacing (chart x))
    (reconstructedReference (chart x)) 0
  simp only [sub_zero] at ht
  linarith [local_reference_value x hx]

theorem local_hessian_sub_le (x y : Plane)
    (hx : x ∈ closedBall position radius) (hy : y ∈ closedBall position radius) :
    ‖euclideanHessianForm ConcreteOperationalAsset.coefficients origin spacing x -
      euclideanHessianForm ConcreteOperationalAsset.coefficients origin spacing y‖ ≤ dist x y := by
  rw [euclideanHessianForm, euclideanHessianForm, ← map_sub, dist_eq_norm]
  refine (norm_pullBilinear_le _).trans ((physicalHessian_sub_le (chart y) (chart x)
    (closedBall_halo y hy) (closedBall_halo x hx)).trans ?_)
  simpa only [← map_sub, one_mul] using (mul_le_mul_of_nonneg_right
    third_error_gain (norm_nonneg (chart x - chart y))).trans
      (by simpa only [← map_sub, one_mul] using norm_chart_apply_le (x - y))

end FractalSync.ConcreteLocalFieldBounds
