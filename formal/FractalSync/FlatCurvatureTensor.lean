import FractalSync.CurvatureTensoriality
import FractalSync.NonlinearCoordinates
import FractalSync.FlatShoreGeodesics

/-!
# Flat-Shore curvature as a continuous trilinear tensor

The existing explicit Christoffel coefficients are bundled as actual continuous
bilinear maps and differentiated as a map into that operator space. Generic
curvature tensoriality then applies to arbitrary regular field extensions.
-/

namespace FractalSync.FlatCurvatureTensor

open FlatShoreGeometry CoordinateCovariance CurvatureTensoriality
open scoped ContDiff Topology

/-- Fixed radial-radial coefficient in the coordinate frame. -/
noncomputable def basisRR : Coord →L[ℝ] Coord →L[ℝ] Coord :=
  (ContinuousLinearMap.fst ℝ ℝ ℝ).smulRight
    ((ContinuousLinearMap.fst ℝ ℝ ℝ).smulRight (1, 0))

/-- Fixed tangential-tangential coefficient with normal output. -/
noncomputable def basisYY : Coord →L[ℝ] Coord →L[ℝ] Coord :=
  (ContinuousLinearMap.snd ℝ ℝ ℝ).smulRight
    ((ContinuousLinearMap.snd ℝ ℝ ℝ).smulRight (1, 0))

/-- Fixed symmetric mixed coefficient with tangential output. -/
noncomputable def basisRY : Coord →L[ℝ] Coord →L[ℝ] Coord :=
  (ContinuousLinearMap.fst ℝ ℝ ℝ).smulRight
    ((ContinuousLinearMap.snd ℝ ℝ ℝ).smulRight (0, 1)) +
  (ContinuousLinearMap.snd ℝ ℝ ℝ).smulRight
    ((ContinuousLinearMap.fst ℝ ℝ ℝ).smulRight (0, 1))

/-- The actual flat Christoffel field, bundled continuously bilinearly. -/
noncomputable def connection (epsilon a : ℝ) (x : Coord) : Coord →L[ℝ] Coord →L[ℝ] Coord :=
  gammaRR epsilon a x.1 • basisRR + gammaYY epsilon a x.1 • basisYY +
    gammaRY epsilon x.1 • basisRY

/-- The bundled field evaluates exactly to the existing flat coefficients. -/
theorem connection_apply (epsilon a : ℝ) (x v w : Coord) :
    connection epsilon a x v w = christoffel epsilon a x.1 v w := by
  ext
  all_goals dsimp [connection, basisRR, basisYY, basisRY, christoffel]
  all_goals ring


/-- The radial coefficient is smooth everywhere for a positive floor. -/
theorem contDiff_gammaRR (epsilon a : ℝ) (he : 0 < epsilon) :
    ContDiff ℝ ∞ (gammaRR epsilon a) := by
  unfold gammaRR
  exact ((contDiff_const.mul contDiff_id).div
    (contDiff_const.add (contDiff_const.mul (contDiff_id.pow 2)))
    (fun r => (FlatShore.normal_sq_pos epsilon a r he).ne')).sub
      ((contDiff_const.mul contDiff_id).div (contDiff_const.add (contDiff_id.pow 2))
        (fun r => (FlatShore.ruler_sq_pos epsilon r he).ne'))


/-- The tangential acceleration coefficient is smooth at a positive floor. -/
theorem contDiff_gammaYY (epsilon a : ℝ) (he : 0 < epsilon) :
    ContDiff ℝ ∞ (gammaYY epsilon a) := by
  exact contDiff_id.div (contDiff_const.add (contDiff_const.mul (contDiff_id.pow 2)))
    (fun r => (FlatShore.normal_sq_pos epsilon a r he).ne')


/-- The mixed coefficient is smooth at a positive floor. -/
theorem contDiff_gammaRY (epsilon : ℝ) (he : 0 < epsilon) :
    ContDiff ℝ ∞ (gammaRY epsilon) := by
  exact contDiff_id.neg.div (contDiff_const.add (contDiff_id.pow 2))
    (fun r => (FlatShore.ruler_sq_pos epsilon r he).ne')


/-- The concrete flat connection is smooth as a field of continuous bilinear maps. -/
theorem contDiff_connection (epsilon a : ℝ) (he : 0 < epsilon) :
    ContDiff ℝ ∞ (connection epsilon a) := by
  exact ((((contDiff_gammaRR epsilon a he).comp contDiff_fst).smul contDiff_const).add
    (((contDiff_gammaYY epsilon a he).comp contDiff_fst).smul contDiff_const)).add
      (((contDiff_gammaRY epsilon he).comp contDiff_fst).smul contDiff_const)


/-- The bundled connection's coefficient operation is exactly the prior flat definition. -/
theorem coefficients_connection (epsilon a : ℝ) :
    coefficients (connection epsilon a) = fun x => christoffel epsilon a x.1 := by
  exact funext fun x => funext fun v => funext fun w => connection_apply epsilon a x v w


/-- Every regular flat-Shore curvature calculation equals the bundled tensor
on the three point values; no constant-extension assumption remains. -/
theorem curvature_eq_tensor (epsilon a : ℝ) (he : 0 < epsilon)
    (X Y Z : Coord → Coord) (x : Coord)
    (hX : DifferentiableAt ℝ X x) (hY : DifferentiableAt ℝ Y x)
    (hZ : ContDiffAt ℝ 2 Z x) :
    FlatShoreGeometry.curvature epsilon a X Y Z x =
      curvatureTensor (connection epsilon a) x (X x) (Y x) (Z x) := by
  simpa only [coefficients_connection, CoordinateCovariance.flat_curvature_eq] using
    CurvatureTensoriality.curvature_eq_tensor (connection epsilon a) X Y Z x
      ((contDiff_connection epsilon a he).differentiable (by simp) x) hX hY hZ


/-- The actual normal derivative of the mixed first covariant derivative. -/
theorem fderiv_gammaRY_normal (epsilon : ℝ) (x : Coord) (he : 0 < epsilon) :
    fderiv ℝ (fun p : Coord => ((0 : ℝ), gammaRY epsilon p.1)) x (1, 0) =
      (0, (x.1 ^ 2 - epsilon ^ 2) / (epsilon ^ 2 + x.1 ^ 2) ^ 2) := by
  have hd := (hasFDerivAt_const (0 : ℝ) x).prodMk
    ((hasDerivAt_gammaRY epsilon x.1 he).comp_hasFDerivAt x
      (ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt)
  simpa using congrArg (fun L : Coord →L[ℝ] Coord => L (1, 0)) hd.fderiv


/-- The actual tangential derivative of the radial first covariant derivative vanishes. -/
theorem fderiv_gammaRR_tangent (epsilon a : ℝ) (x : Coord) (he : 0 < epsilon) :
    fderiv ℝ (fun p : Coord => (gammaRR epsilon a p.1, (0 : ℝ))) x (0, 1) = 0 := by
  have hr := ((contDiff_gammaRR epsilon a he).differentiable (by simp) x.1).hasDerivAt
  have hd := (hr.comp_hasFDerivAt x (ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt).prodMk
    (hasFDerivAt_const (0 : ℝ) x)
  simpa using congrArg (fun L : Coord →L[ℝ] Coord => L (0, 1)) hd.fderiv


/-- The missing normal third-input component of the flat curvature operator is computed
from the actual commutator and differentiated intermediate fields. -/
theorem curvature_normal_tangent_normal (epsilon a : ℝ) (x : Coord) (he : 0 < epsilon) :
    FlatShoreGeometry.curvature epsilon a normalField tangentField normalField x =
      (0, -radialMetric epsilon a x.1 * FlatShore.radialCurvature epsilon a x.1) := by
  rw [FlatShoreGeometry.curvature, lieBracket_normal_tangent]
  unfold FlatShoreGeometry.normalField FlatShoreGeometry.tangentField
  rw [FlatShoreGeodesics.covariantDerivative_const, FlatShoreGeodesics.covariantDerivative_const]
  simp only [christoffel, mul_zero, mul_one, add_zero, zero_add]
  simp only [FlatShoreGeometry.covariantDerivative, Pi.zero_apply]
  rw [fderiv_gammaRY_normal epsilon x he, fderiv_gammaRR_tangent epsilon a x he]
  simp [christoffel]
  dsimp only [gammaRR, gammaRY, radialMetric, FlatShore.radialCurvature]
  field_simp [(FlatShore.ruler_sq_pos epsilon x.1 he).ne',
    (FlatShore.normal_sq_pos epsilon a x.1 he).ne']
  ring


/-- The bundled tensor's normal third-input basis component. -/
theorem tensor_normal_tangent_normal (epsilon a : ℝ) (x : Coord) (he : 0 < epsilon) :
    curvatureTensor (connection epsilon a) x (1, 0) (0, 1) (1, 0) =
      (0, -radialMetric epsilon a x.1 * FlatShore.radialCurvature epsilon a x.1) := by
  simpa only [normalField, tangentField] using
    (curvature_eq_tensor epsilon a he normalField tangentField normalField x
      (differentiableAt_const _) (differentiableAt_const _) contDiffAt_const).symm.trans
      (curvature_normal_tangent_normal epsilon a x he)


/-- The bundled tensor's tangential third-input basis component. -/
theorem tensor_normal_tangent_tangent (epsilon a : ℝ) (x : Coord) (he : 0 < epsilon) :
    curvatureTensor (connection epsilon a) x (1, 0) (0, 1) (0, 1) =
      (tangentialMetric epsilon x.1 * FlatShore.radialCurvature epsilon a x.1, 0) := by
  simpa only [normalField, tangentField] using
    (curvature_eq_tensor epsilon a he normalField tangentField tangentField x
      (differentiableAt_const _) (differentiableAt_const _) contDiffAt_const).symm.trans
      (FlatShoreGeometry.curvature_normal_tangent_tangent_eq epsilon a x he)


/-- Repeated first two directions give zero curvature. -/
theorem tensor_self (epsilon a : ℝ) (x u w : Coord) :
    curvatureTensor (connection epsilon a) x u u w = 0 := by
  simp [curvatureTensor_apply, curvatureValue]


/-- Decomposition in the two coordinate directions. -/
theorem coord_basis (u : Coord) : u = u.1 • (1, 0) + u.2 • (0, 1) := by
  ext <;> simp


/-- Full coordinate evaluation follows by continuous trilinearity from the two
actual commutator computations and alternation. -/
theorem tensor_apply_coords (epsilon a : ℝ) (x u v w : Coord) (he : 0 < epsilon) :
    curvatureTensor (connection epsilon a) x u v w =
      (FlatShore.radialCurvature epsilon a x.1 * tangentialMetric epsilon x.1 *
          (u.1 * v.2 - u.2 * v.1) * w.2,
        -FlatShore.radialCurvature epsilon a x.1 * radialMetric epsilon a x.1 *
          (u.1 * v.2 - u.2 * v.1) * w.1) := by
  have hswap : curvatureTensor (connection epsilon a) x (0, 1) (1, 0) =
      -curvatureTensor (connection epsilon a) x (1, 0) (0, 1) := by
    exact ContinuousLinearMap.ext (fun z =>
      curvatureTensor_swap (connection epsilon a) x (0, 1) (1, 0) z)
  conv_lhs => rw [coord_basis u, coord_basis v, coord_basis w]
  simp only [map_add, map_smul, ContinuousLinearMap.add_apply,
    ContinuousLinearMap.smul_apply, hswap, ContinuousLinearMap.neg_apply, tensor_self,
    tensor_normal_tangent_normal epsilon a x he, tensor_normal_tangent_tangent epsilon a x he]
  ext <;> simp [smul_eq_mul] <;> ring


/-- The full two-dimensional curvature tensor has the metric form associated
with the already checked scalar curvature, in arbitrary input directions. -/
theorem tensor_eq_metric (epsilon a : ℝ) (x u v w : Coord) (he : 0 < epsilon) :
    curvatureTensor (connection epsilon a) x u v w =
      FlatShore.radialCurvature epsilon a x.1 •
        (metric epsilon a x.1 v w • u - metric epsilon a x.1 u w • v) := by
  rw [tensor_apply_coords epsilon a x u v w he]
  ext <;> dsimp [metric] <;> ring


/-- The actual curvature commutator in arbitrary regular vector fields is the
metric tensor formula, hence independent of every permitted field extension. -/
theorem curvature_eq_metric (epsilon a : ℝ) (he : 0 < epsilon)
    (X Y Z : Coord → Coord) (x : Coord)
    (hX : DifferentiableAt ℝ X x) (hY : DifferentiableAt ℝ Y x)
    (hZ : ContDiffAt ℝ 2 Z x) :
    FlatShoreGeometry.curvature epsilon a X Y Z x =
      FlatShore.radialCurvature epsilon a x.1 •
        (metric epsilon a x.1 (Y x) (Z x) • X x -
          metric epsilon a x.1 (X x) (Z x) • Y x) := by
  exact (curvature_eq_tensor epsilon a he X Y Z x hX hY hZ).trans
    (tensor_eq_metric epsilon a x (X x) (Y x) (Z x) he)


/-- The Gram determinant records exactly the coordinate-frame determinant. -/
theorem gram_eq_determinant (epsilon a : ℝ) (x u v : Coord) :
    gram (fun p => metric epsilon a p.1) x u v =
      radialMetric epsilon a x.1 * tangentialMetric epsilon x.1 *
        (u.1 * v.2 - u.2 * v.1) ^ 2 := by
  dsimp [gram, metric]
  ring


/-- Every nondegenerate coordinate frame has strictly positive metric Gram determinant. -/
theorem gram_pos (epsilon a : ℝ) (x u v : Coord) (he : 0 < epsilon)
    (hdet : u.1 * v.2 - u.2 * v.1 ≠ 0) :
    0 < gram (fun p => metric epsilon a p.1) x u v := by
  rw [gram_eq_determinant]
  exact mul_pos (mul_pos (radialMetric_pos epsilon a x.1 he)
    (tangentialMetric_pos epsilon x.1 he)) (sq_pos_of_ne_zero hdet)


/-- The curvature numerator in any regular frame is scalar curvature times its
Gram determinant, without a nondegeneracy assumption. -/
theorem curvature_pairing (epsilon a : ℝ) (he : 0 < epsilon)
    (X Y : Coord → Coord) (x : Coord)
    (hX : DifferentiableAt ℝ X x) (hY : ContDiffAt ℝ 2 Y x) :
    metric epsilon a x.1 (FlatShoreGeometry.curvature epsilon a X Y Y x) (X x) =
      FlatShore.radialCurvature epsilon a x.1 *
        gram (fun p => metric epsilon a p.1) x (X x) (Y x) := by
  rw [curvature_eq_metric epsilon a he X Y Y x hX (hY.differentiableAt one_le_two) hY]
  dsimp [gram, metric]
  ring


/-- The scalar curvature is independent of every regular nondegenerate frame,
including nonconstant fields and arbitrary permitted extensions. -/
theorem gramRatio_eq_radialCurvature (epsilon a : ℝ) (he : 0 < epsilon)
    (X Y : Coord → Coord) (x : Coord)
    (hX : DifferentiableAt ℝ X x) (hY : ContDiffAt ℝ 2 Y x)
    (hdet : (X x).1 * (Y x).2 - (X x).2 * (Y x).1 ≠ 0) :
    gramRatio (fun p => metric epsilon a p.1) (fun p => christoffel epsilon a p.1) X Y x =
      FlatShore.radialCurvature epsilon a x.1 := by
  rw [gramRatio, flat_curvature_eq, curvature_pairing epsilon a he X Y x hX hY]
  exact mul_div_cancel_right₀ _ (gram_pos epsilon a x (X x) (Y x) he hdet).ne'


/-- The intermediate flat covariant derivative is actually differentiable
under local first/second derivative hypotheses on the fields. -/
theorem differentiableAt_covariantDerivative (epsilon a : ℝ) (he : 0 < epsilon)
    (X Y : Coord → Coord) (x : Coord)
    (hX : DifferentiableAt ℝ X x) (hY : ContDiffAt ℝ 2 Y x) :
    DifferentiableAt ℝ (FlatShoreGeometry.covariantDerivative epsilon a X Y) x := by
  simpa only [coefficients_connection, flat_covariantDerivative_eq] using
    CurvatureTensoriality.differentiableAt_covariantDerivative (connection epsilon a) X Y x
      ((contDiff_connection epsilon a he).differentiable (by simp) x) hX
      (hY.differentiableAt one_le_two)
      ((hY.fderiv_right (m := 1) (by norm_num)).differentiableAt le_rfl)


/-- A nonlinear chart and an arbitrary regular frame give the same flat scalar
curvature. All intermediate differentiability premises are discharged here. -/
theorem frame_gramRatio_pullback {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    [CompleteSpace E] (f : E → Coord) (epsilon a : ℝ) (x : E) (he : 0 < epsilon)
    (hf : ContDiffAt ℝ 2 f x) (hi : (fderiv ℝ f x).IsInvertible)
    (X Y : Coord → Coord) (hX : DifferentiableAt ℝ X (f x))
    (hY : ContDiffAt ℝ 2 Y (f x))
    (hdet : (X (f x)).1 * (Y (f x)).2 - (X (f x)).2 * (Y (f x)).1 ≠ 0) :
    gramRatio
      (NonlinearCoordinates.pullbackMetric f (fun p => metric epsilon a p.1))
      (NonlinearCoordinates.pullbackConnection f (fun p => christoffel epsilon a p.1))
      (VectorField.pullback ℝ f X) (VectorField.pullback ℝ f Y) x =
      FlatShore.radialCurvature epsilon a (f x).1 := by
  refine (NonlinearCoordinates.gramRatio_pullback f
    (fun p => metric epsilon a p.1) (fun p => christoffel epsilon a p.1)
    X Y x hf hi hX ?_ ?_ ?_).trans
      (gramRatio_eq_radialCurvature epsilon a he X Y (f x) hX hY hdet)
  · exact (hY.eventually (by norm_num)).mono fun z hz => hz.differentiableAt one_le_two
  · simpa only [flat_covariantDerivative_eq] using
      differentiableAt_covariantDerivative epsilon a he Y Y (f x)
        (hY.differentiableAt one_le_two) hY
  · simpa only [flat_covariantDerivative_eq] using
      differentiableAt_covariantDerivative epsilon a he X Y (f x) hX hY


/-- The denominator in the nonlinear arbitrary-frame curvature theorem is
strictly positive; nondegeneracy is preserved by the actual invertible Jacobian. -/
theorem frame_gram_pullback_pos {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
    (f : E → Coord) (epsilon a : ℝ) (x : E) (he : 0 < epsilon)
    (hi : (fderiv ℝ f x).IsInvertible) (X Y : Coord → Coord)
    (hdet : (X (f x)).1 * (Y (f x)).2 - (X (f x)).2 * (Y (f x)).1 ≠ 0) :
    0 < gram (NonlinearCoordinates.pullbackMetric f (fun p => metric epsilon a p.1)) x
      (VectorField.pullback ℝ f X x) (VectorField.pullback ℝ f Y x) := by
  rw [NonlinearCoordinates.gram_pullback f _ X Y x hi]
  exact gram_pos epsilon a (f x) (X (f x)) (Y (f x)) he hdet

end FractalSync.FlatCurvatureTensor
