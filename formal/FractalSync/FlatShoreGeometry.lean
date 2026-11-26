import FractalSync.FlatShore
import Mathlib.Analysis.Calculus.FDeriv.Prod
import Mathlib.Analysis.Calculus.Deriv.Prod
import Mathlib.Analysis.Calculus.VectorField

/-!
# Flat-Shore connection and coordinate curvature

The pinned Mathlib has Riemannian metric bundles but no connection or curvature
API. This module therefore works in the single global coordinate chart (r,y).
It defines the covariant differential using the actual Fréchet derivative,
checks torsion and metric compatibility, and evaluates its curvature commutator
on the constant coordinate fields. The coordinate metric is explicit; the
default product norm is used only for ordinary derivative calculus.

This does not construct a Mathlib RiemannianBundle curvature tensor, establish
coordinate-change naturality, or formalize Jacobi fields or cut-locus results.
-/

namespace FractalSync.FlatShoreGeometry

abbrev Coord := ℝ × ℝ

noncomputable def radialMetric (epsilon a r : ℝ) : ℝ :=
  (epsilon ^ 2 + (1 + a ^ 2) * r ^ 2) / (epsilon ^ 2 + r ^ 2) ^ 2

noncomputable def tangentialMetric (epsilon r : ℝ) : ℝ := 1 / (epsilon ^ 2 + r ^ 2)

/-- The actual flat-Shore bilinear metric in its global signed-normal chart. -/
noncomputable def metric (epsilon a r : ℝ) (v w : Coord) : ℝ :=
  radialMetric epsilon a r * v.1 * w.1 + tangentialMetric epsilon r * v.2 * w.2

noncomputable def gammaRR (epsilon a r : ℝ) : ℝ :=
  (1 + a ^ 2) * r / (epsilon ^ 2 + (1 + a ^ 2) * r ^ 2) -
    2 * r / (epsilon ^ 2 + r ^ 2)

noncomputable def gammaYY (epsilon a r : ℝ) : ℝ :=
  r / (epsilon ^ 2 + (1 + a ^ 2) * r ^ 2)

noncomputable def gammaRY (epsilon r : ℝ) : ℝ := -r / (epsilon ^ 2 + r ^ 2)

/-- The symmetric coordinate connection coefficients, evaluated bilinearly. -/
noncomputable def christoffel (epsilon a r : ℝ) (v w : Coord) : Coord :=
  (gammaRR epsilon a r * v.1 * w.1 + gammaYY epsilon a r * v.2 * w.2,
    gammaRY epsilon r * (v.1 * w.2 + v.2 * w.1))

theorem radialMetric_pos (epsilon a r : ℝ) (he : 0 < epsilon) :
    0 < radialMetric epsilon a r := by
  exact div_pos (FlatShore.normal_sq_pos epsilon a r he)
    (sq_pos_of_pos (FlatShore.ruler_sq_pos epsilon r he))

theorem tangentialMetric_pos (epsilon r : ℝ) (he : 0 < epsilon) :
    0 < tangentialMetric epsilon r := by
  exact div_pos zero_lt_one (FlatShore.ruler_sq_pos epsilon r he)

theorem metric_symm (epsilon a r : ℝ) (v w : Coord) :
    metric epsilon a r v w = metric epsilon a r w v := by
  simp only [metric, mul_left_comm, mul_comm]

theorem metric_self_pos (epsilon a r : ℝ) (v : Coord) (he : 0 < epsilon) (hv : v ≠ 0) :
    0 < metric epsilon a r v v := by
  by_cases hfirst : v.1 = 0
  · have hsecond : v.2 ≠ 0 := fun hs => hv (Prod.ext hfirst hs)
    simpa only [metric, hfirst, mul_zero, zero_add, pow_two, mul_assoc] using
      mul_pos (tangentialMetric_pos epsilon r he) (sq_pos_of_ne_zero hsecond)
  · simpa only [metric, pow_two, mul_assoc] using
      add_pos_of_pos_of_nonneg
        (mul_pos (radialMetric_pos epsilon a r he) (sq_pos_of_ne_zero hfirst))
        (mul_nonneg (tangentialMetric_pos epsilon r he).le (sq_nonneg v.2))

/-- The radial metric coefficient is actually differentiated, with its
logarithmic derivative identified with the radial Christoffel coefficient. -/
theorem hasDerivAt_radialMetric (epsilon a r : ℝ) (he : 0 < epsilon) :
    HasDerivAt (radialMetric epsilon a)
      (2 * radialMetric epsilon a r * gammaRR epsilon a r) r := by
  convert ((((hasDerivAt_id r).pow 2).const_mul (1 + a ^ 2)).const_add
    (epsilon ^ 2)).div ((((hasDerivAt_id r).pow 2).const_add (epsilon ^ 2)).pow 2)
      (pow_ne_zero 2 (FlatShore.ruler_sq_pos epsilon r he).ne') using 1
  norm_num [radialMetric, gammaRR]
  field_simp [(FlatShore.ruler_sq_pos epsilon r he).ne',
    (FlatShore.normal_sq_pos epsilon a r he).ne']

theorem hasDerivAt_tangentialMetric (epsilon r : ℝ) (he : 0 < epsilon) :
    HasDerivAt (tangentialMetric epsilon)
      (2 * tangentialMetric epsilon r * gammaRY epsilon r) r := by
  convert (((hasDerivAt_id r).pow 2).const_add (epsilon ^ 2)).inv
    (FlatShore.ruler_sq_pos epsilon r he).ne' using 1
  all_goals norm_num [tangentialMetric, gammaRY, one_div]
  · exact funext fun s => one_div (epsilon ^ 2 + s ^ 2)
  · field_simp [(FlatShore.ruler_sq_pos epsilon r he).ne']

theorem hasDerivAt_gammaYY (epsilon a r : ℝ) (he : 0 < epsilon) :
    HasDerivAt (gammaYY epsilon a)
      ((epsilon ^ 2 - (1 + a ^ 2) * r ^ 2) /
        (epsilon ^ 2 + (1 + a ^ 2) * r ^ 2) ^ 2) r := by
  convert (hasDerivAt_id r).div
    ((((hasDerivAt_id r).pow 2).const_mul (1 + a ^ 2)).const_add (epsilon ^ 2))
      (FlatShore.normal_sq_pos epsilon a r he).ne' using 1
  norm_num
  ring

theorem hasDerivAt_gammaRY (epsilon r : ℝ) (he : 0 < epsilon) :
    HasDerivAt (gammaRY epsilon)
      ((r ^ 2 - epsilon ^ 2) / (epsilon ^ 2 + r ^ 2) ^ 2) r := by
  convert (hasDerivAt_id r).neg.div
    (((hasDerivAt_id r).pow 2).const_add (epsilon ^ 2))
      (FlatShore.ruler_sq_pos epsilon r he).ne' using 1
  norm_num
  ring

/-- Symmetry in the two input directions is the zero-torsion coefficient identity. -/
theorem christoffel_symm (epsilon a r : ℝ) (v w : Coord) :
    christoffel epsilon a r v w = christoffel epsilon a r w v := by
  simp only [christoffel, mul_comm, mul_left_comm, add_comm]

noncomputable def metricRate (epsilon a r : ℝ) (d v w : Coord) : ℝ :=
  2 * radialMetric epsilon a r * gammaRR epsilon a r * d.1 * v.1 * w.1 +
    2 * tangentialMetric epsilon r * gammaRY epsilon r * d.1 * v.2 * w.2

/-- Compatibility with the actual coordinate differential of the metric. -/
theorem christoffel_metric_compatible (epsilon a r : ℝ) (d v w : Coord) (he : 0 < epsilon) :
    metricRate epsilon a r d v w =
      metric epsilon a r (christoffel epsilon a r d v) w +
        metric epsilon a r v (christoffel epsilon a r d w) := by
  dsimp only [metricRate, metric, christoffel, radialMetric, tangentialMetric,
    gammaRR, gammaYY, gammaRY]
  field_simp [(FlatShore.ruler_sq_pos epsilon r he).ne',
    (FlatShore.normal_sq_pos epsilon a r he).ne']
  ring

/-- Actual metric-pairing differentiation along a curve with varying tangent
vectors. The scalar metric derivatives above discharge the field calculus. -/
theorem hasDerivAt_metric_pairing (epsilon a t : ℝ) (r : ℝ → ℝ)
    (v w : ℝ → Coord) (d dv dw : Coord) (he : 0 < epsilon)
    (hr : HasDerivAt r d.1 t) (hv : HasDerivAt v dv t) (hw : HasDerivAt w dw t) :
    HasDerivAt (fun s => metric epsilon a (r s) (v s) (w s))
      (metricRate epsilon a (r t) d (v t) (w t) +
        metric epsilon a (r t) dv (w t) + metric epsilon a (r t) (v t) dw) t := by
  have hv1 := (ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt.comp_hasDerivAt t hv
  have hv2 := (ContinuousLinearMap.snd ℝ ℝ ℝ).hasFDerivAt.comp_hasDerivAt t hv
  have hw1 := (ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt.comp_hasDerivAt t hw
  have hw2 := (ContinuousLinearMap.snd ℝ ℝ ℝ).hasFDerivAt.comp_hasDerivAt t hw
  convert ((((hasDerivAt_radialMetric epsilon a (r t) he).comp t hr).mul hv1).mul hw1).add
    ((((hasDerivAt_tangentialMetric epsilon (r t) he).comp t hr).mul hv2).mul hw2) using 1
  dsimp [metricRate, metric]
  ring

/-- The coordinate covariant differential uses the actual field derivative. -/
noncomputable def covariantDerivative (epsilon a : ℝ) (X Y : Coord → Coord) (x : Coord) : Coord :=
  fderiv ℝ Y x (X x) + christoffel epsilon a x.1 (X x) (Y x)

/-- The actual covariant differential has zero torsion relative to Mathlib's
Lie bracket. The coordinate cancellation remains valid for total derivatives. -/
theorem covariantDerivative_torsion (epsilon a : ℝ) (X Y : Coord → Coord) (x : Coord) :
    covariantDerivative epsilon a X Y x - covariantDerivative epsilon a Y X x =
      VectorField.lieBracket ℝ X Y x := by
  rw [covariantDerivative, covariantDerivative,
    christoffel_symm epsilon a x.1 (Y x) (X x), VectorField.lieBracket]
  abel

/-- The covariant differential is pointwise linear in its direction field,
including arbitrary scalar functions in that first slot. -/
theorem covariantDerivative_first_linear (epsilon a : ℝ) (f g : Coord → ℝ)
    (X Y Z : Coord → Coord) (x : Coord) :
    covariantDerivative epsilon a (fun p => f p • X p + g p • Z p) Y x =
      f x • covariantDerivative epsilon a X Y x +
        g x • covariantDerivative epsilon a Z Y x := by
  simp only [covariantDerivative, map_add, map_smul]
  apply Prod.ext
  all_goals simp only [christoffel, Prod.fst_add, Prod.snd_add, Prod.smul_fst, Prod.smul_snd,
    smul_eq_mul]
  all_goals ring

theorem covariantDerivative_second_add (epsilon a : ℝ) (X Y Z : Coord → Coord)
    (x : Coord) (hY : DifferentiableAt ℝ Y x) (hZ : DifferentiableAt ℝ Z x) :
    covariantDerivative epsilon a X (Y + Z) x =
      covariantDerivative epsilon a X Y x + covariantDerivative epsilon a X Z x := by
  simp only [covariantDerivative, fderiv_add hY hZ, ContinuousLinearMap.add_apply, Pi.add_apply]
  apply Prod.ext
  all_goals simp only [christoffel, Prod.fst_add, Prod.snd_add]
  all_goals ring

/-- The Leibniz rule in the differentiated field slot uses the actual scalar
and vector derivatives, under their necessary differentiability premises. -/
theorem covariantDerivative_leibniz (epsilon a : ℝ) (f : Coord → ℝ)
    (X Y : Coord → Coord) (x : Coord)
    (hf : DifferentiableAt ℝ f x) (hY : DifferentiableAt ℝ Y x) :
    covariantDerivative epsilon a X (fun p => f p • Y p) x =
      (fderiv ℝ f x (X x)) • Y x + f x • covariantDerivative epsilon a X Y x := by
  simp only [covariantDerivative, fderiv_fun_smul hf hY,
    ContinuousLinearMap.add_apply, ContinuousLinearMap.smul_apply,
    ContinuousLinearMap.smulRight_apply]
  apply Prod.ext
  all_goals simp only [christoffel, Prod.fst_add, Prod.snd_add, Prod.smul_fst, Prod.smul_snd,
    smul_eq_mul]
  all_goals ring

/-- Genuine metric compatibility for the coordinate covariant differential:
along a curve tangent to X, differentiate the pairing of two actual fields. -/
theorem covariantDerivative_metric_pairing (epsilon a t : ℝ)
    (X Y Z : Coord → Coord) (gamma : ℝ → Coord) (he : 0 < epsilon)
    (hgamma : HasDerivAt gamma (X (gamma t)) t)
    (hY : DifferentiableAt ℝ Y (gamma t)) (hZ : DifferentiableAt ℝ Z (gamma t)) :
    HasDerivAt (fun s => metric epsilon a (gamma s).1 (Y (gamma s)) (Z (gamma s)))
      (metric epsilon a (gamma t).1 (covariantDerivative epsilon a X Y (gamma t)) (Z (gamma t)) +
        metric epsilon a (gamma t).1 (Y (gamma t)) (covariantDerivative epsilon a X Z (gamma t))) t := by
  convert hasDerivAt_metric_pairing epsilon a t (fun s => (gamma s).1)
    (Y ∘ gamma) (Z ∘ gamma) (X (gamma t))
    (fderiv ℝ Y (gamma t) (X (gamma t))) (fderiv ℝ Z (gamma t) (X (gamma t))) he
    ((ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt.comp_hasDerivAt t hgamma)
    (hY.hasFDerivAt.comp_hasDerivAt t hgamma)
    (hZ.hasFDerivAt.comp_hasDerivAt t hgamma) using 1
  rw [christoffel_metric_compatible epsilon a (gamma t).1 _ _ _ he]
  simp only [covariantDerivative, metric, Prod.fst_add, Prod.snd_add, Function.comp_apply]
  ring

/-- The two constant fields in the global (r,y) chart. -/
def normalField (_ : Coord) : Coord := (1, 0)

def tangentField (_ : Coord) : Coord := (0, 1)

theorem covariantDerivative_normal_tangent (epsilon a : ℝ) :
    covariantDerivative epsilon a normalField tangentField =
      fun x => (0, gammaRY epsilon x.1) := by
  funext x
  simp [covariantDerivative, normalField, tangentField, christoffel]
  exact congrArg (fun L : Coord →L[ℝ] Coord => L (1, 0)) (fderiv_const_apply (x := x) (0, 1))

theorem covariantDerivative_tangent_tangent (epsilon a : ℝ) :
    covariantDerivative epsilon a tangentField tangentField =
      fun x => (gammaYY epsilon a x.1, 0) := by
  funext x
  simp [covariantDerivative, tangentField, christoffel]
  exact congrArg (fun L : Coord →L[ℝ] Coord => L (0, 1)) (fderiv_const_apply (x := x) (0, 1))

theorem fderiv_gammaYY_normal (epsilon a : ℝ) (x : Coord) (he : 0 < epsilon) :
    fderiv ℝ (fun p : Coord => (gammaYY epsilon a p.1, (0 : ℝ))) x (1, 0) =
      ((epsilon ^ 2 - (1 + a ^ 2) * x.1 ^ 2) /
        (epsilon ^ 2 + (1 + a ^ 2) * x.1 ^ 2) ^ 2, 0) := by
  have hd := ((hasDerivAt_gammaYY epsilon a x.1 he).comp_hasFDerivAt x
    (ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt).prodMk (hasFDerivAt_const (0 : ℝ) x)
  simpa using congrArg (fun L : Coord →L[ℝ] Coord => L (1, 0)) hd.fderiv

theorem fderiv_gammaRY_tangent (epsilon : ℝ) (x : Coord) (he : 0 < epsilon) :
    fderiv ℝ (fun p : Coord => ((0 : ℝ), gammaRY epsilon p.1)) x (0, 1) = 0 := by
  have hd := (hasFDerivAt_const (0 : ℝ) x).prodMk
    ((hasDerivAt_gammaRY epsilon x.1 he).comp_hasFDerivAt x
      (ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt)
  simpa using congrArg (fun L : Coord →L[ℝ] Coord => L (0, 1)) hd.fderiv

/-- The curvature commutator of the actual coordinate covariant differential.
The sign convention is R(X,Y)Z = nabla_X nabla_Y Z - nabla_Y nabla_X Z - nabla_[X,Y] Z. -/
noncomputable def curvature (epsilon a : ℝ) (X Y Z : Coord → Coord) (x : Coord) : Coord :=
  covariantDerivative epsilon a X (covariantDerivative epsilon a Y Z) x -
    covariantDerivative epsilon a Y (covariantDerivative epsilon a X Z) x -
    covariantDerivative epsilon a (VectorField.lieBracket ℝ X Y) Z x

theorem lieBracket_normal_tangent : VectorField.lieBracket ℝ normalField tangentField = 0 := by
  unfold VectorField.lieBracket normalField tangentField
  simp
  rfl

/-- The curvature commutator is computed from actual derivatives of the two
nonconstant first covariant derivatives, without assuming a curvature formula. -/
theorem curvature_normal_tangent_tangent (epsilon a : ℝ) (x : Coord) (he : 0 < epsilon) :
    curvature epsilon a normalField tangentField tangentField x =
      ((epsilon ^ 2 - (1 + a ^ 2) * x.1 ^ 2) /
          (epsilon ^ 2 + (1 + a ^ 2) * x.1 ^ 2) ^ 2 +
        gammaRR epsilon a x.1 * gammaYY epsilon a x.1 -
        gammaYY epsilon a x.1 * gammaRY epsilon x.1, 0) := by
  rw [curvature, covariantDerivative_tangent_tangent, covariantDerivative_normal_tangent,
    lieBracket_normal_tangent]
  simp only [covariantDerivative, normalField, tangentField, Pi.zero_apply]
  rw [fderiv_gammaYY_normal epsilon a x he, fderiv_gammaRY_tangent epsilon x he]
  simp [christoffel]

/-- The actual commutator component equals the previously checked scalar
curvature times the tangential metric coefficient. -/
theorem curvature_normal_tangent_tangent_eq (epsilon a : ℝ) (x : Coord) (he : 0 < epsilon) :
    curvature epsilon a normalField tangentField tangentField x =
      (tangentialMetric epsilon x.1 * FlatShore.radialCurvature epsilon a x.1, 0) := by
  rw [curvature_normal_tangent_tangent epsilon a x he]
  congr 1
  dsimp only [gammaRR, gammaYY, gammaRY, tangentialMetric, FlatShore.radialCurvature]
  field_simp [(FlatShore.ruler_sq_pos epsilon x.1 he).ne',
    (FlatShore.normal_sq_pos epsilon a x.1 he).ne']
  ring

/-- Sectional curvature in the global coordinate two-plane, defined from the
actual curvature commutator and its metric Gram determinant. -/
noncomputable def sectionalCurvature (epsilon a : ℝ) (x : Coord) : ℝ :=
  metric epsilon a x.1 (curvature epsilon a normalField tangentField tangentField x)
    (normalField x) /
    (metric epsilon a x.1 (normalField x) (normalField x) *
      metric epsilon a x.1 (tangentField x) (tangentField x) -
      metric epsilon a x.1 (normalField x) (tangentField x) ^ 2)

/-- The coordinate sectional curvature, obtained from the actual covariant
commutator, is the flat-Shore rational expression. This fixes the convention
in which Shore curvature is positive one. -/
theorem sectionalCurvature_eq_radialCurvature (epsilon a : ℝ) (x : Coord) (he : 0 < epsilon) :
    sectionalCurvature epsilon a x = FlatShore.radialCurvature epsilon a x.1 := by
  simp only [sectionalCurvature, curvature_normal_tangent_tangent_eq epsilon a x he,
    metric, normalField, tangentField, mul_zero, mul_one, add_zero, zero_add, zero_pow two_ne_zero,
    sub_zero]
  field_simp [(radialMetric_pos epsilon a x.1 he).ne', (tangentialMetric_pos epsilon x.1 he).ne']

/-- Flat Shore has coordinate sectional curvature positive one at every
tangential location, independently of epsilon and the graph parameter. -/
theorem sectionalCurvature_on_shore (epsilon a y : ℝ) (he : 0 < epsilon) :
    sectionalCurvature epsilon a (0, y) = 1 := by
  rw [sectionalCurvature_eq_radialCurvature epsilon a (0, y) he,
    FlatShore.radialCurvature_zero epsilon a he]

end FractalSync.FlatShoreGeometry
