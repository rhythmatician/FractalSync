import FractalSync.OperationalDynamics
import FractalSync.OperationalFieldHessianError
import FractalSync.RulerJetBounds

/-!
# Explicit Euclidean jet bounds for the literal operational ruler

The coefficient bound is on the actual lattice. All conclusions concern the
exact real floor-selected spline and the actual gradient and Hessian on the L2
plane. The coordinate map into the max-norm product is contractive, not an
isometry. These statements do not certify floating-point evaluation.
-/

noncomputable section

namespace FractalSync.OperationalDynamics

open OperationalField FlatShoreIntrinsic RulerJetBounds Set Filter
open scoped Topology

/-- The Euclidean-to-coordinate chart is contractive into the product max norm. -/
theorem norm_chart_apply_le (x : Plane) : ‖chart x‖ ≤ ‖x‖ := by
  exact max_le (WithLp.norm_fst_le ℝ x) (WithLp.norm_snd_le ℝ x)

theorem norm_chart_le : ‖chart.toContinuousLinearMap‖ ≤ 1 := by
  exact ContinuousLinearMap.opNorm_le_bound _ zero_le_one
    (fun x => by simpa only [one_mul] using norm_chart_apply_le x)

/-- Pull coordinate covectors back to Euclidean covectors. -/
def pullDifferential : (Coords →L[ℝ] ℝ) →L[ℝ] (Plane →L[ℝ] ℝ) :=
  (ContinuousLinearMap.compL ℝ Plane Coords ℝ).flip chart.toContinuousLinearMap

theorem norm_pullDifferential_le (d : Coords →L[ℝ] ℝ) : ‖pullDifferential d‖ ≤ ‖d‖ := by
  exact (ContinuousLinearMap.opNorm_comp_le d chart.toContinuousLinearMap).trans
    (by simpa only [mul_one] using mul_le_mul_of_nonneg_left norm_chart_le (norm_nonneg d))

theorem norm_pullDifferential_map_le : ‖pullDifferential‖ ≤ 1 := by
  exact ContinuousLinearMap.opNorm_le_bound _ zero_le_one
    (fun d => by simpa only [one_mul] using norm_pullDifferential_le d)

/-- Pull both arguments of a coordinate bilinear map into the Euclidean plane. -/
def pullBilinear : (Coords →L[ℝ] Coords →L[ℝ] ℝ) →L[ℝ] (Plane →L[ℝ] Plane →L[ℝ] ℝ) :=
  (ContinuousLinearMap.compL ℝ Plane (Coords →L[ℝ] ℝ) (Plane →L[ℝ] ℝ) pullDifferential).comp
    ((ContinuousLinearMap.compL ℝ Plane Coords (Coords →L[ℝ] ℝ)).flip chart.toContinuousLinearMap)

theorem norm_pullBilinear_le (H : Coords →L[ℝ] Coords →L[ℝ] ℝ) :
    ‖pullBilinear H‖ ≤ ‖H‖ := by
  refine (ContinuousLinearMap.opNorm_comp_le pullDifferential
    (H.comp chart.toContinuousLinearMap)).trans ?_
  calc
    _ ≤ 1 * (‖H‖ * ‖chart.toContinuousLinearMap‖) :=
      mul_le_mul norm_pullDifferential_map_le
        (ContinuousLinearMap.opNorm_comp_le H chart.toContinuousLinearMap)
        (norm_nonneg (H.comp chart.toContinuousLinearMap)) zero_le_one
    _ ≤ ‖H‖ := by
      simpa only [one_mul, mul_one] using mul_le_mul_of_nonneg_left norm_chart_le (norm_nonneg H)

def euclideanDifferential (C : Lattice) (o : Coords) (h : ℝ) (x : Plane) : Plane →L[ℝ] ℝ :=
  pullDifferential (physicalDifferential C o h (chart x))

def euclideanHessianForm (C : Lattice) (o : Coords) (h : ℝ) (x : Plane) :
    Plane →L[ℝ] Plane →L[ℝ] ℝ :=
  pullBilinear (physicalHessian C o h (chart x))

theorem hasFDerivAt_euclideanDifferential (C : Lattice) (o : Coords) (h : ℝ) (x : Plane) :
    HasFDerivAt (euclideanDifferential C o h) (euclideanHessianForm C o h x) x := by
  exact pullDifferential.hasFDerivAt.comp x
    ((hasFDerivAt_physicalDifferential C o (chart x) h).comp x chart.hasFDerivAt)

/-- The actual ruler's second-chain-rule bilinear map on the Euclidean plane. -/
def rulerBilinear (C : Lattice) (o : Coords) (h epsilon : ℝ) (x : Plane) :
    Plane →L[ℝ] Plane →L[ℝ] ℝ :=
  regularizedForm epsilon (physicalField C o h (chart x))
    (euclideanDifferential C o h x) (euclideanHessianForm C o h x)

theorem hasFDerivAt_fderiv_rulerField (C : Lattice) (o : Coords) (h epsilon : ℝ)
    (he : 0 < epsilon) (x : Plane) :
    HasFDerivAt (fderiv ℝ (rulerField C o h epsilon)) (rulerBilinear C o h epsilon x) x := by
  apply (hasFDerivAt_ruler_differential epsilon he
    (fun y => physicalField C o h (chart y)) (euclideanDifferential C o h)
    (euclideanHessianForm C o h x) x
    ((hasFDerivAt_physicalField C o (chart x) h).comp x chart.hasFDerivAt)
    (hasFDerivAt_euclideanDifferential C o h x)).congr_of_eventuallyEq
  exact Filter.Eventually.of_forall (fun y => (hasFDerivAt_rulerField C o h epsilon he y).fderiv)

/-- Riesz dualization is an isometry for the actual Euclidean norm. -/
def rieszMap : (Plane →L[ℝ] ℝ) →L[ℝ] Plane :=
  (InnerProductSpace.toDual ℝ Plane).symm.toContinuousLinearEquiv.toContinuousLinearMap

def rieszHessian : (Plane →L[ℝ] Plane →L[ℝ] ℝ) →L[ℝ] (Plane →L[ℝ] Plane) :=
  ContinuousLinearMap.compL ℝ Plane (Plane →L[ℝ] ℝ) Plane rieszMap

theorem hasFDerivAt_gradient_rulerField (C : Lattice) (o : Coords) (h epsilon : ℝ)
    (he : 0 < epsilon) (x : Plane) :
    HasFDerivAt (gradient (rulerField C o h epsilon))
      (rieszHessian (rulerBilinear C o h epsilon x)) x := by
  exact rieszMap.hasFDerivAt.comp x (hasFDerivAt_fderiv_rulerField C o h epsilon he x)

theorem norm_rieszHessian_le (H : Plane →L[ℝ] Plane →L[ℝ] ℝ) :
    ‖rieszHessian H‖ ≤ ‖H‖ := by
  apply ContinuousLinearMap.opNorm_le_bound _ (norm_nonneg H)
  intro v
  change ‖(InnerProductSpace.toDual ℝ Plane).symm (H v)‖ ≤ ‖H‖ * ‖v‖
  simpa only [(InnerProductSpace.toDual ℝ Plane).symm.norm_map] using H.le_opNorm v

def splineGradientBound (B h : ℝ) : ℝ := (8/3 : ℝ)*B/h

def splineHessianBound (B h : ℝ) : ℝ := (68/9 : ℝ)*B/h^2

def splineThirdBound (B h : ℝ) : ℝ := 32*B/h^3

def rulerHessianBound (B h epsilon : ℝ) : ℝ :=
  splineHessianBound B h + (splineGradientBound B h)^2/epsilon

def rulerThirdBound (B h epsilon : ℝ) : ℝ :=
  splineThirdBound B h +
    3*splineGradientBound B h*splineHessianBound B h/epsilon +
    3*(splineGradientBound B h)^3/epsilon^2

theorem physicalField_abs_le (C : Lattice) (o q : Coords) (h B : ℝ)
    (hh : 0 < h) (hB : 0 ≤ B) (hC : ∀ i j, |C i j| ≤ B) :
    |physicalField C o h q| ≤ B := by
  simpa [physicalField, jetBound] using
    physicalJet_abs_le 0 0 (by norm_num) (by norm_num) C o q h B hh hB hC

theorem rulerField_bounds (C : Lattice) (o : Coords) (h epsilon B : ℝ)
    (hh : 0 < h) (he : 0 < epsilon) (hB : 0 ≤ B) (hC : ∀ i j, |C i j| ≤ B)
    (x : Plane) :
    epsilon ≤ rulerField C o h epsilon x ∧ rulerField C o h epsilon x ≤ ruler epsilon B := by
  refine ⟨epsilon_le_ruler epsilon _ he.le, ?_⟩
  exact Real.sqrt_le_sqrt (add_le_add_right (sq_le_sq.2
    (by simpa only [abs_of_nonneg hB] using physicalField_abs_le C o (chart x) h B hh hB hC)) _)

theorem norm_euclideanDifferential_le (C : Lattice) (o : Coords) (h B : ℝ)
    (hh : 0 < h) (hB : 0 ≤ B) (hC : ∀ i j, |C i j| ≤ B) (x : Plane) :
    ‖euclideanDifferential C o h x‖ ≤ splineGradientBound B h := by
  exact (norm_pullDifferential_le _).trans
    (norm_physicalDifferential_le C o (chart x) h B hh hB hC)

theorem norm_euclideanHessianForm_le (C : Lattice) (o : Coords) (h B : ℝ)
    (hh : 0 < h) (hB : 0 ≤ B) (hC : ∀ i j, |C i j| ≤ B) (x : Plane) :
    ‖euclideanHessianForm C o h x‖ ≤ splineHessianBound B h := by
  exact (norm_pullBilinear_le _).trans
    (norm_physicalHessian_le C o (chart x) h B hh hB hC)

theorem norm_gradient_rulerField_le (C : Lattice) (o : Coords) (h epsilon B : ℝ)
    (hh : 0 < h) (he : 0 < epsilon) (hB : 0 ≤ B) (hC : ∀ i j, |C i j| ≤ B) (x : Plane) :
    ‖gradient (rulerField C o h epsilon) x‖ ≤ splineGradientBound B h := by
  rw [gradient, (InnerProductSpace.toDual ℝ Plane).symm.norm_map,
    (hasFDerivAt_rulerField C o h epsilon he x).fderiv, norm_smul, Real.norm_eq_abs]
  simpa only [one_mul] using mul_le_mul
    (ruler_slope_abs_le epsilon (physicalField C o h (chart x)) he)
    (norm_euclideanDifferential_le C o h B hh hB hC x)
    (norm_nonneg (euclideanDifferential C o h x)) zero_le_one

theorem norm_hessian_rulerField_le (C : Lattice) (o : Coords) (h epsilon B : ℝ)
    (hh : 0 < h) (he : 0 < epsilon) (hB : 0 ≤ B) (hC : ∀ i j, |C i j| ≤ B) (x : Plane) :
    ‖fderiv ℝ (gradient (rulerField C o h epsilon)) x‖ ≤ rulerHessianBound B h epsilon := by
  rw [(hasFDerivAt_gradient_rulerField C o h epsilon he x).fderiv]
  exact (norm_rieszHessian_le _).trans
    (norm_regularizedForm_le epsilon _ _ _ he _ _
      (norm_euclideanDifferential_le C o h B hh hB hC x)
      (norm_euclideanHessianForm_le C o h B hh hB hC x))

theorem rulerField_sub_le (C : Lattice) (o : Coords) (h epsilon B : ℝ)
    (hh : 0 < h) (he : 0 < epsilon) (hB : 0 ≤ B) (hC : ∀ i j, |C i j| ≤ B) (x y : Plane) :
    |rulerField C o h epsilon y - rulerField C o h epsilon x| ≤
      splineGradientBound B h * ‖y-x‖ := by
  simpa only [Real.norm_eq_abs] using Convex.norm_image_sub_le_of_norm_fderiv_le
    (fun z (_ : z ∈ (univ : Set Plane)) => (hasFDerivAt_rulerField C o h epsilon he z).differentiableAt)
    (fun z _ => by
      simpa only [gradient, (InnerProductSpace.toDual ℝ Plane).symm.norm_map] using
        norm_gradient_rulerField_le C o h epsilon B hh he hB hC z)
    convex_univ (mem_univ x) (mem_univ y)

theorem gradient_rulerField_sub_le (C : Lattice) (o : Coords) (h epsilon B : ℝ)
    (hh : 0 < h) (he : 0 < epsilon) (hB : 0 ≤ B) (hC : ∀ i j, |C i j| ≤ B) (x y : Plane) :
    ‖gradient (rulerField C o h epsilon) y - gradient (rulerField C o h epsilon) x‖ ≤
      rulerHessianBound B h epsilon * ‖y-x‖ := by
  exact Convex.norm_image_sub_le_of_norm_fderiv_le
    (fun z (_ : z ∈ (univ : Set Plane)) =>
      (hasFDerivAt_gradient_rulerField C o h epsilon he z).differentiableAt)
    (fun z _ => norm_hessian_rulerField_le C o h epsilon B hh he hB hC z)
    convex_univ (mem_univ x) (mem_univ y)

theorem euclideanField_sub_le (C : Lattice) (o : Coords) (h B : ℝ)
    (hh : 0 < h) (hB : 0 ≤ B) (hC : ∀ i j, |C i j| ≤ B) (x y : Plane) :
    |physicalField C o h (chart y) - physicalField C o h (chart x)| ≤
      splineGradientBound B h * ‖y-x‖ := by
  refine (physicalField_query_error_bound C o (chart x) (chart y) h B hh hB hC).trans ?_
  exact mul_le_mul_of_nonneg_left
    (by simpa only [map_sub] using norm_chart_apply_le (y-x)) (by positivity)

theorem euclideanDifferential_sub_le (C : Lattice) (o : Coords) (h B : ℝ)
    (hh : 0 < h) (hB : 0 ≤ B) (hC : ∀ i j, |C i j| ≤ B) (x y : Plane) :
    ‖euclideanDifferential C o h y - euclideanDifferential C o h x‖ ≤
      splineHessianBound B h * ‖y-x‖ := by
  exact Convex.norm_image_sub_le_of_norm_hasFDerivWithin_le
    (fun z (_ : z ∈ (univ : Set Plane)) => (hasFDerivAt_euclideanDifferential C o h z).hasFDerivWithinAt)
    (fun z _ => norm_euclideanHessianForm_le C o h B hh hB hC z)
    convex_univ (mem_univ x) (mem_univ y)

theorem euclideanHessianForm_sub_le (C : Lattice) (o : Coords) (h B : ℝ)
    (hh : 0 < h) (hB : 0 ≤ B) (hC : ∀ i j, |C i j| ≤ B) (x y : Plane) :
    ‖euclideanHessianForm C o h y - euclideanHessianForm C o h x‖ ≤
      splineThirdBound B h * ‖y-x‖ := by
  change ‖pullBilinear (physicalHessian C o h (chart y)) -
    pullBilinear (physicalHessian C o h (chart x))‖ ≤ _
  rw [← map_sub]
  refine (norm_pullBilinear_le _).trans
    ((physicalHessian_query_error_bound C o (chart x) (chart y) h B hh hB hC).trans ?_)
  exact mul_le_mul_of_nonneg_left
    (by simpa only [map_sub] using norm_chart_apply_le (y-x)) (by positivity)

theorem rulerBilinear_sub_le (C : Lattice) (o : Coords) (h epsilon B : ℝ)
    (hh : 0 < h) (he : 0 < epsilon) (hB : 0 ≤ B) (hC : ∀ i j, |C i j| ≤ B) (x y : Plane) :
    ‖rulerBilinear C o h epsilon y - rulerBilinear C o h epsilon x‖ ≤
      rulerThirdBound B h epsilon * ‖y-x‖ := by
  exact regularizedForm_sub_le epsilon _ _
    (splineGradientBound B h) (splineHessianBound B h) (splineThirdBound B h) ‖y-x‖
    he (by unfold splineGradientBound; positivity) (norm_nonneg _) _ _ _ _
    (norm_euclideanDifferential_le C o h B hh hB hC x)
    (norm_euclideanDifferential_le C o h B hh hB hC y)
    (norm_euclideanHessianForm_le C o h B hh hB hC x)
    (euclideanField_sub_le C o h B hh hB hC x y)
    (euclideanDifferential_sub_le C o h B hh hB hC x y)
    (euclideanHessianForm_sub_le C o h B hh hB hC x y)

theorem hessian_rulerField_sub_le (C : Lattice) (o : Coords) (h epsilon B : ℝ)
    (hh : 0 < h) (he : 0 < epsilon) (hB : 0 ≤ B) (hC : ∀ i j, |C i j| ≤ B) (x y : Plane) :
    ‖fderiv ℝ (gradient (rulerField C o h epsilon)) y -
      fderiv ℝ (gradient (rulerField C o h epsilon)) x‖ ≤
      rulerThirdBound B h epsilon * ‖y-x‖ := by
  rw [(hasFDerivAt_gradient_rulerField C o h epsilon he y).fderiv,
    (hasFDerivAt_gradient_rulerField C o h epsilon he x).fderiv, ← map_sub]
  exact (norm_rieszHessian_le _).trans (rulerBilinear_sub_le C o h epsilon B hh he hB hC x y)

/-- The third ruler differential is the genuine derivative of the actual
Riesz Hessian field; C2 regularity rules out a totalized-derivative shortcut. -/
theorem actual_third_rulerField_bound (C : Lattice) (o : Coords) (h epsilon B : ℝ)
    (hh : 0 < h) (he : 0 < epsilon) (hB : 0 ≤ B) (hC : ∀ i j, |C i j| ≤ B) (x : Plane) :
    HasFDerivAt (fderiv ℝ (gradient (rulerField C o h epsilon)))
      (fderiv ℝ (fderiv ℝ (gradient (rulerField C o h epsilon))) x) x ∧
    ‖fderiv ℝ (fderiv ℝ (gradient (rulerField C o h epsilon))) x‖ ≤
      rulerThirdBound B h epsilon := by
  refine ⟨((actual_jet_regularity C o h epsilon he).2.differentiable (by norm_num) x).hasFDerivAt, ?_⟩
  apply norm_fderiv_le_of_lip' ℝ
  · unfold rulerThirdBound splineThirdBound splineHessianBound splineGradientBound
    positivity
  · exact Filter.Eventually.of_forall (fun y => hessian_rulerField_sub_le C o h epsilon B hh he hB hC x y)

end FractalSync.OperationalDynamics
