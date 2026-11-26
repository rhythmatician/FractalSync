import FractalSync.CoordinateCovariance

/-!
# Actual nonlinear coordinate covariance

The transformed connection contains the actual second Fréchet derivative of
the coordinate map. Vector fields use Mathlib's actual derivative-inverse
pullback. The connection and curvature are recomputed using the existing
actual coordinate differential and commutator.
-/

namespace FractalSync.NonlinearCoordinates

open CoordinateCovariance
open scoped Topology

variable {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [CompleteSpace E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

/-- Connection coefficients in a nonlinear coordinate map, including its
actual second derivative. The geometric domain requires an invertible first derivative. -/
noncomputable def pullbackConnection (f : E → F) (Gamma : F → F → F → F)
    (x v w : E) : E :=
  (fderiv ℝ f x).inverse (fderiv ℝ (fderiv ℝ f) x v w +
    Gamma (f x) (fderiv ℝ f x v) (fderiv ℝ f x w))

/-- Pullback of the pointwise metric by the actual derivative of the coordinate map. -/
noncomputable def pullbackMetric (f : E → F) (g : F → F → F → ℝ)
    (x v w : E) : ℝ := g (f x) (fderiv ℝ f x v) (fderiv ℝ f x w)

/-- The actual derivative of a transported vector field includes the
second derivative of the nonlinear coordinate map. -/
theorem fderiv_pullback_apply (f : E → F) (Y : F → F) (x v : E)
    (hf : ContDiffAt ℝ 2 f x) (hi : (fderiv ℝ f x).IsInvertible)
    (hY : DifferentiableAt ℝ Y (f x)) :
    fderiv ℝ (VectorField.pullback ℝ f Y) x v =
      (fderiv ℝ f x).inverse
        (fderiv ℝ Y (f x) (fderiv ℝ f x v) -
          fderiv ℝ (fderiv ℝ f) x v (VectorField.pullback ℝ f Y x)) := by
  rcases exists_continuousLinearEquiv_fderiv_symm_eq hf hi with
    ⟨M, _, hMs, hM, hMd⟩
  have hMx : (M x : E →L[ℝ] F) = fderiv ℝ f x := hM.self_of_nhds
  have hEq : VectorField.pullback ℝ f Y =ᶠ[𝓝 x]
      (fun y => (M y).symm (Y (f y))) := by
    filter_upwards [hM] with y hy
    exact VectorField.pullback_eq_of_fderiv_eq hy Y
  rw [hEq.fderiv_eq]
  change (fderiv ℝ (fun y => ((M y).symm : F →L[ℝ] E) ((Y ∘ f) y)) x) v = _
  rw [fderiv_clm_apply (hMs.differentiableAt le_rfl)
    (hY.comp x (hf.differentiableAt one_le_two))]
  simp [fderiv_comp x hY (hf.differentiableAt one_le_two), hMd, ← hMx,
    VectorField.pullback, sub_eq_add_neg]


/-- The actual nonlinear covariant differential transforms after cancellation
of the coordinate-map Hessian and the derivative of its inverse Jacobian. -/
theorem covariantDerivative_pullback (f : E → F) (Gamma : F → F → F → F)
    (X Y : F → F) (x : E) (hf : ContDiffAt ℝ 2 f x)
    (hi : (fderiv ℝ f x).IsInvertible) (hY : DifferentiableAt ℝ Y (f x)) :
    covariantDerivative (pullbackConnection f Gamma)
      (VectorField.pullback ℝ f X) (VectorField.pullback ℝ f Y) x =
      VectorField.pullback ℝ f (covariantDerivative Gamma X Y) x := by
  rw [covariantDerivative, fderiv_pullback_apply f Y x _ hf hi hY]
  rcases hi with ⟨M, hM⟩
  simp only [pullbackConnection, VectorField.pullback, ← hM, covariantDerivative,
    ContinuousLinearMap.inverse_equiv, ContinuousLinearEquiv.coe_coe,
    ContinuousLinearEquiv.apply_symm_apply, map_sub, map_add]
  abel


/-- The nonlinear Lie-bracket covariance uses Mathlib's actual second-derivative
symmetry theorem over the reals. -/
theorem lieBracket_pullback (f : E → F) (X Y : F → F) (x : E)
    (hf : ContDiffAt ℝ 2 f x) (hX : DifferentiableAt ℝ X (f x))
    (hY : DifferentiableAt ℝ Y (f x)) :
    VectorField.lieBracket ℝ (VectorField.pullback ℝ f X) (VectorField.pullback ℝ f Y) x =
      VectorField.pullback ℝ f (VectorField.lieBracket ℝ X Y) x := by
  exact (VectorField.pullback_lieBracket (by norm_num) hf hX hY).symm


omit [CompleteSpace E] in
/-- Covariant differentiation in the field slot depends only on its germ. -/
theorem covariantDerivative_congr_eventually (Gamma : E → E → E → E)
    (X Y Z : E → E) (x : E) (h : Y =ᶠ[𝓝 x] Z) :
    covariantDerivative Gamma X Y x = covariantDerivative Gamma X Z x := by
  simp only [covariantDerivative, h.fderiv_eq, h.self_of_nhds]


/-- An invertible Jacobian remains invertible near a C² coordinate point. -/
theorem eventually_isInvertible_fderiv (f : E → F) (x : E)
    (hf : ContDiffAt ℝ 2 f x) (hi : (fderiv ℝ f x).IsInvertible) :
    ∀ᶠ z in 𝓝 x, (fderiv ℝ f z).IsInvertible := by
  rcases exists_continuousLinearEquiv_fderiv_symm_eq hf hi with ⟨M, _, _, hM, _⟩
  exact hM.mono fun z hz => ⟨M z, hz⟩


/-- Nonlinear connection covariance holds as an actual neighborhood identity,
so it can be differentiated in the curvature calculation. -/
theorem covariantDerivative_pullback_eventually (f : E → F)
    (Gamma : F → F → F → F) (X Y : F → F) (x : E)
    (hf : ContDiffAt ℝ 2 f x) (hi : (fderiv ℝ f x).IsInvertible)
    (hY : ∀ᶠ z in 𝓝 (f x), DifferentiableAt ℝ Y z) :
    covariantDerivative (pullbackConnection f Gamma)
      (VectorField.pullback ℝ f X) (VectorField.pullback ℝ f Y) =ᶠ[𝓝 x]
      VectorField.pullback ℝ f (covariantDerivative Gamma X Y) := by
  filter_upwards [hf.eventually (by norm_num), eventually_isInvertible_fderiv f x hf hi,
    hf.continuousAt.tendsto.eventually hY] with z hfz hiz hYz
  exact covariantDerivative_pullback f Gamma X Y z hfz hiz hYz


/-- The bracket-direction term of curvature also transforms by actual calculus. -/
theorem covariantDerivative_lieBracket_pullback (f : E → F)
    (Gamma : F → F → F → F) (X Y Z : F → F) (x : E)
    (hf : ContDiffAt ℝ 2 f x) (hi : (fderiv ℝ f x).IsInvertible)
    (hX : DifferentiableAt ℝ X (f x)) (hY : DifferentiableAt ℝ Y (f x))
    (hZ : DifferentiableAt ℝ Z (f x)) :
    covariantDerivative (pullbackConnection f Gamma)
      (VectorField.lieBracket ℝ (VectorField.pullback ℝ f X) (VectorField.pullback ℝ f Y))
      (VectorField.pullback ℝ f Z) x =
      VectorField.pullback ℝ f (covariantDerivative Gamma (VectorField.lieBracket ℝ X Y) Z) x := by
  simpa only [covariantDerivative, lieBracket_pullback f X Y x hf hX hY] using
    covariantDerivative_pullback f Gamma (VectorField.lieBracket ℝ X Y) Z x hf hi hZ


/-- Actual curvature covariance under a local nonlinear coordinate map.
Only local chart regularity/invertibility is used. The third field must be
differentiable near the represented point, and both intermediate covariant
fields must be differentiable there. No transformation formula is assumed. -/
theorem curvature_pullback (f : E → F) (Gamma : F → F → F → F)
    (X Y Z : F → F) (x : E) (hf : ContDiffAt ℝ 2 f x)
    (hi : (fderiv ℝ f x).IsInvertible)
    (hX : DifferentiableAt ℝ X (f x)) (hY : DifferentiableAt ℝ Y (f x))
    (hZ : ∀ᶠ z in 𝓝 (f x), DifferentiableAt ℝ Z z)
    (hYZ : DifferentiableAt ℝ (covariantDerivative Gamma Y Z) (f x))
    (hXZ : DifferentiableAt ℝ (covariantDerivative Gamma X Z) (f x)) :
    curvature (pullbackConnection f Gamma)
      (VectorField.pullback ℝ f X) (VectorField.pullback ℝ f Y)
      (VectorField.pullback ℝ f Z) x =
      VectorField.pullback ℝ f (curvature Gamma X Y Z) x := by
  rw [curvature,
    covariantDerivative_congr_eventually _ _ _ _ x
      (covariantDerivative_pullback_eventually f Gamma Y Z x hf hi hZ),
    covariantDerivative_congr_eventually _ _ _ _ x
      (covariantDerivative_pullback_eventually f Gamma X Z x hf hi hZ),
    covariantDerivative_pullback f Gamma X (covariantDerivative Gamma Y Z) x hf hi hYZ,
    covariantDerivative_pullback f Gamma Y (covariantDerivative Gamma X Z) x hf hi hXZ,
    covariantDerivative_lieBracket_pullback f Gamma X Y Z x hf hi hX hY hZ.self_of_nhds]
  simp only [VectorField.pullback, curvature, map_sub]


omit [CompleteSpace E] in
/-- Pairings of transported fields agree in the metric pulled back by the actual Jacobian. -/
theorem pullbackMetric_fields (f : E → F) (g : F → F → F → ℝ)
    (X Y : F → F) (x : E) (hi : (fderiv ℝ f x).IsInvertible) :
    pullbackMetric f g x (VectorField.pullback ℝ f X x) (VectorField.pullback ℝ f Y x) =
      g (f x) (X (f x)) (Y (f x)) := by
  simp only [pullbackMetric, VectorField.fderiv_pullback f X x hi,
    VectorField.fderiv_pullback f Y x hi]


omit [CompleteSpace E] in
/-- Positive definiteness of the old metric transfers at every invertible Jacobian. -/
theorem pullbackMetric_pos (f : E → F) (g : F → F → F → ℝ)
    (x v : E) (hi : (fderiv ℝ f x).IsInvertible)
    (hg : ∀ w, w ≠ 0 → 0 < g (f x) w w) (hv : v ≠ 0) :
    0 < pullbackMetric f g x v v := by
  rcases hi with ⟨M, hM⟩
  simpa only [pullbackMetric, ← hM, ContinuousLinearEquiv.coe_coe] using
    hg (M v) (fun h => hv (M.injective (h.trans (map_zero M).symm)))


omit [CompleteSpace E] in
/-- The actual metric Gram expression is preserved at an invertible Jacobian. -/
theorem gram_pullback (f : E → F) (g : F → F → F → ℝ)
    (X Y : F → F) (x : E) (hi : (fderiv ℝ f x).IsInvertible) :
    gram (pullbackMetric f g) x (VectorField.pullback ℝ f X x) (VectorField.pullback ℝ f Y x) =
      gram g (f x) (X (f x)) (Y (f x)) := by
  simp only [gram, pullbackMetric_fields f g _ _ x hi]


/-- The actual curvature/Gram ratio is invariant under a local nonlinear chart.
As a real equality this uses totalized division; geometric interpretation also
requires a genuine metric/connection and a nonzero Gram denominator. -/
theorem gramRatio_pullback (f : E → F) (g : F → F → F → ℝ)
    (Gamma : F → F → F → F) (X Y : F → F) (x : E)
    (hf : ContDiffAt ℝ 2 f x) (hi : (fderiv ℝ f x).IsInvertible)
    (hX : DifferentiableAt ℝ X (f x))
    (hY : ∀ᶠ z in 𝓝 (f x), DifferentiableAt ℝ Y z)
    (hYY : DifferentiableAt ℝ (covariantDerivative Gamma Y Y) (f x))
    (hXY : DifferentiableAt ℝ (covariantDerivative Gamma X Y) (f x)) :
    gramRatio (pullbackMetric f g) (pullbackConnection f Gamma)
      (VectorField.pullback ℝ f X) (VectorField.pullback ℝ f Y) x =
      gramRatio g Gamma X Y (f x) := by
  rw [gramRatio, curvature_pullback f Gamma X Y Y x hf hi hX hY.self_of_nhds hY hYY hXY,
    gram_pullback f g X Y x hi, pullbackMetric_fields f g _ _ x hi]
  rfl


omit [CompleteSpace E] in
/-- The pulled-back flat coordinate frame has a positive Gram denominator. -/
theorem flat_gram_pullback_pos (f : E → FlatShoreGeometry.Coord)
    (epsilon a : ℝ) (x : E) (he : 0 < epsilon)
    (hi : (fderiv ℝ f x).IsInvertible) :
    0 < gram (pullbackMetric f (fun p => FlatShoreGeometry.metric epsilon a p.1)) x
      (VectorField.pullback ℝ f FlatShoreGeometry.normalField x)
      (VectorField.pullback ℝ f FlatShoreGeometry.tangentField x) := by
  rw [gram_pullback f _ _ _ x hi]
  simpa [gram, FlatShoreGeometry.metric, FlatShoreGeometry.normalField,
    FlatShoreGeometry.tangentField] using
    mul_pos (FlatShoreGeometry.radialMetric_pos epsilon a (f x).1 he)
      (FlatShoreGeometry.tangentialMetric_pos epsilon (f x).1 he)


/-- The checked flat curvature formula in every locally invertible C² nonlinear chart.
All old field derivative hypotheses are discharged by the actual flat coefficients. -/
theorem flat_gramRatio_pullback (f : E → FlatShoreGeometry.Coord)
    (epsilon a : ℝ) (x : E) (he : 0 < epsilon)
    (hf : ContDiffAt ℝ 2 f x) (hi : (fderiv ℝ f x).IsInvertible) :
    gramRatio
      (pullbackMetric f (fun p => FlatShoreGeometry.metric epsilon a p.1))
      (pullbackConnection f (fun p => FlatShoreGeometry.christoffel epsilon a p.1))
      (VectorField.pullback ℝ f FlatShoreGeometry.normalField)
      (VectorField.pullback ℝ f FlatShoreGeometry.tangentField) x =
      FlatShore.radialCurvature epsilon a (f x).1 := by
  apply Eq.trans (gramRatio_pullback f
    (fun p => FlatShoreGeometry.metric epsilon a p.1)
    (fun p => FlatShoreGeometry.christoffel epsilon a p.1)
    FlatShoreGeometry.normalField FlatShoreGeometry.tangentField x hf hi ?_ ?_ ?_ ?_)
  · exact FlatShoreGeometry.sectionalCurvature_eq_radialCurvature epsilon a (f x) he
  · exact differentiableAt_const _
  · exact Filter.Eventually.of_forall fun _ => differentiableAt_const _
  · rw [flat_covariantDerivative_eq, FlatShoreGeometry.covariantDerivative_tangent_tangent]
    exact (((FlatShoreGeometry.hasDerivAt_gammaYY epsilon a (f x).1 he).comp_hasFDerivAt
      (f x) (ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt).prodMk
        (hasFDerivAt_const (0 : ℝ) (f x))).differentiableAt
  · rw [flat_covariantDerivative_eq, FlatShoreGeometry.covariantDerivative_normal_tangent]
    exact ((hasFDerivAt_const (0 : ℝ) (f x)).prodMk
      ((FlatShoreGeometry.hasDerivAt_gammaRY epsilon (f x).1 he).comp_hasFDerivAt
        (f x) (ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt)).differentiableAt

end FractalSync.NonlinearCoordinates
