import FractalSync.NonlinearCoordinates
import FractalSync.CurvatureTensoriality

/-!
# Actual connection forms in moving frames

Base direction fields remain in the fixed coordinate space. A moving frame
changes the coordinates of sections in the fiber and contributes its actual
first derivative to the transformed connection form.
-/

namespace FractalSync.MovingFrames

open scoped Topology

variable {E F G : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]
  [NormedAddCommGroup G] [NormedSpace ℝ G]

/-- A coordinate connection form is continuous bilinear in direction and section value. -/
abbrev ConnectionForm (E F : Type*) [NormedAddCommGroup E] [NormedSpace ℝ E]
    [NormedAddCommGroup F] [NormedSpace ℝ F] := E → E →L[ℝ] F →L[ℝ] F

/-- Actual covariant differentiation of a section in a coordinate connection form. -/
noncomputable def covariantDerivative (Gamma : ConnectionForm E F)
    (X : E → E) (Z : E → F) (x : E) : F :=
  fderiv ℝ Z x (X x) + Gamma x (X x) (Z x)

/-- Curvature is the actual connection commutator, with the base Lie bracket. -/
noncomputable def curvature (Gamma : ConnectionForm E F)
    (X Y : E → E) (Z : E → F) (x : E) : F :=
  covariantDerivative Gamma X (covariantDerivative Gamma Y Z) x -
    covariantDerivative Gamma Y (covariantDerivative Gamma X Z) x -
    covariantDerivative Gamma (VectorField.lieBracket ℝ X Y) Z x

/-- Coordinates of an actual section in the supplied frame field. -/
noncomputable def inFrame (P : E → F →L[ℝ] G) (Z : E → G) (x : E) : F :=
  (P x).inverse (Z x)

/-- The moving-frame connection form includes the actual frame derivative. -/
noncomputable def gauge (P : E → F →L[ℝ] G) (Gamma : ConnectionForm E G) :
    ConnectionForm E F := fun x =>
  ((ContinuousLinearMap.compL ℝ F G F) (P x).inverse).comp
    (fderiv ℝ P x +
      ((ContinuousLinearMap.compL ℝ F G G).flip (P x)).comp (Gamma x))

/-- Evaluation exposes the inhomogeneous connection-form gauge law. -/
theorem gauge_apply (P : E → F →L[ℝ] G) (Gamma : ConnectionForm E G) (x v : E) (w : F) :
    gauge P Gamma x v w =
      (P x).inverse (fderiv ℝ P x v w + Gamma x v (P x w)) := by
  rfl

omit [NormedSpace ℝ E] in
/-- A continuous frame with an invertible value remains invertible nearby. -/
theorem eventually_isInvertible [CompleteSpace F] (P : E → F →L[ℝ] G)
    (x : E) (hP : ContinuousAt P x) (hi : (P x).IsInvertible) :
    ∀ᶠ y in 𝓝 x, (P y).IsInvertible := by
  rcases hi with ⟨M, hM⟩
  exact hP (by simpa only [← hM, ContinuousLinearMap.IsInvertible] using M.nhds)

/-- Inversion of an actual frame field is differentiable at each invertible value. -/
theorem differentiableAt_inverseFrame [CompleteSpace F] (P : E → F →L[ℝ] G)
    (x : E) (hP : DifferentiableAt ℝ P x) (hi : (P x).IsInvertible) :
    DifferentiableAt ℝ (fun y => (P y).inverse) x := by
  exact ((hi.contDiffAt_map_inverse (n := 1)).differentiableAt le_rfl).comp x hP

/-- The derivative of the actual inverse-frame field, obtained by differentiating
its local inverse identity. -/
theorem fderiv_inverseFrame [CompleteSpace F] (P : E → F →L[ℝ] G)
    (x v : E) (hP : DifferentiableAt ℝ P x) (hi : (P x).IsInvertible) :
    fderiv ℝ (fun y => (P y).inverse) x v =
      -((P x).inverse.comp (fderiv ℝ P x v)).comp (P x).inverse := by
  have heq : (fun y => (P y).comp (P y).inverse) =ᶠ[𝓝 x]
      (fun _ => ContinuousLinearMap.id ℝ G) := by
    exact (eventually_isInvertible P x hP.continuousAt hi).mono fun y ⟨M, hM⟩ => by
      simp [← hM]
  have hz : fderiv ℝ (fun y => (P y).comp (P y).inverse) x v = 0 := by
    simpa only [fderiv_const_apply, ContinuousLinearMap.zero_apply] using
      congrArg (fun L : E →L[ℝ] G →L[ℝ] G => L v) heq.fderiv_eq
  rw [fderiv_clm_comp hP (differentiableAt_inverseFrame P x hP hi)] at hz
  rcases hi with ⟨M, hM⟩
  refine ContinuousLinearMap.ext fun w => ?_
  have hw := congrArg (fun L : G →L[ℝ] G => M.symm (L w)) hz
  simp [← hM] at hw ⊢
  exact eq_neg_of_add_eq_zero_left hw

/-- The actual derivative of inverse-frame section coordinates. -/
theorem fderiv_inFrame [CompleteSpace F] (P : E → F →L[ℝ] G)
    (Z : E → G) (x v : E) (hP : DifferentiableAt ℝ P x)
    (hi : (P x).IsInvertible) (hZ : DifferentiableAt ℝ Z x) :
    fderiv ℝ (inFrame P Z) x v =
      (P x).inverse (fderiv ℝ Z x v - fderiv ℝ P x v (inFrame P Z x)) := by
  change fderiv ℝ (fun y => (P y).inverse (Z y)) x v = _
  rw [fderiv_clm_apply (differentiableAt_inverseFrame P x hP hi) hZ]
  simp [fderiv_inverseFrame P x v hP hi, inFrame, sub_eq_add_neg]

/-- Covariant differentiation in the section slot depends only on the section germ. -/
theorem covariantDerivative_congr_eventually (Gamma : ConnectionForm E F)
    (X : E → E) (Y Z : E → F) (x : E) (h : Y =ᶠ[𝓝 x] Z) :
    covariantDerivative Gamma X Y x = covariantDerivative Gamma X Z x := by
  simp only [covariantDerivative, h.fderiv_eq, h.self_of_nhds]

/-- Actual moving-frame covariance of the covariant derivative. -/
theorem covariantDerivative_gauge [CompleteSpace F]
    (P : E → F →L[ℝ] G) (Gamma : ConnectionForm E G)
    (X : E → E) (Z : E → G) (x : E)
    (hP : DifferentiableAt ℝ P x) (hi : (P x).IsInvertible)
    (hZ : DifferentiableAt ℝ Z x) :
    covariantDerivative (gauge P Gamma) X (inFrame P Z) x =
      inFrame P (covariantDerivative Gamma X Z) x := by
  rw [covariantDerivative, fderiv_inFrame P Z x _ hP hi hZ, gauge_apply]
  rcases hi with ⟨M, hM⟩
  simp [inFrame, covariantDerivative, ← hM, map_add, map_sub]
  abel

/-- The moving-frame covariance identity holds on an actual neighborhood. -/
theorem covariantDerivative_gauge_eventually [CompleteSpace F]
    (P : E → F →L[ℝ] G) (Gamma : ConnectionForm E G)
    (X : E → E) (Z : E → G) (x : E)
    (hP : ContDiffAt ℝ 1 P x) (hi : (P x).IsInvertible)
    (hZ : ∀ᶠ y in 𝓝 x, DifferentiableAt ℝ Z y) :
    covariantDerivative (gauge P Gamma) X (inFrame P Z) =ᶠ[𝓝 x]
      inFrame P (covariantDerivative Gamma X Z) := by
  filter_upwards [hP.eventually (by norm_num), eventually_isInvertible P x hP.continuousAt hi, hZ]
    with y hPy hiy hZy
  exact covariantDerivative_gauge P Gamma X Z y (hPy.differentiableAt le_rfl) hiy hZy

/-- The actual curvature commutator changes fiber coordinates by the inverse
frame. The base direction fields and their Lie bracket are unchanged. -/
theorem curvature_gauge [CompleteSpace F]
    (P : E → F →L[ℝ] G) (Gamma : ConnectionForm E G)
    (X Y : E → E) (Z : E → G) (x : E)
    (hP : ContDiffAt ℝ 1 P x) (hi : (P x).IsInvertible)
    (hZ : ∀ᶠ y in 𝓝 x, DifferentiableAt ℝ Z y)
    (hYZ : DifferentiableAt ℝ (covariantDerivative Gamma Y Z) x)
    (hXZ : DifferentiableAt ℝ (covariantDerivative Gamma X Z) x) :
    curvature (gauge P Gamma) X Y (inFrame P Z) x =
      inFrame P (curvature Gamma X Y Z) x := by
  rw [curvature,
    covariantDerivative_congr_eventually _ _ _ _ x
      (covariantDerivative_gauge_eventually P Gamma Y Z x hP hi hZ),
    covariantDerivative_congr_eventually _ _ _ _ x
      (covariantDerivative_gauge_eventually P Gamma X Z x hP hi hZ),
    covariantDerivative_gauge P Gamma X (covariantDerivative Gamma Y Z) x
      (hP.differentiableAt le_rfl) hi hYZ,
    covariantDerivative_gauge P Gamma Y (covariantDerivative Gamma X Z) x
      (hP.differentiableAt le_rfl) hi hXZ,
    covariantDerivative_gauge P Gamma (VectorField.lieBracket ℝ X Y) Z x
      (hP.differentiableAt le_rfl) hi hZ.self_of_nhds]
  simp only [inFrame, curvature, map_sub]


/-- A C² frame and C¹ connection form give an actually C¹ transformed
coefficient field, in the continuous-bilinear operator topology. -/
theorem contDiffAt_gauge [CompleteSpace F]
    (P : E → F →L[ℝ] G) (Gamma : ConnectionForm E G) (x : E)
    (hP : ContDiffAt ℝ 2 P x) (hi : (P x).IsInvertible)
    (hG : ContDiffAt ℝ 1 Gamma x) :
    ContDiffAt ℝ 1 (gauge P Gamma) x := by
  have hP1 : ContDiffAt ℝ 1 P x := hP.of_le (by norm_num)
  have hInv : ContDiffAt ℝ 1 (fun y => (P y).inverse) x :=
    hi.contDiffAt_map_inverse.comp x hP1
  have hD : ContDiffAt ℝ 1 (fderiv ℝ P) x := hP.fderiv_right (by norm_num)
  exact (contDiffAt_const.clm_apply hInv).clm_comp
    (hD.add ((contDiffAt_const.clm_apply hP1).clm_comp hG))


/-- Successive moving-frame changes agree with the composite frame, including
both actual frame derivatives. This is the coefficient overlap law. -/
theorem gauge_comp {H : Type*} [NormedAddCommGroup H] [NormedSpace ℝ H]
    (P : E → F →L[ℝ] G) (Q : E → H →L[ℝ] F) (Gamma : ConnectionForm E G) (x : E)
    (hP : DifferentiableAt ℝ P x) (hQ : DifferentiableAt ℝ Q x)
    (hiP : (P x).IsInvertible) (hiQ : (Q x).IsInvertible) :
    gauge Q (gauge P Gamma) x = gauge (fun y => (P y).comp (Q y)) Gamma x := by
  refine ContinuousLinearMap.ext fun v => ContinuousLinearMap.ext fun w => ?_
  rcases hiP with ⟨M, hM⟩
  rcases hiQ with ⟨N, hN⟩
  simp [gauge_apply, fderiv_clm_comp hP hQ, ← hM, ← hN]
  abel


/-- The constant identity frame leaves the actual coefficient field unchanged. -/
theorem gauge_id (Gamma : ConnectionForm E F) :
    gauge (fun _ => ContinuousLinearMap.id ℝ F) Gamma = Gamma := by
  exact funext fun x => ContinuousLinearMap.ext fun v => ContinuousLinearMap.ext fun w => by
    simp [gauge_apply]


/-- Section regularity is preserved in an invertible moving frame of the same regularity. -/
theorem contDiffAt_inFrame [CompleteSpace F] (P : E → F →L[ℝ] G)
    (Z : E → G) (x : E) (n : WithTop ℕ∞)
    (hP : ContDiffAt ℝ n P x) (hi : (P x).IsInvertible)
    (hZ : ContDiffAt ℝ n Z x) :
    ContDiffAt ℝ n (inFrame P Z) x := by
  exact (hi.contDiffAt_map_inverse.comp x hP).clm_apply hZ

end FractalSync.MovingFrames
