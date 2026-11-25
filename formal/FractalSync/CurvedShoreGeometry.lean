import FractalSync.FlatShoreGeometry

/-!
# Curved-Shore normal-coordinate curvature

The coordinate metric has Euclidean tangential factor `1 - k(y) * r`.
The connection and curvature below use actual Fréchet derivatives. The Shore
curvature computation is local, with explicit differentiability hypotheses on
the supplied curvature function; constructing a normal chart of an embedded
Shore is a separate geometric obligation.
-/

namespace FractalSync.CurvedShoreGeometry

abbrev Coord := FlatShoreGeometry.Coord

noncomputable def metric (epsilon a k r : ℝ) (v w : Coord) : ℝ :=
  FlatShoreGeometry.radialMetric epsilon a r * v.1 * w.1 +
    FlatShoreGeometry.tangentialMetric epsilon r * (1 - k * r) ^ 2 * v.2 * w.2

theorem metric_eq_flat_pullback (epsilon a k r : ℝ) (v w : Coord) :
    metric epsilon a k r v w = FlatShoreGeometry.metric epsilon a r
      (v.1, (1 - k * r) * v.2) (w.1, (1 - k * r) * w.2) := by
  dsimp [metric, FlatShoreGeometry.metric]
  ring

theorem metric_self_pos (epsilon a k r : ℝ) (v : Coord) (he : 0 < epsilon)
    (hh : 1 - k * r ≠ 0) (hv : v ≠ 0) : 0 < metric epsilon a k r v v := by
  simpa only [metric_eq_flat_pullback] using
    FlatShoreGeometry.metric_self_pos epsilon a r (v.1, (1 - k * r) * v.2) he
      (by simpa [Prod.ext_iff, mul_eq_zero, hh] using hv)

theorem metric_symm (epsilon a k r : ℝ) (v w : Coord) :
    metric epsilon a k r v w = metric epsilon a k r w v := by
  simpa only [metric_eq_flat_pullback] using
    FlatShoreGeometry.metric_symm epsilon a r
      (v.1, (1 - k * r) * v.2) (w.1, (1 - k * r) * w.2)

/-- The Gram denominator in an arbitrary coordinate frame factors by the
square of its ordinary determinant. -/
theorem metric_gram_determinant (epsilon a k r : ℝ) (v w : Coord) :
    metric epsilon a k r v v * metric epsilon a k r w w -
        metric epsilon a k r v w ^ 2 =
      FlatShoreGeometry.radialMetric epsilon a r *
        FlatShoreGeometry.tangentialMetric epsilon r * (1 - k * r) ^ 2 *
          (v.1 * w.2 - v.2 * w.1) ^ 2 := by
  dsimp only [metric]
  ring

theorem metric_zero_curvature (epsilon a r : ℝ) (v w : Coord) :
    metric epsilon a 0 r v w = FlatShoreGeometry.metric epsilon a r v w := by
  simp [metric, FlatShoreGeometry.metric]

noncomputable def gammaYY (epsilon a k r : ℝ) : ℝ :=
  (1 - k * r) * (r + k * epsilon ^ 2) /
    (epsilon ^ 2 + (1 + a ^ 2) * r ^ 2)

noncomputable def gammaRY (epsilon k r : ℝ) : ℝ :=
  -r / (epsilon ^ 2 + r ^ 2) - k / (1 - k * r)

noncomputable def gammaY (k dk r : ℝ) : ℝ := -r * dk / (1 - k * r)

noncomputable def christoffel (epsilon a : ℝ) (k : ℝ → ℝ)
    (x : Coord) (v w : Coord) : Coord :=
  (FlatShoreGeometry.gammaRR epsilon a x.1 * v.1 * w.1 +
      gammaYY epsilon a (k x.2) x.1 * v.2 * w.2,
    gammaRY epsilon (k x.2) x.1 * (v.1 * w.2 + v.2 * w.1) +
      gammaY (k x.2) (deriv k x.2) x.1 * v.2 * w.2)

noncomputable def metricRate (epsilon a k dk r : ℝ) (d v w : Coord) : ℝ :=
  2 * FlatShoreGeometry.radialMetric epsilon a r *
      FlatShoreGeometry.gammaRR epsilon a r * d.1 * v.1 * w.1 +
    (2 * FlatShoreGeometry.tangentialMetric epsilon r *
        FlatShoreGeometry.gammaRY epsilon r * d.1 * (1 - k * r) ^ 2 -
      2 * FlatShoreGeometry.tangentialMetric epsilon r * (1 - k * r) *
        (dk * d.2 * r + k * d.1)) * v.2 * w.2

theorem christoffel_symm (epsilon a : ℝ) (k : ℝ → ℝ) (x v w : Coord) :
    christoffel epsilon a k x v w = christoffel epsilon a k x w v := by
  simp only [christoffel, mul_comm, mul_left_comm, add_comm]

theorem christoffel_on_shore (epsilon a y : ℝ) (k : ℝ → ℝ) (v w : Coord)
    (he : 0 < epsilon) :
    christoffel epsilon a k (0, y) v w =
      (k y * v.2 * w.2, -(k y) * (v.1 * w.2 + v.2 * w.1)) := by
  simp [christoffel, FlatShoreGeometry.gammaRR, gammaYY, gammaRY, gammaY, he.ne']

theorem christoffel_metric_compatible (epsilon a : ℝ) (k : ℝ → ℝ)
    (x d v w : Coord) (he : 0 < epsilon) (hh : 1 - k x.2 * x.1 ≠ 0) :
    metricRate epsilon a (k x.2) (deriv k x.2) x.1 d v w =
      metric epsilon a (k x.2) x.1 (christoffel epsilon a k x d v) w +
        metric epsilon a (k x.2) x.1 v (christoffel epsilon a k x d w) := by
  dsimp only [metricRate, metric, christoffel, gammaYY, gammaRY, gammaY,
    FlatShoreGeometry.radialMetric, FlatShoreGeometry.tangentialMetric,
    FlatShoreGeometry.gammaRR, FlatShoreGeometry.gammaRY]
  field_simp [hh, (FlatShore.ruler_sq_pos epsilon x.1 he).ne',
    (FlatShore.normal_sq_pos epsilon a x.1 he).ne']
  field_simp [show 1 - x.1 * k x.2 ≠ 0 by simpa [mul_comm] using hh]
  ring

noncomputable def covariantDerivative (epsilon a : ℝ) (k : ℝ → ℝ)
    (X Y : Coord → Coord) (x : Coord) : Coord :=
  fderiv ℝ Y x (X x) + christoffel epsilon a k x (X x) (Y x)

theorem covariantDerivative_torsion (epsilon a : ℝ) (k : ℝ → ℝ)
    (X Y : Coord → Coord) (x : Coord) :
    covariantDerivative epsilon a k X Y x - covariantDerivative epsilon a k Y X x =
      VectorField.lieBracket ℝ X Y x := by
  rw [covariantDerivative, covariantDerivative,
    christoffel_symm epsilon a k x (Y x) (X x), VectorField.lieBracket]
  abel

theorem covariantDerivative_first_linear (epsilon a : ℝ) (k : ℝ → ℝ) (f g : Coord → ℝ)
    (X Y Z : Coord → Coord) (x : Coord) :
    covariantDerivative epsilon a k (fun p => f p • X p + g p • Z p) Y x =
      f x • covariantDerivative epsilon a k X Y x +
        g x • covariantDerivative epsilon a k Z Y x := by
  simp only [covariantDerivative, map_add, map_smul]
  apply Prod.ext
  all_goals dsimp [christoffel]
  all_goals ring

theorem covariantDerivative_second_add (epsilon a : ℝ) (k : ℝ → ℝ)
    (X Y Z : Coord → Coord) (x : Coord)
    (hY : DifferentiableAt ℝ Y x) (hZ : DifferentiableAt ℝ Z x) :
    covariantDerivative epsilon a k X (Y + Z) x =
      covariantDerivative epsilon a k X Y x + covariantDerivative epsilon a k X Z x := by
  simp only [covariantDerivative, fderiv_add hY hZ, ContinuousLinearMap.add_apply, Pi.add_apply]
  apply Prod.ext
  all_goals dsimp [christoffel]
  all_goals ring

theorem covariantDerivative_leibniz (epsilon a : ℝ) (k : ℝ → ℝ) (f : Coord → ℝ)
    (X Y : Coord → Coord) (x : Coord)
    (hf : DifferentiableAt ℝ f x) (hY : DifferentiableAt ℝ Y x) :
    covariantDerivative epsilon a k X (fun p => f p • Y p) x =
      (fderiv ℝ f x (X x)) • Y x + f x • covariantDerivative epsilon a k X Y x := by
  simp only [covariantDerivative, fderiv_fun_smul hf hY, ContinuousLinearMap.add_apply,
    ContinuousLinearMap.smul_apply, ContinuousLinearMap.smulRight_apply]
  apply Prod.ext
  all_goals dsimp [christoffel]
  all_goals ring

/-- Differentiate the actual varying metric pairing along a coordinate curve. -/
theorem hasDerivAt_metric_pairing (epsilon a t : ℝ) (k : ℝ → ℝ)
    (gamma v w : ℝ → Coord) (d dv dw : Coord) (he : 0 < epsilon)
    (hg : HasDerivAt gamma d t) (hv : HasDerivAt v dv t) (hw : HasDerivAt w dw t)
    (hk : DifferentiableAt ℝ k (gamma t).2) :
    HasDerivAt (fun s => metric epsilon a (k (gamma s).2) (gamma s).1 (v s) (w s))
      (metricRate epsilon a (k (gamma t).2) (deriv k (gamma t).2) (gamma t).1 d (v t) (w t) +
        metric epsilon a (k (gamma t).2) (gamma t).1 dv (w t) +
        metric epsilon a (k (gamma t).2) (gamma t).1 (v t) dw) t := by
  have hr := (ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt.comp_hasDerivAt t hg
  have hy := (ContinuousLinearMap.snd ℝ ℝ ℝ).hasFDerivAt.comp_hasDerivAt t hg
  have hv1 := (ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt.comp_hasDerivAt t hv
  have hv2 := (ContinuousLinearMap.snd ℝ ℝ ℝ).hasFDerivAt.comp_hasDerivAt t hv
  have hw1 := (ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt.comp_hasDerivAt t hw
  have hw2 := (ContinuousLinearMap.snd ℝ ℝ ℝ).hasFDerivAt.comp_hasDerivAt t hw
  have hh := ((hk.hasDerivAt.comp t hy).mul hr).const_sub 1
  convert ((((FlatShoreGeometry.hasDerivAt_radialMetric epsilon a (gamma t).1 he).comp
    t hr).mul hv1).mul hw1).add
      (((((FlatShoreGeometry.hasDerivAt_tangentialMetric epsilon (gamma t).1 he).comp
        t hr).mul (hh.pow 2)).mul hv2).mul hw2) using 1
  dsimp [metricRate, metric]
  ring

/-- Genuine connection compatibility, with the actual derivative of a pairing
of differentiable vector fields along a curve tangent to `X`. -/
theorem covariantDerivative_metric_pairing (epsilon a t : ℝ) (k : ℝ → ℝ)
    (X Y Z : Coord → Coord) (gamma : ℝ → Coord) (he : 0 < epsilon)
    (hg : HasDerivAt gamma (X (gamma t)) t)
    (hY : DifferentiableAt ℝ Y (gamma t)) (hZ : DifferentiableAt ℝ Z (gamma t))
    (hk : DifferentiableAt ℝ k (gamma t).2)
    (hh : 1 - k (gamma t).2 * (gamma t).1 ≠ 0) :
    HasDerivAt (fun s => metric epsilon a (k (gamma s).2) (gamma s).1
        (Y (gamma s)) (Z (gamma s)))
      (metric epsilon a (k (gamma t).2) (gamma t).1
          (covariantDerivative epsilon a k X Y (gamma t)) (Z (gamma t)) +
        metric epsilon a (k (gamma t).2) (gamma t).1 (Y (gamma t))
          (covariantDerivative epsilon a k X Z (gamma t))) t := by
  convert hasDerivAt_metric_pairing epsilon a t k gamma (Y ∘ gamma) (Z ∘ gamma)
    (X (gamma t)) (fderiv ℝ Y (gamma t) (X (gamma t)))
    (fderiv ℝ Z (gamma t) (X (gamma t))) he hg
    (hY.hasFDerivAt.comp_hasDerivAt t hg) (hZ.hasFDerivAt.comp_hasDerivAt t hg) hk using 1
  rw [christoffel_metric_compatible epsilon a k (gamma t) _ _ _ he hh]
  simp only [covariantDerivative, metric, Prod.fst_add, Prod.snd_add, Function.comp_apply]
  ring

noncomputable def curvature (epsilon a : ℝ) (k : ℝ → ℝ)
    (X Y Z : Coord → Coord) (x : Coord) : Coord :=
  covariantDerivative epsilon a k X (covariantDerivative epsilon a k Y Z) x -
    covariantDerivative epsilon a k Y (covariantDerivative epsilon a k X Z) x -
    covariantDerivative epsilon a k (VectorField.lieBracket ℝ X Y) Z x

theorem covariantDerivative_const (epsilon a : ℝ) (k : ℝ → ℝ) (u v : Coord) :
    covariantDerivative epsilon a k (fun _ => u) (fun _ => v) =
      fun x => christoffel epsilon a k x u v := by
  funext x
  simp [covariantDerivative]

theorem fderiv_gammaYY_pair_normal_on_shore (epsilon a y dk ddk : ℝ)
    (k : ℝ → ℝ) (he : 0 < epsilon)
    (hk : HasDerivAt k dk y) (hdk : HasDerivAt (deriv k) ddk y) :
    fderiv ℝ (fun x : Coord =>
      (gammaYY epsilon a (k x.2) x.1, gammaY (k x.2) (deriv k x.2) x.1))
      (0, y) (1, 0) = (1 / epsilon ^ 2 - (k y) ^ 2, -dk) := by
  have hr : HasFDerivAt (fun x : Coord => x.1)
      (ContinuousLinearMap.fst ℝ ℝ ℝ) (0, y) :=
    (ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt
  have hk2 := hk.comp_hasFDerivAt ((0, y) : Coord)
    (ContinuousLinearMap.snd ℝ ℝ ℝ).hasFDerivAt
  have hdk2 := hdk.comp_hasFDerivAt ((0, y) : Coord)
    (ContinuousLinearMap.snd ℝ ℝ ℝ).hasFDerivAt
  have hden := ((hr.pow 2).const_mul (1 + a ^ 2)).const_add (epsilon ^ 2)
  have hi := (hasDerivAt_inv (by simpa using pow_ne_zero 2 he.ne')).comp_hasFDerivAt
    ((0, y) : Coord) hden
  have hYY := (((hk2.mul hr).const_sub 1).mul
    (hr.add (hk2.mul_const (epsilon ^ 2)))).mul hi
  have hi2 := (hasDerivAt_inv (by norm_num : 1 - (k y) * (0 : ℝ) ≠ 0)).comp_hasFDerivAt
    ((0, y) : Coord) ((hk2.mul hr).const_sub 1)
  have hY := (hr.neg.mul hdk2).mul hi2
  have hd := congrArg (fun L : Coord →L[ℝ] Coord => L (1, 0)) (hYY.prodMk hY).fderiv
  simp only [gammaYY, gammaY, div_eq_mul_inv]
  simp only [Function.comp_apply, Pi.mul_apply, Pi.add_apply, Pi.neg_apply,
    ContinuousLinearMap.coe_snd'] at hd
  rw [hd]
  simp [hk.deriv]
  field_simp
  ring

theorem fderiv_gammaRY_tangent_on_shore (epsilon y dk : ℝ)
    (k : ℝ → ℝ) (he : 0 < epsilon) (hk : HasDerivAt k dk y) :
    fderiv ℝ (fun x : Coord => ((0 : ℝ), gammaRY epsilon (k x.2) x.1))
      (0, y) (0, 1) = (0, -dk) := by
  have hr : HasFDerivAt (fun x : Coord => x.1)
      (ContinuousLinearMap.fst ℝ ℝ ℝ) (0, y) :=
    (ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt
  have hk2 := hk.comp_hasFDerivAt ((0, y) : Coord)
    (ContinuousLinearMap.snd ℝ ℝ ℝ).hasFDerivAt
  have hi := (hasDerivAt_inv (by norm_num : 1 - (k y) * (0 : ℝ) ≠ 0)).comp_hasFDerivAt
    ((0, y) : Coord) ((hk2.mul hr).const_sub 1)
  have hf := (FlatShoreGeometry.hasDerivAt_gammaRY epsilon 0 he).comp_hasFDerivAt
    ((0, y) : Coord) hr
  have hd := congrArg (fun L : Coord →L[ℝ] Coord => L (0, 1))
    ((hasFDerivAt_const (0 : ℝ) ((0, y) : Coord)).prodMk (hf.sub (hk2.mul hi))).fderiv
  simpa [gammaRY, FlatShoreGeometry.gammaRY, div_eq_mul_inv] using hd

/-- Actual curvature commutator at Shore for a varying curvature profile. -/
theorem curvature_normal_tangent_tangent_on_shore (epsilon a y dk ddk : ℝ)
    (k : ℝ → ℝ) (he : 0 < epsilon)
    (hk : HasDerivAt k dk y) (hdk : HasDerivAt (deriv k) ddk y) :
    curvature epsilon a k (fun _ => (1, 0)) (fun _ => (0, 1))
      (fun _ => (0, 1)) (0, y) = (1 / epsilon ^ 2, 0) := by
  rw [curvature, covariantDerivative_const, covariantDerivative_const]
  simp only [christoffel, mul_one, mul_zero, add_zero, zero_add]
  simp only [covariantDerivative, VectorField.lieBracket, fderiv_const_apply,
    ContinuousLinearMap.zero_apply, sub_self, zero_add]
  rw [fderiv_gammaYY_pair_normal_on_shore epsilon a y dk ddk k he hk hdk,
    fderiv_gammaRY_tangent_on_shore epsilon y dk k he hk]
  simp [christoffel, FlatShoreGeometry.gammaRR, gammaYY, gammaRY, gammaY, he.ne']
  ring

/-- Coordinate sectional curvature from the actual commutator and metric Gram
determinant. The definition is used geometrically away from normal focal points. -/
noncomputable def sectionalCurvature (epsilon a : ℝ) (k : ℝ → ℝ) (x : Coord) : ℝ :=
  metric epsilon a (k x.2) x.1
      (curvature epsilon a k (fun _ => (1, 0)) (fun _ => (0, 1)) (fun _ => (0, 1)) x)
      (1, 0) /
    (metric epsilon a (k x.2) x.1 (1, 0) (1, 0) *
      metric epsilon a (k x.2) x.1 (0, 1) (0, 1) -
        metric epsilon a (k x.2) x.1 (1, 0) (0, 1) ^ 2)

/-- Every point of the Shore has curvature positive one in this curved normal
coordinate metric, independently of the curvature profile and its local jets. -/
theorem sectionalCurvature_on_shore (epsilon a y dk ddk : ℝ) (k : ℝ → ℝ)
    (he : 0 < epsilon) (hk : HasDerivAt k dk y) (hdk : HasDerivAt (deriv k) ddk y) :
    sectionalCurvature epsilon a k (0, y) = 1 := by
  rw [sectionalCurvature, curvature_normal_tangent_tangent_on_shore epsilon a y dk ddk k he hk hdk]
  simp [metric, FlatShoreGeometry.radialMetric, FlatShoreGeometry.tangentialMetric, he.ne']

end FractalSync.CurvedShoreGeometry
