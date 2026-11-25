import FractalSync.EmbeddedShoreConnection
import FractalSync.CurvedShoreCurvature

/-!
# Curvature of the actual ambient connection in normal coordinates

The derived ambient connection identification holds on a neighborhood. The
actual curvature commutator can therefore be transferred to the checked curved
coordinate calculation. In particular its metric Gram ratio is +1 on Shore.
-/

namespace FractalSync.CoordinateCovariance

open scoped Topology

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Actual curvature depends only on the germ of its connection coefficients. -/
theorem curvature_congr_connection (Gamma Lambda : E → E → E → E)
    (X Y Z : E → E) (x : E)
    (h : ∀ᶠ y in 𝓝 x, ∀ v w, Gamma y v w = Lambda y v w) :
    curvature Gamma X Y Z x = curvature Lambda X Y Z x := by
  have hcov (A B : E → E) : covariantDerivative Gamma A B =ᶠ[𝓝 x]
      covariantDerivative Lambda A B :=
    h.mono fun y hy => by simp only [covariantDerivative, hy]
  rw [curvature, curvature,
    NonlinearCoordinates.covariantDerivative_congr_eventually Gamma X _ _ x (hcov Y Z),
    NonlinearCoordinates.covariantDerivative_congr_eventually Gamma Y _ _ x (hcov X Z)]
  simp only [covariantDerivative, h.self_of_nhds]

end FractalSync.CoordinateCovariance

namespace FractalSync.EmbeddedShoreConnection

open EmbeddedShoreMetric EmbeddedNormalPatch TubularNeighborhood
open scoped Topology RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

/-- The actual connection identification persists on a full normal-coordinate
neighborhood, allowing its coefficients to be differentiated in curvature. -/
theorem eventually_ambientConnection_normalMap (S : Set E) (epsilon a R : ℝ)
    (C N T : ℝ → E) (k : ℝ → ℝ) (V : Set ℝ) (x : Coord)
    (hpatch : NormalPatch S R C N V) (hV : V ∈ 𝓝 x.2)
    (he : 0 < epsilon) (hr : |x.1| < R) (hh : 1 - k x.2 * x.1 ≠ 0)
    (hf : ContDiffAt ℝ 2 (normalMap C N) x)
    (hi : (fderiv ℝ (normalMap C N) x).IsInvertible)
    (hk : ContDiffAt ℝ 1 k x.2)
    (hF : ∀ᶠ s in 𝓝 x.2,
      HasDerivAt C (T s) s ∧ HasDerivAt N (-(k s) • T s) s ∧
        ‖T s‖ = 1 ∧ inner ℝ (N s) (T s) = 0) :
    ∀ᶠ y in 𝓝 x, ∀ v w,
      NonlinearCoordinates.pullbackConnection (normalMap C N) (ambientConnection S epsilon a)
        y v w = CurvedShoreGeometry.christoffel epsilon a k y v w := by
  have hnonfocal : ∀ᶠ y : Coord in 𝓝 x, 1 - k y.2 * y.1 ≠ 0 :=
    (continuousAt_const.sub ((hk.continuousAt.comp continuousAt_snd).mul
      continuousAt_fst)).eventually_ne hh
  filter_upwards [hnonfocal, hf.eventually (by norm_num),
    NonlinearCoordinates.eventually_isInvertible_fderiv (normalMap C N) x hf hi,
    (continuous_snd.tendsto x).eventually (hk.eventually (by norm_num)),
    (continuous_snd.tendsto x).eventually (eventually_mem_nhds_iff.2 hV),
    (continuous_snd.tendsto x).eventually (eventually_eventually_nhds.2 hF),
    (continuous_fst.abs.continuousAt).eventually_lt_const hr]
      with y hyh hyf hyi hyk hyV hyF hyr
  exact fun v w => ambientConnection_normalMap S epsilon a R C N T k V y v w hpatch hyV
    he hyr hyh hyf hyi (hyk.differentiableAt le_rfl) hyF

/-- The actual commutator of the pulled-back ambient connection equals the
checked curved-coordinate commutator for arbitrary coordinate fields. -/
theorem curvature_normalMap (S : Set E) (epsilon a R : ℝ)
    (C N T : ℝ → E) (k : ℝ → ℝ) (V : Set ℝ) (x : Coord)
    (X Y Z : Coord → Coord)
    (hpatch : NormalPatch S R C N V) (hV : V ∈ 𝓝 x.2)
    (he : 0 < epsilon) (hr : |x.1| < R) (hh : 1 - k x.2 * x.1 ≠ 0)
    (hf : ContDiffAt ℝ 2 (normalMap C N) x)
    (hi : (fderiv ℝ (normalMap C N) x).IsInvertible)
    (hk : ContDiffAt ℝ 1 k x.2)
    (hF : ∀ᶠ s in 𝓝 x.2,
      HasDerivAt C (T s) s ∧ HasDerivAt N (-(k s) • T s) s ∧
        ‖T s‖ = 1 ∧ inner ℝ (N s) (T s) = 0) :
    CoordinateCovariance.curvature
      (NonlinearCoordinates.pullbackConnection (normalMap C N) (ambientConnection S epsilon a))
      X Y Z x = CurvedShoreGeometry.curvature epsilon a k X Y Z x := by
  exact CoordinateCovariance.curvature_congr_connection _ _ X Y Z x
    (eventually_ambientConnection_normalMap S epsilon a R C N T k V x hpatch hV
      he hr hh hf hi hk hF)

omit [CompleteSpace E] in
/-- The actual ambient metric has the curved metric as its derivative pullback.
Ambient differentiability is obtained from the same actual normal chart. -/
theorem ambientGraphMetric_normalMap (S : Set E) (epsilon a R : ℝ)
    (C N T : ℝ → E) (k : ℝ → ℝ) (V : Set ℝ) (x v w : Coord)
    (hpatch : NormalPatch S R C N V) (hV : V ∈ 𝓝 x.2)
    (he : 0 < epsilon) (hr : |x.1| < R)
    (hf : ContDiffAt ℝ 2 (normalMap C N) x)
    (hi : (fderiv ℝ (normalMap C N) x).IsInvertible)
    (hF : HasDerivAt C (T x.2) x.2 ∧ HasDerivAt N (-(k x.2) • T x.2) x.2 ∧
      ‖T x.2‖ = 1 ∧ inner ℝ (N x.2) (T x.2) = 0) :
    NonlinearCoordinates.pullbackMetric (normalMap C N) (ambientGraphMetric S epsilon a)
      x v w = CurvedShoreGeometry.metric epsilon a (k x.2) x.1 v w := by
  have hR := contDiffAt_ambientRuler_of_normalMap S epsilon R C N V hpatch x hV hr he one_le_two hf hi
  have hchain : fderiv ℝ (normalRuler S epsilon C N) x =
      (fderiv ℝ (fun y => ruler epsilon (Metric.infDist y S)) (normalMap C N x)).comp
        (fderiv ℝ (normalMap C N) x) :=
    ((hR.differentiableAt one_le_two).hasFDerivAt.comp x
      (hf.differentiableAt one_le_two).hasFDerivAt).fderiv
  rw [← EmbeddedNormalPatch.normalGraphMetric_eq S epsilon a R C N V hpatch x v w
    (T x.2) (k x.2) hV he hr hF.1 hF.2.1 hF.2.2.1 hF.2.2.2]
  simp only [NonlinearCoordinates.pullbackMetric, ambientGraphMetric, normalGraphMetric,
    hchain, ContinuousLinearMap.comp_apply, normalRuler]

/-- Curvature/Gram ratio formed from the actual ambient metric and actual
ambient connection, both transported by the actual normal-map derivative. -/
noncomputable def ambientNormalCurvature (S : Set E) (epsilon a : ℝ)
    (C N : ℝ → E) (x : Coord) : ℝ :=
  CoordinateCovariance.gramRatio
    (NonlinearCoordinates.pullbackMetric (normalMap C N) (ambientGraphMetric S epsilon a))
    (NonlinearCoordinates.pullbackConnection (normalMap C N) (ambientConnection S epsilon a))
    (fun _ => (1, 0)) (fun _ => (0, 1)) x

/-- The actual ambient connection/metric ratio is the previously checked
curved-coordinate sectional curvature. -/
theorem ambientNormalCurvature_eq_coordinate (S : Set E) (epsilon a R : ℝ)
    (C N T : ℝ → E) (k : ℝ → ℝ) (V : Set ℝ) (x : Coord)
    (hpatch : NormalPatch S R C N V) (hV : V ∈ 𝓝 x.2)
    (he : 0 < epsilon) (hr : |x.1| < R) (hh : 1 - k x.2 * x.1 ≠ 0)
    (hf : ContDiffAt ℝ 2 (normalMap C N) x)
    (hi : (fderiv ℝ (normalMap C N) x).IsInvertible)
    (hk : ContDiffAt ℝ 1 k x.2)
    (hF : ∀ᶠ s in 𝓝 x.2,
      HasDerivAt C (T s) s ∧ HasDerivAt N (-(k s) • T s) s ∧
        ‖T s‖ = 1 ∧ inner ℝ (N s) (T s) = 0) :
    ambientNormalCurvature S epsilon a C N x =
      CurvedShoreGeometry.sectionalCurvature epsilon a k x := by
  rw [ambientNormalCurvature, CoordinateCovariance.gramRatio,
    curvature_normalMap S epsilon a R C N T k V x _ _ _ hpatch hV he hr hh hf hi hk hF]
  simp only [CoordinateCovariance.gram,
    ambientGraphMetric_normalMap S epsilon a R C N T k V x _ _ hpatch hV he hr hf hi hF.self_of_nhds,
    CurvedShoreGeometry.sectionalCurvature]

omit [CompleteSpace E] in
/-- The actual pulled-back metric has a strictly positive Gram denominator
in the normal/tangential coordinate frame at every nonfocal point. -/
theorem normalGram_pos (S : Set E) (epsilon a R : ℝ)
    (C N T : ℝ → E) (k : ℝ → ℝ) (V : Set ℝ) (x : Coord)
    (hpatch : NormalPatch S R C N V) (hV : V ∈ 𝓝 x.2)
    (he : 0 < epsilon) (hr : |x.1| < R) (hh : 1 - k x.2 * x.1 ≠ 0)
    (hf : ContDiffAt ℝ 2 (normalMap C N) x)
    (hi : (fderiv ℝ (normalMap C N) x).IsInvertible)
    (hF : HasDerivAt C (T x.2) x.2 ∧ HasDerivAt N (-(k x.2) • T x.2) x.2 ∧
      ‖T x.2‖ = 1 ∧ inner ℝ (N x.2) (T x.2) = 0) :
    0 < CoordinateCovariance.gram
      (NonlinearCoordinates.pullbackMetric (normalMap C N) (ambientGraphMetric S epsilon a))
        x (1, 0) (0, 1) := by
  simp only [CoordinateCovariance.gram,
    ambientGraphMetric_normalMap S epsilon a R C N T k V x _ _ hpatch hV he hr hf hi hF,
    CurvedShoreGeometry.metric_gram_determinant]
  simpa only [mul_one, mul_zero, sub_zero, one_pow] using
    mul_pos (mul_pos (FlatShoreGeometry.radialMetric_pos epsilon a x.1 he)
      (FlatShoreGeometry.tangentialMetric_pos epsilon x.1 he)) (sq_pos_of_ne_zero hh)

/-- The actual ambient connection/metric ratio in a C³ normal map has the
full curved-Shore formula; the supplied curvature profile is C² locally. -/
theorem ambientNormalCurvature_eq_flat_correction (S : Set E) (epsilon a R : ℝ)
    (C N T : ℝ → E) (k : ℝ → ℝ) (V : Set ℝ) (x : Coord)
    (hpatch : NormalPatch S R C N V) (hV : V ∈ 𝓝 x.2)
    (he : 0 < epsilon) (hr : |x.1| < R) (hh : 1 - k x.2 * x.1 ≠ 0)
    (hf : ContDiffAt ℝ 3 (normalMap C N) x)
    (hi : (fderiv ℝ (normalMap C N) x).IsInvertible)
    (hk : ContDiffAt ℝ 2 k x.2)
    (hF : ∀ᶠ s in 𝓝 x.2,
      HasDerivAt C (T s) s ∧ HasDerivAt N (-(k s) • T s) s ∧
        ‖T s‖ = 1 ∧ inner ℝ (N s) (T s) = 0) :
    ambientNormalCurvature S epsilon a C N x =
      FlatShore.radialCurvature epsilon a x.1 -
        k x.2 * (1 + a ^ 2) * x.1 * (epsilon ^ 2 + x.1 ^ 2) ^ 2 /
          ((1 - k x.2 * x.1) * (epsilon ^ 2 + (1 + a ^ 2) * x.1 ^ 2) ^ 2) := by
  have hdk : DifferentiableAt ℝ (deriv k) x.2 := by
    simpa only [fderiv_deriv] using
      (((hk.fderiv_right (m := 1) (by norm_num)).clm_apply
        (contDiffAt_const (c := (1 : ℝ)))).differentiableAt le_rfl)
  rw [ambientNormalCurvature_eq_coordinate S epsilon a R C N T k V x hpatch hV he hr hh
    (hf.of_le (by norm_num)) hi (hk.of_le (by norm_num)) hF]
  exact CurvedShoreGeometry.sectionalCurvature_eq_flat_correction epsilon a x.1 x.2
    (deriv k x.2) (deriv (deriv k) x.2) k he hh
    (hk.differentiableAt one_le_two).hasDerivAt hdk.hasDerivAt

/-- Every represented smooth Shore point has curvature ratio +1 for the
actual ambient metric connection, independently of the Euclidean curvature. -/
theorem ambientNormalCurvature_on_shore (S : Set E) (epsilon a R s : ℝ)
    (C N T : ℝ → E) (k : ℝ → ℝ) (V : Set ℝ)
    (hpatch : NormalPatch S R C N V) (hV : V ∈ 𝓝 s) (he : 0 < epsilon)
    (hf : ContDiffAt ℝ 3 (normalMap C N) (0, s))
    (hi : (fderiv ℝ (normalMap C N) (0, s)).IsInvertible)
    (hk : ContDiffAt ℝ 2 k s)
    (hF : ∀ᶠ t in 𝓝 s,
      HasDerivAt C (T t) t ∧ HasDerivAt N (-(k t) • T t) t ∧
        ‖T t‖ = 1 ∧ inner ℝ (N t) (T t) = 0) :
    ambientNormalCurvature S epsilon a C N (0, s) = 1 := by
  simpa [FlatShore.radialCurvature_zero epsilon a he] using
    ambientNormalCurvature_eq_flat_correction S epsilon a R C N T k V (0, s) hpatch hV
      he (by simpa using hpatch.radius_pos) (by simp) hf hi hk hF

/-- The normal-coordinate curvature is the actual ambient-coordinate
commutator transported by the inverse Jacobian. Intermediate covariant-field
differentiability follows from actual C³ ruler regularity. -/
theorem ambient_curvature_pullback (S : Set E) (epsilon a R : ℝ)
    (C N T : ℝ → E) (k : ℝ → ℝ) (V : Set ℝ) (x : Coord)
    (X Y Z : E → E)
    (hpatch : NormalPatch S R C N V) (hV : V ∈ 𝓝 x.2)
    (he : 0 < epsilon) (hr : |x.1| < R) (hh : 1 - k x.2 * x.1 ≠ 0)
    (hf : ContDiffAt ℝ 3 (normalMap C N) x)
    (hi : (fderiv ℝ (normalMap C N) x).IsInvertible)
    (hk : ContDiffAt ℝ 1 k x.2)
    (hF : ∀ᶠ s in 𝓝 x.2,
      HasDerivAt C (T s) s ∧ HasDerivAt N (-(k s) • T s) s ∧
        ‖T s‖ = 1 ∧ inner ℝ (N s) (T s) = 0)
    (hX : ContDiffAt ℝ 1 X (normalMap C N x))
    (hY : ContDiffAt ℝ 1 Y (normalMap C N x))
    (hZ : ContDiffAt ℝ 2 Z (normalMap C N x)) :
    CurvedShoreGeometry.curvature epsilon a k
      (VectorField.pullback ℝ (normalMap C N) X)
      (VectorField.pullback ℝ (normalMap C N) Y)
      (VectorField.pullback ℝ (normalMap C N) Z) x =
        VectorField.pullback ℝ (normalMap C N)
          (CoordinateCovariance.curvature (ambientConnection S epsilon a) X Y Z) x := by
  have hR := contDiffAt_ambientRuler_of_normalMap S epsilon R C N V hpatch x hV hr he
    (by norm_num) hf hi
  rw [← curvature_normalMap S epsilon a R C N T k V x _ _ _ hpatch hV he hr hh
    (hf.of_le (by norm_num)) hi hk hF]
  exact NonlinearCoordinates.curvature_pullback (normalMap C N) (ambientConnection S epsilon a)
    X Y Z x (hf.of_le (by norm_num)) hi (hX.differentiableAt le_rfl) (hY.differentiableAt le_rfl)
    ((hZ.eventually (by norm_num)).mono fun _ hz => hz.differentiableAt one_le_two)
    ((contDiffAt_gradientConnection_covariantDerivative _ a Y Z _ hR hY hZ
      (ruler_pos epsilon _ he).ne').differentiableAt le_rfl)
    ((contDiffAt_gradientConnection_covariantDerivative _ a X Z _ hR hX hZ
      (ruler_pos epsilon _ he).ne').differentiableAt le_rfl)

/-- Curvature/Gram ratios agree with the actual ambient-coordinate ratio for
regular ambient fields. All intermediate differentiability is derived. -/
theorem ambient_gramRatio_pullback (S : Set E) (epsilon a R : ℝ)
    (C N T : ℝ → E) (k : ℝ → ℝ) (V : Set ℝ) (x : Coord) (X Y : E → E)
    (hpatch : NormalPatch S R C N V) (hV : V ∈ 𝓝 x.2)
    (he : 0 < epsilon) (hr : |x.1| < R) (hh : 1 - k x.2 * x.1 ≠ 0)
    (hf : ContDiffAt ℝ 3 (normalMap C N) x)
    (hi : (fderiv ℝ (normalMap C N) x).IsInvertible)
    (hk : ContDiffAt ℝ 1 k x.2)
    (hF : ∀ᶠ s in 𝓝 x.2,
      HasDerivAt C (T s) s ∧ HasDerivAt N (-(k s) • T s) s ∧
        ‖T s‖ = 1 ∧ inner ℝ (N s) (T s) = 0)
    (hX : ContDiffAt ℝ 1 X (normalMap C N x))
    (hY : ContDiffAt ℝ 2 Y (normalMap C N x)) :
    CoordinateCovariance.gramRatio
      (fun y => CurvedShoreGeometry.metric epsilon a (k y.2) y.1)
      (CurvedShoreGeometry.christoffel epsilon a k)
      (VectorField.pullback ℝ (normalMap C N) X)
      (VectorField.pullback ℝ (normalMap C N) Y) x =
    CoordinateCovariance.gramRatio (ambientGraphMetric S epsilon a) (ambientConnection S epsilon a)
      X Y (normalMap C N x) := by
  have hR := contDiffAt_ambientRuler_of_normalMap S epsilon R C N V hpatch x hV hr he
    (by norm_num) hf hi
  have htransport := NonlinearCoordinates.gramRatio_pullback (normalMap C N)
    (ambientGraphMetric S epsilon a) (ambientConnection S epsilon a) X Y x
    (hf.of_le (by norm_num)) hi (hX.differentiableAt le_rfl)
    ((hY.eventually (by norm_num)).mono fun _ hy => hy.differentiableAt one_le_two)
    ((contDiffAt_gradientConnection_covariantDerivative _ a Y Y _ hR (hY.of_le one_le_two) hY
      (ruler_pos epsilon _ he).ne').differentiableAt le_rfl)
    ((contDiffAt_gradientConnection_covariantDerivative _ a X Y _ hR hX hY
      (ruler_pos epsilon _ he).ne').differentiableAt le_rfl)
  simpa only [CoordinateCovariance.gramRatio, CoordinateCovariance.gram,
    curvature_normalMap S epsilon a R C N T k V x _ _ _ hpatch hV he hr hh
      (hf.of_le (by norm_num)) hi hk hF,
    ambientGraphMetric_normalMap S epsilon a R C N T k V x _ _ hpatch hV he hr
      (hf.of_le (by norm_num)) hi hF.self_of_nhds] using htransport

end FractalSync.EmbeddedShoreConnection
