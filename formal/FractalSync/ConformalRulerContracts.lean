import FractalSync.ConformalFloorCurvature
import FractalSync.ContinuousMechanics
import FractalSync.CoordinateCovariance
import FractalSync.RulerDifferential
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv

/-!
# Conformal ruler comparisons and the flat normal model

The geometric distance comparison enters as a hypothesis. All transferred
bounds use the same positive floor and compare the same velocity or path.
-/

open scoped Interval InnerProductSpace

namespace FractalSync.ConformalRulerContracts

/-- Flooring the reciprocal of a positive density gives a bounded positive density. -/
theorem regularizedDensity (epsilon density : ℝ) (he : 0 < epsilon) (h : 0 < density) :
    (ruler epsilon density⁻¹)⁻¹ = density / ruler (epsilon * density) 1 := by
  have hr := ruler_pos epsilon density⁻¹ he
  have hrScaled := ruler_pos (epsilon * density) 1 (mul_pos he h)
  have scaled : ruler (epsilon * density) 1 = ruler epsilon density⁻¹ * density := by
    apply (sq_eq_sq₀ hrScaled.le (mul_nonneg hr.le h.le)).1
    rw [mul_pow, ruler_sq, ruler_sq]
    field_simp [h.ne']
  rw [scaled]
  field_simp [hr.ne', h.ne']

/-- A common quadratic floor preserves the pointwise factor-two ruler comparison. -/
theorem flooredRulerComparison (epsilon d r : ℝ) (he : 0 < epsilon)
    (hd : 0 ≤ d) (lower : d / 2 ≤ r) (upper : r ≤ 2 * d) :
    ruler epsilon d / 2 ≤ ruler epsilon r ∧ ruler epsilon r ≤ 2 * ruler epsilon d := by
  have hr : 0 ≤ r := by linarith
  have hdPos := ruler_pos epsilon d he
  have hrPos := ruler_pos epsilon r he
  have lowerSq := (sq_le_sq₀ (by positivity) hr).2 lower
  have upperSq := (sq_le_sq₀ hr (by positivity)).2 upper
  constructor
  · apply (sq_le_sq₀ (by positivity) hrPos.le).1
    rw [div_pow, ruler_sq, ruler_sq]
    nlinarith [sq_nonneg epsilon]
  · apply (sq_le_sq₀ hrPos.le (by positivity)).1
    rw [mul_pow, ruler_sq, ruler_sq]
    nlinarith [sq_nonneg epsilon]

/-- The same Euclidean velocity has conformal speeds within a factor of two. -/
theorem conformalSpeedComparison (epsilon d r v : ℝ) (he : 0 < epsilon)
    (hd : 0 ≤ d) (lower : d / 2 ≤ r) (upper : r ≤ 2 * d) (hv : 0 ≤ v) :
    (1 / 2) * (v / ruler epsilon d) ≤ v / ruler epsilon r ∧
      v / ruler epsilon r ≤ 2 * (v / ruler epsilon d) := by
  have ⟨lo, hi⟩ := flooredRulerComparison epsilon d r he hd lower upper
  have hdPos := ruler_pos epsilon d he
  have hrPos := ruler_pos epsilon r he
  constructor
  · have h := div_le_div_of_nonneg_left hv hrPos hi
    convert h using 1
    field_simp [hdPos.ne']
  · have h := div_le_div_of_nonneg_left hv (show 0 < ruler epsilon d / 2 by positivity) lo
    convert h using 1
    field_simp [hdPos.ne']

/-- At a fixed velocity the conformal kinetic energies are within a factor of four. -/
theorem conformalKineticComparison (epsilon d r v : ℝ) (he : 0 < epsilon)
    (hd : 0 ≤ d) (lower : d / 2 ≤ r) (upper : r ≤ 2 * d) (hv : 0 ≤ v) :
    (1 / 4) * ((v / ruler epsilon d) ^ 2 / 2) ≤ (v / ruler epsilon r) ^ 2 / 2 ∧
      (v / ruler epsilon r) ^ 2 / 2 ≤ 4 * ((v / ruler epsilon d) ^ 2 / 2) := by
  have ⟨lo, hi⟩ := conformalSpeedComparison epsilon d r v he hd lower upper hv
  have hdPos := ruler_pos epsilon d he
  have hrPos := ruler_pos epsilon r he
  have lowerSq := (sq_le_sq₀ (by positivity) (by positivity)).2 lo
  have upperSq := (sq_le_sq₀ (by positivity) (by positivity)).2 hi
  constructor <;> nlinarith

/-- Integrating the pointwise conformal speeds compares lengths of the same path. -/
theorem conformalLengthComparison (epsilon a b : ℝ) (d r v : ℝ → ℝ)
    (he : 0 < epsilon) (hab : a ≤ b)
    (bounds : ∀ t ∈ Set.Icc a b, 0 ≤ d t ∧ d t / 2 ≤ r t ∧ r t ≤ 2 * d t ∧ 0 ≤ v t)
    (distanceIntegrable : IntervalIntegrable (fun t => v t / ruler epsilon (d t))
      MeasureTheory.volume a b)
    (rulerIntegrable : IntervalIntegrable (fun t => v t / ruler epsilon (r t))
      MeasureTheory.volume a b) :
    (1 / 2) * (∫ t in a..b, v t / ruler epsilon (d t)) ≤
      (∫ t in a..b, v t / ruler epsilon (r t)) ∧
    (∫ t in a..b, v t / ruler epsilon (r t)) ≤
      2 * (∫ t in a..b, v t / ruler epsilon (d t)) := by
  simp only [← intervalIntegral.integral_const_mul]
  constructor
  · apply intervalIntegral.integral_mono_on hab (distanceIntegrable.const_mul _) rulerIntegrable
    intro t ht
    obtain ⟨hd, lower, upper, hv⟩ := bounds t ht
    exact (conformalSpeedComparison epsilon (d t) (r t) (v t) he hd lower upper hv).1
  · apply intervalIntegral.integral_mono_on hab rulerIntegrable (distanceIntegrable.const_mul _)
    intro t ht
    obtain ⟨hd, lower, upper, hv⟩ := bounds t ht
    exact (conformalSpeedComparison epsilon (d t) (r t) (v t) he hd lower upper hv).2

/-- A positive floor bounds conformal length by Euclidean length divided by the floor. -/
theorem flooredLength_le (epsilon a b : ℝ) (r v : ℝ → ℝ) (he : 0 < epsilon) (hab : a ≤ b)
    (nonnegative : ∀ t ∈ Set.Icc a b, 0 ≤ v t)
    (euclideanIntegrable : IntervalIntegrable v MeasureTheory.volume a b)
    (conformalIntegrable : IntervalIntegrable (fun t => v t / ruler epsilon (r t))
      MeasureTheory.volume a b) :
    (∫ t in a..b, v t / ruler epsilon (r t)) ≤ (∫ t in a..b, v t) / epsilon := by
  rw [div_eq_mul_inv, mul_comm, ← intervalIntegral.integral_const_mul]
  apply intervalIntegral.integral_mono_on hab conformalIntegrable (euclideanIntegrable.const_mul _)
  intro t ht
  rw [mul_comm, ← div_eq_mul_inv]
  exact div_le_div_of_nonneg_left (nonnegative t ht) he (epsilon_le_ruler epsilon (r t) he.le)

/-- The shared reference length cancels in the one-octave scale comparison. -/
theorem scaleCoordinateComparison (epsilon dref d r : ℝ) (he : 0 < epsilon)
    (href : 0 < dref) (hd : 0 ≤ d) (lower : d / 2 ≤ r) (upper : r ≤ 2 * d) :
    |scaleCoordinate dref (ruler epsilon r) - scaleCoordinate dref (ruler epsilon d)| ≤ 1 := by
  have ⟨lo, hi⟩ := flooredRulerComparison epsilon d r he hd lower upper
  have hdPos := ruler_pos epsilon d he
  have hrPos := ruler_pos epsilon r he
  have logPos : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have logLower := Real.log_le_log (show 0 < ruler epsilon d / 2 by positivity) lo
  have logUpper := Real.log_le_log hrPos hi
  rw [Real.log_div hdPos.ne' (by norm_num)] at logLower
  rw [Real.log_mul (by norm_num) hdPos.ne'] at logUpper
  rw [scaleCoordinate_eq_log_sub _ _ href hrPos, scaleCoordinate_eq_log_sub _ _ href hdPos,
    ← sub_div, abs_div, abs_of_pos logPos, div_le_iff₀ logPos, one_mul, abs_le]
  constructor <;> linarith

section ConformalMetric

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The graph metric at zero anisotropy is precisely the conformal metric. -/
theorem conformalMetric (rho : ℝ) (gradient v w : E) :
    ScaleMetric.metric rho 0 gradient v w = ⟪v, w⟫_ℝ / rho ^ 2 := by
  simp [ScaleMetric.metric]

/-- The inverse action at zero anisotropy multiplies by the ruler squared. -/
theorem conformalInverse (rho : ℝ) (gradient v : E) :
    ScaleMetric.inverseAction rho 0 gradient v = rho ^ 2 • v := by
  simp [ScaleMetric.inverseAction, ScaleMetric.unstretch]

/-- The conformal pairing is positive on nonzero vectors for a positive ruler. -/
theorem conformalPositive (rho : ℝ) (v : E) (hrho : 0 < rho) (hv : v ≠ 0) :
    0 < ⟪v, v⟫_ℝ / rho ^ 2 := by
  rw [← conformalMetric rho v v v]
  exact ScaleMetric.metric_self_pos rho 0 v v hrho hv

/-- The kinetic comparison uses exactly the zero-anisotropy specialization. -/
theorem conformalKinetic (rho : ℝ) (gradient v : E) :
    ScaleMetric.kinetic rho 0 gradient v = (‖v‖ / rho) ^ 2 / 2 := by
  rw [ScaleMetric.kinetic, conformalMetric, real_inner_self_eq_norm_sq, div_pow]

/-- Only the ruler and its gradient enter the conformal connection. -/
theorem conformalConnection (rho : ℝ) (gradient : E) (H : E →L[ℝ] E) (v w : E) :
    ScaleMetric.connection rho 0 gradient H v w =
      (⟪v, w⟫_ℝ / rho) • gradient - (⟪gradient, w⟫_ℝ / rho) • v -
        (⟪gradient, v⟫_ℝ / rho) • w := by
  simp only [ScaleMetric.connection, ScaleMetric.weight, zero_pow two_ne_zero, zero_mul,
    zero_div, zero_smul, add_zero, mul_one, sub_eq_add_neg]
  abel

end ConformalMetric

/-- The planar conformal metric in Cartesian coordinates. -/
noncomputable def planarMetric (rho : ℝ × ℝ → ℝ) (x v w : ℝ × ℝ) : ℝ :=
  (v.1 * w.1 + v.2 * w.2) / rho x ^ 2

/-- The Cartesian conformal connection with log-ruler gradient components `p` and `q`. -/
noncomputable def planarConnection (p q : ℝ × ℝ → ℝ) (x v w : ℝ × ℝ) : ℝ × ℝ :=
  let diagonal := v.1 * w.1 - v.2 * w.2
  let mixed := v.1 * w.2 + v.2 * w.1
  (-p x * diagonal - q x * mixed, q x * diagonal - p x * mixed)

/-- The actual curvature commutator divided by the metric Gram determinant is the weighted
trace of the derivative of the log-gradient. `p` and `q` are the Cartesian partial derivatives
of `log rho` when applying this identity to the conformal Levi-Civita connection. -/
theorem planarCurvatureTrace (rho p q : ℝ × ℝ → ℝ) (x : ℝ × ℝ)
    (dp dq : (ℝ × ℝ) →L[ℝ] ℝ) (hp : HasFDerivAt p dp x) (hq : HasFDerivAt q dq x)
    (hrho : 0 < rho x) :
    CoordinateCovariance.gramRatio (planarMetric rho) (planarConnection p q)
      (fun _ => (1, 0)) (fun _ => (0, 1)) x =
        rho x ^ 2 * (dp (1, 0) + dq (0, 1)) := by
  let X : ℝ × ℝ → ℝ × ℝ := fun _ => (1, 0)
  let Y : ℝ × ℝ → ℝ × ℝ := fun _ => (0, 1)
  let Gamma := planarConnection p q
  change CoordinateCovariance.gramRatio (planarMetric rho) Gamma X Y x = _
  have yy : CoordinateCovariance.covariantDerivative Gamma Y Y = fun y => (p y, -q y) := by
    funext y
    simp [CoordinateCovariance.covariantDerivative, Gamma, planarConnection, Y]
  have xy : CoordinateCovariance.covariantDerivative Gamma X Y = fun y => (-q y, -p y) := by
    funext y
    simp [CoordinateCovariance.covariantDerivative, Gamma, planarConnection, X, Y]
  have bracket : VectorField.lieBracket ℝ X Y = 0 := by
    funext y
    simp [VectorField.lieBracket, X, Y]
  have dyy : fderiv ℝ (fun y => (p y, -q y)) x = dp.prod (-dq) :=
    (hp.prodMk hq.neg).fderiv
  have dxy : fderiv ℝ (fun y => (-q y, -p y)) x = (-dq).prod (-dp) :=
    (hq.neg.prodMk hp.neg).fderiv
  unfold CoordinateCovariance.gramRatio CoordinateCovariance.gram CoordinateCovariance.curvature
  rw [yy, xy, bracket]
  simp only [CoordinateCovariance.covariantDerivative, dyy, dxy]
  simp [X, Y, Gamma, planarConnection, planarMetric]
  field_simp [hrho.ne']

/-- The planar trace formula for the actual second Fréchet derivative of `log rho`. -/
theorem planarCurvatureOfLogRuler (rho : ℝ × ℝ → ℝ) (x : ℝ × ℝ)
    (H : (ℝ × ℝ) →L[ℝ] ((ℝ × ℝ) →L[ℝ] ℝ)) (hrho : 0 < rho x)
    (hH : HasFDerivAt (fderiv ℝ (fun y => Real.log (rho y))) H x) :
    let logGradient := fderiv ℝ (fun y => Real.log (rho y))
    CoordinateCovariance.gramRatio (planarMetric rho)
      (planarConnection (fun y => logGradient y (1, 0)) (fun y => logGradient y (0, 1)))
      (fun _ => (1, 0)) (fun _ => (0, 1)) x =
        rho x ^ 2 * (H (1, 0) (1, 0) + H (0, 1) (0, 1)) := by
  let logGradient := fderiv ℝ (fun y => Real.log (rho y))
  have component (u : ℝ × ℝ) :
      HasFDerivAt (fun y => logGradient y u) (H.flip u) x := by
    simpa using hH.clm_apply (hasFDerivAt_const (c := u) (x := x))
  exact planarCurvatureTrace rho _ _ x (H.flip (1, 0)) (H.flip (0, 1))
    (component (1, 0)) (component (0, 1)) hrho

/-- Squaring the ideal ruler removes the absolute value at the Shore. -/
theorem idealNormalRuler_eq (epsilon slope normal : ℝ) :
    ruler epsilon (slope * |normal|) = ruler epsilon (slope * normal) := by
  simp only [ruler, mul_pow, sq_abs]

/-- The actual log-floor derivative in the ideal normal coordinate, including at zero. -/
theorem hasDerivAt_idealNormalLogRuler (epsilon slope normal : ℝ) (he : 0 < epsilon) :
    HasDerivAt (fun n => Real.log (ruler epsilon (slope * |n|)))
      ((slope * normal / ((slope * normal) ^ 2 + epsilon ^ 2)) * slope) normal := by
  simp_rw [idealNormalRuler_eq]
  exact (ConformalFloorCurvature.hasDerivAt_log_floor epsilon (slope * normal) he).comp normal
    (by simpa using (hasDerivAt_id normal).const_mul slope)

/-- The actual second log-floor derivative in the ideal normal coordinate. -/
theorem hasDerivAt_deriv_idealNormalLogRuler (epsilon slope normal : ℝ) (he : 0 < epsilon) :
    HasDerivAt (deriv (fun n => Real.log (ruler epsilon (slope * |n|))))
      ((((epsilon ^ 2 - (slope * normal) ^ 2) /
        ((slope * normal) ^ 2 + epsilon ^ 2) ^ 2) * slope) * slope) normal := by
  rw [show deriv (fun n => Real.log (ruler epsilon (slope * |n|))) = _ from
    funext fun n => (hasDerivAt_idealNormalLogRuler epsilon slope n he).deriv]
  exact ((ConformalFloorCurvature.hasDerivAt_log_floor_slope epsilon (slope * normal) he).comp
    normal (by simpa using (hasDerivAt_id normal).const_mul slope)).mul_const slope

/-- The ideal normal model's weighted log Laplacian is its curvature coefficient. -/
theorem idealNormalCurvatureCoefficient (epsilon slope normal : ℝ) (he : 0 < epsilon) :
    (ruler epsilon (slope * |normal|)) ^ 2 *
        deriv (deriv (fun n => Real.log (ruler epsilon (slope * |n|)))) normal =
      slope ^ 2 * (epsilon ^ 2 - slope ^ 2 * normal ^ 2) /
        (epsilon ^ 2 + slope ^ 2 * normal ^ 2) := by
  rw [(hasDerivAt_deriv_idealNormalLogRuler epsilon slope normal he).deriv,
    idealNormalRuler_eq, ruler_sq]
  have denom : (slope * normal) ^ 2 + epsilon ^ 2 ≠ 0 := by positivity
  have denom' : epsilon ^ 2 + slope ^ 2 * normal ^ 2 ≠ 0 := by positivity
  field_simp [denom, denom']
  ring

/-- The ideal normal profile gives this curvature for the actual planar commutator. -/
theorem idealPlanarCurvature (epsilon slope : ℝ) (x : ℝ × ℝ) (he : 0 < epsilon) :
    let rho : ℝ × ℝ → ℝ := fun y => ruler epsilon (slope * |y.1|)
    let logGradient := fderiv ℝ (fun y => Real.log (rho y))
    CoordinateCovariance.gramRatio (planarMetric rho)
      (planarConnection (fun y => logGradient y (1, 0)) (fun y => logGradient y (0, 1)))
      (fun _ => (1, 0)) (fun _ => (0, 1)) x =
      slope ^ 2 * (epsilon ^ 2 - slope ^ 2 * x.1 ^ 2) / (epsilon ^ 2 + slope ^ 2 * x.1 ^ 2) := by
  let rho : ℝ × ℝ → ℝ := fun y => ruler epsilon (slope * |y.1|)
  let profile := fun n : ℝ => Real.log (ruler epsilon (slope * |n|))
  let projection := ContinuousLinearMap.fst ℝ ℝ ℝ
  have logGradient (y : ℝ × ℝ) :
      fderiv ℝ (fun z => Real.log (rho z)) y = deriv profile y.1 • projection := by
    have hy := (hasDerivAt_idealNormalLogRuler epsilon slope y.1 he).differentiableAt.hasDerivAt
    exact (hy.comp_hasFDerivAt y projection.hasFDerivAt).fderiv
  have dp : HasFDerivAt (fun y : ℝ × ℝ => deriv profile y.1)
      (deriv (deriv profile) x.1 • projection) x := by
    have hx := (hasDerivAt_deriv_idealNormalLogRuler epsilon slope x.1 he).differentiableAt.hasDerivAt
    exact hx.comp_hasFDerivAt x projection.hasFDerivAt
  have dq : HasFDerivAt (fun _ : ℝ × ℝ => (0 : ℝ)) 0 x :=
    hasFDerivAt_const (𝕜 := ℝ) (c := (0 : ℝ)) (x := x)
  change CoordinateCovariance.gramRatio (planarMetric rho)
    (planarConnection (fun y => fderiv ℝ (fun z => Real.log (rho z)) y (1, 0))
      (fun y => fderiv ℝ (fun z => Real.log (rho z)) y (0, 1))) _ _ x = _
  have first : projection (1, 0) = 1 := rfl
  have second : projection (0, 1) = 0 := rfl
  simp only [logGradient, ContinuousLinearMap.smul_apply, first, second,
    smul_eq_mul, mul_one, mul_zero]
  rw [planarCurvatureTrace _ _ _ _ _ _ dp dq (ruler_pos epsilon _ he)]
  simp only [ContinuousLinearMap.smul_apply, smul_eq_mul, first, mul_one,
    ContinuousLinearMap.zero_apply, add_zero]
  exact idealNormalCurvatureCoefficient epsilon slope x.1 he

/-- The actual normal Jacobi profile along the ideal Shore satisfies the harmonic equation. -/
theorem idealNormalJacobiEquation (slope s : ℝ) :
    deriv (deriv (fun u => Real.sin (slope * u))) s =
      -slope ^ 2 * Real.sin (slope * s) := by
  have hFirst : deriv (fun u => Real.sin (slope * u)) =
      fun u => Real.cos (slope * u) * slope := by
    funext u
    exact ((Real.hasDerivAt_sin (slope * u)).comp u
      (by simpa using (hasDerivAt_id u).const_mul slope)).deriv
  rw [hFirst]
  convert (((Real.hasDerivAt_cos (slope * s)).comp s
    (by simpa using (hasDerivAt_id s).const_mul slope)).mul_const slope).deriv using 1
  ring

/-- The first positive zero has intrinsic arclength `π / slope`. -/
theorem idealNormalJacobiFirstZero (slope : ℝ) (hs : 0 < slope) :
    Real.sin (slope * (Real.pi / slope)) = 0 ∧
      ∀ s ∈ Set.Ioo 0 (Real.pi / slope), Real.sin (slope * s) ≠ 0 := by
  constructor
  · rw [mul_div_cancel₀ Real.pi hs.ne', Real.sin_pi]
  · intro s ht
    apply (Real.sin_pos_of_pos_of_lt_pi (mul_pos hs ht.1) ?_).ne'
    rw [mul_comm]
    exact (lt_div_iff₀ hs).1 ht.2

/-- The first positive zero in tangential distance is `π * epsilon / slope`. -/
theorem idealNormalJacobiEuclideanSpan (epsilon slope : ℝ) (he : 0 < epsilon) (hs : 0 < slope) :
    Real.sin (slope * ((Real.pi * epsilon / slope) / epsilon)) = 0 ∧
      ∀ y ∈ Set.Ioo 0 (Real.pi * epsilon / slope), Real.sin (slope * (y / epsilon)) ≠ 0 := by
  obtain ⟨zero, before⟩ := idealNormalJacobiFirstZero slope hs
  have span : Real.pi * epsilon / slope = (Real.pi / slope) * epsilon := by ring
  constructor
  · rwa [span, mul_div_cancel_right₀ _ he.ne']
  · intro y hy
    apply before (y / epsilon)
    exact ⟨div_pos hy.1 he, (div_lt_iff₀ he).2 (hy.2.trans_eq span)⟩

end FractalSync.ConformalRulerContracts
