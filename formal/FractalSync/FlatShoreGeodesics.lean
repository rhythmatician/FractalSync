import FractalSync.FlatShoreIntrinsic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv

/-!
# Smooth flat-Shore geodesic and Jacobi contracts

These theorems concern actual differentiable solutions of the explicit
coordinate geodesic equation. They do not assume that every AC minimizer is
smooth, and do not identify conjugate points or a global cut locus.
-/

namespace FractalSync.FlatShoreGeodesics

open Set
open FlatShoreGeometry

/-- Actual kinetic energy of a coordinate velocity. -/
noncomputable def energy (epsilon a : ℝ) (q v : Coord) : ℝ :=
  metric epsilon a q.1 v v / 2

/-- Tangential momentum corresponding to translation in the y coordinate. -/
noncomputable def momentum (epsilon : ℝ) (q v : Coord) : ℝ :=
  tangentialMetric epsilon q.1 * v.2

/-- The actual smooth coordinate geodesic ODE at one parameter value. -/
def GeodesicAt (epsilon a : ℝ) (q v : ℝ → Coord) (t : ℝ) : Prop :=
  HasDerivAt q (v t) t ∧
    HasDerivAt v (-christoffel epsilon a (q t).1 (v t) (v t)) t

/-- Actual metric compatibility makes energy stationary along a geodesic. -/
theorem hasDerivAt_energy (epsilon a t : ℝ) (he : 0 < epsilon)
    (q v : ℝ → Coord) (h : GeodesicAt epsilon a q v t) :
    HasDerivAt (fun s => energy epsilon a (q s) (v s)) 0 t := by
  have hd := (hasDerivAt_metric_pairing epsilon a t (fun s => (q s).1)
    v v (v t) (-christoffel epsilon a (q t).1 (v t) (v t))
    (-christoffel epsilon a (q t).1 (v t) (v t)) he
    ((ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt.comp_hasDerivAt t h.1) h.2 h.2).div_const 2
  convert hd using 1
  rw [christoffel_metric_compatible epsilon a (q t).1 (v t) (v t) (v t) he]
  simp only [metric, Prod.fst_neg, Prod.snd_neg]
  ring

/-- Tangential momentum is differentiated from its actual metric coefficient. -/
theorem hasDerivAt_momentum (epsilon a t : ℝ) (he : 0 < epsilon)
    (q v : ℝ → Coord) (h : GeodesicAt epsilon a q v t) :
    HasDerivAt (fun s => momentum epsilon (q s) (v s)) 0 t := by
  convert (((hasDerivAt_tangentialMetric epsilon (q t).1 he).comp t
    ((ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt.comp_hasDerivAt t h.1)).mul
    ((ContinuousLinearMap.snd ℝ ℝ ℝ).hasFDerivAt.comp_hasDerivAt t h.2)) using 1
  change 0 = 2 * tangentialMetric epsilon (q t).1 * gammaRY epsilon (q t).1 *
      (v t).1 * (v t).2 + tangentialMetric epsilon (q t).1 *
        (-(gammaRY epsilon (q t).1 * ((v t).1 * (v t).2 + (v t).2 * (v t).1)))
  ring

/-- Energy is constant on any open interval carrying a differentiable geodesic solution. -/
theorem energy_constant (epsilon a lo hi s t : ℝ) (he : 0 < epsilon)
    (q v : ℝ → Coord) (h : ∀ u ∈ Ioo lo hi, GeodesicAt epsilon a q v u)
    (hs : s ∈ Ioo lo hi) (ht : t ∈ Ioo lo hi) :
    energy epsilon a (q s) (v s) = energy epsilon a (q t) (v t) := by
  exact isOpen_Ioo.is_const_of_deriv_eq_zero isPreconnected_Ioo
    (fun u hu => (hasDerivAt_energy epsilon a u he q v (h u hu)).differentiableAt.differentiableWithinAt)
    (fun u hu => (hasDerivAt_energy epsilon a u he q v (h u hu)).deriv) hs ht

/-- Clairaut's conserved tangential momentum follows from the actual coordinate ODE. -/
theorem momentum_constant (epsilon a lo hi s t : ℝ) (he : 0 < epsilon)
    (q v : ℝ → Coord) (h : ∀ u ∈ Ioo lo hi, GeodesicAt epsilon a q v u)
    (hs : s ∈ Ioo lo hi) (ht : t ∈ Ioo lo hi) :
    momentum epsilon (q s) (v s) = momentum epsilon (q t) (v t) := by
  exact isOpen_Ioo.is_const_of_deriv_eq_zero isPreconnected_Ioo
    (fun u hu => (hasDerivAt_momentum epsilon a u he q v (h u hu)).differentiableAt.differentiableWithinAt)
    (fun u hu => (hasDerivAt_momentum epsilon a u he q v (h u hu)).deriv) hs ht

/-- Momentum reconstructs the tangential velocity at every positive-floor point. -/
theorem tangent_velocity_eq_momentum (epsilon : ℝ) (he : 0 < epsilon) (q v : Coord) :
    v.2 = (epsilon ^ 2 + q.1 ^ 2) * momentum epsilon q v := by
  simp only [momentum, tangentialMetric]
  field_simp [(FlatShore.ruler_sq_pos epsilon q.1 he).ne']

/-- The conserved energy and momentum give the radial first integral. -/
theorem radial_first_integral (epsilon a : ℝ) (he : 0 < epsilon) (q v : Coord) :
    radialMetric epsilon a q.1 * v.1 ^ 2 =
      2 * energy epsilon a q v - (epsilon ^ 2 + q.1 ^ 2) * momentum epsilon q v ^ 2 := by
  simp only [energy, momentum, metric, tangentialMetric]
  field_simp [(FlatShore.ruler_sq_pos epsilon q.1 he).ne']
  ring

/-- The Shore line; positive epsilon makes this an intrinsic unit-speed parameterization. -/
def shoreCurve (epsilon y₀ t : ℝ) : Coord := (0, y₀ + epsilon * t)

def shoreTangent (epsilon : ℝ) : Coord := (0, epsilon)

def shoreNormal (epsilon : ℝ) : Coord := (epsilon, 0)

/-- Every connection coefficient vanishes along the flat Shore. -/
theorem christoffel_on_shore (epsilon a : ℝ) (v w : Coord) :
    christoffel epsilon a 0 v w = 0 := by
  simp [christoffel, gammaRR, gammaYY, gammaRY]

/-- The Shore curve has the asserted actual velocity. -/
theorem hasDerivAt_shoreCurve (epsilon y₀ t : ℝ) :
    HasDerivAt (shoreCurve epsilon y₀) (shoreTangent epsilon) t := by
  simpa only [shoreCurve, shoreTangent, mul_one] using
    (hasDerivAt_const t (0 : ℝ)).prodMk (((hasDerivAt_id t).const_mul epsilon).const_add y₀)

/-- The Shore line satisfies the actual coordinate geodesic equation. -/
theorem geodesicAt_shoreCurve (epsilon a y₀ t : ℝ) :
    GeodesicAt epsilon a (shoreCurve epsilon y₀) (fun _ => shoreTangent epsilon) t := by
  refine ⟨hasDerivAt_shoreCurve epsilon y₀ t, ?_⟩
  simpa only [shoreCurve, christoffel_on_shore, neg_zero] using hasDerivAt_const t (shoreTangent epsilon)

/-- The signed normal and tangential fields form a metric-orthonormal Shore frame. -/
theorem shore_frame_orthonormal (epsilon a : ℝ) (he : 0 < epsilon) :
    metric epsilon a 0 (shoreNormal epsilon) (shoreNormal epsilon) = 1 ∧
    metric epsilon a 0 (shoreTangent epsilon) (shoreTangent epsilon) = 1 ∧
    metric epsilon a 0 (shoreNormal epsilon) (shoreTangent epsilon) = 0 := by
  simp only [metric, radialMetric, tangentialMetric, shoreNormal, shoreTangent,
    zero_pow two_ne_zero, mul_zero, zero_mul, add_zero, zero_add]
  field_simp [he.ne']
  trivial

/-- The covariant derivative of an actual vector field along a coordinate curve. -/
noncomputable def alongDerivative (epsilon a : ℝ) (q J : ℝ → Coord) (t : ℝ) : Coord :=
  deriv J t + christoffel epsilon a (q t).1 (deriv q t) (J t)

/-- Along Shore the actual covariant derivative reduces to the ordinary one. -/
theorem alongDerivative_shore (epsilon a y₀ : ℝ) (J : ℝ → Coord) :
    alongDerivative epsilon a (shoreCurve epsilon y₀) J = deriv J := by
  funext t
  simp [alongDerivative, shoreCurve, christoffel_on_shore]

/-- The named normal field is parallel along the Shore geodesic; the frame theorem
provides its unit norm when epsilon is positive. -/
theorem shoreNormal_parallel (epsilon a y₀ : ℝ) :
    alongDerivative epsilon a (shoreCurve epsilon y₀) (fun _ => shoreNormal epsilon) = 0 := by
  rw [alongDerivative_shore]
  exact funext fun t => deriv_const t (shoreNormal epsilon)

/-- Actual differentiation of a normal profile in an arbitrary coordinate direction. -/
theorem fderiv_normalProfile (f : ℝ → ℝ) (x d : Coord) (f' : ℝ)
    (hf : HasDerivAt f f' x.1) :
    fderiv ℝ (fun p : Coord => (f p.1, (0 : ℝ))) x d = (f' * d.1, 0) := by
  have hd := (hf.comp_hasFDerivAt x (ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt).prodMk
    (hasFDerivAt_const (0 : ℝ) x)
  simpa using congrArg (fun L : Coord →L[ℝ] Coord => L d) hd.fderiv

/-- Actual differentiation of a tangential profile in an arbitrary direction. -/
theorem fderiv_tangentProfile (f : ℝ → ℝ) (x d : Coord) (f' : ℝ)
    (hf : HasDerivAt f f' x.1) :
    fderiv ℝ (fun p : Coord => ((0 : ℝ), f p.1)) x d = (0, f' * d.1) := by
  have hd := (hasFDerivAt_const (0 : ℝ) x).prodMk
    (hf.comp_hasFDerivAt x (ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt)
  simpa using congrArg (fun L : Coord →L[ℝ] Coord => L d) hd.fderiv

/-- Covariant differentiation of two constant coordinate fields retains the connection. -/
theorem covariantDerivative_const (epsilon a : ℝ) (v w : Coord) :
    covariantDerivative epsilon a (fun _ => v) (fun _ => w) =
      fun x => christoffel epsilon a x.1 v w := by
  funext x
  simp only [covariantDerivative, fderiv_const_apply, ContinuousLinearMap.zero_apply, zero_add]

/-- The actual curvature commutator on Shore for a normal vector and unit tangent. -/
theorem curvature_shore_normal_tangent (epsilon a y j : ℝ) (he : 0 < epsilon) :
    curvature epsilon a (fun _ => (j, 0)) (fun _ => shoreTangent epsilon)
      (fun _ => shoreTangent epsilon) (0, y) = (j, 0) := by
  rw [curvature, covariantDerivative_const, covariantDerivative_const]
  simp only [covariantDerivative, christoffel_on_shore, add_zero, fderiv_const_apply,
    ContinuousLinearMap.zero_apply, sub_zero]
  simp only [christoffel, shoreTangent, mul_zero, zero_mul, add_zero, zero_add]
  rw [fderiv_normalProfile (fun r => gammaYY epsilon a r * epsilon * epsilon)
    (0, y) (j, 0) _ (((hasDerivAt_gammaYY epsilon a 0 he).mul_const epsilon).mul_const epsilon)]
  rw [fderiv_tangentProfile (fun r => gammaRY epsilon r * (j * epsilon))
    (0, y) (0, epsilon) _ ((hasDerivAt_gammaRY epsilon 0 he).mul_const (j * epsilon))]
  simp only [zero_pow two_ne_zero, mul_zero, sub_zero, add_zero, Prod.mk_sub_mk]
  field_simp [he.ne']

/-- The coordinate Jacobi residual uses actual along-curve derivatives and the
actual curvature commutator evaluated on constant extensions of the vectors. -/
noncomputable def jacobiResidual (epsilon a : ℝ) (q J : ℝ → Coord) (t : ℝ) : Coord :=
  alongDerivative epsilon a q (alongDerivative epsilon a q J) t +
    curvature epsilon a (fun _ => J t) (fun _ => deriv q t) (fun _ => deriv q t) (q t)

/-- Along Shore the actual coordinate Jacobi equation for a normal profile
reduces to f''+f=0. The first and second derivatives are actual derivatives. -/
theorem normal_profile_jacobiResidual (epsilon a y₀ t f'' : ℝ) (he : 0 < epsilon)
    (f f' : ℝ → ℝ) (hf : ∀ s, HasDerivAt f (f' s) s) (hf' : HasDerivAt f' f'' t) :
    jacobiResidual epsilon a (shoreCurve epsilon y₀) (fun s => (epsilon * f s, 0)) t =
      (epsilon * (f'' + f t), 0) := by
  have hd : deriv (fun s => (epsilon * f s, (0 : ℝ))) =
      fun s => (epsilon * f' s, 0) := funext fun s =>
    (((hf s).const_mul epsilon).prodMk (hasDerivAt_const s (0 : ℝ))).deriv
  rw [jacobiResidual, alongDerivative_shore, alongDerivative_shore, hd,
    (hasDerivAt_shoreCurve epsilon y₀ t).deriv]
  rw [((hf'.const_mul epsilon).prodMk (hasDerivAt_const t (0 : ℝ))).deriv]
  simp only [shoreCurve, curvature_shore_normal_tangent epsilon a (y₀ + epsilon * t)
    (epsilon * f t) he, Prod.mk_add_mk, add_zero, mul_add]

/-- The normal coordinate Jacobi residual vanishes exactly when the scalar ODE does. -/
theorem normal_profile_jacobiResidual_eq_zero_iff (epsilon a y₀ t f'' : ℝ)
    (he : 0 < epsilon) (f f' : ℝ → ℝ) (hf : ∀ s, HasDerivAt f (f' s) s)
    (hf' : HasDerivAt f' f'' t) :
    jacobiResidual epsilon a (shoreCurve epsilon y₀) (fun s => (epsilon * f s, 0)) t = 0 ↔
      f'' + f t = 0 := by
  rw [normal_profile_jacobiResidual epsilon a y₀ t f'' he f f' hf hf']
  simp [Prod.mk_eq_zero, he.ne']

/-- The normal Jacobi candidate; its initial derivative is the named normal field. -/
noncomputable def shoreJacobi (epsilon t : ℝ) : Coord := (epsilon * Real.sin t, 0)

/-- Actual derivative of the normal Jacobi candidate. -/
theorem hasDerivAt_shoreJacobi (epsilon t : ℝ) :
    HasDerivAt (shoreJacobi epsilon) (epsilon * Real.cos t, 0) t := by
  exact ((Real.hasDerivAt_sin t).const_mul epsilon).prodMk (hasDerivAt_const t (0 : ℝ))

theorem deriv_shoreJacobi (epsilon : ℝ) :
    deriv (shoreJacobi epsilon) = fun t => (epsilon * Real.cos t, 0) := by
  exact funext fun t => (hasDerivAt_shoreJacobi epsilon t).deriv

/-- The second actual derivative is minus the field itself. -/
theorem deriv_deriv_shoreJacobi (epsilon t : ℝ) :
    deriv (deriv (shoreJacobi epsilon)) t = -shoreJacobi epsilon t := by
  rw [deriv_shoreJacobi]
  simpa only [shoreJacobi, Prod.neg_mk, neg_zero, mul_neg] using
    (((Real.hasDerivAt_cos t).const_mul epsilon).prodMk (hasDerivAt_const t (0 : ℝ))).deriv

/-- The actual coordinate Jacobi equation holds along the unit-speed Shore line. -/
theorem shoreJacobi_equation (epsilon a y₀ t : ℝ) (he : 0 < epsilon) :
    jacobiResidual epsilon a (shoreCurve epsilon y₀) (shoreJacobi epsilon) t = 0 := by
  rw [jacobiResidual, alongDerivative_shore, alongDerivative_shore,
    deriv_deriv_shoreJacobi, (hasDerivAt_shoreCurve epsilon y₀ t).deriv]
  simp only [shoreJacobi, shoreCurve,
    curvature_shore_normal_tangent epsilon a (y₀ + epsilon * t) (epsilon * Real.sin t) he,
    neg_add_cancel]

/-- The Jacobi field has zero initial value and the named normal initial derivative.
Positive epsilon gives the unit-norm conclusion through the frame theorem. -/
theorem shoreJacobi_initial_data (epsilon a y₀ : ℝ) :
    shoreJacobi epsilon 0 = 0 ∧
      alongDerivative epsilon a (shoreCurve epsilon y₀) (shoreJacobi epsilon) 0 =
        shoreNormal epsilon := by
  simp [alongDerivative_shore, deriv_shoreJacobi, shoreJacobi, shoreNormal]

/-- Its first positive zero is pi in intrinsic arclength. This is a field-zero
statement, not a bundled conjugate-point theorem. -/
theorem shoreJacobi_first_positive_zero (epsilon : ℝ) (he : 0 < epsilon) :
    shoreJacobi epsilon Real.pi = 0 ∧
      ∀ t ∈ Ioo (0 : ℝ) Real.pi, shoreJacobi epsilon t ≠ 0 := by
  constructor
  · simp [shoreJacobi]
  · exact fun t ht hz => (mul_pos he (Real.sin_pos_of_pos_of_lt_pi ht.1 ht.2)).ne'
      (congrArg Prod.fst hz)

/-- Signed tangential displacement at parameter pi is pi*epsilon. For positive
epsilon this is the Euclidean span of the intrinsic unit-speed Shore curve. -/
theorem shoreCurve_pi_span (epsilon y₀ : ℝ) :
    (shoreCurve epsilon y₀ Real.pi).2 - (shoreCurve epsilon y₀ 0).2 = Real.pi * epsilon := by
  simp only [shoreCurve, mul_zero, add_zero]
  ring

end FractalSync.FlatShoreGeodesics
