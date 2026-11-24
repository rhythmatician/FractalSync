import FractalSync.Ruler
import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Analysis.Calculus.FDeriv.Mul
import Mathlib.Tactic.LinearCombination

/-!
# Ruler Hessians and logarithmic scale coordinates

The second derivative is a continuous linear map into continuous linear maps.
These results differentiate actual fields; no smoothness follows merely from
positive regularization. In particular, the Hessian statements state the local
differentiability of the underlying field and the derivative of its differential.
-/

namespace FractalSync

open Filter
open scoped Topology

/-- The position-induced base-two scale, with reference ruler `dref`. -/
noncomputable def scaleCoordinate (dref rho : ℝ) : ℝ :=
  Real.log (dref / rho) / Real.log 2

theorem scaleCoordinate_eq_log_sub (dref rho : ℝ)
    (hd : 0 < dref) (hr : 0 < rho) :
    scaleCoordinate dref rho = (Real.log dref - Real.log rho) / Real.log 2 := by
  rw [scaleCoordinate, Real.log_div hd.ne' hr.ne']

theorem hasDerivAt_scaleCoordinate (dref rho : ℝ)
    (hd : 0 < dref) (hr : 0 < rho) :
    HasDerivAt (scaleCoordinate dref) (-(1 / (rho * Real.log 2))) rho := by
  have hlog : Real.log 2 ≠ 0 := ne_of_gt (Real.log_pos (by norm_num))
  convert (((hasDerivAt_const rho dref).div (hasDerivAt_id rho) hr.ne').log
    (div_ne_zero hd.ne' hr.ne')).div_const (Real.log 2) using 1
  simp only [id_eq, zero_mul, mul_one, zero_sub, Pi.div_apply]
  field_simp

theorem hasDerivAt_scaleSlope (rho : ℝ) (hr : 0 < rho) :
    HasDerivAt (fun r => -(1 / (r * Real.log 2)))
      (1 / (rho ^ 2 * Real.log 2)) rho := by
  have hlog : Real.log 2 ≠ 0 := ne_of_gt (Real.log_pos (by norm_num))
  convert (((hasDerivAt_id rho).mul_const (Real.log 2)).inv
    (mul_ne_zero hr.ne' hlog)).neg using 1
  · ext r
    simp only [Pi.neg_apply, Pi.inv_apply, one_div, id_eq]
  · simp only [id_eq, one_mul]
    field_simp [hr.ne', hlog]

/-- Changing the reference shifts scale by a constant, independently of the
position-dependent ruler. -/
theorem scaleCoordinate_reference_shift (dref rho alpha : ℝ)
    (hd : 0 < dref) (hr : 0 < rho) (ha : 0 < alpha) :
    scaleCoordinate (alpha * dref) rho =
      scaleCoordinate dref rho + Real.log alpha / Real.log 2 := by
  rw [scaleCoordinate_eq_log_sub _ _ (mul_pos ha hd) hr,
    scaleCoordinate_eq_log_sub _ _ hd hr, Real.log_mul ha.ne' hd.ne']
  ring

/-- Scale is order-reversing in a positive ruler. -/
theorem scaleCoordinate_le_iff (dref rho tau : ℝ)
    (hd : 0 < dref) (hr : 0 < rho) (ht : 0 < tau) :
    scaleCoordinate dref rho ≤ scaleCoordinate dref tau ↔ tau ≤ rho := by
  rw [scaleCoordinate_eq_log_sub _ _ hd hr,
    scaleCoordinate_eq_log_sub _ _ hd ht,
    div_le_div_iff_of_pos_right (Real.log_pos (by norm_num)),
    sub_le_sub_iff_left, Real.log_le_log_iff ht hr]

/-- The regularization floor gives the finite upper scale bound. -/
theorem scaleCoordinate_ruler_le_summit (epsilon dref r : ℝ)
    (he : 0 < epsilon) (hd : 0 < dref) :
    scaleCoordinate dref (ruler epsilon r) ≤ scaleCoordinate dref epsilon := by
  exact (scaleCoordinate_le_iff dref (ruler epsilon r) epsilon hd
    (ruler_pos epsilon r he) he).2 (epsilon_le_ruler epsilon r he.le)

theorem ruler_zero (epsilon : ℝ) (he : 0 ≤ epsilon) : ruler epsilon 0 = epsilon := by
  simp [ruler, Real.sqrt_sq_eq_abs, abs_of_nonneg he]

/-- The finite summit is attained precisely on the zero level of the field. -/
theorem scaleCoordinate_ruler_eq_summit_iff (epsilon dref r : ℝ)
    (he : 0 < epsilon) (hd : 0 < dref) :
    scaleCoordinate dref (ruler epsilon r) = scaleCoordinate dref epsilon ↔ r = 0 := by
  constructor
  · intro h
    have hl := (scaleCoordinate_le_iff dref epsilon (ruler epsilon r) hd he
      (ruler_pos epsilon r he)).1 h.ge
    have hr := epsilon_le_ruler epsilon r he.le
    have heq : ruler epsilon r = epsilon := le_antisymm hl hr
    have hs := ruler_sq epsilon r
    rw [heq] at hs
    nlinarith [sq_nonneg r]
  · rintro rfl
    rw [ruler_zero epsilon he.le]

/-- Scale derivative expressed directly in the unregularized scalar field. -/
theorem hasDerivAt_scaleCoordinate_ruler (epsilon dref r : ℝ)
    (he : 0 < epsilon) (hd : 0 < dref) :
    HasDerivAt (fun s => scaleCoordinate dref (ruler epsilon s))
      (-(r / ((ruler epsilon r) ^ 2 * Real.log 2))) r := by
  convert (hasDerivAt_scaleCoordinate dref (ruler epsilon r) hd
    (ruler_pos epsilon r he)).comp r (hasDerivAt_ruler epsilon r he) using 1
  ring

/-- Second scalar derivative of scale with respect to the underlying field. -/
theorem hasDerivAt_scaleRulerSlope (epsilon r : ℝ) (he : 0 < epsilon) :
    HasDerivAt (fun s => -(s / ((ruler epsilon s) ^ 2 * Real.log 2)))
      ((r ^ 2 - epsilon ^ 2) / ((ruler epsilon r) ^ 4 * Real.log 2)) r := by
  have hr := (ruler_pos epsilon r he).ne'
  have hlog : Real.log 2 ≠ 0 := ne_of_gt (Real.log_pos (by norm_num))
  convert ((hasDerivAt_id r).div
    (((hasDerivAt_ruler epsilon r he).pow 2).mul_const (Real.log 2))
    (mul_ne_zero (pow_ne_zero 2 hr) hlog)).neg using 1
  simp only [Nat.cast_ofNat, one_mul, id_eq, Pi.pow_apply]
  field_simp [hr, hlog]
  norm_num only [Nat.reduceSub, pow_one]
  linear_combination (ruler epsilon r) * ruler_sq epsilon r

section Differential

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The actual derivative of the ruler differential field. The rank-one term
is a continuous bilinear map, not merely a coordinate formula. -/
theorem hasFDerivAt_ruler_differential (epsilon : ℝ) (he : 0 < epsilon)
    (D : E → ℝ) (D' : E → E →L[ℝ] ℝ) (D'' : E →L[ℝ] E →L[ℝ] ℝ) (x : E)
    (hD : HasFDerivAt D (D' x) x) (hD' : HasFDerivAt D' D'' x) :
    HasFDerivAt (fun y => (D y / ruler epsilon (D y)) • D' y)
      ((D x / ruler epsilon (D x)) • D'' +
       (epsilon ^ 2 / (ruler epsilon (D x)) ^ 3) •
         (D' x).smulRight (D' x)) x := by
  convert ((hasDerivAt_ruler_slope epsilon (D x) he).comp_hasFDerivAt x hD).smul hD'
    using 1
  ext u v
  simp
  ring

/-- Full second Fréchet derivative of the regularized ruler field. The local
first-derivative premise ensures that the differentiated field really is the
`fderiv` of the composite. -/
theorem hasFDerivAt_fderiv_ruler_comp (epsilon : ℝ) (he : 0 < epsilon)
    (D : E → ℝ) (D'' : E →L[ℝ] E →L[ℝ] ℝ) (x : E)
    (hD : ∀ᶠ y in 𝓝 x, DifferentiableAt ℝ D y)
    (hD' : HasFDerivAt (fderiv ℝ D) D'' x) :
    HasFDerivAt (fderiv ℝ (fun y => ruler epsilon (D y)))
      ((D x / ruler epsilon (D x)) • D'' +
       (epsilon ^ 2 / (ruler epsilon (D x)) ^ 3) •
         (fderiv ℝ D x).smulRight (fderiv ℝ D x)) x := by
  have hDx : DifferentiableAt ℝ D x := Filter.Eventually.self_of_nhds hD
  apply (hasFDerivAt_ruler_differential epsilon he D (fderiv ℝ D) D'' x
    hDx.hasFDerivAt hD').congr_of_eventuallyEq
  filter_upwards [hD] with y hy
  exact (hasFDerivAt_ruler_comp epsilon he D (fderiv ℝ D y) y hy.hasFDerivAt).fderiv

/-- Logarithmic scale differential for an arbitrary differentiable positive
ruler field. -/
theorem hasFDerivAt_scaleCoordinate_comp (dref : ℝ) (hd : 0 < dref)
    (rho : E → ℝ) (rho' : E →L[ℝ] ℝ) (x : E)
    (hr : 0 < rho x) (hR : HasFDerivAt rho rho' x) :
    HasFDerivAt (fun y => scaleCoordinate dref (rho y))
      (-(1 / (rho x * Real.log 2)) • rho') x := by
  exact (hasDerivAt_scaleCoordinate dref (rho x) hd hr).comp_hasFDerivAt x hR

/-- Derivative of the scale differential field, including the positive
rank-one correction to the negative ruler Hessian. -/
theorem hasFDerivAt_scale_differential
    (rho : E → ℝ) (rho' : E → E →L[ℝ] ℝ) (rho'' : E →L[ℝ] E →L[ℝ] ℝ) (x : E)
    (hr : 0 < rho x) (hR : HasFDerivAt rho (rho' x) x)
    (hR' : HasFDerivAt rho' rho'' x) :
    HasFDerivAt (fun y => -(1 / (rho y * Real.log 2)) • rho' y)
      (-(1 / (rho x * Real.log 2)) • rho'' +
       (1 / ((rho x) ^ 2 * Real.log 2)) •
         (rho' x).smulRight (rho' x)) x := by
  convert ((hasDerivAt_scaleSlope (rho x) hr).comp_hasFDerivAt x hR).smul hR' using 1
  ext u v
  simp
  ring

/-- Full second Fréchet derivative of a logarithmic scale field. Positivity is
needed only at the point: continuity supplies a positive neighborhood. -/
theorem hasFDerivAt_fderiv_scaleCoordinate_comp (dref : ℝ) (hd : 0 < dref)
    (rho : E → ℝ) (rho'' : E →L[ℝ] E →L[ℝ] ℝ) (x : E)
    (hr : 0 < rho x) (hR : ∀ᶠ y in 𝓝 x, DifferentiableAt ℝ rho y)
    (hR' : HasFDerivAt (fderiv ℝ rho) rho'' x) :
    HasFDerivAt (fderiv ℝ (fun y => scaleCoordinate dref (rho y)))
      (-(1 / (rho x * Real.log 2)) • rho'' +
       (1 / ((rho x) ^ 2 * Real.log 2)) •
         (fderiv ℝ rho x).smulRight (fderiv ℝ rho x)) x := by
  have hRx : DifferentiableAt ℝ rho x := Filter.Eventually.self_of_nhds hR
  have hpos : ∀ᶠ y in 𝓝 x, 0 < rho y :=
    continuousAt_const.eventually_lt hRx.continuousAt hr
  apply (hasFDerivAt_scale_differential rho (fderiv ℝ rho) rho'' x hr
    hRx.hasFDerivAt hR').congr_of_eventuallyEq
  filter_upwards [hR, hpos] with y hy hpy
  exact (hasFDerivAt_scaleCoordinate_comp dref hd rho (fderiv ℝ rho y) y hpy
    hy.hasFDerivAt).fderiv

/-- First differential of scale expressed in the original field `D`. -/
theorem hasFDerivAt_scaleCoordinate_ruler_comp (epsilon dref : ℝ)
    (he : 0 < epsilon) (hd : 0 < dref)
    (D : E → ℝ) (D' : E →L[ℝ] ℝ) (x : E) (hD : HasFDerivAt D D' x) :
    HasFDerivAt (fun y => scaleCoordinate dref (ruler epsilon (D y)))
      (-(D x / ((ruler epsilon (D x)) ^ 2 * Real.log 2)) • D') x := by
  exact (hasDerivAt_scaleCoordinate_ruler epsilon dref (D x) he hd).comp_hasFDerivAt x hD

/-- Derivative of the scale differential field in unregularized field jets. -/
theorem hasFDerivAt_scale_ruler_differential (epsilon : ℝ) (he : 0 < epsilon)
    (D : E → ℝ) (D' : E → E →L[ℝ] ℝ) (D'' : E →L[ℝ] E →L[ℝ] ℝ) (x : E)
    (hD : HasFDerivAt D (D' x) x) (hD' : HasFDerivAt D' D'' x) :
    HasFDerivAt (fun y => -(D y / ((ruler epsilon (D y)) ^ 2 * Real.log 2)) • D' y)
      (-(D x / ((ruler epsilon (D x)) ^ 2 * Real.log 2)) • D'' +
       ((D x ^ 2 - epsilon ^ 2) / ((ruler epsilon (D x)) ^ 4 * Real.log 2)) •
         (D' x).smulRight (D' x)) x := by
  convert ((hasDerivAt_scaleRulerSlope epsilon (D x) he).comp_hasFDerivAt x hD).smul hD'
    using 1
  ext u v
  simp
  ring

/-- Full scale Hessian directly in `D`, its differential, and its Hessian. -/
theorem hasFDerivAt_fderiv_scaleCoordinate_ruler_comp (epsilon dref : ℝ)
    (he : 0 < epsilon) (hd : 0 < dref)
    (D : E → ℝ) (D'' : E →L[ℝ] E →L[ℝ] ℝ) (x : E)
    (hD : ∀ᶠ y in 𝓝 x, DifferentiableAt ℝ D y)
    (hD' : HasFDerivAt (fderiv ℝ D) D'' x) :
    HasFDerivAt (fderiv ℝ (fun y => scaleCoordinate dref (ruler epsilon (D y))))
      (-(D x / ((ruler epsilon (D x)) ^ 2 * Real.log 2)) • D'' +
       ((D x ^ 2 - epsilon ^ 2) / ((ruler epsilon (D x)) ^ 4 * Real.log 2)) •
         (fderiv ℝ D x).smulRight (fderiv ℝ D x)) x := by
  have hDx : DifferentiableAt ℝ D x := Filter.Eventually.self_of_nhds hD
  apply (hasFDerivAt_scale_ruler_differential epsilon he D (fderiv ℝ D) D'' x
    hDx.hasFDerivAt hD').congr_of_eventuallyEq
  filter_upwards [hD] with y hy
  exact (hasFDerivAt_scaleCoordinate_ruler_comp epsilon dref he hd D (fderiv ℝ D y) y
    hy.hasFDerivAt).fderiv

/-- At a smooth zero level, the ruler Hessian has rank at most one. The Hessian
of the original field drops out of this formula. -/
theorem hasFDerivAt_fderiv_ruler_on_zero (epsilon : ℝ) (he : 0 < epsilon)
    (D : E → ℝ) (D'' : E →L[ℝ] E →L[ℝ] ℝ) (x : E)
    (hD : ∀ᶠ y in 𝓝 x, DifferentiableAt ℝ D y)
    (hD' : HasFDerivAt (fderiv ℝ D) D'' x) (hz : D x = 0) :
    HasFDerivAt (fderiv ℝ (fun y => ruler epsilon (D y)))
      ((1 / epsilon) • (fderiv ℝ D x).smulRight (fderiv ℝ D x)) x := by
  have hcoef : epsilon ^ 2 / epsilon ^ 3 = 1 / epsilon := by field_simp
  convert hasFDerivAt_fderiv_ruler_comp epsilon he D D'' x hD hD' using 1
  ext u v
  simp [hz, ruler_zero epsilon he.le, hcoef]

/-- At a smooth zero level the scale Hessian is nonpositive and has rank at most one, independently
of the field's Hessian. Tangent directions in the kernel of `dD` are flat to
second order. -/
theorem hasFDerivAt_fderiv_scale_on_zero (epsilon dref : ℝ)
    (he : 0 < epsilon) (hd : 0 < dref)
    (D : E → ℝ) (D'' : E →L[ℝ] E →L[ℝ] ℝ) (x : E)
    (hD : ∀ᶠ y in 𝓝 x, DifferentiableAt ℝ D y)
    (hD' : HasFDerivAt (fderiv ℝ D) D'' x) (hz : D x = 0) :
    HasFDerivAt (fderiv ℝ (fun y => scaleCoordinate dref (ruler epsilon (D y))))
      (-(1 / (epsilon ^ 2 * Real.log 2)) •
        (fderiv ℝ D x).smulRight (fderiv ℝ D x)) x := by
  have hcoef : -(epsilon ^ 2) / (epsilon ^ 4 * Real.log 2) =
      -(1 / (epsilon ^ 2 * Real.log 2)) := by field_simp
  convert hasFDerivAt_fderiv_scaleCoordinate_ruler_comp epsilon dref he hd D D'' x hD hD'
    using 1
  ext u v
  simp [hz, ruler_zero epsilon he.le, hcoef]

end Differential

end FractalSync
