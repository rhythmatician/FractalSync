import FractalSync.CovariantSteering
import Mathlib.Analysis.SpecialFunctions.Sqrt

/-!
# Endpoint-normalized covariant Euler steering

A concrete exact-real direction update is normalized in the endpoint metric.
Actual derivative calculations establish its first-order consistency with the
covariant steering ODE. It is not asserted to be exact parallel transport or
an implementation of the current runtime controls.
-/

open Set
open scoped InnerProductSpace

namespace FractalSync.ScaleMetric

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Normalize a nonzero vector in the actual metric at the supplied endpoint. -/
noncomputable def metricNormalize (rho a : ℝ) (u d : E) : E :=
  (Real.sqrt (metric rho a u d d))⁻¹ • d

/-- The actual coordinate rate associated with a frozen covariant steering law. -/
noncomputable def covariantSteeringRate (rho a omega : ℝ)
    (u baseVelocity e f d : E) (H : E →L[ℝ] E) : E :=
  omega • frameTurn rho a u e f d - connection rho a u H baseVelocity d

/-- One explicit covariant Euler trial followed by normalization in the new metric. -/
noncomputable def covariantSteeringStep (a dt rho0 rho1 omega : ℝ)
    (u0 u1 baseVelocity e f d : E) (H : E →L[ℝ] E) : E :=
  metricNormalize rho1 a u1
    (d + dt • covariantSteeringRate rho0 a omega u0 baseVelocity e f d H)

/-- At a unit vector with zero metric-norm-square derivative, endpoint
normalization does not change the actual first derivative. -/
theorem hasDerivAt_metricNormalize (a : ℝ) (rho : ℝ → ℝ) (u d : ℝ → E)
    (dd : E) (hd : HasDerivAt d dd 0)
    (hunit : metric (rho 0) a (u 0) (d 0) (d 0) = 1)
    (henergy : HasDerivAt (fun t => metric (rho t) a (u t) (d t) (d t)) 0 0) :
    HasDerivAt (fun t => metricNormalize (rho t) a (u t) (d t)) dd 0 := by
  have hs := henergy.sqrt (by rw [hunit]; norm_num)
  have hi := hs.inv (by rw [hunit]; norm_num)
  simpa [metricNormalize, hunit] using hi.smul hd

/-- The endpoint-normalized explicit update has the actual covariant ODE rate
as its derivative at zero step, for genuine ruler/gradient trace derivatives. -/
theorem hasDerivAt_covariantSteeringStep (a omega : ℝ)
    (rho : ℝ → ℝ) (u : ℝ → E) (baseVelocity e f d : E) (H : E →L[ℝ] E)
    (hr : HasDerivAt rho ⟪u 0, baseVelocity⟫_ℝ 0)
    (hu : HasDerivAt u (H baseVelocity) 0) (hrho : rho 0 ≠ 0)
    (hunit : metric (rho 0) a (u 0) d d = 1) :
    HasDerivAt
      (fun t => covariantSteeringStep a t (rho 0) (rho t) omega
        (u 0) (u t) baseVelocity e f d H)
      (covariantSteeringRate (rho 0) a omega (u 0) baseVelocity e f d H) 0 := by
  let k := covariantSteeringRate (rho 0) a omega (u 0) baseVelocity e f d H
  have htrial : HasDerivAt (fun t : ℝ => d + t • k) k 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const k).const_add d
  have henergy := hasDerivAt_metric_self_steering a 0 omega rho u
    (fun t : ℝ => d + t • k) baseVelocity k e f H hr hu htrial hrho
    (by simp [k, covariantSteeringRate])
  simpa only [covariantSteeringStep, k] using
    hasDerivAt_metricNormalize a rho u (fun t : ℝ => d + t • k) k htrial
      (by simpa using hunit) henergy

/-- Actual normalization has metric norm one whenever its squared norm is positive. -/
theorem metricNormalize_unit (rho a : ℝ) (u d : E)
    (hpos : 0 < metric rho a u d d) :
    metric rho a u (metricNormalize rho a u d) (metricNormalize rho a u d) = 1 := by
  rw [metricNormalize, metric_smul_left, metric_smul_right]
  rw [← mul_assoc, ← pow_two, inv_pow, Real.sq_sqrt hpos.le, inv_mul_cancel₀ hpos.ne']

/-- Unit inputs are unchanged by normalization. -/
theorem metricNormalize_eq_self (rho a : ℝ) (u d : E)
    (hunit : metric rho a u d d = 1) : metricNormalize rho a u d = d := by
  simp [metricNormalize, hunit]

/-- A strict step budget prevents the Euler trial from vanishing. -/
theorem steering_trial_ne_zero (dt : ℝ) (d k : E)
    (hstep : |dt| * ‖k‖ < ‖d‖) : d + dt • k ≠ 0 := by
  intro hz
  have hn := congrArg norm (eq_neg_of_add_eq_zero_left hz)
  exact (ne_of_gt hstep) (by simpa [norm_smul] using hn)

/-- The concrete endpoint update is exactly metric unit under a positive
endpoint ruler and a nonvanishing trial. -/
theorem covariantSteeringStep_unit (a dt rho0 rho1 omega : ℝ)
    (u0 u1 baseVelocity e f d : E) (H : E →L[ℝ] E) (hrho : 0 < rho1)
    (hstep : |dt| * ‖covariantSteeringRate rho0 a omega u0 baseVelocity e f d H‖ < ‖d‖) :
    metric rho1 a u1 (covariantSteeringStep a dt rho0 rho1 omega u0 u1 baseVelocity e f d H)
      (covariantSteeringStep a dt rho0 rho1 omega u0 u1 baseVelocity e f d H) = 1 := by
  exact metricNormalize_unit rho1 a u1 _
    (metric_self_pos rho1 a u1 _ hrho (steering_trial_ne_zero dt d _ hstep))

/-- A zero step fixes an initially unit direction. -/
theorem covariantSteeringStep_zero (a rho omega : ℝ)
    (u baseVelocity e f d : E) (H : E →L[ℝ] E)
    (hunit : metric rho a u d d = 1) :
    covariantSteeringStep a 0 rho rho omega u u baseVelocity e f d H = d := by
  simpa only [covariantSteeringStep, zero_smul, add_zero] using
    metricNormalize_eq_self rho a u d hunit

/-- Actual field derivatives and a differentiable base curve discharge the
trace hypotheses in the discrete consistency theorem. -/
theorem hasDerivAt_covariantSteeringStep_of_fields (a omega : ℝ)
    (rho : E → ℝ) (gamma : ℝ → E) (rulerGradient : E → E)
    (baseVelocity e f d : E) (H : E →L[ℝ] E)
    (hgamma : HasDerivAt gamma baseVelocity 0)
    (hr : HasFDerivAt rho (innerSL ℝ (rulerGradient (gamma 0))) (gamma 0))
    (hu : HasFDerivAt rulerGradient H (gamma 0)) (hrho : rho (gamma 0) ≠ 0)
    (hunit : metric (rho (gamma 0)) a (rulerGradient (gamma 0)) d d = 1) :
    HasDerivAt
      (fun t => covariantSteeringStep a t (rho (gamma 0)) (rho (gamma t)) omega
        (rulerGradient (gamma 0)) (rulerGradient (gamma t)) baseVelocity e f d H)
      (covariantSteeringRate (rho (gamma 0)) a omega
        (rulerGradient (gamma 0)) baseVelocity e f d H) 0 := by
  exact hasDerivAt_covariantSteeringStep a omega (rho ∘ gamma) (rulerGradient ∘ gamma)
    baseVelocity e f d H (hr.comp_hasDerivAt 0 hgamma)
    (hu.comp_hasDerivAt 0 hgamma) hrho hunit

end FractalSync.ScaleMetric
