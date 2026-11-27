import FractalSync.ControlMechanics

/-!
# Steering and transport in a moving ruler metric

The connection term is included in the actual derivative of a heading along
a moving configuration. The statements prove preservation under the specified
ODE; they do not assert existence of its solutions or verify a discrete update.
-/

open Set
open scoped InnerProductSpace

namespace FractalSync.ScaleMetric

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The oriented quarter-turn in a metric orthonormal two-frame. Without frame
normalization the same expression is still metric-skew, which suffices for
the norm-preservation theorem. -/
noncomputable def frameTurn (rho a : ℝ) (u e f d : E) : E :=
  metric rho a u e d • f - metric rho a u f d • e

theorem frameTurn_orthogonal (rho a : ℝ) (u e f d : E) :
    metric rho a u d (frameTurn rho a u e f d) = 0 := by
  simp only [frameTurn, metric, inner_sub_right, real_inner_smul_right,
    real_inner_comm d e, real_inner_comm d f]
  ring

theorem frameTurn_first (rho a : ℝ) (u e f : E)
    (he : metric rho a u e e = 1) (hef : metric rho a u e f = 0) :
    frameTurn rho a u e f e = f := by
  simp [frameTurn, he, metric_symm rho a u f e, hef]

theorem frameTurn_second (rho a : ℝ) (u e f : E)
    (hf : metric rho a u f f = 1) (hef : metric rho a u e f = 0) :
    frameTurn rho a u e f f = -e := by
  simp [frameTurn, hf, hef]

theorem frameTurn_mix (rho a c s : ℝ) (u e f v w : E) :
    frameTurn rho a u e f (c • v + s • w) =
      c • frameTurn rho a u e f v + s • frameTurn rho a u e f w := by
  simp only [frameTurn, metric_add_right, metric_smul_right, add_smul, mul_smul]
  module

/-- Along a moving configuration, actual differentiation pairs the covariant
rates of both vector traces. The base velocity need not equal either vector. -/
theorem hasDerivAt_metric_pairing_covariant (a t : ℝ) (rho : ℝ → ℝ)
    (u v w : ℝ → E) (baseVelocity dv dw : E) (H : E →L[ℝ] E)
    (hr : HasDerivAt rho ⟪u t, baseVelocity⟫_ℝ t)
    (hu : HasDerivAt u (H baseVelocity) t)
    (hv : HasDerivAt v dv t) (hw : HasDerivAt w dw t) (hrho : rho t ≠ 0) :
    HasDerivAt (fun s => metric (rho s) a (u s) (v s) (w s))
      (metric (rho t) a (u t)
          (dv + connection (rho t) a (u t) H baseVelocity (v t)) (w t) +
        metric (rho t) a (u t) (v t)
          (dw + connection (rho t) a (u t) H baseVelocity (w t))) t := by
  convert hasDerivAt_metric_pairing a t rho u v w baseVelocity dv dw H hr hu hv hw hrho using 1
  rw [connection_metric_compatible (rho t) a (u t) H baseVelocity (v t) (w t) hrho,
    metric_add_left, metric_add_right]
  ring

/-- Covariant steering has zero actual metric-norm-square derivative even
when the ruler, gradient, and heading all vary along the base trajectory. -/
theorem hasDerivAt_metric_self_steering (a t omega : ℝ) (rho : ℝ → ℝ)
    (u d : ℝ → E) (baseVelocity dd e f : E) (H : E →L[ℝ] E)
    (hr : HasDerivAt rho ⟪u t, baseVelocity⟫_ℝ t)
    (hu : HasDerivAt u (H baseVelocity) t) (hd : HasDerivAt d dd t)
    (hrho : rho t ≠ 0)
    (hmotion : dd + connection (rho t) a (u t) H baseVelocity (d t) =
      omega • frameTurn (rho t) a (u t) e f (d t)) :
    HasDerivAt (fun s => metric (rho s) a (u s) (d s) (d s)) 0 t := by
  have h := hasDerivAt_metric_pairing_covariant a t rho u d d baseVelocity dd dd H
    hr hu hd hd hrho
  simpa only [hmotion, metric_smul_left, metric_smul_right,
    metric_symm (rho t) a (u t) (frameTurn (rho t) a (u t) e f (d t)) (d t),
    frameTurn_orthogonal, mul_zero, zero_add] using h

/-- Integrated preservation. Frames and turn rates may vary with time; their
pointwise metric-skew action is enough. Existence of an ODE solution is not assumed
as a theorem: an actual differentiable heading satisfying the ODE is supplied. -/
theorem metric_self_steering_eq (a lo hi : ℝ) (rho omega : ℝ → ℝ)
    (u d baseVelocity dd e f : ℝ → E) (H : ℝ → E →L[ℝ] E)
    (horder : lo ≤ hi)
    (hcont : ContinuousOn (fun s => metric (rho s) a (u s) (d s) (d s)) (Icc lo hi))
    (hr : ∀ t ∈ Ioo lo hi, HasDerivAt rho ⟪u t, baseVelocity t⟫_ℝ t)
    (hu : ∀ t ∈ Ioo lo hi, HasDerivAt u (H t (baseVelocity t)) t)
    (hd : ∀ t ∈ Ioo lo hi, HasDerivAt d (dd t) t)
    (hrho : ∀ t ∈ Ioo lo hi, rho t ≠ 0)
    (hmotion : ∀ t ∈ Ioo lo hi,
      dd t + connection (rho t) a (u t) (H t) (baseVelocity t) (d t) =
        omega t • frameTurn (rho t) a (u t) (e t) (f t) (d t)) :
    metric (rho hi) a (u hi) (d hi) (d hi) =
      metric (rho lo) a (u lo) (d lo) (d lo) := by
  have h := intervalIntegral.integral_eq_sub_of_hasDerivAt_of_le horder hcont
    (fun t ht => hasDerivAt_metric_self_steering a t (omega t) rho u d
      (baseVelocity t) (dd t) (e t) (f t) (H t)
      (hr t ht) (hu t ht) (hd t ht) (hrho t ht) (hmotion t ht))
    (intervalIntegrable_const : IntervalIntegrable (fun _ : ℝ => (0 : ℝ)) MeasureTheory.volume lo hi)
  exact sub_eq_zero.mp (by simpa using h.symm)

/-- Field-level steering: actual position, ruler, and gradient derivatives
discharge the trace calculus used by the moving-metric theorem. -/
theorem hasDerivAt_metric_self_steering_of_fields (a t omega : ℝ)
    (rho : E → ℝ) (gradient : E → E) (gamma d : ℝ → E)
    (baseVelocity dd e f : E) (H : E →L[ℝ] E)
    (hgamma : HasDerivAt gamma baseVelocity t)
    (hr : HasFDerivAt rho (innerSL ℝ (gradient (gamma t))) (gamma t))
    (hu : HasFDerivAt gradient H (gamma t)) (hd : HasDerivAt d dd t)
    (hrho : rho (gamma t) ≠ 0)
    (hmotion : dd + connection (rho (gamma t)) a (gradient (gamma t)) H baseVelocity (d t) =
      omega • frameTurn (rho (gamma t)) a (gradient (gamma t)) e f (d t)) :
    HasDerivAt (fun s => metric (rho (gamma s)) a (gradient (gamma s)) (d s) (d s)) 0 t := by
  exact hasDerivAt_metric_self_steering a t omega (rho ∘ gamma) (gradient ∘ gamma) d
    baseVelocity dd e f H (hr.comp_hasDerivAt t hgamma) (hu.comp_hasDerivAt t hgamma)
    hd hrho hmotion

end FractalSync.ScaleMetric
