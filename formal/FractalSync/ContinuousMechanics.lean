import FractalSync.MetricImpulse
import Mathlib.Analysis.InnerProductSpace.Calculus
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus

/-!
# The local connection and continuous work-energy balance

Exact-real, development-time contracts for issue #159. The local-ruler metric
and inverse are the existing definitions from `ScaleMetric`. Differentiability
is stated explicitly; positive regularization alone does not provide a Hessian
at a distance-field cut locus. Nothing here concerns discrete integration.
-/

open scoped InnerProductSpace

namespace FractalSync.ScaleMetric

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The compact connection evaluated on two tangent vectors. The operator `H`
is the derivative of the ruler's gradient when used geometrically. -/
noncomputable def connection (rho a : ℝ) (u : E) (H : E →L[ℝ] E) (v w : E) : E :=
  (a ^ 2 * ⟪H v, w⟫_ℝ / weight a u) • u -
    (⟪u, w⟫_ℝ / rho) • v - (⟪u, v⟫_ℝ / rho) • w +
    ((⟪v, w⟫_ℝ + a ^ 2 * ⟪u, v⟫_ℝ * ⟪u, w⟫_ℝ) /
      (rho * weight a u)) • u

/-- The directional metric differential supplied by a ruler gradient and
gradient derivative. Actual calculus is identified with it below. -/
noncomputable def metricDifferential (rho a : ℝ) (u : E) (H : E →L[ℝ] E)
    (v w z : E) : ℝ :=
  -2 * (⟪u, v⟫_ℝ / rho) * metric rho a u w z +
    a ^ 2 / rho ^ 2 * (⟪H v, w⟫_ℝ * ⟪u, z⟫_ℝ +
      ⟪u, w⟫_ℝ * ⟪H v, z⟫_ℝ)

/-- A symmetric Hessian gives the torsion-free symmetry. -/
theorem connection_symm (rho a : ℝ) (u : E) (H : E →L[ℝ] E)
    (hH : ∀ v w, ⟪H v, w⟫_ℝ = ⟪H w, v⟫_ℝ) (v w : E) :
    connection rho a u H v w = connection rho a u H w v := by
  simp only [connection, hH v w, real_inner_comm v w]
  module

/-- Pairing the connection with the metric removes its rank-one inverse
denominator, exposing the first-kind connection coefficient. -/
theorem metric_connection (rho a : ℝ) (u : E) (H : E →L[ℝ] E)
    (v w z : E) (hrho : rho ≠ 0) :
    metric rho a u (connection rho a u H v w) z =
      a ^ 2 * ⟪H v, w⟫_ℝ * ⟪u, z⟫_ℝ / rho ^ 2 -
      (⟪u, w⟫_ℝ * metric rho a u v z +
        ⟪u, v⟫_ℝ * metric rho a u w z -
        ⟪u, z⟫_ℝ * metric rho a u v w) / rho := by
  simp only [connection, metric, inner_add_left, inner_sub_left,
    real_inner_smul_left, inner_add_right, inner_sub_right, real_inner_smul_right]
  have hW := ne_of_gt (weight_pos a u)
  field_simp
  simp only [weight]
  ring

/-- The compact connection is compatible with the differentiated metric.
Symmetry of H is not needed for this identity, only for torsion freedom. -/
theorem connection_metric_compatible (rho a : ℝ) (u : E) (H : E →L[ℝ] E)
    (v w z : E) (hrho : rho ≠ 0) :
    metricDifferential rho a u H v w z =
      metric rho a u (connection rho a u H v w) z +
      metric rho a u w (connection rho a u H v z) := by
  rw [metric_symm rho a u w (connection rho a u H v z)]
  rw [metric_connection rho a u H v w z hrho,
    metric_connection rho a u H v z w hrho]
  rw [metric_symm rho a u z w, metric_symm rho a u v z]
  simp only [metricDifferential, metric]
  rw [real_inner_comm z w, real_inner_comm w v]
  ring

/-- Actual differentiation of a metric pairing along a curve, allowing both
vectors to vary. The ruler and gradient derivatives are the stated local jets. -/
theorem hasDerivAt_metric_pairing (a t : ℝ) (rho : ℝ → ℝ)
    (u v w : ℝ → E) (d dv dw : E) (H : E →L[ℝ] E)
    (hr : HasDerivAt rho ⟪u t, d⟫_ℝ t)
    (hu : HasDerivAt u (H d) t) (hv : HasDerivAt v dv t)
    (hw : HasDerivAt w dw t) (hrho : rho t ≠ 0) :
    HasDerivAt (fun s => metric (rho s) a (u s) (v s) (w s))
      (metricDifferential (rho t) a (u t) H d (v t) (w t) +
        metric (rho t) a (u t) dv (w t) +
        metric (rho t) a (u t) (v t) dw) t := by
  convert ((hv.inner ℝ hw).add
    (((hasDerivAt_const t (a ^ 2)).mul (hu.inner ℝ hv)).mul (hu.inner ℝ hw))).div
      (hr.pow 2) (pow_ne_zero 2 hrho) using 1
  simp only [metricDifferential, metric, Pi.mul_apply, Pi.pow_apply, Pi.add_apply,
    Nat.cast_ofNat, Nat.reduceSub, pow_one, zero_mul, zero_add]
  field_simp
  ring

/-- The first-kind Koszul formula identifies the compact expression with the
Levi-Civita coefficients once these are actual ruler jets. -/
theorem connection_koszul (rho a : ℝ) (u : E) (H : E →L[ℝ] E)
    (hH : ∀ v w, ⟪H v, w⟫_ℝ = ⟪H w, v⟫_ℝ)
    (v w z : E) (hrho : rho ≠ 0) :
    2 * metric rho a u (connection rho a u H v w) z =
      metricDifferential rho a u H v w z +
        metricDifferential rho a u H w v z -
        metricDifferential rho a u H z v w := by
  rw [metric_connection rho a u H v w z hrho]
  simp only [metricDifferential, metric, hH w v, hH z v, hH z w,
    real_inner_comm w v]
  ring

/-- The actual kinetic derivative is the metric pairing with covariant
acceleration. Neither this derivative nor metric compatibility is assumed. -/
theorem hasDerivAt_kinetic (a t : ℝ) (rho : ℝ → ℝ) (u v : ℝ → E)
    (acc : E) (H : E →L[ℝ] E)
    (hr : HasDerivAt rho ⟪u t, v t⟫_ℝ t)
    (hu : HasDerivAt u (H (v t)) t) (hv : HasDerivAt v acc t)
    (hrho : rho t ≠ 0) :
    HasDerivAt (fun s => kinetic (rho s) a (u s) (v s))
      (metric (rho t) a (u t) (v t)
        (acc + connection (rho t) a (u t) H (v t) (v t))) t := by
  convert (hasDerivAt_metric_pairing a t rho u v v (v t) acc acc H
    hr hu hv hv hrho).div_const 2 using 1
  rw [connection_metric_compatible (rho t) a (u t) H (v t) (v t) (v t) hrho,
    metric_add_right,
    metric_symm (rho t) a (u t) (connection (rho t) a (u t) H (v t) (v t)) (v t),
    metric_symm (rho t) a (u t) acc (v t)]
  ring

/-- Continuous work-energy for the actual rank-one metric and compact
connection. Force and potential gradient are covectors represented with the
ambient inner product. The motion equation is the sole dynamics hypothesis. -/
theorem hasDerivAt_totalEnergy (a t : ℝ) (rho potential : ℝ → ℝ)
    (u v : ℝ → E) (acc gradU Q : E) (H : E →L[ℝ] E)
    (hr : HasDerivAt rho ⟪u t, v t⟫_ℝ t)
    (hu : HasDerivAt u (H (v t)) t) (hv : HasDerivAt v acc t)
    (hU : HasDerivAt potential ⟪gradU, v t⟫_ℝ t) (hrho : rho t ≠ 0)
    (hmotion : acc + connection (rho t) a (u t) H (v t) (v t) =
      inverseAction (rho t) a (u t) (Q - gradU)) :
    HasDerivAt (fun s => kinetic (rho s) a (u s) (v s) + potential s)
      ⟪v t, Q⟫_ℝ t := by
  convert (hasDerivAt_kinetic a t rho u v acc H hr hu hv hrho).add hU using 1
  rw [hmotion, metric_inverse_pairing (rho t) a (u t) (v t) (Q - gradU) hrho,
    inner_sub_right, real_inner_comm gradU (v t)]
  ring

/-- Field-level work-energy: the trace jet hypotheses are discharged by the
Fréchet chain rule for a position curve, a ruler, its gradient, and a potential.
No time dependence of the configuration fields is introduced. -/
theorem hasDerivAt_totalEnergy_of_fields (a t : ℝ) (rho potential : E → ℝ)
    (gradient : E → E) (gamma velocity : ℝ → E) (acc gradU Q : E)
    (H : E →L[ℝ] E)
    (hgamma : HasDerivAt gamma (velocity t) t)
    (hr : HasFDerivAt rho (innerSL ℝ (gradient (gamma t))) (gamma t))
    (hu : HasFDerivAt gradient H (gamma t))
    (hv : HasDerivAt velocity acc t)
    (hU : HasFDerivAt potential (innerSL ℝ gradU) (gamma t))
    (hrho : rho (gamma t) ≠ 0)
    (hmotion : acc + connection (rho (gamma t)) a (gradient (gamma t)) H
      (velocity t) (velocity t) =
      inverseAction (rho (gamma t)) a (gradient (gamma t)) (Q - gradU)) :
    HasDerivAt (fun s => kinetic (rho (gamma s)) a (gradient (gamma s)) (velocity s) +
      potential (gamma s)) ⟪velocity t, Q⟫_ℝ t := by
  exact hasDerivAt_totalEnergy a t (rho ∘ gamma) (potential ∘ gamma)
    (gradient ∘ gamma) velocity acc gradU Q H
    (hr.comp_hasDerivAt t hgamma) (hu.comp_hasDerivAt t hgamma) hv
    (hU.comp_hasDerivAt t hgamma) hrho hmotion

/-- A positive-semidefinite friction operator cannot supply positive power.
Symmetry is not needed once nonnegativity of its quadratic form is given. -/
theorem friction_power_nonpos (B : E →L[ℝ] E)
    (hB : ∀ w, 0 ≤ ⟪w, B w⟫_ℝ) (v : E) : ⟪v, -(B v)⟫_ℝ ≤ 0 := by
  simpa only [inner_neg_right] using neg_nonpos.mpr (hB v)

/-- With metric-proportional friction, the exact dissipative power is -2 beta K. -/
theorem metric_friction_power (rho a beta : ℝ) (u v : E) :
    ⟪v, -(beta • metricAction rho a u v)⟫_ℝ =
      -2 * beta * kinetic rho a u v := by
  rw [inner_neg_right, real_inner_smul_right, inner_metricAction, ← two_kinetic]
  ring

/-- Integrating the proved dynamics gives the full signed work-energy balance.
The continuous endpoint trace and integrable power are explicit; derivatives
are needed only in the open time interval. -/
theorem integral_work_eq_energy_change (a lo hi : ℝ) (rho potential : ℝ → ℝ)
    (u v acc gradU Q : ℝ → E) (H : ℝ → E →L[ℝ] E)
    (horder : lo ≤ hi)
    (hcont : ContinuousOn
      (fun s => kinetic (rho s) a (u s) (v s) + potential s) (Set.Icc lo hi))
    (hr : ∀ t ∈ Set.Ioo lo hi, HasDerivAt rho ⟪u t, v t⟫_ℝ t)
    (hu : ∀ t ∈ Set.Ioo lo hi, HasDerivAt u (H t (v t)) t)
    (hv : ∀ t ∈ Set.Ioo lo hi, HasDerivAt v (acc t) t)
    (hU : ∀ t ∈ Set.Ioo lo hi, HasDerivAt potential ⟪gradU t, v t⟫_ℝ t)
    (hrho : ∀ t ∈ Set.Ioo lo hi, rho t ≠ 0)
    (hmotion : ∀ t ∈ Set.Ioo lo hi,
      acc t + connection (rho t) a (u t) (H t) (v t) (v t) =
        inverseAction (rho t) a (u t) (Q t - gradU t))
    (hpower : IntervalIntegrable (fun t => ⟪v t, Q t⟫_ℝ) MeasureTheory.volume lo hi) :
    (∫ t in lo..hi, ⟪v t, Q t⟫_ℝ) =
      (kinetic (rho hi) a (u hi) (v hi) + potential hi) -
      (kinetic (rho lo) a (u lo) (v lo) + potential lo) := by
  exact intervalIntegral.integral_eq_sub_of_hasDerivAt_of_le horder hcont
    (fun t ht => hasDerivAt_totalEnergy a t rho potential u v (acc t) (gradU t) (Q t)
      (H t) (hr t ht) (hu t ht) (hv t ht) (hU t ht) (hrho t ht) (hmotion t ht)) hpower

/-- The compact connection is the unique vector satisfying the metric's
Koszul equation. This uses actual positive definiteness, not a coordinate
inverse chosen by convention. -/
theorem connection_eq_of_koszul (rho a : ℝ) (u : E) (H : E →L[ℝ] E)
    (hH : ∀ v w, ⟪H v, w⟫_ℝ = ⟪H w, v⟫_ℝ)
    (c v w : E) (hrho : 0 < rho)
    (hc : ∀ z, 2 * metric rho a u c z =
      metricDifferential rho a u H v w z + metricDifferential rho a u H w v z -
      metricDifferential rho a u H z v w) :
    c = connection rho a u H v w := by
  have hp : ∀ z, metric rho a u c z = metric rho a u (connection rho a u H v w) z :=
    fun z => by linarith [hc z, connection_koszul rho a u H hH v w z hrho.ne']
  have hz : metric rho a u (c - connection rho a u H v w)
      (c - connection rho a u H v w) = 0 := by
    change (metricBilinForm rho a u) (c - connection rho a u H v w)
      (c - connection rho a u H v w) = 0
    simp only [map_sub, LinearMap.sub_apply]
    change metric rho a u c c - metric rho a u (connection rho a u H v w) c -
      (metric rho a u c (connection rho a u H v w) -
        metric rho a u (connection rho a u H v w) (connection rho a u H v w)) = 0
    simp only [hp, sub_self]
  by_contra hne
  exact (ne_of_gt (metric_self_pos rho a u (c - connection rho a u H v w)
    hrho (sub_ne_zero.mpr hne))) hz

/-- A work-energy balance imposes a necessary potential-rise bound because
the final kinetic energy is nonnegative. `work` is the full signed work,
including dissipation; the potential may include both Shore and wall terms. -/
theorem potential_increase_le_initial_kinetic_add_work (rho0 rho1 a U0 U1 work : ℝ)
    (u0 u1 v0 v1 : E)
    (hbalance : work = (kinetic rho1 a u1 v1 + U1) - (kinetic rho0 a u0 v0 + U0)) :
    U1 - U0 ≤ kinetic rho0 a u0 v0 + work := by
  linarith [kinetic_nonneg rho1 a u1 v1]

/-- Free geodesic dynamics has zero actual kinetic-energy derivative. -/
theorem hasDerivAt_kinetic_geodesic (a t : ℝ) (rho : ℝ → ℝ) (u v : ℝ → E)
    (H : E →L[ℝ] E)
    (hr : HasDerivAt rho ⟪u t, v t⟫_ℝ t)
    (hu : HasDerivAt u (H (v t)) t)
    (hv : HasDerivAt v (-connection (rho t) a (u t) H (v t) (v t)) t)
    (hrho : rho t ≠ 0) :
    HasDerivAt (fun s => kinetic (rho s) a (u s) (v s)) 0 t := by
  simpa only [neg_add_cancel, metric, inner_zero_right, mul_zero, add_zero, zero_div] using
    hasDerivAt_kinetic a t rho u v (-connection (rho t) a (u t) H (v t) (v t)) H
      hr hu hv hrho

/-- Conservative forces have zero actual total-energy derivative. -/
theorem hasDerivAt_totalEnergy_conservative (a t : ℝ) (rho potential : ℝ → ℝ)
    (u v : ℝ → E) (acc gradU : E) (H : E →L[ℝ] E)
    (hr : HasDerivAt rho ⟪u t, v t⟫_ℝ t)
    (hu : HasDerivAt u (H (v t)) t) (hv : HasDerivAt v acc t)
    (hU : HasDerivAt potential ⟪gradU, v t⟫_ℝ t) (hrho : rho t ≠ 0)
    (hmotion : acc + connection (rho t) a (u t) H (v t) (v t) =
      inverseAction (rho t) a (u t) (-gradU)) :
    HasDerivAt (fun s => kinetic (rho s) a (u s) (v s) + potential s) 0 t := by
  simpa only [inner_zero_right] using hasDerivAt_totalEnergy a t rho potential u v acc gradU 0 H
    hr hu hv hU hrho (by simpa only [zero_sub] using hmotion)

end FractalSync.ScaleMetric
