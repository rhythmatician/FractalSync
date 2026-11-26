import FractalSync.DiscreteMechanics
import FractalSync.ContinuousMechanics
import FractalSync.NormalizedMapDifferential
import Mathlib.Analysis.Calculus.Deriv.MeanValue

/-!
# Changing-configuration steps and covariant direction evolution

Exact-real contracts for the actual velocity-then-position update. The metric
and potential are evaluated at the new position in the final energy. Local
error estimates require actual derivatives on the traversed phase-space
segment; they do not certify a numerical provider or floating-point execution.
-/

open Set
open scoped InnerProductSpace NNReal

namespace FractalSync.ScaleMetric

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

-- Expose the existing product instance for nested operator-norm inference.
noncomputable local instance : NormedSpace ℝ (E × E) := Prod.normedSpace

/-- The actual energy on position-velocity phase space. Its product norm is
used only to state a quantitative derivative bound, not as a physical metric. -/
noncomputable def phaseEnergy (a : ℝ) (rho potential : E → ℝ)
    (gradient : E → E) (z : E × E) : ℝ :=
  kinetic (rho z.1) a (gradient z.1) z.2 + potential z.1

/-- The runtime's exact-real ordering: update velocity, then use that velocity
to update position. No impulse is included in this continuous-force step. -/
def positionVelocityStep (dt : ℝ) (x v acc : E) : E × E :=
  (x + dt • velocityEulerStep dt v acc, velocityEulerStep dt v acc)

theorem differentiableAt_phaseEnergy (a : ℝ) (rho potential : E → ℝ)
    (gradient : E → E) (x v : E)
    (hr : DifferentiableAt ℝ rho x) (hu : DifferentiableAt ℝ gradient x)
    (hU : DifferentiableAt ℝ potential x) (hrho : rho x ≠ 0) :
    DifferentiableAt ℝ (phaseEnergy a rho potential gradient) (x, v) := by
  unfold phaseEnergy kinetic metric
  have hnorm := (differentiableAt_snd : DifferentiableAt ℝ (Prod.snd : E × E → E) (x, v)).inner ℝ
    differentiableAt_snd
  have hcross := (hu.comp (x, v) differentiableAt_fst).inner ℝ differentiableAt_snd
  have hrp : DifferentiableAt ℝ (fun z : E × E => rho z.1) (x, v) :=
    hr.comp (x, v) differentiableAt_fst
  simpa only [div_eq_mul_inv] using
    (((hnorm.add ((hcross.const_mul (a ^ 2)).mul hcross)).mul
      ((hrp.pow 2).inv (pow_ne_zero 2 hrho))).mul_const
        (2 : ℝ)⁻¹).add (hU.comp (x, v) differentiableAt_fst)

theorem hasDerivAt_phaseEnergy_line (a : ℝ) (rho potential : E → ℝ)
    (gradient : E → E) (x v d w gradU : E) (H : E →L[ℝ] E)
    (hr : HasFDerivAt rho (innerSL ℝ (gradient x)) x)
    (hu : HasFDerivAt gradient H x)
    (hU : HasFDerivAt potential (innerSL ℝ gradU) x) (hrho : rho x ≠ 0) :
    HasDerivAt
      (fun s : ℝ => phaseEnergy a rho potential gradient (x + s • d, v + s • w))
      (metricDifferential (rho x) a (gradient x) H d v v / 2 +
        metric (rho x) a (gradient x) v w + ⟪gradU, d⟫_ℝ) 0 := by
  have hx : HasDerivAt (fun s : ℝ => x + s • d) d 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const d).const_add x
  have hv : HasDerivAt (fun s : ℝ => v + s • w) w 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const w).const_add v
  have hk := (hasDerivAt_metric_pairing a 0 (fun s => rho (x + s • d))
    (fun s => gradient (x + s • d)) (fun s => v + s • w) (fun s => v + s • w)
    d w w H (by simpa using (show HasFDerivAt rho (innerSL ℝ (gradient x))
      (x + (0 : ℝ) • d) by simpa using hr).comp_hasDerivAt 0 hx)
    (by simpa using (show HasFDerivAt gradient H (x + (0 : ℝ) • d)
      by simpa using hu).comp_hasDerivAt 0 hx) hv hv (by simpa using hrho)).div_const 2
  convert hk.add ((show HasFDerivAt potential (innerSL ℝ gradU) (x + (0 : ℝ) • d)
    by simpa using hU).comp_hasDerivAt 0 hx) using 1
  simp only [zero_smul, add_zero, innerSL_apply, metric_symm (rho x) a (gradient x) w v]
  ring

theorem fderiv_phaseEnergy_apply (a : ℝ) (rho potential : E → ℝ)
    (gradient : E → E) (x v d w gradU : E) (H : E →L[ℝ] E)
    (hr : HasFDerivAt rho (innerSL ℝ (gradient x)) x)
    (hu : HasFDerivAt gradient H x)
    (hU : HasFDerivAt potential (innerSL ℝ gradU) x) (hrho : rho x ≠ 0) :
    fderiv ℝ (phaseEnergy a rho potential gradient) (x, v) (d, w) =
      metricDifferential (rho x) a (gradient x) H d v v / 2 +
        metric (rho x) a (gradient x) v w + ⟪gradU, d⟫_ℝ := by
  have hp := differentiableAt_phaseEnergy a rho potential gradient x v
    hr.differentiableAt hu.differentiableAt hU.differentiableAt hrho
  have hline : HasDerivAt (fun s : ℝ => (x, v) + s • (d, w)) (d, w) 0 := by
    simpa using ((hasDerivAt_id (0 : ℝ)).smul_const (d, w)).const_add (x, v)
  refine ((show HasFDerivAt (phaseEnergy a rho potential gradient)
    (fderiv ℝ (phaseEnergy a rho potential gradient) (x, v)) ((x, v) + (0 : ℝ) • (d, w))
      by simpa using hp.hasFDerivAt).comp_hasDerivAt 0 hline).unique ?_
  simpa only [Function.comp_def, Prod.smul_mk, Prod.mk_add_mk] using
    hasDerivAt_phaseEnergy_line a rho potential gradient x v d w gradU H hr hu hU hrho

/-- The actual phase-energy differential in the dynamics direction is the
nonconservative power; the connection and conservative terms cancel. -/
theorem fderiv_phaseEnergy_motion (a : ℝ) (rho potential : E → ℝ)
    (gradient : E → E) (x v acc gradU Q : E) (H : E →L[ℝ] E)
    (hr : HasFDerivAt rho (innerSL ℝ (gradient x)) x)
    (hu : HasFDerivAt gradient H x)
    (hU : HasFDerivAt potential (innerSL ℝ gradU) x) (hrho : rho x ≠ 0)
    (hmotion : acc + connection (rho x) a (gradient x) H v v =
      inverseAction (rho x) a (gradient x) (Q - gradU)) :
    fderiv ℝ (phaseEnergy a rho potential gradient) (x, v) (v, acc) = ⟪v, Q⟫_ℝ := by
  rw [fderiv_phaseEnergy_apply a rho potential gradient x v v acc gradU H hr hu hU hrho,
    connection_metric_compatible (rho x) a (gradient x) H v v v hrho,
    metric_symm (rho x) a (gradient x) (connection (rho x) a (gradient x) H v v) v]
  have h := congrArg (metric (rho x) a (gradient x) v) hmotion
  rw [metric_add_right, metric_inverse_pairing (rho x) a (gradient x) v (Q - gradU) hrho,
    inner_sub_right, real_inner_comm gradU v] at h
  linarith

theorem positionVelocityStep_eq_add (dt : ℝ) (x v acc : E) :
    positionVelocityStep dt x v acc =
      (x, v) + dt • (velocityEulerStep dt v acc, acc) := by
  rfl

theorem positionVelocityStep_increment (dt : ℝ) (v acc : E) :
    dt • (velocityEulerStep dt v acc, acc) =
      dt • (v, acc) + dt ^ 2 • (acc, (0 : E)) := by
  ext <;> simp [velocityEulerStep, smul_add, smul_smul, pow_two]

/-- The actual energy rate under a position variation, holding velocity fixed. -/
noncomputable def positionEnergyRate (rho a : ℝ) (u : E) (H : E →L[ℝ] E)
    (v gradU d : E) : ℝ :=
  metricDifferential rho a u H d v v / 2 + ⟪gradU, d⟫_ℝ

theorem fderiv_phaseEnergy_step_increment (a dt : ℝ) (rho potential : E → ℝ)
    (gradient : E → E) (x v acc gradU Q : E) (H : E →L[ℝ] E)
    (hr : HasFDerivAt rho (innerSL ℝ (gradient x)) x)
    (hu : HasFDerivAt gradient H x)
    (hU : HasFDerivAt potential (innerSL ℝ gradU) x) (hrho : rho x ≠ 0)
    (hmotion : acc + connection (rho x) a (gradient x) H v v =
      inverseAction (rho x) a (gradient x) (Q - gradU)) :
    fderiv ℝ (phaseEnergy a rho potential gradient) (x, v)
      (dt • (velocityEulerStep dt v acc, acc)) =
      dt * ⟪v, Q⟫_ℝ + dt ^ 2 * positionEnergyRate (rho x) a (gradient x) H v gradU acc := by
  rw [positionVelocityStep_increment, map_add, map_smul, map_smul,
    fderiv_phaseEnergy_motion a rho potential gradient x v acc gradU Q H hr hu hU hrho hmotion,
    fderiv_phaseEnergy_apply a rho potential gradient x v acc 0 gradU H hr hu hU hrho]
  simp [positionEnergyRate, metric, smul_eq_mul]

/-- A second-order local energy defect for the complete velocity-then-position
step. The bound is on the actual phase-energy differential on the whole step
segment, not on a presumed energy residual. The phase product uses the max norm. -/
theorem positionVelocityStep_energy_defect (a dt L : ℝ) (rho potential : E → ℝ)
    (gradient : E → E) (x v acc gradU Q : E) (H : E →L[ℝ] E)
    (hr : HasFDerivAt rho (innerSL ℝ (gradient x)) x)
    (hu : HasFDerivAt gradient H x)
    (hU : HasFDerivAt potential (innerSL ℝ gradU) x) (hrho : rho x ≠ 0)
    (hmotion : acc + connection (rho x) a (gradient x) H v v =
      inverseAction (rho x) a (gradient x) (Q - gradU))
    (hD : ∀ t ∈ Icc (0 : ℝ) 1,
      DifferentiableAt ℝ (phaseEnergy a rho potential gradient)
        ((x, v) + t • (dt • (velocityEulerStep dt v acc, acc))))
    (hbound : ∀ t ∈ Icc (0 : ℝ) 1,
      ‖fderiv ℝ (phaseEnergy a rho potential gradient)
          ((x, v) + t • (dt • (velocityEulerStep dt v acc, acc))) -
        fderiv ℝ (phaseEnergy a rho potential gradient) (x, v)‖ ≤
          L * ‖t • (dt • (velocityEulerStep dt v acc, acc))‖) :
    |phaseEnergy a rho potential gradient (positionVelocityStep dt x v acc) -
      phaseEnergy a rho potential gradient (x, v) - dt * ⟪v, Q⟫_ℝ| ≤
      dt ^ 2 * (|positionEnergyRate (rho x) a (gradient x) H v gradU acc| +
        L / 2 * (max ‖velocityEulerStep dt v acc‖ ‖acc‖) ^ 2) := by
  have hR := NormalizedDistanceFields.field_quadratic_remainder
    (phaseEnergy a rho potential gradient) (x, v)
    (dt • (velocityEulerStep dt v acc, acc)) L hD hbound
  rw [← positionVelocityStep_eq_add,
    fderiv_phaseEnergy_step_increment a dt rho potential gradient x v acc gradU Q H
      hr hu hU hrho hmotion] at hR
  have ht := abs_sub_le
    (phaseEnergy a rho potential gradient (positionVelocityStep dt x v acc) -
      phaseEnergy a rho potential gradient (x, v))
    (dt * ⟪v, Q⟫_ℝ + dt ^ 2 * positionEnergyRate (rho x) a (gradient x) H v gradU acc)
    (dt * ⟪v, Q⟫_ℝ)
  simp only [add_sub_cancel_left, abs_mul, abs_sq] at ht
  simp only [norm_smul, Real.norm_eq_abs, Prod.norm_def, mul_pow, sq_abs] at hR
  linarith

/-- An actual Hessian bound on phase energy supplies the differential variation
used above. This is a local certificate on the entire proposed phase segment. -/
theorem positionVelocityStep_energy_defect_of_hessian_bound
    (a dt : ℝ) (L : ℝ≥0) (rho potential : E → ℝ) (gradient : E → E)
    (x v acc gradU Q : E) (H : E →L[ℝ] E)
    (J : E × E → (E × E) →L[ℝ] ((E × E) →L[ℝ] ℝ))
    (hr : HasFDerivAt rho (innerSL ℝ (gradient x)) x)
    (hu : HasFDerivAt gradient H x)
    (hU : HasFDerivAt potential (innerSL ℝ gradU) x) (hrho : rho x ≠ 0)
    (hmotion : acc + connection (rho x) a (gradient x) H v v =
      inverseAction (rho x) a (gradient x) (Q - gradU))
    (hD : ∀ z ∈ segment ℝ (x, v) (positionVelocityStep dt x v acc),
      DifferentiableAt ℝ (phaseEnergy a rho potential gradient) z)
    (hJ : ∀ z ∈ segment ℝ (x, v) (positionVelocityStep dt x v acc),
      HasFDerivAt (fderiv ℝ (phaseEnergy a rho potential gradient)) (J z) z)
    (hbound : ∀ z ∈ segment ℝ (x, v) (positionVelocityStep dt x v acc), ‖J z‖ ≤ L) :
    |phaseEnergy a rho potential gradient (positionVelocityStep dt x v acc) -
      phaseEnergy a rho potential gradient (x, v) - dt * ⟪v, Q⟫_ℝ| ≤
      dt ^ 2 * (|positionEnergyRate (rho x) a (gradient x) H v gradU acc| +
        (L : ℝ) / 2 * (max ‖velocityEulerStep dt v acc‖ ‖acc‖) ^ 2) := by
  have hLip := (convex_segment (x, v) (positionVelocityStep dt x v acc)).lipschitzOnWith_of_nnnorm_hasFDerivWithin_le
      (fun z hz => (hJ z hz).hasFDerivWithinAt) (fun z hz => hbound z hz)
  have hmem : ∀ t ∈ Icc (0 : ℝ) 1,
      (x, v) + t • (dt • (velocityEulerStep dt v acc, acc)) ∈
        segment ℝ (x, v) (positionVelocityStep dt x v acc) := by
    rw [positionVelocityStep_eq_add]
    exact fun t ht => (convex_segment _ _).add_smul_mem
      (left_mem_segment ℝ _ _) (right_mem_segment ℝ _ _) ht
  refine positionVelocityStep_energy_defect a dt L rho potential gradient x v acc gradU Q H
    hr hu hU hrho hmotion (fun t ht => hD _ (hmem t ht)) ?_
  exact fun t ht => by
    simpa [dist_eq_norm] using hLip.dist_le_mul
      ((x, v) + t • (dt • (velocityEulerStep dt v acc, acc))) (hmem t ht)
      (x, v) (left_mem_segment ℝ _ _)

/-- Exact separation of the frozen-velocity, metric-change, and potential-change
terms, all evaluated at the actual old and proposed configurations. -/
theorem positionVelocityStep_energy_decomposition (a dt : ℝ)
    (rho potential : E → ℝ) (gradient : E → E) (x v acc : E) :
    phaseEnergy a rho potential gradient (positionVelocityStep dt x v acc) -
      phaseEnergy a rho potential gradient (x, v) =
      dt * metric (rho x) a (gradient x) v acc +
      dt ^ 2 * kinetic (rho x) a (gradient x) acc +
      (kinetic (rho (positionVelocityStep dt x v acc).1) a
          (gradient (positionVelocityStep dt x v acc).1) (velocityEulerStep dt v acc) -
        kinetic (rho x) a (gradient x) (velocityEulerStep dt v acc)) +
      (potential (positionVelocityStep dt x v acc).1 - potential x) := by
  have h := velocityEulerStep_energy_change (rho x) a dt (gradient x) v acc
  dsimp only [phaseEnergy, positionVelocityStep]
  linarith

theorem positionEnergyRate_eq_connection (rho a : ℝ) (u : E)
    (H : E →L[ℝ] E) (v gradU d : E) (hrho : rho ≠ 0) :
    positionEnergyRate rho a u H v gradU d =
      metric rho a u v (connection rho a u H d v) + ⟪gradU, d⟫_ℝ := by
  rw [positionEnergyRate, connection_metric_compatible rho a u H d v v hrho,
    metric_symm rho a u (connection rho a u H d v) v]
  ring

theorem positionEnergyRate_expanded (rho a : ℝ) (u : E)
    (H : E →L[ℝ] E) (v gradU d : E) :
    positionEnergyRate rho a u H v gradU d =
      -(⟪u, d⟫_ℝ / rho) * metric rho a u v v +
        a ^ 2 / rho ^ 2 * ⟪H d, v⟫_ℝ * ⟪u, v⟫_ℝ + ⟪gradU, d⟫_ℝ := by
  unfold positionEnergyRate metricDifferential
  ring

/-- The extra first-derivative coefficient in the local step defect has an
explicit bound in terms of the actual ruler gradient, Hessian, and potential
gradient. The separate phase-Hessian remainder still needs its own certificate. -/
theorem abs_positionEnergyRate_le (rho a : ℝ) (u : E)
    (H : E →L[ℝ] E) (v gradU d : E) (hrho : 0 < rho) :
    |positionEnergyRate rho a u H v gradU d| ≤
      (‖u‖ / rho * metric rho a u v v +
        a ^ 2 / rho ^ 2 * ‖H‖ * ‖u‖ * ‖v‖ ^ 2 + ‖gradU‖) * ‖d‖ := by
  rw [positionEnergyRate_expanded]
  have h1 : |-(⟪u, d⟫_ℝ / rho) * metric rho a u v v| ≤
      (‖u‖ / rho * metric rho a u v v) * ‖d‖ := by
    rw [abs_mul, abs_neg, abs_div, abs_of_pos hrho,
      abs_of_nonneg (metric_self_nonneg rho a u v)]
    convert mul_le_mul_of_nonneg_right
      (div_le_div_of_nonneg_right (abs_real_inner_le_norm u d) hrho.le)
      (metric_self_nonneg rho a u v) using 1
    ring
  have hHv : |⟪H d, v⟫_ℝ| ≤ (‖H‖ * ‖d‖) * ‖v‖ :=
    (abs_real_inner_le_norm (H d) v).trans
      (mul_le_mul_of_nonneg_right (H.le_opNorm d) (norm_nonneg v))
  have h2 : |a ^ 2 / rho ^ 2 * ⟪H d, v⟫_ℝ * ⟪u, v⟫_ℝ| ≤
      (a ^ 2 / rho ^ 2 * ‖H‖ * ‖u‖ * ‖v‖ ^ 2) * ‖d‖ := by
    have hc : 0 ≤ a ^ 2 / rho ^ 2 := div_nonneg (sq_nonneg a) (sq_nonneg rho)
    rw [abs_mul, abs_mul, abs_of_nonneg hc]
    convert mul_le_mul_of_nonneg_left
      (mul_le_mul hHv (abs_real_inner_le_norm u v) (abs_nonneg _) (by positivity)) hc using 1 <;> ring
  have ht := (abs_add_le
    (-(⟪u, d⟫_ℝ / rho) * metric rho a u v v + a ^ 2 / rho ^ 2 * ⟪H d, v⟫_ℝ * ⟪u, v⟫_ℝ)
    ⟪gradU, d⟫_ℝ).trans (add_le_add_right
      (abs_add_le (-(⟪u, d⟫_ℝ / rho) * metric rho a u v v)
        (a ^ 2 / rho ^ 2 * ⟪H d, v⟫_ℝ * ⟪u, v⟫_ℝ)) _)
  convert ht.trans (add_le_add (add_le_add h1 h2) (abs_real_inner_le_norm gradU d)) using 1
  ring

/-- A displacement budget keeps the entire interpolating segment in the open
ball, including both endpoints. -/
theorem affine_segment_norm_lt (x d : E) (R t : ℝ)
    (hbudget : ‖x‖ + ‖d‖ < R) (ht : t ∈ Icc (0 : ℝ) 1) :
    ‖x + t • d‖ < R := by
  have hn := norm_add_le x (t • d)
  rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg ht.1] at hn
  nlinarith [norm_nonneg d, ht.2]

/-- Exact proposed-velocity condition for whole-segment wall-domain safety.
Taking R=2 gives the canonical secant-wall disk. -/
theorem positionVelocityStep_segment_norm_lt (dt R : ℝ) (x v acc : E)
    (hbudget : ‖x‖ + |dt| * ‖velocityEulerStep dt v acc‖ < R)
    (t : ℝ) (ht : t ∈ Icc (0 : ℝ) 1) :
    ‖x + t • (dt • velocityEulerStep dt v acc)‖ < R := by
  exact affine_segment_norm_lt x (dt • velocityEulerStep dt v acc) R t
    (by simpa only [norm_smul, Real.norm_eq_abs] using hbudget) ht

/-- An a priori old-state/acceleration budget supplies the preceding condition. -/
theorem positionVelocityStep_segment_norm_lt_of_acceleration_bound (dt R : ℝ) (x v acc : E)
    (hbudget : ‖x‖ + |dt| * ‖v‖ + dt ^ 2 * ‖acc‖ < R)
    (t : ℝ) (ht : t ∈ Icc (0 : ℝ) 1) :
    ‖x + t • (dt • velocityEulerStep dt v acc)‖ < R := by
  refine positionVelocityStep_segment_norm_lt dt R x v acc ?_ t ht
  have hn := mul_le_mul_of_nonneg_left (norm_add_le v (dt • acc)) (abs_nonneg dt)
  simp only [norm_smul, Real.norm_eq_abs, mul_add, ← mul_assoc, ← pow_two, sq_abs] at hn
  exact lt_of_le_of_lt (add_le_add_left hn ‖x‖) (by simpa only [add_assoc] using hbudget)

/-- An impulse at the supplied configuration, matching the wrapper's operation
after the continuous-force position update. -/
noncomputable def impulseAtPhase (a : ℝ) (rho : E → ℝ) (gradient : E → E)
    (z : E × E) (J : E) : E × E :=
  (z.1, z.2 + inverseAction (rho z.1) a (gradient z.1) J)

theorem impulseAtPhase_energy_change (a : ℝ) (rho potential : E → ℝ)
    (gradient : E → E) (z : E × E) (J : E) (hrho : rho z.1 ≠ 0) :
    phaseEnergy a rho potential gradient (impulseAtPhase a rho gradient z J) -
      phaseEnergy a rho potential gradient z =
      ⟪z.2, J⟫_ℝ + dualSq (rho z.1) a (gradient z.1) J / 2 := by
  have h := impulse_energy_jump (rho z.1) a (gradient z.1) z.2 J hrho
  dsimp only [phaseEnergy, impulseAtPhase]
  linarith

/-- The wrapper's full ordered energy ledger evaluates the impulse metric at
the updated position. The complete changing-metric force-step defect remains
the one bounded above, plus this exact impulse contribution. -/
theorem positionVelocityImpulse_energy_ledger (a dt : ℝ) (rho potential : E → ℝ)
    (gradient : E → E) (x v acc J : E)
    (hrho : rho (positionVelocityStep dt x v acc).1 ≠ 0) :
    phaseEnergy a rho potential gradient
        (impulseAtPhase a rho gradient (positionVelocityStep dt x v acc) J) -
      phaseEnergy a rho potential gradient (x, v) =
      (phaseEnergy a rho potential gradient (positionVelocityStep dt x v acc) -
        phaseEnergy a rho potential gradient (x, v)) +
      ⟪velocityEulerStep dt v acc, J⟫_ℝ +
      dualSq (rho (positionVelocityStep dt x v acc).1) a
        (gradient (positionVelocityStep dt x v acc).1) J / 2 := by
  have h := impulseAtPhase_energy_change a rho potential gradient
    (positionVelocityStep dt x v acc) J hrho
  dsimp only [positionVelocityStep] at h ⊢
  linarith

end FractalSync.ScaleMetric
