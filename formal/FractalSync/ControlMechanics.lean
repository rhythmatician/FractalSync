import FractalSync.PotentialForces

/-!
# Metric-consistent local control contracts

Exact-real contracts for the drive, steering, and friction formulas listed in
the current equations and #159. These reuse the shared metric and inverse.
The board-frame dissipation and rotation results validate the stated candidate
formulas; they do not promote a candidate to runtime authority or prescribe a
control policy. No finite-step integration, normalization tolerance, or sampled
provider claim is made.
-/

open scoped InnerProductSpace
open MeasureTheory Set
open FractalSync.ScaleMetric FractalSync.PotentialForces

namespace FractalSync.ControlMechanics

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The actual velocity differential of kinetic energy is the momentum
covector Gv. No differentiation of an independently named algebraic jet is used. -/
theorem hasFDerivAt_kinetic (rho a : ℝ) (u v : E) :
    HasFDerivAt (kinetic rho a u) (innerSL ℝ (metricAction rho a u v)) v := by
  convert (((hasStrictFDerivAt_norm_sq v).hasFDerivAt.add
    ((((innerSL ℝ u).hasFDerivAt).pow 2).const_mul (a ^ 2))).const_mul
      (1 / (2 * rho ^ 2))) using 1
  all_goals ext w
  all_goals norm_num [kinetic_formula, metricAction, stretch, innerSL_apply,
    ContinuousLinearMap.smul_apply, ContinuousLinearMap.add_apply, smul_eq_mul,
    inner_add_left, real_inner_smul_left]
  all_goals ring

/-- At a fixed configuration, the full Lagrangian has the same momentum
differential because the potential does not depend on velocity. -/
theorem hasFDerivAt_lagrangian_velocity (rho a U : ℝ) (u v : E) :
    HasFDerivAt (fun w => kinetic rho a u w - U)
      (innerSL ℝ (metricAction rho a u v)) v := by
  exact (hasFDerivAt_kinetic rho a u v).sub_const U

/-- Exact normalization underlying a metric-unit control heading. -/
noncomputable def metricNormalize (rho a : ℝ) (u v : E) : E :=
  (Real.sqrt (metric rho a u v v))⁻¹ • v

theorem metricNormalize_unit (rho a : ℝ) (u v : E) (hr : 0 < rho) (hv : v ≠ 0) :
    metric rho a u (metricNormalize rho a u v) (metricNormalize rho a u v) = 1 := by
  rw [metricNormalize, metric_smul_left, metric_smul_right]
  field_simp [ne_of_gt (Real.sqrt_pos.mpr (metric_self_pos rho a u v hr hv))]
  exact (Real.sq_sqrt (metric_self_nonneg rho a u v)).symm

/-- The canonical intrinsic drive covector Q=F G d. -/
noncomputable def driveForce (rho a F : ℝ) (u d : E) : E :=
  F • metricAction rho a u d

/-- Metric conversion returns the intended coordinate acceleration F d. -/
theorem drive_acceleration (rho a F : ℝ) (u d : E) (hr : rho ≠ 0) :
    inverseAction rho a u (driveForce rho a F u d) = F • d := by
  rw [driveForce, inverseAction_smul, inverse_metricAction rho a u d hr]

/-- Drive power is the signed metric projection of velocity along the heading. -/
theorem drive_power (rho a F : ℝ) (u d v : E) :
    ⟪v, driveForce rho a F u d⟫_ℝ = F * metric rho a u v d := by
  rw [driveForce, real_inner_smul_right, inner_metricAction]

/-- Squared drive magnitude uses the dual metric, before heading normalization. -/
theorem drive_dualSq (rho a F : ℝ) (u d : E) (hr : rho ≠ 0) :
    dualSq rho a u (driveForce rho a F u d) = F ^ 2 * metric rho a u d d := by
  rw [dualSq_eq_metric rho a u (driveForce rho a F u d) hr, drive_acceleration rho a F u d hr,
    metric_smul_left, metric_smul_right]
  ring

/-- A metric-unit heading makes the drive magnitude exactly abs(F), with no
extra ruler factor. Negative F is permitted and reverses the signed drive. -/
theorem drive_dualNorm_unit (rho a F : ℝ) (u d : E) (hr : rho ≠ 0)
    (hd : metric rho a u d d = 1) :
    dualNorm rho a u (driveForce rho a F u d) = |F| := by
  simp only [dualNorm, drive_dualSq rho a F u d hr, hd, mul_one, Real.sqrt_sq_eq_abs]

/-- Metric-unit drive has a speed-times-intrinsic-force power bound. -/
theorem drive_power_bound (rho a F : ℝ) (u d v : E) (hr : rho ≠ 0)
    (hd : metric rho a u d d = 1) :
    |⟪v, driveForce rho a F u d⟫_ℝ| ≤ Real.sqrt (2 * kinetic rho a u v) * |F| := by
  simpa only [drive_dualNorm_unit rho a F u d hr hd] using
    impulse_pairing_bound rho a u v (driveForce rho a F u d) hr

/-- The canonical viscous covector -beta Gv produces coordinate damping -beta v. -/
theorem metric_drag_acceleration (rho a beta : ℝ) (u v : E) (hr : rho ≠ 0) :
    inverseAction rho a u (-(beta • metricAction rho a u v)) = (-beta) • v := by
  rw [← neg_smul, inverseAction_smul, inverse_metricAction rho a u v hr]

theorem metric_drag_power_nonpos (rho a beta : ℝ) (u v : E) (hb : 0 ≤ beta) :
    ⟪v, -(beta • metricAction rho a u v)⟫_ℝ ≤ 0 := by
  rw [metric_friction_power]
  exact mul_nonpos_of_nonpos_of_nonneg
    (mul_nonpos_of_nonpos_of_nonneg (by norm_num) hb) (kinetic_nonneg rho a u v)

/-- The board-frame candidate friction tensor, bundled as a continuous linear
map. Its covectors are metric lowerings of the specified headings. -/
noncomputable def boardFriction (rho a betaParallel betaNormal : ℝ) (u e n : E) : E →L[ℝ] E :=
  betaParallel • (innerSL ℝ (metricAction rho a u e)).smulRight (metricAction rho a u e) +
    betaNormal • (innerSL ℝ (metricAction rho a u n)).smulRight (metricAction rho a u n)

theorem boardFriction_apply (rho a betaParallel betaNormal : ℝ) (u e n v : E) :
    boardFriction rho a betaParallel betaNormal u e n v =
      (betaParallel * metric rho a u e v) • metricAction rho a u e +
      (betaNormal * metric rho a u n v) • metricAction rho a u n := by
  simp only [boardFriction, ContinuousLinearMap.add_apply, ContinuousLinearMap.smul_apply,
    ContinuousLinearMap.smulRight_apply, innerSL_apply, smul_smul]
  rw [real_inner_comm v (metricAction rho a u e), real_inner_comm v (metricAction rho a u n),
    inner_metricAction, inner_metricAction, metric_symm rho a u v e, metric_symm rho a u v n]

/-- The candidate friction dissipates exactly the two squared metric velocity
components. This identity does not require a nonzero or orthogonal frame. -/
theorem boardFriction_quadratic (rho a betaParallel betaNormal : ℝ) (u e n v : E) :
    ⟪v, boardFriction rho a betaParallel betaNormal u e n v⟫_ℝ =
      betaParallel * metric rho a u e v ^ 2 + betaNormal * metric rho a u n v ^ 2 := by
  rw [boardFriction_apply, inner_add_right, real_inner_smul_right, real_inner_smul_right,
    inner_metricAction, inner_metricAction, metric_symm rho a u v e, metric_symm rho a u v n]
  ring

/-- Nonnegative directional coefficients are sufficient for a positive
semidefinite friction tensor, including degenerate frames and zero coefficients. -/
theorem boardFriction_nonneg (rho a betaParallel betaNormal : ℝ) (u e n v : E)
    (hp : 0 ≤ betaParallel) (hn : 0 ≤ betaNormal) :
    0 ≤ ⟪v, boardFriction rho a betaParallel betaNormal u e n v⟫_ℝ := by
  rw [boardFriction_quadratic]
  exact add_nonneg (mul_nonneg hp (sq_nonneg _)) (mul_nonneg hn (sq_nonneg _))

theorem board_friction_power_nonpos (rho a betaParallel betaNormal : ℝ) (u e n v : E)
    (hp : 0 ≤ betaParallel) (hn : 0 ≤ betaNormal) :
    ⟪v, -boardFriction rho a betaParallel betaNormal u e n v⟫_ℝ ≤ 0 := by
  exact friction_power_nonpos (boardFriction rho a betaParallel betaNormal u e n)
    (fun w => boardFriction_nonneg rho a betaParallel betaNormal u e n w hp hn) v

/-- Linearity of the shared inverse is useful when summing control covectors. -/
theorem inverseAction_add (rho a : ℝ) (u J K : E) :
    inverseAction rho a u (J + K) = inverseAction rho a u J + inverseAction rho a u K := by
  simp only [inverseAction, unstretch, inner_add_right, mul_add, add_div, add_smul,
    smul_sub, smul_add]
  abel

/-- The velocity-jump formula realizes exactly the canonical covector momentum
update at a fixed configuration. -/
theorem momentum_impulse_jump (rho a : ℝ) (u v J : E) (hr : rho ≠ 0) :
    metricAction rho a u (v + inverseAction rho a u J) = metricAction rho a u v + J := by
  conv_lhs => rw [← inverse_metricAction rho a u v hr, ← inverseAction_add]
  exact metric_inverseAction rho a u (metricAction rho a u v + J) hr

/-- Metric conversion of board friction damps the two metric projections.
No Euclidean heading normalization or ambient projection is substituted. -/
theorem board_friction_acceleration (rho a betaParallel betaNormal : ℝ) (u e n v : E)
    (hr : rho ≠ 0) :
    inverseAction rho a u (-boardFriction rho a betaParallel betaNormal u e n v) =
      -((betaParallel * metric rho a u e v) • e +
        (betaNormal * metric rho a u n v) • n) := by
  rw [← neg_one_smul ℝ (boardFriction rho a betaParallel betaNormal u e n v), inverseAction_smul,
    boardFriction_apply, inverseAction_add, inverseAction_smul, inverseAction_smul,
    inverse_metricAction rho a u e hr, inverse_metricAction rho a u n hr, neg_one_smul]

/-- A positive-semidefinite friction field gives a nonnegative dissipation
integral. The work-energy ledger separately requires this integrand integrable. -/
theorem friction_integral_nonneg (lo hi : ℝ) (velocity : ℝ → E)
    (B : ℝ → E →L[ℝ] E) (horder : lo ≤ hi)
    (hB : ∀ᵐ t ∂volume.restrict (Icc lo hi), ∀ w, 0 ≤ ⟪w, B t w⟫_ℝ) :
    0 ≤ ∫ t in lo..hi, ⟪velocity t, B t (velocity t)⟫_ℝ := by
  exact intervalIntegral.integral_nonneg_of_ae_restrict horder
    (hB.mono fun t ht => ht (velocity t))

/-- The metric-native projector along a specified metric-unit tangent. -/
noncomputable def headingProjection (rho a : ℝ) (u e v : E) : E :=
  metric rho a u e v • e

theorem headingProjection_idempotent (rho a : ℝ) (u e v : E)
    (he : metric rho a u e e = 1) :
    headingProjection rho a u e (headingProjection rho a u e v) =
      headingProjection rho a u e v := by
  simp only [headingProjection, metric_smul_right, he, mul_one]

/-- The discarded velocity component is metric-orthogonal to the heading. -/
theorem headingProjection_residual_orthogonal (rho a : ℝ) (u e v : E)
    (he : metric rho a u e e = 1) :
    metric rho a u e (v - headingProjection rho a u e v) = 0 := by
  change (metricBilinForm rho a u e) (v - headingProjection rho a u e v) = 0
  rw [map_sub]
  simp only [metricBilinForm, LinearMap.mk₂_apply, headingProjection,
    metric_smul_right, he, mul_one, sub_self]

/-- A valid iso-scale tangent remains tangent after metric projection. -/
theorem headingProjection_tangent (rho a : ℝ) (u e v : E) (he : ⟪u, e⟫_ℝ = 0) :
    ⟪u, headingProjection rho a u e v⟫_ℝ = 0 := by
  simp only [headingProjection, real_inner_smul_right, he, mul_zero]

/-- The pre-normalization vector in the documented soft carve-lock candidate. -/
noncomputable def carveBias (rho a alpha : ℝ) (u e d : E) : E :=
  (1 - alpha) • d + alpha • headingProjection rho a u e d

/-- Strictly subunit carve bias cannot erase a nonzero heading. The result
holds even before requiring the tangent to be metric-unit. -/
theorem carveBias_ne_zero (rho a alpha : ℝ) (u e d : E) (hr : 0 < rho)
    (hd : d ≠ 0) (ha0 : 0 ≤ alpha) (ha1 : alpha < 1) :
    carveBias rho a alpha u e d ≠ 0 := by
  intro hz
  have hpair := congrArg (fun z => metric rho a u d z) hz
  simp only [carveBias, metric_add_right, metric_smul_right, headingProjection] at hpair
  rw [metric_symm rho a u d e, show metric rho a u d (0 : E) = 0 by simp [metric]] at hpair
  nlinarith [mul_pos (sub_pos.mpr ha1) (metric_self_pos rho a u d hr hd),
    mul_nonneg ha0 (sq_nonneg (metric rho a u e d))]

/-- The candidate carve-lock normalization is well-defined and metric-unit
under its stated strict alpha bound. No extra nonvanishing premise is needed. -/
theorem normalized_carveBias_unit (rho a alpha : ℝ) (u e d : E) (hr : 0 < rho)
    (hd : d ≠ 0) (ha0 : 0 ≤ alpha) (ha1 : alpha < 1) :
    metric rho a u (metricNormalize rho a u (carveBias rho a alpha u e d))
      (metricNormalize rho a u (carveBias rho a alpha u e d)) = 1 := by
  exact metricNormalize_unit rho a u (carveBias rho a alpha u e d) hr
    (carveBias_ne_zero rho a alpha u e d hr hd ha0 ha1)

/-- Linearity of the existing orientation-compatible quarter-turn, in plane
coordinates. These coordinate pairs are not assigned the product's default norm. -/
theorem quarter_mix (A B C c s : ℝ) (v w : ℝ × ℝ) :
    Plane.quarter A B C (c • v + s • w) =
      c • Plane.quarter A B C v + s • Plane.quarter A B C w := by
  apply Prod.ext
  all_goals simp only [Plane.quarter, Prod.fst_add, Prod.snd_add, Prod.smul_fst, Prod.smul_snd,
    smul_eq_mul]
  all_goals ring

/-- At a fixed metric configuration, differentiating the actual steering
rotation with respect to its angle gives its quarter-turned heading. -/
theorem hasDerivAt_steering_angle (A B C theta : ℝ) (d : ℝ × ℝ)
    (hdelta : 0 < Plane.delta A B C) :
    HasDerivAt (fun angle => Plane.rotation A B C angle d)
      (Plane.quarter A B C (Plane.rotation A B C theta d)) theta := by
  convert ((Real.hasDerivAt_cos theta).smul_const d).add
    ((Real.hasDerivAt_sin theta).smul_const (Plane.quarter A B C d)) using 1
  rw [Plane.rotation, quarter_mix, Plane.quarter_square A B C d hdelta]
  module

/-- A differentiable steering-angle trace has the stated intrinsic turn rate
at a fixed configuration. A changing metric needs the connection separately. -/
theorem hasDerivAt_steering (A B C rate t : ℝ) (theta : ℝ → ℝ) (d : ℝ × ℝ)
    (hdelta : 0 < Plane.delta A B C) (hangle : HasDerivAt theta rate t) :
    HasDerivAt (fun s => Plane.rotation A B C (theta s) d)
      (rate • Plane.quarter A B C (Plane.rotation A B C (theta t) d)) t := by
  exact (hasDerivAt_steering_angle A B C (theta t) d hdelta).scomp t hangle

/-- The documented finite steering update preserves a metric-unit heading. -/
theorem steering_preserves_unit (A B C theta : ℝ) (d : ℝ × ℝ)
    (hdelta : 0 < Plane.delta A B C) (hd : Plane.form A B C d d = 1) :
    Plane.form A B C (Plane.rotation A B C theta d) (Plane.rotation A B C theta d) = 1 := by
  rw [Plane.rotation_isometry A B C theta d d hdelta, hd]

/-- For an energy-preserving redirection followed by the documented release
impulse, every energy change is the ordinary impulse ledger at the redirected
velocity. The existing plane rotation theorem supplies the equal-length premise. -/
theorem redirection_impulse_energy_jump (rho a : ℝ) (u v redirected J : E)
    (hr : rho ≠ 0) (hredirect : metric rho a u redirected redirected = metric rho a u v v) :
    kinetic rho a u (redirected + inverseAction rho a u J) - kinetic rho a u v =
      ⟪redirected, J⟫_ℝ + dualSq rho a u J / 2 := by
  have hK : kinetic rho a u redirected = kinetic rho a u v :=
    congrArg (fun z => z / 2) hredirect
  simpa only [hK] using impulse_energy_jump rho a u redirected J hr

end FractalSync.ControlMechanics
