import FractalSync.PotentialForces

/-!
# Sustained work limits for Shore arrival

The continuous covector power is bounded using the primal and dual metric
norms. The crest budget takes the signed ledger as input, including its
wall difference, dissipation, and any explicitly accounted event work.
-/

open MeasureTheory Set Filter
open scoped Interval InnerProductSpace

namespace FractalSync.ShoreCrossingWorkBound

open ScaleMetric PotentialForces

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Speed in the authoritative graph metric, expressed through its kinetic energy. -/
noncomputable def intrinsicSpeed (rho a : ℝ) (gradient velocity : E) : ℝ :=
  Real.sqrt (2 * kinetic rho a gradient velocity)

/-- A sustained generalized-force cap bounds signed mechanical power. -/
theorem power_le_force_mul_speed (rho a F : ℝ) (gradient velocity force : E)
    (hrho : rho ≠ 0) (cap : dualNorm rho a gradient force ≤ F) :
    ⟪velocity, force⟫_ℝ ≤ F * intrinsicSpeed rho a gradient velocity := by
  calc
    ⟪velocity, force⟫_ℝ ≤ |⟪velocity, force⟫_ℝ| := le_abs_self _
    _ ≤ intrinsicSpeed rho a gradient velocity * dualNorm rho a gradient force :=
      impulse_pairing_bound rho a gradient velocity force hrho
    _ ≤ intrinsicSpeed rho a gradient velocity * F :=
      mul_le_mul_of_nonneg_left cap (Real.sqrt_nonneg _)
    _ = F * intrinsicSpeed rho a gradient velocity := mul_comm _ _

/-- The signed sustained work is at most the force cap times the actual integrated speed. -/
theorem sustained_work_le_force_mul_length (a F lo hi : ℝ) (rho : ℝ → ℝ)
    (gradient velocity force : ℝ → E) (horder : lo ≤ hi)
    (workIntegrable : IntervalIntegrable (fun t => ⟪velocity t, force t⟫_ℝ) volume lo hi)
    (speedIntegrable : IntervalIntegrable
      (fun t => intrinsicSpeed (rho t) a (gradient t) (velocity t)) volume lo hi)
    (forceCap : ∀ᵐ t ∂volume.restrict (Icc lo hi),
      rho t ≠ 0 ∧ dualNorm (rho t) a (gradient t) (force t) ≤ F) :
    (∫ t in lo..hi, ⟪velocity t, force t⟫_ℝ) ≤
      F * ∫ t in lo..hi, intrinsicSpeed (rho t) a (gradient t) (velocity t) := by
  rw [← intervalIntegral.integral_const_mul]
  apply intervalIntegral.integral_mono_ae_restrict horder workIntegrable (speedIntegrable.const_mul F)
  filter_upwards [forceCap] with t ht
  exact power_le_force_mul_speed (rho t) a F (gradient t) (velocity t) (force t) ht.1 ht.2

/-- A verified speed cap yields an upper work budget on a fixed interval. -/
theorem sustained_work_le_force_mul_speed_mul_time (a F V lo hi : ℝ) (rho : ℝ → ℝ)
    (gradient velocity force : ℝ → E) (horder : lo ≤ hi) (hF : 0 ≤ F)
    (workIntegrable : IntervalIntegrable (fun t => ⟪velocity t, force t⟫_ℝ) volume lo hi)
    (speedIntegrable : IntervalIntegrable
      (fun t => intrinsicSpeed (rho t) a (gradient t) (velocity t)) volume lo hi)
    (forceCap : ∀ᵐ t ∂volume.restrict (Icc lo hi),
      rho t ≠ 0 ∧ dualNorm (rho t) a (gradient t) (force t) ≤ F)
    (speedCap : ∀ᵐ t ∂volume.restrict (Icc lo hi),
      intrinsicSpeed (rho t) a (gradient t) (velocity t) ≤ V) :
    (∫ t in lo..hi, ⟪velocity t, force t⟫_ℝ) ≤ F * V * (hi - lo) := by
  have lengthCap : (∫ t in lo..hi, intrinsicSpeed (rho t) a (gradient t) (velocity t)) ≤
      V * (hi - lo) := by
    simpa only [intervalIntegral.integral_const, smul_eq_mul, mul_comm] using
      intervalIntegral.integral_mono_ae_restrict horder speedIntegrable
        (intervalIntegrable_const (c := V)) speedCap
  rw [mul_assoc]
  exact (sustained_work_le_force_mul_length a F lo hi rho gradient velocity force horder
    workIntegrable speedIntegrable forceCap).trans (mul_le_mul_of_nonneg_left lengthCap hF)

/-- A fixed-position event can be added to the work budget with its own dual impulse cap. -/
theorem impulse_energy_le_speed_cap (rho a V Jmax : ℝ) (gradient velocity impulse : E)
    (hrho : rho ≠ 0) (hJ : dualNorm rho a gradient impulse ≤ Jmax)
    (hV : intrinsicSpeed rho a gradient velocity ≤ V) :
    kinetic rho a gradient (velocity + inverseAction rho a gradient impulse) -
      kinetic rho a gradient velocity ≤ V * Jmax + Jmax ^ 2 / 2 := by
  have hJmax : 0 ≤ Jmax := (Real.sqrt_nonneg (dualSq rho a gradient impulse)).trans hJ
  exact (impulse_energy_bound rho a Jmax gradient velocity impulse hrho hJ).trans
    (add_le_add_right (mul_le_mul_of_nonneg_right hV hJmax) _)

omit [InnerProductSpace ℝ E] in
/-- The full Shore rise plus dissipation and required arrival kinetic energy must
fit within the initial kinetic energy and an upper bound on signed supplied work. -/
theorem crest_deficit_le_work_cap (kappa mu dref epsilon initialK arrivalK targetK
    work dissipation workCap : ℝ) (D : E → ℝ) (x0 x1 : E)
    (hd : 0 < dref) (he : 0 < epsilon) (hcrest : D x1 = 0) (hK : targetK ≤ arrivalK)
    (hbalance : arrivalK + totalPotential kappa mu dref (fun x => ruler epsilon (D x)) x1 -
      (initialK + totalPotential kappa mu dref (fun x => ruler epsilon (D x)) x0) =
        work - dissipation)
    (hcap : work ≤ workCap) :
    kappa * Real.log (ruler epsilon (D x0) / epsilon) / Real.log 2 +
      (wallPotential mu x1 - wallPotential mu x0) + dissipation + targetK - initialK ≤ workCap := by
  rw [← totalPotential_crest_difference kappa mu dref epsilon D x0 x1 hd he hcrest]
  linarith

/-- Independent geometry and work conditions both contribute to the necessary duration.
The length condition may use a certified lower bound; the work condition uses a cap. -/
theorem duration_ge_geometry_and_work (F V duration lengthLower workDeficit : ℝ)
    (hF : 0 < F) (hV : 0 < V) (hT : 0 ≤ duration)
    (hLength : lengthLower ≤ V * duration) (hWork : workDeficit ≤ F * V * duration) :
    max (lengthLower / V) (max 0 workDeficit / (F * V)) ≤ duration := by
  apply max_le
  · exact (div_le_iff₀ hV).2 (by simpa only [mul_comm] using hLength)
  · apply (div_le_iff₀ (mul_pos hF hV)).2
    rw [mul_comm duration]
    exact max_le (mul_nonneg (mul_nonneg hF.le hV.le) hT) hWork

/-- When potential is at least its baseline, control power less dissipation
is bounded by the root of twice the total energy excess. -/
theorem net_power_le_force_mul_excess_root (rho a F excess dissipationRate : ℝ)
    (gradient velocity force : E) (hrho : rho ≠ 0) (hF : 0 ≤ F)
    (hD : 0 ≤ dissipationRate) (hK : kinetic rho a gradient velocity ≤ excess)
    (forceCap : dualNorm rho a gradient force ≤ F) :
    ⟪velocity, force⟫_ℝ - dissipationRate ≤ F * Real.sqrt (2 * excess) := by
  calc
    ⟪velocity, force⟫_ℝ - dissipationRate ≤ ⟪velocity, force⟫_ℝ := sub_le_self _ hD
    _ ≤ F * intrinsicSpeed rho a gradient velocity :=
      power_le_force_mul_speed rho a F gradient velocity force hrho forceCap
    _ ≤ F * Real.sqrt (2 * excess) :=
      mul_le_mul_of_nonneg_left (Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left hK (by norm_num))) hF

/-- Starting with zero energy excess, the root of twice the excess grows at most
linearly while its rate is bounded by force times that root. The physical
application uses excess above the initial potential on an impulse-free segment
where the potential never drops below its initial value. -/
theorem rest_energy_excess_le (excess rate : ℝ → ℝ) (F lo hi : ℝ) (hF : 0 ≤ F)
    (horder : lo < hi) (hstart : excess lo = 0)
    (nonnegative : ∀ t ∈ Icc lo hi, 0 ≤ excess t)
    (continuous : ContinuousOn excess (Icc lo hi))
    (derivative : ∀ t ∈ Ioo lo hi, HasDerivAt excess (rate t) t)
    (powerCap : ∀ t ∈ Ioo lo hi, rate t ≤ F * Real.sqrt (2 * excess t)) :
    excess hi ≤ (F * (hi - lo)) ^ 2 / 2 := by
  have endpointNonnegative : 0 ≤ excess hi := nonnegative hi ⟨horder.le, le_rfl⟩
  have rootEnvelope (delta : ℝ) (hd : 0 < delta) :
      Real.sqrt (2 * excess hi) ≤ F * (hi - lo) + delta := by
    let root := fun t : ℝ => Real.sqrt (2 * excess t + delta ^ 2)
    let rootRate := fun t : ℝ => 2 * rate t / (2 * root t)
    have rootContinuous : ContinuousOn root (Icc lo hi) :=
      (((continuousOn_const (c := (2 : ℝ))).mul continuous).add continuousOn_const).sqrt
    have rootPositive (t : ℝ) (ht : t ∈ Ioo lo hi) : 0 < 2 * excess t + delta ^ 2 := by
      have ht0 := nonnegative t ⟨ht.1.le, ht.2.le⟩
      positivity
    have rootDerivative (t : ℝ) (ht : t ∈ Ioo lo hi) : HasDerivAt root (rootRate t) t :=
      (((derivative t ht).const_mul 2).add_const (delta ^ 2)).sqrt (rootPositive t ht).ne'
    obtain ⟨c, hc, slope⟩ := exists_hasDerivAt_eq_slope root rootRate horder
      rootContinuous rootDerivative
    have rateCap : rootRate c ≤ F := by
      change 2 * rate c / (2 * Real.sqrt (2 * excess c + delta ^ 2)) ≤ F
      apply (div_le_iff₀ (mul_pos (by norm_num) (Real.sqrt_pos.2 (rootPositive c hc)))).2
      have sqrtComparison : Real.sqrt (2 * excess c) ≤ Real.sqrt (2 * excess c + delta ^ 2) :=
        Real.sqrt_le_sqrt (le_add_of_nonneg_right (sq_nonneg delta))
      nlinarith only [powerCap c hc, mul_le_mul_of_nonneg_left sqrtComparison hF]
    rw [slope] at rateCap
    have endpoint := (div_le_iff₀ (sub_pos.mpr horder)).1 rateCap
    change Real.sqrt (2 * excess hi + delta ^ 2) -
      Real.sqrt (2 * excess lo + delta ^ 2) ≤ F * (hi - lo) at endpoint
    rw [hstart, mul_zero, zero_add, Real.sqrt_sq hd.le] at endpoint
    exact (Real.sqrt_le_sqrt (le_add_of_nonneg_right (sq_nonneg delta))).trans (by linarith)
  have rootBound : Real.sqrt (2 * excess hi) ≤ F * (hi - lo) := by
    by_contra! violated
    let delta := (Real.sqrt (2 * excess hi) - F * (hi - lo)) / 2
    have hd : 0 < delta := by
      dsimp [delta]
      linarith
    have budget := rootEnvelope delta hd
    dsimp [delta] at budget
    linarith
  have squared := (sq_le_sq₀ (Real.sqrt_nonneg _)
    (mul_nonneg hF (sub_nonneg.mpr horder.le))).2 rootBound
  rw [Real.sq_sqrt (by positivity)] at squared
  linarith

end FractalSync.ShoreCrossingWorkBound
