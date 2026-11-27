import FractalSync.RadialProfile
import FractalSync.DistanceToSet

/-!
# Integrated radial calibration for differentiable traces

This closes the integration step for traces with pointwise derivatives and
integrable densities. It does not assume that distance to an arbitrary closed
set is differentiable along every absolutely continuous curve. The unrestricted
almost-everywhere chain rule and calibration are now established separately in
`AbsolutelyContinuousPaths`; these pointwise contracts remain reusable.
-/

namespace FractalSync.DistanceGraph

/-- The actual, nonnegative graph speed used in the length integral. -/
noncomputable def graphSpeed (a epsilon r speed radialSpeed : ℝ) : ℝ :=
  Real.sqrt (graphSpeedSq a epsilon r speed radialSpeed)

theorem graphSpeedSq_nonneg (a epsilon r speed w : ℝ) :
    0 ≤ graphSpeedSq a epsilon r speed w := by
  unfold graphSpeedSq
  positivity

/-- A scalar calibration inequality for the actual square-root densities. -/
theorem radial_speed_le_graphSpeed (epsilon a r speed w : ℝ)
    (he : 0 < epsilon) (hbound : w ^ 2 ≤ speed ^ 2) :
    |radialDensity epsilon a r * w| ≤ graphSpeed a epsilon r speed w := by
  apply (Real.le_sqrt (abs_nonneg _) (graphSpeedSq_nonneg a epsilon r speed w)).2
  rw [sq_abs, mul_pow, radialDensity_sq epsilon a r w he]
  exact radial_lower_bound a epsilon r speed w he hbound

/-- Pure radial motion attains the pointwise calibration, with either sign of
radial velocity. -/
theorem graphSpeed_radial (epsilon a r v : ℝ) (he : 0 < epsilon) :
    graphSpeed a epsilon r v v = radialDensity epsilon a r * |v| := by
  have hden : r ^ 2 + epsilon ^ 2 ≠ 0 := by positivity
  have hrem := radial_remainder a epsilon r v v hden
  have heq : graphSpeedSq a epsilon r v v = radialSpeedSq a epsilon r v := by
    simpa only [sub_self, zero_div, sub_eq_zero] using hrem
  unfold graphSpeed
  rw [heq, ← radialDensity_sq epsilon a r v he, ← mul_pow,
    Real.sqrt_sq_eq_abs, abs_mul, abs_of_pos (radialDensity_pos epsilon a r he)]

/-- The calibrated nearest-segment integrand has length F(r), including r=0.
This is an equality of actual integrals, rather than an algebraic density check. -/
theorem radial_segment_length (epsilon a r : ℝ) (he : 0 < epsilon) (hr : 0 ≤ r) :
    (∫ t in (0 : ℝ)..1, graphSpeed a epsilon (t * r) r r) =
      radialProfile epsilon a r := by
  have hint : IntervalIntegrable
      (fun t => radialDensity epsilon a (t * r) * r) MeasureTheory.volume 0 1 :=
    (((continuous_radialDensity epsilon a he).comp
      (continuous_id.mul continuous_const)).mul continuous_const).intervalIntegrable 0 1
  have hftc := intervalIntegral.integral_eq_sub_of_hasDerivAt
    (a := (0 : ℝ)) (b := (1 : ℝ))
    (f := fun t => radialProfile epsilon a (t * r))
    (f' := fun t => radialDensity epsilon a (t * r) * r)
    (fun t _ => by
      simpa only [one_mul] using
        (hasDerivAt_radialProfile epsilon a (t * r) he).comp t
          ((hasDerivAt_id t).mul_const r)) hint
  simpa only [graphSpeed_radial epsilon a _ _ he, abs_of_nonneg hr,
    zero_mul, one_mul, radialProfile_zero, sub_zero] using hftc

/-- Exact graph length of an actual nearest-point segment. The Euclidean speed
and its interior radial speed are both `dist x p`; the distance field can be
nonsmooth elsewhere. -/
theorem nearest_segment_length {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E]
    (S : Set E) (x p : E) (hp : p ∈ S)
    (hnearest : Metric.infDist x S = dist x p)
    (epsilon a : ℝ) (he : 0 < epsilon) :
    (∫ t in (0 : ℝ)..1,
      graphSpeed a epsilon (Metric.infDist (AffineMap.lineMap p x t) S)
        (dist x p) (dist x p)) = radialProfile epsilon a (Metric.infDist x S) := by
  rw [← hnearest]
  calc
    (∫ t in (0 : ℝ)..1,
      graphSpeed a epsilon (Metric.infDist (AffineMap.lineMap p x t) S)
        (Metric.infDist x S) (Metric.infDist x S)) =
        ∫ t in (0 : ℝ)..1,
          graphSpeed a epsilon (t * Metric.infDist x S)
            (Metric.infDist x S) (Metric.infDist x S) := by
      apply intervalIntegral.integral_congr
      intro t ht
      dsimp only
      rw [infDist_nearest_segment S x p hp hnearest t (by simpa using ht)]
    _ = radialProfile epsilon a (Metric.infDist x S) :=
      radial_segment_length epsilon a (Metric.infDist x S) he Metric.infDist_nonneg

/-- The endpoint calibration, with the scalar chain rule and integral comparison
checked. Derivatives are needed only in the interval's interior; extending this
to arbitrary absolutely continuous distance traces is not asserted here. -/
theorem radialProfile_endpoint_bound (epsilon a lo hi : ℝ)
    (r dr speed : ℝ → ℝ) (he : 0 < epsilon) (horder : lo ≤ hi)
    (hcont : ContinuousOn r (Set.Icc lo hi))
    (hderiv : ∀ t ∈ Set.Ioo lo hi, HasDerivAt r (dr t) t)
    (hint : IntervalIntegrable
      (fun t => radialDensity epsilon a (r t) * dr t) MeasureTheory.volume lo hi)
    (hgraph : IntervalIntegrable
      (fun t => graphSpeed a epsilon (r t) (speed t) (dr t))
      MeasureTheory.volume lo hi)
    (hbound : ∀ t ∈ Set.Ioo lo hi, (dr t) ^ 2 ≤ (speed t) ^ 2) :
    |radialProfile epsilon a (r hi) - radialProfile epsilon a (r lo)| ≤
      ∫ t in lo..hi, graphSpeed a epsilon (r t) (speed t) (dr t) := by
  have hFcont : Continuous (radialProfile epsilon a) :=
    continuous_iff_continuousAt.mpr
      (fun s => (hasDerivAt_radialProfile epsilon a s he).continuousAt)
  have hftc : (∫ t in lo..hi, radialDensity epsilon a (r t) * dr t) =
      radialProfile epsilon a (r hi) - radialProfile epsilon a (r lo) :=
    intervalIntegral.integral_eq_sub_of_hasDerivAt_of_le horder
      (hFcont.comp_continuousOn hcont)
      (fun t ht => (hasDerivAt_radialProfile epsilon a (r t) he).comp t (hderiv t ht))
      hint
  rw [← hftc]
  apply (intervalIntegral.abs_integral_le_integral_abs horder).trans
  apply intervalIntegral.integral_mono_on_of_le_Ioo horder hint.abs hgraph
  intro t ht
  exact radial_speed_le_graphSpeed epsilon a (r t) (speed t) (dr t) he (hbound t ht)

/-- Specialization to the actual distance-to-set trace of a curve. There is no
ambient differentiability assumption on the distance field. -/
theorem distance_trace_endpoint_bound {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E]
    (S : Set E) (gamma velocity : ℝ → E) (w : ℝ → ℝ)
    (epsilon a lo hi : ℝ) (he : 0 < epsilon) (horder : lo ≤ hi)
    (hcont : ContinuousOn gamma (Set.Icc lo hi))
    (hgamma : ∀ t ∈ Set.Ioo lo hi, HasDerivAt gamma (velocity t) t)
    (hradial : ∀ t ∈ Set.Ioo lo hi,
      HasDerivAt (fun s => Metric.infDist (gamma s) S) (w t) t)
    (hint : IntervalIntegrable
      (fun t => radialDensity epsilon a (Metric.infDist (gamma t) S) * w t)
      MeasureTheory.volume lo hi)
    (hgraph : IntervalIntegrable
      (fun t => graphSpeed a epsilon (Metric.infDist (gamma t) S) ‖velocity t‖ (w t))
      MeasureTheory.volume lo hi) :
    |radialProfile epsilon a (Metric.infDist (gamma hi) S) -
      radialProfile epsilon a (Metric.infDist (gamma lo) S)| ≤
      ∫ t in lo..hi,
        graphSpeed a epsilon (Metric.infDist (gamma t) S) ‖velocity t‖ (w t) := by
  apply radialProfile_endpoint_bound epsilon a lo hi
    (fun t => Metric.infDist (gamma t) S) w (fun t => ‖velocity t‖) he horder
    ((Metric.continuous_infDist_pt S).comp_continuousOn hcont) hradial hint hgraph
  intro t ht
  have hb := infDist_deriv_bound S gamma t (w t) (velocity t) (hgamma t ht) (hradial t ht)
  simpa only [sq_abs] using (sq_le_sq₀ (abs_nonneg (w t)) (norm_nonneg (velocity t))).2 hb


/-- A Shore hit calibrates both endpoint-to-Shore legs. The distance trace may
have a cusp at the hit: no derivative is required at the splitting time. -/
theorem distance_trace_crossing_bound {E : Type*}
    [NormedAddCommGroup E] [NormedSpace ℝ E]
    (S : Set E) (gamma velocity : ℝ → E) (w : ℝ → ℝ)
    (epsilon a lo c hi : ℝ) (he : 0 < epsilon) (hlc : lo ≤ c) (hchi : c ≤ hi)
    (hcont : ContinuousOn gamma (Set.Icc lo hi))
    (hgamma : ∀ t ∈ Set.Ioo lo c ∪ Set.Ioo c hi, HasDerivAt gamma (velocity t) t)
    (hradial : ∀ t ∈ Set.Ioo lo c ∪ Set.Ioo c hi,
      HasDerivAt (fun s => Metric.infDist (gamma s) S) (w t) t)
    (hint : IntervalIntegrable
      (fun t => radialDensity epsilon a (Metric.infDist (gamma t) S) * w t)
      MeasureTheory.volume lo hi)
    (hgraph : IntervalIntegrable
      (fun t => graphSpeed a epsilon (Metric.infDist (gamma t) S) ‖velocity t‖ (w t))
      MeasureTheory.volume lo hi)
    (hshore : gamma c ∈ S) :
    radialProfile epsilon a (Metric.infDist (gamma lo) S) +
      radialProfile epsilon a (Metric.infDist (gamma hi) S) ≤
      ∫ t in lo..hi,
        graphSpeed a epsilon (Metric.infDist (gamma t) S) ‖velocity t‖ (w t) := by
  have hleft_sub : Set.uIcc lo c ⊆ Set.uIcc lo hi := by
    simpa only [Set.uIcc_of_le hlc, Set.uIcc_of_le (hlc.trans hchi)] using
      (Set.Icc_subset_Icc le_rfl hchi)
  have hright_sub : Set.uIcc c hi ⊆ Set.uIcc lo hi := by
    simpa only [Set.uIcc_of_le hchi, Set.uIcc_of_le (hlc.trans hchi)] using
      (Set.Icc_subset_Icc hlc le_rfl)
  have hleft := distance_trace_endpoint_bound S gamma velocity w epsilon a lo c he hlc
    (hcont.mono (Set.Icc_subset_Icc le_rfl hchi))
    (fun t ht => hgamma t (Or.inl ht)) (fun t ht => hradial t (Or.inl ht))
    (hint.mono_set hleft_sub) (hgraph.mono_set hleft_sub)
  have hright := distance_trace_endpoint_bound S gamma velocity w epsilon a c hi he hchi
    (hcont.mono (Set.Icc_subset_Icc hlc le_rfl))
    (fun t ht => hgamma t (Or.inr ht)) (fun t ht => hradial t (Or.inr ht))
    (hint.mono_set hright_sub) (hgraph.mono_set hright_sub)
  simp only [Metric.infDist_zero_of_mem hshore, radialProfile_zero,
    zero_sub, sub_zero, abs_neg] at hleft hright
  exact (add_le_add ((le_abs_self _).trans hleft) ((le_abs_self _).trans hright)).trans_eq
    (intervalIntegral.integral_add_adjacent_intervals
      (hgraph.mono_set hleft_sub) (hgraph.mono_set hright_sub))

end FractalSync.DistanceGraph
