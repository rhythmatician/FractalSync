import FractalSync.FlatShoreSecondVariation
import FractalSync.FlatShoreIntrinsic
import FractalSync.ACSubpaths

/-!
# Actual flat-Shore pairwise shortcuts beyond the normalized threshold

The sine competitor is lifted to the Euclidean plane and reparameterized to
the unit interval. Its actual AC graph length is the checked amplitude
functional, so strict shortening passes to the existing pairwise infimum.
-/

namespace FractalSync.FlatShoreShortcuts

open Set FlatShoreIntrinsic FlatShoreSecondVariation DistanceGraph

/-- The sine curve in the Euclidean ambient plane, on its original interval. -/
noncomputable def euclideanMode (ell A : ℝ) (t : ℝ) : Plane :=
  chart.symm (modeCurve ell A t)

/-- The actual competitor in the existing unit-interval AC path class. -/
noncomputable def modePath (ell A : ℝ) : ℝ → Plane :=
  affineSubpath (euclideanMode ell A) 0 Real.pi

/-- The Euclidean lift has its actual derivative under the linear chart. -/
theorem hasDerivAt_euclideanMode (ell A t : ℝ) :
    HasDerivAt (euclideanMode ell A) (chart.symm (modeVelocity ell A t)) t := by
  exact chart.symm.hasFDerivAt.comp_hasDerivAt t (hasDerivAt_modeCurve ell A t)

/-- The unit-interval competitor is absolutely continuous in the Euclidean norm. -/
theorem absolutelyContinuous_modePath (ell A : ℝ) :
    AbsolutelyContinuousOnInterval (modePath ell A) 0 1 := by
  exact ac_affineSubpath (euclideanMode ell A) 0 Real.pi Real.pi_pos.le
    (chart.symm.lipschitz.comp_absolutelyContinuousOnInterval
      (absolutelyContinuousOnInterval_modeCurve ell A 0 Real.pi))

/-- Both endpoints are exactly the designated Shore points. -/
theorem modePath_endpoints (ell A : ℝ) :
    modePath ell A 0 = chart.symm (0, 0) ∧ modePath ell A 1 = chart.symm (0, ell) := by
  simp only [modePath, affineSubpath_zero, affineSubpath_one, euclideanMode,
    (modeCurve_endpoints ell A).1, (modeCurve_endpoints ell A).2, and_self]

/-- The actual unrestricted graph length of the unit-interval competitor is
the previously differentiated sine-amplitude functional. -/
theorem acGraphLength_modePath (ell a A : ℝ) (hell : 0 < ell) :
    acGraphLength shore 1 a (modePath ell A) = oneModeLength ell a A := by
  rw [modePath, acGraphLength_affineSubpath shore (euclideanMode ell A) 1 a 0 Real.pi
    Real.pi_pos.le, oneModeLength_eq_coordinateIntegral ell a A hell]
  have hAC : AbsolutelyContinuousOnInterval (euclideanMode ell A) 0 Real.pi :=
    chart.symm.lipschitz.comp_absolutelyContinuousOnInterval
      (absolutelyContinuousOnInterval_modeCurve ell A 0 Real.pi)
  apply intervalIntegral.integral_congr_ae
  filter_upwards [ae_graphSpeed_eq_coordinateSpeed 1 a 0 Real.pi (by norm_num)
    (euclideanMode ell A) hAC] with t ht hmem
  rw [ht (uIoc_subset_uIcc hmem), (hasDerivAt_euclideanMode ell A t).deriv,
    (hasDerivAt_modeCurve ell A t).deriv, ContinuousLinearEquiv.apply_symm_apply]
  rfl

/-- The actual pairwise AC infimum is bounded by every sine competitor. -/
theorem intrinsicDistance_le_oneModeLength (ell a A : ℝ) (hell : 0 < ell) :
    intrinsicDistance shore 1 a (chart.symm (0, 0)) (chart.symm (0, ell)) ≤
      oneModeLength ell a A := by
  exact (intrinsicDistance_le_length shore 1 a (chart.symm (0, 0)) (chart.symm (0, ell))
    (modePath ell A) (absolutelyContinuous_modePath ell A)
    (modePath_endpoints ell A).1 (modePath_endpoints ell A).2).trans_eq
      (acGraphLength_modePath ell a A hell)

/-- Above normalized span pi, the actual pairwise distance is strictly less
than the Shore segment's length. No minimizing curve or geodesic is assumed. -/
theorem intrinsicDistance_lt_shore_span (ell a : ℝ) (hell : Real.pi < ell) :
    intrinsicDistance shore 1 a (chart.symm (0, 0)) (chart.symm (0, ell)) < ell := by
  obtain ⟨delta, hd, hshort⟩ := exists_shortening_amplitude_interval ell a hell
  exact (intrinsicDistance_le_oneModeLength ell a (delta / 2) (Real.pi_pos.trans hell)).trans_lt
    (hshort (delta / 2) ⟨by linarith, by linarith⟩)

/-- Scaling both the ruler floor and ambient coordinates preserves the flat
metric pairing. -/
theorem metric_dilation (epsilon a r : ℝ) (v w : FlatShoreGeometry.Coord)
    (he : epsilon ≠ 0) :
    FlatShoreGeometry.metric epsilon a (epsilon * r) (epsilon • v) (epsilon • w) =
      FlatShoreGeometry.metric 1 a r v w := by
  dsimp [FlatShoreGeometry.metric, FlatShoreGeometry.radialMetric,
    FlatShoreGeometry.tangentialMetric]
  have hD : epsilon ^ 2 + (epsilon * r) ^ 2 = epsilon ^ 2 * (1 + r ^ 2) := by ring
  rw [hD]
  field_simp

/-- Actual coordinate length is invariant under the corresponding dilation. -/
theorem coordinateLength_dilation (epsilon a : ℝ) (he : epsilon ≠ 0)
    (gamma : ℝ → Plane) :
    coordinateLength epsilon a (fun t => epsilon • gamma t) = coordinateLength 1 a gamma := by
  apply intervalIntegral.integral_congr
  intro t _
  dsimp only
  rw [deriv_fun_const_smul' epsilon, map_smul]
  change Real.sqrt (FlatShoreGeometry.metric epsilon a (epsilon * (gamma t).1)
    (epsilon • chart (deriv gamma t)) (epsilon • chart (deriv gamma t))) = _
  rw [metric_dilation epsilon a (gamma t).1 _ _ he]

/-- The unrestricted AC graph length has the same exact dilation law. -/
theorem acGraphLength_dilation (epsilon a : ℝ) (he : 0 < epsilon)
    (gamma : ℝ → Plane) (hg : AbsolutelyContinuousOnInterval gamma 0 1) :
    acGraphLength shore epsilon a (fun t => epsilon • gamma t) = acGraphLength shore 1 a gamma := by
  have hscaled : AbsolutelyContinuousOnInterval (fun t => epsilon • gamma t) 0 1 :=
    (epsilon • ContinuousLinearMap.id ℝ Plane).lipschitz.comp_absolutelyContinuousOnInterval hg
  rw [← coordinateLength_eq_acGraphLength epsilon a he _ hscaled,
    coordinateLength_dilation epsilon a he.ne' gamma,
    coordinateLength_eq_acGraphLength 1 a (by norm_num) gamma hg]

/-- Dilation of the actual sine competitor gives the same length bound at
every positive physical ruler floor. -/
theorem intrinsicDistance_le_oneModeLength_scaled (epsilon ell a A : ℝ)
    (he : 0 < epsilon) (hell : 0 < ell) :
    intrinsicDistance shore epsilon a (chart.symm (0, 0)) (chart.symm (0, epsilon * ell)) ≤
      oneModeLength ell a A := by
  have hscaled : AbsolutelyContinuousOnInterval
      (fun t => epsilon • modePath ell A t) 0 1 :=
    (epsilon • ContinuousLinearMap.id ℝ Plane).lipschitz.comp_absolutelyContinuousOnInterval
      (absolutelyContinuous_modePath ell A)
  have hends : epsilon • modePath ell A 0 = chart.symm (0, 0) ∧
      epsilon • modePath ell A 1 = chart.symm (0, epsilon * ell) := by
    rw [(modePath_endpoints ell A).1, (modePath_endpoints ell A).2,
      ← chart.symm.map_smul, ← chart.symm.map_smul]
    simp
  exact (intrinsicDistance_le_length shore epsilon a _ _
    (fun t => epsilon • modePath ell A t) hscaled hends.1 hends.2).trans_eq
      ((acGraphLength_dilation epsilon a he (modePath ell A) (absolutelyContinuous_modePath ell A)).trans
        (acGraphLength_modePath ell a A hell))

/-- At every positive floor, Euclidean Shore separation greater than pi times
the floor has actual pairwise distance strictly below the Shore length L/epsilon. -/
theorem intrinsicDistance_lt_shore_length (epsilon L a : ℝ)
    (he : 0 < epsilon) (hL : Real.pi * epsilon < L) :
    intrinsicDistance shore epsilon a (chart.symm (0, 0)) (chart.symm (0, L)) < L / epsilon := by
  have hell : Real.pi < L / epsilon := (lt_div_iff₀ he).mpr hL
  obtain ⟨delta, hd, hshort⟩ := exists_shortening_amplitude_interval (L / epsilon) a hell
  have hbound := intrinsicDistance_le_oneModeLength_scaled epsilon (L / epsilon) a (delta / 2)
    he (Real.pi_pos.trans hell)
  have hcancel : epsilon * (L / epsilon) = L := by field_simp
  rw [hcancel] at hbound
  exact hbound.trans_lt (hshort (delta / 2) ⟨by linarith, by linarith⟩)

end FractalSync.FlatShoreShortcuts
