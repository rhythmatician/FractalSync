import FractalSync.FlatShoreShortcuts

/-!
# Actual flat-Shore path-infimum symmetries

Dilation of the positive floor and ambient coordinates, and translation along
Shore, preserve the full admissible AC path length sets. Distance equalities
therefore follow directly from infima, without a minimizing-curve assumption.
-/

namespace FractalSync.FlatShoreIsometries

open Set FlatShoreIntrinsic FlatShoreShortcuts DistanceGraph

/-- Dilation gives equal candidate-length sets in the full AC path class. -/
theorem pairwisePathLengths_dilation (epsilon a : ℝ) (he : 0 < epsilon) (x y : Plane) :
    pairwisePathLengths shore epsilon a (epsilon • x) (epsilon • y) =
      pairwisePathLengths shore 1 a x y := by
  ext L
  constructor <;> rintro ⟨gamma, hg, h0, h1, hL⟩
  · have hscaled : AbsolutelyContinuousOnInterval (fun t => epsilon⁻¹ • gamma t) 0 1 :=
      (epsilon⁻¹ • ContinuousLinearMap.id ℝ Plane).lipschitz.comp_absolutelyContinuousOnInterval hg
    have hlength := acGraphLength_dilation epsilon a he (fun t => epsilon⁻¹ • gamma t) hscaled
    simp only [smul_smul, mul_inv_cancel₀ he.ne', one_smul] at hlength
    exact ⟨fun t => epsilon⁻¹ • gamma t, hscaled,
      by simp [h0, smul_smul, he.ne'], by simp [h1, smul_smul, he.ne'], hlength.symm.trans hL⟩
  · exact ⟨fun t => epsilon • gamma t,
      (epsilon • ContinuousLinearMap.id ℝ Plane).lipschitz.comp_absolutelyContinuousOnInterval hg,
      congrArg (epsilon • ·) h0, congrArg (epsilon • ·) h1,
      (acGraphLength_dilation epsilon a he gamma hg).trans hL⟩


/-- Equality of the full pairwise AC infima under physical dilation. -/
theorem intrinsicDistance_dilation (epsilon a : ℝ) (he : 0 < epsilon) (x y : Plane) :
    intrinsicDistance shore epsilon a (epsilon • x) (epsilon • y) =
      intrinsicDistance shore 1 a x y :=
  congrArg sInf (pairwisePathLengths_dilation epsilon a he x y)


/-- Translation in the tangential chart coordinate, in the Euclidean ambient plane. -/
noncomputable def tangentialTranslate (b : ℝ) (x : Plane) : Plane :=
  x + chart.symm (0, b)

theorem chart_tangentialTranslate (b : ℝ) (x : Plane) :
    chart (tangentialTranslate b x) = ((chart x).1, (chart x).2 + b) := by
  ext <;> simp [tangentialTranslate]

@[simp] theorem tangentialTranslate_neg (b : ℝ) (x : Plane) :
    tangentialTranslate (-b) (tangentialTranslate b x) = x := by
  apply chart.injective
  simp [chart_tangentialTranslate]


theorem absolutelyContinuous_tangentialTranslate (b lo hi : ℝ) (gamma : ℝ → Plane)
    (hg : AbsolutelyContinuousOnInterval gamma lo hi) :
    AbsolutelyContinuousOnInterval (fun t => tangentialTranslate b (gamma t)) lo hi :=
  (isometry_add_right (chart.symm (0, b))).lipschitz.comp_absolutelyContinuousOnInterval hg

/-- The actual coordinate integral is unchanged by tangential translation. -/
theorem coordinateLength_tangentialTranslate (epsilon a b : ℝ) (gamma : ℝ → Plane) :
    coordinateLength epsilon a (fun t => tangentialTranslate b (gamma t)) =
      coordinateLength epsilon a gamma := by
  apply intervalIntegral.integral_congr
  intro t _
  simp [tangentialTranslate, deriv_add_const, chart]


/-- Translation preserves the unrestricted AC graph length at every positive floor. -/
theorem acGraphLength_tangentialTranslate (epsilon a b : ℝ) (he : 0 < epsilon)
    (gamma : ℝ → Plane) (hg : AbsolutelyContinuousOnInterval gamma 0 1) :
    acGraphLength shore epsilon a (fun t => tangentialTranslate b (gamma t)) =
      acGraphLength shore epsilon a gamma := by
  rw [← coordinateLength_eq_acGraphLength epsilon a he _
    (absolutelyContinuous_tangentialTranslate b 0 1 gamma hg),
    coordinateLength_tangentialTranslate,
    coordinateLength_eq_acGraphLength epsilon a he gamma hg]

/-- Tangential translation preserves every admissible AC length in both directions. -/
theorem pairwisePathLengths_tangentialTranslate (epsilon a b : ℝ) (he : 0 < epsilon)
    (x y : Plane) :
    pairwisePathLengths shore epsilon a (tangentialTranslate b x) (tangentialTranslate b y) =
      pairwisePathLengths shore epsilon a x y := by
  ext L
  constructor <;> rintro ⟨gamma, hg, h0, h1, hL⟩
  · exact ⟨fun t => tangentialTranslate (-b) (gamma t),
      absolutelyContinuous_tangentialTranslate (-b) 0 1 gamma hg,
      by simp [h0], by simp [h1],
      (acGraphLength_tangentialTranslate epsilon a (-b) he gamma hg).trans hL⟩
  · exact ⟨fun t => tangentialTranslate b (gamma t),
      absolutelyContinuous_tangentialTranslate b 0 1 gamma hg,
      congrArg (tangentialTranslate b) h0, congrArg (tangentialTranslate b) h1,
      (acGraphLength_tangentialTranslate epsilon a b he gamma hg).trans hL⟩


/-- The actual pairwise path infimum is invariant under tangential translation. -/
theorem intrinsicDistance_tangentialTranslate (epsilon a b : ℝ) (he : 0 < epsilon)
    (x y : Plane) :
    intrinsicDistance shore epsilon a (tangentialTranslate b x) (tangentialTranslate b y) =
      intrinsicDistance shore epsilon a x y :=
  congrArg sInf (pairwisePathLengths_tangentialTranslate epsilon a b he x y)

end FractalSync.FlatShoreIsometries
