import FractalSync.FlatShoreMinimizers
import FractalSync.FlatShoreIsometries

/-!
# The physical flat-Shore shortcut threshold and all-AC equality

The normalized global calibration and equality theorem transfer through actual
path symmetries. The endpoint case is included, with pauses permitted in every
minimizing Shore traversal. Above the threshold an actual AC shortcut exists.
-/

namespace FractalSync.FlatShoreThreshold

open Set FlatShoreIntrinsic FlatShoreIsometries FlatShoreShortcuts DistanceGraph

@[simp] theorem tangentialTranslate_chart_symm (b r y : ℝ) :
    tangentialTranslate b (chart.symm (r, y)) = chart.symm (r, y+b) := by
  apply chart.injective
  simp [chart_tangentialTranslate]

/-- Physical Shore distance below and at the threshold, for any tangential source. -/
theorem intrinsicDistance_eq_shore_length (epsilon a b L : ℝ) (he : 0 < epsilon)
    (hL : L ∈ Icc (0 : ℝ) (Real.pi * epsilon)) :
    intrinsicDistance shore epsilon a (chart.symm (0, b)) (chart.symm (0, b+L)) = L/epsilon := by
  have hnorm : L/epsilon ∈ Icc (0 : ℝ) Real.pi :=
    ⟨div_nonneg hL.1 he.le, (div_le_iff₀ he).mpr hL.2⟩
  have hd := (intrinsicDistance_dilation epsilon a he (chart.symm (0, 0))
    (chart.symm (0, L/epsilon))).trans
      (FlatShoreMinimizers.intrinsicDistance_eq_shore_span (L/epsilon) a hnorm)
  have hends : epsilon • chart.symm (0, 0) = chart.symm (0,0) ∧
      epsilon • chart.symm (0, L/epsilon) = chart.symm (0,L) := by
    rw [← map_smul, ← map_smul]
    simp
    field_simp
  rw [hends.1, hends.2] at hd
  simpa only [tangentialTranslate_chart_symm, zero_add, add_comm L b] using
    (intrinsicDistance_tangentialTranslate epsilon a b he (chart.symm (0,0)) (chart.symm (0,L))).trans hd


/-- Full physical equality classification, including the threshold and pauses. -/
theorem minimizing_path_iff (epsilon a b L : ℝ) (he : 0 < epsilon)
    (hL : L ∈ Icc (0 : ℝ) (Real.pi * epsilon)) (gamma : ℝ → Plane)
    (hg : AbsolutelyContinuousOnInterval gamma 0 1)
    (hstart : gamma 0 = chart.symm (0,b)) (hend : gamma 1 = chart.symm (0,b+L)) :
    acGraphLength shore epsilon a gamma = L/epsilon ↔
      (∀ t ∈ uIcc (0 : ℝ) 1, (gamma t).1 = 0) ∧
        MonotoneOn (fun t => (gamma t).2) (uIcc (0 : ℝ) 1) := by
  let eta : ℝ → Plane := fun t => epsilon⁻¹ • tangentialTranslate (-b) (gamma t)
  have heta : AbsolutelyContinuousOnInterval eta 0 1 :=
    (epsilon⁻¹ • ContinuousLinearMap.id ℝ Plane).lipschitz.comp_absolutelyContinuousOnInterval
      (absolutelyContinuous_tangentialTranslate (-b) 0 1 gamma hg)
  have hstartEta : eta 0 = chart.symm (0,0) := by
    apply chart.injective
    simp [eta, hstart]
  have hendEta : eta 1 = chart.symm (0,L/epsilon) := by
    apply chart.injective
    simp [eta, hend, div_eq_mul_inv, mul_comm]
  have hs := acGraphLength_dilation epsilon a he eta heta
  have hscaled : (fun t => epsilon • eta t) = (fun t => tangentialTranslate (-b) (gamma t)) := by
    funext t
    simp [eta, smul_smul, he.ne']
  rw [hscaled, acGraphLength_tangentialTranslate epsilon a (-b) he gamma hg] at hs
  have hiff := FlatShoreMinimizers.minimizing_path_iff (L/epsilon) a
    ⟨div_nonneg hL.1 he.le, (div_le_iff₀ he).mpr hL.2⟩ eta heta hstartEta hendEta
  rw [← hs] at hiff
  simpa [eta, tangentialTranslate, chart, MonotoneOn, he.ne',
    mul_le_mul_iff_right₀ (inv_pos.mpr he)] using hiff


/-- The physical threshold is exact for strict improvement of the actual pairwise distance. -/
theorem intrinsicDistance_lt_shore_length_iff (epsilon a b L : ℝ) (he : 0 < epsilon)
    (hL : 0 ≤ L) :
    intrinsicDistance shore epsilon a (chart.symm (0,b)) (chart.symm (0,b+L)) < L/epsilon ↔
      Real.pi * epsilon < L := by
  constructor
  · intro hshort
    by_contra hnot
    have heq := intrinsicDistance_eq_shore_length epsilon a b L he ⟨hL, le_of_not_gt hnot⟩
    exact (lt_irrefl (L/epsilon)) (heq ▸ hshort)
  · intro hlong
    have hshort := FlatShoreShortcuts.intrinsicDistance_lt_shore_length epsilon L a he hlong
    simpa only [tangentialTranslate_chart_symm, zero_add, add_comm L b] using
      (intrinsicDistance_tangentialTranslate epsilon a b he (chart.symm (0,0)) (chart.symm (0,L))).trans_lt hshort


/-- A strictly shorter admissible AC path exists exactly beyond the physical threshold.
The forward implication and the infimum-to-path passage do not assume attainment. -/
theorem exists_shorter_path_iff (epsilon a b L : ℝ) (he : 0 < epsilon) (hL : 0 ≤ L) :
    (∃ gamma : ℝ → Plane, AbsolutelyContinuousOnInterval gamma 0 1 ∧
      gamma 0 = chart.symm (0,b) ∧ gamma 1 = chart.symm (0,b+L) ∧
      acGraphLength shore epsilon a gamma < L/epsilon) ↔ Real.pi * epsilon < L := by
  constructor
  · rintro ⟨gamma, hg, hstart, hend, hshort⟩
    exact (intrinsicDistance_lt_shore_length_iff epsilon a b L he hL).mp
      ((intrinsicDistance_le_length shore epsilon a _ _ gamma hg hstart hend).trans_lt hshort)
  · intro hlong
    obtain ⟨ell, ⟨gamma, hg, hstart, hend, hlength⟩, hshort⟩ :=
      exists_lt_of_csInf_lt (pairwisePathLengths_nonempty shore epsilon a
        (chart.symm (0,b)) (chart.symm (0,b+L)))
        ((intrinsicDistance_lt_shore_length_iff epsilon a b L he hL).mpr hlong)
    exact ⟨gamma, hg, hstart, hend, hlength.trans_lt hshort⟩

end FractalSync.FlatShoreThreshold
