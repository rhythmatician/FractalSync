import FractalSync.PathCalibration
import FractalSync.AbsolutelyContinuousPaths
import Mathlib.MeasureTheory.Function.AbsolutelyContinuous

/-!
# Intrinsic distance to a closed Shore

The admissible class consists of all absolutely continuous curves on `[0,1]`
with one endpoint on `S`. Length uses the actual derivatives of the curve and
its distance trace. The nearest-point segment has exactly the radial-profile
length, including the zero-length case.
-/

namespace FractalSync.DistanceGraph

open MeasureTheory Set
open scoped Topology

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Regularized graph length, expressed using the actual distance-trace
derivative. Derivatives may fail on a null set. -/
noncomputable def acGraphLength (S : Set E) (epsilon a : ℝ) (gamma : ℝ → E) : ℝ :=
  ∫ t in (0 : ℝ)..1,
    graphSpeed a epsilon (Metric.infDist (gamma t) S)
      ‖deriv gamma t‖ (deriv (fun s => Metric.infDist (gamma s) S) t)

/-- The set of lengths of all absolutely continuous paths from `S` to `x`.
There is no restriction on where the interior of the curve travels. -/
def shorePathLengths (S : Set E) (epsilon a : ℝ) (x : E) : Set ℝ :=
  {L | ∃ gamma : ℝ → E, AbsolutelyContinuousOnInterval gamma 0 1 ∧
    gamma 0 ∈ S ∧ gamma 1 = x ∧ acGraphLength S epsilon a gamma = L}

/-- The opposite orientation: all absolutely continuous paths from `x` to the
Shore. Its infimum is proved equal independently, using a reversed segment. -/
def pointToShorePathLengths (S : Set E) (epsilon a : ℝ) (x : E) : Set ℝ :=
  {L | ∃ gamma : ℝ → E, AbsolutelyContinuousOnInterval gamma 0 1 ∧
    gamma 0 = x ∧ gamma 1 ∈ S ∧ acGraphLength S epsilon a gamma = L}

/-- Actual infimum over the admissible path class. The theorem below is stated
for a nonempty closed Shore, where the length set is nonempty and bounded below. -/
noncomputable def intrinsicShoreDistance (S : Set E) (epsilon a : ℝ) (x : E) : ℝ :=
  sInf (shorePathLengths S epsilon a x)

theorem acGraphLength_nonneg (S : Set E) (epsilon a : ℝ) (gamma : ℝ → E) :
    0 ≤ acGraphLength S epsilon a gamma := by
  apply intervalIntegral.integral_nonneg (by norm_num : (0 : ℝ) ≤ 1)
  intro t _
  exact Real.sqrt_nonneg _

/-- The expanded distance-trace density is exactly the original ruler-graph
length functional. This uses the actual almost-everywhere scalar chain rule,
including curves whose distance trace has cusps. -/
theorem acGraphLength_eq_ruler_integral (S : Set E) (epsilon a : ℝ)
    (gamma : ℝ → E) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) :
    acGraphLength S epsilon a gamma =
      ∫ t in (0 : ℝ)..1,
        Real.sqrt (‖deriv gamma t‖ ^ 2 + a ^ 2 *
          (deriv (fun s => ruler epsilon (Metric.infDist (gamma s) S)) t) ^ 2) /
            ruler epsilon (Metric.infDist (gamma t) S) := by
  apply intervalIntegral.integral_congr_ae
  filter_upwards [(ac_distance_trace S gamma 0 1 hgamma).ae_differentiableAt]
    with t ht hmem
  have hdr := ((hasDerivAt_ruler epsilon (Metric.infDist (gamma t) S) he).comp t
    (ht (uIoc_subset_uIcc hmem)).hasDerivAt).deriv
  simp only [Function.comp_def] at hdr
  rw [graphSpeed_eq_rulerSpeed epsilon a _ _ _ he, hdr]

theorem shorePathLengths_bddBelow (S : Set E) (epsilon a : ℝ) (x : E) :
    BddBelow (shorePathLengths S epsilon a x) := by
  refine ⟨0, ?_⟩
  rintro L ⟨gamma, _, _, _, rfl⟩
  exact acGraphLength_nonneg S epsilon a gamma

/-- An affine segment is an admissible absolutely continuous curve. -/
theorem absolutelyContinuousOnInterval_lineMap (x p : E) :
    AbsolutelyContinuousOnInterval (fun t : ℝ => AffineMap.lineMap p x t) 0 1 := by
  exact (lipschitzWith_lineMap p x).lipschitzOnWith.absolutelyContinuousOnInterval

/-- The actual derivatives in the length definition agree almost everywhere
with the calibrated nearest-segment speeds. No derivative at an endpoint is
required. -/
theorem acGraphLength_nearest_segment (S : Set E) (x p : E)
    (hp : p ∈ S) (hnearest : Metric.infDist x S = dist x p)
    (epsilon a : ℝ) (he : 0 < epsilon) :
    acGraphLength S epsilon a (fun t => AffineMap.lineMap p x t) =
      radialProfile epsilon a (Metric.infDist x S) := by
  rw [← nearest_segment_length S x p hp hnearest epsilon a he]
  apply intervalIntegral.integral_congr_ae
  filter_upwards [show ∀ᵐ t : ℝ, t ≠ 1 from by simp [ae_iff]] with t hne
  intro ht
  have ht' : t ∈ Ioo (0 : ℝ) 1 := by
    simp only [uIoc_of_le (by norm_num : (0 : ℝ) ≤ 1), mem_Ioc] at ht
    exact ⟨ht.1, lt_of_le_of_ne ht.2 hne⟩
  rw [(hasDerivAt_nearest_segment x p t).deriv,
    (hasDerivAt_infDist_nearest_segment S x p hp hnearest t ht').deriv,
    hnearest, dist_eq_norm]

/-- The reverse nearest segment has the negative distance-trace derivative at
interior points. -/
theorem hasDerivAt_infDist_reverse_nearest_segment (S : Set E) (x p : E)
    (hp : p ∈ S) (hnearest : Metric.infDist x S = dist x p)
    (t : ℝ) (ht : t ∈ Ioo (0 : ℝ) 1) :
    HasDerivAt (fun s => Metric.infDist (AffineMap.lineMap x p s) S)
      (-Metric.infDist x S) t := by
  have ht' : 1 - t ∈ Ioo (0 : ℝ) 1 := ⟨by linarith [ht.2], by linarith [ht.1]⟩
  simpa only [Function.comp_def, AffineMap.lineMap_apply_one_sub, zero_sub,
    mul_neg, mul_one] using
    (hasDerivAt_infDist_nearest_segment S x p hp hnearest (1 - t) ht').comp t
      ((hasDerivAt_const t (1 : ℝ)).sub (hasDerivAt_id t))

/-- The segment from `x` to a nearest Shore point has the same actual length.
The orientation change is checked by integration, not assumed from metric
symmetry. -/
theorem acGraphLength_reverse_nearest_segment (S : Set E) (x p : E)
    (hp : p ∈ S) (hnearest : Metric.infDist x S = dist x p)
    (epsilon a : ℝ) (he : 0 < epsilon) :
    acGraphLength S epsilon a (fun t => AffineMap.lineMap x p t) =
      radialProfile epsilon a (Metric.infDist x S) := by
  let f : ℝ → ℝ := fun t => graphSpeed a epsilon
    (Metric.infDist (AffineMap.lineMap p x t) S) (dist x p) (dist x p)
  calc
    acGraphLength S epsilon a (fun t => AffineMap.lineMap x p t) =
        ∫ t in (0 : ℝ)..1, f (1 - t) := by
      apply intervalIntegral.integral_congr_ae
      filter_upwards [show ∀ᵐ t : ℝ, t ≠ 1 from by simp [ae_iff]] with t hne
      intro ht
      have ht' : t ∈ Ioo (0 : ℝ) 1 := by
        simp only [uIoc_of_le (by norm_num : (0 : ℝ) ≤ 1), mem_Ioc] at ht
        exact ⟨ht.1, lt_of_le_of_ne ht.2 hne⟩
      rw [(hasDerivAt_nearest_segment p x t).deriv,
        (hasDerivAt_infDist_reverse_nearest_segment S x p hp hnearest t ht').deriv]
      simp only [f, AffineMap.lineMap_apply_one_sub, hnearest, dist_eq_norm,
        norm_sub_rev p x, graphSpeed, graphSpeedSq, neg_sq]
    _ = ∫ t in (0 : ℝ)..1, f t := by
      simp only [intervalIntegral.integral_comp_sub_left, sub_self, sub_zero]
    _ = radialProfile epsilon a (Metric.infDist x S) :=
      nearest_segment_length S x p hp hnearest epsilon a he

/-- A nearest segment realizes the radial value inside the actual admissible
length set. -/
theorem radialProfile_mem_shorePathLengths [ProperSpace E]
    (S : Set E) (hclosed : IsClosed S) (hne : S.Nonempty) (x : E)
    (epsilon a : ℝ) (he : 0 < epsilon) :
    radialProfile epsilon a (Metric.infDist x S) ∈ shorePathLengths S epsilon a x := by
  obtain ⟨p, hp, hnearest⟩ := hclosed.exists_infDist_eq_dist hne x
  refine ⟨(fun t => AffineMap.lineMap p x t),
    absolutelyContinuousOnInterval_lineMap x p, ?_, ?_,
    acGraphLength_nearest_segment S x p hp hnearest epsilon a he⟩
  · simpa only [AffineMap.lineMap_apply_zero] using hp
  · simp only [AffineMap.lineMap_apply_one]

theorem shorePathLengths_nonempty [ProperSpace E]
    (S : Set E) (hclosed : IsClosed S) (hne : S.Nonempty) (x : E)
    (epsilon a : ℝ) (he : 0 < epsilon) :
    (shorePathLengths S epsilon a x).Nonempty := by
  exact ⟨_, radialProfile_mem_shorePathLengths S hclosed hne x epsilon a he⟩

/-- Upper bound for the path infimum, supplied by an admissible nearest
segment. The lower bound requires the absolutely continuous trace theorem. -/
theorem intrinsicShoreDistance_le_radialProfile [ProperSpace E]
    (S : Set E) (hclosed : IsClosed S) (hne : S.Nonempty) (x : E)
    (epsilon a : ℝ) (he : 0 < epsilon) :
    intrinsicShoreDistance S epsilon a x ≤ radialProfile epsilon a (Metric.infDist x S) := by
  exact csInf_le (shorePathLengths_bddBelow S epsilon a x)
    (radialProfile_mem_shorePathLengths S hclosed hne x epsilon a he)

/-- Every admissible absolutely continuous path has at least the radial-profile
length. Almost-everywhere differentiation and density integrability follow from
absolute continuity; neither is an additional premise. -/
theorem radialProfile_le_of_mem_shorePathLengths [FiniteDimensional ℝ E]
    (S : Set E) (epsilon a : ℝ) (x : E) (he : 0 < epsilon)
    (L : ℝ) (hL : L ∈ shorePathLengths S epsilon a x) :
    radialProfile epsilon a (Metric.infDist x S) ≤ L := by
  obtain ⟨gamma, hgamma, hstart, hend, rfl⟩ := hL
  have hb := ac_distance_trace_endpoint_bound S gamma epsilon a 0 1 he
    (by norm_num) hgamma
  simpa only [hend, Metric.infDist_zero_of_mem hstart, radialProfile_zero,
    sub_zero, abs_of_nonneg (radialProfile_nonneg epsilon a (Metric.infDist x S) he
      Metric.infDist_nonneg)] using hb

/-- The radial-profile length is the actual minimum of the unrestricted
absolutely continuous Shore-to-point path class. -/
theorem radialProfile_isLeast_shorePathLengths [FiniteDimensional ℝ E]
    (S : Set E) (hclosed : IsClosed S) (hne : S.Nonempty) (x : E)
    (epsilon a : ℝ) (he : 0 < epsilon) :
    IsLeast (shorePathLengths S epsilon a x)
      (radialProfile epsilon a (Metric.infDist x S)) := by
  exact ⟨radialProfile_mem_shorePathLengths S hclosed hne x epsilon a he,
    fun L hL => radialProfile_le_of_mem_shorePathLengths S epsilon a x he L hL⟩

/-- Exact intrinsic distance to a nonempty closed Shore in finite-dimensional
real normed space, proved for the actual path infimum. -/
theorem intrinsicShoreDistance_eq_radialProfile [FiniteDimensional ℝ E]
    (S : Set E) (hclosed : IsClosed S) (hne : S.Nonempty) (x : E)
    (epsilon a : ℝ) (he : 0 < epsilon) :
    intrinsicShoreDistance S epsilon a x =
      radialProfile epsilon a (Metric.infDist x S) := by
  exact (radialProfile_isLeast_shorePathLengths S hclosed hne x epsilon a he).csInf_eq

/-- The same actual minimum for paths oriented from the point to the Shore. -/
theorem radialProfile_isLeast_pointToShorePathLengths [FiniteDimensional ℝ E]
    (S : Set E) (hclosed : IsClosed S) (hne : S.Nonempty) (x : E)
    (epsilon a : ℝ) (he : 0 < epsilon) :
    IsLeast (pointToShorePathLengths S epsilon a x)
      (radialProfile epsilon a (Metric.infDist x S)) := by
  constructor
  · obtain ⟨p, hp, hnearest⟩ := hclosed.exists_infDist_eq_dist hne x
    refine ⟨(fun t => AffineMap.lineMap x p t),
      absolutelyContinuousOnInterval_lineMap p x, ?_, ?_,
      acGraphLength_reverse_nearest_segment S x p hp hnearest epsilon a he⟩
    · simp only [AffineMap.lineMap_apply_zero]
    · simpa only [AffineMap.lineMap_apply_one] using hp
  · rintro L ⟨gamma, hgamma, hstart, hend, rfl⟩
    have hb := ac_distance_trace_endpoint_bound S gamma epsilon a 0 1 he
      (by norm_num) hgamma
    simpa only [hstart, Metric.infDist_zero_of_mem hend, radialProfile_zero,
      zero_sub, abs_neg, abs_of_nonneg (radialProfile_nonneg epsilon a
        (Metric.infDist x S) he Metric.infDist_nonneg)] using hb

/-- The point-to-Shore infimum equals the radial profile as well. -/
theorem pointToShorePathLengths_sInf_eq [FiniteDimensional ℝ E]
    (S : Set E) (hclosed : IsClosed S) (hne : S.Nonempty) (x : E)
    (epsilon a : ℝ) (he : 0 < epsilon) :
    sInf (pointToShorePathLengths S epsilon a x) =
      radialProfile epsilon a (Metric.infDist x S) := by
  exact (radialProfile_isLeast_pointToShorePathLengths S hclosed hne x epsilon a he).csInf_eq

/-- Both orientations of the unrestricted Shore-path infimum coincide. No
general pairwise metric or symmetry theorem is presupposed. -/
theorem pointToShorePathLengths_sInf_eq_intrinsicShoreDistance [FiniteDimensional ℝ E]
    (S : Set E) (hclosed : IsClosed S) (hne : S.Nonempty) (x : E)
    (epsilon a : ℝ) (he : 0 < epsilon) :
    sInf (pointToShorePathLengths S epsilon a x) = intrinsicShoreDistance S epsilon a x := by
  rw [pointToShorePathLengths_sInf_eq S hclosed hne x epsilon a he,
    intrinsicShoreDistance_eq_radialProfile S hclosed hne x epsilon a he]

/-- The path infimum vanishes exactly on the closed Shore. -/
theorem intrinsicShoreDistance_eq_zero_iff [FiniteDimensional ℝ E]
    (S : Set E) (hclosed : IsClosed S) (hne : S.Nonempty) (x : E)
    (epsilon a : ℝ) (he : 0 < epsilon) :
    intrinsicShoreDistance S epsilon a x = 0 ↔ x ∈ S := by
  rw [intrinsicShoreDistance_eq_radialProfile S hclosed hne x epsilon a he]
  constructor
  · intro h
    apply (hclosed.mem_iff_infDist_zero hne).2
    apply (radialProfile_strictMono epsilon a he).injective
    exact h.trans (radialProfile_zero epsilon a).symm
  · intro hx
    rw [Metric.infDist_zero_of_mem hx, radialProfile_zero]

/-- The zero-floor divergence is now a statement about the actual Shore-path
infimum. This does not assert the full pairwise componentwise metric limit. -/
theorem intrinsicShoreDistance_tendsto_zero_floor [FiniteDimensional ℝ E]
    (S : Set E) (hclosed : IsClosed S) (hne : S.Nonempty) (x : E) (hx : x ∉ S)
    (a : ℝ) :
    Filter.Tendsto (fun epsilon => intrinsicShoreDistance S epsilon a x)
      (𝓝[>] (0 : ℝ)) Filter.atTop := by
  have hr : 0 < Metric.infDist x S := (hclosed.notMem_iff_infDist_pos hne).1 hx
  apply (radialProfile_tendsto_zero_floor a (Metric.infDist x S) hr).congr'
  filter_upwards [self_mem_nhdsWithin] with epsilon he
  exact (intrinsicShoreDistance_eq_radialProfile S hclosed hne x epsilon a he).symm

end FractalSync.DistanceGraph
