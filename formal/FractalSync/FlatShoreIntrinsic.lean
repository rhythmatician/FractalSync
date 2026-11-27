import FractalSync.IntrinsicCurveLength
import FractalSync.FlatShoreGeometry

/-!
# The Euclidean flat Shore and its actual intrinsic length

The plane uses the L2 product norm. The explicit coordinate metric from
`FlatShoreGeometry` is evaluated in the ordinary global (r,y) chart, but every
ambient norm in the distance-to-set and AC path construction is Euclidean.
-/

namespace FractalSync.FlatShoreIntrinsic

open MeasureTheory Set Filter
open scoped Topology

abbrev Plane := WithLp 2 (ℝ × ℝ)

/-- The signed-normal coordinate chart; its source carries the Euclidean norm. -/
noncomputable def chart : Plane ≃L[ℝ] ℝ × ℝ :=
  WithLp.prodContinuousLinearEquiv 2 ℝ ℝ ℝ

/-- The actual flat Shore, as a subset of the Euclidean plane. -/
def shore : Set Plane := {x | x.1 = 0}

/-- Length using the explicit coordinate bilinear metric and actual velocity. -/
noncomputable def coordinateLength (epsilon a : ℝ) (gamma : ℝ → Plane) : ℝ :=
  ∫ t in (0 : ℝ)..1,
    Real.sqrt (FlatShoreGeometry.metric epsilon a (gamma t).1
      (chart (deriv gamma t)) (chart (deriv gamma t)))

/-- The candidate length set uses ordinary Euclidean absolute continuity. -/
def coordinatePathLengths (epsilon a : ℝ) (x y : Plane) : Set ℝ :=
  {L | ∃ gamma : ℝ → Plane, AbsolutelyContinuousOnInterval gamma 0 1 ∧
    gamma 0 = x ∧ gamma 1 = y ∧ coordinateLength epsilon a gamma = L}

noncomputable def coordinateDistance (epsilon a : ℝ) (x y : Plane) : ℝ :=
  sInf (coordinatePathLengths epsilon a x y)

/-- The same explicit coordinate integral written directly on coordinate curves. -/
noncomputable def chartLength (epsilon a : ℝ) (q : ℝ → ℝ × ℝ) : ℝ :=
  ∫ t in (0 : ℝ)..1,
    Real.sqrt (FlatShoreGeometry.metric epsilon a (q t).1 (deriv q t) (deriv q t))

/-- Actual Euclidean distance to the flat Shore is absolute signed normal coordinate. -/
theorem infDist_shore (x : Plane) : Metric.infDist x shore = |x.1| := by
  apply le_antisymm
  · simpa [dist_eq_norm, WithLp.prod_norm_eq_of_L2, Real.sqrt_sq_eq_abs] using
      (Metric.infDist_le_dist_of_mem (x := x) (show WithLp.toLp 2 (0, x.2) ∈ shore from rfl))
  · apply (Metric.le_infDist (show shore.Nonempty from ⟨0, rfl⟩)).2
    intro y hy
    simpa only [dist_eq_norm, WithLp.sub_fst, show y.1 = 0 from hy, sub_zero,
      Real.norm_eq_abs] using WithLp.norm_fst_le ℝ (x - y)

/-- The signed coordinate is differentiated through the actual global chart. -/
theorem hasDerivAt_normal (gamma : ℝ → Plane) (t : ℝ) (v : Plane)
    (hg : HasDerivAt gamma v t) :
    HasDerivAt (fun s => (gamma s).1) v.1 t := by
  exact ((ContinuousLinearMap.fst ℝ ℝ ℝ).comp chart.toContinuousLinearMap).hasFDerivAt.comp_hasDerivAt t hg

/-- Equality of squared distances avoids differentiation of absolute value at
Shore. Only the two trace derivatives, which exist a.e. on AC paths, are used. -/
theorem distance_weighted_deriv_eq (gamma : ℝ → Plane) (t w : ℝ) (v : Plane)
    (hg : HasDerivAt gamma v t)
    (hr : HasDerivAt (fun s => Metric.infDist (gamma s) shore) w t) :
    Metric.infDist (gamma t) shore * w = (gamma t).1 * v.1 := by
  have hnormal := (hasDerivAt_normal gamma t v hg).pow 2
  have hdistance : HasDerivAt ((fun s => (gamma s).1) ^ 2)
      (2 * Metric.infDist (gamma t) shore * w) t := by
    convert hr.pow 2 using 1
    · funext s
      simp only [Pi.pow_apply, infDist_shore, sq_abs]
    · simp only [Nat.cast_ofNat, Nat.reduceSub, pow_one]
  have heq : 2 * Metric.infDist (gamma t) shore * w = 2 * (gamma t).1 * v.1 := by
    simpa only [Nat.cast_ofNat, Nat.reduceSub, pow_one] using hdistance.unique hnormal
  linarith

/-- The actual flat-Shore density equals the coordinate metric speed wherever
both trace derivatives exist, including signed-coordinate zeroes. -/
theorem graphSpeed_eq_coordinateSpeed (epsilon a : ℝ) (he : 0 < epsilon)
    (gamma : ℝ → Plane) (t w : ℝ) (v : Plane)
    (hg : HasDerivAt gamma v t)
    (hr : HasDerivAt (fun s => Metric.infDist (gamma s) shore) w t) :
    DistanceGraph.graphSpeed a epsilon (Metric.infDist (gamma t) shore) ‖v‖ w =
      Real.sqrt (FlatShoreGeometry.metric epsilon a (gamma t).1 (chart v) (chart v)) := by
  have hweighted : (Metric.infDist (gamma t) shore) ^ 2 * w ^ 2 =
      (gamma t).1 ^ 2 * v.1 ^ 2 := by
    convert congrArg (fun z : ℝ => z ^ 2)
      (distance_weighted_deriv_eq gamma t w v hg hr) using 1 <;> ring
  apply congrArg Real.sqrt
  dsimp only [DistanceGraph.graphSpeedSq, FlatShoreGeometry.metric,
    FlatShoreGeometry.radialMetric, FlatShoreGeometry.tangentialMetric, chart]
  simp only [WithLp.prod_norm_sq_eq_of_L2, Real.norm_eq_abs, sq_abs, infDist_shore,
    WithLp.prodContinuousLinearEquiv_apply, WithLp.ofLp_fst, WithLp.ofLp_snd] at *
  rw [mul_assoc (a ^ 2), hweighted]
  field_simp [(FlatShore.ruler_sq_pos epsilon (gamma t).1 he).ne',
    (show (gamma t).1 ^ 2 + epsilon ^ 2 ≠ 0 by positivity)]
  ring

/-- The coordinate speed agrees a.e. with the unrestricted graph density;
absolute continuity alone supplies both derivatives. -/
theorem ae_graphSpeed_eq_coordinateSpeed (epsilon a lo hi : ℝ) (he : 0 < epsilon)
    (gamma : ℝ → Plane) (hg : AbsolutelyContinuousOnInterval gamma lo hi) :
    ∀ᵐ t, t ∈ uIcc lo hi →
      DistanceGraph.graphSpeed a epsilon (Metric.infDist (gamma t) shore)
        ‖deriv gamma t‖ (deriv (fun s => Metric.infDist (gamma s) shore) t) =
      Real.sqrt (FlatShoreGeometry.metric epsilon a (gamma t).1
        (chart (deriv gamma t)) (chart (deriv gamma t))) := by
  filter_upwards [hg.ae_differentiableAt,
    (DistanceGraph.ac_distance_trace shore gamma lo hi hg).ae_differentiableAt] with t hgamma hr ht
  exact graphSpeed_eq_coordinateSpeed epsilon a he gamma t _ _ (hgamma ht).hasDerivAt (hr ht).hasDerivAt

/-- Coordinate speed is integrable for every ordinary AC Euclidean curve. -/
theorem coordinateSpeed_intervalIntegrable (epsilon a lo hi : ℝ) (he : 0 < epsilon)
    (gamma : ℝ → Plane) (hg : AbsolutelyContinuousOnInterval gamma lo hi) :
    IntervalIntegrable (fun t => Real.sqrt (FlatShoreGeometry.metric epsilon a (gamma t).1
      (chart (deriv gamma t)) (chart (deriv gamma t)))) volume lo hi := by
  exact (DistanceGraph.ac_graphSpeed_intervalIntegrable shore gamma epsilon a lo hi he hg).congr_ae
    ((ae_restrict_iff' measurableSet_uIoc).2
      ((ae_graphSpeed_eq_coordinateSpeed epsilon a lo hi he gamma hg).mono
        (fun _ h ht => h (uIoc_subset_uIcc ht))))

/-- The actual integral of the explicit flat coordinate metric equals the
original graph length, including AC curves crossing or resting on Shore. -/
theorem coordinateLength_eq_acGraphLength (epsilon a : ℝ) (he : 0 < epsilon)
    (gamma : ℝ → Plane) (hg : AbsolutelyContinuousOnInterval gamma 0 1) :
    coordinateLength epsilon a gamma = DistanceGraph.acGraphLength shore epsilon a gamma := by
  exact intervalIntegral.integral_congr_ae
    ((ae_graphSpeed_eq_coordinateSpeed epsilon a 0 1 he gamma hg).mono
      (fun _ h ht => (h (uIoc_subset_uIcc ht)).symm))

/-- The two path-infimum constructions have exactly the same admissible lengths. -/
theorem coordinatePathLengths_eq (epsilon a : ℝ) (he : 0 < epsilon) (x y : Plane) :
    coordinatePathLengths epsilon a x y = DistanceGraph.pairwisePathLengths shore epsilon a x y := by
  ext L
  constructor <;> rintro ⟨gamma, hg, h0, h1, hL⟩
  · exact ⟨gamma, hg, h0, h1, (coordinateLength_eq_acGraphLength epsilon a he gamma hg).symm.trans hL⟩
  · exact ⟨gamma, hg, h0, h1, (coordinateLength_eq_acGraphLength epsilon a he gamma hg).trans hL⟩

/-- Equality of the actual path infima follows from equality of candidate lengths. -/
theorem coordinateDistance_eq_intrinsicDistance (epsilon a : ℝ) (he : 0 < epsilon)
    (x y : Plane) :
    coordinateDistance epsilon a x y = DistanceGraph.intrinsicDistance shore epsilon a x y := by
  exact congrArg sInf (coordinatePathLengths_eq epsilon a he x y)

/-- The coordinate integral equals actual partition-supremum metric length. -/
theorem eVariationOn_eq_coordinateLength (epsilon a : ℝ) (he : 0 < epsilon)
    (gamma : ℝ → Plane) (hg : AbsolutelyContinuousOnInterval gamma 0 1) :
    @eVariationOn ℝ inferInstance Plane
      (DistanceGraph.intrinsicMetricSpace shore epsilon a he).toPseudoMetricSpace.toPseudoEMetricSpace
        gamma (Icc (0 : ℝ) 1) = ENNReal.ofReal (coordinateLength epsilon a gamma) := by
  simpa only [coordinateLength_eq_acGraphLength epsilon a he gamma hg] using
    DistanceGraph.eVariationOn_intrinsic_eq_acGraphLength shore epsilon a gamma he hg

/-- Every pair has an ordinary AC minimizer for the explicit flat coordinate
integral, with constant speed in its actual path-infimum distance. -/
theorem exists_coordinateLength_minimizer (epsilon a : ℝ) (he : 0 < epsilon)
    (x y : Plane) :
    ∃ gamma : ℝ → Plane, AbsolutelyContinuousOnInterval gamma 0 1 ∧
      gamma 0 = x ∧ gamma 1 = y ∧
      coordinateLength epsilon a gamma = coordinateDistance epsilon a x y ∧
      ∀ s ∈ uIcc (0 : ℝ) 1, ∀ t ∈ uIcc (0 : ℝ) 1,
        coordinateDistance epsilon a (gamma s) (gamma t) =
          coordinateDistance epsilon a x y * |s - t| := by
  simp_rw [coordinateDistance_eq_intrinsicDistance epsilon a he]
  obtain ⟨gamma, hg, h0, h1, hL, hspeed⟩ :=
    DistanceGraph.exists_acGraphLength_minimizer shore epsilon a x y he
  exact ⟨gamma, hg, h0, h1,
    (coordinateLength_eq_acGraphLength epsilon a he gamma hg).trans hL, hspeed⟩

/-- Equivalent norms in the global chart preserve the full AC class in both directions. -/
theorem absolutelyContinuous_chart_iff (gamma : ℝ → Plane) (lo hi : ℝ) :
    AbsolutelyContinuousOnInterval (fun t => chart (gamma t)) lo hi ↔
      AbsolutelyContinuousOnInterval gamma lo hi := by
  constructor
  · intro h
    simpa only [Function.comp_def, ContinuousLinearEquiv.symm_apply_apply] using
      chart.symm.lipschitz.comp_absolutelyContinuousOnInterval h
  · exact fun h => chart.lipschitz.comp_absolutelyContinuousOnInterval h

/-- Actual chart derivatives give the familiar coordinate-curve integral. -/
theorem coordinateLength_eq_chartLength (epsilon a : ℝ) (gamma : ℝ → Plane)
    (hg : AbsolutelyContinuousOnInterval gamma 0 1) :
    coordinateLength epsilon a gamma = chartLength epsilon a (fun t => chart (gamma t)) := by
  apply intervalIntegral.integral_congr_ae
  filter_upwards [hg.ae_differentiableAt] with t ht hmem
  have hd : deriv (fun s => chart (gamma s)) t = chart (deriv gamma t) :=
    (chart.hasFDerivAt.comp_hasDerivAt t (ht (uIoc_subset_uIcc hmem)).hasDerivAt).deriv
  rw [hd]
  rfl

/-- Passing to ordinary coordinate curves preserves the candidate length set.
The chart is used only as a continuous linear equivalence, not an isometry. -/
theorem coordinatePathLengths_eq_chartPathLengths (epsilon a : ℝ) (x y : Plane) :
    coordinatePathLengths epsilon a x y =
      {L | ∃ q : ℝ → ℝ × ℝ, AbsolutelyContinuousOnInterval q 0 1 ∧
        q 0 = chart x ∧ q 1 = chart y ∧ chartLength epsilon a q = L} := by
  ext L
  constructor <;> rintro ⟨q, hq, h0, h1, hL⟩
  · exact ⟨fun t => chart (q t), (absolutelyContinuous_chart_iff q 0 1).2 hq,
      congrArg chart h0, congrArg chart h1,
      (coordinateLength_eq_chartLength epsilon a q hq).symm.trans hL⟩
  · have hq' : AbsolutelyContinuousOnInterval (fun t => chart.symm (q t)) 0 1 :=
      chart.symm.lipschitz.comp_absolutelyContinuousOnInterval hq
    refine ⟨fun t => chart.symm (q t), hq', ?_, ?_, ?_⟩
    · simp only [h0, ContinuousLinearEquiv.symm_apply_apply]
    · simp only [h1, ContinuousLinearEquiv.symm_apply_apply]
    · simpa only [coordinateLength_eq_chartLength epsilon a _ hq',
        ContinuousLinearEquiv.apply_symm_apply] using hL

/-- The general Euclidean flat-Shore distance is exactly the infimum of the
explicit coordinate integral over all ordinary AC coordinate curves. -/
theorem intrinsicDistance_eq_coordinate_path_infimum (epsilon a : ℝ) (he : 0 < epsilon)
    (x y : Plane) :
    DistanceGraph.intrinsicDistance shore epsilon a x y =
      sInf {L | ∃ q : ℝ → ℝ × ℝ, AbsolutelyContinuousOnInterval q 0 1 ∧
        q 0 = chart x ∧ q 1 = chart y ∧ chartLength epsilon a q = L} := by
  exact (coordinateDistance_eq_intrinsicDistance epsilon a he x y).symm.trans
    (congrArg sInf (coordinatePathLengths_eq_chartPathLengths epsilon a x y))

end FractalSync.FlatShoreIntrinsic
