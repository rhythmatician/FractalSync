import FractalSync.TubularNeighborhood
import FractalSync.ScaleMetric
import Mathlib.Analysis.Calculus.InverseFunctionTheorem.ContDiff

/-!
# Normal charts and the actual regularized graph metric

The normal map is built from an actual Shore curve and normal field. Its
derivative follows from the Frenet data, and its distance trace follows from
the geometric chord certificate. These identify the previously supplied
curved-coordinate metric with an actual induced graph metric.
-/

namespace FractalSync.EmbeddedShoreMetric

open TubularNeighborhood
open scoped Topology RealInnerProductSpace

abbrev Coord := ℝ × ℝ

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The actual normal-map differential in a tangent/normal frame. -/
noncomputable def normalDifferential (n t : E) (h : ℝ) : Coord →L[ℝ] E :=
  (ContinuousLinearMap.fst ℝ ℝ ℝ).smulRight n +
    (ContinuousLinearMap.snd ℝ ℝ ℝ).smulRight (h • t)

theorem normalDifferential_apply (n t : E) (h : ℝ) (v : Coord) :
    normalDifferential n t h v = v.1 • n + (h * v.2) • t := by
  simp [normalDifferential, smul_smul, mul_comm]

/-- The Frenet derivative is derived by differentiating C(y)+r N(y). -/
theorem hasFDerivAt_normalMap (C N : ℝ → E) (z : Coord) (t : E) (k : ℝ)
    (hC : HasDerivAt C t z.2) (hN : HasDerivAt N (-k • t) z.2) :
    HasFDerivAt (normalMap C N) (normalDifferential (N z.2) t (1 - k * z.1)) z := by
  convert (hC.hasFDerivAt.comp z (ContinuousLinearMap.snd ℝ ℝ ℝ).hasFDerivAt).add
    ((ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt.smul
      (hN.hasFDerivAt.comp z (ContinuousLinearMap.snd ℝ ℝ ℝ).hasFDerivAt)) using 1
  refine ContinuousLinearMap.ext (fun v => ?_)
  dsimp [normalDifferential]
  module

/-- The Euclidean metric of the actual normal-map differential. -/
theorem inner_normalDifferential (n t : E) (h : ℝ) (v w : Coord)
    (hn : ‖n‖ = 1) (ht : ‖t‖ = 1) (hnt : inner ℝ n t = 0) :
    inner ℝ (normalDifferential n t h v) (normalDifferential n t h w) =
      v.1 * w.1 + h ^ 2 * v.2 * w.2 := by
  simp [normalDifferential_apply, inner_add_left, inner_add_right,
    real_inner_smul_left, real_inner_smul_right, real_inner_comm n t, hnt,
    real_inner_self_eq_norm_sq, hn, ht]
  ring

/-- A uniform geometric certificate for the actual Shore and normal family. -/
structure NormalTube (S : Set E) (R : ℝ) (C N : ℝ → E) : Prop where
  radius_pos : 0 < R
  foot_mem : ∀ y, C y ∈ S
  normal_unit : ∀ y, ‖N y‖ = 1
  chord_bound : ∀ y, NormalChordBound S R (C y) (N y)

/-- Actual regularized distance, evaluated on the normal map. -/
noncomputable def normalRuler (S : Set E) (epsilon : ℝ) (C N : ℝ → E) (z : Coord) : ℝ :=
  ruler epsilon (Metric.infDist (normalMap C N z) S)

/-- The actual ruler equals the signed-coordinate ruler throughout the tube. -/
theorem normalRuler_eq (S : Set E) (epsilon R : ℝ) (C N : ℝ → E)
    (h : NormalTube S R C N) (z : Coord) (hz : |z.1| ≤ R) :
    normalRuler S epsilon C N z = ruler epsilon z.1 := by
  rw [normalRuler, normalMap, infDist_normalOffset S R z.1 (C z.2) (N z.2)
    h.radius_pos (h.foot_mem z.2) (h.normal_unit z.2) (h.chord_bound z.2) hz]
  simp [ruler]

/-- Differentiation of actual distance regularization across signed-coordinate zero. -/
theorem hasFDerivAt_normalRuler (S : Set E) (epsilon R : ℝ) (C N : ℝ → E)
    (h : NormalTube S R C N) (z : Coord) (he : 0 < epsilon) (hz : |z.1| < R) :
    HasFDerivAt (normalRuler S epsilon C N)
      ((z.1 / ruler epsilon z.1) • ContinuousLinearMap.fst ℝ ℝ ℝ) z := by
  have hcoord := (hasDerivAt_ruler epsilon z.1 he).comp_hasFDerivAt z
    (ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt
  apply hcoord.congr_of_eventuallyEq
  filter_upwards [(continuous_fst.abs.continuousAt).eventually_lt_const hz] with q hq
  exact normalRuler_eq S epsilon R C N h q hq.le

/-- Induced graph metric computed from actual chart and regularized-distance derivatives. -/
noncomputable def normalGraphMetric (S : Set E) (epsilon a : ℝ) (C N : ℝ → E)
    (z v w : Coord) : ℝ :=
  (inner ℝ (fderiv ℝ (normalMap C N) z v) (fderiv ℝ (normalMap C N) z w) +
    a ^ 2 * fderiv ℝ (normalRuler S epsilon C N) z v *
      fderiv ℝ (normalRuler S epsilon C N) z w) / normalRuler S epsilon C N z ^ 2

/-- The curved-coordinate metric is the actual induced graph metric in a certified tube. -/
theorem normalGraphMetric_eq (S : Set E) (epsilon a R : ℝ) (C N : ℝ → E)
    (h : NormalTube S R C N) (z v w : Coord) (t : E) (k : ℝ)
    (he : 0 < epsilon) (hz : |z.1| < R)
    (hC : HasDerivAt C t z.2) (hN : HasDerivAt N (-k • t) z.2)
    (ht : ‖t‖ = 1) (hnt : inner ℝ (N z.2) t = 0) :
    normalGraphMetric S epsilon a C N z v w =
      CurvedShoreGeometry.metric epsilon a k z.1 v w := by
  rw [normalGraphMetric, (hasFDerivAt_normalMap C N z t k hC hN).fderiv,
    (hasFDerivAt_normalRuler S epsilon R C N h z he hz).fderiv,
    normalRuler_eq S epsilon R C N h z hz.le,
    inner_normalDifferential (N z.2) t (1 - k * z.1) v w (h.normal_unit z.2) ht hnt]
  dsimp [CurvedShoreGeometry.metric, FlatShoreGeometry.radialMetric, FlatShoreGeometry.tangentialMetric]
  have hprod : a ^ 2 * (z.1 / ruler epsilon z.1 * v.1) *
      (z.1 / ruler epsilon z.1 * w.1) =
      a ^ 2 * z.1 ^ 2 * v.1 * w.1 / ruler epsilon z.1 ^ 2 := by ring
  rw [hprod, ruler_sq]
  field_simp
  ring

/-- An invertible frame with the actual nonfocal tangential scale. -/
noncomputable def normalFrameEquiv (B : Coord ≃L[ℝ] E) (h : ℝ) (hh : h ≠ 0) :
    Coord ≃L[ℝ] E :=
  ((ContinuousLinearEquiv.refl ℝ ℝ).prodCongr
    (ContinuousLinearEquiv.unitsEquivAut ℝ (Units.mk0 h hh))).trans B

theorem normalFrameEquiv_apply (B : Coord ≃L[ℝ] E) (h : ℝ) (hh : h ≠ 0) (v : Coord) :
    normalFrameEquiv B h hh v = B (v.1, h * v.2) := by
  simp [normalFrameEquiv, ContinuousLinearEquiv.unitsEquivAut_apply, mul_comm]

/-- Coordinates of a vector in the supplied actual linear frame. -/
theorem frame_expand (B : Coord ≃L[ℝ] E) (v : Coord) :
    B v = v.1 • B (1, 0) + v.2 • B (0, 1) := by
  rw [← B.map_smul v.1 (1, 0), ← B.map_smul v.2 (0, 1), ← B.map_add]
  congr 1
  exact Prod.ext (by simp) (by simp)

/-- The normal differential is invertible at the stated nonfocal frame scale. -/
theorem normalDifferential_eq_frame (B : Coord ≃L[ℝ] E) (h : ℝ) (hh : h ≠ 0) :
    normalDifferential (B (1, 0)) (B (0, 1)) h =
      (normalFrameEquiv B h hh : Coord →L[ℝ] E) := by
  refine ContinuousLinearMap.ext (fun v => ?_)
  change normalDifferential (B (1, 0)) (B (0, 1)) h v = normalFrameEquiv B h hh v
  rw [normalDifferential_apply, normalFrameEquiv_apply, frame_expand B (v.1, h * v.2)]

/-- Actual C¹ curve and normal data make the normal map C¹. -/
theorem contDiffAt_normalMap (C N : ℝ → E) (z : Coord)
    (hC : ContDiffAt ℝ 1 C z.2) (hN : ContDiffAt ℝ 1 N z.2) :
    ContDiffAt ℝ 1 (normalMap C N) z := by
  exact (hC.comp z contDiffAt_snd).add (contDiffAt_fst.smul (hN.comp z contDiffAt_snd))

/-- The actual normal-map derivative is the explicitly constructed linear equivalence. -/
theorem hasFDerivAt_normalMap_frame (C N : ℝ → E) (z : Coord)
    (B : Coord ≃L[ℝ] E) (k : ℝ) (hh : 1 - k * z.1 ≠ 0)
    (hn : N z.2 = B (1, 0)) (hC : HasDerivAt C (B (0, 1)) z.2)
    (hN : HasDerivAt N (-k • B (0, 1)) z.2) :
    HasFDerivAt (normalMap C N)
      (normalFrameEquiv B (1 - k * z.1) hh : Coord →L[ℝ] E) z := by
  simpa only [hn, normalDifferential_eq_frame B (1 - k * z.1) hh] using
    hasFDerivAt_normalMap C N z (B (0, 1)) k hC hN

/-- The inverse function theorem constructs an actual local normal chart and inverse derivative. -/
theorem exists_normalChart (C N : ℝ → E) (z : Coord) (B : Coord ≃L[ℝ] E)
    (k : ℝ) (hh : 1 - k * z.1 ≠ 0) (hn : N z.2 = B (1, 0))
    (hC : HasDerivAt C (B (0, 1)) z.2) (hN : HasDerivAt N (-k • B (0, 1)) z.2)
    (hC1 : ContDiffAt ℝ 1 C z.2) (hN1 : ContDiffAt ℝ 1 N z.2) :
    ∃ e : OpenPartialHomeomorph Coord E,
      (e : Coord → E) = normalMap C N ∧ z ∈ e.source ∧
      HasFDerivAt e.symm
        ((normalFrameEquiv B (1 - k * z.1) hh).symm : E →L[ℝ] Coord)
        (normalMap C N z) := by
  have hs := (contDiffAt_normalMap C N z hC1 hN1).hasStrictFDerivAt'
    (hasFDerivAt_normalMap_frame C N z B k hh hn hC hN) (by norm_num)
  refine ⟨hs.toOpenPartialHomeomorph (normalMap C N), rfl, hs.mem_toOpenPartialHomeomorph_source, ?_⟩
  exact hs.to_localInverse.hasFDerivAt

/-- The inverse chart's signed coordinate has the actual distance as its magnitude. -/
theorem infDist_normalChart (S : Set E) (R : ℝ) (C N : ℝ → E)
    (h : NormalTube S R C N) (e : OpenPartialHomeomorph Coord E)
    (he : (e : Coord → E) = normalMap C N) (x : E)
    (hx : x ∈ e.target) (hr : |(e.symm x).1| ≤ R) :
    Metric.infDist x S = |(e.symm x).1| := by
  calc
    Metric.infDist x S = Metric.infDist (normalMap C N (e.symm x)) S := by
      rw [← he, e.right_inv hx]
    _ = |(e.symm x).1| := infDist_normalOffset S R (e.symm x).1
      (C (e.symm x).2) (N (e.symm x).2) h.radius_pos (h.foot_mem _)
      (h.normal_unit _) (h.chord_bound _) hr

/-- The distance identity holds on a full ambient neighborhood of the represented point. -/
theorem eventually_infDist_normalChart (S : Set E) (R : ℝ) (C N : ℝ → E)
    (h : NormalTube S R C N) (e : OpenPartialHomeomorph Coord E)
    (he : (e : Coord → E) = normalMap C N) (z : Coord)
    (hz : z ∈ e.source) (hr : |z.1| < R) :
    ∀ᶠ x in 𝓝 (e z), Metric.infDist x S = |(e.symm x).1| := by
  have ht : e z ∈ e.target := e.map_source hz
  have hc := (e.continuousAt_symm ht).fst.abs
  filter_upwards [e.open_target.mem_nhds ht,
    hc.eventually_lt_const (by simpa only [e.left_inv hz] using hr)] with x hx hxr
  exact infDist_normalChart S R C N h e he x hx hxr.le

/-- Actual ambient ruler differentiation, obtained from the constructed inverse chart. -/
theorem hasFDerivAt_ambientRuler (S : Set E) (epsilon R : ℝ) (C N : ℝ → E)
    (h : NormalTube S R C N) (e : OpenPartialHomeomorph Coord E)
    (he : (e : Coord → E) = normalMap C N) (z : Coord)
    (hz : z ∈ e.source) (hr : |z.1| < R) (heps : 0 < epsilon)
    (D : E →L[ℝ] Coord) (hInv : HasFDerivAt e.symm D (e z)) :
    HasFDerivAt (fun x => ruler epsilon (Metric.infDist x S))
      ((z.1 / ruler epsilon z.1) • (ContinuousLinearMap.fst ℝ ℝ ℝ).comp D) (e z) := by
  have hd : HasFDerivAt (fun x => ruler epsilon (e.symm x).1)
      ((z.1 / ruler epsilon z.1) • (ContinuousLinearMap.fst ℝ ℝ ℝ).comp D) (e z) := by
    simpa only [e.left_inv hz] using
      ((hasDerivAt_ruler epsilon (e.symm (e z)).1 heps).comp_hasFDerivAt (e z)
        ((ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt.comp (e z) hInv))
  apply hd.congr_of_eventuallyEq
  filter_upwards [eventually_infDist_normalChart S R C N h e he z hz hr] with x hx
  simp only [hx, ruler, sq_abs]

/-- The ambient graph metric uses the actual derivative of regularized distance. -/
noncomputable def ambientGraphMetric (S : Set E) (epsilon a : ℝ) (x v w : E) : ℝ :=
  (inner ℝ v w + a ^ 2 * fderiv ℝ (fun p => ruler epsilon (Metric.infDist p S)) x v *
    fderiv ℝ (fun p => ruler epsilon (Metric.infDist p S)) x w) /
      ruler epsilon (Metric.infDist x S) ^ 2

/-- The actual ambient metric pulls back to the checked curved normal-coordinate metric. -/
theorem ambientGraphMetric_pullback (S : Set E) (epsilon a R : ℝ) (C N : ℝ → E)
    (h : NormalTube S R C N) (e : OpenPartialHomeomorph Coord E)
    (he : (e : Coord → E) = normalMap C N) (z v w : Coord)
    (hz : z ∈ e.source) (hr : |z.1| < R) (heps : 0 < epsilon)
    (D : E →L[ℝ] Coord) (hInv : HasFDerivAt e.symm D (e z))
    (t : E) (k : ℝ) (hC : HasDerivAt C t z.2) (hN : HasDerivAt N (-k • t) z.2)
    (ht : ‖t‖ = 1) (hnt : inner ℝ (N z.2) t = 0) :
    ambientGraphMetric S epsilon a (normalMap C N z)
      (normalDifferential (N z.2) t (1 - k * z.1) v)
      (normalDifferential (N z.2) t (1 - k * z.1) w) =
      CurvedShoreGeometry.metric epsilon a k z.1 v w := by
  have ha := hasFDerivAt_ambientRuler S epsilon R C N h e he z hz hr heps D hInv
  rw [he] at ha
  have hmap := hasFDerivAt_normalMap C N z t k hC hN
  have hchain : fderiv ℝ (normalRuler S epsilon C N) z =
      (fderiv ℝ (fun p => ruler epsilon (Metric.infDist p S)) (normalMap C N z)).comp
        (normalDifferential (N z.2) t (1 - k * z.1)) :=
    (ha.differentiableAt.hasFDerivAt.comp z hmap).fderiv
  rw [← normalGraphMetric_eq S epsilon a R C N h z v w t k heps hr hC hN ht hnt]
  simp [ambientGraphMetric, normalGraphMetric, hchain, hmap.fderiv, normalRuler]

/-- The derivative-based ambient graph metric is the existing rank-one metric for an actual gradient. -/
theorem ambientGraphMetric_eq_scaleMetric (S : Set E) (epsilon a : ℝ) (x u v w : E)
    (hu : HasFDerivAt (fun p => ruler epsilon (Metric.infDist p S)) (innerSL ℝ u) x) :
    ambientGraphMetric S epsilon a x v w =
      ScaleMetric.metric (ruler epsilon (Metric.infDist x S)) a u v w := by
  simp [ambientGraphMetric, hu.fderiv, ScaleMetric.metric]

/-- The constructed chart carries both actual signed-distance and metric identification. -/
theorem exists_normalChart_with_metric (S : Set E) (epsilon a R : ℝ) (C N : ℝ → E)
    (h : NormalTube S R C N) (z : Coord) (B : Coord ≃L[ℝ] E) (k : ℝ)
    (heps : 0 < epsilon) (hr : |z.1| < R) (hh : 1 - k * z.1 ≠ 0)
    (hn : N z.2 = B (1, 0)) (hC : HasDerivAt C (B (0, 1)) z.2)
    (hN : HasDerivAt N (-k • B (0, 1)) z.2)
    (hC1 : ContDiffAt ℝ 1 C z.2) (hN1 : ContDiffAt ℝ 1 N z.2)
    (ht : ‖B (0, 1)‖ = 1) (hnt : inner ℝ (N z.2) (B (0, 1)) = 0) :
    ∃ e : OpenPartialHomeomorph Coord E,
      (e : Coord → E) = normalMap C N ∧ z ∈ e.source ∧
      (∀ᶠ x in 𝓝 (e z), Metric.infDist x S = |(e.symm x).1|) ∧
      ∀ v w : Coord, ambientGraphMetric S epsilon a (e z)
        (normalFrameEquiv B (1 - k * z.1) hh v)
        (normalFrameEquiv B (1 - k * z.1) hh w) =
        CurvedShoreGeometry.metric epsilon a k z.1 v w := by
  obtain ⟨e, he, hz, hInv⟩ := exists_normalChart C N z B k hh hn hC hN hC1 hN1
  refine ⟨e, he, hz, eventually_infDist_normalChart S R C N h e he z hz hr, ?_⟩
  intro v w
  have hp := ambientGraphMetric_pullback S epsilon a R C N h e he z v w hz hr heps
    ((normalFrameEquiv B (1 - k * z.1) hh).symm : E →L[ℝ] Coord)
    (by simpa only [he] using hInv) (B (0, 1)) k hC hN ht hnt
  simpa only [he, hn, normalDifferential_eq_frame B (1 - k * z.1) hh] using hp

end FractalSync.EmbeddedShoreMetric
