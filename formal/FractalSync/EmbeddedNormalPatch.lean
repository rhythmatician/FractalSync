import FractalSync.NormalRadius
import FractalSync.EmbeddedShoreMetric

/-!
# Derived normal radii on a local curve patch

The geometric certificate is localized to a parameter patch. Actual curve-jet
bounds derive this certificate, injectivity, and signed-distance and metric
identities in the local chart supplied by the inverse function theorem.
-/

namespace FractalSync.EmbeddedNormalPatch

open Set TubularNeighborhood NormalRadius EmbeddedShoreMetric
open scoped Topology RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- A normal-neighborhood certificate restricted to the base parameter patch. -/
structure NormalPatch (S : Set E) (R : ℝ) (C N : ℝ → E) (V : Set ℝ) : Prop where
  radius_pos : 0 < R
  foot_mem : ∀ s ∈ V, C s ∈ S
  normal_unit : ∀ s ∈ V, ‖N s‖ = 1
  chord_bound : ∀ s ∈ V, NormalChordBound S R (C s) (N s)

/-- Actual curve jets and remote separation derive a uniform normal patch
and injectivity of its entire open normal strip. -/
theorem normalPatch_of_curve_jet (C : ℝ → E) (C' : ℝ → ℝ →L[ℝ] E) (N : ℝ → E)
    (S : Set E) (U V : Set ℝ) (L v δ d : ℝ) (hU : Convex ℝ U) (hVU : V ⊆ U)
    (hLpos : 0 < L) (hm : 0 < v - L * δ / 2) (hd : 0 < d)
    (hfoot : ∀ s ∈ V, C s ∈ S) (hN : ∀ s ∈ V, ‖N s‖ = 1)
    (horth : ∀ s ∈ V, inner ℝ (N s) (C' s 1) = 0)
    (hC : ∀ z ∈ U, HasFDerivAt C (C' z) z)
    (hL : ∀ s ∈ V, ∀ z ∈ U, ‖C' z - C' s‖ ≤ L * |z - s|)
    (hspeed : ∀ s ∈ V, v ≤ ‖C' s 1‖)
    (hdiam : ∀ s ∈ V, ∀ z ∈ U, |z - s| ≤ δ)
    (hremote : ∀ s ∈ V, ∀ q ∈ S, q ∉ C '' U → d ≤ ‖q - C s‖) :
    NormalPatch S (certifiedRadius (v - L * δ / 2) L d) C N V ∧
      Set.InjOn (normalMap C N)
        {z : Coord | z.2 ∈ V ∧ |z.1| < certifiedRadius (v - L * δ / 2) L d} := by
  obtain ⟨hR, hbound⟩ := uniform_normal_radius C C' N S U V L v δ d hU hVU hLpos hm hd
    hN horth hC hL hspeed hdiam hremote
  exact ⟨⟨hR, hfoot, hN, hbound⟩, normalMap_injOn_strip S _ C N V hR
    (injOn_of_curve_jet C C' U V L v δ hU hVU hLpos.le hm hC hL hspeed hdiam)
    hfoot hN hbound⟩

/-- An injective compact parameterization supplies remote separation itself.
The stated local jet bounds then produce an actual uniform normal patch. -/
theorem exists_normalPatch_of_compact_injective (C : ℝ → E) (C' : ℝ → ℝ →L[ℝ] E)
    (N : ℝ → E) (P U V : Set ℝ) (L v δ : ℝ)
    (hP : IsCompact P) (hV : IsCompact V) (hUopen : IsOpen U) (hU : Convex ℝ U)
    (hVP : V ⊆ P) (hVU : V ⊆ U) (hcont : ContinuousOn C P) (hinj : Set.InjOn C P)
    (hLpos : 0 < L) (hm : 0 < v - L * δ / 2)
    (hN : ∀ s ∈ V, ‖N s‖ = 1) (horth : ∀ s ∈ V, inner ℝ (N s) (C' s 1) = 0)
    (hC : ∀ z ∈ U, HasFDerivAt C (C' z) z)
    (hL : ∀ s ∈ V, ∀ z ∈ U, ‖C' z - C' s‖ ≤ L * |z - s|)
    (hspeed : ∀ s ∈ V, v ≤ ‖C' s 1‖)
    (hdiam : ∀ s ∈ V, ∀ z ∈ U, |z - s| ≤ δ) :
    ∃ R : ℝ, NormalPatch (C '' P) R C N V ∧
      Set.InjOn (normalMap C N) {z : Coord | z.2 ∈ V ∧ |z.1| < R} := by
  obtain ⟨d, hd, hsep⟩ := exists_remote_separation_of_compact_injective C P U V
    hP hV hUopen hVP hVU hcont hinj
  exact ⟨certifiedRadius (v - L * δ / 2) L d,
    normalPatch_of_curve_jet C C' N (C '' P) U V L v δ d hU hVU hLpos hm hd
      (fun s hs => ⟨s, hVP hs, rfl⟩) hN horth hC hL hspeed hdiam hsep⟩

/-- The actual set distance equals the magnitude of the normal coordinate. -/
theorem infDist_normalMap (S : Set E) (R : ℝ) (C N : ℝ → E) (V : Set ℝ)
    (h : NormalPatch S R C N V) (z : Coord) (hs : z.2 ∈ V) (hr : |z.1| ≤ R) :
    Metric.infDist (normalMap C N z) S = |z.1| := by
  exact infDist_normalOffset S R z.1 (C z.2) (N z.2) h.radius_pos (h.foot_mem _ hs)
    (h.normal_unit _ hs) (h.chord_bound _ hs) hr

/-- Inside the derived radius, the represented Shore foot is unique. -/
theorem unique_nearest (S : Set E) (R : ℝ) (C N : ℝ → E) (V : Set ℝ)
    (h : NormalPatch S R C N V) (z : Coord) (hs : z.2 ∈ V) (hr : |z.1| < R)
    (q : E) (hq : q ∈ S)
    (hnear : dist (normalMap C N z) q = Metric.infDist (normalMap C N z) S) :
    q = C z.2 := by
  exact normalOffset_unique_nearest S R z.1 (C z.2) (N z.2) q h.radius_pos
    (h.foot_mem _ hs) (h.normal_unit _ hs) (h.chord_bound _ hs) hr hq hnear

/-- The actual ruler agrees with the signed-coordinate ruler on the patch. -/
theorem normalRuler_eq (S : Set E) (epsilon R : ℝ) (C N : ℝ → E) (V : Set ℝ)
    (h : NormalPatch S R C N V) (z : Coord) (hs : z.2 ∈ V) (hr : |z.1| ≤ R) :
    normalRuler S epsilon C N z = ruler epsilon z.1 := by
  rw [normalRuler, infDist_normalMap S R C N V h z hs hr]
  simp [ruler]

/-- Actual ruler differentiation requires only a parameter neighborhood within
the certified patch; it includes the Shore coordinate zero. -/
theorem hasFDerivAt_normalRuler (S : Set E) (epsilon R : ℝ) (C N : ℝ → E) (V : Set ℝ)
    (h : NormalPatch S R C N V) (z : Coord) (hV : V ∈ 𝓝 z.2)
    (he : 0 < epsilon) (hr : |z.1| < R) :
    HasFDerivAt (normalRuler S epsilon C N)
      ((z.1 / ruler epsilon z.1) • ContinuousLinearMap.fst ℝ ℝ ℝ) z := by
  have hcoord := (hasDerivAt_ruler epsilon z.1 he).comp_hasFDerivAt z
    (ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt
  apply hcoord.congr_of_eventuallyEq
  filter_upwards [(continuous_fst.abs.continuousAt).eventually_lt_const hr,
    (continuous_snd.tendsto z).eventually hV] with q hq hs
  exact normalRuler_eq S epsilon R C N V h q hs hq.le

/-- The local coordinate metric is the actual graph metric induced by the
normal map and regularized set distance. -/
theorem normalGraphMetric_eq (S : Set E) (epsilon a R : ℝ) (C N : ℝ → E) (V : Set ℝ)
    (h : NormalPatch S R C N V) (z v w : Coord) (t : E) (k : ℝ)
    (hV : V ∈ 𝓝 z.2) (he : 0 < epsilon) (hr : |z.1| < R)
    (hC : HasDerivAt C t z.2) (hN : HasDerivAt N (-k • t) z.2)
    (ht : ‖t‖ = 1) (hnt : inner ℝ (N z.2) t = 0) :
    normalGraphMetric S epsilon a C N z v w =
      CurvedShoreGeometry.metric epsilon a k z.1 v w := by
  have hs : z.2 ∈ V := mem_of_mem_nhds hV
  rw [normalGraphMetric, (hasFDerivAt_normalMap C N z t k hC hN).fderiv,
    (hasFDerivAt_normalRuler S epsilon R C N V h z hV he hr).fderiv,
    normalRuler_eq S epsilon R C N V h z hs hr.le,
    inner_normalDifferential (N z.2) t (1 - k * z.1) v w (h.normal_unit _ hs) ht hnt]
  dsimp [CurvedShoreGeometry.metric, FlatShoreGeometry.radialMetric, FlatShoreGeometry.tangentialMetric]
  have hprod : a ^ 2 * (z.1 / ruler epsilon z.1 * v.1) *
      (z.1 / ruler epsilon z.1 * w.1) =
      a ^ 2 * z.1 ^ 2 * v.1 * w.1 / ruler epsilon z.1 ^ 2 := by ring
  rw [hprod, ruler_sq]
  field_simp
  ring

/-- A local inverse normal chart represents actual unsigned distance by the
absolute value of its signed coordinate, provided its parameter is in the patch. -/
theorem infDist_normalChart (S : Set E) (R : ℝ) (C N : ℝ → E) (V : Set ℝ)
    (h : NormalPatch S R C N V) (e : OpenPartialHomeomorph Coord E)
    (he : (e : Coord → E) = normalMap C N) (x : E)
    (hx : x ∈ e.target) (hs : (e.symm x).2 ∈ V) (hr : |(e.symm x).1| ≤ R) :
    Metric.infDist x S = |(e.symm x).1| := by
  calc
    Metric.infDist x S = Metric.infDist (normalMap C N (e.symm x)) S := by
      rw [← he, e.right_inv hx]
    _ = |(e.symm x).1| := infDist_normalMap S R C N V h (e.symm x) hs hr

/-- Distance identification holds on a full ambient neighborhood, using only
a parameter neighborhood inside the local normal patch. -/
theorem eventually_infDist_normalChart (S : Set E) (R : ℝ) (C N : ℝ → E) (V : Set ℝ)
    (h : NormalPatch S R C N V) (e : OpenPartialHomeomorph Coord E)
    (he : (e : Coord → E) = normalMap C N) (z : Coord)
    (hz : z ∈ e.source) (hV : V ∈ 𝓝 z.2) (hr : |z.1| < R) :
    ∀ᶠ x in 𝓝 (e z), Metric.infDist x S = |(e.symm x).1| := by
  have ht : e z ∈ e.target := e.map_source hz
  have hc := (e.continuousAt_symm ht).fst.abs
  have hs : ∀ᶠ x in 𝓝 (e z), (e.symm x).2 ∈ V :=
    ((e.continuousAt_symm ht).snd).eventually (by simpa only [e.left_inv hz] using hV)
  filter_upwards [e.open_target.mem_nhds ht, hs,
    hc.eventually_lt_const (by simpa only [e.left_inv hz] using hr)] with x hx hxs hxr
  exact infDist_normalChart S R C N V h e he x hx hxs hxr.le

/-- The actual ambient regularized distance is differentiable in the derived
patch, including on Shore itself. -/
theorem hasFDerivAt_ambientRuler (S : Set E) (epsilon R : ℝ) (C N : ℝ → E) (V : Set ℝ)
    (h : NormalPatch S R C N V) (e : OpenPartialHomeomorph Coord E)
    (he : (e : Coord → E) = normalMap C N) (z : Coord)
    (hz : z ∈ e.source) (hV : V ∈ 𝓝 z.2) (hr : |z.1| < R) (heps : 0 < epsilon)
    (D : E →L[ℝ] Coord) (hInv : HasFDerivAt e.symm D (e z)) :
    HasFDerivAt (fun x => ruler epsilon (Metric.infDist x S))
      ((z.1 / ruler epsilon z.1) • (ContinuousLinearMap.fst ℝ ℝ ℝ).comp D) (e z) := by
  have hd : HasFDerivAt (fun x => ruler epsilon (e.symm x).1)
      ((z.1 / ruler epsilon z.1) • (ContinuousLinearMap.fst ℝ ℝ ℝ).comp D) (e z) := by
    simpa only [e.left_inv hz] using
      ((hasDerivAt_ruler epsilon (e.symm (e z)).1 heps).comp_hasFDerivAt (e z)
        ((ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt.comp (e z) hInv))
  apply hd.congr_of_eventuallyEq
  filter_upwards [eventually_infDist_normalChart S R C N V h e he z hz hV hr] with x hx
  simp only [hx, ruler, sq_abs]

/-- The actual ambient graph metric pulls back to the curved coordinate metric
using a local patch certificate. -/
theorem ambientGraphMetric_pullback (S : Set E) (epsilon a R : ℝ) (C N : ℝ → E) (V : Set ℝ)
    (h : NormalPatch S R C N V) (e : OpenPartialHomeomorph Coord E)
    (he : (e : Coord → E) = normalMap C N) (z v w : Coord)
    (hz : z ∈ e.source) (hV : V ∈ 𝓝 z.2) (hr : |z.1| < R) (heps : 0 < epsilon)
    (D : E →L[ℝ] Coord) (hInv : HasFDerivAt e.symm D (e z))
    (t : E) (k : ℝ) (hC : HasDerivAt C t z.2) (hN : HasDerivAt N (-k • t) z.2)
    (ht : ‖t‖ = 1) (hnt : inner ℝ (N z.2) t = 0) :
    ambientGraphMetric S epsilon a (normalMap C N z)
      (normalDifferential (N z.2) t (1 - k * z.1) v)
      (normalDifferential (N z.2) t (1 - k * z.1) w) =
      CurvedShoreGeometry.metric epsilon a k z.1 v w := by
  have ha := hasFDerivAt_ambientRuler S epsilon R C N V h e he z hz hV hr heps D hInv
  rw [he] at ha
  have hmap := hasFDerivAt_normalMap C N z t k hC hN
  have hchain : fderiv ℝ (normalRuler S epsilon C N) z =
      (fderiv ℝ (fun p => ruler epsilon (Metric.infDist p S)) (normalMap C N z)).comp
        (normalDifferential (N z.2) t (1 - k * z.1)) :=
    (ha.differentiableAt.hasFDerivAt.comp z hmap).fderiv
  rw [← normalGraphMetric_eq S epsilon a R C N V h z v w t k hV heps hr hC hN ht hnt]
  simp [ambientGraphMetric, normalGraphMetric, hchain, hmap.fderiv, normalRuler]

/-- Actual Frenet data and the inverse function theorem give a chart with both
the ambient signed-distance identity and the induced metric identification. -/
theorem exists_normalChart_with_metric (S : Set E) (epsilon a R : ℝ) (C N : ℝ → E)
    (V : Set ℝ) (h : NormalPatch S R C N V) (z : Coord) (B : Coord ≃L[ℝ] E) (k : ℝ)
    (hV : V ∈ 𝓝 z.2) (heps : 0 < epsilon) (hr : |z.1| < R) (hh : 1 - k * z.1 ≠ 0)
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
  refine ⟨e, he, hz, eventually_infDist_normalChart S R C N V h e he z hz hV hr, ?_⟩
  intro v w
  have hp := ambientGraphMetric_pullback S epsilon a R C N V h e he z v w hz hV hr heps
    ((normalFrameEquiv B (1 - k * z.1) hh).symm : E →L[ℝ] Coord)
    (by simpa only [he] using hInv) (B (0, 1)) k hC hN ht hnt
  simpa only [he, hn, normalDifferential_eq_frame B (1 - k * z.1) hh] using hp

end FractalSync.EmbeddedNormalPatch
