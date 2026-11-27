import FractalSync.TubularNeighborhood
import FractalSync.ProviderCertificates

/-!
# Normal radii derived from actual curve derivatives and separation

The normal-chord certificate is derived from a Taylor estimate for an actual
curve, quantitative parameter separation, and separation from remote Shore
points. No distance or nearest-point conclusion is assumed.
-/

namespace FractalSync.NormalRadius

open Set
open TubularNeighborhood
open scoped RealInnerProductSpace Topology

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- An actual derivative variation bound gives a quadratic normal chord bound. -/
theorem normal_chord_taylor (C : ℝ → E) (C' : ℝ → ℝ →L[ℝ] E)
    (U : Set ℝ) (s t L : ℝ) (n : E) (hU : Convex ℝ U) (hs : s ∈ U) (ht : t ∈ U)
    (hn : ‖n‖ = 1) (horth : inner ℝ n (C' s 1) = 0)
    (hC : ∀ z ∈ U, HasFDerivAt C (C' z) z)
    (hL : ∀ z ∈ U, ‖C' z - C' s‖ ≤ L * |z - s|) :
    |inner ℝ n (C t - C s)| ≤ L / 2 * |t - s| ^ 2 := by
  have hseg : ∀ u ∈ Icc (0 : ℝ) 1, s + u • (t - s) ∈ U := by
    exact fun u hu => hU.add_smul_sub_mem hs ht hu
  have hrem := ProviderCertificates.first_order_remainder_bound C C' s (t - s) L
    (fun u hu => hC _ (hseg u hu))
    (fun u hu => by simpa [Real.norm_eq_abs] using hL _ (hseg u hu))
  have hlin : C' s (t - s) = (t - s) • C' s 1 := by
    simpa using (C' s).map_smul (t - s) (1 : ℝ)
  have hi := abs_real_inner_le_norm n (C t - C s - C' s (t - s))
  have horth' : inner ℝ n (C' s (t - s)) = 0 := by
    simp only [hlin, real_inner_smul_right, horth, mul_zero]
  have hi' : |inner ℝ n (C t - C s)| ≤ ‖C t - C s - C' s (t - s)‖ := by
    simpa only [inner_sub_right, horth', sub_zero, hn, one_mul] using hi
  exact hi'.trans (by simpa [Real.norm_eq_abs] using hrem)

/-- Actual curve jets and parameter separation imply the normal-chord certificate. -/
theorem normalChordBound_image (C : ℝ → E) (C' : ℝ → ℝ →L[ℝ] E)
    (U : Set ℝ) (s R L m : ℝ) (n : E) (hU : Convex ℝ U) (hs : s ∈ U)
    (hR : 0 ≤ R) (hm : 0 ≤ m) (hn : ‖n‖ = 1)
    (horth : inner ℝ n (C' s 1) = 0)
    (hC : ∀ z ∈ U, HasFDerivAt C (C' z) z)
    (hL : ∀ z ∈ U, ‖C' z - C' s‖ ≤ L * |z - s|)
    (hsep : ∀ z ∈ U, m * |z - s| ≤ ‖C z - C s‖)
    (hscale : R * L ≤ m ^ 2) :
    NormalChordBound (C '' U) R (C s) n := by
  rintro q ⟨t, ht, rfl⟩
  have hTaylor := mul_le_mul_of_nonneg_left
    (normal_chord_taylor C C' U s t L n hU hs ht hn horth hC hL)
    (mul_nonneg (by norm_num : (0 : ℝ) ≤ 2) hR)
  have hsq := mul_self_le_mul_self (mul_nonneg hm (abs_nonneg (t - s))) (hsep t ht)
  nlinarith [mul_le_mul_of_nonneg_right hscale (sq_nonneg |t - s|)]

/-- A local normal certificate extends to the full Shore when remote points
are farther than twice the proposed normal radius. -/
theorem normalChordBound_of_separation (S T : Set E) (p n : E) (R d : ℝ)
    (hR : 0 ≤ R) (hn : ‖n‖ = 1) (hlocal : NormalChordBound T R p n)
    (hremote : ∀ q ∈ S, q ∉ T → d ≤ ‖q - p‖) (hscale : 2 * R ≤ d) :
    NormalChordBound S R p n := by
  intro q hq
  by_cases ht : q ∈ T
  · exact hlocal q ht
  have hi : |inner ℝ n (q - p)| ≤ ‖q - p‖ := by
    simpa only [hn, one_mul] using abs_real_inner_le_norm n (q - p)
  nlinarith [mul_le_mul_of_nonneg_left hi (mul_nonneg (by norm_num : (0 : ℝ) ≤ 2) hR),
    mul_le_mul_of_nonneg_right (hscale.trans (hremote q hq ht)) (norm_nonneg (q - p))]

/-- A radius balancing local curvature control and remote Shore separation. -/
noncomputable def certifiedRadius (m L d : ℝ) : ℝ := min (m ^ 2 / L) (d / 2)

/-- A positive, explicit normal radius follows from actual derivative control,
parameter separation, and separation from all Shore outside the curve patch. -/
theorem certifiedRadius_normalChordBound (C : ℝ → E) (C' : ℝ → ℝ →L[ℝ] E)
    (S : Set E) (U : Set ℝ) (s L m d : ℝ) (n : E)
    (hU : Convex ℝ U) (hs : s ∈ U) (hLpos : 0 < L) (hm : 0 < m) (hd : 0 < d)
    (hn : ‖n‖ = 1) (horth : inner ℝ n (C' s 1) = 0)
    (hC : ∀ z ∈ U, HasFDerivAt C (C' z) z)
    (hL : ∀ z ∈ U, ‖C' z - C' s‖ ≤ L * |z - s|)
    (hsep : ∀ z ∈ U, m * |z - s| ≤ ‖C z - C s‖)
    (hremote : ∀ q ∈ S, q ∉ C '' U → d ≤ ‖q - C s‖) :
    0 < certifiedRadius m L d ∧ NormalChordBound S (certifiedRadius m L d) (C s) n := by
  have hpos : 0 < certifiedRadius m L d :=
    lt_min (div_pos (sq_pos_of_pos hm) hLpos) (div_pos hd (by norm_num))
  have hcurv : certifiedRadius m L d * L ≤ m ^ 2 :=
    (le_div_iff₀ hLpos).mp (min_le_left _ _)
  have hremoteR : 2 * certifiedRadius m L d ≤ d := by
    have hmin : certifiedRadius m L d ≤ d / 2 := min_le_right _ _
    linarith
  exact ⟨hpos, normalChordBound_of_separation S (C '' U) (C s) n
    (certifiedRadius m L d) d hpos.le hn
    (normalChordBound_image C C' U s (certifiedRadius m L d) L m n hU hs hpos.le
      hm.le hn horth hC hL hsep hcurv) hremote hremoteR⟩

/-- A nonzero actual tangent and a small parameter interval imply a quantitative
chord lower bound. No injectivity or chord separation is assumed. -/
theorem chord_lower_bound_of_derivative (C : ℝ → E) (C' : ℝ → ℝ →L[ℝ] E)
    (U : Set ℝ) (s t L v δ m : ℝ) (hU : Convex ℝ U) (hs : s ∈ U) (ht : t ∈ U)
    (hC : ∀ z ∈ U, HasFDerivAt C (C' z) z)
    (hL : ∀ z ∈ U, ‖C' z - C' s‖ ≤ L * |z - s|)
    (hL0 : 0 ≤ L) (hspeed : v ≤ ‖C' s 1‖) (hdiam : |t - s| ≤ δ)
    (hm : m ≤ v - L * δ / 2) :
    m * |t - s| ≤ ‖C t - C s‖ := by
  have hseg : ∀ u ∈ Icc (0 : ℝ) 1, s + u • (t - s) ∈ U := by
    exact fun u hu => hU.add_smul_sub_mem hs ht hu
  have hrem := ProviderCertificates.first_order_remainder_bound C C' s (t - s) L
    (fun u hu => hC _ (hseg u hu))
    (fun u hu => by simpa [Real.norm_eq_abs] using hL _ (hseg u hu))
  have hlin : C' s (t - s) = (t - s) • C' s 1 := by
    simpa using (C' s).map_smul (t - s) (1 : ℝ)
  have htri : ‖C' s (t - s)‖ ≤ ‖C t - C s‖ + ‖C t - C s - C' s (t - s)‖ := by
    calc
      ‖C' s (t - s)‖ = ‖(C t - C s) - (C t - C s - C' s (t - s))‖ := by
        congr 1
        abel
      _ ≤ _ := norm_sub_le _ _
  simp only [hlin, norm_smul, Real.norm_eq_abs] at htri
  simp only [add_sub_cancel, Real.norm_eq_abs, hlin] at hrem
  nlinarith [mul_le_mul_of_nonneg_right hspeed (abs_nonneg (t - s)),
    mul_le_mul_of_nonneg_right hm (abs_nonneg (t - s)),
    mul_le_mul_of_nonneg_left hdiam (mul_nonneg hL0 (abs_nonneg (t - s)))]

/-- Actual tangent, derivative-variation, and parameter-width bounds supply the
local chord hypothesis of the explicit normal-radius theorem. -/
theorem normal_radius_of_curve_jet (C : ℝ → E) (C' : ℝ → ℝ →L[ℝ] E)
    (S : Set E) (U : Set ℝ) (s L v δ d : ℝ) (n : E)
    (hU : Convex ℝ U) (hs : s ∈ U) (hLpos : 0 < L)
    (hm : 0 < v - L * δ / 2) (hd : 0 < d) (hn : ‖n‖ = 1)
    (horth : inner ℝ n (C' s 1) = 0)
    (hC : ∀ z ∈ U, HasFDerivAt C (C' z) z)
    (hL : ∀ z ∈ U, ‖C' z - C' s‖ ≤ L * |z - s|)
    (hspeed : v ≤ ‖C' s 1‖) (hdiam : ∀ z ∈ U, |z - s| ≤ δ)
    (hremote : ∀ q ∈ S, q ∉ C '' U → d ≤ ‖q - C s‖) :
    0 < certifiedRadius (v - L * δ / 2) L d ∧
      NormalChordBound S (certifiedRadius (v - L * δ / 2) L d) (C s) n := by
  exact certifiedRadius_normalChordBound C C' S U s L (v - L * δ / 2) d n
    hU hs hLpos hm hd hn horth hC hL
    (fun z hz => chord_lower_bound_of_derivative C C' U s z L v δ
      (v - L * δ / 2) hU hs hz hC hL hLpos.le hspeed (hdiam z hz) le_rfl) hremote

/-- The radius is uniform over a base patch when all jet and remote-separation
bounds are uniform. The larger patch controls all local chords. -/
theorem uniform_normal_radius (C : ℝ → E) (C' : ℝ → ℝ →L[ℝ] E) (N : ℝ → E)
    (S : Set E) (U V : Set ℝ) (L v δ d : ℝ) (hU : Convex ℝ U) (hVU : V ⊆ U)
    (hLpos : 0 < L) (hm : 0 < v - L * δ / 2) (hd : 0 < d)
    (hN : ∀ s ∈ V, ‖N s‖ = 1) (horth : ∀ s ∈ V, inner ℝ (N s) (C' s 1) = 0)
    (hC : ∀ z ∈ U, HasFDerivAt C (C' z) z)
    (hL : ∀ s ∈ V, ∀ z ∈ U, ‖C' z - C' s‖ ≤ L * |z - s|)
    (hspeed : ∀ s ∈ V, v ≤ ‖C' s 1‖)
    (hdiam : ∀ s ∈ V, ∀ z ∈ U, |z - s| ≤ δ)
    (hremote : ∀ s ∈ V, ∀ q ∈ S, q ∉ C '' U → d ≤ ‖q - C s‖) :
    0 < certifiedRadius (v - L * δ / 2) L d ∧
      ∀ s ∈ V, NormalChordBound S (certifiedRadius (v - L * δ / 2) L d) (C s) (N s) := by
  refine ⟨lt_min (div_pos (sq_pos_of_pos hm) hLpos) (div_pos hd (by norm_num)), ?_⟩
  intro s hs
  exact (normal_radius_of_curve_jet C C' S U s L v δ d (N s) hU (hVU hs)
    hLpos hm hd (hN s hs) (horth s hs) hC (hL s hs) (hspeed s hs)
    (hdiam s hs) (hremote s hs)).2

omit [InnerProductSpace ℝ E] in
/-- A compact base image disjoint from the closed remote Shore has a uniform
positive remote-separation bound. Empty remote sets are included. -/
theorem exists_remote_separation (C : ℝ → E) (S : Set E) (U V : Set ℝ)
    (hV : IsCompact V) (hC : ContinuousOn C V)
    (hdisj : Disjoint (C '' V) (closure (S \ C '' U))) :
    ∃ d : ℝ, 0 < d ∧ ∀ s ∈ V, ∀ q ∈ S, q ∉ C '' U → d ≤ ‖q - C s‖ := by
  obtain ⟨d, hd, hsep⟩ := EMetric.exists_pos_forall_lt_edist
    (hV.image_of_continuousOn hC) isClosed_closure hdisj
  refine ⟨d, by exact_mod_cast hd, ?_⟩
  intro s hs q hq hremote
  have hdist := hsep (C s) ⟨s, hs, rfl⟩ q (subset_closure ⟨hq, hremote⟩)
  rw [edist_nndist, ENNReal.coe_lt_coe] at hdist
  have hreal : (d : ℝ) < dist (C s) q := by exact_mod_cast hdist
  simpa only [dist_eq_norm, norm_sub_rev] using hreal.le

/-- A bound on an actual second derivative supplies the derivative-variation
bound used by the normal-radius construction. -/
theorem derivative_variation_of_second_derivative (C' : ℝ → ℝ →L[ℝ] E)
    (C'' : ℝ → ℝ →L[ℝ] (ℝ →L[ℝ] E)) (U : Set ℝ) (L : ℝ)
    (hU : Convex ℝ U) (hC' : ∀ z ∈ U, HasFDerivAt C' (C'' z) z)
    (hbound : ∀ z ∈ U, ‖C'' z‖ ≤ L) :
    ∀ s ∈ U, ∀ t ∈ U, ‖C' t - C' s‖ ≤ L * |t - s| := by
  intro s hs t ht
  simpa only [Real.norm_eq_abs] using
    Convex.norm_image_sub_le_of_norm_hasFDerivWithin_le
      (fun z hz => (hC' z hz).hasFDerivWithinAt) hbound hU hs ht

/-- Actual `C²` regularity on a compact convex parameter patch supplies a
positive finite derivative-variation constant. Its value is existential. -/
theorem exists_derivative_variation_bound (C : ℝ → E) (U : Set ℝ)
    (hU : IsCompact U) (hconv : Convex ℝ U)
    (hC : ∀ z ∈ U, ContDiffAt ℝ 2 C z) :
    ∃ L : ℝ, 0 < L ∧ ∀ s ∈ U, ∀ t ∈ U,
      ‖fderiv ℝ C t - fderiv ℝ C s‖ ≤ L * |t - s| := by
  have hD : ∀ z ∈ U, ContDiffAt ℝ 1 (fderiv ℝ C) z :=
    fun z hz => (hC z hz).fderiv_right (by norm_num)
  have hcont : ContinuousOn (fderiv ℝ (fderiv ℝ C)) U := by
    intro z hz
    exact ((hD z hz).fderiv_right (m := 0) (by norm_num)).continuousAt.continuousWithinAt
  obtain ⟨L, hL⟩ := hU.exists_bound_of_continuousOn
    (f := fderiv ℝ (fderiv ℝ C)) hcont
  refine ⟨max L 1, lt_of_lt_of_le zero_lt_one (le_max_right _ _), ?_⟩
  exact derivative_variation_of_second_derivative (fderiv ℝ C) (fderiv ℝ (fderiv ℝ C))
    U (max L 1) hconv (fun z hz => ((hD z hz).differentiableAt (by norm_num)).hasFDerivAt)
    (fun z hz => (hL z hz).trans (le_max_left _ _))

/-- The same actual-jet bounds make the curve injective on the base patch. -/
theorem injOn_of_curve_jet (C : ℝ → E) (C' : ℝ → ℝ →L[ℝ] E)
    (U V : Set ℝ) (L v δ : ℝ) (hU : Convex ℝ U) (hVU : V ⊆ U)
    (hL0 : 0 ≤ L) (hm : 0 < v - L * δ / 2)
    (hC : ∀ z ∈ U, HasFDerivAt C (C' z) z)
    (hL : ∀ s ∈ V, ∀ z ∈ U, ‖C' z - C' s‖ ≤ L * |z - s|)
    (hspeed : ∀ s ∈ V, v ≤ ‖C' s 1‖)
    (hdiam : ∀ s ∈ V, ∀ z ∈ U, |z - s| ≤ δ) : Set.InjOn C V := by
  intro s hs t ht heq
  have hchord := chord_lower_bound_of_derivative C C' U s t L v δ
    (v - L * δ / 2) hU (hVU hs) (hVU ht) hC (hL s hs) hL0
    (hspeed s hs) (hdiam s hs t (hVU ht)) le_rfl
  rw [← heq, sub_self, norm_zero] at hchord
  have habs : |t - s| = 0 := by nlinarith [abs_nonneg (t - s)]
  exact (sub_eq_zero.mp (abs_eq_zero.mp habs)).symm

omit [InnerProductSpace ℝ E] in
/-- For an injectively parameterized compact curve, a compact base patch inside
an open parameter patch is uniformly separated from the remaining curve.
The separation is derived from compactness and injectivity. -/
theorem exists_remote_separation_of_compact_injective (C : ℝ → E) (P U V : Set ℝ)
    (hP : IsCompact P) (hV : IsCompact V) (hU : IsOpen U)
    (hVP : V ⊆ P) (hVU : V ⊆ U) (hC : ContinuousOn C P) (hinj : Set.InjOn C P) :
    ∃ d : ℝ, 0 < d ∧ ∀ s ∈ V, ∀ q ∈ C '' P,
      q ∉ C '' U → d ≤ ‖q - C s‖ := by
  have hcompact : IsCompact (C '' (P \ U)) :=
    (hP.diff hU).image_of_continuousOn (hC.mono diff_subset)
  have hsub : C '' P \ C '' U ⊆ C '' (P \ U) := by
    rintro q ⟨⟨t, ht, rfl⟩, hq⟩
    exact ⟨t, ⟨ht, fun htU => hq ⟨t, htU, rfl⟩⟩, rfl⟩
  have hcl : closure (C '' P \ C '' U) ⊆ C '' (P \ U) :=
    closure_minimal hsub hcompact.isClosed
  have hdisj : Disjoint (C '' V) (closure (C '' P \ C '' U)) := by
    apply disjoint_left.mpr
    rintro q ⟨s, hs, rfl⟩ hq
    obtain ⟨t, ht, heq⟩ := hcl hq
    have heq' : t = s := hinj ht.1 (hVP hs) heq
    exact ht.2 (heq' ▸ hVU hs)
  exact exists_remote_separation C (C '' P) U V hV (hC.mono hVP) hdisj

end FractalSync.NormalRadius
