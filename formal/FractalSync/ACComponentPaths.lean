import FractalSync.PathOperations
import Mathlib.Analysis.Normed.Module.Convex
import Mathlib.Topology.Connected.Basic

/-!
# Absolutely continuous paths inside open connected components

AC reachability inside an open subset is relatively clopen: sufficiently
small balls admit affine segments, and AC paths concatenate. Connectedness
therefore supplies an AC path between any two points of one open component.
This discharges the off-Shore reference-path premise in the zero-floor limit.
-/

namespace FractalSync.DistanceGraph

open Set

variable {E : Type*} [NormedAddCommGroup E]

/-- Two points are joined by an actual AC path whose unit-interval image lies
inside the specified set. -/
def ACJoinedIn (U : Set E) (x y : E) : Prop :=
  ∃ gamma : ℝ → E, AbsolutelyContinuousOnInterval gamma 0 1 ∧
    gamma 0 = x ∧ gamma 1 = y ∧ MapsTo gamma (Icc (0 : ℝ) 1) U

/-- An AC path inside a set has its terminal endpoint in that set. -/
theorem ACJoinedIn.mem_right {U : Set E} {x y : E} (h : ACJoinedIn U x y) : y ∈ U := by
  obtain ⟨gamma, _, _, hend, hmem⟩ := h
  exact hend ▸ hmem ⟨zero_le_one, le_rfl⟩

/-- A constant path witnesses AC reachability at every point of the domain. -/
theorem ACJoinedIn.refl {U : Set E} {x : E} (hx : x ∈ U) : ACJoinedIn U x x := by
  exact ⟨fun _ => x, (LipschitzWith.const x).lipschitzOnWith.absolutelyContinuousOnInterval,
    rfl, rfl, fun _ _ => hx⟩

/-- Concatenation preserves AC reachability within a domain. -/
theorem ACJoinedIn.trans {U : Set E} {x y z : E}
    (hxy : ACJoinedIn U x y) (hyz : ACJoinedIn U y z) : ACJoinedIn U x z := by
  obtain ⟨gamma, hgamma, hgamma₀, hgamma₁, hgammaU⟩ := hxy
  obtain ⟨eta, heta, heta₀, heta₁, hetaU⟩ := hyz
  refine ⟨concatPath gamma eta,
    ac_concatPath gamma eta hgamma heta (hgamma₁.trans heta₀.symm),
    by simpa only [concatPath_zero] using hgamma₀,
    by simpa only [concatPath_one] using heta₁, ?_⟩
  intro t ht
  exact if h : t ≤ 1 / 2 then
    (by simpa only [concatPath, ACComposition.gluePath, if_pos h] using
      hgammaU ⟨by linarith [ht.1], by linarith⟩)
  else
    (by simpa only [concatPath, ACComposition.gluePath, if_neg h] using
      hetaU ⟨by linarith, by linarith [ht.2]⟩)

variable [NormedSpace ℝ E]

/-- A segment inside a convex subset of a domain is an admissible AC path. -/
theorem ACJoinedIn.of_convex_subset {U V : Set E} {x y : E}
    (hV : Convex ℝ V) (hVU : V ⊆ U) (hx : x ∈ V) (hy : y ∈ V) :
    ACJoinedIn U x y := by
  exact ⟨(fun t => AffineMap.lineMap x y t), absolutelyContinuousOnInterval_lineMap y x,
    by simp only [AffineMap.lineMap_apply_zero],
    by simp only [AffineMap.lineMap_apply_one], fun _ ht => hVU (hV.lineMap_mem hx hy ht)⟩

/-- AC reachability from a fixed point is open inside an open domain, because
small terminal segments may be appended. -/
theorem isOpen_acJoinedIn (U : Set E) (hU : IsOpen U) (x : E) :
    IsOpen {y | ACJoinedIn U x y} := by
  refine Metric.isOpen_iff.2 fun y hy => ?_
  obtain ⟨r, hr, hball⟩ := Metric.isOpen_iff.1 hU y (ACJoinedIn.mem_right hy)
  exact ⟨r, hr, fun _ hz => ACJoinedIn.trans hy
    (ACJoinedIn.of_convex_subset (convex_ball y r) hball (Metric.mem_ball_self hr) hz)⟩

/-- The relative complement of an AC reachability class is open: reaching
any sufficiently close point would also reach the original endpoint. -/
theorem isOpen_diff_acJoinedIn (U : Set E) (hU : IsOpen U) (x : E) :
    IsOpen (U \ {y | ACJoinedIn U x y}) := by
  refine Metric.isOpen_iff.2 fun y hy => ?_
  obtain ⟨r, hr, hball⟩ := Metric.isOpen_iff.1 hU y hy.1
  exact ⟨r, hr, fun _ hz => ⟨hball hz, fun hxy =>
    hy.2 (ACJoinedIn.trans hxy (ACJoinedIn.of_convex_subset (convex_ball y r)
      hball hz (Metric.mem_ball_self hr)))⟩⟩

/-- Every two points of the same connected component of an open subset of a
real normed space are joined inside that subset by an actual AC path. -/
theorem acJoinedIn_of_mem_connectedComponentIn (U : Set E) (hU : IsOpen U)
    (x y : E) (hx : x ∈ U) (hy : y ∈ connectedComponentIn U x) :
    ACJoinedIn U x y := by
  have hcover : connectedComponentIn U x ⊆
      {z | ACJoinedIn U x z} ∪ (U \ {z | ACJoinedIn U x z}) :=
    fun z hz => (Classical.em (ACJoinedIn U x z)).elim Or.inl
      (fun hn => Or.inr ⟨connectedComponentIn_subset U x hz, hn⟩)
  have hsub : connectedComponentIn U x ⊆ {z | ACJoinedIn U x z} :=
    IsPreconnected.subset_left_of_subset_union (isOpen_acJoinedIn U hU x)
      (isOpen_diff_acJoinedIn U hU x)
      (Set.disjoint_left.2 fun _ hreach hnot => hnot.2 hreach) hcover
      ⟨x, mem_connectedComponentIn hx, ACJoinedIn.refl hx⟩
      isPreconnected_connectedComponentIn
  exact hsub hy

/-- Points in one connected component of the complement of a closed Shore
admit an actual AC unit-interval path avoiding Shore everywhere. No finite
dimensionality, nonemptiness of Shore, or length-integrability premise is needed. -/
theorem exists_ac_path_avoiding_of_mem_connectedComponentIn
    (S : Set E) (hclosed : IsClosed S) (x y : E) (hx : x ∉ S)
    (hy : y ∈ connectedComponentIn Sᶜ x) :
    ∃ gamma : ℝ → E, AbsolutelyContinuousOnInterval gamma 0 1 ∧
      gamma 0 = x ∧ gamma 1 = y ∧
        ∀ t ∈ uIcc (0 : ℝ) 1, gamma t ∉ S := by
  simpa only [ACJoinedIn, MapsTo, uIcc_of_le (zero_le_one : (0 : ℝ) ≤ 1),
    Set.mem_compl_iff] using
      acJoinedIn_of_mem_connectedComponentIn Sᶜ hclosed.isOpen_compl x y hx hy

end FractalSync.DistanceGraph
