import FractalSync.PairwiseDistance
import FractalSync.PathOperations
import FractalSync.DisplacementBounds

/-!
# Metric axioms for the actual absolutely continuous path infimum

Reversal identifies the admissible length classes. Concatenation and paths
arbitrarily close to their infima prove the triangle inequality without
assuming minimizing-path existence. The logarithmic displacement bound
supplies separation, yielding an explicit metric structure.
-/

namespace FractalSync.DistanceGraph

open Set Filter
open scoped Topology

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Reversal identifies the actual admissible length sets in both directions. -/
theorem pairwisePathLengths_symm (S : Set E) (epsilon a : ℝ) (x y : E) :
    pairwisePathLengths S epsilon a x y = pairwisePathLengths S epsilon a y x := by
  ext L
  constructor <;> rintro ⟨gamma, hgamma, hstart, hend, hlength⟩
  all_goals
    exact ⟨reversePath gamma, ac_reversePath gamma hgamma,
      by simpa only [reversePath_zero] using hend,
      by simpa only [reversePath_one] using hstart,
      (acGraphLength_reversePath S epsilon a gamma).trans hlength⟩

/-- The actual pairwise path infimum is symmetric, by reversal of every path. -/
theorem intrinsicDistance_symm (S : Set E) (epsilon a : ℝ) (x y : E) :
    intrinsicDistance S epsilon a x y = intrinsicDistance S epsilon a y x := by
  exact congrArg sInf (pairwisePathLengths_symm S epsilon a x y)

/-- Every positive tolerance admits an actual AC path whose length is within
that tolerance of the pairwise infimum. No minimizing path is presupposed. -/
theorem exists_path_length_lt_intrinsicDistance_add (S : Set E) (epsilon a : ℝ)
    (x y : E) (delta : ℝ) (hdelta : 0 < delta) :
    ∃ gamma : ℝ → E, AbsolutelyContinuousOnInterval gamma 0 1 ∧
      gamma 0 = x ∧ gamma 1 = y ∧
        acGraphLength S epsilon a gamma < intrinsicDistance S epsilon a x y + delta := by
  obtain ⟨L, ⟨gamma, hgamma, hstart, hend, hlength⟩, hL⟩ :=
    exists_lt_of_csInf_lt (pairwisePathLengths_nonempty S epsilon a x y)
      (lt_add_of_pos_right (intrinsicDistance S epsilon a x y) hdelta)
  exact ⟨gamma, hgamma, hstart, hend, hlength.symm ▸ hL⟩

/-- Concatenating paths arbitrarily close to their infima proves the triangle
inequality for the actual pairwise distance. -/
theorem intrinsicDistance_triangle [FiniteDimensional ℝ E]
    (S : Set E) (epsilon a : ℝ) (x y z : E) (he : 0 < epsilon) :
    intrinsicDistance S epsilon a x z ≤
      intrinsicDistance S epsilon a x y + intrinsicDistance S epsilon a y z := by
  refine le_of_forall_pos_le_add fun delta hdelta => ?_
  obtain ⟨gamma, hgamma, hgamma₀, hgamma₁, hgammaLength⟩ :=
    exists_path_length_lt_intrinsicDistance_add S epsilon a x y (delta / 2) (half_pos hdelta)
  obtain ⟨eta, heta, heta₀, heta₁, hetaLength⟩ :=
    exists_path_length_lt_intrinsicDistance_add S epsilon a y z (delta / 2) (half_pos hdelta)
  have hbound : intrinsicDistance S epsilon a x z ≤
      acGraphLength S epsilon a gamma + acGraphLength S epsilon a eta :=
    (intrinsicDistance_le_length S epsilon a x z (concatPath gamma eta)
      (ac_concatPath gamma eta hgamma heta (hgamma₁.trans heta₀.symm))
      (by simpa only [concatPath_zero] using hgamma₀)
      (by simpa only [concatPath_one] using heta₁)).trans_eq
        (acGraphLength_concatPath S epsilon a gamma eta he hgamma heta
          (hgamma₁.trans heta₀.symm))
  linarith

/-- The actual positive-floor path infimum defines a metric on the ambient
space. This is an explicit value, not a global instance replacing its norm
metric. Its uniformity is the intrinsic metric's own uniformity. -/
noncomputable def intrinsicMetricSpace [FiniteDimensional ℝ E]
    (S : Set E) (epsilon a : ℝ) (he : 0 < epsilon) : MetricSpace E where
  dist := intrinsicDistance S epsilon a
  dist_self := intrinsicDistance_self S epsilon a
  dist_comm := intrinsicDistance_symm S epsilon a
  dist_triangle := fun x y z => intrinsicDistance_triangle S epsilon a x y z he
  eq_of_dist_eq_zero := fun {x y} h =>
    (intrinsicDistance_eq_zero_iff S epsilon a x y he).1 h

@[simp] theorem intrinsicMetricSpace_dist [FiniteDimensional ℝ E]
    (S : Set E) (epsilon a : ℝ) (he : 0 < epsilon) (x y : E) :
    @dist E (intrinsicMetricSpace S epsilon a he).toDist x y =
      intrinsicDistance S epsilon a x y := rfl

/-- Shore-endpoint zero-floor divergence in the reverse orientation. -/
theorem intrinsicDistance_tendsto_zero_floor_of_mem_right [FiniteDimensional ℝ E]
    (S : Set E) (x y : E) (hy : y ∈ S) (hxy : x ≠ y) (a : ℝ) :
    Tendsto (fun epsilon => intrinsicDistance S epsilon a x y)
      (𝓝[>] (0 : ℝ)) atTop := by
  exact (intrinsicDistance_tendsto_zero_floor_of_mem_left S y x hy hxy.symm a).congr'
    (Eventually.of_forall fun epsilon => intrinsicDistance_symm S epsilon a y x)

/-- Distinct endpoints become infinitely far apart if either endpoint lies on Shore. -/
theorem intrinsicDistance_tendsto_zero_floor_of_mem_or_mem [FiniteDimensional ℝ E]
    (S : Set E) (x y : E) (hshore : x ∈ S ∨ y ∈ S) (hxy : x ≠ y) (a : ℝ) :
    Tendsto (fun epsilon => intrinsicDistance S epsilon a x y)
      (𝓝[>] (0 : ℝ)) atTop := by
  exact hshore.elim
    (fun hx => intrinsicDistance_tendsto_zero_floor_of_mem_left S x y hx hxy a)
    (fun hy => intrinsicDistance_tendsto_zero_floor_of_mem_right S x y hy hxy a)

end FractalSync.DistanceGraph
