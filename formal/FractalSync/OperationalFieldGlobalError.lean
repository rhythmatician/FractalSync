import FractalSync.OperationalFieldQueryCorrespondence

/-!
# Global query-error bounds for the actual floor-selected operational field

Uniform lattice coefficient bounds supply actual global derivative bounds.
Queries may cross arbitrarily many grid lines; no same-cell hypothesis remains.
Arithmetic and query-rounding certificates remain separate.
-/

open Set

noncomputable section

namespace FractalSync.OperationalField

theorem mem_query_cell (o q : Coords) (h : ℝ) (hh : 0 < h) :
    q ∈ cell (queryCellOrigin o h q) h := by
  have hxlo := (le_div_iff₀ hh).mp (Int.floor_le ((q.1-o.1)/h))
  have hxhi := (div_lt_iff₀ hh).mp (Int.lt_floor_add_one ((q.1-o.1)/h))
  have hylo := (le_div_iff₀ hh).mp (Int.floor_le ((q.2-o.2)/h))
  have hyhi := (div_lt_iff₀ hh).mp (Int.lt_floor_add_one ((q.2-o.2)/h))
  change (o.1 + h * (⌊(q.1-o.1)/h⌋ : ℝ) ≤ q.1 ∧
      q.1 ≤ o.1 + h * (⌊(q.1-o.1)/h⌋ : ℝ) + h) ∧
    (o.2 + h * (⌊(q.2-o.2)/h⌋ : ℝ) ≤ q.2 ∧
      q.2 ≤ o.2 + h * (⌊(q.2-o.2)/h⌋ : ℝ) + h)
  constructor <;> constructor <;> nlinarith

theorem norm_physicalDifferential_le (C : Lattice) (o q : Coords) (h B : ℝ)
    (hh : 0 < h) (hB : 0 ≤ B) (hC : ∀ i j, |C i j| ≤ B) :
    ‖physicalDifferential C o h q‖ ≤ (8/3 : ℝ) * B / h := by
  rw [physicalDifferential_eq_query_patch C o q h hh.ne']
  exact norm_fieldDifferential_le _ _ q h B hh hB (mem_query_cell o q h hh)
    (fun i j => hC _ _)

theorem norm_physicalHessian_le (C : Lattice) (o q : Coords) (h B : ℝ)
    (hh : 0 < h) (hB : 0 ≤ B) (hC : ∀ i j, |C i j| ≤ B) :
    ‖physicalHessian C o h q‖ ≤ (68/9 : ℝ) * B / h^2 := by
  rw [physicalHessian_eq_query_patch C o q h hh.ne']
  exact norm_fieldHessian_le _ _ q h B hh hB (mem_query_cell o q h hh)
    (fun i j => hC _ _)


/-- Global value stability for real queries, without a same-cell restriction. -/
theorem physicalField_query_error_bound (C : Lattice) (o q q' : Coords) (h B : ℝ)
    (hh : 0 < h) (hB : 0 ≤ B) (hC : ∀ i j, |C i j| ≤ B) :
    |physicalField C o h q' - physicalField C o h q| ≤
      (8/3 : ℝ) * B / h * ‖q'-q‖ := by
  simpa only [Real.norm_eq_abs] using
    Convex.norm_image_sub_le_of_norm_hasFDerivWithin_le
      (fun z (_ : z ∈ (univ : Set Coords)) => (hasFDerivAt_physicalField C o z h).hasFDerivWithinAt)
      (fun z _ => norm_physicalDifferential_le C o z h B hh hB hC)
      convex_univ (mem_univ q) (mem_univ q')

/-- Global derivative-map stability through all grid boundaries. -/
theorem physicalDifferential_query_error_bound (C : Lattice) (o q q' : Coords) (h B : ℝ)
    (hh : 0 < h) (hB : 0 ≤ B) (hC : ∀ i j, |C i j| ≤ B) :
    ‖physicalDifferential C o h q' - physicalDifferential C o h q‖ ≤
      (68/9 : ℝ) * B / h^2 * ‖q'-q‖ := by
  exact Convex.norm_image_sub_le_of_norm_hasFDerivWithin_le
    (fun z (_ : z ∈ (univ : Set Coords)) => (hasFDerivAt_physicalDifferential C o z h).hasFDerivWithinAt)
    (fun z _ => norm_physicalHessian_le C o z h B hh hB hC)
    convex_univ (mem_univ q) (mem_univ q')

/-- Global coefficient perturbation still has unit gain at every real query. -/
theorem physicalField_coefficient_error_bound (C D : Lattice) (o q : Coords)
    (h eta : ℝ) (hh : 0 < h) (hCD : ∀ i j, |C i j - D i j| ≤ eta) :
    |physicalField C o h q - physicalField D o h q| ≤ eta := by
  have hq := normalized_mem_of_mem_cell (queryCellOrigin o h q) q h hh (mem_query_cell o q h hh)
  rw [physicalField_eq_query_patch C o q h hh.ne', physicalField_eq_query_patch D o q h hh.ne']
  exact field_coefficient_error_bound _ _ _ q h eta hq.1 hq.2 (fun i j => hCD _ _)

/-- Combined real coefficient and query perturbations can cross cell boundaries. -/
theorem physicalField_combined_error_bound (C D : Lattice) (o q q' : Coords)
    (h eta B : ℝ) (hh : 0 < h) (hB : 0 ≤ B)
    (hCD : ∀ i j, |C i j - D i j| ≤ eta) (hD : ∀ i j, |D i j| ≤ B) :
    |physicalField C o h q' - physicalField D o h q| ≤
      eta + (8/3 : ℝ) * B / h * ‖q'-q‖ := by
  calc
    _ ≤ |physicalField C o h q' - physicalField D o h q'| +
        |physicalField D o h q' - physicalField D o h q| := abs_sub_le _ _ _
    _ ≤ eta + (8/3 : ℝ) * B / h * ‖q'-q‖ := add_le_add
      (physicalField_coefficient_error_bound C D o q' h eta hh hCD)
      (physicalField_query_error_bound D o q q' h B hh hB hD)

end FractalSync.OperationalField
