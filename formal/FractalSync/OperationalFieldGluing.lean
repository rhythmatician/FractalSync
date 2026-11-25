import FractalSync.OperationalFieldQueryError

/-!
# Differentiating floor-selected polynomial cells

A continuous real cell coordinate can select adjacent differentiable pieces
with actual matching values and derivatives. The concrete operational-field
join identities are supplied separately, rather than postulated for that field.
-/

open Set Filter Topology

noncomputable section

namespace FractalSync.OperationalField

variable {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

def floorSelect (p : E → ℝ) (f : ℤ → E → F) (x : E) : F := f ⌊p x⌋ x

omit [NormedAddCommGroup E] [NormedSpace ℝ E] [NormedAddCommGroup F] [NormedSpace ℝ F] in
theorem floorSelect_eq_on_closed_cell (p : E → ℝ) (f : ℤ → E → F)
    (hj : ∀ (k : ℤ) x, p x = (k : ℝ) + 1 → f k x = f (k+1) x)
    (k : ℤ) (x : E) (hx : p x ∈ Icc (k : ℝ) ((k : ℝ)+1)) :
    floorSelect p f x = f k x := by
  by_cases he : p x = (k : ℝ) + 1
  · have hf : ⌊p x⌋ = k+1 := by
      rw [he, ← Int.cast_one, ← Int.cast_add, Int.floor_intCast]
    exact (show floorSelect p f x = f (k+1) x by simp [floorSelect, hf]).trans (hj k x he).symm
  · exact congrArg (fun n => f n x) (Int.floor_eq_iff.mpr ⟨hx.1, lt_of_le_of_ne hx.2 he⟩)

theorem hasFDerivWithinAt_floorSelect_cell (p : E → ℝ) (f : ℤ → E → F)
    (f' : ℤ → E → E →L[ℝ] F)
    (hd : ∀ k x, HasFDerivAt (f k) (f' k x) x)
    (hj : ∀ (k : ℤ) x, p x = (k : ℝ) + 1 → f k x = f (k+1) x)
    (k : ℤ) (x : E) (hx : p x ∈ Icc (k : ℝ) ((k : ℝ)+1)) :
    HasFDerivWithinAt (floorSelect p f) (f' k x)
      (p ⁻¹' Icc (k : ℝ) ((k : ℝ)+1)) x := by
  exact (hd k x).hasFDerivWithinAt.congr
    (fun y hy => floorSelect_eq_on_closed_cell p f hj k y hy)
    (floorSelect_eq_on_closed_cell p f hj k x hx)

/-- Actual differentiability across all integer cell boundaries. -/
theorem hasFDerivAt_floorSelect (p : E → ℝ) (hp : Continuous p)
    (f : ℤ → E → F) (f' : ℤ → E → E →L[ℝ] F)
    (hd : ∀ k x, HasFDerivAt (f k) (f' k x) x)
    (hj : ∀ (k : ℤ) x, p x = (k : ℝ) + 1 → f k x = f (k+1) x)
    (hj' : ∀ (k : ℤ) x, p x = (k : ℝ) + 1 → f' k x = f' (k+1) x)
    (x : E) : HasFDerivAt (floorSelect p f) (floorSelect p f' x) x := by
  set k : ℤ := ⌊p x⌋ with hk
  have hcell : p x ∈ Icc (k : ℝ) ((k : ℝ)+1) :=
    ⟨Int.floor_le _, (Int.lt_floor_add_one _).le⟩
  by_cases he : p x = (k : ℝ)
  · have hl : p x ∈ Icc ((k-1 : ℤ) : ℝ) (((k-1 : ℤ) : ℝ)+1) := by
      simp [he, Int.cast_sub]
    have hmatch : f' (k-1) x = f' k x := by
      simpa using hj' (k-1) x (by simpa [Int.cast_sub] using he)
    apply (((hasFDerivWithinAt_floorSelect_cell p f f' hd hj (k-1) x hl).congr_fderiv hmatch).union
      (hasFDerivWithinAt_floorSelect_cell p f f' hd hj k x hcell)).hasFDerivAt
    simp only [Int.cast_sub, Int.cast_one, sub_add_cancel, ← preimage_union,
      Icc_union_Icc_eq_Icc (sub_le_self _ zero_le_one) (le_add_of_nonneg_right zero_le_one)]
    exact hp.continuousAt.preimage_mem_nhds (Icc_mem_nhds (by simp [he]) (by simp [he]))
  · exact (hasFDerivWithinAt_floorSelect_cell p f f' hd hj k x hcell).hasFDerivAt
      (hp.continuousAt.preimage_mem_nhds
        (Icc_mem_nhds (lt_of_le_of_ne hcell.1 (Ne.symm he)) (Int.lt_floor_add_one _)))


omit [NormedSpace ℝ E] [NormedSpace ℝ F] in
theorem continuousWithinAt_floorSelect_cell (p : E → ℝ) (f : ℤ → E → F)
    (hc : ∀ k x, ContinuousAt (f k) x)
    (hj : ∀ (k : ℤ) x, p x = (k : ℝ) + 1 → f k x = f (k+1) x)
    (k : ℤ) (x : E) (hx : p x ∈ Set.Icc (k : ℝ) ((k : ℝ)+1)) :
    ContinuousWithinAt (floorSelect p f) (p ⁻¹' Set.Icc (k : ℝ) ((k : ℝ)+1)) x := by
  exact (hc k x).continuousWithinAt.congr
    (fun y hy => floorSelect_eq_on_closed_cell p f hj k y hy)
    (floorSelect_eq_on_closed_cell p f hj k x hx)

omit [NormedSpace ℝ E] [NormedSpace ℝ F] in
theorem continuousAt_floorSelect (p : E → ℝ) (hp : Continuous p)
    (f : ℤ → E → F) (hc : ∀ k x, ContinuousAt (f k) x)
    (hj : ∀ (k : ℤ) x, p x = (k : ℝ) + 1 → f k x = f (k+1) x)
    (x : E) : ContinuousAt (floorSelect p f) x := by
  set k : ℤ := ⌊p x⌋ with hk
  have hcell : p x ∈ Icc (k : ℝ) ((k : ℝ)+1) :=
    ⟨Int.floor_le _, (Int.lt_floor_add_one _).le⟩
  by_cases he : p x = (k : ℝ)
  · have hl : p x ∈ Icc ((k-1 : ℤ) : ℝ) (((k-1 : ℤ) : ℝ)+1) := by
      simp [he, Int.cast_sub]
    apply ((continuousWithinAt_floorSelect_cell p f hc hj (k-1) x hl).union
      (continuousWithinAt_floorSelect_cell p f hc hj k x hcell)).continuousAt
    simp only [Int.cast_sub, Int.cast_one, sub_add_cancel, ← preimage_union,
      Icc_union_Icc_eq_Icc (sub_le_self _ zero_le_one) (le_add_of_nonneg_right zero_le_one)]
    exact hp.continuousAt.preimage_mem_nhds (Icc_mem_nhds (by simp [he]) (by simp [he]))
  · exact (continuousWithinAt_floorSelect_cell p f hc hj k x hcell).continuousAt
      (hp.continuousAt.preimage_mem_nhds
        (Icc_mem_nhds (lt_of_le_of_ne hcell.1 (Ne.symm he)) (Int.lt_floor_add_one _)))

end FractalSync.OperationalField
