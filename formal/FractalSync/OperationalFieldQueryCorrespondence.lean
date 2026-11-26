import FractalSync.GlobalOperationalField

/-!
# Physical queries, ideal grid indices, and finite-asset correspondence

The query uses mathematical integer floor after exact-real normalization. Its
36 indices lie in the finite asset under the actual full-halo inequalities.
Theorems do not model machine floor, casts, masks, or IEEE arithmetic.
-/

open Set

noncomputable section

namespace FractalSync.OperationalField

def physicalJet (n m : ℕ) (C : Lattice) (o : Coords) (h : ℝ) (q : Coords) : ℝ :=
  globalJet n m C (normalizedCoordinates o h q) / h^(n+m)

theorem hasFDerivAt_normalizedCoordinates (o q : Coords) (h : ℝ) :
    HasFDerivAt (normalizedCoordinates o h) (h⁻¹ • ContinuousLinearMap.id ℝ Coords) q := by
  convert ((hasFDerivAt_id (𝕜 := ℝ) q).sub_const o).const_smul (h⁻¹) using 1
  ext x <;> simp [normalizedCoordinates, div_eq_mul_inv, mul_comm]

/-- Every selected physical mixed jet has the actual normalized-coordinate derivative. -/
theorem hasFDerivAt_physicalJet (n m : ℕ) (hn : n < 4) (hm : m < 4)
    (C : Lattice) (o q : Coords) (h : ℝ) :
    HasFDerivAt (physicalJet n m C o h)
      (linearForm (physicalJet (n+1) m C o h q) (physicalJet n (m+1) C o h q)) q := by
  convert ((hasFDerivAt_globalJet n m hn hm C (normalizedCoordinates o h q)).comp q
    (hasFDerivAt_normalizedCoordinates o q h)).const_mul ((h^(n+m))⁻¹) using 1
  · exact funext (fun z => by simp [physicalJet, div_eq_mul_inv, mul_comm])
  · apply ContinuousLinearMap.ext
    intro v
    simp [linearForm, physicalJet, pow_add, div_eq_mul_inv,
      mul_assoc, mul_left_comm, mul_comm]


def physicalField (C : Lattice) (o : Coords) (h : ℝ) : Coords → ℝ := physicalJet 0 0 C o h

def physicalDifferential (C : Lattice) (o : Coords) (h : ℝ)
    (q : Coords) : Coords →L[ℝ] ℝ :=
  linearForm (physicalJet 1 0 C o h q) (physicalJet 0 1 C o h q)

def physicalHessian (C : Lattice) (o : Coords) (h : ℝ)
    (q : Coords) : Coords →L[ℝ] (Coords →L[ℝ] ℝ) :=
  quadraticForm (physicalJet 2 0 C o h q) (physicalJet 1 1 C o h q) (physicalJet 0 2 C o h q)

theorem hasFDerivAt_physicalField (C : Lattice) (o q : Coords) (h : ℝ) :
    HasFDerivAt (physicalField C o h) (physicalDifferential C o h q) q :=
  hasFDerivAt_physicalJet 0 0 (by norm_num) (by norm_num) C o q h

theorem hasFDerivAt_physicalDifferential (C : Lattice) (o q : Coords) (h : ℝ) :
    HasFDerivAt (physicalDifferential C o h) (physicalHessian C o h q) q :=
  hasFDerivAt_linearForm _ _ _ _ q
    (hasFDerivAt_physicalJet 1 0 (by norm_num) (by norm_num) C o q h)
    (hasFDerivAt_physicalJet 0 1 (by norm_num) (by norm_num) C o q h)

theorem fderiv_physicalField (C : Lattice) (o q : Coords) (h : ℝ) :
    fderiv ℝ (physicalField C o h) q = physicalDifferential C o h q :=
  (hasFDerivAt_physicalField C o q h).fderiv

theorem fderiv_fderiv_physicalField (C : Lattice) (o q : Coords) (h : ℝ) :
    fderiv ℝ (fderiv ℝ (physicalField C o h)) q = physicalHessian C o h q := by
  simpa only [show fderiv ℝ (physicalField C o h) = physicalDifferential C o h from
    funext (fun z => fderiv_physicalField C o z h)] using
      (hasFDerivAt_physicalDifferential C o q h).fderiv

theorem contDiff_normalizedCoordinates (o : Coords) (h : ℝ) :
    ContDiff ℝ 4 (normalizedCoordinates o h) :=
  ((contDiff_fst.sub (contDiff_const (c := o.1))).div_const h).prodMk
    ((contDiff_snd.sub (contDiff_const (c := o.2))).div_const h)

theorem contDiff_physicalField (C : Lattice) (o : Coords) (h : ℝ) :
    ContDiff ℝ 4 (physicalField C o h) := by
  convert (contDiff_globalField C).comp (contDiff_normalizedCoordinates o h) using 1
  exact funext (fun q => by simp [physicalField, physicalJet, globalField])


/-- Mathematical floor produces exactly the half-open fractional coordinate. -/
theorem floor_fraction_mem (x : ℝ) : x - (⌊x⌋ : ℝ) ∈ Ico (0 : ℝ) 1 :=
  ⟨Int.fract_nonneg x, Int.fract_lt_one x⟩

/-- The exact-real runtime halo inequalities validate every one of the six indices. -/
theorem halo_index_bounds (N : ℕ) (x : ℝ) (hx : 2 ≤ x) (hN : x < (N : ℝ)-3)
    (i : Fin 6) :
    0 ≤ ⌊x⌋ + i.val - 2 ∧ ⌊x⌋ + i.val - 2 < (N : ℤ) := by
  have hlo : (2 : ℤ) ≤ ⌊x⌋ := Int.le_floor.mpr hx
  have hhi : ⌊x⌋ < (N : ℤ)-3 := by
    exact_mod_cast (lt_of_le_of_lt (Int.floor_le x) hN)
  have hi := i.isLt
  omega


/-- Row-major integer addresses also stay within the declared finite asset. -/
theorem halo_flattened_index_bounds (Nx Ny : ℕ) (q : Coords)
    (hx : 2 ≤ q.1) (hNx : q.1 < (Nx : ℝ)-3)
    (hy : 2 ≤ q.2) (hNy : q.2 < (Ny : ℝ)-3) (i j : Fin 6) :
    0 ≤ (⌊q.2⌋ + j.val - 2) * Nx + (⌊q.1⌋ + i.val - 2) ∧
      (⌊q.2⌋ + j.val - 2) * Nx + (⌊q.1⌋ + i.val - 2) < (Nx : ℤ) * Ny := by
  have hi := halo_index_bounds Nx q.1 hx hNx i
  have hj := halo_index_bounds Ny q.2 hy hNy j
  constructor
  · exact add_nonneg (mul_nonneg hj.1 (Nat.cast_nonneg _)) hi.1
  · calc
      _ < (⌊q.2⌋ + j.val - 2 + 1) * Nx := by nlinarith [hi.2]
      _ ≤ (Ny : ℤ) * Nx := mul_le_mul_of_nonneg_right (by omega) (Nat.cast_nonneg _)
      _ = (Nx : ℤ) * Ny := mul_comm _ _

/-- Arbitrary extensions of the same finite asset give the same selected coefficient table. -/
theorem latticeCoefficients_eq_of_halo (C D : Lattice) (Nx Ny : ℕ) (q : Coords)
    (hx : 2 ≤ q.1) (hNx : q.1 < (Nx : ℝ)-3)
    (hy : 2 ≤ q.2) (hNy : q.2 < (Ny : ℝ)-3)
    (hCD : ∀ i j, 0 ≤ i → i < (Nx : ℤ) → 0 ≤ j → j < (Ny : ℤ) → C i j = D i j) :
    latticeCoefficients C ⌊q.1⌋ ⌊q.2⌋ = latticeCoefficients D ⌊q.1⌋ ⌊q.2⌋ := by
  funext i j
  exact hCD _ _ (halo_index_bounds Nx q.1 hx hNx i).1 (halo_index_bounds Nx q.1 hx hNx i).2
    (halo_index_bounds Ny q.2 hy hNy j).1 (halo_index_bounds Ny q.2 hy hNy j).2

theorem globalJet_eq_of_halo (n m : ℕ) (C D : Lattice) (Nx Ny : ℕ) (q : Coords)
    (hx : 2 ≤ q.1) (hNx : q.1 < (Nx : ℝ)-3)
    (hy : 2 ≤ q.2) (hNy : q.2 < (Ny : ℝ)-3)
    (hCD : ∀ i j, 0 ≤ i → i < (Nx : ℤ) → 0 ≤ j → j < (Ny : ℤ) → C i j = D i j) :
    globalJet n m C q = globalJet n m D q := by
  simp only [globalJet, latticePatchJet,
    latticeCoefficients_eq_of_halo C D Nx Ny q hx hNx hy hNy hCD]


def queryCellOrigin (o : Coords) (h : ℝ) (q : Coords) : Coords :=
  (o.1 + h * (⌊(normalizedCoordinates o h q).1⌋ : ℝ),
    o.2 + h * (⌊(normalizedCoordinates o h q).2⌋ : ℝ))

theorem normalized_offset (x o h k : ℝ) (hh : h ≠ 0) :
    (x-o)/h-k = (x-(o+h*k))/h := by
  field_simp
  ring

/-- The global evaluator is exactly the pre-existing physical query patch on its selected cell. -/
theorem physicalField_eq_query_patch (C : Lattice) (o q : Coords) (h : ℝ) (hh : h ≠ 0) :
    physicalField C o h q = field
      (latticeCoefficients C ⌊(normalizedCoordinates o h q).1⌋ ⌊(normalizedCoordinates o h q).2⌋)
      (queryCellOrigin o h q) h q := by
  simp [physicalField, physicalJet, globalJet, latticePatchJet, field, tensorPatch,
    queryCellOrigin, normalizedCoordinates, weights, normalized_offset _ _ _ _ hh]


theorem physicalDifferential_eq_query_patch (C : Lattice) (o q : Coords)
    (h : ℝ) (hh : h ≠ 0) :
    physicalDifferential C o h q = fieldDifferential
      (latticeCoefficients C ⌊(normalizedCoordinates o h q).1⌋ ⌊(normalizedCoordinates o h q).2⌋)
      (queryCellOrigin o h q) h q := by
  simp [physicalDifferential, physicalJet, globalJet, latticePatchJet,
    fieldDifferential_eq_query, tensorPatch, normalizedCoordinates, queryCellOrigin,
    normalized_offset _ _ _ _ hh]

theorem physicalHessian_eq_query_patch (C : Lattice) (o q : Coords)
    (h : ℝ) (hh : h ≠ 0) :
    physicalHessian C o h q = fieldHessian
      (latticeCoefficients C ⌊(normalizedCoordinates o h q).1⌋ ⌊(normalizedCoordinates o h q).2⌋)
      (queryCellOrigin o h q) h q := by
  simp [physicalHessian, physicalJet, globalJet, latticePatchJet,
    fieldHessian_eq_query, tensorPatch, normalizedCoordinates, queryCellOrigin,
    normalized_offset _ _ _ _ hh]

end FractalSync.OperationalField
