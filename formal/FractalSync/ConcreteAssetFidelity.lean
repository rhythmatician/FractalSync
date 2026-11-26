import FractalSync.ConcreteOperationalAsset
import FractalSync.OperationalFieldFidelity

/-!
# Reference fidelity of the literal finite quadratic fixture

The actual floor-selected field and its actual derivatives are compared with
the biased quadratic reproduced by ideal raw samples. The literal table's
finite sample certificate supplies every active coefficient premise throughout
the halo. These are exact-real bounds, without an IEEE evaluation claim.
-/

noncomputable section

namespace FractalSync.ConcreteAssetFidelity

open OperationalField OperationalHalo OperationalDynamics ConcreteOperationalAsset FlatShoreIntrinsic Set

def reference : Coords → ℝ := quadratic 1 2 (-3) 0 0 0

def sampleError : ℝ := 1 / 281474976710656

def reconstructedReference (q : Coords) : ℝ := reference q - spacing ^ 2

def selectedCoefficients (q : Coords) : Coefficients :=
  latticeCoefficients ConcreteOperationalAsset.coefficients
    ⌊(normalizedCoordinates origin spacing q).1⌋
    ⌊(normalizedCoordinates origin spacing q).2⌋

theorem finite_sample_error (i j : ℤ) (hi : 0 ≤ i) (hix : i < 16)
    (hj : 0 ≤ j) (hjy : j < 16) :
    |ConcreteOperationalAsset.coefficients i j -
      reference ((i : ℝ) * spacing, (j : ℝ) * spacing)| ≤ sampleError := by
  let fi : Fin 16 := ⟨i.toNat, by omega⟩
  let fj : Fin 16 := ⟨j.toNat, by omega⟩
  have hiR : (i.toNat : ℝ) = (i : ℝ) := by exact_mod_cast Int.toNat_of_nonneg hi
  have hjR : (j.toNat : ℝ) = (j : ℝ) := by exact_mod_cast Int.toNat_of_nonneg hj
  simpa [reference, quadratic, sampleError, ← coefficients_of_fin, fi, fj,
    hi, hj, hiR, hjR, sub_eq_add_neg] using coefficient_sample_error fi fj

/-- Every selected coefficient satisfies the reference-sample certificate,
including changes of the floor-selected query cell. -/
theorem selected_sample_error (q : Coords)
    (hq : q ∈ physicalHalo 16 16 origin spacing) (i j : Fin 6) :
    |selectedCoefficients q i j -
      sampledCoefficients reference (queryCellOrigin origin spacing q) spacing i j| ≤ sampleError := by
  have hi := halo_index_bounds 16 (normalizedCoordinates origin spacing q).1 hq.1.1 hq.1.2 i
  have hj := halo_index_bounds 16 (normalizedCoordinates origin spacing q).2 hq.2.1 hq.2.2 j
  convert finite_sample_error _ _ hi.1 hi.2 hj.1 hj.2 using 1
  simp only [selectedCoefficients, latticeCoefficients, sampledCoefficients, node,
    queryCellOrigin, origin, Int.cast_sub, Int.cast_add, Int.cast_natCast, Int.cast_ofNat]
  congr 4 <;> ring

/-- The actual reconstructed value stays within the literal sample error
of `P - h²` everywhere in the full half-open halo. -/
theorem physicalField_reference_error (q : Coords)
    (hq : q ∈ physicalHalo 16 16 origin spacing) :
    |physicalField ConcreteOperationalAsset.coefficients origin spacing q -
      reconstructedReference q| ≤ sampleError := by
  have hn := normalized_mem_of_mem_cell (queryCellOrigin origin spacing q) q spacing spacing_pos
    (mem_query_cell origin q spacing spacing_pos)
  have he := field_coefficient_error_bound (selectedCoefficients q)
    (sampledCoefficients reference (queryCellOrigin origin spacing q) spacing)
    (queryCellOrigin origin spacing q) q spacing sampleError hn.1 hn.2
    (selected_sample_error q hq)
  rw [reference, field_quadratic _ _ _ _ _ _ _ _ _ spacing_pos.ne'] at he
  norm_num [physicalField_eq_query_patch _ _ _ _ spacing_pos.ne', selectedCoefficients,
    reconstructedReference, reference, div_eq_mul_inv, mul_assoc, sub_eq_add_neg] at he ⊢
  exact he

/-- The actual derivative query has the coefficient-noise gain `8/(3h)`
on the full halo, without a fixed-cell restriction. -/
theorem physicalDifferential_reference_error (q : Coords)
    (hq : q ∈ physicalHalo 16 16 origin spacing) :
    ‖physicalDifferential ConcreteOperationalAsset.coefficients origin spacing q -
      fderiv ℝ reference q‖ ≤ (8 / 3 : ℝ) * sampleError / spacing := by
  have he := norm_fieldDifferential_coefficient_error (selectedCoefficients q)
    (sampledCoefficients reference (queryCellOrigin origin spacing q) spacing)
    (queryCellOrigin origin spacing q) q spacing sampleError spacing_pos
    (by norm_num [sampleError]) (mem_query_cell origin q spacing spacing_pos)
    (selected_sample_error q hq)
  simpa only [physicalDifferential_eq_query_patch _ _ _ _ spacing_pos.ne',
    selectedCoefficients, reference, fieldDifferential_quadratic _ _ _ _ _ _ _ _ _ spacing_pos.ne'] using he

/-- The actual Hessian query has coefficient-noise gain `68/(9h²)`
on the complete physical halo. -/
theorem physicalHessian_reference_error (q : Coords)
    (hq : q ∈ physicalHalo 16 16 origin spacing) :
    ‖physicalHessian ConcreteOperationalAsset.coefficients origin spacing q -
      fderiv ℝ (fderiv ℝ reference) q‖ ≤ (68 / 9 : ℝ) * sampleError / spacing ^ 2 := by
  have he := norm_fieldHessian_coefficient_error (selectedCoefficients q)
    (sampledCoefficients reference (queryCellOrigin origin spacing q) spacing)
    (queryCellOrigin origin spacing q) q spacing sampleError spacing_pos
    (by norm_num [sampleError]) (mem_query_cell origin q spacing spacing_pos)
    (selected_sample_error q hq)
  simpa only [physicalHessian_eq_query_patch _ _ _ _ spacing_pos.ne', selectedCoefficients,
    reference, fieldHessian_quadratic _ _ _ _ _ _ _ _ _ spacing_pos.ne'] using he

theorem fderiv_reconstructedReference :
    fderiv ℝ reconstructedReference = fderiv ℝ reference :=
  funext (fun _ => fderiv_sub_const (spacing ^ 2))

/-- The comparison is with the actual Fréchet differential of the actual
floor-selected field and of its biased quadratic reference. -/
theorem fderiv_reference_error (q : Coords)
    (hq : q ∈ physicalHalo 16 16 origin spacing) :
    ‖fderiv ℝ (physicalField ConcreteOperationalAsset.coefficients origin spacing) q -
      fderiv ℝ reconstructedReference q‖ ≤ (8 / 3 : ℝ) * sampleError / spacing := by
  simpa only [fderiv_reconstructedReference, fderiv_physicalField] using
    physicalDifferential_reference_error q hq

theorem fderiv_fderiv_reference_error (q : Coords)
    (hq : q ∈ physicalHalo 16 16 origin spacing) :
    ‖fderiv ℝ (fderiv ℝ (physicalField ConcreteOperationalAsset.coefficients origin spacing)) q -
      fderiv ℝ (fderiv ℝ reconstructedReference) q‖ ≤
        (68 / 9 : ℝ) * sampleError / spacing ^ 2 := by
  simpa only [fderiv_reconstructedReference, fderiv_fderiv_physicalField] using
    physicalHessian_reference_error q hq

/-- Convenient dyadic upper bounds for both exact derivative-noise gains. -/
theorem reference_gains :
    (8 / 3 : ℝ) * sampleError / spacing ≤ (1 / 8796093022208 : ℝ) ∧
    (68 / 9 : ℝ) * sampleError / spacing ^ 2 ≤ (1 / 274877906944 : ℝ) := by
  norm_num [sampleError, spacing]

/-- A concrete value/differential/Hessian fidelity certificate with no
unverified finite-asset or reference-sample hypotheses. -/
theorem actual_reference_errors (q : Coords)
    (hq : q ∈ physicalHalo 16 16 origin spacing) :
    |physicalField ConcreteOperationalAsset.coefficients origin spacing q -
      reconstructedReference q| ≤ (1 / 281474976710656 : ℝ) ∧
    ‖fderiv ℝ (physicalField ConcreteOperationalAsset.coefficients origin spacing) q -
      fderiv ℝ reconstructedReference q‖ ≤ (1 / 8796093022208 : ℝ) ∧
    ‖fderiv ℝ (fderiv ℝ (physicalField ConcreteOperationalAsset.coefficients origin spacing)) q -
      fderiv ℝ (fderiv ℝ reconstructedReference) q‖ ≤ (1 / 274877906944 : ℝ) :=
  ⟨physicalField_reference_error q hq,
    (fderiv_reference_error q hq).trans reference_gains.1,
    (fderiv_fderiv_reference_error q hq).trans reference_gains.2⟩

/-- The same differential accuracy controls the actual Euclidean gradient.
The chart pullback is contractive and Riesz dualization preserves the norm. -/
theorem euclidean_gradient_reference_error (x : Plane)
    (hx : chart x ∈ physicalHalo 16 16 origin spacing) :
    ‖gradient (physicalField ConcreteOperationalAsset.coefficients origin spacing ∘ chart) x -
      gradient (reconstructedReference ∘ chart) x‖ ≤ (1 / 8796093022208 : ℝ) := by
  rw [gradient, gradient, ← map_sub, (InnerProductSpace.toDual ℝ Plane).symm.norm_map,
    chart.comp_right_fderiv, chart.comp_right_fderiv, ← ContinuousLinearMap.sub_comp]
  exact (norm_pullDifferential_le _).trans (actual_reference_errors (chart x) hx).2.1

end FractalSync.ConcreteAssetFidelity
