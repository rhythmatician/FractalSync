import FractalSync.OperationalFieldFidelity

/-!
# Reference fidelity for the actual floor-selected global evaluator

The lattice is sampled at the physical origin and spacing. The selected
coefficient table is identified with the existing 36-node patch, so the
reference bounds do not require the query to stay in one preselected cell.
-/

open Set Filter Topology

noncomputable section

namespace FractalSync.OperationalField

/-- Exact reference samples on the physical integer lattice. -/
def sampledLattice (f : Coords → ℝ) (o : Coords) (h : ℝ) : Lattice :=
  fun i j => f (o.1 + h*(i : ℝ), o.2 + h*(j : ℝ))

theorem latticeCoefficients_sampledLattice (f : Coords → ℝ) (o q : Coords) (h : ℝ) :
    latticeCoefficients (sampledLattice f o h)
      ⌊(normalizedCoordinates o h q).1⌋ ⌊(normalizedCoordinates o h q).2⌋ =
      sampledCoefficients f (queryCellOrigin o h q) h := by
  funext i j
  simp only [latticeCoefficients, sampledLattice, sampledCoefficients, queryCellOrigin, node,
    Int.cast_sub, Int.cast_add, Int.cast_natCast, Int.cast_ofNat]
  congr 1
  ext <;> ring


theorem selected_reference_sample_error (C : Lattice) (f : Coords → ℝ) (o q : Coords)
    (h eta : ℝ) (hC : ∀ i j, |C i j - sampledLattice f o h i j| ≤ eta) (i j : Fin 6) :
    |latticeCoefficients C ⌊(normalizedCoordinates o h q).1⌋ ⌊(normalizedCoordinates o h q).2⌋ i j -
      sampledCoefficients f (queryCellOrigin o h q) h i j| ≤ eta := by
  rw [← latticeCoefficients_sampledLattice f o q h]
  exact hC _ _


/-- A certificate only for the 36 stored nodes selected at this query. -/
def SampleAccuracyAt (C : Lattice) (f : Coords → ℝ) (o : Coords)
    (h : ℝ) (q : Coords) (eta : ℝ) : Prop :=
  ∀ i j, |latticeCoefficients C ⌊(normalizedCoordinates o h q).1⌋
      ⌊(normalizedCoordinates o h q).2⌋ i j -
    sampledCoefficients f (queryCellOrigin o h q) h i j| ≤ eta

theorem sampleAccuracyAt_of_lattice (C : Lattice) (f : Coords → ℝ) (o q : Coords)
    (h eta : ℝ) (hC : ∀ i j, |C i j - sampledLattice f o h i j| ≤ eta) :
    SampleAccuracyAt C f o h q eta :=
  selected_reference_sample_error C f o q h eta hC

/-- Bias-corrected fidelity for the literal global floor-selected value. -/
theorem physicalField_reference_value_error {f : Coords → ℝ} {q : Coords}
    {S : Set Coords} {L : ℝ} (J : ProviderCertificates.LocalJetControl f q S L)
    (C : Lattice) (o : Coords) (h eta : ℝ) (hh : 0 < h) (hL : 0 ≤ L)
    (hnodes : ∀ i j, (node (queryCellOrigin o h q).1 h i, node (queryCellOrigin o h q).2 h j) ∈ S)
    (hC : SampleAccuracyAt C f o h q eta) :
    |physicalField C o h q - f q - h^2/4 * (J.hessian q (1,0) (1,0) + J.hessian q (0,1) (0,1))| ≤
      eta + (9/2 : ℝ)*L*h^3 := by
  rw [physicalField_eq_query_patch C o q h hh.ne']
  exact field_reference_value_error J _ _ h eta hh hL (mem_query_cell o q h hh) hnodes
    hC

/-- The physical selected gradient is compared to the actual reference differential. -/
theorem physicalField_reference_differential_error {f : Coords → ℝ} {q : Coords}
    {S : Set Coords} {L : ℝ} (J : ProviderCertificates.LocalJetControl f q S L)
    (C : Lattice) (o : Coords) (h eta : ℝ) (hh : 0 < h) (heta : 0 ≤ eta)
    (hL : 0 ≤ L) (hS : S ∈ 𝓝 q)
    (hnodes : ∀ i j, (node (queryCellOrigin o h q).1 h i, node (queryCellOrigin o h q).2 h j) ∈ S)
    (hC : SampleAccuracyAt C f o h q eta) :
    ‖physicalDifferential C o h q - J.differential q‖ ≤ (8/3 : ℝ)*eta/h + 12*L*h^2 := by
  rw [physicalDifferential_eq_query_patch C o q h hh.ne']
  exact field_reference_differential_error J _ _ h eta hh heta hL hS
    (mem_query_cell o q h hh) hnodes hC

/-- The physical selected Hessian is compared to the actual reference Hessian. -/
theorem physicalField_reference_hessian_error {f : Coords → ℝ} {q : Coords}
    {S : Set Coords} {L : ℝ} (J : ProviderCertificates.LocalJetControl f q S L)
    (C : Lattice) (o : Coords) (h eta : ℝ) (hh : 0 < h) (heta : 0 ≤ eta)
    (hL : 0 ≤ L) (hS : S ∈ 𝓝 q)
    (hnodes : ∀ i j, (node (queryCellOrigin o h q).1 h i, node (queryCellOrigin o h q).2 h j) ∈ S)
    (hC : SampleAccuracyAt C f o h q eta) :
    ‖physicalHessian C o h q - J.hessian q‖ ≤ (68/9 : ℝ)*eta/h^2 + 34*L*h := by
  rw [physicalHessian_eq_query_patch C o q h hh.ne']
  exact field_reference_hessian_error J _ _ h eta hh heta hL hS
    (mem_query_cell o q h hh) hnodes hC

/-- Uncorrected value fidelity, including the actual reference Hessian at the query. -/
theorem physicalField_reference_value_total_error {f : Coords → ℝ} {q : Coords}
    {S : Set Coords} {L : ℝ} (J : ProviderCertificates.LocalJetControl f q S L)
    (C : Lattice) (o : Coords) (h eta : ℝ) (hh : 0 < h) (hL : 0 ≤ L)
    (hnodes : ∀ i j, (node (queryCellOrigin o h q).1 h i, node (queryCellOrigin o h q).2 h j) ∈ S)
    (hC : SampleAccuracyAt C f o h q eta) :
    |physicalField C o h q - f q| ≤ eta + h^2/2 * ‖J.hessian q‖ + (9/2 : ℝ)*L*h^3 := by
  rw [physicalField_eq_query_patch C o q h hh.ne']
  exact field_reference_value_total_error J _ _ h eta hh hL
    (mem_query_cell o q h hh) hnodes hC


theorem reference_actual_hessian {f : Coords → ℝ} {q : Coords}
    {S : Set Coords} {L : ℝ} (J : ProviderCertificates.LocalJetControl f q S L)
    (hS : S ∈ 𝓝 q) : fderiv ℝ (fderiv ℝ f) q = J.hessian q := by
  have he : fderiv ℝ f =ᶠ[𝓝 q] J.differential :=
    Filter.mem_of_superset hS (fun z hz => (J.first z hz).fderiv)
  exact ((J.second q J.center_mem).congr_of_eventuallyEq he).fderiv


/-- Simultaneous accuracy of the global value and its literal first two derivatives. -/
theorem physicalField_reference_actual_jet_error {f : Coords → ℝ} {q : Coords}
    {S : Set Coords} {L : ℝ} (J : ProviderCertificates.LocalJetControl f q S L)
    (C : Lattice) (o : Coords) (h eta : ℝ) (hh : 0 < h) (heta : 0 ≤ eta)
    (hL : 0 ≤ L) (hS : S ∈ 𝓝 q)
    (hnodes : ∀ i j, (node (queryCellOrigin o h q).1 h i, node (queryCellOrigin o h q).2 h j) ∈ S)
    (hC : SampleAccuracyAt C f o h q eta) :
    |physicalField C o h q - f q| ≤
        eta + h^2/2 * ‖fderiv ℝ (fderiv ℝ f) q‖ + (9/2 : ℝ)*L*h^3 ∧
      ‖fderiv ℝ (physicalField C o h) q - fderiv ℝ f q‖ ≤ (8/3 : ℝ)*eta/h + 12*L*h^2 ∧
      ‖fderiv ℝ (fderiv ℝ (physicalField C o h)) q - fderiv ℝ (fderiv ℝ f) q‖ ≤
        (68/9 : ℝ)*eta/h^2 + 34*L*h := by
  simpa only [fderiv_physicalField, fderiv_fderiv_physicalField,
    (J.first q J.center_mem).fderiv, reference_actual_hessian J hS] using
    And.intro (physicalField_reference_value_total_error J C o h eta hh hL hnodes hC)
      ⟨physicalField_reference_differential_error J C o h eta hh heta hL hS hnodes hC,
        physicalField_reference_hessian_error J C o h eta hh heta hL hS hnodes hC⟩

end FractalSync.OperationalField
