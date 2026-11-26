import FractalSync.OperationalFieldReference
import FractalSync.OperationalFieldHessianError

/-!
# Combined reference, query, and reported-jet error

This contract keeps three independent sources of error explicit: stored sample
fidelity, the displacement of the actual real query, and the reported result's
arithmetic discrepancy from the mathematical evaluator at that query. No
floating-point operation is presumed to provide its discrepancy certificate.
-/

open Set Filter Topology

noncomputable section

namespace FractalSync.OperationalField

/-- A complete sufficient error contract for a reported value and both jets. -/
theorem reported_reference_jet_error {f : Coords → ℝ} {q : Coords}
    {S : Set Coords} {L : ℝ} (J : ProviderCertificates.LocalJetControl f q S L)
    (C : Lattice) (o q' : Coords) (h eta B : ℝ)
    (value : ℝ) (D : Coords →L[ℝ] ℝ) (H : Coords →L[ℝ] (Coords →L[ℝ] ℝ))
    (ev eD eH : ℝ) (hh : 0 < h) (heta : 0 ≤ eta) (hL : 0 ≤ L) (hB : 0 ≤ B)
    (hS : S ∈ 𝓝 q)
    (hnodes : ∀ i j, (node (queryCellOrigin o h q).1 h i, node (queryCellOrigin o h q).2 h j) ∈ S)
    (hsample : SampleAccuracyAt C f o h q eta) (hC : ∀ i j, |C i j| ≤ B)
    (hv : |value - physicalField C o h q'| ≤ ev)
    (hD : ‖D - physicalDifferential C o h q'‖ ≤ eD)
    (hH : ‖H - physicalHessian C o h q'‖ ≤ eH) :
    |value - f q| ≤ ev + (8/3 : ℝ)*B/h * ‖q'-q‖ +
        (eta + h^2/2 * ‖fderiv ℝ (fderiv ℝ f) q‖ + (9/2 : ℝ)*L*h^3) ∧
      ‖D - fderiv ℝ f q‖ ≤ eD + (68/9 : ℝ)*B/h^2 * ‖q'-q‖ +
        ((8/3 : ℝ)*eta/h + 12*L*h^2) ∧
      ‖H - fderiv ℝ (fderiv ℝ f) q‖ ≤ eH + 32*B/h^3 * ‖q'-q‖ +
        ((68/9 : ℝ)*eta/h^2 + 34*L*h) := by
  have href := physicalField_reference_actual_jet_error J C o h eta hh heta hL hS hnodes hsample
  simp only [fderiv_physicalField, fderiv_fderiv_physicalField] at href
  have htH : ‖H - fderiv ℝ (fderiv ℝ f) q‖ ≤
      ‖H-physicalHessian C o h q'‖ + ‖physicalHessian C o h q'-physicalHessian C o h q‖ +
      ‖physicalHessian C o h q-fderiv ℝ (fderiv ℝ f) q‖ := by
    simpa only [dist_eq_norm] using dist_triangle4 H (physicalHessian C o h q')
      (physicalHessian C o h q) (fderiv ℝ (fderiv ℝ f) q)
  have htD : ‖D - fderiv ℝ f q‖ ≤
      ‖D-physicalDifferential C o h q'‖ + ‖physicalDifferential C o h q'-physicalDifferential C o h q‖ +
      ‖physicalDifferential C o h q-fderiv ℝ f q‖ := by
    simpa only [dist_eq_norm] using dist_triangle4 D (physicalDifferential C o h q')
      (physicalDifferential C o h q) (fderiv ℝ f q)
  have htv : |value-f q| ≤ |value-physicalField C o h q'| +
      |physicalField C o h q'-physicalField C o h q| + |physicalField C o h q-f q| := by
    simpa only [Real.dist_eq] using dist_triangle4 value (physicalField C o h q')
      (physicalField C o h q) (f q)
  exact ⟨htv.trans (add_le_add (add_le_add hv
    (physicalField_query_error_bound C o q q' h B hh hB hC)) href.1),
    htD.trans (add_le_add (add_le_add hD
      (physicalDifferential_query_error_bound C o q q' h B hh hB hC)) href.2.1),
    htH.trans (add_le_add (add_le_add hH
      (physicalHessian_query_error_bound C o q q' h B hh hB hC)) href.2.2)⟩

end FractalSync.OperationalField
