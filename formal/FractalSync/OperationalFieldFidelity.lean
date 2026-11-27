import FractalSync.OperationalFieldGlobalError

/-!
# Reference-field fidelity of the actual quintic sample operator

The input certificates concern the reference field's actual derivatives and
the values stored at the physical lattice nodes. The resulting value and jet
bounds apply to the concrete quintic evaluator. IEEE arithmetic remains separate.
-/

open Set Filter Topology

noncomputable section

namespace FractalSync.OperationalField

theorem linear_apply_coords (D : Coords →L[ℝ] ℝ) (z : Coords) :
    D z = z.1 * D (1, 0) + z.2 * D (0, 1) := by
  have hz : z = z.1 • (1, 0) + z.2 • (0, 1) := by ext <;> simp
  simpa only [map_add, map_smul, smul_eq_mul] using congrArg D hz


theorem bilinear_apply_coords (H : Coords →L[ℝ] (Coords →L[ℝ] ℝ)) (u v : Coords) :
    H u v = u.1 * (v.1 * H (1, 0) (1, 0) + v.2 * H (1, 0) (0, 1)) +
      u.2 * (v.1 * H (0, 1) (1, 0) + v.2 * H (0, 1) (0, 1)) := by
  calc
    _ = u.1 * H (1, 0) v + u.2 * H (0, 1) v := linear_apply_coords (H.flip v) u
    _ = _ := by rw [linear_apply_coords (H (1, 0)) v, linear_apply_coords (H (0, 1)) v]


/-- The actual second-order Taylor polynomial, centered at the physical query. -/
def taylorPolynomial (value : ℝ) (D : Coords →L[ℝ] ℝ)
    (H : Coords →L[ℝ] (Coords →L[ℝ] ℝ)) (q z : Coords) : ℝ :=
  value + D (z - q) + H (z - q) (z - q) / 2

theorem taylorPolynomial_eq_quadratic (value : ℝ) (D : Coords →L[ℝ] ℝ)
    (H : Coords →L[ℝ] (Coords →L[ℝ] ℝ)) (q z : Coords) :
    taylorPolynomial value D H q z =
      quadratic (H (1,0) (1,0) / 2) ((H (1,0) (0,1) + H (0,1) (1,0)) / 2)
        (H (0,1) (0,1) / 2)
        (D (1,0) - H (1,0) (1,0) * q.1 - (H (1,0) (0,1) + H (0,1) (1,0)) * q.2 / 2)
        (D (0,1) - H (0,1) (0,1) * q.2 - (H (1,0) (0,1) + H (0,1) (1,0)) * q.1 / 2)
        (value - D q + H q q / 2) z := by
  simp only [taylorPolynomial, quadratic]
  rw [linear_apply_coords D (z-q), linear_apply_coords D q,
    bilinear_apply_coords H (z-q) (z-q), bilinear_apply_coords H q q]
  simp only [Prod.fst_sub, Prod.snd_sub]
  ring


/-- The concrete raw-sample operator adds the exact quarter-Laplacian bias. -/
theorem field_taylorPolynomial (value : ℝ) (D : Coords →L[ℝ] ℝ)
    (H : Coords →L[ℝ] (Coords →L[ℝ] ℝ)) (c o z : Coords) (h : ℝ) (hh : h ≠ 0) :
    field (sampledCoefficients (taylorPolynomial value D H c) o h) o h z =
      taylorPolynomial value D H c z + h^2/4 * (H (1,0) (1,0) + H (0,1) (0,1)) := by
  have hfun := funext (taylorPolynomial_eq_quadratic value D H c)
  rw [hfun, field_quadratic _ _ _ _ _ _ o z h hh]
  ring


/-- The polynomial has the specified differential and symmetric Hessian. -/
theorem hasFDerivAt_taylorPolynomial (value : ℝ) (D : Coords →L[ℝ] ℝ)
    (H : Coords →L[ℝ] (Coords →L[ℝ] ℝ)) (c z : Coords)
    (hH : ∀ u v, H u v = H v u) :
    HasFDerivAt (taylorPolynomial value D H c) (D + H (z-c)) z := by
  convert ((D.hasFDerivAt.comp z ((hasFDerivAt_id z).sub_const c)).const_add value).add
    ((H.hasFDerivAt_of_bilinear ((hasFDerivAt_id z).sub_const c)
      ((hasFDerivAt_id z).sub_const c)).const_mul (1/2 : ℝ)) using 1
  · exact funext (fun w => by simp [taylorPolynomial, div_eq_mul_inv, mul_comm])
  · apply ContinuousLinearMap.ext
    intro w
    simp [hH w z, hH w c]
    ring


theorem fieldDifferential_taylorPolynomial (value : ℝ) (D : Coords →L[ℝ] ℝ)
    (H : Coords →L[ℝ] (Coords →L[ℝ] ℝ)) (c o z : Coords) (h : ℝ) (hh : h ≠ 0)
    (hH : ∀ u v, H u v = H v u) :
    fieldDifferential (sampledCoefficients (taylorPolynomial value D H c) o h) o h z =
      D + H (z-c) := by
  rw [← fderiv_field, show field (sampledCoefficients (taylorPolynomial value D H c) o h) o h =
    (fun w => taylorPolynomial value D H c w + h^2/4 * (H (1,0) (1,0) + H (0,1) (0,1))) from
      funext (fun w => field_taylorPolynomial value D H c o w h hh), fderiv_add_const]
  exact (hasFDerivAt_taylorPolynomial value D H c z hH).fderiv


theorem fieldHessian_taylorPolynomial (value : ℝ) (D : Coords →L[ℝ] ℝ)
    (H : Coords →L[ℝ] (Coords →L[ℝ] ℝ)) (c o z : Coords) (h : ℝ) (hh : h ≠ 0)
    (hH : ∀ u v, H u v = H v u) :
    fieldHessian (sampledCoefficients (taylorPolynomial value D H c) o h) o h z = H := by
  rw [← (hasFDerivAt_fieldDifferential _ o h z).fderiv,
    show fieldDifferential (sampledCoefficients (taylorPolynomial value D H c) o h) o h =
      (fun w => D + H (w-c)) from funext (fun w => fieldDifferential_taylorPolynomial value D H c o w h hh hH)]
  simpa using ((H.hasFDerivAt.comp z ((hasFDerivAt_id z).sub_const c)).const_add D).fderiv


theorem fieldDifferential_sub (C D : Coefficients) (o q : Coords) (h : ℝ) :
    fieldDifferential (fun i j => C i j - D i j) o h q =
      fieldDifferential C o h q - fieldDifferential D o h q := by
  rw [← fderiv_field, show field (fun i j => C i j - D i j) o h = field C o h - field D o h from
    funext (fun z => field_sub C D o z h)]
  exact ((hasFDerivAt_field C o h q).sub (hasFDerivAt_field D o h q)).fderiv


theorem fieldHessian_sub (C D : Coefficients) (o q : Coords) (h : ℝ) :
    fieldHessian (fun i j => C i j - D i j) o h q =
      fieldHessian C o h q - fieldHessian D o h q := by
  rw [← (hasFDerivAt_fieldDifferential _ o h q).fderiv,
    show fieldDifferential (fun i j => C i j - D i j) o h =
      fieldDifferential C o h - fieldDifferential D o h from
        funext (fun z => fieldDifferential_sub C D o z h)]
  exact ((hasFDerivAt_fieldDifferential C o h q).sub (hasFDerivAt_fieldDifferential D o h q)).fderiv


theorem norm_fieldDifferential_coefficient_error (C D : Coefficients) (o q : Coords)
    (h eta : ℝ) (hh : 0 < h) (heta : 0 ≤ eta) (hq : q ∈ cell o h)
    (hCD : ∀ i j, |C i j - D i j| ≤ eta) :
    ‖fieldDifferential C o h q - fieldDifferential D o h q‖ ≤ (8/3 : ℝ) * eta / h := by
  simpa only [fieldDifferential_sub] using
    norm_fieldDifferential_le (fun i j => C i j-D i j) o q h eta hh heta hq hCD

theorem norm_fieldHessian_coefficient_error (C D : Coefficients) (o q : Coords)
    (h eta : ℝ) (hh : 0 < h) (heta : 0 ≤ eta) (hq : q ∈ cell o h)
    (hCD : ∀ i j, |C i j - D i j| ≤ eta) :
    ‖fieldHessian C o h q - fieldHessian D o h q‖ ≤ (68/9 : ℝ) * eta / h^2 := by
  simpa only [fieldHessian_sub] using
    norm_fieldHessian_le (fun i j => C i j-D i j) o q h eta hh heta hq hCD


/-- Every active node is within three spacings of the query along its axis. -/
theorem abs_node_sub_le (o x h : ℝ) (hh : 0 ≤ h) (hx : x ∈ Icc o (o+h)) (i : Fin 6) :
    |node o h i - x| ≤ 3*h := by
  have hi : (i.val : ℝ) ≤ 5 := by exact_mod_cast Nat.le_of_lt_succ i.isLt
  exact abs_le.mpr ⟨by dsimp [node]; nlinarith [hx.2, mul_nonneg hh (Nat.cast_nonneg i.val)],
    by dsimp [node]; nlinarith [hx.1, mul_le_mul_of_nonneg_left hi hh]⟩


theorem norm_sample_node_sub_le (o q : Coords) (h : ℝ) (hh : 0 ≤ h)
    (hq : q ∈ cell o h) (i j : Fin 6) :
    ‖(node o.1 h i, node o.2 h j) - q‖ ≤ 3*h := by
  exact max_le (abs_node_sub_le o.1 q.1 h hh hq.1 i)
    (abs_node_sub_le o.2 q.2 h hh hq.2 j)


theorem reference_taylor_remainder {f : Coords → ℝ} {q : Coords} {S : Set Coords} {L : ℝ}
    (J : ProviderCertificates.LocalJetControl f q S L) (z : Coords) (hz : z ∈ S) :
    |f z - taylorPolynomial (f q) (J.differential q) (J.hessian q) q z| ≤ L/6 * ‖z-q‖^3 := by
  simpa [taylorPolynomial, sub_add_eq_sub_sub] using
    J.remainder (z-q) (by simpa using hz)


/-- Actual Hessian variation bounds every one of the 36 sample remainders. -/
theorem reference_sample_remainder {f : Coords → ℝ} {q : Coords} {S : Set Coords} {L : ℝ}
    (J : ProviderCertificates.LocalJetControl f q S L) (o : Coords) (h : ℝ)
    (hh : 0 ≤ h) (hL : 0 ≤ L) (hq : q ∈ cell o h)
    (hnodes : ∀ i j, (node o.1 h i, node o.2 h j) ∈ S) (i j : Fin 6) :
    |sampledCoefficients f o h i j -
      sampledCoefficients (taylorPolynomial (f q) (J.differential q) (J.hessian q) q) o h i j| ≤
      (9/2 : ℝ) * L * h^3 := by
  calc
    _ ≤ L/6 * ‖(node o.1 h i, node o.2 h j)-q‖^3 :=
      reference_taylor_remainder J _ (hnodes i j)
    _ ≤ L/6 * (3*h)^3 := mul_le_mul_of_nonneg_left
      (pow_le_pow_left₀ (norm_nonneg _) (norm_sample_node_sub_le o q h hh hq i j) 3)
      (div_nonneg hL (by norm_num))
    _ = _ := by ring


theorem coefficient_taylor_error_bound {f : Coords → ℝ} {q : Coords} {S : Set Coords} {L : ℝ}
    (J : ProviderCertificates.LocalJetControl f q S L) (C : Coefficients) (o : Coords)
    (h eta : ℝ) (hh : 0 ≤ h) (hL : 0 ≤ L) (hq : q ∈ cell o h)
    (hnodes : ∀ i j, (node o.1 h i, node o.2 h j) ∈ S)
    (hC : ∀ i j, |C i j - sampledCoefficients f o h i j| ≤ eta) (i j : Fin 6) :
    |C i j - sampledCoefficients
      (taylorPolynomial (f q) (J.differential q) (J.hessian q) q) o h i j| ≤
      eta + (9/2 : ℝ) * L * h^3 := by
  exact (abs_sub_le (C i j) (sampledCoefficients f o h i j) _).trans
    (add_le_add (hC i j) (reference_sample_remainder J o h hh hL hq hnodes i j))

/-- Value accuracy includes the intrinsic raw-sample Laplacian bias. -/
theorem field_reference_value_error {f : Coords → ℝ} {q : Coords} {S : Set Coords} {L : ℝ}
    (J : ProviderCertificates.LocalJetControl f q S L) (C : Coefficients) (o : Coords)
    (h eta : ℝ) (hh : 0 < h) (hL : 0 ≤ L) (hq : q ∈ cell o h)
    (hnodes : ∀ i j, (node o.1 h i, node o.2 h j) ∈ S)
    (hC : ∀ i j, |C i j - sampledCoefficients f o h i j| ≤ eta) :
    |field C o h q - f q - h^2/4 * (J.hessian q (1,0) (1,0) + J.hessian q (0,1) (0,1))| ≤
      eta + (9/2 : ℝ) * L * h^3 := by
  have hq' := normalized_mem_of_mem_cell o q h hh hq
  have he := field_coefficient_error_bound C
    (sampledCoefficients (taylorPolynomial (f q) (J.differential q) (J.hessian q) q) o h)
    o q h (eta + (9/2 : ℝ)*L*h^3) hq'.1 hq'.2
    (coefficient_taylor_error_bound J C o h eta hh.le hL hq hnodes hC)
  simpa [field_taylorPolynomial _ _ _ _ _ _ _ hh.ne', taylorPolynomial, sub_add_eq_sub_sub] using he


/-- Actual differential accuracy for noisy raw samples of a controlled reference field. -/
theorem field_reference_differential_error {f : Coords → ℝ} {q : Coords} {S : Set Coords} {L : ℝ}
    (J : ProviderCertificates.LocalJetControl f q S L) (C : Coefficients) (o : Coords)
    (h eta : ℝ) (hh : 0 < h) (heta : 0 ≤ eta) (hL : 0 ≤ L) (hS : S ∈ 𝓝 q)
    (hq : q ∈ cell o h) (hnodes : ∀ i j, (node o.1 h i, node o.2 h j) ∈ S)
    (hC : ∀ i j, |C i j - sampledCoefficients f o h i j| ≤ eta) :
    ‖fieldDifferential C o h q - J.differential q‖ ≤ (8/3 : ℝ) * eta / h + 12*L*h^2 := by
  have he := norm_fieldDifferential_coefficient_error C
    (sampledCoefficients (taylorPolynomial (f q) (J.differential q) (J.hessian q) q) o h)
    o q h (eta + (9/2 : ℝ)*L*h^3) hh (by positivity) hq
    (coefficient_taylor_error_bound J C o h eta hh.le hL hq hnodes hC)
  rw [fieldDifferential_taylorPolynomial _ _ _ _ _ _ _ hh.ne' (J.hessian_symm hS)] at he
  have heq : (8/3 : ℝ) * (eta+(9/2 : ℝ)*L*h^3)/h = (8/3 : ℝ)*eta/h + 12*L*h^2 := by
    field_simp
    ring
  simpa only [sub_self, map_zero, add_zero, heq] using he


/-- Actual Hessian accuracy follows with an explicit first-order spacing term. -/
theorem field_reference_hessian_error {f : Coords → ℝ} {q : Coords} {S : Set Coords} {L : ℝ}
    (J : ProviderCertificates.LocalJetControl f q S L) (C : Coefficients) (o : Coords)
    (h eta : ℝ) (hh : 0 < h) (heta : 0 ≤ eta) (hL : 0 ≤ L) (hS : S ∈ 𝓝 q)
    (hq : q ∈ cell o h) (hnodes : ∀ i j, (node o.1 h i, node o.2 h j) ∈ S)
    (hC : ∀ i j, |C i j - sampledCoefficients f o h i j| ≤ eta) :
    ‖fieldHessian C o h q - J.hessian q‖ ≤ (68/9 : ℝ) * eta / h^2 + 34*L*h := by
  have he := norm_fieldHessian_coefficient_error C
    (sampledCoefficients (taylorPolynomial (f q) (J.differential q) (J.hessian q) q) o h)
    o q h (eta + (9/2 : ℝ)*L*h^3) hh (by positivity) hq
    (coefficient_taylor_error_bound J C o h eta hh.le hL hq hnodes hC)
  rw [fieldHessian_taylorPolynomial _ _ _ _ _ _ _ hh.ne' (J.hessian_symm hS)] at he
  convert he using 1
  field_simp
  ring


theorem abs_coordinate_trace_le (H : Coords →L[ℝ] (Coords →L[ℝ] ℝ)) :
    |H (1,0) (1,0) + H (0,1) (0,1)| ≤ 2 * ‖H‖ := by
  have hx : |H (1,0) (1,0)| ≤ ‖H‖ := by simpa using H.le_opNorm₂ (1,0) (1,0)
  have hy : |H (0,1) (0,1)| ≤ ‖H‖ := by simpa using H.le_opNorm₂ (0,1) (0,1)
  exact (abs_add_le _ _).trans ((add_le_add hx hy).trans_eq (two_mul ‖H‖).symm)


/-- An uncorrected value bound that can be used without cancellation of the bias. -/
theorem field_reference_value_total_error {f : Coords → ℝ} {q : Coords} {S : Set Coords} {L : ℝ}
    (J : ProviderCertificates.LocalJetControl f q S L) (C : Coefficients) (o : Coords)
    (h eta : ℝ) (hh : 0 < h) (hL : 0 ≤ L) (hq : q ∈ cell o h)
    (hnodes : ∀ i j, (node o.1 h i, node o.2 h j) ∈ S)
    (hC : ∀ i j, |C i j - sampledCoefficients f o h i j| ≤ eta) :
    |field C o h q - f q| ≤ eta + h^2/2 * ‖J.hessian q‖ + (9/2 : ℝ)*L*h^3 := by
  have he := field_reference_value_error J C o h eta hh hL hq hnodes hC
  have hb : |h^2/4 * (J.hessian q (1,0) (1,0) + J.hessian q (0,1) (0,1))| ≤
      h^2/2 * ‖J.hessian q‖ := by
    rw [abs_mul, abs_of_nonneg (by positivity : 0 ≤ h^2/4)]
    convert mul_le_mul_of_nonneg_left (abs_coordinate_trace_le (J.hessian q))
      (show 0 ≤ h^2/4 by positivity) using 1
    ring
  have ht := abs_sub_le (field C o h q - f q)
    (h^2/4 * (J.hessian q (1,0) (1,0) + J.hessian q (0,1) (0,1))) 0
  simp only [sub_zero] at ht
  linarith

end FractalSync.OperationalField
