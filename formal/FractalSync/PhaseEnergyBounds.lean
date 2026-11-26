import FractalSync.ControlForceBounds
import FractalSync.PhaseEnergyRegularity

/-!
# Explicit regional bounds for the actual phase-energy differential

Arithmetic norm/Lipschitz budgets for field jets determine a concrete bound
for the actual energy differential. This refines the earlier existence of a
compact-region regularity constant. Exact-real field-jet premises remain
explicit; no numerical provider or global continuation claim is made.
-/

noncomputable section

namespace FractalSync.ScaleMetric.PhaseEnergyBounds

open Set MechanicalBounds ControlMechanics
open scoped RealInnerProductSpace

/-- Explicit position-differential norm and Lipschitz budget. -/
def positionCovectorBudget (epsilon a : ℝ) (br bu bH bv bU : Budget) : Budget :=
  let ri := inverseBudget epsilon br
  let ri2 := productBudget ri ri
  let uv := productBudget bu bv
  let first := productBudget (productBudget (productBudget (a ^ 2, 0) uv) ri2)
    (productBudget bv bH)
  let numerator := addBudget (productBudget bv bv)
    (productBudget (productBudget (a ^ 2, 0) uv) uv)
  let second := productBudget (productBudget numerator (productBudget ri2 ri)) bu
  addBudget (addBudget first second) bU

/-- Explicit norm and Lipschitz budget for the complete energy differential. -/
def energyCovectorBudget (epsilon a : ℝ) (br bu bH bv bU : Budget) : Budget :=
  addBudget (positionCovectorBudget epsilon a br bu bH bv bU)
    (metricActionBudget epsilon a br bu bv)

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

noncomputable local instance : NormedSpace ℝ (E × E) := Prod.normedSpace

/-- Position part of the actual phase-energy differential. -/
def positionCovector (rho a : ℝ) (u : E) (H : E →L[ℝ] E) (v gradU : E) : E →L[ℝ] ℝ :=
  (a ^ 2 * inner ℝ u v / rho ^ 2) • ((innerSL ℝ v).comp H) -
    ((inner ℝ v v + a ^ 2 * inner ℝ u v * inner ℝ u v) / rho ^ 3) • innerSL ℝ u +
    innerSL ℝ gradU

/-- Bundled differential on the actual position-velocity product. -/
def energyCovector (rho a : ℝ) (u : E) (H : E →L[ℝ] E) (v gradU : E) : (E × E) →L[ℝ] ℝ :=
  (positionCovector rho a u H v gradU).comp (ContinuousLinearMap.fst ℝ E E) +
    (innerSL ℝ (metricAction rho a u v)).comp (ContinuousLinearMap.snd ℝ E E)

/-- Exact identification, assuming actual ruler/gradient/potential jets. -/
theorem fderiv_phaseEnergy_eq (a : ℝ) (rho potential : E → ℝ)
    (gradient : E → E) (x v gradU : E) (H : E →L[ℝ] E)
    (hr : HasFDerivAt rho (innerSL ℝ (gradient x)) x)
    (hu : HasFDerivAt gradient H x)
    (hU : HasFDerivAt potential (innerSL ℝ gradU) x) (hrho : rho x ≠ 0) :
    fderiv ℝ (phaseEnergy a rho potential gradient) (x, v) =
      energyCovector (rho x) a (gradient x) H v gradU := by
  apply ContinuousLinearMap.ext
  rintro ⟨d, w⟩
  rw [fderiv_phaseEnergy_apply a rho potential gradient x v d w gradU H hr hu hU hrho]
  simp only [energyCovector, positionCovector, ContinuousLinearMap.add_apply,
    ContinuousLinearMap.sub_apply, ContinuousLinearMap.smul_apply,
    ContinuousLinearMap.comp_apply, ContinuousLinearMap.coe_fst',
    ContinuousLinearMap.coe_snd', innerSL_apply, smul_eq_mul, metricAction, stretch,
    real_inner_smul_left, inner_add_left, metricDifferential, metric,
    real_inner_comm v (H d)]
  field_simp [hrho]
  ring

variable {X : Type*} [PseudoMetricSpace X]

/-- The Riesz covector preserves the supplied vector-field budget. -/
theorem innerSL_bound {f : X → E} {s : Set X} {M L : ℝ}
    (hf : BoundedLipschitzOn f s M L) :
    BoundedLipschitzOn (fun z => innerSL ℝ (f z)) s M L := by
  exact ⟨hf.bound_nonneg, hf.lip_nonneg,
    fun z hz => by simpa only [innerSL_apply_norm] using hf.norm_le z hz,
    fun z hz w hw => by
      simpa only [← map_sub, innerSL_apply_norm] using hf.norm_sub_le z hz w hw⟩

/-- Variable covector/operator composition is a norm-one bilinear operation. -/
theorem covector_comp_bound {f : X → E →L[ℝ] ℝ} {g : X → E →L[ℝ] E}
    {s : Set X} {M L N K : ℝ} (hf : BoundedLipschitzOn f s M L)
    (hg : BoundedLipschitzOn g s N K) :
    BoundedLipschitzOn (fun z => (f z).comp (g z)) s (M * N) (M * K + N * L) := by
  exact hf.bilinear hg (ContinuousLinearMap.compL ℝ E E ℝ)
    (fun f g => f.opNorm_comp_le g)

/-- Projection or any fixed contraction does not increase a covector budget. -/
theorem phase_precomp_bound {f : X → E →L[ℝ] ℝ} {s : Set X} {M L : ℝ}
    (hf : BoundedLipschitzOn f s M L) (P : (E × E) →L[ℝ] E) (hP : ‖P‖ ≤ 1) :
    BoundedLipschitzOn (fun z => (f z).comp P) s M L := by
  have hconst := (BoundedLipschitzOn.const s P).enlarge hP le_rfl
  simpa only [mul_one, mul_zero, one_mul, zero_add] using
    hf.bilinear hconst (ContinuousLinearMap.compL ℝ (E × E) E ℝ)
      (fun f g => f.opNorm_comp_le g)

theorem positionCovector_bound (epsilon a : ℝ) (s : Set X)
    (rho : X → ℝ) (u : X → E) (H : X → E →L[ℝ] E) (v gradU : X → E)
    (br bu bH bv bU : Budget) (he : 0 < epsilon)
    (hr : BoundedLipschitzOn rho s br.1 br.2)
    (hu : BoundedLipschitzOn u s bu.1 bu.2)
    (hH : BoundedLipschitzOn H s bH.1 bH.2)
    (hv : BoundedLipschitzOn v s bv.1 bv.2)
    (hU : BoundedLipschitzOn gradU s bU.1 bU.2)
    (hfloor : ∀ z ∈ s, epsilon ≤ rho z) :
    BoundedLipschitzOn (fun z => positionCovector (rho z) a (u z) (H z) (v z) (gradU z)) s
      (positionCovectorBudget epsilon a br bu bH bv bU).1
      (positionCovectorBudget epsilon a br bu bH bv bU).2 := by
  have hA : BoundedLipschitzOn (fun _ : X => a ^ 2) s (a ^ 2) 0 :=
    ⟨sq_nonneg a, le_rfl, fun _ _ => (Real.norm_of_nonneg (sq_nonneg a)).le,
      fun _ _ _ _ => by simp⟩
  have hri := hr.inv he hfloor
  have hFirst := ((hA.mul (hu.inner hv)).mul (hri.mul hri)).smul
    (covector_comp_bound (innerSL_bound hv) hH)
  have hNum := (hv.inner hv).add ((hA.mul (hu.inner hv)).mul (hu.inner hv))
  have hSecond := (hNum.mul ((hri.mul hri).mul hri)).smul (innerSL_bound hu)
  simpa only [positionCovectorBudget, addBudget, productBudget, inverseBudget,
    positionCovector, div_eq_mul_inv, pow_succ, pow_zero, mul_inv_rev, inv_one,
    one_mul, mul_one, mul_assoc, mul_left_comm, mul_comm] using
    (hFirst.sub hSecond).add (innerSL_bound hU)

/-- Uniform operator-norm and difference bounds for the bundled differential. -/
theorem energyCovector_bound (epsilon a : ℝ) (s : Set X)
    (rho : X → ℝ) (u : X → E) (H : X → E →L[ℝ] E) (v gradU : X → E)
    (br bu bH bv bU : Budget) (he : 0 < epsilon)
    (hr : BoundedLipschitzOn rho s br.1 br.2)
    (hu : BoundedLipschitzOn u s bu.1 bu.2)
    (hH : BoundedLipschitzOn H s bH.1 bH.2)
    (hv : BoundedLipschitzOn v s bv.1 bv.2)
    (hU : BoundedLipschitzOn gradU s bU.1 bU.2)
    (hfloor : ∀ z ∈ s, epsilon ≤ rho z) :
    BoundedLipschitzOn (fun z => energyCovector (rho z) a (u z) (H z) (v z) (gradU z)) s
      (energyCovectorBudget epsilon a br bu bH bv bU).1
      (energyCovectorBudget epsilon a br bu bH bv bU).2 := by
  simpa only [energyCovectorBudget, addBudget, energyCovector] using
    (phase_precomp_bound
      (positionCovector_bound epsilon a s rho u H v gradU br bu bH bv bU he hr hu hH hv hU hfloor)
      (ContinuousLinearMap.fst ℝ E E) (ContinuousLinearMap.opNorm_le_bound _ zero_le_one
        (fun z => by simpa only [one_mul] using norm_fst_le z))).add
      (phase_precomp_bound (innerSL_bound
        (metricAction_boundedLipschitz epsilon a s rho u v br bu bv he hr hu hv hfloor))
        (ContinuousLinearMap.snd ℝ E E) (ContinuousLinearMap.opNorm_le_bound _ zero_le_one
          (fun z => by simpa only [one_mul] using norm_snd_le z)))

/-- Actual field-jet identities transfer the explicit coefficient budget to
`fderiv` of the actual phase energy on the whole supplied region. -/
theorem fderiv_phaseEnergy_bound (epsilon a : ℝ) (K : Set (E × E))
    (rho potential : E → ℝ) (gradient gradU : E → E) (H : E → E →L[ℝ] E)
    (br bu bH bv bU : Budget) (he : 0 < epsilon)
    (hr : BoundedLipschitzOn (fun z : E × E => rho z.1) K br.1 br.2)
    (hu : BoundedLipschitzOn (fun z : E × E => gradient z.1) K bu.1 bu.2)
    (hH : BoundedLipschitzOn (fun z : E × E => H z.1) K bH.1 bH.2)
    (hv : BoundedLipschitzOn (Prod.snd : E × E → E) K bv.1 bv.2)
    (hU : BoundedLipschitzOn (fun z : E × E => gradU z.1) K bU.1 bU.2)
    (hfloor : ∀ z ∈ K, epsilon ≤ rho z.1)
    (hr' : ∀ z ∈ K, HasFDerivAt rho (innerSL ℝ (gradient z.1)) z.1)
    (hu' : ∀ z ∈ K, HasFDerivAt gradient (H z.1) z.1)
    (hU' : ∀ z ∈ K, HasFDerivAt potential (innerSL ℝ (gradU z.1)) z.1) :
    BoundedLipschitzOn (fderiv ℝ (phaseEnergy a rho potential gradient)) K
      (energyCovectorBudget epsilon a br bu bH bv bU).1
      (energyCovectorBudget epsilon a br bu bH bv bU).2 := by
  have h := energyCovector_bound epsilon a K (fun z => rho z.1)
    (fun z => gradient z.1) (fun z => H z.1) Prod.snd (fun z => gradU z.1)
    br bu bH bv bU he hr hu hH hv hU hfloor
  have heq : ∀ z ∈ K, fderiv ℝ (phaseEnergy a rho potential gradient) z =
      energyCovector (rho z.1) a (gradient z.1) (H z.1) z.2 (gradU z.1) :=
    fun z hz => fderiv_phaseEnergy_eq a rho potential gradient z.1 z.2 (gradU z.1) (H z.1)
      (hr' z hz) (hu' z hz) (hU' z hz) (he.trans_le (hfloor z hz)).ne'
  exact ⟨h.bound_nonneg, h.lip_nonneg,
    fun z hz => by simpa only [heq z hz] using h.norm_le z hz,
    fun z hz w hw => by simpa only [heq z hz, heq w hw] using h.norm_sub_le z hz w hw⟩

/-- A step inside a convex certified region receives the explicit differential
Lipschitz constant computed above, without a phase-Hessian or compactness premise. -/
def stepJetCertificate (epsilon a dt : ℝ) (K : Set (E × E))
    (rho potential : E → ℝ) (gradient gradU : E → E) (H : E → E →L[ℝ] E)
    (br bu bH bv bU : Budget) (x v acc : E) (he : 0 < epsilon)
    (hr : BoundedLipschitzOn (fun z : E × E => rho z.1) K br.1 br.2)
    (hu : BoundedLipschitzOn (fun z : E × E => gradient z.1) K bu.1 bu.2)
    (hH : BoundedLipschitzOn (fun z : E × E => H z.1) K bH.1 bH.2)
    (hv : BoundedLipschitzOn (Prod.snd : E × E → E) K bv.1 bv.2)
    (hU : BoundedLipschitzOn (fun z : E × E => gradU z.1) K bU.1 bU.2)
    (hfloor : ∀ z ∈ K, epsilon ≤ rho z.1)
    (hr' : ∀ z ∈ K, HasFDerivAt rho (innerSL ℝ (gradient z.1)) z.1)
    (hu' : ∀ z ∈ K, HasFDerivAt gradient (H z.1) z.1)
    (hU' : ∀ z ∈ K, HasFDerivAt potential (innerSL ℝ (gradU z.1)) z.1)
    (hconv : Convex ℝ K) (hstart : (x, v) ∈ K)
    (hend : positionVelocityStep dt x v acc ∈ K) :
    StepJetCertificate a dt rho potential gradient x v acc := by
  have h := fderiv_phaseEnergy_bound epsilon a K rho potential gradient gradU H
    br bu bH bv bU he hr hu hH hv hU hfloor hr' hu' hU'
  exact StepJetCertificate.ofLipschitzRegion a dt rho potential gradient x v acc (gradU x) (H x)
    K ⟨(energyCovectorBudget epsilon a br bu bH bv bU).2, h.lip_nonneg⟩ hconv hstart hend
    (he.trans_le (hfloor (x, v) hstart)).ne'
    (hr' (x, v) hstart) (hu' (x, v) hstart) (hU' (x, v) hstart)
    (fun z hz => differentiableAt_phaseEnergy a rho potential gradient z.1 z.2
      (hr' z hz).differentiableAt (hu' z hz).differentiableAt (hU' z hz).differentiableAt
      (he.trans_le (hfloor z hz)).ne') h.lipschitzOnWith

end FractalSync.ScaleMetric.PhaseEnergyBounds
