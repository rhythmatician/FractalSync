import FractalSync.AccumulatedDynamics

/-!
# Actual phase-energy regularity from smooth configuration fields

Smooth ruler, gradient, and potential fields imply actual phase-energy
regularity. Compact regions then admit a uniform Hessian bound. This is an
existence statement for a finite constant, not a numerical certificate for
its magnitude or for an actual provider's field derivatives.
-/

open Set
open scoped InnerProductSpace NNReal

namespace FractalSync.ScaleMetric

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

noncomputable local instance : NormedSpace ℝ (E × E) := Prod.normedSpace

/-- Actual smoothness of the configuration fields gives smoothness of the
actual phase energy wherever the ruler is nonzero. -/
theorem contDiffAt_phaseEnergy (a : ℝ) (rho potential : E → ℝ)
    (gradient : E → E) (x v : E) (n : WithTop ℕ∞)
    (hr : ContDiffAt ℝ n rho x) (hu : ContDiffAt ℝ n gradient x)
    (hU : ContDiffAt ℝ n potential x) (hrho : rho x ≠ 0) :
    ContDiffAt ℝ n (phaseEnergy a rho potential gradient) (x, v) := by
  unfold phaseEnergy kinetic metric
  have hnorm := (contDiffAt_snd : ContDiffAt ℝ n (Prod.snd : E × E → E) (x, v)).inner ℝ
    contDiffAt_snd
  have hcross := (hu.comp (x, v) contDiffAt_fst).inner ℝ contDiffAt_snd
  have hrp : ContDiffAt ℝ n (fun z : E × E => rho z.1) (x, v) :=
    hr.comp (x, v) contDiffAt_fst
  exact (((hnorm.add (((contDiffAt_const (c := a ^ 2)).mul hcross).mul hcross)).div
    (hrp.pow 2) (pow_ne_zero 2 hrho)).div (contDiffAt_const (c := (2 : ℝ)))
      (by norm_num)).add (hU.comp (x, v) contDiffAt_fst)

/-- Actual `C²` fields give a finite uniform actual phase-energy Hessian bound
on any compact region where the ruler is nonzero. Compactness supplies the
constant; this statement does not supply a numerical value for it. -/
theorem exists_phaseEnergy_hessian_bound_on_compact (a : ℝ)
    (rho potential : E → ℝ) (gradient : E → E) (K : Set (E × E))
    (hK : IsCompact K)
    (hr : ∀ z ∈ K, ContDiffAt ℝ 2 rho z.1)
    (hu : ∀ z ∈ K, ContDiffAt ℝ 2 gradient z.1)
    (hU : ∀ z ∈ K, ContDiffAt ℝ 2 potential z.1)
    (hrho : ∀ z ∈ K, rho z.1 ≠ 0) :
    ∃ L : ℝ≥0,
      (∀ z ∈ K, DifferentiableAt ℝ (phaseEnergy a rho potential gradient) z) ∧
      (∀ z ∈ K, HasFDerivAt (fderiv ℝ (phaseEnergy a rho potential gradient))
        (fderiv ℝ (fderiv ℝ (phaseEnergy a rho potential gradient)) z) z) ∧
      (∀ z ∈ K, ‖fderiv ℝ (fderiv ℝ (phaseEnergy a rho potential gradient)) z‖ ≤ L) := by
  have hp : ∀ z ∈ K, ContDiffAt ℝ 2 (phaseEnergy a rho potential gradient) z := by
    intro z hz
    exact contDiffAt_phaseEnergy a rho potential gradient z.1 z.2 2
      (hr z hz) (hu z hz) (hU z hz) (hrho z hz)
  have hDJ : ∀ z ∈ K, ContDiffAt ℝ 1
      (fderiv ℝ (phaseEnergy a rho potential gradient)) z :=
    fun z hz => (hp z hz).fderiv_right (by norm_num)
  have hcont : ContinuousOn
      (fderiv ℝ (fderiv ℝ (phaseEnergy a rho potential gradient))) K := by
    intro z hz
    exact ((hDJ z hz).fderiv_right (m := 0) (by norm_num)).continuousAt.continuousWithinAt
  obtain ⟨C, hC⟩ := hK.exists_bound_of_continuousOn
    (f := fderiv ℝ (fderiv ℝ (phaseEnergy a rho potential gradient))) hcont
  refine ⟨⟨max C 0, le_max_right C 0⟩,
    fun z hz => (hp z hz).differentiableAt (by norm_num),
    fun z hz => ((hDJ z hz).differentiableAt (by norm_num)).hasFDerivAt, ?_⟩
  exact fun z hz => (hC z hz).trans (le_max_left C 0)

/-- On a compact convex phase region the actual energy differential has one
uniform Lipschitz constant, derived from the actual field regularity. -/
theorem exists_phaseEnergy_lipschitz_fderiv_on_compact (a : ℝ)
    (rho potential : E → ℝ) (gradient : E → E) (K : Set (E × E))
    (hK : IsCompact K) (hconv : Convex ℝ K)
    (hr : ∀ z ∈ K, ContDiffAt ℝ 2 rho z.1)
    (hu : ∀ z ∈ K, ContDiffAt ℝ 2 gradient z.1)
    (hU : ∀ z ∈ K, ContDiffAt ℝ 2 potential z.1)
    (hrho : ∀ z ∈ K, rho z.1 ≠ 0) :
    ∃ L : ℝ≥0,
      (∀ z ∈ K, DifferentiableAt ℝ (phaseEnergy a rho potential gradient) z) ∧
      LipschitzOnWith L (fderiv ℝ (phaseEnergy a rho potential gradient)) K := by
  obtain ⟨L, hD, hJ, hbound⟩ := exists_phaseEnergy_hessian_bound_on_compact
    a rho potential gradient K hK hr hu hU hrho
  exact ⟨L, hD, hconv.lipschitzOnWith_of_nnnorm_hasFDerivWithin_le
    (fun z hz => (hJ z hz).hasFDerivWithinAt) (fun z hz => hbound z hz)⟩

/-- A uniform actual differential bound on a convex phase region supplies a
step certificate whenever both phase endpoints lie in that region. Together
with the compact-field theorem above this supplies a shared regularity
constant for every such step. Actual initial field jets are still required. -/
noncomputable def StepJetCertificate.ofLipschitzRegion (a dt : ℝ)
    (rho potential : E → ℝ) (gradient : E → E) (x v acc gradU : E)
    (H : E →L[ℝ] E) (K : Set (E × E)) (L : ℝ≥0)
    (hconv : Convex ℝ K) (hstart : (x, v) ∈ K)
    (hend : positionVelocityStep dt x v acc ∈ K)
    (hrho : rho x ≠ 0)
    (hr : HasFDerivAt rho (innerSL ℝ (gradient x)) x)
    (hu : HasFDerivAt gradient H x)
    (hU : HasFDerivAt potential (innerSL ℝ gradU) x)
    (hD : ∀ z ∈ K, DifferentiableAt ℝ (phaseEnergy a rho potential gradient) z)
    (hLip : LipschitzOnWith L (fderiv ℝ (phaseEnergy a rho potential gradient)) K) :
    StepJetCertificate a dt rho potential gradient x v acc := by
  have hmem : ∀ t ∈ Icc (0 : ℝ) 1,
      (x, v) + t • (dt • (velocityEulerStep dt v acc, acc)) ∈ K := by
    intro t ht
    apply hconv.add_smul_mem hstart
    · simpa only [positionVelocityStep_eq_add] using hend
    · exact ht
  refine ⟨H, gradU, L, hrho, hr, hu, hU, fun t ht => hD _ (hmem t ht), ?_⟩
  exact fun t ht => by
    simpa [dist_eq_norm] using hLip.dist_le_mul
      ((x, v) + t • (dt • (velocityEulerStep dt v acc, acc))) (hmem t ht)
      (x, v) hstart

end FractalSync.ScaleMetric
