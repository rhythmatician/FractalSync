import FractalSync.RulerDifferential
import FractalSync.ContinuousMechanics
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Deriv

/-!
# Canonical conservative potentials and force covectors

Exact-real contracts for #159 and the adopted geometry in #120. The Shore
potential and the outer secant wall are the current equations in
`runtime-core/src/manifold.rs`. The field derivatives below are actual Fréchet
derivatives. Force covectors are represented using the ambient inner product;
the existing `inverseAction` converts them to coordinate accelerations.

The outer-wall covector is radially inward. Its metric-raised acceleration need
not be Euclidean radial. No result here certifies sampled derivatives, floating
point evaluation, discrete conservation, or finite-step domain safety.
-/

open scoped InnerProductSpace Topology
open MeasureTheory Set Filter
open FractalSync.ScaleMetric

namespace FractalSync.PotentialForces

/-- The canonical Shore-ridge potential, including its reference origin. -/
noncomputable def shorePotential (kappa dref rho : ℝ) : ℝ :=
  kappa * scaleCoordinate dref rho

/-- The actual scalar derivative of the Shore potential. -/
theorem hasDerivAt_shorePotential (kappa dref rho : ℝ)
    (hd : 0 < dref) (hr : 0 < rho) :
    HasDerivAt (shorePotential kappa dref) (-kappa / (rho * Real.log 2)) rho := by
  convert (hasDerivAt_scaleCoordinate dref rho hd hr).const_mul kappa using 1
  ring

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The Shore force covector under the ambient Riesz identification. -/
noncomputable def shoreForce (kappa rho : ℝ) (u : E) : E :=
  (kappa / (rho * Real.log 2)) • u

/-- The displayed Shore force is the negative actual differential of the
potential, requiring only the ruler derivative at the configuration. -/
theorem hasFDerivAt_shorePotential (kappa dref : ℝ) (rho : E → ℝ) (x u : E)
    (hd : 0 < dref) (hr : 0 < rho x)
    (hR : HasFDerivAt rho (innerSL ℝ u) x) :
    HasFDerivAt (fun y => shorePotential kappa dref (rho y))
      (-(innerSL ℝ (shoreForce kappa (rho x) u))) x := by
  convert (hasDerivAt_shorePotential kappa dref (rho x) hd hr).comp_hasFDerivAt x hR using 1
  simp [shoreForce, neg_div, neg_smul]

/-- The exact inverse metric in the ruler-gradient direction, including u=0. -/
theorem inverseAction_gradient (rho a : ℝ) (u : E) :
    inverseAction rho a u u = (rho ^ 2 / weight a u) • u := by
  simp only [inverseAction, unstretch]
  match_scalars
  field_simp [ne_of_gt (weight_pos a u)]
  simp [weight]

theorem inverseAction_smul (rho a k : ℝ) (u J : E) :
    inverseAction rho a u (k • J) = k • inverseAction rho a u J := by
  simp only [inverseAction, unstretch_smul, smul_smul, mul_comm]

/-- Raising the Shore covector cancels its inverse-ruler factor. The formula
has no reference origin and remains valid when the ruler gradient vanishes. -/
theorem shore_acceleration (kappa rho a : ℝ) (u : E) (hr : rho ≠ 0) :
    inverseAction rho a u (shoreForce kappa rho u) =
      (kappa * rho / (Real.log 2 * weight a u)) • u := by
  rw [shoreForce, inverseAction_smul, inverseAction_gradient, smul_smul]
  congr 1
  field_simp

/-- A reference-ruler change adds a configuration-independent potential origin. -/
theorem shorePotential_reference_shift (kappa dref rho alpha : ℝ)
    (hd : 0 < dref) (hr : 0 < rho) (ha : 0 < alpha) :
    shorePotential kappa (alpha * dref) rho =
      shorePotential kappa dref rho + kappa * Real.log alpha / Real.log 2 := by
  rw [shorePotential, scaleCoordinate_reference_shift dref rho alpha hd hr ha]
  simp only [shorePotential, mul_add, mul_div_assoc]

/-- The exact potential difference is independent of the reference origin. -/
theorem shorePotential_difference (kappa dref rho0 rho1 : ℝ)
    (hd : 0 < dref) (h0 : 0 < rho0) (h1 : 0 < rho1) :
    shorePotential kappa dref rho1 - shorePotential kappa dref rho0 =
      kappa * Real.log (rho0 / rho1) / Real.log 2 := by
  rw [shorePotential, shorePotential,
    scaleCoordinate_eq_log_sub dref rho1 hd h1,
    scaleCoordinate_eq_log_sub dref rho0 hd h0, Real.log_div h0.ne' h1.ne']
  ring

/-- Canonical radial wall coordinate: s=norm(x)^2/4. -/
noncomputable def wallCoordinate (x : E) : ℝ := ‖x‖ ^ 2 / 4

/-- The p=8 wall phase, expressed without a radial square root. -/
noncomputable def wallPhase (x : E) : ℝ := Real.pi / 2 * wallCoordinate x ^ 4

/-- The canonical outer secant-wall potential. The physical domain is norm(x)<2. -/
noncomputable def wallPotential (mu : ℝ) (x : E) : ℝ :=
  mu * ((Real.cos (wallPhase x))⁻¹ - 1)

noncomputable def wallCoefficient (mu : ℝ) (x : E) : ℝ :=
  mu * Real.pi * wallCoordinate x ^ 3 * (Real.cos (wallPhase x))⁻¹ * Real.tan (wallPhase x)

/-- The wall force covector, as an ambient Riesz vector. -/
noncomputable def wallForce (mu : ℝ) (x : E) : E := -(wallCoefficient mu x) • x

theorem hasFDerivAt_wallCoordinate (x : E) :
    HasFDerivAt wallCoordinate ((1 / 2 : ℝ) • innerSL ℝ x) x := by
  convert (hasStrictFDerivAt_norm_sq x).hasFDerivAt.const_mul (1 / 4 : ℝ) using 1
  all_goals ext z
  all_goals norm_num [wallCoordinate, ContinuousLinearMap.smul_apply, smul_eq_mul]
  all_goals ring

theorem hasFDerivAt_wallPhase (x : E) :
    HasFDerivAt wallPhase ((Real.pi * wallCoordinate x ^ 3) • innerSL ℝ x) x := by
  convert ((hasFDerivAt_wallCoordinate x).pow 4).const_mul (Real.pi / 2) using 1
  ext v
  norm_num [ContinuousLinearMap.smul_apply, smul_eq_mul]
  ring

/-- Actual Fréchet differentiation identifies the implemented secant-wall
force covector, at every point where the cosine denominator is nonzero. -/
theorem hasFDerivAt_wallPotential_of_cos_ne_zero (mu : ℝ) (x : E)
    (hc : Real.cos (wallPhase x) ≠ 0) :
    HasFDerivAt (wallPotential mu) (-(innerSL ℝ (wallForce mu x))) x := by
  convert (((hasDerivAt_inv hc).comp_hasFDerivAt x
    (hasFDerivAt_wallPhase x).cos).sub_const 1).const_mul mu using 1
  ext v
  simp only [wallForce, wallCoefficient, ContinuousLinearMap.neg_apply,
    ContinuousLinearMap.smul_apply, innerSL_apply, real_inner_smul_left, smul_eq_mul,
    Real.tan_eq_sin_div_cos]
  field_simp

omit [InnerProductSpace ℝ E] in
theorem wallCoordinate_nonneg (x : E) : 0 ≤ wallCoordinate x := by
  exact div_nonneg (sq_nonneg _) (by norm_num)

omit [InnerProductSpace ℝ E] in
theorem wallCoordinate_lt_one (x : E) (hx : ‖x‖ < 2) : wallCoordinate x < 1 := by
  dsimp only [wallCoordinate]
  nlinarith [norm_nonneg x]

omit [InnerProductSpace ℝ E] in
theorem wallPhase_mem (x : E) (hx : ‖x‖ < 2) : wallPhase x ∈ Ico 0 (Real.pi / 2) := by
  have hs := pow_lt_one₀ (wallCoordinate_nonneg x) (wallCoordinate_lt_one x hx)
    (by norm_num : (4 : ℕ) ≠ 0)
  exact ⟨mul_nonneg (by positivity) (pow_nonneg (wallCoordinate_nonneg x) _),
    (mul_lt_mul_of_pos_left hs (by positivity : 0 < Real.pi / 2)).trans_eq (mul_one _)⟩

omit [InnerProductSpace ℝ E] in
theorem cos_wallPhase_pos (x : E) (hx : ‖x‖ < 2) : 0 < Real.cos (wallPhase x) := by
  exact Real.cos_pos_of_mem_Ioo
    ⟨lt_of_lt_of_le (by linarith [Real.pi_pos]) (wallPhase_mem x hx).1,
      (wallPhase_mem x hx).2⟩

/-- The full physical open disk automatically discharges the denominator
hypothesis of the actual wall-force differential. -/
theorem hasFDerivAt_wallPotential (mu : ℝ) (x : E) (hx : ‖x‖ < 2) :
    HasFDerivAt (wallPotential mu) (-(innerSL ℝ (wallForce mu x))) x := by
  exact hasFDerivAt_wallPotential_of_cos_ne_zero mu x (cos_wallPhase_pos x hx).ne'

omit [InnerProductSpace ℝ E] in
theorem wallCoefficient_nonneg (mu : ℝ) (x : E) (hm : 0 ≤ mu) (hx : ‖x‖ < 2) :
    0 ≤ wallCoefficient mu x := by
  have ht := Real.tan_nonneg_of_nonneg_of_le_pi_div_two
    (wallPhase_mem x hx).1 (wallPhase_mem x hx).2.le
  exact mul_nonneg (mul_nonneg (mul_nonneg (mul_nonneg hm Real.pi_pos.le)
    (pow_nonneg (wallCoordinate_nonneg x) _)) (inv_nonneg.mpr (cos_wallPhase_pos x hx).le)) ht

omit [InnerProductSpace ℝ E] in
theorem wallPotential_nonneg (mu : ℝ) (x : E) (hm : 0 ≤ mu) (hx : ‖x‖ < 2) :
    0 ≤ wallPotential mu x := by
  exact mul_nonneg hm (sub_nonneg.mpr
    ((one_le_inv₀ (cos_wallPhase_pos x hx)).2 (Real.cos_le_one _)))

/-- The wall covector does nonpositive work against an outward radial vector. -/
theorem wallForce_radial_nonpos (mu : ℝ) (x : E) (hm : 0 ≤ mu) (hx : ‖x‖ < 2) :
    ⟪x, wallForce mu x⟫_ℝ ≤ 0 := by
  rw [wallForce, real_inner_smul_right]
  exact mul_nonpos_of_nonpos_of_nonneg (neg_nonpos.mpr (wallCoefficient_nonneg mu x hm hx))
    real_inner_self_nonneg

/-- The implemented mu=1/pi specialization cancels pi in the force coefficient. -/
theorem wallForce_default (x : E) :
    wallForce (1 / Real.pi) x =
      (-(wallCoordinate x ^ 3 * (Real.cos (wallPhase x))⁻¹ * Real.tan (wallPhase x))) • x := by
  simp [wallForce, wallCoefficient, Real.pi_ne_zero]

/-- The metric-raised wall force includes the inverse-metric projection term;
Euclidean radiality of the covector does not assert radial acceleration. -/
theorem wall_acceleration (rho a mu : ℝ) (u x : E) :
    inverseAction rho a u (wallForce mu x) =
      (-(wallCoefficient mu x) * rho ^ 2) •
        (x - (a ^ 2 * ⟪u, x⟫_ℝ / weight a u) • u) := by
  rw [wallForce, inverseAction_smul, inverseAction, unstretch, smul_smul]

/-- The full current conservative potential, retaining both Shore and wall. -/
noncomputable def totalPotential (kappa mu dref : ℝ) (rho : E → ℝ) (x : E) : ℝ :=
  shorePotential kappa dref (rho x) + wallPotential mu x

noncomputable def totalForce (kappa mu rho : ℝ) (u x : E) : E :=
  shoreForce kappa rho u + wallForce mu x

/-- The sum of the canonical force covectors is the negative actual
differential of the full conservative potential. -/
theorem hasFDerivAt_totalPotential (kappa mu dref : ℝ) (rho : E → ℝ) (x u : E)
    (hd : 0 < dref) (hr : 0 < rho x) (hx : ‖x‖ < 2)
    (hR : HasFDerivAt rho (innerSL ℝ u) x) :
    HasFDerivAt (totalPotential kappa mu dref rho)
      (-(innerSL ℝ (totalForce kappa mu (rho x) u x))) x := by
  convert (hasFDerivAt_shorePotential kappa dref rho x u hd hr hR).add
    (hasFDerivAt_wallPotential mu x hx) using 1
  simp [totalForce, add_comm]

omit [InnerProductSpace ℝ E] in
theorem totalPotential_reference_shift (kappa mu dref alpha : ℝ) (rho : E → ℝ) (x : E)
    (hd : 0 < dref) (hr : 0 < rho x) (ha : 0 < alpha) :
    totalPotential kappa mu (alpha * dref) rho x =
      totalPotential kappa mu dref rho x + kappa * Real.log alpha / Real.log 2 := by
  simp only [totalPotential, shorePotential_reference_shift kappa dref (rho x) alpha hd hr ha,
    add_right_comm]

/-- Reference-origin invariance of the actual conservative differential. -/
theorem totalPotential_fderiv_reference_invariant (kappa mu dref alpha : ℝ)
    (rho : E → ℝ) (x u : E) (hd : 0 < dref) (ha : 0 < alpha)
    (hr : 0 < rho x) (hx : ‖x‖ < 2) (hR : HasFDerivAt rho (innerSL ℝ u) x) :
    fderiv ℝ (totalPotential kappa mu (alpha * dref) rho) x =
      fderiv ℝ (totalPotential kappa mu dref rho) x := by
  rw [(hasFDerivAt_totalPotential kappa mu (alpha * dref) rho x u (mul_pos ha hd) hr hx hR).fderiv,
    (hasFDerivAt_totalPotential kappa mu dref rho x u hd hr hx hR).fderiv]

/-- The explicitly differentiated Shore and wall forces satisfy the actual
continuous work-energy law in the adopted metric. -/
theorem hasDerivAt_canonicalEnergy (kappa mu dref a t : ℝ)
    (rho : E → ℝ) (gradient : E → E) (gamma velocity : ℝ → E)
    (acc Q : E) (H : E →L[ℝ] E) (hd : 0 < dref)
    (hgamma : HasDerivAt gamma (velocity t) t)
    (hR : HasFDerivAt rho (innerSL ℝ (gradient (gamma t))) (gamma t))
    (hG : HasFDerivAt gradient H (gamma t)) (hv : HasDerivAt velocity acc t)
    (hr : 0 < rho (gamma t)) (hx : ‖gamma t‖ < 2)
    (hmotion : acc + connection (rho (gamma t)) a (gradient (gamma t)) H
      (velocity t) (velocity t) = inverseAction (rho (gamma t)) a (gradient (gamma t))
        (Q + totalForce kappa mu (rho (gamma t)) (gradient (gamma t)) (gamma t))) :
    HasDerivAt (fun s => kinetic (rho (gamma s)) a (gradient (gamma s)) (velocity s) +
      totalPotential kappa mu dref rho (gamma s)) ⟪velocity t, Q⟫_ℝ t := by
  apply hasDerivAt_totalEnergy_of_fields a t rho (totalPotential kappa mu dref rho)
    gradient gamma velocity acc
    (-totalForce kappa mu (rho (gamma t)) (gradient (gamma t)) (gamma t)) Q H
    hgamma hR hG hv _ hr.ne'
  · simpa only [sub_neg_eq_add] using hmotion
  · simpa only [map_neg] using
      hasFDerivAt_totalPotential kappa mu dref rho (gamma t) (gradient (gamma t)) hd hr hx hR

omit [InnerProductSpace ℝ E] in
/-- At an actual zero of D, the regularized ruler reaches epsilon. The full
potential rise retains the wall difference as well as the Shore barrier. -/
theorem totalPotential_crest_difference (kappa mu dref epsilon : ℝ) (D : E → ℝ)
    (x0 x1 : E) (hd : 0 < dref) (he : 0 < epsilon) (hcrest : D x1 = 0) :
    totalPotential kappa mu dref (fun x => ruler epsilon (D x)) x1 -
      totalPotential kappa mu dref (fun x => ruler epsilon (D x)) x0 =
      kappa * Real.log (ruler epsilon (D x0) / epsilon) / Real.log 2 +
        (wallPotential mu x1 - wallPotential mu x0) := by
  simp only [totalPotential, hcrest, ruler_zero epsilon he.le]
  linarith [shorePotential_difference kappa dref (ruler epsilon (D x0)) epsilon
    hd (ruler_pos epsilon (D x0) he) he]

/-- The actual canonical dynamics integrates to signed control work minus the
friction quadratic form, with both conservative potentials retained. -/
theorem canonical_work_energy (kappa mu dref a lo hi : ℝ)
    (rho : E → ℝ) (gradient : E → E) (gamma velocity acc Qc : ℝ → E)
    (H B : ℝ → E →L[ℝ] E) (hd : 0 < dref) (horder : lo ≤ hi)
    (hcont : ContinuousOn (fun t =>
      kinetic (rho (gamma t)) a (gradient (gamma t)) (velocity t) +
        totalPotential kappa mu dref rho (gamma t)) (Icc lo hi))
    (hgamma : ∀ t ∈ Ioo lo hi, HasDerivAt gamma (velocity t) t)
    (hR : ∀ t ∈ Ioo lo hi,
      HasFDerivAt rho (innerSL ℝ (gradient (gamma t))) (gamma t))
    (hG : ∀ t ∈ Ioo lo hi, HasFDerivAt gradient (H t) (gamma t))
    (hv : ∀ t ∈ Ioo lo hi, HasDerivAt velocity (acc t) t)
    (hr : ∀ t ∈ Ioo lo hi, 0 < rho (gamma t))
    (hx : ∀ t ∈ Ioo lo hi, ‖gamma t‖ < 2)
    (hmotion : ∀ t ∈ Ioo lo hi,
      acc t + connection (rho (gamma t)) a (gradient (gamma t)) (H t)
        (velocity t) (velocity t) = inverseAction (rho (gamma t)) a (gradient (gamma t))
          (Qc t - B t (velocity t) +
            totalForce kappa mu (rho (gamma t)) (gradient (gamma t)) (gamma t)))
    (hwork : IntervalIntegrable (fun t => ⟪velocity t, Qc t⟫_ℝ) volume lo hi)
    (hfriction : IntervalIntegrable (fun t => ⟪velocity t, B t (velocity t)⟫_ℝ) volume lo hi) :
    (kinetic (rho (gamma hi)) a (gradient (gamma hi)) (velocity hi) +
      totalPotential kappa mu dref rho (gamma hi)) -
    (kinetic (rho (gamma lo)) a (gradient (gamma lo)) (velocity lo) +
      totalPotential kappa mu dref rho (gamma lo)) =
      (∫ t in lo..hi, ⟪velocity t, Qc t⟫_ℝ) -
        ∫ t in lo..hi, ⟪velocity t, B t (velocity t)⟫_ℝ := by
  have hderiv : ∀ t ∈ Ioo lo hi, HasDerivAt (fun s =>
      kinetic (rho (gamma s)) a (gradient (gamma s)) (velocity s) +
        totalPotential kappa mu dref rho (gamma s))
      (⟪velocity t, Qc t⟫_ℝ - ⟪velocity t, B t (velocity t)⟫_ℝ) t := by
    intro t ht
    simpa only [inner_sub_right] using hasDerivAt_canonicalEnergy kappa mu dref a t
      rho gradient gamma velocity (acc t) (Qc t - B t (velocity t)) (H t) hd
      (hgamma t ht) (hR t ht) (hG t ht) (hv t ht) (hr t ht) (hx t ht) (hmotion t ht)
  have hftc := intervalIntegral.integral_eq_sub_of_hasDerivAt_of_le
    horder hcont hderiv (hwork.sub hfriction)
  simpa only [intervalIntegral.integral_sub hwork hfriction] using hftc.symm

/-- The necessary crest budget extracted from the full signed energy ledger.
The wall difference and dissipation remain explicit; this is not a sufficient
crossing condition. `canonical_work_energy` supplies the ledger from dynamics. -/
theorem crest_budget_of_work_energy (kappa mu dref epsilon a work dissipation : ℝ)
    (D : E → ℝ) (x0 x1 u0 u1 v0 v1 : E)
    (hd : 0 < dref) (he : 0 < epsilon) (hcrest : D x1 = 0)
    (hbalance : work - dissipation =
      (kinetic (ruler epsilon (D x1)) a u1 v1 +
        totalPotential kappa mu dref (fun x => ruler epsilon (D x)) x1) -
      (kinetic (ruler epsilon (D x0)) a u0 v0 +
        totalPotential kappa mu dref (fun x => ruler epsilon (D x)) x0)) :
    kappa * Real.log (ruler epsilon (D x0) / epsilon) / Real.log 2 +
        (wallPotential mu x1 - wallPotential mu x0) ≤
      kinetic (ruler epsilon (D x0)) a u0 v0 + work - dissipation := by
  have hb := potential_increase_le_initial_kinetic_add_work
    (ruler epsilon (D x0)) (ruler epsilon (D x1)) a
    (totalPotential kappa mu dref (fun x => ruler epsilon (D x)) x0)
    (totalPotential kappa mu dref (fun x => ruler epsilon (D x)) x1)
    (work - dissipation) u0 u1 v0 v1 hbalance
  simpa only [totalPotential_crest_difference kappa mu dref epsilon D x0 x1 hd he hcrest,
    ← add_sub_assoc] using hb

/-- Necessary Shore-crest budget for a trajectory satisfying the actual
canonical equation. Positive and negative control work are retained, as are
the full wall-potential difference and the integrated friction quadratic form. -/
theorem canonical_crest_budget (kappa mu dref epsilon a lo hi : ℝ)
    (D : E → ℝ) (gradient : E → E) (gamma velocity acc Qc : ℝ → E)
    (H B : ℝ → E →L[ℝ] E) (hd : 0 < dref) (he : 0 < epsilon) (horder : lo ≤ hi)
    (hcont : ContinuousOn (fun t =>
      kinetic (ruler epsilon (D (gamma t))) a (gradient (gamma t)) (velocity t) +
        totalPotential kappa mu dref (fun x => ruler epsilon (D x)) (gamma t)) (Icc lo hi))
    (hgamma : ∀ t ∈ Ioo lo hi, HasDerivAt gamma (velocity t) t)
    (hR : ∀ t ∈ Ioo lo hi, HasFDerivAt (fun x => ruler epsilon (D x))
      (innerSL ℝ (gradient (gamma t))) (gamma t))
    (hG : ∀ t ∈ Ioo lo hi, HasFDerivAt gradient (H t) (gamma t))
    (hv : ∀ t ∈ Ioo lo hi, HasDerivAt velocity (acc t) t)
    (hx : ∀ t ∈ Ioo lo hi, ‖gamma t‖ < 2)
    (hmotion : ∀ t ∈ Ioo lo hi,
      acc t + connection (ruler epsilon (D (gamma t))) a (gradient (gamma t)) (H t)
        (velocity t) (velocity t) =
        inverseAction (ruler epsilon (D (gamma t))) a (gradient (gamma t))
          (Qc t - B t (velocity t) +
            totalForce kappa mu (ruler epsilon (D (gamma t))) (gradient (gamma t)) (gamma t)))
    (hwork : IntervalIntegrable (fun t => ⟪velocity t, Qc t⟫_ℝ) volume lo hi)
    (hfriction : IntervalIntegrable (fun t => ⟪velocity t, B t (velocity t)⟫_ℝ) volume lo hi)
    (hcrest : D (gamma hi) = 0) :
    kappa * Real.log (ruler epsilon (D (gamma lo)) / epsilon) / Real.log 2 +
        (wallPotential mu (gamma hi) - wallPotential mu (gamma lo)) ≤
      kinetic (ruler epsilon (D (gamma lo))) a (gradient (gamma lo)) (velocity lo) +
        (∫ t in lo..hi, ⟪velocity t, Qc t⟫_ℝ) -
        ∫ t in lo..hi, ⟪velocity t, B t (velocity t)⟫_ℝ := by
  apply crest_budget_of_work_energy kappa mu dref epsilon a
    (∫ t in lo..hi, ⟪velocity t, Qc t⟫_ℝ)
    (∫ t in lo..hi, ⟪velocity t, B t (velocity t)⟫_ℝ)
    D (gamma lo) (gamma hi) (gradient (gamma lo)) (gradient (gamma hi))
    (velocity lo) (velocity hi) hd he hcrest
  exact (canonical_work_energy kappa mu dref a lo hi (fun x => ruler epsilon (D x))
    gradient gamma velocity acc Qc H B hd horder hcont hgamma hR hG hv
    (fun t _ => ruler_pos epsilon (D (gamma t)) he) hx hmotion hwork hfriction).symm

omit [InnerProductSpace ℝ E] in
/-- The actual secant-wall potential diverges along every approach to radius 2
from the physical open disk, provided the wall strength is strictly positive.
This limit alone does not certify a finite-step integrator's domain safety. -/
theorem wallPotential_tendsto_atTop {α : Type*} {l : Filter α}
    (x : α → E) (mu : ℝ) (hm : 0 < mu)
    (hboundary : Tendsto (fun t => ‖x t‖) l (𝓝 2))
    (hinside : ∀ᶠ t in l, ‖x t‖ < 2) :
    Tendsto (fun t => wallPotential mu (x t)) l atTop := by
  have hphase : Tendsto (fun t => wallPhase (x t)) l (𝓝 (Real.pi / 2)) := by
    convert (((hboundary.pow 2).div_const 4).pow 4).const_mul (Real.pi / 2) using 1
    norm_num
  have hcos : Tendsto (fun t => Real.cos (wallPhase (x t))) l (𝓝 0) := by
    simpa only [Real.cos_pi_div_two] using Real.continuous_cos.continuousAt.tendsto.comp hphase
  have hcospos : Tendsto (fun t => Real.cos (wallPhase (x t))) l (𝓝[>] 0) :=
    tendsto_nhdsWithin_iff.mpr ⟨hcos, hinside.mono fun t ht => cos_wallPhase_pos (x t) ht⟩
  simpa only [wallPotential, sub_eq_add_neg] using
    (tendsto_atTop_add_const_right l (-1 : ℝ)
      (tendsto_inv_nhdsGT_zero.comp hcospos)).const_mul_atTop hm

end FractalSync.PotentialForces
