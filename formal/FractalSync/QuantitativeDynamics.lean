import FractalSync.LocalDynamics

/-!
# Quantitative local trajectories on a specified phase ball

An actual acceleration Lipschitz constant and phase-field norm bound give
an explicit positive existence horizon, a reference tube, and constants
usable by the continuous numerical-trajectory estimate.
-/

open Set Filter Metric
open scoped Topology NNReal InnerProductSpace

namespace FractalSync.ScaleMetric

/-- A conservative positive horizon with room for endpoint derivatives and
an interior numerical tube. The `+1` also handles a zero speed bound. -/
noncomputable def quantitativeHorizon (R : ℝ) (B : ℝ≥0) : ℝ :=
  R / (4 * ((B : ℝ) + 1))

theorem quantitativeHorizon_properties (R : ℝ) (B : ℝ≥0) (hR : 0 < R) :
    0 < quantitativeHorizon R B ∧
      (B : ℝ) * (2 * quantitativeHorizon R B) ≤ R ∧
      (B : ℝ) * quantitativeHorizon R B < R / 4 := by
  have hden : 0 < (B : ℝ) + 1 := by positivity
  have hT : 0 < quantitativeHorizon R B := div_pos hR (mul_pos (by norm_num) hden)
  have hidentity : ((B : ℝ) + 1) * quantitativeHorizon R B = R / 4 := by
    unfold quantitativeHorizon
    field_simp
  exact ⟨hT, by nlinarith, by nlinarith⟩

section IntegralCurve

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V] [CompleteSpace V]

/-- Regional Lipschitz and speed bounds construct an actual integral curve
on an explicit open interval. The chosen extension is globally B-Lipschitz. -/
theorem exists_quantitative_integralCurve (f : V → V) (p : V) (R : ℝ)
    (K B : ℝ≥0) (hR : 0 < R)
    (hLip : LipschitzOnWith K f (closedBall p R))
    (hB : ∀ x ∈ closedBall p R, ‖f x‖ ≤ B) :
    ∃ z : ℝ → V, z 0 = p ∧ LipschitzWith B z ∧
      ∀ t ∈ Ioo (-2 * quantitativeHorizon R B) (2 * quantitativeHorizon R B),
        HasDerivAt z (f (z t)) t := by
  obtain ⟨hT, htime, _⟩ := quantitativeHorizon_properties R B hR
  let t₀ : Icc (-2 * quantitativeHorizon R B) (2 * quantitativeHorizon R B) :=
    ⟨0, by constructor <;> linarith⟩
  have hpicard : IsPicardLindelof (fun _ => f) t₀ p ⟨R, hR.le⟩ 0 B K :=
    IsPicardLindelof.of_time_independent hB hLip (by simpa [t₀] using htime)
  obtain ⟨α, hα⟩ := ODE.FunSpace.exists_isFixedPt_next hpicard
    (mem_closedBall_self (show (0 : ℝ) ≤ 0 from le_rfl))
  have heq : EqOn α.compProj (ODE.picard (fun _ => f) 0 p α.compProj)
      (Icc (-2 * quantitativeHorizon R B) (2 * quantitativeHorizon R B)) := by
    intro t ht
    rw [ODE.FunSpace.compProj_of_mem ht]
    nth_rw 1 [← hα]
    rfl
  have hderiv : ∀ t ∈ Ioo (-2 * quantitativeHorizon R B) (2 * quantitativeHorizon R B),
      HasDerivAt α.compProj (f (α.compProj t)) t := by
    intro t ht
    exact ((ODE.hasDerivWithinAt_picard_Icc t₀.property hpicard.continuousOn_uncurry
      α.continuous_compProj.continuousOn
      (fun _ _ => α.compProj_mem_closedBall hpicard.mul_max_le) p
      (Ioo_subset_Icc_self ht)).congr_of_mem heq (Ioo_subset_Icc_self ht)).hasDerivAt
        (Icc_mem_nhds ht.1 ht.2)
  refine ⟨α.compProj, ?_, ?_, hderiv⟩
  · simpa only [show (t₀ : ℝ) = 0 from rfl, ODE.picard_apply₀] using heq t₀.property
  · simpa only [mul_one] using
      α.lipschitzWith.comp (LipschitzWith.projIcc (le_trans t₀.property.1 t₀.property.2))

end IntegralCurve

section PhaseSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [CompleteSpace E]

noncomputable local instance quantitativeDynamicsPhaseNormedSpace : NormedSpace ℝ (E × E) :=
  Prod.normedSpace

/-- A regional acceleration bound also bounds phase speed, accounting for
the entire velocity range in the supplied phase ball. -/
noncomputable def phaseBallSpeed (p : E × E) (R : ℝ) (A : ℝ≥0) : ℝ≥0 :=
  ⟨max (‖p.2‖ + R) (A : ℝ), A.coe_nonneg.trans (le_max_right _ _)⟩

omit [InnerProductSpace ℝ E] [CompleteSpace E] in
theorem phaseField_norm_le_phaseBallSpeed (acc : E × E → E) (p z : E × E)
    (R : ℝ) (A : ℝ≥0) (hz : z ∈ closedBall p R) (hacc : ‖acc z‖ ≤ A) :
    ‖phaseField acc z‖ ≤ phaseBallSpeed p R A := by
  have hv : ‖z.2‖ ≤ ‖p.2‖ + R := by
    have hsub : ‖z.2 - p.2‖ ≤ R :=
      (norm_snd_le (z - p)).trans (mem_closedBall_iff_norm.mp hz)
    linarith [norm_le_norm_sub_add z.2 p.2]
  exact max_le_max hv hacc

/-- An explicit phase-ball hypothesis constructs the reference and its tube,
with exactly the supplied acceleration and speed constants in the certificate. -/
theorem exists_quantitative_ODE_certificate (acc : E × E → E) (p : E × E)
    (R : ℝ) (L B : ℝ≥0) (hR : 0 < R)
    (hLip : LipschitzOnWith L acc (closedBall p R))
    (hB : ∀ x ∈ closedBall p R, ‖phaseField acc x‖ ≤ B) :
    ∃ z : ℝ → E × E, z 0 = p ∧
      ∃ c : UniformODECertificate acc z (quantitativeHorizon R B) (ball p R),
        c.accelerationLip = L ∧ c.phaseSpeed = B ∧
        ∀ t ∈ Icc (0 : ℝ) (quantitativeHorizon R B),
          ‖z t - p‖ ≤ (B : ℝ) * t ∧
          ∀ w : E × E, ‖w - z t‖ ≤ R / 2 → w ∈ ball p R := by
  have hphase : LipschitzOnWith (max 1 L) (phaseField acc) (closedBall p R) := by
    apply LipschitzOnWith.of_dist_le_mul
    intro u hu v hv
    simpa only [dist_eq_norm, NNReal.coe_max, NNReal.coe_one] using
      phaseField_sub_le acc u v L (by simpa only [dist_eq_norm] using hLip.dist_le_mul u hu v hv)
  obtain ⟨z, hz0, hzLip, hz⟩ :=
    exists_quantitative_integralCurve (phaseField acc) p R (max 1 L) B hR hphase hB
  have hdisplacement : ∀ t ∈ Icc (0 : ℝ) (quantitativeHorizon R B),
      ‖z t - p‖ ≤ (B : ℝ) * t := by
    intro t ht
    simpa only [dist_eq_norm, hz0, sub_zero, Real.norm_eq_abs, abs_of_nonneg ht.1] using
      hzLip.dist_le_mul t 0
  have hsmall : ∀ t ∈ Icc (0 : ℝ) (quantitativeHorizon R B), ‖z t - p‖ < R / 4 := by
    intro t ht
    exact (hdisplacement t ht).trans_lt
      ((mul_le_mul_of_nonneg_left ht.2 B.coe_nonneg).trans_lt
        (quantitativeHorizon_properties R B hR).2.2)
  have hmem : ∀ t ∈ Icc (0 : ℝ) (quantitativeHorizon R B), z t ∈ ball p R := by
    intro t ht
    exact mem_ball_iff_norm.mpr ((hsmall t ht).trans (by linarith))
  have hequation : ∀ t ∈ Icc (0 : ℝ) (quantitativeHorizon R B),
      HasDerivAt z (phaseField acc (z t)) t := by
    intro t ht
    have hT := (quantitativeHorizon_properties R B hR).1
    exact hz t ⟨by linarith [ht.1], by linarith [ht.2]⟩
  let c : UniformODECertificate acc z (quantitativeHorizon R B) (ball p R) :=
    ⟨L, B, hequation, hmem, fun t ht => hB _ (ball_subset_closedBall (hmem t ht)),
      fun u hu v hv => by
        simpa only [dist_eq_norm] using
          hLip.dist_le_mul u (ball_subset_closedBall hu) v (ball_subset_closedBall hv)⟩
  refine ⟨z, hz0, c, rfl, rfl, fun t ht => ⟨hdisplacement t ht, ?_⟩⟩
  intro w hw
  exact mem_ball_iff_norm.mpr
    ((norm_sub_le_norm_sub_add_norm_sub w (z t) p).trans_lt (by linarith [hsmall t ht]))

/-- Regional constants construct one actual reference, prove its uniqueness,
and certify every sufficiently resolved numerical history on an explicit horizon.
The step-count condition uses only the given constants, not an unknown margin. -/
theorem exists_quantitative_trajectory_control (acc : E × E → E) (p : E × E)
    (R : ℝ) (L B : ℝ≥0) (hR : 0 < R)
    (hLip : LipschitzOnWith L acc (closedBall p R))
    (hB : ∀ x ∈ closedBall p R, ‖phaseField acc x‖ ≤ B)
    (residualCoeff roundingCoeff initialCoeff : ℝ≥0) :
    let T := quantitativeHorizon R B
    let C := refinementCoefficient T L B residualCoeff roundingCoeff initialCoeff + (B : ℝ) / 2
    ∃ z : ℝ → E × E, z 0 = p ∧
      (∀ t ∈ Icc (0 : ℝ) T, HasDerivAt z (phaseField acc (z t)) t) ∧
      (∀ w : ℝ → E × E,
        (∀ t ∈ Icc (0 : ℝ) T, HasDerivAt w (phaseField acc (w t)) t) →
        (∀ t ∈ Icc (0 : ℝ) T, w t ∈ ball p R) → w 0 = p → EqOn z w (Icc (0 : ℝ) T)) ∧
      ∀ (N : ℕ) (y : ℕ → E × E) (A : ℕ → E),
        ‖y 0 - p‖ ≤ refinementStep T N * initialCoeff →
        (∀ i < N + 1, y i ∈ ball p R →
          ‖A i - acc (y i)‖ ≤ refinementStep T N * residualCoeff) →
        (∀ i < N + 1, y i ∈ ball p R →
          ‖y (i + 1) - positionVelocityStep (refinementStep T N) (y i).1 (y i).2 (A i)‖ ≤
            (refinementStep T N) ^ 2 * roundingCoeff) →
        2 * T * C ≤ R * ((N : ℝ) + 1) →
        Continuous (trajectoryInterpolant (refinementStep T N) y (N + 1)) ∧
        ∀ t ∈ Icc (0 : ℝ) T,
          ‖trajectoryInterpolant (refinementStep T N) y (N + 1) t - z t‖ ≤
            refinementStep T N * C ∧
          trajectoryInterpolant (refinementStep T N) y (N + 1) t ∈ ball p R := by
  obtain ⟨z, hz0, c, hcL, hcB, htube⟩ :=
    exists_quantitative_ODE_certificate acc p R L B hR hLip hB
  refine ⟨z, hz0, c.equation, ?_, ?_⟩
  · intro w hw hmem hw0
    exact c.solution_unique w hw hmem (hz0.trans hw0.symm)
  · intro N y A hinit hres hround hcount
    have hfit : refinementStep (quantitativeHorizon R B) N *
        (refinementCoefficient (quantitativeHorizon R B) c.accelerationLip c.phaseSpeed
          residualCoeff roundingCoeff initialCoeff + c.phaseSpeed / 2) ≤ R / 2 := by
      rw [hcL, hcB]
      unfold refinementStep
      rw [div_mul_eq_mul_div, div_le_iff₀ (by positivity)]
      push_cast
      nlinarith [hcount]
    have hbound := c.interpolation_error_bound (quantitativeHorizon_properties R B hR).1
      N y A residualCoeff roundingCoeff initialCoeff (R / 2)
      (by simpa only [hz0] using hinit) hres hround (fun t ht => (htube t ht).2) hfit
    exact ⟨continuous_trajectoryInterpolant _ y
      (div_pos (quantitativeHorizon_properties R B hR).1 (by positivity)) _,
      by simpa only [hcL, hcB] using hbound⟩

end PhaseSpace

end FractalSync.ScaleMetric
