import FractalSync.TrajectoryError
import Mathlib.Analysis.SpecificLimits.Basic

/-!
# Fixed-horizon refinement of the actual ordered dynamics

Uniform field/ODE certificates and explicit residual/rounding scaling give a
checked refinement limit. Domain membership is obtained from an exact-curve
interior margin and the previously derived trajectory-error induction.
-/

open Set Filter
open scoped Topology BigOperators NNReal

namespace FractalSync.ScaleMetric

/-- A closed exponential bound for the finite recurrence, retaining the
kappa=1 case without division by kappa-1. -/
theorem trajectoryEnvelope_le_exp (kappa delta initial h C : ℝ) (n : ℕ)
    (hk : 1 ≤ kappa) (hd : 0 ≤ delta) (hi : 0 ≤ initial)
    (hstep : kappa ≤ 1 + h * C) :
    trajectoryEnvelope kappa delta initial n ≤
      Real.exp ((n : ℝ) * h * C) * (initial + (n : ℝ) * delta) := by
  have hsum : (∑ j ∈ Finset.range n, kappa ^ j) ≤ (n : ℝ) * kappa ^ n := by
    calc
      (∑ j ∈ Finset.range n, kappa ^ j) ≤ ∑ _j ∈ Finset.range n, kappa ^ n :=
        Finset.sum_le_sum (fun j hj => pow_le_pow_right₀ hk (Finset.mem_range.mp hj).le)
      _ = (n : ℝ) * kappa ^ n := by simp
  have hpow : kappa ^ n ≤ Real.exp ((n : ℝ) * h * C) := by
    calc
      kappa ^ n ≤ (Real.exp (h * C)) ^ n := pow_le_pow_left₀ (zero_le_one.trans hk)
        (hstep.trans (by simpa only [add_comm] using Real.add_one_le_exp (h * C))) n
      _ = Real.exp ((n : ℝ) * h * C) := by rw [← Real.exp_nat_mul, mul_assoc]
  rw [trajectoryEnvelope_eq]
  calc
    kappa ^ n * initial + delta * (∑ j ∈ Finset.range n, kappa ^ j) ≤
        kappa ^ n * (initial + (n : ℝ) * delta) := by
      nlinarith [mul_le_mul_of_nonneg_left hsum hd]
    _ ≤ Real.exp ((n : ℝ) * h * C) * (initial + (n : ℝ) * delta) :=
      mul_le_mul_of_nonneg_right hpow (by positivity)

noncomputable def refinementCoefficient (T L B residualCoeff roundingCoeff initialCoeff : ℝ) : ℝ :=
  Real.exp (T * (max 1 L + T * L)) *
    (initialCoeff + T * ((1 + T) * residualCoeff + roundingCoeff + B + max 1 L * B / 2))

/-- With O(h) acceleration residual, O(h²) stored-state rounding, and O(h)
initial error, the derived finite-horizon envelope has an explicit O(h) bound. -/
theorem trajectoryEnvelope_fixed_horizon_bound
    (T h L B residualCoeff roundingCoeff initialCoeff : ℝ) (n : ℕ)
    (hh : 0 ≤ h) (hstep : h ≤ T) (hL : 0 ≤ L) (hB : 0 ≤ B)
    (he : 0 ≤ residualCoeff) (hr : 0 ≤ roundingCoeff) (hi : 0 ≤ initialCoeff)
    (horizon : (n : ℝ) * h = T) :
    trajectoryEnvelope (stepAmplification h L)
      (localTrajectoryBudget h L B (h * residualCoeff) (h ^ 2 * roundingCoeff))
      (h * initialCoeff) n ≤ h * refinementCoefficient T L B residualCoeff roundingCoeff initialCoeff := by
  have hAmp : stepAmplification h L ≤ 1 + h * (max 1 L + T * L) := by
    unfold stepAmplification
    rw [abs_of_nonneg hh]
    nlinarith [mul_nonneg hh (mul_nonneg (sub_nonneg.mpr hstep) hL)]
  let D := (1 + T) * residualCoeff + roundingCoeff + B + max 1 L * B / 2
  have hdelta : localTrajectoryBudget h L B (h * residualCoeff) (h ^ 2 * roundingCoeff) ≤
      h ^ 2 * D := by
    unfold localTrajectoryBudget
    rw [abs_of_nonneg hh]
    dsimp [D]
    nlinarith [mul_nonneg (sq_nonneg h) (mul_nonneg (sub_nonneg.mpr hstep) he)]
  have hexp := trajectoryEnvelope_le_exp (stepAmplification h L)
    (localTrajectoryBudget h L B (h * residualCoeff) (h ^ 2 * roundingCoeff))
    (h * initialCoeff) h (max 1 L + T * L) n (one_le_stepAmplification h L hL)
    (localTrajectoryBudget_nonneg h L B _ _ hB (mul_nonneg hh he) (mul_nonneg (sq_nonneg h) hr))
    (mul_nonneg hh hi) hAmp
  rw [horizon] at hexp
  calc
    trajectoryEnvelope (stepAmplification h L)
        (localTrajectoryBudget h L B (h * residualCoeff) (h ^ 2 * roundingCoeff))
        (h * initialCoeff) n ≤
      Real.exp (T * (max 1 L + T * L)) *
        (h * initialCoeff + (n : ℝ) * localTrajectoryBudget h L B
          (h * residualCoeff) (h ^ 2 * roundingCoeff)) := hexp
    _ ≤ Real.exp (T * (max 1 L + T * L)) *
        (h * initialCoeff + (n : ℝ) * (h ^ 2 * D)) := by
      gcongr
    _ = h * refinementCoefficient T L B residualCoeff roundingCoeff initialCoeff := by
      have heq : (n : ℝ) * (h ^ 2 * D) = h * T * D := by
        calc
          (n : ℝ) * (h ^ 2 * D) = h * ((n : ℝ) * h) * D := by ring
          _ = h * T * D := by rw [horizon]
      rw [heq]
      unfold refinementCoefficient
      dsimp [D]
      ring

/-- N indexes a refinement with N+1 nondegenerate step counts. -/
noncomputable def refinementStep (T : ℝ) (N : ℕ) : ℝ := T / ((N + 1 : ℕ) : ℝ)

noncomputable def refinementEnvelope
    (T L B residualCoeff roundingCoeff initialCoeff : ℝ) (N : ℕ) : ℝ :=
  trajectoryEnvelope (stepAmplification (refinementStep T N) L)
    (localTrajectoryBudget (refinementStep T N) L B
      (refinementStep T N * residualCoeff) ((refinementStep T N) ^ 2 * roundingCoeff))
    (refinementStep T N * initialCoeff) (N + 1)

theorem refinementStep_properties (T : ℝ) (N : ℕ) (hT : 0 ≤ T) :
    0 ≤ refinementStep T N ∧ refinementStep T N ≤ T ∧
      ((N + 1 : ℕ) : ℝ) * refinementStep T N = T := by
  have hn : (0 : ℝ) < ((N + 1 : ℕ) : ℝ) := by positivity
  have hn1 : (1 : ℝ) ≤ ((N + 1 : ℕ) : ℝ) := by exact_mod_cast Nat.succ_le_succ (Nat.zero_le N)
  refine ⟨div_nonneg hT hn.le, ?_, ?_⟩
  · unfold refinementStep
    apply (div_le_iff₀ hn).2
    nlinarith
  · unfold refinementStep
    field_simp

theorem tendsto_refinementStep_zero (T : ℝ) :
    Tendsto (refinementStep T) atTop (𝓝 0) := by
  exact (tendsto_add_atTop_iff_nat 1).2 (tendsto_const_div_atTop_nhds_zero_nat T)

/-- The derived trajectory envelope itself tends to zero at fixed horizon
under the stated residual, rounding, and initial-error scaling. -/
theorem tendsto_refinementEnvelope_zero
    (T L B residualCoeff roundingCoeff initialCoeff : ℝ)
    (hT : 0 ≤ T) (hL : 0 ≤ L) (hB : 0 ≤ B)
    (he : 0 ≤ residualCoeff) (hr : 0 ≤ roundingCoeff) (hi : 0 ≤ initialCoeff) :
    Tendsto (refinementEnvelope T L B residualCoeff roundingCoeff initialCoeff) atTop (𝓝 0) := by
  have hupper : ∀ N, refinementEnvelope T L B residualCoeff roundingCoeff initialCoeff N ≤
      refinementStep T N * refinementCoefficient T L B residualCoeff roundingCoeff initialCoeff := by
    intro N
    exact trajectoryEnvelope_fixed_horizon_bound T (refinementStep T N) L B
      residualCoeff roundingCoeff initialCoeff (N + 1) (refinementStep_properties T N hT).1
      (refinementStep_properties T N hT).2.1 hL hB he hr hi (refinementStep_properties T N hT).2.2
  have hnonneg : ∀ N, 0 ≤ refinementEnvelope T L B residualCoeff roundingCoeff initialCoeff N := by
    intro N
    have hh := (refinementStep_properties T N hT).1
    exact trajectoryEnvelope_nonneg _ _ _
      (zero_le_one.trans (one_le_stepAmplification _ L hL))
      (localTrajectoryBudget_nonneg _ L B _ _ hB (mul_nonneg hh he)
        (mul_nonneg (sq_nonneg _) hr)) (mul_nonneg hh hi) (N + 1)
  apply squeeze_zero hnonneg hupper
  simpa using (tendsto_refinementStep_zero T).mul_const
    (refinementCoefficient T L B residualCoeff roundingCoeff initialCoeff)

theorem refinement_interval_subset (T : ℝ) (N i : ℕ) (hT : 0 ≤ T) (hi : i < N + 1) :
    Icc ((i : ℝ) * refinementStep T N) ((i : ℝ) * refinementStep T N + refinementStep T N) ⊆
      Icc (0 : ℝ) T := by
  intro s hs
  have hh := (refinementStep_properties T N hT).1
  have horizon := (refinementStep_properties T N hT).2.2
  have hi' : (i : ℝ) + 1 ≤ ((N + 1 : ℕ) : ℝ) := by exact_mod_cast Nat.succ_le_of_lt hi
  exact ⟨(mul_nonneg (Nat.cast_nonneg i) hh).trans hs.1,
    by nlinarith [hs.2, mul_le_mul_of_nonneg_right hi' hh]⟩

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

noncomputable local instance trajectoryRefinementNormedSpace : NormedSpace ℝ (E × E) := Prod.normedSpace

/-- One actual ODE/field certificate on a fixed horizon, independent of the
number of steps used to approximate it. -/
structure UniformODECertificate (acc : E × E → E) (z : ℝ → E × E)
    (T : ℝ) (K : Set (E × E)) where
  accelerationLip : ℝ≥0
  phaseSpeed : ℝ≥0
  equation : ∀ s ∈ Icc (0 : ℝ) T, HasDerivAt z (phaseField acc (z s)) s
  reference_mem : ∀ s ∈ Icc (0 : ℝ) T, z s ∈ K
  speed_bound : ∀ s ∈ Icc (0 : ℝ) T, ‖phaseField acc (z s)‖ ≤ phaseSpeed
  acceleration_lipschitz : ∀ p ∈ K, ∀ q ∈ K,
    ‖acc p - acc q‖ ≤ accelerationLip * ‖p - q‖

/-- The same actual solution and field bounds supply every refinement. -/
noncomputable def UniformODECertificate.onRefinement {acc : E × E → E} {z : ℝ → E × E}
    {T : ℝ} {K : Set (E × E)} (c : UniformODECertificate acc z T K) (hT : 0 ≤ T) (N : ℕ) :
    ODERegionCertificate acc z (fun i => (i : ℝ) * refinementStep T N)
      (refinementStep T N) (N + 1) K where
  accelerationLip := c.accelerationLip
  phaseSpeed := c.phaseSpeed
  grid := by intro i _; push_cast; ring
  equation := fun i hi s hs => c.equation s (refinement_interval_subset T N i hT hi hs)
  reference_mem := fun i hi s hs => c.reference_mem s (refinement_interval_subset T N i hT hi hs)
  speed_bound := fun i hi s hs => c.speed_bound s (refinement_interval_subset T N i hT hi hs)
  acceleration_lipschitz := c.acceleration_lipschitz

/-- Actual grid trajectories converge uniformly at the grid points as the
fixed-horizon mesh is refined. An exact-curve interior margin closes domain
membership for all sufficiently fine meshes; it is not assumed for the
numerical histories. Residual and rounding budgets scale as h and h². -/
theorem UniformODECertificate.refinement_converges {acc : E × E → E} {z : ℝ → E × E}
    {T : ℝ} {K : Set (E × E)} (c : UniformODECertificate acc z T K) (hT : 0 ≤ T)
    (y : ℕ → ℕ → E × E) (A : ℕ → ℕ → E)
    (residualCoeff roundingCoeff initialCoeff : ℝ≥0) (radius : ℝ) (hradius : 0 < radius)
    (hinit : ∀ N, ‖y N 0 - z 0‖ ≤ refinementStep T N * initialCoeff)
    (hres : ∀ N i, i < N + 1 → y N i ∈ K →
      ‖A N i - acc (y N i)‖ ≤ refinementStep T N * residualCoeff)
    (hround : ∀ N i, i < N + 1 → y N i ∈ K →
      ‖y N (i + 1) - positionVelocityStep (refinementStep T N) (y N i).1 (y N i).2 (A N i)‖ ≤
        (refinementStep T N) ^ 2 * roundingCoeff)
    (htube : ∀ s ∈ Icc (0 : ℝ) T, ∀ w : E × E, ‖w - z s‖ ≤ radius → w ∈ K) :
    ∀ epsilon > 0, ∀ᶠ N in atTop, ∀ i ≤ N + 1,
      ‖y N i - z ((i : ℝ) * refinementStep T N)‖ < epsilon ∧ y N i ∈ K := by
  have hE := tendsto_refinementEnvelope_zero T c.accelerationLip c.phaseSpeed
    residualCoeff roundingCoeff initialCoeff hT c.accelerationLip.coe_nonneg c.phaseSpeed.coe_nonneg
    residualCoeff.coe_nonneg roundingCoeff.coe_nonneg initialCoeff.coe_nonneg
  intro epsilon hepsilon
  have hsmall := hE.eventually (eventually_lt_nhds (lt_min hradius hepsilon))
  filter_upwards [hsmall] with N hsmallN
  have hh := (refinementStep_properties T N hT).1
  let eta : ℝ≥0 := ⟨refinementStep T N * residualCoeff, mul_nonneg hh residualCoeff.coe_nonneg⟩
  let rounding : ℝ≥0 := ⟨(refinementStep T N) ^ 2 * roundingCoeff,
    mul_nonneg (sq_nonneg _) roundingCoeff.coe_nonneg⟩
  let cn := c.onRefinement hT N
  have hfinal : cn.errorEnvelope eta rounding (refinementStep T N * initialCoeff) (N + 1) <
      min radius epsilon := hsmallN
  have htubeN : ∀ i ≤ N + 1, ∀ w : E × E,
      ‖w - z ((i : ℝ) * refinementStep T N)‖ ≤
        cn.errorEnvelope eta rounding (refinementStep T N * initialCoeff) (N + 1) → w ∈ K := by
    intro i hi w hw
    have hi' : (i : ℝ) ≤ ((N + 1 : ℕ) : ℝ) := by exact_mod_cast hi
    have ht : (i : ℝ) * refinementStep T N ∈ Icc (0 : ℝ) T := by
      refine ⟨mul_nonneg (Nat.cast_nonneg i) hh, ?_⟩
      exact (mul_le_mul_of_nonneg_right hi' hh).trans_eq (refinementStep_properties T N hT).2.2
    exact htube _ ht w (hw.trans (hfinal.trans_le (min_le_left radius epsilon)).le)
  have hrun := cn.trajectory_error_domain (y N) (A N) eta rounding
    (refinementStep T N * initialCoeff) hh
    (by simpa only [Nat.cast_zero, zero_mul] using hinit N) (hres N) (hround N) htubeN
  have hmono : Monotone (cn.errorEnvelope eta rounding (refinementStep T N * initialCoeff)) :=
    trajectoryEnvelope_mono _ _ _
      (one_le_stepAmplification _ cn.accelerationLip cn.accelerationLip.coe_nonneg)
      (localTrajectoryBudget_nonneg _ cn.accelerationLip cn.phaseSpeed eta rounding
        cn.phaseSpeed.coe_nonneg eta.coe_nonneg rounding.coe_nonneg)
      (mul_nonneg hh initialCoeff.coe_nonneg)
  intro i hi
  obtain ⟨herr, hmem⟩ := hrun i hi
  exact ⟨(herr.trans (hmono hi)).trans_lt (hfinal.trans_le (min_le_right radius epsilon)), hmem⟩

end FractalSync.ScaleMetric
