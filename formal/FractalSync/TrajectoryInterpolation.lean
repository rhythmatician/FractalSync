import FractalSync.TrajectoryRefinement

/-!
# Continuous interpolation of actual numerical trajectories

The piecewise-affine interpolant is built from the stored phase states.
Its continuity, domain membership, and off-grid convergence are proved from
the actual grid-error theorem and the reference curve's derivative bound.
-/

open Set Filter
open scoped Topology NNReal

namespace FractalSync.ScaleMetric

section Normed

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

/-- The affine segment with its dimensional time argument. -/
noncomputable def trajectorySegment (h : ℝ) (p q : V) (t : ℝ) : V := p + (t / h) • (q - p)

/-- Continuous interpolation of the first `n+1` stored states. The last
segment continues affinely after the horizon; before zero the path is constant. -/
noncomputable def trajectoryInterpolant (h : ℝ) (y : ℕ → V) : ℕ → ℝ → V
  | 0 => fun _ => y 0
  | n + 1 => fun t => if t ≤ (n : ℝ) * h then trajectoryInterpolant h y n t
      else trajectorySegment h (y n) (y (n + 1)) (t - (n : ℝ) * h)

theorem trajectorySegment_eq_weighted (h : ℝ) (p q : V) (t : ℝ) :
    trajectorySegment h p q t = (1 - t / h) • p + (t / h) • q := by
  simp only [trajectorySegment, sub_smul, one_smul, smul_sub]
  abel

/-- The interpolation error between two stored values and a reference point,
including the reference displacement from both ends of the time cell. -/
theorem trajectorySegment_error (h s error B : ℝ) (p q z z₀ z₁ : V)
    (hh : 0 < h) (hs : s ∈ Icc (0 : ℝ) h) (hB : 0 ≤ B)
    (hp : ‖p - z₀‖ ≤ error) (hq : ‖q - z₁‖ ≤ error)
    (hz₀ : ‖z₀ - z‖ ≤ B * s) (hz₁ : ‖z₁ - z‖ ≤ B * (h - s)) :
    ‖trajectorySegment h p q s - z‖ ≤ error + B * h / 2 := by
  have ht : s / h ∈ Icc (0 : ℝ) 1 := ⟨div_nonneg hs.1 hh.le, (div_le_one hh).2 hs.2⟩
  have heq : trajectorySegment h p q s - z =
      (1 - s / h) • (p - z) + (s / h) • (q - z) := by
    rw [trajectorySegment_eq_weighted]
    module
  have hpz : ‖p - z‖ ≤ error + B * s :=
    (norm_sub_le_norm_sub_add_norm_sub p z₀ z).trans (add_le_add hp hz₀)
  have hqz : ‖q - z‖ ≤ error + B * (h - s) :=
    (norm_sub_le_norm_sub_add_norm_sub q z₁ z).trans (add_le_add hq hz₁)
  have hnorm : ‖trajectorySegment h p q s - z‖ ≤
      (1 - s / h) * ‖p - z‖ + (s / h) * ‖q - z‖ := by
    simpa only [heq, norm_smul, Real.norm_eq_abs, abs_of_nonneg ht.1,
      abs_of_nonneg (sub_nonneg.mpr ht.2)] using
      norm_add_le ((1 - s / h) • (p - z)) ((s / h) • (q - z))
  have hfinal := hnorm.trans (add_le_add
    (mul_le_mul_of_nonneg_left hpz (sub_nonneg.mpr ht.2))
    (mul_le_mul_of_nonneg_left hqz ht.1))
  have hquot : s / h * h = s := div_mul_cancel₀ s hh.ne'
  nlinarith [mul_nonneg (mul_nonneg hB hh.le) (sq_nonneg (s / h - 1 / 2)),
    congrArg (fun x => B * x) hquot, congrArg (fun x => B * (s / h) * x) hquot]

theorem trajectoryInterpolant_endpoint (h : ℝ) (y : ℕ → V) (hh : 0 < h) (n : ℕ) :
    trajectoryInterpolant h y n ((n : ℝ) * h) = y n := by
  cases n with
  | zero => rfl
  | succ n =>
    simp only [trajectoryInterpolant, Nat.cast_add, Nat.cast_one]
    rw [if_neg (by nlinarith : ¬ ((n : ℝ) + 1) * h ≤ (n : ℝ) * h)]
    simp [trajectorySegment, add_mul, hh.ne']

theorem continuous_trajectoryInterpolant (h : ℝ) (y : ℕ → V) (hh : 0 < h) (n : ℕ) :
    Continuous (trajectoryInterpolant h y n) := by
  induction n with
  | zero => exact continuous_const
  | succ n ih =>
    apply ih.if_le ?_ continuous_id continuous_const
    · intro x hx
      change x = (n : ℝ) * h at hx
      simpa [hx, trajectorySegment] using trajectoryInterpolant_endpoint h y hh n
    · unfold trajectorySegment
      fun_prop

/-- Every stored state is reproduced at its actual grid time. -/
theorem trajectoryInterpolant_grid (h : ℝ) (y : ℕ → V) (hh : 0 < h) (n : ℕ) :
    ∀ i ≤ n, trajectoryInterpolant h y n ((i : ℝ) * h) = y i := by
  induction n with
  | zero =>
    intro i hi
    simpa only [Nat.le_zero.mp hi] using trajectoryInterpolant_endpoint h y hh 0
  | succ n ih =>
    intro i hi
    by_cases hin : i ≤ n
    · simpa only [trajectoryInterpolant,
        if_pos (mul_le_mul_of_nonneg_right (Nat.cast_le.mpr hin : (i : ℝ) ≤ n) hh.le)] using ih i hin
    · have heq : i = n + 1 := by omega
      simpa only [heq] using trajectoryInterpolant_endpoint h y hh (n + 1)

/-- The continuous interpolant inherits a uniform grid error, with an
additional half-cell displacement controlled by the actual reference speed. -/
theorem trajectoryInterpolant_error (h : ℝ) (y : ℕ → V) (z : ℝ → V)
    (error : ℝ) (B : ℝ≥0) (hh : 0 < h) (n : ℕ)
    (hgrid : ∀ i ≤ n, ‖y i - z ((i : ℝ) * h)‖ ≤ error)
    (hLip : LipschitzOnWith B z (Icc (0 : ℝ) ((n : ℝ) * h))) :
    ∀ t ∈ Icc (0 : ℝ) ((n : ℝ) * h),
      ‖trajectoryInterpolant h y n t - z t‖ ≤ error + B * h / 2 := by
  induction n with
  | zero =>
    intro t ht
    have ht0 : t = 0 := by simpa using ht
    simpa [trajectoryInterpolant, ht0] using (hgrid 0 le_rfl).trans
      (le_add_of_nonneg_right (div_nonneg (mul_nonneg B.coe_nonneg hh.le) (by norm_num)))
  | succ n ih =>
    intro t ht
    by_cases hcell : t ≤ (n : ℝ) * h
    · simp only [trajectoryInterpolant, if_pos hcell]
      exact ih (fun i hi => hgrid i (hi.trans (Nat.le_succ n)))
        (hLip.mono (Icc_subset_Icc_right
          (mul_le_mul_of_nonneg_right (by exact_mod_cast Nat.le_succ n) hh.le))) t ⟨ht.1, hcell⟩
    · simp only [trajectoryInterpolant, if_neg hcell]
      have hs : t - (n : ℝ) * h ∈ Icc (0 : ℝ) h := by
        constructor <;> push_cast at ht <;> linarith [ht.2]
      have hn : (n : ℝ) * h ∈ Icc (0 : ℝ) ((↑(n + 1) : ℝ) * h) :=
        ⟨mul_nonneg (Nat.cast_nonneg n) hh.le,
          mul_le_mul_of_nonneg_right (by exact_mod_cast Nat.le_succ n) hh.le⟩
      have hn1 : (↑(n + 1) : ℝ) * h ∈ Icc (0 : ℝ) ((↑(n + 1) : ℝ) * h) :=
        ⟨mul_nonneg (Nat.cast_nonneg (n + 1)) hh.le, le_rfl⟩
      apply trajectorySegment_error h (t - (n : ℝ) * h) error B (y n) (y (n + 1))
        (z t) (z ((n : ℝ) * h)) (z ((↑(n + 1) : ℝ) * h)) hh hs B.coe_nonneg
        (hgrid n (Nat.le_succ n)) (hgrid (n + 1) le_rfl)
      · simpa only [dist_eq_norm, Real.norm_eq_abs,
          abs_of_nonpos (sub_nonpos.mpr (not_le.mp hcell).le), neg_sub] using
          hLip.dist_le_mul ((n : ℝ) * h) hn t ht
      · have hnear := hLip.dist_le_mul ((↑(n + 1) : ℝ) * h) hn1 t ht
        simp only [dist_eq_norm, Real.norm_eq_abs, abs_of_nonneg (sub_nonneg.mpr ht.2)] at hnear
        simpa only [Nat.cast_add, Nat.cast_one, add_mul, one_mul, sub_sub_eq_add_sub,
          add_comm h] using hnear

end Normed

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

noncomputable local instance trajectoryInterpolationNormedSpace : NormedSpace ℝ (E × E) :=
  Prod.normedSpace

/-- The speed certificate gives actual reference displacement bounds between
all times of the interval, not only grid times. -/
theorem UniformODECertificate.reference_lipschitz {acc : E × E → E} {z : ℝ → E × E}
    {T : ℝ} {K : Set (E × E)} (c : UniformODECertificate acc z T K) :
    LipschitzOnWith c.phaseSpeed z (Icc (0 : ℝ) T) := by
  exact (convex_Icc (0 : ℝ) T).lipschitzOnWith_of_nnnorm_hasDerivWithin_le
    (fun s hs => (c.equation s hs).hasDerivWithinAt) (fun s hs => c.speed_bound s hs)

/-- A computable inequality in the supplied regional constants controls the
entire interpolated history and establishes its phase-domain membership. -/
theorem UniformODECertificate.interpolation_error_bound {acc : E × E → E} {z : ℝ → E × E}
    {T : ℝ} {K : Set (E × E)} (c : UniformODECertificate acc z T K) (hT : 0 < T)
    (N : ℕ) (y : ℕ → E × E) (A : ℕ → E)
    (residualCoeff roundingCoeff initialCoeff : ℝ≥0) (radius : ℝ)
    (hinit : ‖y 0 - z 0‖ ≤ refinementStep T N * initialCoeff)
    (hres : ∀ i < N + 1, y i ∈ K →
      ‖A i - acc (y i)‖ ≤ refinementStep T N * residualCoeff)
    (hround : ∀ i < N + 1, y i ∈ K →
      ‖y (i + 1) - positionVelocityStep (refinementStep T N) (y i).1 (y i).2 (A i)‖ ≤
        (refinementStep T N) ^ 2 * roundingCoeff)
    (htube : ∀ s ∈ Icc (0 : ℝ) T, ∀ w : E × E, ‖w - z s‖ ≤ radius → w ∈ K)
    (hbudget : refinementStep T N *
      (refinementCoefficient T c.accelerationLip c.phaseSpeed residualCoeff roundingCoeff initialCoeff +
        c.phaseSpeed / 2) ≤ radius) :
    ∀ t ∈ Icc (0 : ℝ) T,
      ‖trajectoryInterpolant (refinementStep T N) y (N + 1) t - z t‖ ≤
        refinementStep T N *
          (refinementCoefficient T c.accelerationLip c.phaseSpeed residualCoeff roundingCoeff initialCoeff +
            c.phaseSpeed / 2) ∧
      trajectoryInterpolant (refinementStep T N) y (N + 1) t ∈ K := by
  have hh : 0 < refinementStep T N := div_pos hT (by positivity)
  let cn := c.onRefinement hT.le N
  let eta : ℝ≥0 := ⟨refinementStep T N * residualCoeff, mul_nonneg hh.le residualCoeff.coe_nonneg⟩
  let rounding : ℝ≥0 := ⟨(refinementStep T N) ^ 2 * roundingCoeff,
    mul_nonneg (sq_nonneg _) roundingCoeff.coe_nonneg⟩
  have hfinal : cn.errorEnvelope eta rounding (refinementStep T N * initialCoeff) (N + 1) ≤
      refinementStep T N * refinementCoefficient T c.accelerationLip c.phaseSpeed
        residualCoeff roundingCoeff initialCoeff :=
    trajectoryEnvelope_fixed_horizon_bound T (refinementStep T N) c.accelerationLip c.phaseSpeed
      residualCoeff roundingCoeff initialCoeff (N + 1) hh.le
      (refinementStep_properties T N hT.le).2.1 c.accelerationLip.coe_nonneg c.phaseSpeed.coe_nonneg
      residualCoeff.coe_nonneg roundingCoeff.coe_nonneg initialCoeff.coe_nonneg
      (refinementStep_properties T N hT.le).2.2
  have hfit : cn.errorEnvelope eta rounding (refinementStep T N * initialCoeff) (N + 1) ≤
      radius := by nlinarith [mul_nonneg hh.le c.phaseSpeed.coe_nonneg]
  have htubeN : ∀ i ≤ N + 1, ∀ w : E × E,
      ‖w - z ((i : ℝ) * refinementStep T N)‖ ≤
        cn.errorEnvelope eta rounding (refinementStep T N * initialCoeff) (N + 1) → w ∈ K := by
    intro i hi w hw
    apply htube ((i : ℝ) * refinementStep T N) ?_ w (hw.trans hfit)
    exact ⟨mul_nonneg (Nat.cast_nonneg i) hh.le,
      (mul_le_mul_of_nonneg_right (Nat.cast_le.mpr hi) hh.le).trans_eq
        (refinementStep_properties T N hT.le).2.2⟩
  have hrun := cn.trajectory_error_domain y A eta rounding
    (refinementStep T N * initialCoeff) hh.le
    (by simpa only [Nat.cast_zero, zero_mul] using hinit) hres hround htubeN
  have hmono : Monotone (cn.errorEnvelope eta rounding (refinementStep T N * initialCoeff)) :=
    trajectoryEnvelope_mono _ _ _
      (one_le_stepAmplification _ cn.accelerationLip cn.accelerationLip.coe_nonneg)
      (localTrajectoryBudget_nonneg _ cn.accelerationLip cn.phaseSpeed eta rounding
        cn.phaseSpeed.coe_nonneg eta.coe_nonneg rounding.coe_nonneg)
      (mul_nonneg hh.le initialCoeff.coe_nonneg)
  have hbound := trajectoryInterpolant_error (refinementStep T N) y z
    (refinementStep T N * refinementCoefficient T c.accelerationLip c.phaseSpeed
      residualCoeff roundingCoeff initialCoeff) c.phaseSpeed hh (N + 1)
    (fun i hi => ((hrun i hi).1.trans (hmono hi)).trans hfinal)
    (by simpa only [(refinementStep_properties T N hT.le).2.2] using c.reference_lipschitz)
  rw [(refinementStep_properties T N hT.le).2.2] at hbound
  intro t ht
  have herr : ‖trajectoryInterpolant (refinementStep T N) y (N + 1) t - z t‖ ≤
      refinementStep T N *
        (refinementCoefficient T c.accelerationLip c.phaseSpeed residualCoeff roundingCoeff initialCoeff +
          c.phaseSpeed / 2) := by nlinarith [hbound t ht]
  exact ⟨herr, htube t ht _ (herr.trans hbudget)⟩

/-- Actual piecewise-affine stored-state trajectories converge uniformly at
all times. The exact-curve tube establishes off-grid domain membership too,
without requiring the region to be convex. -/
theorem UniformODECertificate.interpolation_converges {acc : E × E → E} {z : ℝ → E × E}
    {T : ℝ} {K : Set (E × E)} (c : UniformODECertificate acc z T K) (hT : 0 < T)
    (y : ℕ → ℕ → E × E) (A : ℕ → ℕ → E)
    (residualCoeff roundingCoeff initialCoeff : ℝ≥0) (radius : ℝ) (hradius : 0 < radius)
    (hinit : ∀ N, ‖y N 0 - z 0‖ ≤ refinementStep T N * initialCoeff)
    (hres : ∀ N i, i < N + 1 → y N i ∈ K →
      ‖A N i - acc (y N i)‖ ≤ refinementStep T N * residualCoeff)
    (hround : ∀ N i, i < N + 1 → y N i ∈ K →
      ‖y N (i + 1) - positionVelocityStep (refinementStep T N) (y N i).1 (y N i).2 (A N i)‖ ≤
        (refinementStep T N) ^ 2 * roundingCoeff)
    (htube : ∀ s ∈ Icc (0 : ℝ) T, ∀ w : E × E, ‖w - z s‖ ≤ radius → w ∈ K) :
    ∀ epsilon > 0, ∀ᶠ N in atTop,
      Continuous (trajectoryInterpolant (refinementStep T N) (y N) (N + 1)) ∧
      ∀ t ∈ Icc (0 : ℝ) T,
        ‖trajectoryInterpolant (refinementStep T N) (y N) (N + 1) t - z t‖ < epsilon ∧
        trajectoryInterpolant (refinementStep T N) (y N) (N + 1) t ∈ K := by
  intro epsilon hepsilon
  have hdelta : 0 < min radius epsilon := lt_min hradius hepsilon
  have hg := c.refinement_converges hT.le y A residualCoeff roundingCoeff initialCoeff
    radius hradius hinit hres hround htube (min radius epsilon / 2) (half_pos hdelta)
  have htime : Tendsto (fun N => (c.phaseSpeed : ℝ) * refinementStep T N / 2) atTop (𝓝 0) := by
    simpa using ((tendsto_refinementStep_zero T).const_mul (c.phaseSpeed : ℝ)).div_const 2
  filter_upwards [hg, htime.eventually (eventually_lt_nhds (half_pos hdelta))] with N hgN htN
  have hh : 0 < refinementStep T N := div_pos hT (by positivity)
  have hbound := trajectoryInterpolant_error (refinementStep T N) (y N) z
    (min radius epsilon / 2) c.phaseSpeed hh (N + 1) (fun i hi => (hgN i hi).1.le)
    (by simpa only [(refinementStep_properties T N hT.le).2.2] using c.reference_lipschitz)
  rw [(refinementStep_properties T N hT.le).2.2] at hbound
  refine ⟨continuous_trajectoryInterpolant _ _ hh _, fun t ht => ?_⟩
  have herr : ‖trajectoryInterpolant (refinementStep T N) (y N) (N + 1) t - z t‖ <
      min radius epsilon := by linarith [hbound t ht]
  exact ⟨herr.trans_le (min_le_right radius epsilon),
    htube t ht _ (herr.le.trans (min_le_left radius epsilon))⟩

end FractalSync.ScaleMetric
