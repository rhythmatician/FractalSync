import FractalSync.AccumulatedDynamics
import Mathlib.Analysis.Calculus.Deriv.MeanValue

/-!
# Actual ordered-step stability and finite-horizon trajectory error

The phase-space max norm is an error-control norm. Local defects are derived
from actual ODE derivatives, and provider/rounding errors enter as explicit
residuals. These statements do not certify any floating-point implementation.
-/

open Set
open scoped InnerProductSpace BigOperators NNReal

namespace FractalSync.ScaleMetric

section CurveRemainder

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

noncomputable local instance trajectoryCurveNormedSpace : NormedSpace ℝ (E × E) := Prod.normedSpace

/-- The actual first-order phase field associated with an acceleration field. -/
def phaseField (acc : E × E → E) (z : E × E) : E × E := (z.2, acc z)

/-- A vector-valued quadratic remainder derived from actual curve derivatives
and their variation along a nonnegative time segment. -/
theorem curve_quadratic_remainder (f f' : ℝ → E) (t h M : ℝ) (hh : 0 ≤ h)
    (hd : ∀ s ∈ Icc t (t + h), HasDerivAt f (f' s) s)
    (hbound : ∀ s ∈ Icc t (t + h), ‖f' s - f' t‖ ≤ M * (s - t)) :
    ‖f (t + h) - f t - h • f' t‖ ≤ M / 2 * h ^ 2 := by
  have hg : ∀ s ∈ Icc t (t + h),
      HasDerivAt (fun u => f u - f t - (u - t) • f' t) (f' s - f' t) s := by
    intro s hs
    simpa using ((hd s hs).sub_const (f t)).sub
      (((hasDerivAt_id s).sub_const t).smul_const (f' t))
  have hB : ∀ s : ℝ, HasDerivAt (fun u => M / 2 * (u - t) ^ 2) (M * (s - t)) s := by
    intro s
    convert (((hasDerivAt_id s).sub_const t).pow 2).const_mul (M / 2) using 1
    dsimp
    ring
  have hb := image_norm_le_of_norm_deriv_right_le_deriv_boundary
    (B := fun s => M / 2 * (s - t) ^ 2)
    (fun s hs => (hg s hs).continuousAt.continuousWithinAt)
    (fun s hs => (hg s (Ico_subset_Icc_self hs)).hasDerivWithinAt)
    (by simp) hB (fun s hs => hbound s (Ico_subset_Icc_self hs))
    (show t + h ∈ Icc t (t + h) from ⟨le_add_of_nonneg_right hh, le_rfl⟩)
  simpa using hb

end CurveRemainder

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

noncomputable local instance trajectoryPhaseNormedSpace : NormedSpace ℝ (E × E) := Prod.normedSpace

/-- The actual ordered update evaluated at the acceleration field. -/
def orderedStep (acc : E × E → E) (h : ℝ) (z : E × E) : E × E :=
  positionVelocityStep h z.1 z.2 (acc z)

/-- Exact phase increment of the existing velocity-then-position update. -/
theorem orderedStep_eq (acc : E × E → E) (h : ℝ) (z : E × E) :
    orderedStep acc h z = z + h • phaseField acc z + h ^ 2 • (acc z, 0) := by
  rw [orderedStep, positionVelocityStep_eq_add, positionVelocityStep_increment]
  exact (add_assoc _ _ _).symm

omit [InnerProductSpace ℝ E] in
/-- Acceleration regularity controls the actual first-order phase field. -/
theorem phaseField_sub_le (acc : E × E → E) (z w : E × E) (L : ℝ)
    (hacc : ‖acc z - acc w‖ ≤ L * ‖z - w‖) :
    ‖phaseField acc z - phaseField acc w‖ ≤ max 1 L * ‖z - w‖ := by
  change max ‖z.2 - w.2‖ ‖acc z - acc w‖ ≤ max 1 L * ‖z - w‖
  refine max_le ?_ ?_
  · exact (norm_snd_le (z - w)).trans
      (by nlinarith [le_max_left (1 : ℝ) L, norm_nonneg (z - w)])
  · exact hacc.trans (mul_le_mul_of_nonneg_right (le_max_right (1 : ℝ) L) (norm_nonneg _))

def stepAmplification (h L : ℝ) : ℝ := 1 + |h| * max 1 L + h ^ 2 * L

theorem one_le_stepAmplification (h L : ℝ) (hL : 0 ≤ L) : 1 ≤ stepAmplification h L := by
  unfold stepAmplification
  have hm : 0 ≤ max 1 L := le_trans zero_le_one (le_max_left 1 L)
  nlinarith [mul_nonneg (abs_nonneg h) hm, mul_nonneg (sq_nonneg h) hL]

/-- Stability is derived from the actual ordered update, including its h²
position correction. Only the acceleration difference is assumed controlled. -/
theorem orderedStep_stability (acc : E × E → E) (h L : ℝ) (z w : E × E)
    (hacc : ‖acc z - acc w‖ ≤ L * ‖z - w‖) :
    ‖orderedStep acc h z - orderedStep acc h w‖ ≤ stepAmplification h L * ‖z - w‖ := by
  rw [orderedStep_eq, orderedStep_eq]
  have heq : (z + h • phaseField acc z + h ^ 2 • (acc z, 0)) -
      (w + h • phaseField acc w + h ^ 2 • (acc w, 0)) =
      (z - w) + h • (phaseField acc z - phaseField acc w) + h ^ 2 • (acc z - acc w, 0) := by
    rw [show (acc z - acc w, (0 : E)) = (acc z, 0) - (acc w, 0) by simp]
    simp only [smul_sub]
    abel
  rw [heq]
  have hn := (norm_add_le ((z - w) + h • (phaseField acc z - phaseField acc w))
    (h ^ 2 • (acc z - acc w, (0 : E)))).trans
      (add_le_add_right (norm_add_le (z - w) (h • (phaseField acc z - phaseField acc w))) _)
  have hp : ‖(acc z - acc w, (0 : E))‖ = ‖acc z - acc w‖ := by simp
  simp only [norm_smul, Real.norm_eq_abs, abs_sq, hp] at hn
  have hF := mul_le_mul_of_nonneg_left (phaseField_sub_le acc z w L hacc) (abs_nonneg h)
  have hA := mul_le_mul_of_nonneg_left hacc (sq_nonneg h)
  unfold stepAmplification
  nlinarith

/-- A defect against an actual ODE curve, derived from actual curve derivatives,
an acceleration Lipschitz bound, and a phase-speed bound along the step. -/
theorem orderedStep_ode_defect (acc : E × E → E) (z : ℝ → E × E)
    (t h L B : ℝ) (hh : 0 ≤ h)
    (hODE : ∀ s ∈ Icc t (t + h), HasDerivAt z (phaseField acc (z s)) s)
    (hB : ∀ s ∈ Icc t (t + h), ‖phaseField acc (z s)‖ ≤ B)
    (hLip : ∀ s ∈ Icc t (t + h), ‖acc (z s) - acc (z t)‖ ≤ L * ‖z s - z t‖) :
    ‖orderedStep acc h (z t) - z (t + h)‖ ≤ h ^ 2 * (B + max 1 L * B / 2) := by
  have hmove : ∀ s ∈ Icc t (t + h), ‖z s - z t‖ ≤ B * (s - t) :=
    norm_image_sub_le_of_norm_deriv_le_segment'
      (fun s hs => (hODE s hs).hasDerivWithinAt)
      (fun s hs => hB s (Ico_subset_Icc_self hs))
  have hvar : ∀ s ∈ Icc t (t + h),
      ‖phaseField acc (z s) - phaseField acc (z t)‖ ≤ (max 1 L * B) * (s - t) := by
    intro s hs
    have hm : 0 ≤ max 1 L := le_trans zero_le_one (le_max_left 1 L)
    exact (phaseField_sub_le acc (z s) (z t) L (hLip s hs)).trans
      (by simpa only [mul_assoc] using mul_le_mul_of_nonneg_left (hmove s hs) hm)
  have hR := curve_quadratic_remainder z (fun s => phaseField acc (z s))
    t h (max 1 L * B) hh hODE hvar
  have heq : orderedStep acc h (z t) - z (t + h) =
      h ^ 2 • (acc (z t), 0) - (z (t + h) - z t - h • phaseField acc (z t)) := by
    rw [orderedStep_eq]
    abel
  rw [heq]
  have hA : ‖acc (z t)‖ ≤ B := (norm_snd_le (phaseField acc (z t))).trans
    (hB t ⟨le_rfl, le_add_of_nonneg_right hh⟩)
  have hn := norm_sub_le (h ^ 2 • (acc (z t), (0 : E)))
    (z (t + h) - z t - h • phaseField acc (z t))
  have hp : ‖(acc (z t), (0 : E))‖ = ‖acc (z t)‖ := by simp
  simp only [norm_smul, Real.norm_eq_abs, abs_sq, hp] at hn
  nlinarith [mul_le_mul_of_nonneg_left hA (sq_nonneg h)]

/-- The actual ordered update amplifies an acceleration evaluation error by
at most |h|+h² in the phase max norm. -/
theorem positionVelocityStep_acceleration_error (h : ℝ) (x v A B : E) :
    ‖positionVelocityStep h x v A - positionVelocityStep h x v B‖ ≤
      (|h| + h ^ 2) * ‖A - B‖ := by
  have heq : positionVelocityStep h x v A - positionVelocityStep h x v B =
      (h ^ 2 • (A - B), h • (A - B)) := by
    apply Prod.ext <;> dsimp [positionVelocityStep, velocityEulerStep] <;> module
  rw [heq, Prod.norm_def]
  simp only [norm_smul, Real.norm_eq_abs, abs_sq]
  apply max_le
  · nlinarith [mul_nonneg (abs_nonneg h) (norm_nonneg (A - B))]
  · nlinarith [mul_nonneg (sq_nonneg h) (norm_nonneg (A - B))]

noncomputable def localTrajectoryBudget (h L B eta rounding : ℝ) : ℝ :=
  (|h| + h ^ 2) * eta + rounding + h ^ 2 * (B + max 1 L * B / 2)

/-- One actual rounded step compared with an actual ODE endpoint. Evaluation
residuals and stored-state rounding enter separately from the derived defect. -/
theorem roundedStep_ode_error (acc : E × E → E) (z : ℝ → E × E)
    (t h L B eta rounding : ℝ) (y next : E × E) (A : E) (hh : 0 ≤ h)
    (hODE : ∀ s ∈ Icc t (t + h), HasDerivAt z (phaseField acc (z s)) s)
    (hB : ∀ s ∈ Icc t (t + h), ‖phaseField acc (z s)‖ ≤ B)
    (hLip : ∀ s ∈ Icc t (t + h), ‖acc (z s) - acc (z t)‖ ≤ L * ‖z s - z t‖)
    (hstable : ‖acc y - acc (z t)‖ ≤ L * ‖y - z t‖)
    (hres : ‖A - acc y‖ ≤ eta)
    (hround : ‖next - positionVelocityStep h y.1 y.2 A‖ ≤ rounding) :
    ‖next - z (t + h)‖ ≤
      stepAmplification h L * ‖y - z t‖ + localTrajectoryBudget h L B eta rounding := by
  have hpert := (positionVelocityStep_acceleration_error h y.1 y.2 A (acc y)).trans
    (mul_le_mul_of_nonneg_left hres (add_nonneg (abs_nonneg h) (sq_nonneg h)))
  have hstable' := orderedStep_stability acc h L y (z t) hstable
  have hlocal := orderedStep_ode_defect acc z t h L B hh hODE hB hLip
  change ‖positionVelocityStep h y.1 y.2 A - orderedStep acc h y‖ ≤ (|h| + h ^ 2) * eta at hpert
  have htri1 := dist_triangle next (positionVelocityStep h y.1 y.2 A) (orderedStep acc h y)
  have htri2 := dist_triangle next (orderedStep acc h y) (orderedStep acc h (z t))
  have htri3 := dist_triangle next (orderedStep acc h (z t)) (z (t + h))
  simp only [dist_eq_norm] at htri1 htri2 htri3
  unfold localTrajectoryBudget
  linarith

/-- A finite-error envelope, with no division by kappa-1 at kappa=1. -/
def trajectoryEnvelope (kappa delta initial : ℝ) : ℕ → ℝ
  | 0 => initial
  | n + 1 => kappa * trajectoryEnvelope kappa delta initial n + delta

theorem trajectoryEnvelope_eq (kappa delta initial : ℝ) (n : ℕ) :
    trajectoryEnvelope kappa delta initial n =
      kappa ^ n * initial + delta * ∑ j ∈ Finset.range n, kappa ^ j := by
  induction n with
  | zero => simp [trajectoryEnvelope]
  | succ n ih =>
    rw [trajectoryEnvelope, ih, Finset.sum_range_succ']
    simp only [pow_succ, pow_zero, ← Finset.sum_mul]
    ring

theorem trajectoryEnvelope_nonneg (kappa delta initial : ℝ)
    (hk : 0 ≤ kappa) (hd : 0 ≤ delta) (hi : 0 ≤ initial) (n : ℕ) :
    0 ≤ trajectoryEnvelope kappa delta initial n := by
  induction n with
  | zero => exact hi
  | succ n ih => exact add_nonneg (mul_nonneg hk ih) hd

theorem trajectoryEnvelope_mono (kappa delta initial : ℝ)
    (hk : 1 ≤ kappa) (hd : 0 ≤ delta) (hi : 0 ≤ initial) :
    Monotone (trajectoryEnvelope kappa delta initial) := by
  apply monotone_nat_of_le_succ
  intro n
  rw [trajectoryEnvelope]
  nlinarith [trajectoryEnvelope_nonneg kappa delta initial (zero_le_one.trans hk) hd hi n]

theorem localTrajectoryBudget_nonneg (h L B eta rounding : ℝ)
    (hB : 0 ≤ B) (he : 0 ≤ eta) (hr : 0 ≤ rounding) :
    0 ≤ localTrajectoryBudget h L B eta rounding := by
  have hm : 0 ≤ max 1 L := zero_le_one.trans (le_max_left 1 L)
  unfold localTrajectoryBudget
  positivity

/-- Actual ODE and field data on a finite family of time intervals. This
record contains no local truncation estimate or numerical-trajectory bound. -/
structure ODERegionCertificate (acc : E × E → E) (z : ℝ → E × E)
    (time : ℕ → ℝ) (h : ℝ) (N : ℕ) (K : Set (E × E)) where
  accelerationLip : ℝ≥0
  phaseSpeed : ℝ≥0
  grid : ∀ i < N, time (i + 1) = time i + h
  equation : ∀ i < N, ∀ s ∈ Icc (time i) (time i + h),
    HasDerivAt z (phaseField acc (z s)) s
  reference_mem : ∀ i < N, ∀ s ∈ Icc (time i) (time i + h), z s ∈ K
  speed_bound : ∀ i < N, ∀ s ∈ Icc (time i) (time i + h),
    ‖phaseField acc (z s)‖ ≤ phaseSpeed
  acceleration_lipschitz : ∀ p ∈ K, ∀ q ∈ K,
    ‖acc p - acc q‖ ≤ accelerationLip * ‖p - q‖

theorem ODERegionCertificate.step_error {acc : E × E → E} {z : ℝ → E × E}
    {time : ℕ → ℝ} {h : ℝ} {N : ℕ} {K : Set (E × E)}
    (c : ODERegionCertificate acc z time h N K) (i : ℕ) (hi : i < N)
    (eta rounding : ℝ) (y next : E × E) (A : E) (hh : 0 ≤ h) (hy : y ∈ K)
    (hres : ‖A - acc y‖ ≤ eta)
    (hround : ‖next - positionVelocityStep h y.1 y.2 A‖ ≤ rounding) :
    ‖next - z (time (i + 1))‖ ≤ stepAmplification h c.accelerationLip * ‖y - z (time i)‖ +
      localTrajectoryBudget h c.accelerationLip c.phaseSpeed eta rounding := by
  have ht : time i ∈ Icc (time i) (time i + h) := ⟨le_rfl, le_add_of_nonneg_right hh⟩
  rw [c.grid i hi]
  exact roundedStep_ode_error acc z (time i) h c.accelerationLip c.phaseSpeed
    eta rounding y next A hh (c.equation i hi) (c.speed_bound i hi)
    (fun s hs => c.acceleration_lipschitz (z s) (c.reference_mem i hi s hs)
      (z (time i)) (c.reference_mem i hi (time i) ht))
    (c.acceleration_lipschitz y hy (z (time i)) (c.reference_mem i hi (time i) ht)) hres hround

noncomputable def ODERegionCertificate.errorEnvelope {acc : E × E → E} {z : ℝ → E × E}
    {time : ℕ → ℝ} {h : ℝ} {N : ℕ} {K : Set (E × E)}
    (c : ODERegionCertificate acc z time h N K) (eta rounding initial : ℝ) (n : ℕ) : ℝ :=
  trajectoryEnvelope (stepAmplification h c.accelerationLip)
    (localTrajectoryBudget h c.accelerationLip c.phaseSpeed eta rounding) initial n

/-- Simultaneous trajectory-error and domain control. Only points in the
certified region need provider/rounding contracts; the conclusion establishes
that the numerical history stays in that region from an exact-curve margin. -/
theorem ODERegionCertificate.trajectory_error_domain {acc : E × E → E} {z : ℝ → E × E}
    {time : ℕ → ℝ} {h : ℝ} {N : ℕ} {K : Set (E × E)}
    (c : ODERegionCertificate acc z time h N K) (y : ℕ → E × E) (A : ℕ → E)
    (eta rounding : ℝ≥0) (initial : ℝ) (hh : 0 ≤ h)
    (hinit : ‖y 0 - z (time 0)‖ ≤ initial)
    (hres : ∀ i < N, y i ∈ K → ‖A i - acc (y i)‖ ≤ eta)
    (hround : ∀ i < N, y i ∈ K →
      ‖y (i + 1) - positionVelocityStep h (y i).1 (y i).2 (A i)‖ ≤ rounding)
    (htube : ∀ i ≤ N, ∀ w : E × E,
      ‖w - z (time i)‖ ≤ c.errorEnvelope eta rounding initial N → w ∈ K) :
    ∀ i ≤ N, ‖y i - z (time i)‖ ≤ c.errorEnvelope eta rounding initial i ∧ y i ∈ K := by
  have hi0 : 0 ≤ initial := (norm_nonneg _).trans hinit
  have hk := one_le_stepAmplification h c.accelerationLip c.accelerationLip.coe_nonneg
  have hd := localTrajectoryBudget_nonneg h c.accelerationLip c.phaseSpeed eta rounding
    c.phaseSpeed.coe_nonneg eta.coe_nonneg rounding.coe_nonneg
  have hmono : Monotone (c.errorEnvelope eta rounding initial) :=
    trajectoryEnvelope_mono _ _ _ hk hd hi0
  intro i
  induction i with
  | zero =>
    intro hi
    have herr : ‖y 0 - z (time 0)‖ ≤ c.errorEnvelope eta rounding initial 0 := hinit
    exact ⟨herr, htube 0 hi (y 0) (herr.trans (hmono hi))⟩
  | succ i ih =>
    intro hi
    have hiN : i < N := Nat.lt_of_lt_of_le (Nat.lt_succ_self i) hi
    obtain ⟨herr, hmem⟩ := ih hiN.le
    have hstep := c.step_error i hiN eta rounding (y i) (y (i + 1)) (A i) hh hmem
      (hres i hiN hmem) (hround i hiN hmem)
    have hnext : ‖y (i + 1) - z (time (i + 1))‖ ≤
        c.errorEnvelope eta rounding initial (i + 1) := by
      exact hstep.trans (add_le_add_right
        (mul_le_mul_of_nonneg_left herr (zero_le_one.trans hk)) _)
    exact ⟨hnext, htube (i + 1) hi (y (i + 1)) (hnext.trans (hmono hi))⟩

/-- Actual acceleration derivatives on a convex region supply the field
Lipschitz hypothesis used by the stability and truncation theorems. -/
theorem acceleration_sub_le_of_derivative_bound (acc : E × E → E)
    (D : E × E → (E × E) →L[ℝ] E) (K : Set (E × E)) (L : ℝ)
    (hconv : Convex ℝ K) (hd : ∀ z ∈ K, HasFDerivAt acc (D z) z)
    (hbound : ∀ z ∈ K, ‖D z‖ ≤ L) (p q : E × E) (hp : p ∈ K) (hq : q ∈ K) :
    ‖acc p - acc q‖ ≤ L * ‖p - q‖ := by
  exact hconv.norm_image_sub_le_of_norm_hasFDerivWithin_le
    (fun z hz => (hd z hz).hasFDerivWithinAt) hbound hq hp

/-- Position and velocity may have separate regional radii. -/
def phaseBox (positionRadius velocityRadius : ℝ) : Set (E × E) :=
  {z | ‖z.1‖ < positionRadius ∧ ‖z.2‖ < velocityRadius}

omit [InnerProductSpace ℝ E] in
/-- An explicit exact-reference margin turns a phase-error bound into both
position-domain and velocity-domain membership. -/
theorem phaseBox_mem_of_error (reference candidate : E × E)
    (error positionRadius velocityRadius : ℝ)
    (herr : ‖candidate - reference‖ ≤ error)
    (hx : ‖reference.1‖ + error < positionRadius)
    (hv : ‖reference.2‖ + error < velocityRadius) :
    candidate ∈ phaseBox positionRadius velocityRadius := by
  have hfirst := (norm_fst_le (candidate - reference)).trans herr
  have hsecond := (norm_snd_le (candidate - reference)).trans herr
  change ‖candidate.1 - reference.1‖ ≤ error at hfirst
  change ‖candidate.2 - reference.2‖ ≤ error at hsecond
  have hxf : ‖candidate.1‖ ≤ ‖candidate.1 - reference.1‖ + ‖reference.1‖ := by
    simpa using norm_add_le (candidate.1 - reference.1) reference.1
  have hvf : ‖candidate.2‖ ≤ ‖candidate.2 - reference.2‖ + ‖reference.2‖ := by
    simpa using norm_add_le (candidate.2 - reference.2) reference.2
  exact ⟨by linarith, by linarith⟩

/-- Concrete position/velocity margins discharge the abstract domain tube.
This proves both numerical phase-region membership and trajectory error. -/
theorem ODERegionCertificate.trajectory_error_phaseBox {acc : E × E → E} {z : ℝ → E × E}
    {time : ℕ → ℝ} {h : ℝ} {N : ℕ} {positionRadius velocityRadius : ℝ}
    (c : ODERegionCertificate acc z time h N (phaseBox positionRadius velocityRadius))
    (y : ℕ → E × E) (A : ℕ → E) (eta rounding : ℝ≥0) (initial : ℝ) (hh : 0 ≤ h)
    (hinit : ‖y 0 - z (time 0)‖ ≤ initial)
    (hres : ∀ i < N, y i ∈ phaseBox positionRadius velocityRadius →
      ‖A i - acc (y i)‖ ≤ eta)
    (hround : ∀ i < N, y i ∈ phaseBox positionRadius velocityRadius →
      ‖y (i + 1) - positionVelocityStep h (y i).1 (y i).2 (A i)‖ ≤ rounding)
    (hx : ∀ i ≤ N, ‖(z (time i)).1‖ + c.errorEnvelope eta rounding initial N < positionRadius)
    (hv : ∀ i ≤ N, ‖(z (time i)).2‖ + c.errorEnvelope eta rounding initial N < velocityRadius) :
    ∀ i ≤ N, ‖y i - z (time i)‖ ≤ c.errorEnvelope eta rounding initial i ∧
      ‖(y i).1‖ < positionRadius ∧ ‖(y i).2‖ < velocityRadius := by
  exact c.trajectory_error_domain y A eta rounding initial hh hinit hres hround
    (fun i hi w hw => phaseBox_mem_of_error (z (time i)) w _ positionRadius velocityRadius
      hw (hx i hi) (hv i hi))

/-- The exact proposed affine phase segment stays in a convex certified region
when the reference margin covers the derived next-step error. -/
theorem ODERegionCertificate.proposed_segment_mem {acc : E × E → E} {z : ℝ → E × E}
    {time : ℕ → ℝ} {h : ℝ} {N : ℕ} {K : Set (E × E)}
    (c : ODERegionCertificate acc z time h N K) (i : ℕ) (hi : i < N)
    (eta rounding : ℝ≥0) (initial : ℝ) (y : E × E) (A : E) (hh : 0 ≤ h)
    (hy : y ∈ K) (herr : ‖y - z (time i)‖ ≤ c.errorEnvelope eta rounding initial i)
    (hres : ‖A - acc y‖ ≤ eta) (hconv : Convex ℝ K)
    (htube : ∀ w : E × E, ‖w - z (time (i + 1))‖ ≤
      c.errorEnvelope eta rounding initial (i + 1) → w ∈ K)
    (s : ℝ) (hs : s ∈ Icc (0 : ℝ) 1) :
    y + s • (positionVelocityStep h y.1 y.2 A - y) ∈ K := by
  have hstep := c.step_error i hi eta rounding y (positionVelocityStep h y.1 y.2 A) A
    hh hy hres (by simp)
  have hnext : ‖positionVelocityStep h y.1 y.2 A - z (time (i + 1))‖ ≤
      c.errorEnvelope eta rounding initial (i + 1) := by
    exact hstep.trans (add_le_add_right (mul_le_mul_of_nonneg_left herr
      (zero_le_one.trans (one_le_stepAmplification h c.accelerationLip c.accelerationLip.coe_nonneg))) _)
  apply hconv.add_smul_mem hy
  · convert htube _ hnext using 1
    abel
  · exact hs

end FractalSync.ScaleMetric
