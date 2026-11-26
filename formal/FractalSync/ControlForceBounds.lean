import FractalSync.MechanicalBounds
import FractalSync.ControlMechanics

/-!
# Explicit regional bounds for metric-consistent controls

The actual metric lowering, drive covector and viscous covector inherit
explicit norm/Lipschitz pairs from the ruler, gradient and control inputs.
These are exact-real local field estimates; machine clamping, deadzones,
time-varying commands and combined steering remain separate contracts.
-/

noncomputable section

namespace FractalSync.ControlMechanics

open Set ScaleMetric ScaleMetric.MechanicalBounds
open scoped RealInnerProductSpace

/-- The actual metric-lowering budget, including its changing ruler and gradient. -/
def metricActionBudget (epsilon a : ℝ) (br bu bv : Budget) : Budget :=
  let ri := inverseBudget epsilon br
  productBudget (productBudget ri ri)
    (addBudget bv (productBudget (productBudget (a ^ 2, 0) (productBudget bu bv)) bu))

def driveForceBudget (epsilon a : ℝ) (br bu bF bd : Budget) : Budget :=
  productBudget bF (metricActionBudget epsilon a br bu bd)

def dragForceBudget (epsilon a : ℝ) (br bu bBeta bv : Budget) : Budget :=
  productBudget bBeta (metricActionBudget epsilon a br bu bv)

variable {X E : Type*} [PseudoMetricSpace X] [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem metricAction_boundedLipschitz (epsilon a : ℝ) (s : Set X)
    (rho : X → ℝ) (u v : X → E) (br bu bv : Budget) (he : 0 < epsilon)
    (hr : BoundedLipschitzOn rho s br.1 br.2)
    (hu : BoundedLipschitzOn u s bu.1 bu.2)
    (hv : BoundedLipschitzOn v s bv.1 bv.2)
    (hfloor : ∀ x ∈ s, epsilon ≤ rho x) :
    BoundedLipschitzOn (fun x => metricAction (rho x) a (u x) (v x)) s
      (metricActionBudget epsilon a br bu bv).1 (metricActionBudget epsilon a br bu bv).2 := by
  have hA : BoundedLipschitzOn (fun _ : X => a ^ 2) s (a ^ 2) 0 :=
    ⟨sq_nonneg a, le_rfl, fun _ _ => (Real.norm_of_nonneg (sq_nonneg a)).le,
      fun _ _ _ _ => by simp⟩
  have hri := hr.inv he hfloor
  simpa only [metricActionBudget, productBudget, addBudget, inverseBudget, metricAction,
    stretch, pow_two, mul_inv_rev] using
    (hri.mul hri).smul (hv.add ((hA.mul (hu.inner hv)).smul hu))

theorem driveForce_boundedLipschitz (epsilon a : ℝ) (s : Set X)
    (rho F : X → ℝ) (u d : X → E) (br bu bF bd : Budget) (he : 0 < epsilon)
    (hr : BoundedLipschitzOn rho s br.1 br.2)
    (hu : BoundedLipschitzOn u s bu.1 bu.2)
    (hF : BoundedLipschitzOn F s bF.1 bF.2)
    (hd : BoundedLipschitzOn d s bd.1 bd.2)
    (hfloor : ∀ x ∈ s, epsilon ≤ rho x) :
    BoundedLipschitzOn (fun x => driveForce (rho x) a (F x) (u x) (d x)) s
      (driveForceBudget epsilon a br bu bF bd).1 (driveForceBudget epsilon a br bu bF bd).2 := by
  exact hF.smul (metricAction_boundedLipschitz epsilon a s rho u d br bu bd he hr hu hd hfloor)

theorem dragForce_boundedLipschitz (epsilon a : ℝ) (s : Set X)
    (rho beta : X → ℝ) (u v : X → E) (br bu bBeta bv : Budget) (he : 0 < epsilon)
    (hr : BoundedLipschitzOn rho s br.1 br.2)
    (hu : BoundedLipschitzOn u s bu.1 bu.2)
    (hb : BoundedLipschitzOn beta s bBeta.1 bBeta.2)
    (hv : BoundedLipschitzOn v s bv.1 bv.2)
    (hfloor : ∀ x ∈ s, epsilon ≤ rho x) :
    BoundedLipschitzOn (fun x => -(beta x • metricAction (rho x) a (u x) (v x))) s
      (dragForceBudget epsilon a br bu bBeta bv).1 (dragForceBudget epsilon a br bu bBeta bv).2 := by
  have hd := hb.smul (metricAction_boundedLipschitz epsilon a s rho u v br bu bv he hr hu hv hfloor)
  exact ⟨hd.bound_nonneg, hd.lip_nonneg,
    fun x hx => by simpa only [norm_neg] using hd.norm_le x hx,
    fun x hx y hy => by simpa only [neg_sub_neg, dist_comm y x] using hd.norm_sub_le y hy x hx⟩

/-- The square root loses at most a factor one-half above the unit floor. -/
theorem sqrt_sub_le_of_one_le (x y : ℝ) (hx : 1 ≤ x) (hy : 1 ≤ y) :
    ‖Real.sqrt x - Real.sqrt y‖ ≤ (1 / 2 : ℝ) * ‖x - y‖ := by
  apply Convex.norm_image_sub_le_of_norm_hasDerivWithin_le
    (f' := fun t => 1 / (2 * Real.sqrt t))
    (fun t (ht : t ∈ Ici (1 : ℝ)) =>
      (Real.hasDerivAt_sqrt (lt_of_lt_of_le zero_lt_one ht).ne').hasDerivWithinAt)
    _ (convex_Ici (1 : ℝ)) hy hx
  exact fun t ht => (Real.norm_of_nonneg (by positivity : 0 ≤ 1 / (2 * Real.sqrt t))).le.trans
    (div_le_div₀ (by norm_num) le_rfl (by norm_num)
      (by nlinarith [show (1 : ℝ) ≤ Real.sqrt t from by simpa using Real.sqrt_le_sqrt ht]))

theorem sqrt_boundedLipschitz_of_one_le (f : X → ℝ) (s : Set X) (M L : ℝ)
    (hf : BoundedLipschitzOn f s M L) (hfloor : ∀ x ∈ s, 1 ≤ f x) :
    BoundedLipschitzOn (fun x => Real.sqrt (f x)) s (Real.sqrt M) (L / 2) := by
  exact ⟨Real.sqrt_nonneg _, div_nonneg hf.lip_nonneg (by norm_num),
    fun x hx => (Real.norm_of_nonneg (Real.sqrt_nonneg _)).le.trans
      (Real.sqrt_le_sqrt ((le_abs_self _).trans
        (by simpa only [Real.norm_eq_abs] using hf.norm_le x hx))),
    fun x hx y hy => (sqrt_sub_le_of_one_le _ _ (hfloor x hx) (hfloor y hy)).trans
      (by simpa only [div_eq_mul_inv, one_mul, mul_assoc, mul_left_comm, mul_comm] using
        mul_le_mul_of_nonneg_left (hf.norm_sub_le x hx y hy) (by norm_num : (0 : ℝ) ≤ 1 / 2))⟩

/-- A fixed Euclidean-unit direction exposes the positive unit denominator
inside the actual metric normalization. -/
theorem metricNormalize_unit_formula (rho a : ℝ) (u e : E)
    (hr : 0 < rho) (he : ‖e‖ = 1) :
    metricNormalize rho a u e =
      (rho / Real.sqrt (1 + a ^ 2 * (inner ℝ u e) ^ 2)) • e := by
  simp only [metricNormalize, metric_self_formula, he, one_pow,
    Real.sqrt_div (by positivity : 0 ≤ 1 + a ^ 2 * (inner ℝ u e) ^ 2),
    Real.sqrt_sq hr.le, inv_div]

/-- Bounds for the actual normalization of a fixed nonzero direction. -/
def normalizedHeadingBudget (a : ℝ) (br bu : Budget) : Budget :=
  (br.1, br.2 + br.1 * a ^ 2 * bu.1 * bu.2)

theorem metricNormalize_unit_boundedLipschitz (a : ℝ) (s : Set X)
    (rho : X → ℝ) (u : X → E) (e : E) (br bu : Budget)
    (hr : BoundedLipschitzOn rho s br.1 br.2)
    (hu : BoundedLipschitzOn u s bu.1 bu.2)
    (hpos : ∀ x ∈ s, 0 < rho x) (he : ‖e‖ = 1) :
    BoundedLipschitzOn (fun x => metricNormalize (rho x) a (u x) e) s
      (normalizedHeadingBudget a br bu).1 (normalizedHeadingBudget a br bu).2 := by
  have hec : BoundedLipschitzOn (fun _ : X => e) s 1 0 := by
    simpa only [he] using BoundedLipschitzOn.const s e
  have hd : BoundedLipschitzOn (fun x => inner ℝ (u x) e) s bu.1 bu.2 := by
    simpa only [mul_one, mul_zero, one_mul, zero_add] using hu.inner hec
  have hA : BoundedLipschitzOn (fun _ : X => a ^ 2) s (a ^ 2) 0 :=
    ⟨sq_nonneg a, le_rfl, fun _ _ => (Real.norm_of_nonneg (sq_nonneg a)).le,
      fun _ _ _ _ => by simp⟩
  have hOne : BoundedLipschitzOn (fun _ : X => (1 : ℝ)) s 1 0 :=
    ⟨zero_le_one, le_rfl, fun _ _ => (norm_one : ‖(1 : ℝ)‖ = 1).le,
      fun _ _ _ _ => by simp⟩
  have hq := hOne.add (hA.mul (hd.mul hd))
  have hfloorq : ∀ x ∈ s, (1 : ℝ) ≤ 1 + a ^ 2 * (inner ℝ (u x) e * inner ℝ (u x) e) :=
    fun _ _ => le_add_of_nonneg_right (mul_nonneg (sq_nonneg _) (mul_self_nonneg _))
  have hsqrt := sqrt_boundedLipschitz_of_one_le _ s _ _ hq hfloorq
  have hi := hsqrt.inv zero_lt_one (fun x hx => by
    simpa only [Real.sqrt_one] using Real.sqrt_le_sqrt (hfloorq x hx))
  have hform : BoundedLipschitzOn
      (fun x => (rho x / Real.sqrt (1 + a ^ 2 * (inner ℝ (u x) e) ^ 2)) • e) s
      br.1 (br.2 + br.1 * a ^ 2 * bu.1 * bu.2) := by
    convert (hr.mul hi).smul hec using 1
    · simp only [div_eq_mul_inv, pow_two]
    · simp only [inv_one, mul_one]
    · ring
  exact ⟨hform.bound_nonneg, hform.lip_nonneg,
    fun x hx => by
      simpa only [metricNormalize_unit_formula (rho x) a (u x) e (hpos x hx) he] using
        hform.norm_le x hx,
    fun x hx y hy => by
      simpa only [metricNormalize_unit_formula (rho x) a (u x) e (hpos x hx) he,
        metricNormalize_unit_formula (rho y) a (u y) e (hpos y hy) he] using
        hform.norm_sub_le x hx y hy⟩

/-- Positive rescaling of a command does not change its metric-normalized direction. -/
theorem metricNormalize_pos_smul (rho a c : ℝ) (u e : E) (hc : 0 < c) :
    metricNormalize rho a u (c • e) = metricNormalize rho a u e := by
  simp only [metricNormalize, metric_smul_left, metric_smul_right]
  rw [← mul_assoc, ← pow_two, Real.sqrt_mul (sq_nonneg c), Real.sqrt_sq hc.le,
    mul_inv_rev, smul_smul, mul_assoc, inv_mul_cancel₀ hc.ne', mul_one]

/-- The fixed-command bound is independent of the original nonzero command magnitude. -/
theorem metricNormalize_boundedLipschitz (a : ℝ) (s : Set X)
    (rho : X → ℝ) (u : X → E) (e : E) (br bu : Budget)
    (hr : BoundedLipschitzOn rho s br.1 br.2)
    (hu : BoundedLipschitzOn u s bu.1 bu.2)
    (hpos : ∀ x ∈ s, 0 < rho x) (he : e ≠ 0) :
    BoundedLipschitzOn (fun x => metricNormalize (rho x) a (u x) e) s
      (normalizedHeadingBudget a br bu).1 (normalizedHeadingBudget a br bu).2 := by
  have heunit : ‖‖e‖⁻¹ • e‖ = 1 := by
    simp only [norm_smul, Real.norm_eq_abs, abs_inv, abs_norm,
      inv_mul_cancel₀ (norm_ne_zero_iff.mpr he)]
  simpa only [metricNormalize_pos_smul (hc := inv_pos.mpr (norm_pos_iff.mpr he))] using
    metricNormalize_unit_boundedLipschitz a s rho u (‖e‖⁻¹ • e) br bu hr hu hpos heunit

def normalizedDriveForceBudget (epsilon a : ℝ) (br bu bF : Budget) : Budget :=
  driveForceBudget epsilon a br bu bF (normalizedHeadingBudget a br bu)

/-- Regional constants for the actual fixed-direction intrinsically normalized drive. -/
theorem normalizedDriveForce_boundedLipschitz (epsilon a : ℝ) (s : Set X)
    (rho F : X → ℝ) (u : X → E) (e : E) (br bu bF : Budget) (hepsilon : 0 < epsilon)
    (hr : BoundedLipschitzOn rho s br.1 br.2)
    (hu : BoundedLipschitzOn u s bu.1 bu.2)
    (hF : BoundedLipschitzOn F s bF.1 bF.2)
    (hfloor : ∀ x ∈ s, epsilon ≤ rho x) (he : e ≠ 0) :
    BoundedLipschitzOn
      (fun x => driveForce (rho x) a (F x) (u x)
        (metricNormalize (rho x) a (u x) e)) s
      (normalizedDriveForceBudget epsilon a br bu bF).1
      (normalizedDriveForceBudget epsilon a br bu bF).2 := by
  exact driveForce_boundedLipschitz epsilon a s rho F u
    (fun x => metricNormalize (rho x) a (u x) e) br bu bF
    (normalizedHeadingBudget a br bu) hepsilon hr hu hF
    (metricNormalize_boundedLipschitz a s rho u e br bu hr hu
      (fun x hx => hepsilon.trans_le (hfloor x hx)) he) hfloor

end FractalSync.ControlMechanics
