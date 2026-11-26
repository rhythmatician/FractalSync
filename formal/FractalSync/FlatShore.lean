import FractalSync.Ruler
import Mathlib.Analysis.SpecialFunctions.Integrals.Basic
import Mathlib.MeasureTheory.Integral.DominatedConvergence
import Mathlib.Topology.Order.IntermediateValue
import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.Calculus.Deriv.Add
import Mathlib.Analysis.Calculus.Deriv.Inv
import Mathlib.Analysis.Calculus.Deriv.Mul
import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Basic
import Mathlib.MeasureTheory.Integral.IntervalIntegral.Basic
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Scalar analysis for a flat Shore

The formulas here use the signed Euclidean normal coordinate `r`. The parameter
`epsilon` is positive; the formulas depend on `a` only through `a²`, so the
statements allow all real `a`. These are scalar derivative, inequality, and
integral theorems. They do not construct a Riemannian manifold or identify its
curvature tensor, and do not formalize the global cut-locus argument.
-/

namespace FractalSync.FlatShore

/-- Use the shared ruler authority; the local name keeps flat-model formulas compact. -/
noncomputable abbrev ruler := FractalSync.ruler

noncomputable def normalNumerator (epsilon a r : ℝ) : ℝ :=
  Real.sqrt (epsilon ^ 2 + (1 + a ^ 2) * r ^ 2)

noncomputable def radialDensity (epsilon a r : ℝ) : ℝ :=
  normalNumerator epsilon a r / (epsilon ^ 2 + r ^ 2)

noncomputable def warp (epsilon r : ℝ) : ℝ := (ruler epsilon r)⁻¹

noncomputable def warpSlope (epsilon a r : ℝ) : ℝ :=
  -r / (ruler epsilon r * normalNumerator epsilon a r)

noncomputable def radialCurvature (epsilon a r : ℝ) : ℝ :=
  (epsilon ^ 4 - (1 + a ^ 2) * r ^ 4) /
    (epsilon ^ 2 + (1 + a ^ 2) * r ^ 2) ^ 2

noncomputable def tangentialCurvature (epsilon a r : ℝ) : ℝ :=
  -(r ^ 2) / (epsilon ^ 2 + (1 + a ^ 2) * r ^ 2)

lemma ruler_sq_pos (epsilon r : ℝ) (he : 0 < epsilon) :
    0 < epsilon ^ 2 + r ^ 2 :=
  add_pos_of_pos_of_nonneg (sq_pos_of_pos he) (sq_nonneg r)

lemma normal_sq_pos (epsilon a r : ℝ) (he : 0 < epsilon) :
    0 < epsilon ^ 2 + (1 + a ^ 2) * r ^ 2 := by
  exact add_pos_of_pos_of_nonneg (sq_pos_of_pos he)
    (mul_nonneg (by positivity) (sq_nonneg r))

lemma normalNumerator_pos (epsilon a r : ℝ) (he : 0 < epsilon) :
    0 < normalNumerator epsilon a r :=
  Real.sqrt_pos.mpr (normal_sq_pos epsilon a r he)

/-- The scalar radial-curvature formula equals one on Shore. -/
theorem radialCurvature_zero (epsilon a : ℝ) (he : 0 < epsilon) :
    radialCurvature epsilon a 0 = 1 := by
  unfold radialCurvature
  simp only [zero_pow (by decide : 2 ≠ 0), zero_pow (by decide : 4 ≠ 0),
    mul_zero, sub_zero, add_zero]
  field_simp

/-- The scalar tangential-curvature formula vanishes on Shore. -/
theorem tangentialCurvature_zero (epsilon a : ℝ) :
    tangentialCurvature epsilon a 0 = 0 := by
  simp [tangentialCurvature]

/-- Derivative of the numerator of the intrinsic normal density. -/
theorem hasDerivAt_normalNumerator (epsilon a r : ℝ) (he : 0 < epsilon) :
    HasDerivAt (normalNumerator epsilon a)
      ((1 + a ^ 2) * r / normalNumerator epsilon a r) r := by
  convert ((((hasDerivAt_id r).pow 2).const_mul (1 + a ^ 2)).const_add
    (epsilon ^ 2)).sqrt (normal_sq_pos epsilon a r he).ne' using 1
  simp [normalNumerator]
  ring

/-- The ordinary derivative of the warp factor in Euclidean normal coordinates. -/
theorem hasDerivAt_warp (epsilon r : ℝ) (he : 0 < epsilon) :
    HasDerivAt (warp epsilon) (-r / (ruler epsilon r) ^ 3) r := by
  convert (hasDerivAt_ruler epsilon r he).inv (ruler_pos epsilon r he).ne' using 1
  ring

/-- Differentiating in intrinsic normal distance divides by `dt/dr`. -/
theorem warpSlope_eq (epsilon a r : ℝ) (he : 0 < epsilon) :
    deriv (warp epsilon) r / radialDensity epsilon a r = warpSlope epsilon a r := by
  rw [(hasDerivAt_warp epsilon r he).deriv]
  have hrho : (ruler epsilon r) ^ 2 = epsilon ^ 2 + r ^ 2 :=
    by simpa only [add_comm] using FractalSync.ruler_sq epsilon r
  unfold radialDensity warpSlope
  rw [← hrho]
  field_simp

/-- Euclidean-normal derivative of the intrinsic warp slope. -/
theorem hasDerivAt_warpSlope (epsilon a r : ℝ) (he : 0 < epsilon) :
    HasDerivAt (warpSlope epsilon a)
      (((1 + a ^ 2) * r ^ 4 - epsilon ^ 4) /
        ((ruler epsilon r) ^ 3 * (normalNumerator epsilon a r) ^ 3)) r := by
  convert (hasDerivAt_id r).neg.div
    ((hasDerivAt_ruler epsilon r he).mul (hasDerivAt_normalNumerator epsilon a r he))
    (mul_ne_zero (ruler_pos epsilon r he).ne'
      (normalNumerator_pos epsilon a r he).ne') using 1
  have hrho : (ruler epsilon r) ^ 2 = epsilon ^ 2 + r ^ 2 :=
    by simpa only [add_comm] using FractalSync.ruler_sq epsilon r
  have hQ : (normalNumerator epsilon a r) ^ 2 =
      epsilon ^ 2 + (1 + a ^ 2) * r ^ 2 :=
    Real.sq_sqrt (normal_sq_pos epsilon a r he).le
  simp only [Pi.mul_apply, Pi.neg_apply, id_eq]
  field_simp
  rw [hrho, hQ]
  ring

/-- The scalar expression `-f_tt/f`, after the normal-coordinate change,
reduces to the stated rational formula. The Riemannian curvature identification
is not part of this scalar theorem. -/
theorem warp_curvature_identity (epsilon a r : ℝ) (he : 0 < epsilon) :
    -(deriv (warpSlope epsilon a) r / radialDensity epsilon a r) /
      warp epsilon r = radialCurvature epsilon a r := by
  rw [(hasDerivAt_warpSlope epsilon a r he).deriv]
  have hrho : (ruler epsilon r) ^ 2 = epsilon ^ 2 + r ^ 2 :=
    by simpa only [add_comm] using FractalSync.ruler_sq epsilon r
  have hQ : (normalNumerator epsilon a r) ^ 2 =
      epsilon ^ 2 + (1 + a ^ 2) * r ^ 2 :=
    Real.sq_sqrt (normal_sq_pos epsilon a r he).le
  unfold radialDensity warp radialCurvature
  rw [← hrho, ← hQ]
  field_simp [(ruler_pos epsilon r he).ne']
  ring

/-- Exact outward-coordinate derivative of the radial-curvature formula. -/
theorem hasDerivAt_radialCurvature (epsilon a r : ℝ) (he : 0 < epsilon) :
    HasDerivAt (radialCurvature epsilon a)
      (-4 * epsilon ^ 2 * (1 + a ^ 2) * r * (epsilon ^ 2 + r ^ 2) /
        (epsilon ^ 2 + (1 + a ^ 2) * r ^ 2) ^ 3) r := by
  convert (((hasDerivAt_id r).pow 4).const_mul (1 + a ^ 2)).const_sub
    (epsilon ^ 4) |>.div
      (((((hasDerivAt_id r).pow 2).const_mul (1 + a ^ 2)).const_add
        (epsilon ^ 2)).pow 2)
      (pow_ne_zero 2 (normal_sq_pos epsilon a r he).ne') using 1
  simp only [id_eq, Pi.pow_apply, Nat.cast_ofNat, Nat.reduceSub, pow_one, mul_one]
  field_simp [(normal_sq_pos epsilon a r he).ne']
  ring

/-- Curvature strictly decreases along each outward Euclidean normal ray. -/
theorem radialCurvature_deriv_neg (epsilon a r : ℝ) (he : 0 < epsilon)
    (hr : 0 < r) : deriv (radialCurvature epsilon a) r < 0 := by
  rw [(hasDerivAt_radialCurvature epsilon a r he).deriv]
  apply div_neg_of_neg_of_pos
  · have hpos : 0 < 4 * epsilon ^ 2 * (1 + a ^ 2) * r * (epsilon ^ 2 + r ^ 2) := by
      positivity
    nlinarith only [hpos]
  · exact pow_pos (normal_sq_pos epsilon a r he) 3

/-- Strict curvature monotonicity, including the Shore endpoint. -/
theorem radialCurvature_strictAntiOn (epsilon a : ℝ) (he : 0 < epsilon) :
    StrictAntiOn (radialCurvature epsilon a) (Set.Ici 0) := by
  apply strictAntiOn_of_deriv_neg (convex_Ici 0)
  · exact fun r _ => (hasDerivAt_radialCurvature epsilon a r he).continuousAt.continuousWithinAt
  · intro r hr
    exact radialCurvature_deriv_neg epsilon a r he (by simpa using hr)

/-- The radial and purely tangential scalar curvatures have a positive gap. -/
theorem curvature_gap (epsilon a r : ℝ) (he : 0 < epsilon) :
    radialCurvature epsilon a r - tangentialCurvature epsilon a r =
      epsilon ^ 2 * (epsilon ^ 2 + r ^ 2) /
        (epsilon ^ 2 + (1 + a ^ 2) * r ^ 2) ^ 2 := by
  unfold radialCurvature tangentialCurvature
  field_simp [(normal_sq_pos epsilon a r he).ne']
  ring

/-- The tangential formula stays strictly above its asymptotic lower value. -/
theorem tangentialCurvature_lower (epsilon a r : ℝ) (he : 0 < epsilon) :
    -1 / (1 + a ^ 2) < tangentialCurvature epsilon a r := by
  have hb : 0 < 1 + a ^ 2 := by positivity
  unfold tangentialCurvature
  apply (div_lt_div_iff₀ hb (normal_sq_pos epsilon a r he)).mpr
  nlinarith only [sq_pos_of_pos he]

/-- The radial scalar curvature lies strictly above the tangential one. -/
theorem tangentialCurvature_lt_radialCurvature (epsilon a r : ℝ) (he : 0 < epsilon) :
    tangentialCurvature epsilon a r < radialCurvature epsilon a r := by
  apply sub_pos.mp
  rw [curvature_gap epsilon a r he]
  exact div_pos (mul_pos (sq_pos_of_pos he) (ruler_sq_pos epsilon r he))
    (sq_pos_of_pos (normal_sq_pos epsilon a r he))

/-- Global upper bound for the radial scalar formula. -/
theorem radialCurvature_le_one (epsilon a r : ℝ) (he : 0 < epsilon) :
    radialCurvature epsilon a r ≤ 1 := by
  unfold radialCurvature
  apply (div_le_iff₀ (sq_pos_of_pos (normal_sq_pos epsilon a r he))).mpr
  nlinarith [mul_nonneg (by positivity : 0 ≤ 1 + a ^ 2) (sq_nonneg r),
    mul_nonneg (by positivity : 0 ≤ 1 + a ^ 2) (sq_nonneg (r ^ 2))]

/-- The entire radial scalar curvature range lies in `(-1/(1+a²),1]`. -/
theorem radialCurvature_bounds (epsilon a r : ℝ) (he : 0 < epsilon) :
    -1 / (1 + a ^ 2) < radialCurvature epsilon a r ∧
      radialCurvature epsilon a r ≤ 1 :=
  ⟨(tangentialCurvature_lower epsilon a r he).trans
    (tangentialCurvature_lt_radialCurvature epsilon a r he),
    radialCurvature_le_one epsilon a r he⟩

/-- The tangential scalar formula is nonpositive. -/
theorem tangentialCurvature_nonpos (epsilon a r : ℝ) (he : 0 < epsilon) :
    tangentialCurvature epsilon a r ≤ 0 :=
  div_nonpos_of_nonpos_of_nonneg (neg_nonpos.mpr (sq_nonneg r))
    (normal_sq_pos epsilon a r he).le

/-- Integrand after the substitution `r = R sin θ` in one geodesic excursion. -/
noncomputable def excursionIntegrand (epsilon a R theta : ℝ) : ℝ :=
  Real.sqrt (epsilon ^ 2 + (1 + a ^ 2) * R ^ 2 * (Real.sin theta) ^ 2)

/-- Tangential displacement of a full flat-Shore excursion of turning radius R. -/
noncomputable def excursionSpan (epsilon a R : ℝ) : ℝ :=
  2 * ∫ theta in (0 : ℝ)..(Real.pi / 2), excursionIntegrand epsilon a R theta

lemma continuous_excursionIntegrand (epsilon a R : ℝ) :
    Continuous (excursionIntegrand epsilon a R) := by
  exact (continuous_const.add (continuous_const.mul (Real.continuous_sin.pow 2))).sqrt

/-- The zero-amplitude endpoint of the excursion displacement is πε. -/
theorem excursionSpan_zero (epsilon a : ℝ) (he : 0 ≤ epsilon) :
    excursionSpan epsilon a 0 = Real.pi * epsilon := by
  unfold excursionSpan excursionIntegrand
  simp [Real.sqrt_sq he]
  ring

/-- A larger nonnegative turning radius gives strictly larger displacement. -/
theorem excursionSpan_strictMonoOn (epsilon a : ℝ) :
    StrictMonoOn (excursionSpan epsilon a) (Set.Ici 0) := by
  intro R₁ h₁ R₂ h₂ hR
  apply mul_lt_mul_of_pos_left _ (by norm_num : (0 : ℝ) < 2)
  apply intervalIntegral.integral_lt_integral_of_continuousOn_of_le_of_exists_lt
    Real.pi_div_two_pos (continuous_excursionIntegrand epsilon a R₁).continuousOn
    (continuous_excursionIntegrand epsilon a R₂).continuousOn
  · intro theta _
    unfold excursionIntegrand
    apply Real.sqrt_le_sqrt
    exact add_le_add_left (mul_le_mul_of_nonneg_right
      (mul_le_mul_of_nonneg_left ((sq_le_sq₀ h₁ h₂).mpr hR.le) (by positivity))
      (sq_nonneg _)) _
  · refine ⟨Real.pi / 2, ⟨Real.pi_div_two_pos.le, le_rfl⟩, ?_⟩
    simp only [excursionIntegrand, Real.sin_pi_div_two, one_pow, mul_one]
    exact Real.sqrt_lt_sqrt (by positivity)
      (add_lt_add_left (mul_lt_mul_of_pos_left ((sq_lt_sq₀ h₁ h₂).mpr hR)
        (by positivity)) _)

/-- Excursion displacements vary continuously with their radius. -/
theorem continuous_excursionSpan (epsilon a : ℝ) :
    Continuous (excursionSpan epsilon a) := by
  apply continuous_const.mul
  apply intervalIntegral.continuous_parametric_intervalIntegral_of_continuous'
  exact (continuous_const.add ((continuous_const.mul (continuous_fst.pow 2)).mul
    (Real.continuous_sin.comp continuous_snd |>.pow 2))).sqrt

/-- Every nonzero excursion has displacement strictly greater than πε. -/
theorem excursionSpan_gt_threshold (epsilon a R : ℝ) (he : 0 ≤ epsilon)
    (hR : 0 < R) : Real.pi * epsilon < excursionSpan epsilon a R := by
  rw [← excursionSpan_zero epsilon a he]
  exact excursionSpan_strictMonoOn epsilon a (by simp) hR.le hR

/-- For nonnegative radii, the threshold is attained only at zero radius. -/
theorem excursionSpan_eq_threshold_iff (epsilon a R : ℝ) (he : 0 ≤ epsilon)
    (hR : 0 ≤ R) : excursionSpan epsilon a R = Real.pi * epsilon ↔ R = 0 := by
  rw [← excursionSpan_zero epsilon a he]
  exact (excursionSpan_strictMonoOn epsilon a).injOn.eq_iff hR (by simp)

/-- A useful linear lower bound on displacement, hence unboundedness. -/
theorem excursionSpan_linear_lower (epsilon a R : ℝ) :
    2 * Real.sqrt (1 + a ^ 2) * R ≤ excursionSpan epsilon a R := by
  have hle : (∫ theta in (0 : ℝ)..(Real.pi / 2),
      (Real.sqrt (1 + a ^ 2) * R) * Real.sin theta) ≤
      ∫ theta in (0 : ℝ)..(Real.pi / 2), excursionIntegrand epsilon a R theta := by
    apply intervalIntegral.integral_mono Real.pi_div_two_pos.le
      (continuous_const.mul Real.continuous_sin |>.intervalIntegrable _ _)
      (continuous_excursionIntegrand epsilon a R |>.intervalIntegrable _ _)
    intro theta
    apply Real.le_sqrt_of_sq_le
    rw [mul_pow, mul_pow, Real.sq_sqrt (by positivity)]
    exact le_add_of_nonneg_left (sq_nonneg epsilon)
  simpa [excursionSpan, intervalIntegral.integral_const_mul, mul_assoc] using
    mul_le_mul_of_nonneg_left hle (by norm_num : (0 : ℝ) ≤ 2)

/-- Each tangential separation above πε has exactly one positive turning radius.
This is an inversion theorem for the excursion integral, not a proof that the
corresponding geodesic is globally minimizing. -/
theorem excursionSpan_existsUnique_radius (epsilon a L : ℝ) (he : 0 < epsilon)
    (hL : Real.pi * epsilon < L) :
    ∃! R : ℝ, 0 < R ∧ excursionSpan epsilon a R = L := by
  have hLp : 0 < L := (mul_pos Real.pi_pos he).trans hL
  have hb : 0 < Real.sqrt (1 + a ^ 2) := Real.sqrt_pos.mpr (by positivity)
  let T := L / (2 * Real.sqrt (1 + a ^ 2))
  have hT : 0 ≤ T := (div_pos hLp (mul_pos (by norm_num) hb)).le
  have htop : L ≤ excursionSpan epsilon a T := by
    convert excursionSpan_linear_lower epsilon a T using 1
    dsimp [T]
    field_simp
  obtain ⟨R, hR, hval⟩ := intermediate_value_Icc hT
    (continuous_excursionSpan epsilon a).continuousOn
    (show L ∈ Set.Icc (excursionSpan epsilon a 0) (excursionSpan epsilon a T) from
      ⟨by simpa [excursionSpan_zero epsilon a he.le] using hL.le, htop⟩)
  have hRp : 0 < R := by
    by_contra hn
    have hzero : R = 0 := le_antisymm (le_of_not_gt hn) hR.1
    rw [hzero, excursionSpan_zero epsilon a he.le] at hval
    exact hL.ne hval
  refine ⟨R, ⟨hRp, hval⟩, ?_⟩
  intro S hS
  exact (excursionSpan_strictMonoOn epsilon a).injOn hS.1.le hRp.le
    (hS.2.trans hval.symm)

/-- Quartic density obtained in the conventional Taylor expansion at ℓ=π.
The Taylor expansion itself is not asserted by this definition. -/
noncomputable def quarticDensity (a theta : ℝ) : ℝ :=
  (1 + a ^ 2 - 2) / 2 * ((Real.sin theta) ^ 2 * (Real.cos theta) ^ 2) +
    1 / 2 * (Real.sin theta) ^ 4 -
    1 / 8 * ((Real.cos theta) ^ 2 - (Real.sin theta) ^ 2) ^ 2

/-- A trigonometric reduction of the proposed quartic density. -/
theorem quarticDensity_eq (a theta : ℝ) :
    quarticDensity a theta =
      a ^ 2 / 2 * ((Real.sin theta) ^ 2 * (Real.cos theta) ^ 2) +
      1 / 2 * (Real.sin theta) ^ 4 - 1 / 8 := by
  unfold quarticDensity
  have hsq := congrArg (fun z : ℝ => z ^ 2) (Real.sin_sq_add_cos_sq theta)
  nlinarith only [hsq]

/-- Exact integration of the quartic density: the coefficient is positive
for every real graph parameter. This does not formalize the surrounding
Taylor theorem or the Lyapunov–Schmidt reduction. -/
theorem integral_quarticDensity (a : ℝ) :
    (∫ theta in (0 : ℝ)..Real.pi, quarticDensity a theta) =
      Real.pi * (1 + a ^ 2) / 16 := by
  have h4 := integral_sin_pow_even 2
  norm_num [Finset.prod_range_succ] at h4
  have hc : Continuous (fun theta : ℝ =>
      a ^ 2 / 2 * ((Real.sin theta) ^ 2 * (Real.cos theta) ^ 2)) :=
    continuous_const.mul ((Real.continuous_sin.pow 2).mul (Real.continuous_cos.pow 2))
  have hs : Continuous (fun theta : ℝ => (1 : ℝ) / 2 * (Real.sin theta) ^ 4) :=
    continuous_const.mul (Real.continuous_sin.pow 4)
  simp_rw [quarticDensity_eq]
  rw [intervalIntegral.integral_sub ((hc.add hs).intervalIntegrable _ _)
    (continuous_const.intervalIntegrable _ _),
    intervalIntegral.integral_add (hc.intervalIntegrable _ _) (hs.intervalIntegrable _ _),
    intervalIntegral.integral_const_mul, intervalIntegral.integral_const_mul,
    integral_sin_sq_mul_cos_sq, h4]
  rw [show Real.sin (4 * Real.pi) = 0 from Real.sin_nat_mul_pi 4]
  norm_num
  ring

/-- The proposed reduced quartic coefficient is strictly positive. -/
theorem integral_quarticDensity_pos (a : ℝ) :
    0 < ∫ theta in (0 : ℝ)..Real.pi, quarticDensity a theta := by
  rw [integral_quarticDensity]
  positivity

/-- Unit speed and a specified conserved tangential momentum imply the radial
first-integral formula. Conservation along geodesics is a separate theorem. -/
theorem radial_first_integral (epsilon a r vr vy p : ℝ) (he : 0 < epsilon)
    (hunit : (epsilon ^ 2 + (1 + a ^ 2) * r ^ 2) * vr ^ 2 /
      (epsilon ^ 2 + r ^ 2) ^ 2 + vy ^ 2 / (epsilon ^ 2 + r ^ 2) = 1)
    (hmomentum : vy = p * (epsilon ^ 2 + r ^ 2)) :
    vr ^ 2 = (epsilon ^ 2 + r ^ 2) ^ 2 * (1 - p ^ 2 * (epsilon ^ 2 + r ^ 2)) /
      (epsilon ^ 2 + (1 + a ^ 2) * r ^ 2) := by
  rw [hmomentum] at hunit
  field_simp [(ruler_sq_pos epsilon r he).ne'] at hunit
  apply (eq_div_iff (normal_sq_pos epsilon a r he).ne').mpr
  nlinarith only [hunit]

/-- Simultaneous Euclidean dilation scales the excursion displacement exactly. -/
theorem excursionSpan_scale (epsilon a R beta : ℝ) (hbeta : 0 ≤ beta) :
    excursionSpan (beta * epsilon) a (beta * R) = beta * excursionSpan epsilon a R := by
  have hi : ∀ theta : ℝ, excursionIntegrand (beta * epsilon) a (beta * R) theta =
      beta * excursionIntegrand epsilon a R theta := by
    intro theta
    unfold excursionIntegrand
    rw [show (beta * epsilon) ^ 2 + (1 + a ^ 2) * (beta * R) ^ 2 *
        (Real.sin theta) ^ 2 =
        beta ^ 2 * (epsilon ^ 2 + (1 + a ^ 2) * R ^ 2 * (Real.sin theta) ^ 2) by ring,
      Real.sqrt_mul (sq_nonneg beta), Real.sqrt_sq hbeta]
  unfold excursionSpan
  simp_rw [hi]
  rw [intervalIntegral.integral_const_mul]
  ring

end FractalSync.FlatShore
