import FractalSync.RulerDifferential

/-!
# Quantitative regularization of bounded field jets

Scalar regularizer bounds and operator-norm estimates for the genuine
second-chain-rule expression. The field inputs are separate from these
algebraic and scalar-calculus statements.
-/

noncomputable section

namespace FractalSync.RulerJetBounds

open Set Filter
open scoped Topology

theorem ruler_slope_abs_le (epsilon r : ℝ) (he : 0 < epsilon) :
    |r / ruler epsilon r| ≤ 1 := by
  simpa only [abs_div, abs_of_pos (ruler_pos epsilon r he)] using
    (div_le_one (ruler_pos epsilon r he)).2 (abs_le_ruler epsilon r)

theorem ruler_second_le (epsilon r : ℝ) (he : 0 < epsilon) :
    epsilon ^ 2 / ruler epsilon r ^ 3 ≤ 1 / epsilon := by
  rw [div_le_div_iff₀ (pow_pos (ruler_pos epsilon r he) 3) he]
  simpa only [← pow_succ, one_mul] using
    pow_le_pow_left₀ he.le (epsilon_le_ruler epsilon r he.le) 3

theorem hasDerivAt_ruler_second (epsilon r : ℝ) (he : 0 < epsilon) :
    HasDerivAt (fun s => epsilon ^ 2 / ruler epsilon s ^ 3)
      (-3 * epsilon ^ 2 * r / ruler epsilon r ^ 5) r := by
  convert (hasDerivAt_const r (epsilon ^ 2)).div
    ((hasDerivAt_ruler epsilon r he).pow 3)
    (pow_ne_zero 3 (ruler_pos epsilon r he).ne') using 1
  simp only [Pi.pow_apply, Nat.cast_ofNat, Nat.reduceSub, zero_mul, zero_sub]
  field_simp

theorem ruler_third_abs_le (epsilon r : ℝ) (he : 0 < epsilon) :
    |-3 * epsilon ^ 2 * r / ruler epsilon r ^ 5| ≤ 3 / epsilon ^ 2 := by
  rw [abs_div, abs_mul, abs_mul, abs_of_nonneg (sq_nonneg epsilon),
    abs_of_pos (pow_pos (ruler_pos epsilon r he) 5)]
  norm_num only [abs_neg, abs_of_nonneg (show (0 : ℝ) ≤ 3 by norm_num)]
  calc
    _ ≤ 3 * epsilon ^ 2 * ruler epsilon r / ruler epsilon r ^ 5 := by
      exact div_le_div_of_nonneg_right
        (mul_le_mul_of_nonneg_left (abs_le_ruler epsilon r) (by positivity))
        (pow_pos (ruler_pos epsilon r he) 5).le
    _ = 3 * epsilon ^ 2 / ruler epsilon r ^ 4 := by field_simp
    _ ≤ 3 * epsilon ^ 2 / epsilon ^ 4 := by
      gcongr
      exact epsilon_le_ruler epsilon r he.le
    _ = _ := by field_simp

theorem ruler_slope_sub_le (epsilon r s : ℝ) (he : 0 < epsilon) :
    |s / ruler epsilon s - r / ruler epsilon r| ≤ (1 / epsilon) * |s-r| := by
  simpa only [Real.norm_eq_abs, abs_of_nonneg
    (div_nonneg (sq_nonneg epsilon) (pow_pos (ruler_pos epsilon _ he) 3).le)] using
    Convex.norm_image_sub_le_of_norm_hasDerivWithin_le
      (fun t (_ : t ∈ (univ : Set ℝ)) => (hasDerivAt_ruler_slope epsilon t he).hasDerivWithinAt)
      (fun t _ => by
        simpa only [Real.norm_eq_abs, abs_of_nonneg
          (div_nonneg (sq_nonneg epsilon) (pow_pos (ruler_pos epsilon t he) 3).le)] using
          ruler_second_le epsilon t he)
      convex_univ (mem_univ r) (mem_univ s)

theorem ruler_second_sub_le (epsilon r s : ℝ) (he : 0 < epsilon) :
    |epsilon ^ 2 / ruler epsilon s ^ 3 - epsilon ^ 2 / ruler epsilon r ^ 3| ≤
      (3 / epsilon ^ 2) * |s-r| := by
  simpa only [Real.norm_eq_abs] using
    Convex.norm_image_sub_le_of_norm_hasDerivWithin_le
      (fun t (_ : t ∈ (univ : Set ℝ)) => (hasDerivAt_ruler_second epsilon t he).hasDerivWithinAt)
      (fun t _ => ruler_third_abs_le epsilon t he)
      convex_univ (mem_univ r) (mem_univ s)

section Normed

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

local instance rulerBoundsBilinearNorm : NormSMulClass ℝ (E →L[ℝ] E →L[ℝ] ℝ) :=
  @NormedSpace.toNormSMulClass ℝ (E →L[ℝ] E →L[ℝ] ℝ) _ _ inferInstance

theorem norm_smul_difference_le (c d : ℝ) (x y : E) :
    ‖d • y - c • x‖ ≤ |d| * ‖y-x‖ + |d-c| * ‖x‖ := by
  have heq : d • y - c • x = d • (y-x) + (d-c) • x := by
    simp only [smul_sub, sub_smul]
    abel
  simpa only [heq, norm_smul, Real.norm_eq_abs] using norm_add_le (d • (y-x)) ((d-c) • x)

theorem norm_rankOne_difference_le (d e : E →L[ℝ] ℝ) :
    ‖e.smulRight e - d.smulRight d‖ ≤ (‖e‖ + ‖d‖) * ‖e-d‖ := by
  have heq : e.smulRight e - d.smulRight d =
      (e-d).smulRight e + d.smulRight (e-d) := by
    apply ContinuousLinearMap.ext
    intro v
    apply ContinuousLinearMap.ext
    intro w
    simp only [ContinuousLinearMap.sub_apply, ContinuousLinearMap.add_apply,
      ContinuousLinearMap.smulRight_apply, ContinuousLinearMap.smul_apply, smul_eq_mul]
    ring
  calc
    _ ≤ ‖(e-d).smulRight e‖ + ‖d.smulRight (e-d)‖ := by
      simpa only [heq] using norm_add_le ((e-d).smulRight e) (d.smulRight (e-d))
    _ = _ := by rw [ContinuousLinearMap.norm_smulRight_apply, ContinuousLinearMap.norm_smulRight_apply]; ring

/-- The actual second-chain-rule expression, used before Riesz dualization. -/
def regularizedForm (epsilon r : ℝ) (d : E →L[ℝ] ℝ)
    (H : E →L[ℝ] E →L[ℝ] ℝ) : E →L[ℝ] E →L[ℝ] ℝ :=
  (r / ruler epsilon r) • H +
    (epsilon ^ 2 / ruler epsilon r ^ 3) • d.smulRight d

theorem norm_regularizedForm_le (epsilon r G K : ℝ) (he : 0 < epsilon)
    (d : E →L[ℝ] ℝ) (H : E →L[ℝ] E →L[ℝ] ℝ)
    (hd : ‖d‖ ≤ G) (hH : ‖H‖ ≤ K) :
    ‖regularizedForm epsilon r d H‖ ≤ K + G ^ 2 / epsilon := by
  have hG : 0 ≤ G := (norm_nonneg d).trans hd
  calc
    _ ≤ ‖(r / ruler epsilon r) • H‖ +
        ‖(epsilon ^ 2 / ruler epsilon r ^ 3) • d.smulRight d‖ := by
      exact norm_add_le ((r / ruler epsilon r) • H)
        ((epsilon ^ 2 / ruler epsilon r ^ 3) • d.smulRight d)
    _ ≤ 1 * K + (1 / epsilon) * (G * G) := by
      rw [norm_smul, norm_smul, Real.norm_eq_abs, Real.norm_eq_abs,
        abs_of_nonneg (div_nonneg (sq_nonneg epsilon) (pow_pos (ruler_pos epsilon r he) 3).le),
        ContinuousLinearMap.norm_smulRight_apply]
      exact add_le_add
        (mul_le_mul (ruler_slope_abs_le epsilon r he) hH (norm_nonneg H) zero_le_one)
        (mul_le_mul (ruler_second_le epsilon r he)
          (mul_le_mul hd hd (norm_nonneg d) hG) (mul_self_nonneg _) (by positivity))
    _ = _ := by ring

theorem regularizedForm_sub_le (epsilon r s G K T delta : ℝ)
    (he : 0 < epsilon) (hG : 0 ≤ G) (hdelta : 0 ≤ delta)
    (d e : E →L[ℝ] ℝ) (H J : E →L[ℝ] E →L[ℝ] ℝ)
    (hd : ‖d‖ ≤ G) (he' : ‖e‖ ≤ G) (hH : ‖H‖ ≤ K)
    (hrs : |s-r| ≤ G*delta) (hde : ‖e-d‖ ≤ K*delta) (hHJ : ‖J-H‖ ≤ T*delta) :
    ‖regularizedForm epsilon s e J - regularizedForm epsilon r d H‖ ≤
      (T + 3*G*K/epsilon + 3*G^3/epsilon^2) * delta := by
  have hs : |s / ruler epsilon s - r / ruler epsilon r| ≤ (1/epsilon)*(G*delta) :=
    (ruler_slope_sub_le epsilon r s he).trans (mul_le_mul_of_nonneg_left hrs (by positivity))
  have hc : |epsilon^2 / ruler epsilon s^3 - epsilon^2 / ruler epsilon r^3| ≤
      (3/epsilon^2)*(G*delta) :=
    (ruler_second_sub_le epsilon r s he).trans (mul_le_mul_of_nonneg_left hrs (by positivity))
  have hR : ‖e.smulRight e - d.smulRight d‖ ≤ (G+G)*(K*delta) :=
    (norm_rankOne_difference_le d e).trans
      (mul_le_mul (add_le_add he' hd) hde (norm_nonneg _) (by positivity))
  have hlin : ‖(s / ruler epsilon s) • J - (r / ruler epsilon r) • H‖ ≤
      1*(T*delta) + ((1/epsilon)*(G*delta))*K :=
    (norm_smul_difference_le _ _ H J).trans (add_le_add
      (mul_le_mul (ruler_slope_abs_le epsilon s he) hHJ (norm_nonneg (J-H)) zero_le_one)
      (mul_le_mul hs hH (norm_nonneg H) (by positivity)))
  have hca : |epsilon^2 / ruler epsilon s^3| ≤ 1/epsilon := by
    simpa only [abs_of_nonneg (div_nonneg (sq_nonneg epsilon)
      (pow_pos (ruler_pos epsilon s he) 3).le)] using ruler_second_le epsilon s he
  have hquad : ‖(epsilon^2 / ruler epsilon s^3) • e.smulRight e -
      (epsilon^2 / ruler epsilon r^3) • d.smulRight d‖ ≤
      (1/epsilon)*((G+G)*(K*delta)) + ((3/epsilon^2)*(G*delta))*(G*G) := by
    refine (norm_smul_difference_le _ _ (d.smulRight d) (e.smulRight e)).trans ?_
    exact add_le_add
      (mul_le_mul hca hR (norm_nonneg (e.smulRight e - d.smulRight d)) (by positivity))
      (mul_le_mul hc (by
        rw [ContinuousLinearMap.norm_smulRight_apply]
        exact mul_le_mul hd hd (norm_nonneg d) hG)
        (norm_nonneg (d.smulRight d)) (by positivity))
  calc
    _ ≤ ‖(s / ruler epsilon s) • J - (r / ruler epsilon r) • H‖ +
        ‖(epsilon^2 / ruler epsilon s^3) • e.smulRight e -
          (epsilon^2 / ruler epsilon r^3) • d.smulRight d‖ := by
      rw [regularizedForm, regularizedForm, add_sub_add_comm]
      exact norm_add_le
        ((s / ruler epsilon s) • J - (r / ruler epsilon r) • H)
        ((epsilon^2 / ruler epsilon s^3) • e.smulRight e -
          (epsilon^2 / ruler epsilon r^3) • d.smulRight d)
    _ ≤ _ := by
      convert add_le_add hlin hquad using 1
      ring

end Normed

end FractalSync.RulerJetBounds
