import FractalSync.OperationalRulerBounds
import FractalSync.BoundedLipschitz

/-!
# Local regularizer estimates near a small field value

The scalar third-derivative estimate retains the small field factor. The
resulting regional certificates concern actual derivatives on a convex set.
-/

noncomputable section

namespace FractalSync.LocalRulerBounds

open Set RulerJetBounds

theorem unit_slope_abs_le (r : ℝ) (hr : |r| ≤ 1 / 4) :
    |r / ruler 1 r| ≤ 1 / 4 := by
  rw [abs_div, abs_of_pos (ruler_pos 1 r (by norm_num))]
  exact (div_le_self (abs_nonneg r) (epsilon_le_ruler 1 r (by norm_num))).trans hr

theorem unit_ruler_le_two (r : ℝ) (hr : |r| ≤ 1 / 4) : ruler 1 r ≤ 2 := by
  apply (Real.sqrt_le_iff).2
  exact ⟨by norm_num, by nlinarith [sq_abs r,
    (sq_le_sq₀ (abs_nonneg r) (by norm_num : (0 : ℝ) ≤ 1 / 4)).2 hr]⟩

theorem unit_third_abs_le (r : ℝ) (hr : |r| ≤ 1 / 4) :
    |-3 * (1 : ℝ) ^ 2 * r / ruler 1 r ^ 5| ≤ 3 / 4 := by
  rw [abs_div, abs_mul, abs_mul, abs_of_pos (pow_pos (ruler_pos 1 r (by norm_num)) 5)]
  norm_num only [abs_neg, one_pow, abs_one, mul_one]
  have hp : (1 : ℝ) ≤ ruler 1 r ^ 5 := one_le_pow₀ (epsilon_le_ruler 1 r (by norm_num))
  exact (div_le_self (by positivity) hp).trans (by linarith)

theorem unit_second_sub_le (r s : ℝ) (hr : |r| ≤ 1 / 4) (hs : |s| ≤ 1 / 4) :
    |1 / ruler 1 s ^ 3 - 1 / ruler 1 r ^ 3| ≤ (3 / 4 : ℝ) * |s - r| := by
  have h := Convex.norm_image_sub_le_of_norm_hasDerivWithin_le
    (fun t (_ : t ∈ Icc (-(1 / 4) : ℝ) (1 / 4)) =>
      (hasDerivAt_ruler_second 1 t (by norm_num)).hasDerivWithinAt)
    (fun t ht => by simpa only [Real.norm_eq_abs] using
      unit_third_abs_le t (abs_le.mpr ht))
    (convex_Icc (-(1 / 4) : ℝ) (1 / 4)) (abs_le.mp hr) (abs_le.mp hs)
  simpa only [one_pow, Real.norm_eq_abs] using h

section Normed

variable {E X : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E] [PseudoMetricSpace X]

local instance localRulerBilinearNorm : NormSMulClass ℝ (E →L[ℝ] E →L[ℝ] ℝ) :=
  @NormedSpace.toNormSMulClass ℝ (E →L[ℝ] E →L[ℝ] ℝ) _ _ inferInstance

theorem slope_boundedLipschitz {f : X → ℝ} {s : Set X}
    (hf : BoundedLipschitzOn f s (1 / 4) 4) :
    BoundedLipschitzOn (fun x => f x / ruler 1 (f x)) s (1 / 4) 4 := by
  refine ⟨by norm_num, by norm_num, ?_, ?_⟩
  case refine_1 =>
    exact fun x hx => unit_slope_abs_le (f x) (hf.norm_le x hx)
  exact fun x hx y hy => (show _ ≤ |f x - f y| from by
    simpa only [one_div, inv_one, one_mul, Real.norm_eq_abs] using
      ruler_slope_sub_le 1 (f y) (f x) (by norm_num)).trans (hf.norm_sub_le x hx y hy)

theorem second_boundedLipschitz {f : X → ℝ} {s : Set X}
    (hf : BoundedLipschitzOn f s (1 / 4) 4) :
    BoundedLipschitzOn (fun x => 1 / ruler 1 (f x) ^ 3) s 1 3 := by
  refine ⟨by norm_num, by norm_num, ?_, ?_⟩
  case refine_1 =>
    exact fun x _ => by
      simpa only [one_pow, div_one, Real.norm_eq_abs,
        abs_of_pos (one_div_pos.mpr (pow_pos (ruler_pos 1 (f x) (by norm_num)) 3))] using
        ruler_second_le 1 (f x) (by norm_num)
  exact fun x hx y hy => (unit_second_sub_le (f y) (f x) (hf.norm_le y hy) (hf.norm_le x hx)).trans
    (by have hb : |f x - f y| ≤ 4 * dist x y := hf.norm_sub_le x hx y hy
        linarith)

theorem rankOne_boundedLipschitz {d : X → E →L[ℝ] ℝ} {s : Set X}
    (hd : BoundedLipschitzOn d s 4 12) :
    BoundedLipschitzOn (fun x => (d x).smulRight (d x)) s 16 96 := by
  refine ⟨by norm_num, by norm_num, ?_, ?_⟩
  case refine_1 =>
    exact fun x hx => by
      rw [ContinuousLinearMap.norm_smulRight_apply]
      nlinarith [hd.norm_le x hx, norm_nonneg (d x)]
  exact fun x hx y hy => (norm_rankOne_difference_le (d y) (d x)).trans (by
    nlinarith [mul_le_mul (add_le_add (hd.norm_le x hx) (hd.norm_le y hy))
      (hd.norm_sub_le x hx y hy) (norm_nonneg (d x - d y)) (by norm_num : (0 : ℝ) ≤ 4 + 4)])

theorem regularizedForm_boundedLipschitz {f : X → ℝ} {d : X → E →L[ℝ] ℝ}
    {H : X → E →L[ℝ] E →L[ℝ] ℝ} {s : Set X}
    (hf : BoundedLipschitzOn f s (1 / 4) 4)
    (hd : BoundedLipschitzOn d s 4 12) (hH : BoundedLipschitzOn H s 12 1) :
    BoundedLipschitzOn (fun x => regularizedForm 1 (f x) (d x) (H x)) s 20 200 := by
  have hb := ((slope_boundedLipschitz hf).smul hH).add
    ((second_boundedLipschitz hf).smul (rankOne_boundedLipschitz hd))
  simpa only [regularizedForm, one_pow] using hb.enlarge
    (by norm_num : (1 / 4 : ℝ) * 12 + 1 * 16 ≤ 20)
    (by norm_num : (1 / 4 : ℝ) * 1 + 12 * 4 + (1 * 96 + 16 * 3) ≤ 200)

end Normed

open OperationalDynamics OperationalField FlatShoreIntrinsic

theorem local_gradient_norm (C : Lattice) (o : Coords) (h : ℝ) (x : Plane)
    (hf : |physicalField C o h (chart x)| ≤ 1 / 4)
    (hd : ‖euclideanDifferential C o h x‖ ≤ 4) :
    ‖gradient (rulerField C o h 1) x‖ ≤ 1 := by
  rw [gradient, (InnerProductSpace.toDual ℝ Plane).symm.norm_map,
    (hasFDerivAt_rulerField C o h 1 (by norm_num) x).fderiv, norm_smul, Real.norm_eq_abs]
  exact (mul_le_mul (unit_slope_abs_le _ hf) hd (norm_nonneg _) (by norm_num)).trans (by norm_num)

theorem local_hessian_boundedLipschitz (C : Lattice) (o : Coords) (h : ℝ) (s : Set Plane)
    (hf : BoundedLipschitzOn (fun x => physicalField C o h (chart x)) s (1 / 4) 4)
    (hd : BoundedLipschitzOn (euclideanDifferential C o h) s 4 12)
    (hH : BoundedLipschitzOn (euclideanHessianForm C o h) s 12 1) :
    BoundedLipschitzOn (fderiv ℝ (gradient (rulerField C o h 1))) s 20 200 := by
  have hb := regularizedForm_boundedLipschitz hf hd hH
  refine ⟨by norm_num, by norm_num, ?_, ?_⟩
  case refine_1 =>
    exact fun x hx => by
      simpa only [(hasFDerivAt_gradient_rulerField C o h 1 (by norm_num) x).fderiv] using
        (norm_rieszHessian_le (rulerBilinear C o h 1 x)).trans (hb.norm_le x hx)
  exact fun x hx y hy => by
    simpa only [(hasFDerivAt_gradient_rulerField C o h 1 (by norm_num) x).fderiv,
      (hasFDerivAt_gradient_rulerField C o h 1 (by norm_num) y).fderiv, ← map_sub] using
      (norm_rieszHessian_le (rulerBilinear C o h 1 x - rulerBilinear C o h 1 y)).trans
        (hb.norm_sub_le x hx y hy)

theorem local_gradient_boundedLipschitz (C : Lattice) (o : Coords) (h : ℝ) (s : Set Plane)
    (hs : Convex ℝ s)
    (hf : BoundedLipschitzOn (fun x => physicalField C o h (chart x)) s (1 / 4) 4)
    (hd : BoundedLipschitzOn (euclideanDifferential C o h) s 4 12)
    (hH : BoundedLipschitzOn (euclideanHessianForm C o h) s 12 1) :
    BoundedLipschitzOn (gradient (rulerField C o h 1)) s 1 20 := by
  refine ⟨by norm_num, by norm_num,
    fun x hx => local_gradient_norm C o h x (hf.norm_le x hx) (hd.norm_le x hx), ?_⟩
  exact fun x hx y hy => by
    simpa only [dist_eq_norm] using Convex.norm_image_sub_le_of_norm_fderiv_le
      (fun z (_ : z ∈ s) => (hasFDerivAt_gradient_rulerField C o h 1 (by norm_num) z).differentiableAt)
      (local_hessian_boundedLipschitz C o h s hf hd hH).norm_le hs hy hx

theorem local_ruler_boundedLipschitz (C : Lattice) (o : Coords) (h : ℝ) (s : Set Plane)
    (hs : Convex ℝ s)
    (hf : BoundedLipschitzOn (fun x => physicalField C o h (chart x)) s (1 / 4) 4)
    (hd : BoundedLipschitzOn (euclideanDifferential C o h) s 4 12) :
    BoundedLipschitzOn (rulerField C o h 1) s 2 1 := by
  refine ⟨by norm_num, by norm_num, ?_, ?_⟩
  case refine_1 =>
    exact fun x hx => by
      simpa only [rulerField, Real.norm_eq_abs,
        abs_of_pos (ruler_pos 1 _ (by norm_num))] using
        unit_ruler_le_two _ (hf.norm_le x hx)
  exact fun x hx y hy => by
    simpa only [dist_eq_norm] using Convex.norm_image_sub_le_of_norm_fderiv_le
      (fun z (_ : z ∈ s) => (hasFDerivAt_rulerField C o h 1 (by norm_num) z).differentiableAt)
      (fun z hz => by
        simpa only [gradient, (InnerProductSpace.toDual ℝ Plane).symm.norm_map] using
          local_gradient_norm C o h z (hf.norm_le z hz) (hd.norm_le z hz)) hs hy hx

end FractalSync.LocalRulerBounds
