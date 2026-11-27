import FractalSync.ConformalRulerContracts
import Mathlib.MeasureTheory.Integral.IntervalIntegral.IntegrationByParts

/-!
# Path replacement and quadrature for the planar rotation angle

The angle is a real lift in the oriented conformal coordinate frame. All
line-integral estimates use exact real arithmetic and the chosen coordinate
norm. The path replacement estimate uses position error and path lengths.
-/

namespace FractalSync.ConformalTransportError

open MeasureTheory Set

section Path

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The scalar line integral of a one-form along a path with the supplied velocity. -/
noncomputable def lineIntegral (α : E → E →L[ℝ] ℝ) (x v : ℝ → E) (a b : ℝ) : ℝ :=
  ∫ t in a..b, α (x t) (v t)

/-- The two path angles differ by at most their positional deviation times
the spatial derivative bound and the sum of their lengths, after subtracting
the oriented endpoint discrepancies paired with the reconstruction's form. -/
theorem lineIntegral_path_error (α : E → E →L[ℝ] ℝ) (x y u v : ℝ → E)
    (a b error L : ℝ) (hab : a ≤ b) (hL : 0 ≤ L)
    (hx : ∀ t ∈ Icc a b, HasDerivAt x (u t) t)
    (hy : ∀ t ∈ Icc a b, HasDerivAt y (v t) t)
    (hu : ContinuousOn u (Icc a b)) (hv : ContinuousOn v (Icc a b))
    (A : ℝ → E →L[ℝ] ℝ) (hA : ContinuousOn A (Icc a b))
    (hd : ∀ t ∈ Icc a b, HasDerivAt (α ∘ y) (A t) t)
    (hα : ContinuousOn (α ∘ x) (Icc a b))
    (hvariation : ∀ t ∈ Icc a b, ‖α (x t) - α (y t)‖ ≤ L * ‖x t - y t‖)
    (hderiv : ∀ t ∈ Icc a b, ‖A t‖ ≤ L * ‖v t‖)
    (herror : ∀ t ∈ Icc a b, ‖x t - y t‖ ≤ error) :
    |lineIntegral α x u a b - lineIntegral α y v a b -
      (α (y b) (x b - y b) - α (y a) (x a - y a))| ≤
      L * error * ((∫ t in a..b, ‖u t‖) + ∫ t in a..b, ‖v t‖) := by
  let f := fun t => (α (x t) - α (y t)) (u t) - A t (x t - y t)
  have hyα : ContinuousOn (α ∘ y) (Icc a b) :=
    fun t ht => (hd t ht).continuousAt.continuousWithinAt
  have hxy : ContinuousOn (fun t => x t - y t) (Icc a b) :=
    fun t ht => ((hx t ht).sub (hy t ht)).continuousAt.continuousWithinAt
  obtain ⟨ix, iy, iA, iΔ⟩ :
      IntervalIntegrable (fun t => α (x t) (u t)) volume a b ∧
      IntervalIntegrable (fun t => α (y t) (v t)) volume a b ∧
      IntervalIntegrable (fun t => A t (x t - y t)) volume a b ∧
      IntervalIntegrable (fun t => α (y t) (u t - v t)) volume a b :=
    ⟨(hα.clm_apply hu).intervalIntegrable_of_Icc hab,
      (hyα.clm_apply hv).intervalIntegrable_of_Icc hab,
      (hA.clm_apply hxy).intervalIntegrable_of_Icc hab,
      (hyα.clm_apply (hu.sub hv)).intervalIntegrable_of_Icc hab⟩
  have parts :
      (∫ t in a..b, A t (x t - y t) + α (y t) (u t - v t)) =
        α (y b) (x b - y b) - α (y a) (x a - y a) :=
    intervalIntegral.integral_eq_sub_of_hasDerivAt
      (fun t ht =>
        let hi := (uIcc_of_le hab ▸ ht)
        (hd t hi).clm_apply ((hx t hi).sub (hy t hi)))
      (iA.add iΔ)
  have integrand (t : ℝ) : f t =
      (α (x t) (u t) - α (y t) (v t)) -
        (A t (x t - y t) + α (y t) (u t - v t)) := by
    simp only [f, ContinuousLinearMap.sub_apply, map_sub]
    ring
  rw [lineIntegral, lineIntegral, ← parts,
    ← intervalIntegral.integral_sub ix iy,
    ← intervalIntegral.integral_sub (ix.sub iy) (iA.add iΔ)]
  simp_rw [← integrand]
  have bound (t : ℝ) (ht : t ∈ Icc a b) : |f t| ≤ L * error * (‖u t‖ + ‖v t‖) :=
    let apply_bound (B : E →L[ℝ] ℝ) (w : E) : |B w| ≤ ‖B‖ * ‖w‖ :=
      Real.norm_eq_abs (B w) ▸ B.le_opNorm w
    calc
      |f t| ≤ |(α (x t) - α (y t)) (u t)| + |A t (x t - y t)| := abs_sub _ _
      _ ≤ L * error * ‖u t‖ + L * ‖v t‖ * error := add_le_add
        ((apply_bound _ _).trans (mul_le_mul_of_nonneg_right
          ((hvariation t ht).trans (mul_le_mul_of_nonneg_left (herror t ht) hL)) (norm_nonneg _)))
        ((apply_bound _ _).trans ((mul_le_mul_of_nonneg_right (hderiv t ht) (norm_nonneg _)).trans
          (mul_le_mul_of_nonneg_left (herror t ht) (mul_nonneg hL (norm_nonneg _)))))
      _ = _ := by ring
  calc
    |∫ t in a..b, f t| ≤ ∫ t in a..b, |f t| :=
      intervalIntegral.abs_integral_le_integral_abs hab
    _ ≤ ∫ t in a..b, L * error * (‖u t‖ + ‖v t‖) :=
      intervalIntegral.integral_mono_on hab
        ((((hα.sub hyα).clm_apply hu).sub
          (hA.clm_apply hxy)).intervalIntegrable_of_Icc hab).abs
        ((continuousOn_const.mul (hu.norm.add hv.norm)).intervalIntegrable_of_Icc hab) bound
    _ = _ := by
      rw [intervalIntegral.integral_const_mul, intervalIntegral.integral_add
        (hu.norm.intervalIntegrable_of_Icc hab) (hv.norm.intervalIntegrable_of_Icc hab)]

end Path

/-- In the positive Cartesian frame the negative conformal connection splits
into a radial dilation and a signed rotation. Here `p,q` are the Cartesian
partials of the log ruler. The coefficient of the quarter turn is the
rotation one-form evaluated on the base velocity. -/
theorem neg_planarConnection_split (p q : ℝ × ℝ → ℝ) (x v w : ℝ × ℝ) :
    -ConformalRulerContracts.planarConnection p q x v w =
      (p x * v.1 + q x * v.2) • w +
        (p x * v.2 - q x * v.1) • ((-w.2, w.1) : ℝ × ℝ) := by
  ext <;> simp [ConformalRulerContracts.planarConnection] <;> ring

end FractalSync.ConformalTransportError
