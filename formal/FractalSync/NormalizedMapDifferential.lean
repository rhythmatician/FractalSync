import FractalSync.NormalizedDistanceFields
import Mathlib.Analysis.Calculus.MeanValue

/-!
# Actual differentials and quadratic remainders of the normalized Map

Development-time contracts for #108/#159. The anchor and physical scale are
held fixed while differentiating the window coordinate. All derivatives are
actual Fréchet derivatives; the hypotheses do not certify a numerical provider.
-/

open Set
open scoped NNReal

namespace FractalSync.NormalizedDistanceFields

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- At fixed anchor and nonzero scale, the window differential is the physical
field differential at the sampled point. The two scale factors cancel. -/
theorem hasFDerivAt_normalizedMap (D : E → ℝ) (c u : E) (h : ℝ)
    (D' : E →L[ℝ] ℝ) (hh : h ≠ 0)
    (hD : HasFDerivAt D D' (c + h • u)) :
    HasFDerivAt (fun v => normalizedMap D c v h) D' u := by
  have ha := ((hasFDerivAt_id (𝕜 := ℝ) u).const_smul h).const_add c
  have hn := ((hD.comp u ha).sub_const (D c)).const_smul h⁻¹
  simpa [normalizedMap, normalizedDifference, Function.comp_def, div_eq_mul_inv,
    Pi.smul_def, smul_smul, smul_eq_mul, hh, mul_comm] using hn

/-- In particular, the window differential at the anchor is exactly `dD(c)`.
This differentiates the window coordinate, not the anchor-dependent scale. -/
theorem hasFDerivAt_normalizedMap_zero (D : E → ℝ) (c : E) (h : ℝ)
    (D' : E →L[ℝ] ℝ) (hh : h ≠ 0) (hD : HasFDerivAt D D' c) :
    HasFDerivAt (fun u => normalizedMap D c u h) D' 0 := by
  simpa using hasFDerivAt_normalizedMap D c 0 h D' hh (by simpa using hD)

/-- Equality of the actual differential fields at every differentiable query. -/
theorem fderiv_normalizedMap (D : E → ℝ) (c u : E) (h : ℝ) (hh : h ≠ 0)
    (hD : DifferentiableAt ℝ D (c + h • u)) :
    fderiv ℝ (fun v => normalizedMap D c v h) u = fderiv ℝ D (c + h • u) := by
  exact (hasFDerivAt_normalizedMap D c u h _ hh hD.hasFDerivAt).fderiv

/-- A field Lipschitz bound on just the physical query segment controls the
normalized value. Taking `L = 1` gives the ideal signed-distance contract. -/
theorem normalizedMap_value_bound_on_segment (D : E → ℝ) (c u : E)
    (h : ℝ) (L : ℝ≥0) (hh : h ≠ 0)
    (hLip : LipschitzOnWith L D (segment ℝ c (c + h • u))) :
    |normalizedMap D c u h| ≤ (L : ℝ) * ‖u‖ := by
  rw [normalizedMap, normalizedDifference, abs_div, div_le_iff₀ (abs_pos.mpr hh)]
  simpa [dist_eq_norm, norm_smul, Real.norm_eq_abs, mul_assoc, mul_left_comm, mul_comm]
    using hLip.dist_le_mul (c + h • u) (right_mem_segment ℝ c (c + h • u))
      c (left_mem_segment ℝ c (c + h • u))

/-- A first-order field remainder with the factor `1/2`. Only the actual
differential's variation from the anchor along this segment is bounded. -/
theorem field_quadratic_remainder (D : E → ℝ) (c v : E) (L : ℝ)
    (hD : ∀ t ∈ Icc (0 : ℝ) 1, DifferentiableAt ℝ D (c + t • v))
    (hbound : ∀ t ∈ Icc (0 : ℝ) 1,
      ‖fderiv ℝ D (c + t • v) - fderiv ℝ D c‖ ≤ L * ‖t • v‖) :
    |D (c + v) - D c - fderiv ℝ D c v| ≤ L / 2 * ‖v‖ ^ 2 := by
  have hg : ∀ t ∈ Icc (0 : ℝ) 1,
      HasDerivAt (fun s => D (c + s • v) - D c - s * fderiv ℝ D c v)
        ((fderiv ℝ D (c + t • v) - fderiv ℝ D c) v) t := by
    intro t ht
    simpa [Function.comp_def] using
      (((hD t ht).hasFDerivAt.comp_hasDerivAt t
        (((hasDerivAt_id t).smul_const v).const_add c)).sub_const (D c)).sub
          ((hasDerivAt_id t).mul_const (fderiv ℝ D c v))
  have hB : ∀ t : ℝ,
      HasDerivAt (fun s => L / 2 * ‖v‖ ^ 2 * s ^ 2) (L * ‖v‖ ^ 2 * t) t := by
    intro t
    convert ((hasDerivAt_id t).mul (hasDerivAt_id t)).const_mul
      (L / 2 * ‖v‖ ^ 2) using 1
    all_goals simp only [pow_two, Pi.mul_apply, id_eq, one_mul, mul_one]
    ring
  have hb : ∀ t ∈ Ico (0 : ℝ) 1,
      ‖(fderiv ℝ D (c + t • v) - fderiv ℝ D c) v‖ ≤ L * ‖v‖ ^ 2 * t := by
    intro t ht
    calc
      ‖(fderiv ℝ D (c + t • v) - fderiv ℝ D c) v‖
          ≤ ‖fderiv ℝ D (c + t • v) - fderiv ℝ D c‖ * ‖v‖ :=
        (fderiv ℝ D (c + t • v) - fderiv ℝ D c).le_opNorm v
      _ ≤ (L * ‖t • v‖) * ‖v‖ :=
        mul_le_mul_of_nonneg_right (hbound t (Ico_subset_Icc_self ht)) (norm_nonneg v)
      _ = L * ‖v‖ ^ 2 * t := by
        rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg ht.1]
        ring
  have he := image_norm_le_of_norm_deriv_right_le_deriv_boundary
    (B := fun s => L / 2 * ‖v‖ ^ 2 * s ^ 2)
    (fun t ht => (hg t ht).continuousAt.continuousWithinAt)
    (fun t ht => (hg t (Ico_subset_Icc_self ht)).hasDerivWithinAt)
    (by simp) hB hb (show (1 : ℝ) ∈ Icc 0 1 by simp)
  simpa [Real.norm_eq_abs] using he

/-- The normalized residual is quadratic in the window coordinate and linear
in the absolute physical scale, under an anchored differential-variation bound. -/
theorem normalizedMap_quadratic_remainder (D : E → ℝ) (c u : E) (h L : ℝ)
    (hh : h ≠ 0)
    (hD : ∀ t ∈ Icc (0 : ℝ) 1, DifferentiableAt ℝ D (c + t • (h • u)))
    (hbound : ∀ t ∈ Icc (0 : ℝ) 1,
      ‖fderiv ℝ D (c + t • (h • u)) - fderiv ℝ D c‖ ≤ L * ‖t • (h • u)‖) :
    |normalizedMap D c u h - fderiv ℝ D c u| ≤ L * |h| / 2 * ‖u‖ ^ 2 := by
  have hR := field_quadratic_remainder D c (h • u) L hD hbound
  have he : normalizedMap D c u h - fderiv ℝ D c u =
      (D (c + h • u) - D c - fderiv ℝ D c (h • u)) / h := by
    simp [normalizedMap, normalizedDifference, sub_div, map_smul, smul_eq_mul, hh]
  rw [he, abs_div, div_le_iff₀ (abs_pos.mpr hh)]
  convert hR using 1
  rw [norm_smul, Real.norm_eq_abs]
  ring

/-- A Lipschitz bound on the actual differential restricted to the physical
query segment supplies the quadratic normalized residual bound. -/
theorem normalizedMap_quadratic_remainder_of_lipschitz_fderiv
    (D : E → ℝ) (c u : E) (h : ℝ) (L : ℝ≥0) (hh : h ≠ 0)
    (hD : ∀ z ∈ segment ℝ c (c + h • u), DifferentiableAt ℝ D z)
    (hLip : LipschitzOnWith L (fderiv ℝ D) (segment ℝ c (c + h • u))) :
    |normalizedMap D c u h - fderiv ℝ D c u| ≤ (L : ℝ) * |h| / 2 * ‖u‖ ^ 2 := by
  have hp : ∀ t ∈ Icc (0 : ℝ) 1,
      c + t • (h • u) ∈ segment ℝ c (c + h • u) := fun _ ht =>
    (convex_segment c (c + h • u)).add_smul_mem
      (left_mem_segment ℝ c (c + h • u)) (right_mem_segment ℝ c (c + h • u)) ht
  refine normalizedMap_quadratic_remainder D c u h L hh (fun t ht => hD _ (hp t ht)) ?_
  exact fun t ht => by
    simpa [dist_eq_norm] using
      hLip.dist_le_mul (c + t • (h • u)) (hp t ht) c (left_mem_segment ℝ c (c + h • u))

/-- The requested Hessian-bound contract: the derivative of the actual
`fderiv` field is bounded on the whole physical query segment. -/
theorem normalizedMap_quadratic_remainder_of_hessian_bound
    (D : E → ℝ) (c u : E) (h : ℝ) (L : ℝ≥0) (hh : h ≠ 0)
    (H : E → E →L[ℝ] (E →L[ℝ] ℝ))
    (hD : ∀ z ∈ segment ℝ c (c + h • u), DifferentiableAt ℝ D z)
    (hH : ∀ z ∈ segment ℝ c (c + h • u), HasFDerivAt (fderiv ℝ D) (H z) z)
    (hbound : ∀ z ∈ segment ℝ c (c + h • u), ‖H z‖ ≤ L) :
    |normalizedMap D c u h - fderiv ℝ D c u| ≤ (L : ℝ) * |h| / 2 * ‖u‖ ^ 2 := by
  refine normalizedMap_quadratic_remainder_of_lipschitz_fderiv D c u h L hh hD ?_
  exact (convex_segment c (c + h • u)).lipschitzOnWith_of_nnnorm_hasFDerivWithin_le
    (fun z hz => (hH z hz).hasFDerivWithinAt) (fun z hz => hbound z hz)

end FractalSync.NormalizedDistanceFields
