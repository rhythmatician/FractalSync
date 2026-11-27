import FractalSync.NormalizedMapDifferential
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.Calculus.FDeriv.Symmetric
import Mathlib.Analysis.InnerProductSpace.PiL2

/-!
# Certified exact-real stencil errors

The provider's central gradient and nine-point Hessian stencil are compared to
actual derivatives of one underlying field. Smoothness and value-error bounds
are explicit hypotheses. No theorem asserts that the runtime's flags, raster,
or floating-point evaluation supplies those certificates automatically.
-/

open Set Filter Topology
open scoped NNReal

namespace FractalSync.ProviderCertificates

variable {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

/-- Vector-valued first-order Taylor remainder on one actual query segment. -/
theorem first_order_remainder_bound (f : E → F) (f' : E → E →L[ℝ] F)
    (c v : E) (L : ℝ)
    (hf : ∀ t ∈ Icc (0 : ℝ) 1, HasFDerivAt f (f' (c + t • v)) (c + t • v))
    (hb : ∀ t ∈ Icc (0 : ℝ) 1, ‖f' (c + t • v) - f' c‖ ≤ L * ‖t • v‖) :
    ‖f (c + v) - f c - f' c v‖ ≤ L / 2 * ‖v‖ ^ 2 := by
  have hd : ∀ t ∈ Icc (0 : ℝ) 1,
      HasDerivAt (fun s => f (c + s • v) - f c - s • f' c v)
        ((f' (c + t • v) - f' c) v) t := by
    intro t ht
    simpa [Function.comp_def] using
      (((hf t ht).comp_hasDerivAt t
        (((hasDerivAt_id t).smul_const v).const_add c)).sub_const (f c)).sub
          ((hasDerivAt_id t).smul_const (f' c v))
  have hB : ∀ t : ℝ,
      HasDerivAt (fun s => L / 2 * ‖v‖ ^ 2 * s ^ 2) (L * ‖v‖ ^ 2 * t) t := by
    intro t
    convert ((hasDerivAt_id t).mul (hasDerivAt_id t)).const_mul
      (L / 2 * ‖v‖ ^ 2) using 1
    all_goals simp only [pow_two, Pi.mul_apply, id_eq, one_mul, mul_one]
    ring
  have hbound : ∀ t ∈ Ico (0 : ℝ) 1,
      ‖(f' (c + t • v) - f' c) v‖ ≤ L * ‖v‖ ^ 2 * t := by
    intro t ht
    calc
      ‖(f' (c + t • v) - f' c) v‖ ≤ ‖f' (c + t • v) - f' c‖ * ‖v‖ :=
        (f' (c + t • v) - f' c).le_opNorm v
      _ ≤ (L * ‖t • v‖) * ‖v‖ :=
        mul_le_mul_of_nonneg_right (hb t (Ico_subset_Icc_self ht)) (norm_nonneg v)
      _ = L * ‖v‖ ^ 2 * t := by
        rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg ht.1]
        ring
  have h := image_norm_le_of_norm_deriv_right_le_deriv_boundary
    (B := fun s => L / 2 * ‖v‖ ^ 2 * s ^ 2)
    (fun t ht => (hd t ht).continuousAt.continuousWithinAt)
    (fun t ht => (hd t (Ico_subset_Icc_self ht)).hasDerivWithinAt)
    (by simp) hB hbound (show (1 : ℝ) ∈ Icc 0 1 by simp)
  simpa using h

/-- Second-order Taylor remainder from the actual Hessian's anchored
Lipschitz variation. Only the queried radial segment is required. -/
theorem second_order_remainder_bound (f : E → ℝ) (f' : E → E →L[ℝ] ℝ)
    (H : E → E →L[ℝ] (E →L[ℝ] ℝ)) (c v : E) (L : ℝ)
    (hf : ∀ t ∈ Icc (0 : ℝ) 1, HasFDerivAt f (f' (c + t • v)) (c + t • v))
    (hH : ∀ t ∈ Icc (0 : ℝ) 1, HasFDerivAt f' (H (c + t • v)) (c + t • v))
    (hb : ∀ t ∈ Icc (0 : ℝ) 1, ‖H (c + t • v) - H c‖ ≤ L * ‖t • v‖) :
    |f (c + v) - f c - f' c v - H c v v / 2| ≤ L / 6 * ‖v‖ ^ 3 := by
  have hfirst : ∀ t ∈ Icc (0 : ℝ) 1,
      ‖f' (c + t • v) - f' c - H c (t • v)‖ ≤ L / 2 * ‖t • v‖ ^ 2 := by
    intro t ht
    refine first_order_remainder_bound f' H c (t • v) L ?_ ?_
    · intro s hs
      simpa only [smul_smul] using hH (s * t)
        ⟨mul_nonneg hs.1 ht.1, mul_le_one₀ hs.2 ht.1 ht.2⟩
    · intro s hs
      simpa only [smul_smul] using hb (s * t)
        ⟨mul_nonneg hs.1 ht.1, mul_le_one₀ hs.2 ht.1 ht.2⟩
  have hd : ∀ t ∈ Icc (0 : ℝ) 1,
      HasDerivAt
        (fun s => f (c + s • v) - f c - s * f' c v - s ^ 2 / 2 * H c v v)
        ((f' (c + t • v) - f' c - H c (t • v)) v) t := by
    intro t ht
    convert ((((hf t ht).comp_hasDerivAt t
      (((hasDerivAt_id t).smul_const v).const_add c)).sub_const (f c)).sub
      ((hasDerivAt_id t).mul_const (f' c v))).sub
      ((((hasDerivAt_id t).pow 2).div_const 2).mul_const (H c v v)) using 1
    simp only [ContinuousLinearMap.sub_apply, map_smul, ContinuousLinearMap.smul_apply,
      smul_eq_mul, id_eq]
    ring
  have hB : ∀ t : ℝ,
      HasDerivAt (fun s => L / 6 * ‖v‖ ^ 3 * s ^ 3)
        (L / 2 * ‖v‖ ^ 3 * t ^ 2) t := by
    intro t
    convert ((hasDerivAt_id t).pow 3).const_mul (L / 6 * ‖v‖ ^ 3) using 1
    simp only [id_eq]
    ring
  have hbound : ∀ t ∈ Ico (0 : ℝ) 1,
      ‖(f' (c + t • v) - f' c - H c (t • v)) v‖ ≤
        L / 2 * ‖v‖ ^ 3 * t ^ 2 := by
    intro t ht
    calc
      ‖(f' (c + t • v) - f' c - H c (t • v)) v‖ ≤
          ‖f' (c + t • v) - f' c - H c (t • v)‖ * ‖v‖ :=
        (f' (c + t • v) - f' c - H c (t • v)).le_opNorm v
      _ ≤ (L / 2 * ‖t • v‖ ^ 2) * ‖v‖ :=
        mul_le_mul_of_nonneg_right (hfirst t (Ico_subset_Icc_self ht)) (norm_nonneg v)
      _ = L / 2 * ‖v‖ ^ 3 * t ^ 2 := by
        rw [norm_smul, Real.norm_eq_abs, abs_of_nonneg ht.1]
        ring
  have h := image_norm_le_of_norm_deriv_right_le_deriv_boundary
    (B := fun s => L / 6 * ‖v‖ ^ 3 * s ^ 3)
    (fun t ht => (hd t ht).continuousAt.continuousWithinAt)
    (fun t ht => (hd t (Ico_subset_Icc_self ht)).hasDerivWithinAt)
    (by simp) hB hbound (show (1 : ℝ) ∈ Icc 0 1 by simp)
  simpa [Real.norm_eq_abs, div_eq_mul_inv, mul_comm, mul_left_comm] using h

/-- Analytic hypotheses on a convex region containing the stencil. These are
not assertions about the runtime's `Regular` flag. -/
structure LocalJetControl (f : E → ℝ) (c : E) (S : Set E) (L : ℝ) where
  differential : E → E →L[ℝ] ℝ
  hessian : E → E →L[ℝ] (E →L[ℝ] ℝ)
  region_convex : Convex ℝ S
  center_mem : c ∈ S
  first : ∀ z ∈ S, HasFDerivAt f (differential z) z
  second : ∀ z ∈ S, HasFDerivAt differential (hessian z) z
  variation : ∀ z ∈ S, ‖hessian z - hessian c‖ ≤ L * ‖z - c‖

theorem LocalJetControl.remainder {f : E → ℝ} {c : E} {S : Set E} {L : ℝ}
    (J : LocalJetControl f c S L) (v : E) (hv : c + v ∈ S) :
    |f (c + v) - f c - J.differential c v - J.hessian c v v / 2| ≤
      L / 6 * ‖v‖ ^ 3 := by
  have hm : ∀ t ∈ Icc (0 : ℝ) 1, c + t • v ∈ S := by
    intro t ht
    simpa only [add_sub_cancel_left] using J.region_convex.add_smul_mem J.center_mem hv ht
  exact second_order_remainder_bound f J.differential J.hessian c v L
    (fun t ht => J.first _ (hm t ht)) (fun t ht => J.second _ (hm t ht))
    (fun t ht => by simpa only [add_sub_cancel_left] using J.variation _ (hm t ht))

/-- Exact-real central first difference at the supplied, possibly clamped step. -/
noncomputable def centralFirst (f : E → ℝ) (c u : E) (h : ℝ) : ℝ :=
  (f (c + h • u) - f (c - h • u)) / (2 * h)

/-- Exact-real diagonal second difference used by the raster bridge. -/
noncomputable def centralSecond (f : E → ℝ) (c u : E) (h : ℝ) : ℝ :=
  (f (c + h • u) - 2 * f c + f (c - h • u)) / h ^ 2

/-- Exact-real four-corner mixed difference used by the raster bridge. -/
noncomputable def centralMixed (f : E → ℝ) (c u v : E) (h : ℝ) : ℝ :=
  (f (c + h • (u + v)) - f (c + h • (u - v)) -
    f (c - h • (u - v)) + f (c - h • (u + v))) / (4 * h ^ 2)

noncomputable def quadraticRemainder (f : E → ℝ) (c : E)
    (D : E →L[ℝ] ℝ) (H : E →L[ℝ] (E →L[ℝ] ℝ)) (v : E) : ℝ :=
  f (c + v) - f c - D v - H v v / 2

theorem centralFirst_remainder_identity (f : E → ℝ) (c u : E) (h : ℝ)
    (D : E →L[ℝ] ℝ) (H : E →L[ℝ] (E →L[ℝ] ℝ)) (hh : h ≠ 0) :
    centralFirst f c u h - D u =
      (quadraticRemainder f c D H (h • u) -
        quadraticRemainder f c D H (-(h • u))) / (2 * h) := by
  simp only [centralFirst, quadraticRemainder, map_smul, map_neg,
    ContinuousLinearMap.smul_apply, ContinuousLinearMap.neg_apply, smul_eq_mul,
    sub_eq_add_neg]
  field_simp [hh]
  ring

/-- Central gradient truncation is quadratic in the actual positive step.
The bound comes from Hessian variation on the two physical query segments. -/
theorem LocalJetControl.centralFirst_error {f : E → ℝ} {c : E} {S : Set E} {L : ℝ}
    (J : LocalJetControl f c S L) (u : E) (h : ℝ) (hh : 0 < h)
    (hp : c + h • u ∈ S) (hm : c - h • u ∈ S) :
    |centralFirst f c u h - J.differential c u| ≤ L / 6 * h ^ 2 * ‖u‖ ^ 3 := by
  have hrp : |quadraticRemainder f c (J.differential c) (J.hessian c) (h • u)| ≤
      L / 6 * (h * ‖u‖) ^ 3 := by
    simpa only [quadraticRemainder, norm_smul, Real.norm_eq_abs, abs_of_pos hh] using
      J.remainder (h • u) hp
  have hrm : |quadraticRemainder f c (J.differential c) (J.hessian c) (-(h • u))| ≤
      L / 6 * (h * ‖u‖) ^ 3 := by
    simpa only [quadraticRemainder, norm_neg, norm_smul, Real.norm_eq_abs, abs_of_pos hh] using
      J.remainder (-(h • u)) (by simpa only [sub_eq_add_neg] using hm)
  rw [centralFirst_remainder_identity f c u h (J.differential c) (J.hessian c) hh.ne', abs_div,
    abs_of_pos (by positivity : 0 < 2 * h), div_le_iff₀ (by positivity : 0 < 2 * h)]
  have hab := abs_sub
    (quadraticRemainder f c (J.differential c) (J.hessian c) (h • u))
    (quadraticRemainder f c (J.differential c) (J.hessian c) (-(h • u)))
  nlinarith

theorem centralSecond_remainder_identity (f : E → ℝ) (c u : E) (h : ℝ)
    (D : E →L[ℝ] ℝ) (H : E →L[ℝ] (E →L[ℝ] ℝ)) (hh : h ≠ 0) :
    centralSecond f c u h - H u u =
      (quadraticRemainder f c D H (h • u) +
        quadraticRemainder f c D H (-(h • u))) / h ^ 2 := by
  simp only [centralSecond, quadraticRemainder, map_smul, map_neg,
    ContinuousLinearMap.smul_apply, ContinuousLinearMap.neg_apply, smul_eq_mul,
    sub_eq_add_neg]
  field_simp [hh]
  ring

/-- Diagonal Hessian truncation under Lipschitz Hessian control. -/
theorem LocalJetControl.centralSecond_error {f : E → ℝ} {c : E} {S : Set E} {L : ℝ}
    (J : LocalJetControl f c S L) (u : E) (h : ℝ) (hh : 0 < h)
    (hp : c + h • u ∈ S) (hm : c - h • u ∈ S) :
    |centralSecond f c u h - J.hessian c u u| ≤ L / 3 * h * ‖u‖ ^ 3 := by
  have hrp : |quadraticRemainder f c (J.differential c) (J.hessian c) (h • u)| ≤
      L / 6 * (h * ‖u‖) ^ 3 := by
    simpa only [quadraticRemainder, norm_smul, Real.norm_eq_abs, abs_of_pos hh] using
      J.remainder (h • u) hp
  have hrm : |quadraticRemainder f c (J.differential c) (J.hessian c) (-(h • u))| ≤
      L / 6 * (h * ‖u‖) ^ 3 := by
    simpa only [quadraticRemainder, norm_neg, norm_smul, Real.norm_eq_abs, abs_of_pos hh] using
      J.remainder (-(h • u)) (by simpa only [sub_eq_add_neg] using hm)
  rw [centralSecond_remainder_identity f c u h (J.differential c) (J.hessian c) hh.ne',
    abs_div, abs_of_pos (sq_pos_of_pos hh), div_le_iff₀ (sq_pos_of_pos hh)]
  have hab := abs_add_le
    (quadraticRemainder f c (J.differential c) (J.hessian c) (h • u))
    (quadraticRemainder f c (J.differential c) (J.hessian c) (-(h • u)))
  nlinarith

/-- Symmetry is derived from the actual local derivatives, provided the
controlled region is a neighborhood of the center. -/
theorem LocalJetControl.hessian_symm {f : E → ℝ} {c : E} {S : Set E} {L : ℝ}
    (J : LocalJetControl f c S L) (hS : S ∈ 𝓝 c) (u v : E) :
    J.hessian c u v = J.hessian c v u := by
  exact second_derivative_symmetric_of_eventually_of_real
    (Filter.mem_of_superset hS J.first) (J.second c J.center_mem) u v

theorem centralMixed_remainder_identity (f : E → ℝ) (c u v : E) (h : ℝ)
    (D : E →L[ℝ] ℝ) (H : E →L[ℝ] (E →L[ℝ] ℝ)) (hh : h ≠ 0)
    (hH : H u v = H v u) :
    centralMixed f c u v h - H u v =
      (quadraticRemainder f c D H (h • (u + v)) -
        quadraticRemainder f c D H (h • (u - v)) -
        quadraticRemainder f c D H (-(h • (u - v))) +
        quadraticRemainder f c D H (-(h • (u + v)))) / (4 * h ^ 2) := by
  simp only [centralMixed, quadraticRemainder, map_smul, map_neg, map_add,
    ContinuousLinearMap.smul_apply, ContinuousLinearMap.neg_apply,
    ContinuousLinearMap.add_apply, smul_eq_mul,
    sub_eq_add_neg, hH]
  field_simp [hh]
  ring

/-- The mixed stencil uses all four actual corners. Hessian symmetry follows
from differentiability on a neighborhood and is not assumed as a numeric flag. -/
theorem LocalJetControl.centralMixed_error {f : E → ℝ} {c : E} {S : Set E} {L : ℝ}
    (J : LocalJetControl f c S L) (hS : S ∈ 𝓝 c) (u v : E) (h : ℝ) (hh : 0 < h)
    (hpp : c + h • (u + v) ∈ S) (hpm : c + h • (u - v) ∈ S)
    (hmp : c - h • (u - v) ∈ S) (hmm : c - h • (u + v) ∈ S) :
    |centralMixed f c u v h - J.hessian c u v| ≤
      L / 12 * h * (‖u + v‖ ^ 3 + ‖u - v‖ ^ 3) := by
  let R := quadraticRemainder f c (J.differential c) (J.hessian c)
  have hrp (w : E) (hw : c + h • w ∈ S) : |R (h • w)| ≤ L / 6 * (h * ‖w‖) ^ 3 := by
    simpa only [R, quadraticRemainder, norm_smul, Real.norm_eq_abs, abs_of_pos hh] using
      J.remainder (h • w) hw
  have hrm (w : E) (hw : c - h • w ∈ S) : |R (-(h • w))| ≤ L / 6 * (h * ‖w‖) ^ 3 := by
    simpa only [R, quadraticRemainder, norm_neg, norm_smul, Real.norm_eq_abs, abs_of_pos hh] using
      J.remainder (-(h • w)) (by simpa only [sub_eq_add_neg] using hw)
  rw [centralMixed_remainder_identity f c u v h (J.differential c) (J.hessian c)
      hh.ne' (J.hessian_symm hS u v), abs_div,
    abs_of_pos (by positivity : 0 < 4 * h ^ 2),
    div_le_iff₀ (by positivity : 0 < 4 * h ^ 2)]
  change |R (h • (u + v)) - R (h • (u - v)) - R (-(h • (u - v))) +
    R (-(h • (u + v)))| ≤ _
  have h1 := abs_sub (R (h • (u + v))) (R (h • (u - v)))
  have h2 := abs_sub (R (h • (u + v)) - R (h • (u - v))) (R (-(h • (u - v))))
  have h3 := abs_add_le (R (h • (u + v)) - R (h • (u - v)) - R (-(h • (u - v))))
    (R (-(h • (u + v))))
  nlinarith [hrp (u + v) hpp, hrp (u - v) hpm, hrm (u - v) hmp, hrm (u + v) hmm]

/-- Additive field errors pass through the actual central stencil linearly. -/
theorem centralFirst_add (f e : E → ℝ) (c u : E) (h : ℝ) :
    centralFirst (fun z => f z + e z) c u h =
      centralFirst f c u h + centralFirst e c u h := by
  simp only [centralFirst]
  ring

/-- Correlated sample-error bound at the two computed query locations. -/
theorem centralFirst_provider_error (f e : E → ℝ) (c u : E) (h η : ℝ) (hh : 0 < h)
    (he : |e (c + h • u) - e (c - h • u)| ≤ η) :
    |centralFirst (fun z => f z + e z) c u h - centralFirst f c u h| ≤ η / (2 * h) := by
  rw [centralFirst_add, add_sub_cancel_left, centralFirst, abs_div,
    abs_of_pos (by positivity : 0 < 2 * h)]
  exact div_le_div_of_nonneg_right he (by positivity)

theorem centralSecond_add (f e : E → ℝ) (c u : E) (h : ℝ) :
    centralSecond (fun z => f z + e z) c u h =
      centralSecond f c u h + centralSecond e c u h := by
  simp only [centralSecond]
  ring

/-- Correlated second-difference sample error, including the center query. -/
theorem centralSecond_provider_error (f e : E → ℝ) (c u : E) (h η : ℝ) (hh : 0 < h)
    (he : |e (c + h • u) - 2 * e c + e (c - h • u)| ≤ η) :
    |centralSecond (fun z => f z + e z) c u h - centralSecond f c u h| ≤ η / h ^ 2 := by
  rw [centralSecond_add, add_sub_cancel_left, centralSecond, abs_div,
    abs_of_pos (sq_pos_of_pos hh)]
  exact div_le_div_of_nonneg_right he (sq_nonneg h)

theorem centralMixed_add (f e : E → ℝ) (c u v : E) (h : ℝ) :
    centralMixed (fun z => f z + e z) c u v h =
      centralMixed f c u v h + centralMixed e c u v h := by
  simp only [centralMixed]
  ring

/-- Correlated four-corner sample error at the actual computed scale. -/
theorem centralMixed_provider_error (f e : E → ℝ) (c u v : E) (h η : ℝ) (hh : 0 < h)
    (he : |e (c + h • (u + v)) - e (c + h • (u - v)) -
      e (c - h • (u - v)) + e (c - h • (u + v))| ≤ η) :
    |centralMixed (fun z => f z + e z) c u v h - centralMixed f c u v h| ≤
      η / (4 * h ^ 2) := by
  rw [centralMixed_add, add_sub_cancel_left, centralMixed, abs_div,
    abs_of_pos (by positivity : 0 < 4 * h ^ 2)]
  exact div_le_div_of_nonneg_right he (by positivity)

theorem centralFirst_provider_endpoint_error (f e : E → ℝ) (c u : E)
    (h ηp ηm : ℝ) (hh : 0 < h)
    (hp : |e (c + h • u)| ≤ ηp) (hm : |e (c - h • u)| ≤ ηm) :
    |centralFirst (fun z => f z + e z) c u h - centralFirst f c u h| ≤
      (ηp + ηm) / (2 * h) := by
  exact centralFirst_provider_error f e c u h (ηp + ηm) hh
    ((abs_sub _ _).trans (add_le_add hp hm))

theorem centralSecond_provider_endpoint_error (f e : E → ℝ) (c u : E)
    (h ηp η0 ηm : ℝ) (hh : 0 < h)
    (hp : |e (c + h • u)| ≤ ηp) (h0 : |e c| ≤ η0) (hm : |e (c - h • u)| ≤ ηm) :
    |centralSecond (fun z => f z + e z) c u h - centralSecond f c u h| ≤
      (ηp + 2 * η0 + ηm) / h ^ 2 := by
  apply centralSecond_provider_error f e c u h _ hh
  rcases abs_le.mp hp with ⟨hp0, hp1⟩
  rcases abs_le.mp h0 with ⟨hc0, hc1⟩
  rcases abs_le.mp hm with ⟨hm0, hm1⟩
  exact abs_le.mpr ⟨by linarith, by linarith⟩

theorem centralMixed_provider_endpoint_error (f e : E → ℝ) (c u v : E)
    (h ηpp ηpm ηmp ηmm : ℝ) (hh : 0 < h)
    (hpp : |e (c + h • (u + v))| ≤ ηpp) (hpm : |e (c + h • (u - v))| ≤ ηpm)
    (hmp : |e (c - h • (u - v))| ≤ ηmp) (hmm : |e (c - h • (u + v))| ≤ ηmm) :
    |centralMixed (fun z => f z + e z) c u v h - centralMixed f c u v h| ≤
      (ηpp + ηpm + ηmp + ηmm) / (4 * h ^ 2) := by
  apply centralMixed_provider_error f e c u v h _ hh
  rcases abs_le.mp hpp with ⟨hpp0, hpp1⟩
  rcases abs_le.mp hpm with ⟨hpm0, hpm1⟩
  rcases abs_le.mp hmp with ⟨hmp0, hmp1⟩
  rcases abs_le.mp hmm with ⟨hmm0, hmm1⟩
  exact abs_le.mpr ⟨by linarith, by linarith⟩

/-- Total central-gradient error: analytic truncation, correlated value error,
and an independently certified error of the final arithmetic evaluation. -/
theorem LocalJetControl.centralFirst_total_error {f : E → ℝ} {c : E} {S : Set E} {L : ℝ}
    (J : LocalJetControl f c S L) (e : E → ℝ) (u : E) (h η τ computed : ℝ)
    (hh : 0 < h) (hp : c + h • u ∈ S) (hm : c - h • u ∈ S)
    (he : |e (c + h • u) - e (c - h • u)| ≤ η)
    (ha : |computed - centralFirst (fun z => f z + e z) c u h| ≤ τ) :
    |computed - J.differential c u| ≤ L / 6 * h ^ 2 * ‖u‖ ^ 3 + η / (2 * h) + τ := by
  have h1 := J.centralFirst_error u h hh hp hm
  have h2 := centralFirst_provider_error f e c u h η hh he
  have h3 := abs_sub_le computed (centralFirst (fun z => f z + e z) c u h)
    (centralFirst f c u h)
  have h4 := abs_sub_le computed (centralFirst f c u h) (J.differential c u)
  linarith

/-- Total diagonal-Hessian error at the actual positive query step. -/
theorem LocalJetControl.centralSecond_total_error {f : E → ℝ} {c : E} {S : Set E} {L : ℝ}
    (J : LocalJetControl f c S L) (e : E → ℝ) (u : E) (h η τ computed : ℝ)
    (hh : 0 < h) (hp : c + h • u ∈ S) (hm : c - h • u ∈ S)
    (he : |e (c + h • u) - 2 * e c + e (c - h • u)| ≤ η)
    (ha : |computed - centralSecond (fun z => f z + e z) c u h| ≤ τ) :
    |computed - J.hessian c u u| ≤ L / 3 * h * ‖u‖ ^ 3 + η / h ^ 2 + τ := by
  have h1 := J.centralSecond_error u h hh hp hm
  have h2 := centralSecond_provider_error f e c u h η hh he
  have h3 := abs_sub_le computed (centralSecond (fun z => f z + e z) c u h)
    (centralSecond f c u h)
  have h4 := abs_sub_le computed (centralSecond f c u h) (J.hessian c u u)
  linarith

/-- Total mixed-Hessian error for the actual four-corner stencil. -/
theorem LocalJetControl.centralMixed_total_error {f : E → ℝ} {c : E} {S : Set E} {L : ℝ}
    (J : LocalJetControl f c S L) (hS : S ∈ 𝓝 c) (e : E → ℝ) (u v : E)
    (h η τ computed : ℝ) (hh : 0 < h)
    (hpp : c + h • (u + v) ∈ S) (hpm : c + h • (u - v) ∈ S)
    (hmp : c - h • (u - v) ∈ S) (hmm : c - h • (u + v) ∈ S)
    (he : |e (c + h • (u + v)) - e (c + h • (u - v)) -
      e (c - h • (u - v)) + e (c - h • (u + v))| ≤ η)
    (ha : |computed - centralMixed (fun z => f z + e z) c u v h| ≤ τ) :
    |computed - J.hessian c u v| ≤
      L / 12 * h * (‖u + v‖ ^ 3 + ‖u - v‖ ^ 3) + η / (4 * h ^ 2) + τ := by
  have h1 := J.centralMixed_error hS u v h hh hpp hpm hmp hmm
  have h2 := centralMixed_provider_error f e c u v h η hh he
  have h3 := abs_sub_le computed (centralMixed (fun z => f z + e z) c u v h)
    (centralMixed f c u v h)
  have h4 := abs_sub_le computed (centralMixed f c u v h) (J.hessian c u v)
  linarith

omit [NormedSpace ℝ E] in
/-- Query-coordinate error and value error combine before applying a stencil.
The field Lipschitz bound is only needed between these two actual locations. -/
theorem sampled_value_error (f : E → ℝ) (z actual : E) (value η δ : ℝ)
    (K : ℝ≥0) (hLip : LipschitzWith K f)
    (he : |value - f actual| ≤ η) (hq : ‖actual - z‖ ≤ δ) :
    |value - f z| ≤ η + (K : ℝ) * δ := by
  have hf : |f actual - f z| ≤ (K : ℝ) * ‖actual - z‖ := by
    simpa only [Real.dist_eq, dist_eq_norm] using hLip.dist_le_mul actual z
  have hf' := hf.trans (mul_le_mul_of_nonneg_left hq K.coe_nonneg)
  exact (abs_sub_le value (f actual) (f z)).trans (add_le_add he hf')

/-- Component certificates assemble into the genuine Euclidean norm bound.
Taking a product index gives the Frobenius norm of a matrix's entries. -/
theorem euclidean_error_bound {ι : Type*} [Fintype ι]
    (x y : EuclideanSpace ℝ ι) (b : ι → ℝ) (hb : ∀ i, |x i - y i| ≤ b i) :
    ‖x - y‖ ≤ Real.sqrt (∑ i, (b i) ^ 2) := by
  calc
    ‖x - y‖ = Real.sqrt (∑ i, ‖(x - y) i‖ ^ 2) := by
      rw [← EuclideanSpace.norm_sq_eq]
      exact (Real.sqrt_sq (norm_nonneg (x - y))).symm
    _ ≤ Real.sqrt (∑ i, (b i) ^ 2) := by
      apply Real.sqrt_le_sqrt
      apply Finset.sum_le_sum
      intro i _
      simpa only [PiLp.sub_apply, Real.norm_eq_abs] using
        pow_le_pow_left₀ (abs_nonneg (x i - y i)) (hb i) 2

/-- Explicit numerator and denominator perturbation bound. This is a real
arithmetic lemma; an IEEE implementation must separately supply its errors. -/
theorem division_perturbation_bound (n nHat d dHat ηn ηd : ℝ)
    (hd : 0 < d) (hdHat : 0 < dHat)
    (hn : |nHat - n| ≤ ηn) (he : |dHat - d| ≤ ηd) :
    |nHat / dHat - n / d| ≤ ηn / dHat + |n| * ηd / (d * dHat) := by
  have hid : nHat / dHat - n / d =
      (nHat - n) / dHat + n * (d - dHat) / (d * dHat) := by
    field_simp
    ring
  rw [hid]
  calc
    |(nHat - n) / dHat + n * (d - dHat) / (d * dHat)| ≤
        |(nHat - n) / dHat| + |n * (d - dHat) / (d * dHat)| := abs_add_le _ _
    _ = |nHat - n| / dHat + |n| * |dHat - d| / (d * dHat) := by
      rw [abs_div, abs_div, abs_mul, abs_of_pos hdHat,
        abs_of_pos (mul_pos hd hdHat), abs_sub_comm d dHat]
    _ ≤ ηn / dHat + |n| * ηd / (d * dHat) :=
      add_le_add (div_le_div_of_nonneg_right hn hdHat.le)
        (div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left he (abs_nonneg n))
          (mul_pos hd hdHat).le)

end FractalSync.ProviderCertificates
