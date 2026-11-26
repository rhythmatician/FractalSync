import FractalSync.NormalizedMapDifferential
import FractalSync.Ruler

/-!
# Normalized query stability with a recomputed ruler

Exact-real contracts for #162. Provider error certificates concern the actual
computed query location. No numerical provider's values or derivatives are
certified by these conditional statements.
-/

namespace FractalSync.NormalizedDistanceFields

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- The ideal value bound also holds for a negative, nonzero physical scale. -/
theorem normalizedMap_value_bound_signed (D : E → ℝ) (hD : LipschitzWith 1 D)
    (x u : E) (h : ℝ) (hh : h ≠ 0) :
    |normalizedMap D x u h| ≤ ‖u‖ := by
  simpa using normalizedMap_value_bound_on_segment D x u h 1 hh hD.lipschitzOnWith

/-- The scale change separates displacement of the sampled point from the
normalizing-denominator change. -/
theorem normalizedMap_scale_difference_identity (D : E → ℝ) (x u : E)
    (h k : ℝ) (hh : h ≠ 0) (hk : k ≠ 0) :
    normalizedMap D x u k - normalizedMap D x u h =
      ((D (x + k • u) - D (x + h • u)) +
        (h - k) * normalizedMap D x u h) / k := by
  unfold normalizedMap normalizedDifference
  field_simp
  ring

/-- A one-sided denominator bound used to obtain the symmetric scale estimate. -/
theorem normalizedMap_scale_change_bound_right (D : E → ℝ) (hD : LipschitzWith 1 D)
    (x u : E) (h k : ℝ) (hh : h ≠ 0) (hk : k ≠ 0) :
    |normalizedMap D x u k - normalizedMap D x u h| ≤ 2 * ‖u‖ * |k - h| / |k| := by
  have hs : |D (x + k • u) - D (x + h • u)| ≤ |k - h| * ‖u‖ := by
    simpa [add_sub_add_left_eq_sub, ← sub_smul, norm_smul, Real.norm_eq_abs] using
      hD.norm_sub_le (x + k • u) (x + h • u)
  rw [normalizedMap_scale_difference_identity D x u h k hh hk, abs_div]
  apply div_le_div_of_nonneg_right _ (abs_nonneg k)
  refine (abs_add_le _ _).trans ?_
  rw [abs_mul, abs_sub_comm h k]
  nlinarith [mul_le_mul_of_nonneg_left (normalizedMap_value_bound_signed D hD x u h hh)
    (abs_nonneg (k - h))]

/-- The exact scale-change estimate, valid for either sign of nonzero scales. -/
theorem normalizedMap_scale_change_bound (D : E → ℝ) (hD : LipschitzWith 1 D)
    (x u : E) (h k : ℝ) (hh : h ≠ 0) (hk : k ≠ 0) :
    |normalizedMap D x u h - normalizedMap D x u k| ≤
      2 * ‖u‖ * |h - k| / max |h| |k| := by
  rcases le_total |h| |k| with hle | hle
  · simpa [max_eq_right hle, abs_sub_comm] using
      normalizedMap_scale_change_bound_right D hD x u h k hh hk
  · simpa [max_eq_left hle] using
      normalizedMap_scale_change_bound_right D hD x u k h hk hh

/-- With a recomputed scale, provider error is evaluated at that scale's
actual query point, before comparison with the nominal window. -/
theorem normalizedMap_provider_scale_error_identity (D e : E → ℝ)
    (x u : E) (h k : ℝ) :
    normalizedMap (fun z => D z + e z) x u k - normalizedMap D x u h =
      (e (x + k • u) - e x) / k +
        (normalizedMap D x u k - normalizedMap D x u h) := by
  unfold normalizedMap normalizedDifference
  ring

/-- Conditional correlated provider error plus the exact scale-change bound. -/
theorem normalizedMap_provider_scale_error_bound (D e : E → ℝ)
    (hD : LipschitzWith 1 D) (x u : E) (h k eta : ℝ)
    (hh : h ≠ 0) (hk : k ≠ 0) (he : |e (x + k • u) - e x| ≤ eta) :
    |normalizedMap (fun z => D z + e z) x u k - normalizedMap D x u h| ≤
      eta / |k| + 2 * ‖u‖ * |k - h| / max |h| |k| := by
  rw [normalizedMap_provider_scale_error_identity]
  exact (abs_add_le _ _).trans (add_le_add
    (by simpa [abs_div] using div_le_div_of_nonneg_right he (abs_nonneg k))
    (by simpa [max_comm] using normalizedMap_scale_change_bound D hD x u k h hk hh))

/-- A common positive window multiplier cancels from the relative ruler term. -/
theorem normalizedMap_provider_scaled_error_bound (D e : E → ℝ)
    (hD : LipschitzWith 1 D) (x u : E) (q rho rhoHat eta : ℝ)
    (hq : 0 < q) (hr : 0 < rho) (hrh : 0 < rhoHat)
    (he : |e (x + (q * rhoHat) • u) - e x| ≤ eta) :
    |normalizedMap (fun z => D z + e z) x u (q * rhoHat) -
        normalizedMap D x u (q * rho)| ≤
      eta / (q * rhoHat) + 2 * ‖u‖ * |rhoHat - rho| / max rho rhoHat := by
  have hb := normalizedMap_provider_scale_error_bound D e hD x u
    (q * rho) (q * rhoHat) eta (mul_pos hq hr).ne' (mul_pos hq hrh).ne' he
  rw [abs_of_pos (mul_pos hq hrh), abs_of_pos (mul_pos hq hr),
    ← mul_sub, abs_mul, abs_of_pos hq, ← mul_max_of_nonneg rho rhoHat hq.le] at hb
  simpa only [mul_left_comm (2 * ‖u‖) q, mul_div_mul_left _ _ hq.ne'] using hb

/-- The combined error budget for the actual ruler recomputed from the
provider's center value. The error certificate uses the computed query. -/
theorem normalizedMap_recomputed_ruler_error_bound (D e : E → ℝ)
    (hD : LipschitzWith 1 D) (x u : E) (epsilon q eta : ℝ)
    (heps : 0 < epsilon) (hq : 0 < q)
    (he : |e (x + (q * ruler epsilon (D x + e x)) • u) - e x| ≤ eta) :
    let rho := ruler epsilon (D x)
    let rhoHat := ruler epsilon (D x + e x)
    |normalizedMap (fun z => D z + e z) x u (q * rhoHat) -
        normalizedMap D x u (q * rho)| ≤
      eta / (q * rhoHat) + 2 * ‖u‖ * |rhoHat - rho| / max rho rhoHat := by
  exact normalizedMap_provider_scaled_error_bound D e hD x u q _ _ eta hq
    (ruler_pos epsilon (D x) heps) (ruler_pos epsilon (D x + e x) heps) he

/-- Separate certified errors at the center and the computed query imply
the endpoint-sum version of the recomputed-ruler budget. -/
theorem normalizedMap_recomputed_ruler_endpoint_bound (D e : E → ℝ)
    (hD : LipschitzWith 1 D) (x u : E) (epsilon q e0 eu : ℝ)
    (heps : 0 < epsilon) (hq : 0 < q) (h0 : |e x| ≤ e0)
    (hu : |e (x + (q * ruler epsilon (D x + e x)) • u)| ≤ eu) :
    let rho := ruler epsilon (D x)
    let rhoHat := ruler epsilon (D x + e x)
    |normalizedMap (fun z => D z + e z) x u (q * rhoHat) -
        normalizedMap D x u (q * rho)| ≤
      (eu + e0) / (q * rhoHat) + 2 * ‖u‖ * |rhoHat - rho| / max rho rhoHat := by
  exact normalizedMap_recomputed_ruler_error_bound D e hD x u epsilon q (eu + e0)
    heps hq (by
      simpa only [sub_eq_add_neg, abs_neg] using
        (abs_add_le (e (x + (q * ruler epsilon (D x + e x)) • u)) (-e x)).trans
          (add_le_add hu (by simpa only [abs_neg] using h0)))

/-- Recomputing the regularized ruler changes it by at most the center error. -/
theorem ruler_perturbation_le (epsilon s b : ℝ) (heps : 0 < epsilon) :
    |ruler epsilon (s + b) - ruler epsilon s| ≤ |b| := by
  simpa [Real.norm_eq_abs] using (ruler_lipschitz epsilon heps).norm_sub_le (s + b) s

/-- Rationalization retains the exact dependence of ruler error on field value
and center bias, including the quadratic cancellation at the zero level. -/
theorem ruler_perturbation_identity (epsilon s b : ℝ) (heps : 0 < epsilon) :
    |ruler epsilon (s + b) - ruler epsilon s| =
      |b * (2 * s + b)| / (ruler epsilon (s + b) + ruler epsilon s) := by
  have hid : (ruler epsilon (s + b) - ruler epsilon s) *
      (ruler epsilon (s + b) + ruler epsilon s) = b * (2 * s + b) := by
    nlinarith [ruler_sq epsilon (s + b), ruler_sq epsilon s]
  exact (eq_div_iff (add_pos (ruler_pos epsilon (s + b) heps)
    (ruler_pos epsilon s heps)).ne').mpr (by
      simpa only [abs_mul, abs_of_pos (add_pos (ruler_pos epsilon (s + b) heps)
        (ruler_pos epsilon s heps))] using congrArg abs hid)

/-- A relative center-error certificate gives a lower bound for the recomputed ruler. -/
theorem ruler_perturbation_relative_lower (epsilon s b kappa : ℝ) (heps : 0 < epsilon)
    (hb : |b| ≤ kappa * ruler epsilon s) :
    (1 - kappa) * ruler epsilon s ≤ ruler epsilon (s + b) := by
  nlinarith [(abs_le.mp ((ruler_perturbation_le epsilon s b heps).trans hb)).1]

/-- The relative ruler change is at most the certified relative center error. -/
theorem ruler_perturbation_relative_bound (epsilon s b kappa : ℝ)
    (heps : 0 < epsilon) (hk : 0 ≤ kappa) (hb : |b| ≤ kappa * ruler epsilon s) :
    |ruler epsilon (s + b) - ruler epsilon s| /
      max (ruler epsilon s) (ruler epsilon (s + b)) ≤ kappa := by
  exact (div_le_iff₀ ((ruler_pos epsilon s heps).trans_le (le_max_left _ _))).mpr
    (((ruler_perturbation_le epsilon s b heps).trans hb).trans
      (mul_le_mul_of_nonneg_left (le_max_left _ _) hk))

/-- A center-error certificate below the local ruler gives a uniform budget
while preserving a sharper, correlated query-error certificate. -/
theorem normalizedMap_recomputed_ruler_relative_bound (D e : E → ℝ)
    (hD : LipschitzWith 1 D) (x u : E) (epsilon q eta kappa : ℝ)
    (heps : 0 < epsilon) (hq : 0 < q) (hk0 : 0 ≤ kappa) (hk1 : kappa < 1)
    (h0 : |e x| ≤ kappa * ruler epsilon (D x))
    (he : |e (x + (q * ruler epsilon (D x + e x)) • u) - e x| ≤ eta) :
    |normalizedMap (fun z => D z + e z) x u (q * ruler epsilon (D x + e x)) -
        normalizedMap D x u (q * ruler epsilon (D x))| ≤
      eta / (q * (1 - kappa) * ruler epsilon (D x)) + 2 * kappa * ‖u‖ := by
  refine (normalizedMap_recomputed_ruler_error_bound D e hD x u epsilon q eta
    heps hq he).trans (add_le_add ?_ ?_)
  · exact div_le_div_of_nonneg_left ((abs_nonneg _).trans he)
      (mul_pos (mul_pos hq (sub_pos.mpr hk1)) (ruler_pos epsilon (D x) heps))
      (by
        simpa only [mul_assoc] using mul_le_mul_of_nonneg_left
          (ruler_perturbation_relative_lower epsilon (D x) (e x) kappa heps h0) hq.le)
  · simpa only [div_eq_mul_inv, mul_assoc, mul_left_comm, mul_comm] using
      mul_le_mul_of_nonneg_left
        (ruler_perturbation_relative_bound epsilon (D x) (e x) kappa heps hk0 h0)
        (show 0 ≤ 2 * ‖u‖ by positivity)

/-- The uniform endpoint-certificate budget when the center error is smaller
than a fixed fraction of the nominal ruler. -/
theorem normalizedMap_recomputed_ruler_uniform_bound (D e : E → ℝ)
    (hD : LipschitzWith 1 D) (x u : E) (epsilon q e0 eu kappa : ℝ)
    (heps : 0 < epsilon) (hq : 0 < q) (hk0 : 0 ≤ kappa) (hk1 : kappa < 1)
    (h0 : |e x| ≤ e0) (hrel : e0 ≤ kappa * ruler epsilon (D x))
    (hu : |e (x + (q * ruler epsilon (D x + e x)) • u)| ≤ eu) :
    |normalizedMap (fun z => D z + e z) x u (q * ruler epsilon (D x + e x)) -
        normalizedMap D x u (q * ruler epsilon (D x))| ≤
      (eu + e0) / (q * (1 - kappa) * ruler epsilon (D x)) + 2 * kappa * ‖u‖ := by
  exact normalizedMap_recomputed_ruler_relative_bound D e hD x u epsilon q
    (eu + e0) kappa heps hq hk0 hk1 (h0.trans hrel) (by
      simpa only [sub_eq_add_neg, abs_neg] using
        (abs_add_le (e (x + (q * ruler epsilon (D x + e x)) • u)) (-e x)).trans
          (add_le_add hu (by simpa only [abs_neg] using h0)))

/-- At an exact zero level, the relative ruler error is quadratic in bias. -/
theorem ruler_crest_relative_bound (epsilon b : ℝ) (heps : 0 < epsilon) :
    |ruler epsilon b - epsilon| / ruler epsilon b ≤ b ^ 2 / (2 * epsilon ^ 2) := by
  have hz : ruler epsilon 0 = epsilon := by simpa [ruler] using Real.sqrt_sq heps.le
  have hi : |ruler epsilon b - epsilon| = b ^ 2 / (ruler epsilon b + epsilon) := by
    simpa [hz, ← pow_two] using ruler_perturbation_identity epsilon 0 b heps
  rw [hi, div_div]
  apply div_le_div_of_nonneg_left (sq_nonneg b) (by positivity)
  nlinarith [mul_nonneg (sub_nonneg.mpr (epsilon_le_ruler epsilon b heps.le))
    (show 0 ≤ ruler epsilon b + 2 * epsilon by
      linarith [ruler_pos epsilon b heps])]

/-- A common additive provider bias contributes only a quadratic scale error
at an exact zero-level center; its anchored query error cancels exactly. -/
theorem normalizedMap_crest_bias_bound (D : E → ℝ) (hD : LipschitzWith 1 D)
    (x u : E) (epsilon q b : ℝ) (heps : 0 < epsilon) (hq : 0 < q) (hx : D x = 0) :
    |normalizedMap (fun z => D z + b) x u (q * ruler epsilon b) -
        normalizedMap D x u (q * epsilon)| ≤ (b / epsilon) ^ 2 * ‖u‖ := by
  have hb := normalizedMap_recomputed_ruler_error_bound D (fun _ => b) hD x u
    epsilon q 0 heps hq (by simp)
  have hz : ruler epsilon 0 = epsilon := by simpa [ruler] using Real.sqrt_sq heps.le
  simp only [hx, zero_add, hz, zero_div, max_eq_right (epsilon_le_ruler epsilon b heps.le)] at hb
  refine hb.trans ?_
  convert mul_le_mul_of_nonneg_left (ruler_crest_relative_bound epsilon b heps)
    (show 0 ≤ 2 * ‖u‖ by positivity) using 1
  all_goals ring

/-- Center stability of the normalized query. The totalized definition also
makes this inequality valid when the scale is zero. -/
theorem normalizedMap_center_change_bound (D : E → ℝ) (hD : LipschitzWith 1 D)
    (x y u : E) (h : ℝ) :
    |normalizedMap D x u h - normalizedMap D y u h| ≤ 2 * ‖x - y‖ / |h| := by
  have hs : |D (x + h • u) - D (y + h • u)| ≤ ‖x - y‖ := by
    simpa [Real.norm_eq_abs] using hD.norm_sub_le (x + h • u) (y + h • u)
  have hc : |D x - D y| ≤ ‖x - y‖ := by simpa [Real.norm_eq_abs] using hD.norm_sub_le x y
  simp only [normalizedMap, normalizedDifference, div_sub_div_same, abs_div]
  apply div_le_div_of_nonneg_right _ (abs_nonneg h)
  exact abs_le.mpr ⟨by linarith [(abs_le.mp hs).1, (abs_le.mp hc).2],
    by linarith [(abs_le.mp hs).2, (abs_le.mp hc).1]⟩

end FractalSync.NormalizedDistanceFields
