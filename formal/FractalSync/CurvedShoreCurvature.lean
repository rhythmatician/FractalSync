import FractalSync.CurvedShoreGeometry

/-!
# Actual off-Shore curvature in the curved normal-coordinate metric

The full coordinate curvature commutator is evaluated at arbitrary normal
radius away from `1 - k(y) * r = 0`. The result depends on the curvature value
but not its derivatives, whose terms cancel. Constructing a tubular chart of
an embedded Shore is not part of these coordinate calculations.
-/

namespace FractalSync.CurvedShoreGeometry

open Filter Topology

noncomputable def gammaYYRadialDerivative (epsilon a k r : ℝ) : ℝ :=
  ((1 - 2 * k * r - k ^ 2 * epsilon ^ 2) *
      (epsilon ^ 2 + (1 + a ^ 2) * r ^ 2) -
    2 * (1 + a ^ 2) * r * (1 - k * r) * (r + k * epsilon ^ 2)) /
      (epsilon ^ 2 + (1 + a ^ 2) * r ^ 2) ^ 2

/-- Actual normal derivative of the vector field `nabla_y y`. -/
theorem fderiv_gammaYY_pair_normal (epsilon a r y dk ddk : ℝ)
    (k : ℝ → ℝ) (he : 0 < epsilon) (hh : 1 - k y * r ≠ 0)
    (hk : HasDerivAt k dk y) (hdk : HasDerivAt (deriv k) ddk y) :
    fderiv ℝ (fun x : Coord =>
      (gammaYY epsilon a (k x.2) x.1, gammaY (k x.2) (deriv k x.2) x.1))
      (r, y) (1, 0) = (gammaYYRadialDerivative epsilon a (k y) r,
        -dk / (1 - k y * r) ^ 2) := by
  have hr : HasFDerivAt (fun x : Coord => x.1)
      (ContinuousLinearMap.fst ℝ ℝ ℝ) (r, y) :=
    (ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt
  have hk2 := hk.comp_hasFDerivAt ((r, y) : Coord)
    (ContinuousLinearMap.snd ℝ ℝ ℝ).hasFDerivAt
  have hdk2 := hdk.comp_hasFDerivAt ((r, y) : Coord)
    (ContinuousLinearMap.snd ℝ ℝ ℝ).hasFDerivAt
  have hden := ((hr.pow 2).const_mul (1 + a ^ 2)).const_add (epsilon ^ 2)
  have hi := (hasDerivAt_inv (FlatShore.normal_sq_pos epsilon a r he).ne').comp_hasFDerivAt
    ((r, y) : Coord) hden
  have hYY := (((hk2.mul hr).const_sub 1).mul
    (hr.add (hk2.mul_const (epsilon ^ 2)))).mul hi
  have hi2 := (hasDerivAt_inv hh).comp_hasFDerivAt
    ((r, y) : Coord) ((hk2.mul hr).const_sub 1)
  have hY := (hr.neg.mul hdk2).mul hi2
  have hd := congrArg (fun L : Coord →L[ℝ] Coord => L (1, 0)) (hYY.prodMk hY).fderiv
  simp only [gammaYY, gammaY, div_eq_mul_inv]
  simp only [Function.comp_apply, Pi.mul_apply, Pi.add_apply, Pi.neg_apply,
    ContinuousLinearMap.coe_snd'] at hd
  rw [hd]
  simp [hk.deriv, gammaYYRadialDerivative]
  constructor
  all_goals field_simp [hh, (FlatShore.normal_sq_pos epsilon a r he).ne']
  all_goals ring

/-- Actual tangential derivative of the vector field `nabla_r y`. -/
theorem fderiv_gammaRY_tangent (epsilon r y dk : ℝ)
    (k : ℝ → ℝ) (he : 0 < epsilon) (hh : 1 - k y * r ≠ 0)
    (hk : HasDerivAt k dk y) :
    fderiv ℝ (fun x : Coord => ((0 : ℝ), gammaRY epsilon (k x.2) x.1))
      (r, y) (0, 1) = (0, -dk / (1 - k y * r) ^ 2) := by
  have hr : HasFDerivAt (fun x : Coord => x.1)
      (ContinuousLinearMap.fst ℝ ℝ ℝ) (r, y) :=
    (ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt
  have hk2 := hk.comp_hasFDerivAt ((r, y) : Coord)
    (ContinuousLinearMap.snd ℝ ℝ ℝ).hasFDerivAt
  have hi := (hasDerivAt_inv hh).comp_hasFDerivAt
    ((r, y) : Coord) ((hk2.mul hr).const_sub 1)
  have hf := (FlatShoreGeometry.hasDerivAt_gammaRY epsilon r he).comp_hasFDerivAt
    ((r, y) : Coord) hr
  have hd := congrArg (fun L : Coord →L[ℝ] Coord => L (0, 1))
    ((hasFDerivAt_const (0 : ℝ) ((r, y) : Coord)).prodMk (hf.sub (hk2.mul hi))).fderiv
  simp only [gammaRY, div_eq_mul_inv]
  simp only [FlatShoreGeometry.gammaRY, Function.comp_apply, Pi.mul_apply, Pi.sub_apply,
    ContinuousLinearMap.coe_snd', div_eq_mul_inv] at hd
  rw [hd]
  simp
  field_simp [hh]
  ring

/-- The full actual commutator at an arbitrary nonfocal coordinate point.
The tangential derivative terms cancel, including the derivatives of `k`. -/
theorem curvature_normal_tangent_tangent (epsilon a r y dk ddk : ℝ)
    (k : ℝ → ℝ) (he : 0 < epsilon) (hh : 1 - k y * r ≠ 0)
    (hk : HasDerivAt k dk y) (hdk : HasDerivAt (deriv k) ddk y) :
    curvature epsilon a k (fun _ => (1, 0)) (fun _ => (0, 1))
      (fun _ => (0, 1)) (r, y) =
        (gammaYYRadialDerivative epsilon a (k y) r +
          FlatShoreGeometry.gammaRR epsilon a r * gammaYY epsilon a (k y) r -
          gammaYY epsilon a (k y) r * gammaRY epsilon (k y) r, 0) := by
  rw [curvature, covariantDerivative_const, covariantDerivative_const]
  simp only [christoffel, mul_one, mul_zero, add_zero, zero_add]
  simp only [covariantDerivative, VectorField.lieBracket, fderiv_const_apply,
    ContinuousLinearMap.zero_apply, sub_self, zero_add]
  rw [fderiv_gammaYY_pair_normal epsilon a r y dk ddk k he hh hk hdk,
    fderiv_gammaRY_tangent epsilon r y dk k he hh hk]
  simp [christoffel]
  ring

/-- Actual coordinate curvature is the flat expression plus the exact normal
chart correction. Derivatives of the curvature profile cancel completely. -/
theorem sectionalCurvature_eq_flat_correction (epsilon a r y dk ddk : ℝ)
    (k : ℝ → ℝ) (he : 0 < epsilon) (hh : 1 - k y * r ≠ 0)
    (hk : HasDerivAt k dk y) (hdk : HasDerivAt (deriv k) ddk y) :
    sectionalCurvature epsilon a k (r, y) =
      FlatShore.radialCurvature epsilon a r -
        k y * (1 + a ^ 2) * r * (epsilon ^ 2 + r ^ 2) ^ 2 /
          ((1 - k y * r) * (epsilon ^ 2 + (1 + a ^ 2) * r ^ 2) ^ 2) := by
  rw [sectionalCurvature, curvature_normal_tangent_tangent epsilon a r y dk ddk k he hh hk hdk]
  dsimp only [metric]
  simp only [mul_one, mul_zero, add_zero, zero_add, zero_pow two_ne_zero, sub_zero]
  rw [mul_div_mul_left _ _ (FlatShoreGeometry.radialMetric_pos epsilon a r he).ne']
  dsimp only [gammaYYRadialDerivative, gammaYY, gammaRY, FlatShoreGeometry.gammaRR,
    FlatShoreGeometry.tangentialMetric, FlatShore.radialCurvature]
  field_simp [hh, (FlatShore.normal_sq_pos epsilon a r he).ne',
    (FlatShore.ruler_sq_pos epsilon r he).ne']
  ring

/-- The curvature's normal slope at Shore detects the signed curvature of
the supplied normal chart. This differentiates the actual curvature function. -/
theorem hasDerivAt_sectionalCurvature_normal_on_shore (epsilon a y dk ddk : ℝ)
    (k : ℝ → ℝ) (he : 0 < epsilon)
    (hk : HasDerivAt k dk y) (hdk : HasDerivAt (deriv k) ddk y) :
    HasDerivAt (fun r => sectionalCurvature epsilon a k (r, y))
      (-(1 + a ^ 2) * k y) 0 := by
  have hflat : HasDerivAt (FlatShore.radialCurvature epsilon a) 0 0 := by
    simpa using FlatShore.hasDerivAt_radialCurvature epsilon a 0 he
  have hn := ((hasDerivAt_id (0 : ℝ)).const_mul (k y * (1 + a ^ 2))).mul
    ((((hasDerivAt_id (0 : ℝ)).pow 2).const_add (epsilon ^ 2)).pow 2)
  have hd := (((hasDerivAt_id (0 : ℝ)).const_mul (k y)).const_sub 1).mul
    (((((hasDerivAt_id (0 : ℝ)).pow 2).const_mul (1 + a ^ 2)).const_add (epsilon ^ 2)).pow 2)
  have hformula : HasDerivAt
      (fun r => FlatShore.radialCurvature epsilon a r -
        k y * (1 + a ^ 2) * r * (epsilon ^ 2 + r ^ 2) ^ 2 /
          ((1 - k y * r) * (epsilon ^ 2 + (1 + a ^ 2) * r ^ 2) ^ 2))
      (-(1 + a ^ 2) * k y) 0 := by
    convert hflat.sub (hn.div hd (by simp [he.ne'])) using 1
    norm_num
    field_simp
    ring
  have hnonzero : ∀ᶠ r in 𝓝 (0 : ℝ), 1 - k y * r ≠ 0 := by
    exact (((hasDerivAt_id (0 : ℝ)).const_mul (k y)).const_sub 1).continuousAt.eventually_ne
      (by simp)
  apply hformula.congr_of_eventuallyEq
  filter_upwards [hnonzero] with r hr
  exact sectionalCurvature_eq_flat_correction epsilon a r y dk ddk k he hr hk hdk

end FractalSync.CurvedShoreGeometry
