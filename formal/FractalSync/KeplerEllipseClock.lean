import FractalSync.KeplerAreaClock
import FractalSync.KeplerAreaDensity
import FractalSync.KeplerKinematics

/-! The intrinsic area clock of a focus-relative ellipse in a normal chart.
The chart and its pullback density bounds are geometric inputs. -/

namespace FractalSync.KeplerEllipseClock

open Set
open scoped Interval
open KeplerAreaClock KeplerAreaDensity KeplerKinematics

noncomputable section

variable {J : ℝ × ℝ → ℝ} {a e E l U M x y : ℝ}

/-- Coordinates in the oriented orthonormal frame at the focus. -/
def ellipse (a e E : ℝ) : ℝ × ℝ :=
  (a * (Real.cos E - e), a * Real.sqrt (1 - e ^ 2) * Real.sin E)

/-- Swept-area rate with the constant ellipse factor `ab/2` divided out. -/
def rate (J : ℝ × ℝ → ℝ) (a e E : ℝ) : ℝ :=
  radialDensity J (ellipse a e E) * (1 - e * Real.cos E)

theorem continuous_rate (hJ : Continuous J) : Continuous (rate J a e) := by
  have hw : Continuous (ellipse a e) := by unfold ellipse; fun_prop
  exact ((continuous_radialDensity hJ).comp hw).mul (by fun_prop)

theorem rate_relative_bounds (hJ : Continuous J) (he : 0 ≤ e) (he' : e < 1)
    (hb : ∀ s ∈ Icc (0 : ℝ) 1,
      l ≤ J (s * (ellipse a e E).1, s * (ellipse a e E).2) ∧
      J (s * (ellipse a e E).1, s * (ellipse a e E).2) ≤ U) :
    l * (1 - e * Real.cos E) ≤ rate J a e E ∧
    rate J a e E ≤ U * (1 - e * Real.cos E) := by
  have hr := radialDensity_bounds hJ (ellipse a e E) l U hb
  have hg := (clock_positive e (Real.cos E) he he' (Real.cos_le_one E)).le
  exact ⟨mul_le_mul_of_nonneg_right hr.1 hg, mul_le_mul_of_nonneg_right hr.2 hg⟩

/-- Integrating the signed Jacobian of the radial fan gives the swept-area rate. -/
theorem swept_area_rate (J : ℝ × ℝ → ℝ) (a e E : ℝ) :
    let b := a * Real.sqrt (1 - e ^ 2)
    let w := ellipse a e E
    let v := (-a * Real.sin E, b * Real.cos E)
    (∫ s in (0 : ℝ)..1, J (s * w.1, s * w.2) * (w.1 * (s * v.2) - w.2 * (s * v.1))) =
      (a * b / 2) * rate J a e E := by
  dsimp only
  rw [integral_fan_density J (ellipse a e E)
    (-a * Real.sin E, a * Real.sqrt (1 - e ^ 2) * Real.cos E)]
  dsimp only
  have hd := focus_determinant a (a * Real.sqrt (1 - e ^ 2)) e (Real.cos E) (Real.sin E)
    (by nlinarith [Real.sin_sq_add_cos_sq E])
  unfold rate
  simp only [ellipse]
  rw [hd]
  ring

theorem flat_phase (a e E : ℝ) :
    phase (rate (fun _ => 1) a e) (2 * Real.pi) E = meanAnomaly e E := by
  have hr : rate (fun _ => 1) a e = fun u => 1 - e * Real.cos u := by
    funext u
    simp [rate, radialDensity_one]
  rw [hr]
  unfold phase area
  rw [integral_classical_rate_period, integral_classical_rate, div_self (by positivity : 2 * Real.pi ≠ 0),
    one_mul]


/-- The density enclosure need only hold on the swept radial fan. -/
def FanBounds (J : ℝ × ℝ → ℝ) (a e l U : ℝ) : Prop :=
  ∀ E ∈ Icc (0 : ℝ) (2 * Real.pi), ∀ s ∈ Icc (0 : ℝ) 1,
    l ≤ J (s * (ellipse a e E).1, s * (ellipse a e E).2) ∧
    J (s * (ellipse a e E).1, s * (ellipse a e E).2) ≤ U

theorem rate_bounds (hJ : Continuous J) (he : 0 ≤ e) (he' : e < 1) (hl : 0 < l)
    (hLU : l ≤ U) (hb : FanBounds J a e l U) :
    ∀ E ∈ Icc (0 : ℝ) (2 * Real.pi),
      l * (1 - e) ≤ rate J a e E ∧ rate J a e E ≤ U * (1 + e) := by
  intro E hE
  have hr := rate_relative_bounds hJ he he' (hb E hE)
  have hlow := clock_lower e (Real.cos E) he (Real.cos_le_one E)
  have hhigh : 1 - e * Real.cos E ≤ 1 + e := by
    have h := mul_le_mul_of_nonneg_left (Real.neg_one_le_cos E) he
    linarith
  exact ⟨(mul_le_mul_of_nonneg_left hlow hl.le).trans hr.1,
    hr.2.trans (mul_le_mul_of_nonneg_left hhigh (hl.le.trans hLU))⟩

theorem ellipse_anomaly_existsUnique (hJ : Continuous J) (he : 0 ≤ e) (he' : e < 1)
    (hl : 0 < l) (hLU : l ≤ U) (hb : FanBounds J a e l U)
    (hM : M ∈ Icc (0 : ℝ) (2 * Real.pi)) :
    ∃! E, E ∈ Icc (0 : ℝ) (2 * Real.pi) ∧ phase (rate J a e) (2 * Real.pi) E = M :=
  anomaly_existsUnique (continuous_rate hJ) (by positivity)
    (mul_pos hl (sub_pos.mpr he')) (rate_bounds hJ he he' hl hLU hb) hM

private theorem meanAnomaly_increment_lower (he : 0 ≤ e) (hxy : x ≤ y) :
    (1 - e) * (y - x) ≤ meanAnomaly e y - meanAnomaly e x := by
  let g : ℝ → ℝ := fun u => 1 - e * Real.cos u
  have hg : Continuous g := by dsimp [g]; fun_prop
  rw [← integral_classical_rate e y, ← integral_classical_rate e x]
  exact (area_sub_bounds hg hxy (l := 1 - e) (U := 1 + e) (fun u _ => by
    refine ⟨clock_lower e (Real.cos u) he (Real.cos_le_one u), ?_⟩
    have h := mul_le_mul_of_nonneg_left (Real.neg_one_le_cos u) he
    dsimp [g]
    linarith)).1

/-- Variable density changes phase increments by at most the ratio of the
density bounds, relative to the classical mean anomaly. -/
theorem ellipse_phase_distortion (hJ : Continuous J) (he : 0 ≤ e) (he' : e < 1)
    (hl : 0 < l) (hLU : l ≤ U) (hb : FanBounds J a e l U)
    (hx : x ∈ Icc (0 : ℝ) (2 * Real.pi)) (hy : y ∈ Icc (0 : ℝ) (2 * Real.pi))
    (hxy : x ≤ y) :
    (l / U) * (meanAnomaly e y - meanAnomaly e x) ≤
      phase (rate J a e) (2 * Real.pi) y - phase (rate J a e) (2 * Real.pi) x ∧
    phase (rate J a e) (2 * Real.pi) y - phase (rate J a e) (2 * Real.pi) x ≤
      (U / l) * (meanAnomaly e y - meanAnomaly e x) := by
  let g : ℝ → ℝ := fun u => 1 - e * Real.cos u
  have hg : Continuous g := by dsimp [g]; fun_prop
  have hgT : area g (2 * Real.pi) = 2 * Real.pi := integral_classical_rate_period e
  have hgxy : 0 ≤ area g y - area g x := by
    rw [show area g y = meanAnomaly e y from integral_classical_rate e y,
      show area g x = meanAnomaly e x from integral_classical_rate e x]
    exact (mul_nonneg (sub_nonneg.mpr he'.le) (sub_nonneg.mpr hxy)).trans
      (meanAnomaly_increment_lower he hxy)
  have hflat (z : ℝ) : phase g (2 * Real.pi) z = meanAnomaly e z := by
    rw [phase, hgT, div_self (by positivity : 2 * Real.pi ≠ 0), one_mul]
    exact integral_classical_rate e z
  have h := relative_phase_sub_bounds (continuous_rate hJ) hg (by positivity) hl
    (hl.trans_le hLU) (by rw [hgT]; positivity)
    (fun u hu => rate_relative_bounds hJ he he' (hb u hu)) hx hy hxy hgxy
  rwa [hflat x, hflat y] at h

/-- A residual in musical phase bounds anomaly error for ordered anomalies. -/
theorem ellipse_anomaly_error_bound (hJ : Continuous J) (he : 0 ≤ e) (he' : e < 1)
    (hl : 0 < l) (hLU : l ≤ U) (hb : FanBounds J a e l U)
    (hx : x ∈ Icc (0 : ℝ) (2 * Real.pi)) (hy : y ∈ Icc (0 : ℝ) (2 * Real.pi))
    (hxy : x ≤ y) :
    y - x ≤ (phase (rate J a e) (2 * Real.pi) y - phase (rate J a e) (2 * Real.pi) x) /
      ((l / U) * (1 - e)) := by
  have hp := div_pos hl (hl.trans_le hLU)
  apply (le_div_iff₀ (mul_pos hp (sub_pos.mpr he'))).2
  rw [mul_comm (y - x), mul_assoc]
  exact (mul_le_mul_of_nonneg_left (meanAnomaly_increment_lower he hxy) hp.le).trans
    (ellipse_phase_distortion hJ he he' hl hLU hb hx hy hxy).1

end
end FractalSync.KeplerEllipseClock
