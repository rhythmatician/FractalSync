import Mathlib.Analysis.Complex.Norm
import Mathlib.Analysis.SpecificLimits.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring
import Mathlib.Tactic.FieldSimp

namespace FractalSync.ExteriorRulerBounds

/-- The quadratic orbit starting from a chosen finite iterate. -/
def orbitTail (c z : ℂ) : ℕ → ℂ
  | 0 => z
  | j + 1 => (orbitTail c z j) ^ 2 + c

theorem guarded_step {c z : ℂ} (hz : 4 ≤ ‖z‖)
    (hc : ‖c‖ ≤ ‖z‖ ^ 2 / 4) :
    (3 / 4 : ℝ) * ‖z‖ ^ 2 ≤ ‖z ^ 2 + c‖ ∧
      3 * ‖z‖ ≤ ‖z ^ 2 + c‖ := by
  have htriangle := norm_sub_le (z ^ 2 + c) c
  rw [add_sub_cancel_right, Complex.norm_pow] at htriangle
  constructor
  · linarith
  · nlinarith [mul_nonneg (show 0 ≤ ‖z‖ by positivity) (sub_nonneg.mpr hz)]

theorem guarded_growth {c z : ℂ} (hz : 4 ≤ ‖z‖)
    (hc : ‖c‖ ≤ ‖z‖ ^ 2 / 4) (j : ℕ) :
    4 ≤ ‖orbitTail c z j‖ ∧ ‖c‖ ≤ ‖orbitTail c z j‖ ^ 2 / 4 ∧
      (3 : ℝ) ^ j * ‖z‖ ≤ ‖orbitTail c z j‖ := by
  induction j with
  | zero => simpa [orbitTail] using And.intro hz hc
  | succ j ih =>
    obtain ⟨hr, hq, hg⟩ := ih
    have hstep := (guarded_step hr hq).2
    rw [← orbitTail] at hstep
    refine ⟨by linarith, hq.trans ?_, ?_⟩
    · have hmono : ‖orbitTail c z j‖ ≤ ‖orbitTail c z (j + 1)‖ := by linarith
      gcongr
    · rw [pow_succ]
      calc
        (3 : ℝ) ^ j * 3 * ‖z‖ = 3 * (3 ^ j * ‖z‖) := by ring
        _ ≤ 3 * ‖orbitTail c z j‖ := by linarith
        _ ≤ ‖orbitTail c z (j + 1)‖ := hstep

theorem potential_increment {c z : ℂ} (t : ℝ) (hz : z ≠ 0) (hn : z ^ 2 + c ≠ 0) :
    (t / 2) * Real.log ‖z ^ 2 + c‖ - t * Real.log ‖z‖ =
      (t / 2) * Real.log ‖1 + c / z ^ 2‖ := by
  have hfactor : z ^ 2 + c = z ^ 2 * (1 + c / z ^ 2) := by field_simp
  have hquot : 1 + c / z ^ 2 ≠ 0 := by
    intro h
    rw [hfactor, h, mul_zero] at hn
    exact hn rfl
  rw [hfactor, Complex.norm_mul, Complex.norm_pow,
    Real.log_mul (pow_ne_zero _ (norm_ne_zero_iff.mpr hz)) (norm_ne_zero_iff.mpr hquot),
    Real.log_pow]
  push_cast
  ring

theorem log_near_one {a : ℂ} (ha : ‖a‖ ≤ 1 / 4) :
    |Real.log ‖1 + a‖| ≤ (4 / 3 : ℝ) * ‖a‖ := by
  have hlow := norm_sub_le ((1 : ℂ) + a) a
  rw [add_sub_cancel_right, norm_one] at hlow
  have hhigh := norm_add_le (1 : ℂ) a
  rw [norm_one] at hhigh
  have hpos : 0 < ‖1 + a‖ := by linarith
  have hup := Real.log_le_sub_one_of_pos hpos
  have hdown := Real.log_le_sub_one_of_pos (inv_pos.mpr hpos)
  rw [Real.log_inv] at hdown
  rw [abs_le]
  constructor
  · have hrecip : ‖1 + a‖⁻¹ - 1 ≤ (4 / 3 : ℝ) * ‖a‖ := by
      rw [inv_eq_one_div, sub_le_iff_le_add, div_le_iff₀ hpos]
      nlinarith [mul_nonneg (norm_nonneg a) (show 0 ≤ ‖1 + a‖ - 3 / 4 by linarith)]
    linarith
  · linarith [norm_nonneg a]

theorem logarithmic_derivative_increment {c z d t : ℂ} (hz : z ≠ 0)
    (hn : z ^ 2 + c ≠ 0) :
    (t / 2) * (2 * z * d + 1) / (z ^ 2 + c) - t * d / z =
      (t / 2 - c * (t * d / z)) / (z ^ 2 + c) := by
  field_simp
  ring

theorem guarded_potential_increment {c z : ℂ} {t : ℝ} (ht : 0 ≤ t)
    (hz : 4 ≤ ‖z‖) (hc : ‖c‖ ≤ ‖z‖ ^ 2 / 4) :
    |(t / 2) * Real.log ‖z ^ 2 + c‖ - t * Real.log ‖z‖| ≤
      (4 / 3 : ℝ) * (t / 2) * (‖c‖ / ‖z‖ ^ 2) := by
  have hr : 0 < ‖z‖ := by linarith
  have hn : 0 < ‖z ^ 2 + c‖ := by linarith [(guarded_step hz hc).2]
  have ha : ‖c / z ^ 2‖ ≤ 1 / 4 := by
    rw [Complex.norm_div, Complex.norm_pow, div_le_iff₀ (pow_pos hr _)]
    linarith
  rw [potential_increment t (norm_pos_iff.mp hr) (norm_pos_iff.mp hn), abs_mul,
    abs_of_nonneg (by positivity : 0 ≤ t / 2)]
  calc
    (t / 2) * |Real.log ‖1 + c / z ^ 2‖| ≤ (t / 2) * ((4 / 3 : ℝ) * ‖c / z ^ 2‖) :=
      mul_le_mul_of_nonneg_left (log_near_one ha) (by positivity)
    _ = _ := by rw [Complex.norm_div, Complex.norm_pow]; ring

theorem guarded_logarithmic_derivative_increment {c z d : ℂ} {t : ℝ} (ht : 0 ≤ t)
    (hz : 4 ≤ ‖z‖) (hc : ‖c‖ ≤ ‖z‖ ^ 2 / 4) :
    ‖((t : ℂ) / 2) * (2 * z * d + 1) / (z ^ 2 + c) - (t : ℂ) * d / z‖ ≤
      (4 / 3 : ℝ) * (‖c‖ * ‖(t : ℂ) * d / z‖ + t / 2) / ‖z‖ ^ 2 := by
  have hr : 0 < ‖z‖ := by linarith
  have hstep := (guarded_step hz hc).1
  have hden : 0 < (3 / 4 : ℝ) * ‖z‖ ^ 2 := by positivity
  have hn := hden.trans_le hstep
  rw [logarithmic_derivative_increment (norm_pos_iff.mp hr) (norm_pos_iff.mp hn), Complex.norm_div]
  calc
    _ ≤ ‖((t / 2 : ℝ) : ℂ)‖ / ‖z ^ 2 + c‖ +
        ‖c * ((t : ℂ) * d / z)‖ / ‖z ^ 2 + c‖ := by
      rw [← add_div, Complex.ofReal_div, Complex.ofReal_ofNat]
      exact div_le_div_of_nonneg_right (norm_sub_le _ _) hn.le
    _ = (‖c‖ * ‖(t : ℂ) * d / z‖ + t / 2) / ‖z ^ 2 + c‖ := by
      rw [Complex.norm_real, Real.norm_eq_abs, abs_of_nonneg (by positivity : 0 ≤ t / 2),
        Complex.norm_mul, ← add_div, add_comm]
    _ ≤ (‖c‖ * ‖(t : ℂ) * d / z‖ + t / 2) / ((3 / 4 : ℝ) * ‖z‖ ^ 2) :=
      div_le_div_of_nonneg_left (by positivity) hden hstep
    _ = _ := by ring

theorem geometric_sums :
    HasSum (fun j : ℕ => (4 / 3 : ℝ) * (1 / 9) ^ j) (3 / 2) ∧
      HasSum (fun j : ℕ => (4 / 3 : ℝ) * (1 / 18) ^ j) (24 / 17) := by
  constructor
  · convert (hasSum_geometric_of_lt_one (r := (1 / 9 : ℝ)) (by norm_num) (by norm_num)).mul_left
      (4 / 3 : ℝ) using 1
    norm_num
  · convert (hasSum_geometric_of_lt_one (r := (1 / 18 : ℝ)) (by norm_num) (by norm_num)).mul_left
      (4 / 3 : ℝ) using 1
    norm_num

theorem finite_tail_envelope (L : ℕ → ℂ) (a b : ℕ → ℝ) {A B : ℝ}
    (ha : ∀ j, 0 ≤ a j) (hA : A < 1) (hB : 0 ≤ B)
    (haSum : ∀ m, ∑ j ∈ Finset.range m, a j ≤ A)
    (hbSum : ∀ m, ∑ j ∈ Finset.range m, b j ≤ B)
    (hstep : ∀ j, ‖L (j + 1) - L j‖ ≤ a j * ‖L j‖ + b j) (m : ℕ) :
    ‖L m - L 0‖ ≤ (A * ‖L 0‖ + B) / (1 - A) := by
  let M := (‖L 0‖ + B) / (1 - A)
  have hden : 0 < 1 - A := by linarith
  have hM : 0 ≤ M := div_nonneg (add_nonneg (norm_nonneg _) hB) hden.le
  have hbalance : ‖L 0‖ + B = M * (1 - A) := (eq_div_iff hden.ne').mp rfl |>.symm
  have hcombine (k : ℕ) : (∑ j ∈ Finset.range k, (a j * M + b j)) ≤ A * M + B := by
    rw [Finset.sum_add_distrib, ← Finset.sum_mul]
    exact add_le_add (mul_le_mul_of_nonneg_right (haSum k) hM) (hbSum k)
  have hprefix (k : ℕ) : ‖L k - L 0‖ ≤ ∑ j ∈ Finset.range k, (a j * M + b j) := by
    induction k with
    | zero => simp
    | succ k ih =>
      have htriangle : ‖L k‖ ≤ ‖L k - L 0‖ + ‖L 0‖ := by
        simpa using norm_add_le (L k - L 0) (L 0)
      have hnorm : ‖L k‖ ≤ M := by
        calc
          ‖L k‖ ≤ ‖L k - L 0‖ + ‖L 0‖ := htriangle
          _ ≤ (A * M + B) + ‖L 0‖ := add_le_add_right (ih.trans (hcombine k)) _
          _ = M := by linarith
      rw [Finset.sum_range_succ]
      calc
        ‖L (k + 1) - L 0‖ ≤ ‖L k - L 0‖ + ‖L (k + 1) - L k‖ := by
          simpa [sub_eq_add_neg, add_assoc, add_left_comm, add_comm] using
            norm_add_le (L k - L 0) (L (k + 1) - L k)
        _ ≤ _ := add_le_add ih ((hstep k).trans (add_le_add_right
          (mul_le_mul_of_nonneg_left hnorm (ha k)) _))
  calc
    ‖L m - L 0‖ ≤ A * M + B := (hprefix m).trans (hcombine m)
    _ = (A * ‖L 0‖ + B) / (1 - A) := (eq_div_iff hden.ne').mpr (by nlinarith)

theorem ratio_enclosure {G Gn Ln EG EL : ℝ} {L : ℂ}
    (hG : 0 ≤ G) (hEG : |G - Gn| ≤ EG)
    (hEL : |‖L‖ - Ln| ≤ EL) (hguard : EL < Ln) :
    max 0 (Gn - EG) / (Ln + EL) ≤ G / ‖L‖ ∧
      G / ‖L‖ ≤ (Gn + EG) / (Ln - EL) := by
  rw [abs_le] at hEG hEL
  have hlow : 0 < Ln - EL := by linarith
  have hnorm : 0 < ‖L‖ := by linarith [hEL.1]
  have hup : 0 < Ln + EL := by linarith [hEL.2]
  constructor
  · calc
      max 0 (Gn - EG) / (Ln + EL) ≤ G / (Ln + EL) :=
        div_le_div_of_nonneg_right (max_le hG (by linarith [hEG.1])) hup.le
      _ ≤ G / ‖L‖ := div_le_div_of_nonneg_left hG hnorm (by linarith [hEL.2])
  · exact (div_le_div_of_nonneg_right (show G ≤ Gn + EG by linarith [hEG.2]) hnorm.le).trans
      (div_le_div_of_nonneg_left (by linarith [hEG.2]) hlow (by linarith [hEL.1]))

end FractalSync.ExteriorRulerBounds
