import Mathlib.Analysis.Calculus.Deriv.MeanValue
import Mathlib.Analysis.Calculus.Deriv.Pow
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.Ring

/-!
# Squared-speed bounds along a prescribed path

Arc length is the scalar path coordinate. The orthonormal-frame calculation takes
the tangential and normal components as inputs to the geometric decomposition.
-/

namespace FractalSync.TrapSteeringBudget

open Set

theorem hasDerivAt_squaredSpeed (v : ℝ → ℝ) (s v' : ℝ)
    (hv : HasDerivAt v v' s) :
    HasDerivAt (fun q => v q ^ 2) (2 * v s * v') s := by
  convert hv.pow 2 using 1; ring

theorem hasDerivAt_timeSpeed (v arc : ℝ → ℝ) (t v' : ℝ)
    (hv : HasDerivAt v v' (arc t)) (harc : HasDerivAt arc (v (arc t)) t) :
    HasDerivAt (v ∘ arc) (deriv (fun q => v q ^ 2) (arc t) / 2) t := by
  rw [(hasDerivAt_squaredSpeed v (arc t) v' hv).deriv]
  convert hv.comp t harc using 1; ring

section Frame

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

theorem acceleration_sq_in_frame (T N : E) (w' κ w : ℝ)
    (hT : ‖T‖ = 1) (hN : ‖N‖ = 1) (hperp : inner ℝ T N = 0) :
    ‖(w' / 2) • T + (κ * w) • N‖ ^ 2 = (w' / 2) ^ 2 + κ ^ 2 * w ^ 2 := by
  rw [norm_add_sq_real, real_inner_smul_left, real_inner_smul_right, hperp]
  simp only [norm_smul, Real.norm_eq_abs, hT, hN, mul_one, sq_abs, mul_zero]
  ring

theorem orthogonal_handoff_stops (T N : E) (vin vout : ℝ)
    (hT : ‖T‖ = 1) (hN : ‖N‖ = 1) (hperp : inner ℝ T N = 0)
    (hmatch : vin • T = vout • N) : vin = 0 ∧ vout = 0 := by
  have hzero : vin • T + (-vout) • N = 0 := by
    rw [hmatch, neg_smul, add_neg_cancel]
  have hsq := acceleration_sq_in_frame T N (2 * vin) 1 (-vout) hT hN hperp
  rw [show 2 * vin / 2 = vin by ring, one_mul, hzero] at hsq
  simp at hsq
  constructor <;> nlinarith [sq_nonneg vin, sq_nonneg vout, hsq]

end Frame

theorem acceleration_sq_le_of_split (w' κ w K atang anorm acap : ℝ)
    (hw : 0 ≤ w) (ht : |w'| ≤ 2 * atang) (hκ : |κ| ≤ K)
    (hn : K * w ≤ anorm) (hcap : atang ^ 2 + anorm ^ 2 ≤ acap ^ 2) :
    (w' / 2) ^ 2 + κ ^ 2 * w ^ 2 ≤ acap ^ 2 := by
  have ht' : |w' / 2| ≤ atang := by
    rw [abs_div, abs_of_pos (by norm_num : (0 : ℝ) < 2)]
    linarith
  have hn' : |κ * w| ≤ anorm := by
    rw [abs_mul, abs_of_nonneg hw]
    exact (mul_le_mul_of_nonneg_right hκ hw).trans hn
  have htSq : (w' / 2) ^ 2 ≤ atang ^ 2 := sq_le_sq.mpr (ht'.trans (le_abs_self _))
  have hnSq : (κ * w) ^ 2 ≤ anorm ^ 2 := sq_le_sq.mpr (hn'.trans (le_abs_self _))
  rw [mul_pow] at hnSq
  linarith

theorem squaredSpeed_ramp (w : ℝ → ℝ) (s₀ s₁ atang : ℝ)
    (hs : s₀ ≤ s₁) (hc : ContinuousOn w (Icc s₀ s₁))
    (hd : DifferentiableOn ℝ w (Ioo s₀ s₁))
    (ht : ∀ s ∈ Ioo s₀ s₁, |deriv w s| ≤ 2 * atang) :
    |w s₁ - w s₀| ≤ 2 * atang * (s₁ - s₀) := by
  rcases hs.eq_or_lt with rfl | hs'
  · simp
  obtain ⟨s, hs, hsecant⟩ := exists_deriv_eq_slope w hs' hc hd
  have h := ht s hs
  rw [hsecant, abs_div, abs_of_pos (sub_pos.mpr hs'), div_le_iff₀ (sub_pos.mpr hs')] at h
  exact h

theorem squaredSpeed_stop_envelope (w : ℝ → ℝ) (s₀ s s₁ atang : ℝ)
    (h₀ : s₀ ≤ s) (h₁ : s ≤ s₁) (hc : ContinuousOn w (Icc s₀ s₁))
    (hd : DifferentiableOn ℝ w (Ioo s₀ s₁))
    (ht : ∀ q ∈ Ioo s₀ s₁, |deriv w q| ≤ 2 * atang)
    (hstart : w s₀ = 0) (hstop : w s₁ = 0) :
    w s ≤ min (2 * atang * (s - s₀)) (2 * atang * (s₁ - s)) := by
  have hleft := squaredSpeed_ramp w s₀ s atang h₀
    (hc.mono fun _ h => ⟨h.1, h.2.trans h₁⟩)
    (hd.mono fun _ h => ⟨h.1, h.2.trans_le h₁⟩)
    (fun q h => ht q ⟨h.1, h.2.trans_le h₁⟩)
  have hright := squaredSpeed_ramp w s s₁ atang h₁
    (hc.mono fun _ h => ⟨h₀.trans h.1, h.2⟩)
    (hd.mono fun _ h => ⟨h₀.trans_lt h.1, h.2⟩)
    (fun q h => ht q ⟨h₀.trans_lt h.1, h.2⟩)
  rw [hstart, sub_zero] at hleft
  rw [hstop, zero_sub, abs_neg] at hright
  exact le_min ((le_abs_self _).trans hleft) ((le_abs_self _).trans hright)

/-- A stop-to-stop squared-speed profile, with peak `W` at the midpoint. -/
noncomputable def stopProfile (s₀ L W s : ℝ) : ℝ :=
  let u := (s - s₀) / L
  4 * W * u * (1 - u)

theorem hasDerivAt_stopProfile (s₀ L W s : ℝ) :
    HasDerivAt (stopProfile s₀ L W) (4 * W * (1 - 2 * ((s - s₀) / L)) / L) s := by
  have hu := ((hasDerivAt_id s).sub_const s₀).div_const L
  unfold stopProfile
  convert (hu.const_mul (4 * W)).mul (hu.const_sub 1) using 1
  simp only [id_eq]
  ring

theorem stopProfile_acceleration_sq_le (s₀ L W s κ K atang anorm acap : ℝ)
    (hL : 0 < L) (hW : 0 ≤ W) (hs : s ∈ Icc s₀ (s₀ + L))
    (ht : 2 * W / L ≤ atang) (hκ : |κ| ≤ K) (hn : K * W ≤ anorm)
    (hcap : atang ^ 2 + anorm ^ 2 ≤ acap ^ 2) :
    (deriv (stopProfile s₀ L W) s / 2) ^ 2 + κ ^ 2 * stopProfile s₀ L W s ^ 2 ≤ acap ^ 2 := by
  let u := (s - s₀) / L
  have hu₀ : 0 ≤ u := div_nonneg (sub_nonneg.mpr hs.1) hL.le
  have hu₁ : u ≤ 1 := (div_le_iff₀ hL).2 (by linarith [hs.2])
  have hw₀ : 0 ≤ stopProfile s₀ L W s :=
    mul_nonneg (mul_nonneg (by positivity) hu₀) (sub_nonneg.mpr hu₁)
  have hw₁ : stopProfile s₀ L W s ≤ W := by
    change 4 * W * u * (1 - u) ≤ W
    nlinarith only [mul_nonneg hW (sq_nonneg (2 * u - 1))]
  have hK : 0 ≤ K := (abs_nonneg κ).trans hκ
  apply acceleration_sq_le_of_split _ κ _ K atang anorm acap hw₀ _ hκ
    ((mul_le_mul_of_nonneg_left hw₁ hK).trans hn) hcap
  rw [(hasDerivAt_stopProfile s₀ L W s).deriv]
  change |4 * W * (1 - 2 * u) / L| ≤ 2 * atang
  have hu : |1 - 2 * u| ≤ 1 := abs_le.mpr ⟨by linarith, by linarith⟩
  have h4W : 0 ≤ 4 * W := by positivity
  rw [abs_div, abs_mul, abs_of_nonneg h4W, abs_of_pos hL]
  calc
    4 * W * |1 - 2 * u| / L ≤ 4 * W * 1 / L :=
      (div_le_div_iff_of_pos_right hL).mpr (mul_le_mul_of_nonneg_left hu h4W)
    _ = 2 * (2 * W / L) := by ring
    _ ≤ 2 * atang := mul_le_mul_of_nonneg_left ht (by norm_num)

end FractalSync.TrapSteeringBudget
