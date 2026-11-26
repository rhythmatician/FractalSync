import Mathlib.Analysis.Calculus.ContDiff.Defs
import Mathlib.Analysis.Normed.Operator.NormedSpace
import Mathlib.Topology.MetricSpace.HausdorffDistance

/-!
# Gluing a quartically vanishing field through a closed set

The derivative fields can come from unrelated components of the complement.
The estimates are needed only in a common collar of the closed set.
-/

namespace FractalSync.ShoreFloorGluing

open Filter Set
open scoped Topology

variable {E F : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [NormedAddCommGroup F] [NormedSpace ℝ F]

/-- A field has a distance-power bound in a collar of `shore`. -/
def CollarBound (shore : Set E) (radius : ℝ) (f : E → F) (power : ℕ) : Prop :=
  ∃ C : ℝ, 0 ≤ C ∧ ∀ x, Metric.infDist x shore < radius →
    ‖f x‖ ≤ C * Metric.infDist x shore ^ power

omit [NormedSpace ℝ E] [NormedSpace ℝ F] in
theorem bound_at_shore {shore : Set E} {radius : ℝ} {f : E → F} {power : ℕ}
    (h : CollarBound shore radius f power) {x : E} (hx : x ∈ shore) :
    ∃ C : ℝ, 0 ≤ C ∧ ∀ y, dist y x < radius → ‖f y‖ ≤ C * dist y x ^ power := by
  obtain ⟨C, hC, hb⟩ := h
  refine ⟨C, hC, fun y hy => ?_⟩
  have hd := Metric.infDist_le_dist_of_mem (x := y) hx
  exact (hb y (hd.trans_lt hy)).trans (mul_le_mul_of_nonneg_left
    (pow_le_pow_left₀ Metric.infDist_nonneg hd _) hC)

omit [NormedSpace ℝ E] [NormedSpace ℝ F] in
theorem eq_zero_on_shore {shore : Set E} {radius : ℝ} {f : E → F} {power : ℕ}
    (h : CollarBound shore radius f power) (hp : power ≠ 0) (hr : 0 < radius)
    {x : E} (hx : x ∈ shore) : f x = 0 := by
  obtain ⟨C, _, hb⟩ := h
  have hd := Metric.infDist_zero_of_mem hx
  exact norm_eq_zero.mp (le_antisymm (by simpa [hd, zero_pow hp] using hb x (hd ▸ hr)) (norm_nonneg _))

theorem hasFDerivAt_zero_on_shore {shore : Set E} {radius : ℝ} {f : E → F} {power : ℕ}
    (h : CollarBound shore radius f power) (hp : 2 ≤ power) (hr : 0 < radius)
    {x : E} (hx : x ∈ shore) : HasFDerivAt f (0 : E →L[ℝ] F) x := by
  have hf := eq_zero_on_shore h (by omega) hr hx
  obtain ⟨C, hC, hb⟩ := bound_at_shore h hx
  rw [hasFDerivAt_iff_isLittleO_nhds_zero, Asymptotics.isLittleO_iff]
  intro ε hε
  have hδ : 0 < min radius (min 1 (ε / (C + 1))) :=
    lt_min hr (lt_min zero_lt_one (div_pos hε (by linarith)))
  filter_upwards [Metric.ball_mem_nhds (0 : E) hδ] with v hv
  have hv' : ‖v‖ < min radius (min 1 (ε / (C + 1))) := by simpa [Metric.mem_ball] using hv
  have hv₁ := lt_of_lt_of_le hv' (min_le_left _ _)
  have hv₂ := lt_of_lt_of_le hv' ((min_le_right _ _).trans (min_le_left _ _))
  have hv₃ := lt_of_lt_of_le hv' ((min_le_right _ _).trans (min_le_right _ _))
  have hd : dist (x + v) x = ‖v‖ := by rw [dist_eq_norm, add_sub_cancel_left]
  have hpv : ‖v‖ ^ power ≤ ‖v‖ ^ 2 := pow_le_pow_of_le_one (norm_nonneg _) hv₂.le hp
  have hCv : C * ‖v‖ ≤ ε := by
    have h := (lt_div_iff₀ (by linarith : 0 < C + 1)).mp hv₃
    nlinarith [norm_nonneg v]
  simp only [hf, sub_zero, ContinuousLinearMap.zero_apply]
  calc
    ‖f (x + v)‖ ≤ C * ‖v‖ ^ power := by simpa [hd] using hb (x + v) (hd.symm ▸ hv₁)
    _ ≤ C * ‖v‖ ^ 2 := mul_le_mul_of_nonneg_left hpv hC
    _ ≤ ε * ‖v‖ := by simpa [pow_two, mul_assoc] using mul_le_mul_of_nonneg_right hCv (norm_nonneg v)

omit [NormedSpace ℝ E] [NormedSpace ℝ F] in
theorem continuousAt_on_shore {shore : Set E} {radius : ℝ} {f : E → F}
    (h : CollarBound shore radius f 1) (hr : 0 < radius)
    {x : E} (hx : x ∈ shore) : ContinuousAt f x := by
  have hf := eq_zero_on_shore h (by decide) hr hx
  obtain ⟨C, hC, hb⟩ := bound_at_shore h hx
  rw [Metric.continuousAt_iff]
  intro ε hε
  refine ⟨min radius (ε / (C + 1)), lt_min hr (div_pos hε (by linarith)), ?_⟩
  intro y hy
  have hy₁ := hy.trans_le (min_le_left _ _)
  have hy₂ := hy.trans_le (min_le_right _ _)
  have hCd : C * dist y x < ε := by
    have h := (lt_div_iff₀ (by linarith : 0 < C + 1)).mp hy₂
    nlinarith [dist_nonneg (x := y) (y := x)]
  simpa [hf, dist_zero_right] using (by simpa using hb y hy₁ : ‖f y‖ ≤ C * dist y x).trans_lt hCd

theorem hasFDerivAt_of_collar_bounds {shore : Set E} {radius : ℝ} (hr : 0 < radius)
    {f : E → F} {f' : E → E →L[ℝ] F} {p q : ℕ} (hp : 2 ≤ p) (hq : q ≠ 0)
    (hb : CollarBound shore radius f p) (hb' : CollarBound shore radius f' q)
    (hd : ∀ x ∉ shore, HasFDerivAt f (f' x) x) (x : E) : HasFDerivAt f (f' x) x := by
  by_cases hx : x ∈ shore
  · rw [eq_zero_on_shore hb' hq hr hx]
    exact hasFDerivAt_zero_on_shore hb hp hr hx
  · exact hd x hx

noncomputable local instance (priority := 2000) thirdDerivativeNorm :
    NormedAddCommGroup (E →L[ℝ] E →L[ℝ] E →L[ℝ] F) :=
  ContinuousLinearMap.toNormedAddCommGroup (σ₁₂ := RingHom.id ℝ) (E := E)
    (F := E →L[ℝ] E →L[ℝ] F)

/-- Three source derivatives and a continuous third derivative glue across the
Shore if their collar orders are four, three, two and one, respectively. -/
theorem contDiff_three_of_collar_bounds {shore : Set E} {radius : ℝ}
    (hr : 0 < radius) (f₀ : E → F) (f₁ : E → E →L[ℝ] F)
    (f₂ : E → E →L[ℝ] E →L[ℝ] F) (f₃ : E → E →L[ℝ] E →L[ℝ] E →L[ℝ] F)
    (hb₀ : CollarBound shore radius f₀ 4) (hb₁ : CollarBound shore radius f₁ 3)
    (hb₂ : CollarBound shore radius f₂ 2) (hb₃ : CollarBound shore radius f₃ 1)
    (h₀ : ∀ x ∉ shore, HasFDerivAt f₀ (f₁ x) x)
    (h₁ : ∀ x ∉ shore, HasFDerivAt f₁ (f₂ x) x)
    (h₂ : ∀ x ∉ shore, HasFDerivAt f₂ (f₃ x) x)
    (h₃ : ∀ x ∉ shore, ContinuousAt f₃ x) :
    ContDiff ℝ 3 f₀ ∧ fderiv ℝ f₀ = f₁ ∧ fderiv ℝ f₁ = f₂ ∧ fderiv ℝ f₂ = f₃ := by
  have hd₀ := hasFDerivAt_of_collar_bounds hr (by decide) (by decide) hb₀ hb₁ h₀
  have hd₁ := hasFDerivAt_of_collar_bounds hr (by decide) (by decide) hb₁ hb₂ h₁
  have hd₂ := hasFDerivAt_of_collar_bounds (p := 2) (q := 1) hr (by decide) (by decide) hb₂ hb₃ h₂
  have he₀ : fderiv ℝ f₀ = f₁ := funext fun x => (hd₀ x).fderiv
  have he₁ : fderiv ℝ f₁ = f₂ := funext fun x => (hd₁ x).fderiv
  have he₂ : fderiv ℝ f₂ = f₃ := by
    funext x
    exact HasFDerivAt.fderiv (𝕜 := ℝ) (E := E) (F := E →L[ℝ] E →L[ℝ] F) (hd₂ x)
  have hc₃ : Continuous f₃ := continuous_iff_continuousAt.2 fun x => by
    by_cases hx : x ∈ shore
    · exact continuousAt_on_shore hb₃ hr hx
    · exact h₃ x hx
  refine ⟨?_, he₀, he₁, he₂⟩
  rw [show (3 : WithTop ℕ∞) = 2 + 1 from rfl, contDiff_succ_iff_fderiv, he₀]
  refine ⟨fun x => (hd₀ x).differentiableAt, by simp, ?_⟩
  rw [show (2 : WithTop ℕ∞) = 1 + 1 from rfl, contDiff_succ_iff_fderiv, he₁]
  refine ⟨fun x => (hd₁ x).differentiableAt, by simp, contDiff_one_iff_fderiv.mpr ⟨?_, by rwa [he₂]⟩⟩
  intro x
  exact HasFDerivAt.differentiableAt (𝕜 := ℝ) (E := E) (F := E →L[ℝ] E →L[ℝ] F) (hd₂ x)

end FractalSync.ShoreFloorGluing
