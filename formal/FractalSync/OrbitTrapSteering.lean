import Mathlib.Analysis.SpecialFunctions.Sqrt
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Linarith
import Mathlib.Tactic.NormNum
import Mathlib.Tactic.Ring

/-!
# Steering along a fixed point-trap preimage

For the critical orbit of `z ↦ z²+c`, the second iterate is `c²+c`.
We use its squared distance to the origin and the metric `ρ⁻² I`.
The coordinate pair carries an explicit Euclidean pairing here.
-/

namespace FractalSync.OrbitTrapSteering

noncomputable section

abbrev Point := ℝ × ℝ

def orbitRe (c : Point) : ℝ := c.1 ^ 2 - c.2 ^ 2 + c.1

def orbitIm (c : Point) : ℝ := (2 * c.1 + 1) * c.2

def trap (c : Point) : ℝ := orbitRe c ^ 2 + orbitIm c ^ 2

def trapGradient (c : Point) : Point :=
  (2 * ((2 * c.1 + 1) * orbitRe c + 2 * c.2 * orbitIm c),
   2 * (-(2 * c.2) * orbitRe c + (2 * c.1 + 1) * orbitIm c))

def pairing (u v : Point) : ℝ := u.1 * v.1 + u.2 * v.2

def differential (g : Point) : Point →L[ℝ] ℝ :=
  g.1 • ContinuousLinearMap.fst ℝ ℝ ℝ + g.2 • ContinuousLinearMap.snd ℝ ℝ ℝ

theorem differential_apply (g v : Point) : differential g v = pairing g v := by
  simp [differential, pairing]

theorem hasFDerivAt_trap (c : Point) : HasFDerivAt trap (differential (trapGradient c)) c := by
  have hx := (ContinuousLinearMap.fst ℝ ℝ ℝ).hasFDerivAt (x := c)
  have hy := (ContinuousLinearMap.snd ℝ ℝ ℝ).hasFDerivAt (x := c)
  have hu := ((hx.mul hx).sub (hy.mul hy)).add hx
  have hv := (((hx.const_mul 2).add_const 1).mul hy)
  convert ((hu.mul hu).add (hv.mul hv)) using 1
  · funext q
    simp [trap, orbitRe, orbitIm, pow_two]
  · apply ContinuousLinearMap.ext
    intro v
    simp [differential, trapGradient, orbitRe, orbitIm, pow_two]
    ring

def euclideanNorm (g : Point) : ℝ := Real.sqrt (pairing g g)

/-- The positive quarter-turn of the metric gradient, normalized in `ρ⁻² I`. -/
def unitTangent (ρ : ℝ) (g : Point) : Point :=
  (ρ * (-g.2) / euclideanNorm g, ρ * g.1 / euclideanNorm g)

def metricSpeedSq (ρ : ℝ) (v : Point) : ℝ := pairing v v / ρ ^ 2

theorem metricSpeedSq_unitTangent (ρ : ℝ) (g : Point)
    (hρ : 0 < ρ) (hg : 0 < pairing g g) :
    metricSpeedSq ρ (unitTangent ρ g) = 1 := by
  have hs : (euclideanNorm g) ^ 2 = pairing g g := Real.sq_sqrt hg.le
  have hn : euclideanNorm g ≠ 0 := (Real.sqrt_pos.2 hg).ne'
  unfold metricSpeedSq unitTangent
  simp only [pairing, div_eq_iff (pow_ne_zero 2 hρ.ne')]
  field_simp [hn]
  rw [hs]
  unfold pairing
  ring

theorem differential_unitTangent (ρ : ℝ) (g : Point) :
    differential g (unitTangent ρ g) = 0 := by
  rw [differential_apply]
  unfold pairing unitTangent
  rcases eq_or_ne (euclideanNorm g) 0 with h | h
  · simp [h]
  · field_simp [h]
    ring

theorem regular_trap_tangent (c : Point) (ρ : ℝ)
    (hρ : 0 < ρ) (hreg : 0 < pairing (trapGradient c) (trapGradient c)) :
    metricSpeedSq ρ (unitTangent ρ (trapGradient c)) = 1 ∧
    fderiv ℝ trap c (unitTangent ρ (trapGradient c)) = 0 := by
  rw [(hasFDerivAt_trap c).fderiv]
  exact ⟨metricSpeedSq_unitTangent ρ _ hρ hreg, differential_unitTangent ρ _⟩

theorem trapGradient_sq (c : Point) :
    pairing (trapGradient c) (trapGradient c) =
      4 * ((2 * c.1 + 1) ^ 2 + (2 * c.2) ^ 2) * trap c := by
  unfold pairing trapGradient trap
  ring

theorem trapGradient_critical (c : Point) :
    pairing (trapGradient c) (trapGradient c) = 0 ↔
      c = (-(1 : ℝ) / 2, 0) ∨ c = (0, 0) ∨ c = (-1, 0) := by
  rw [trapGradient_sq]
  constructor
  · intro h
    have hzero : ((2 * c.1 + 1) ^ 2 + (2 * c.2) ^ 2) = 0 ∨ trap c = 0 := by
      simpa [mul_eq_zero] using (mul_eq_zero.mp h)
    rcases hzero with hd | hz
    · left
      apply Prod.ext
      · have hx : (2 * c.1 + 1) ^ 2 = 0 := by nlinarith [sq_nonneg (2 * c.2)]
        have hx' := sq_eq_zero_iff.mp hx
        change c.1 = -(1 : ℝ) / 2
        linarith
      · have hy : (2 * c.2) ^ 2 = 0 := by nlinarith [sq_nonneg (2 * c.1 + 1)]
        have hy' := sq_eq_zero_iff.mp hy
        change c.2 = 0
        linarith
    · have hu : orbitRe c = 0 := by
        apply sq_eq_zero_iff.mp
        unfold trap at hz
        nlinarith [sq_nonneg (orbitIm c)]
      have hv : orbitIm c = 0 := by
        apply sq_eq_zero_iff.mp
        unfold trap at hz
        nlinarith [sq_nonneg (orbitRe c)]
      have hy : c.2 = 0 := by
        unfold orbitIm at hv
        rcases mul_eq_zero.mp hv with hx | hy
        · unfold orbitRe at hu
          nlinarith [sq_nonneg c.2]
        · exact hy
      have hx : c.1 * (c.1 + 1) = 0 := by
        unfold orbitRe at hu
        rw [hy] at hu
        nlinarith
      right
      rcases mul_eq_zero.mp hx with hx | hx
      · exact Or.inl (Prod.ext hx hy)
      · exact Or.inr (Prod.ext (by linarith) hy)
  · rintro (rfl | rfl | rfl) <;> norm_num [trap, orbitRe, orbitIm]

theorem trap_at_saddle (x y : ℝ) :
    trap (x - 1 / 2, y) - 1 / 16 =
      (x ^ 2 + y ^ 2) ^ 2 - (x ^ 2 - y ^ 2) / 2 := by
  unfold trap orbitRe orbitIm
  dsimp only
  ring

end

end FractalSync.OrbitTrapSteering
