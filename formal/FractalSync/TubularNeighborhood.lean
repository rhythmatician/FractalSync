import FractalSync.CurvedShoreGeometry
import FractalSync.Ruler
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Topology.MetricSpace.HausdorffDistance

/-!
# Actual distance in a certified normal neighborhood

A two-sided quadratic normal-chord bound gives an explicit normal radius,
unique nearest points, and an actual distance-to-set identity. The later
normal-coordinate calculus uses actual curve derivatives.
-/

namespace FractalSync.TubularNeighborhood

open scoped RealInnerProductSpace Topology

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

def normalOffset (p n : E) (r : ℝ) : E := p + r • n

/-- A geometric normal-chord certificate, stated without a distance infimum. -/
def NormalChordBound (S : Set E) (R : ℝ) (p n : E) : Prop :=
  ∀ q ∈ S, 2 * R * |inner ℝ n (q - p)| ≤ ‖q - p‖ ^ 2

/-- Exact squared-distance expansion in a unit normal direction. -/
theorem normalOffset_dist_sq (r : ℝ) (p n q : E) (hn : ‖n‖ = 1) :
    dist (normalOffset p n r) q ^ 2 =
      r ^ 2 + ‖q - p‖ ^ 2 - 2 * r * inner ℝ n (q - p) := by
  rw [dist_eq_norm, show normalOffset p n r - q = r • n - (q - p) by
    simp only [normalOffset]; abel, norm_sub_sq_real]
  simp only [norm_smul, Real.norm_eq_abs, hn, mul_one, sq_abs, real_inner_smul_left]
  ring

/-- Quantitative separation from other Shore points along a normal ray. -/
theorem normalOffset_dist_sq_gap (S : Set E) (R r : ℝ) (p n q : E)
    (hR : 0 < R) (hn : ‖n‖ = 1) (hb : NormalChordBound S R p n) (hq : q ∈ S) :
    (1 - |r| / R) * ‖q - p‖ ^ 2 ≤ dist (normalOffset p n r) q ^ 2 - r ^ 2 := by
  rw [normalOffset_dist_sq r p n q hn]
  have hbscaled := mul_le_mul_of_nonneg_left (hb q hq) (abs_nonneg r)
  have hcross := mul_le_mul_of_nonneg_left
    ((le_abs_self (r * inner ℝ n (q - p))).trans_eq (abs_mul _ _)) hR.le
  have heq : 1 - |r| / R = (R - |r|) / R := by rw [sub_div, div_self hR.ne']
  rw [heq, div_mul_eq_mul_div, div_le_iff₀ hR]
  nlinarith

/-- The certified normal coordinate is the actual unsigned distance to Shore. -/
theorem infDist_normalOffset (S : Set E) (R r : ℝ) (p n : E)
    (hR : 0 < R) (hp : p ∈ S) (hn : ‖n‖ = 1)
    (hb : NormalChordBound S R p n) (hr : |r| ≤ R) :
    Metric.infDist (normalOffset p n r) S = |r| := by
  apply le_antisymm
  · simpa [normalOffset, dist_eq_norm, norm_smul, hn] using
      (Metric.infDist_le_dist_of_mem (x := normalOffset p n r) hp)
  · apply (Metric.le_infDist (show S.Nonempty from ⟨p, hp⟩)).2
    intro q hq
    have hnonneg : 0 ≤ (1 - |r| / R) * ‖q - p‖ ^ 2 :=
      mul_nonneg (sub_nonneg.mpr ((div_le_one hR).2 hr)) (sq_nonneg _)
    nlinarith [normalOffset_dist_sq_gap S R r p n q hR hn hb hq,
      sq_abs r, dist_nonneg (x := normalOffset p n r) (y := q)]

/-- Inside the certified radius, the foot point is the unique nearest point. -/
theorem normalOffset_unique_nearest (S : Set E) (R r : ℝ) (p n q : E)
    (hR : 0 < R) (hp : p ∈ S) (hn : ‖n‖ = 1)
    (hb : NormalChordBound S R p n) (hr : |r| < R) (hq : q ∈ S)
    (hnear : dist (normalOffset p n r) q = Metric.infDist (normalOffset p n r) S) :
    q = p := by
  by_contra hne
  have hpos := mul_pos (sub_pos.mpr ((div_lt_one hR).2 hr))
    (sq_pos_of_pos (norm_pos_iff.mpr (sub_ne_zero.mpr hne)))
  have hgap := normalOffset_dist_sq_gap S R r p n q hR hn hb hq
  rw [hnear, infDist_normalOffset S R r p n hR hp hn hb hr.le, sq_abs, sub_self] at hgap
  exact hpos.not_ge hgap

/-- The inward radial direction on a positive-radius sphere is unit. -/
theorem sphere_inwardNormal_norm (R : ℝ) (p : E) (hR : 0 < R) (hp : ‖p‖ = R) :
    ‖(-(R⁻¹)) • p‖ = 1 := by
  simp [norm_smul, hp, abs_of_pos hR, hR.ne']

/-- Spheres supply the normal-chord certificate with their actual radius. -/
theorem sphere_normalChordBound (R : ℝ) (p : E) (hR : 0 < R) (hp : ‖p‖ = R) :
    NormalChordBound (Metric.sphere (0 : E) R) R p (-(R⁻¹) • p) := by
  intro q hq
  have hqn : ‖q‖ = R := by simpa [Metric.mem_sphere, dist_eq_norm] using hq
  have hid : 2 * R * inner ℝ (-(R⁻¹) • p) (q - p) = ‖q - p‖ ^ 2 := by
    simp only [real_inner_smul_left, inner_sub_right, real_inner_self_eq_norm_sq,
      hp, norm_sub_sq_real, hqn, real_inner_comm p q]
    field_simp
    ring
  have hnonneg : 0 ≤ inner ℝ (-(R⁻¹) • p) (q - p) := by
    nlinarith [sq_nonneg ‖q - p‖]
  rw [abs_of_nonneg hnonneg, hid]

/-- The sphere instance identifies the actual distance along its normal rays. -/
theorem infDist_sphere_normalOffset (R r : ℝ) (p : E)
    (hR : 0 < R) (hp : ‖p‖ = R) (hr : |r| ≤ R) :
    Metric.infDist (normalOffset p (-(R⁻¹) • p) r) (Metric.sphere (0 : E) R) = |r| := by
  exact infDist_normalOffset _ R r p _ hR (by simpa [Metric.mem_sphere, dist_eq_norm] using hp)
    (sphere_inwardNormal_norm R p hR hp) (sphere_normalChordBound R p hR hp) hr

/-- Strictly inside the normal radius of a sphere, its foot is the unique closest point. -/
theorem sphere_normalOffset_unique_nearest (R r : ℝ) (p q : E)
    (hR : 0 < R) (hp : ‖p‖ = R) (hr : |r| < R) (hq : q ∈ Metric.sphere (0 : E) R)
    (hnear : dist (normalOffset p (-(R⁻¹) • p) r) q =
      Metric.infDist (normalOffset p (-(R⁻¹) • p) r) (Metric.sphere (0 : E) R)) :
    q = p := by
  exact normalOffset_unique_nearest _ R r p _ q hR
    (by simpa [Metric.mem_sphere, dist_eq_norm] using hp)
    (sphere_inwardNormal_norm R p hR hp) (sphere_normalChordBound R p hR hp) hr hq hnear

/-- The normal map of an actual parameterized Shore and normal field. -/
def normalMap (C N : ℝ → E) (z : ℝ × ℝ) : E := normalOffset (C z.2) (N z.2) z.1

/-- A common chord certificate prevents distinct normal coordinates from colliding. -/
theorem normalMap_injOn_strip (S : Set E) (R : ℝ) (C N : ℝ → E) (U : Set ℝ)
    (hR : 0 < R) (hC : Set.InjOn C U) (hmem : ∀ y ∈ U, C y ∈ S)
    (hN : ∀ y ∈ U, ‖N y‖ = 1)
    (hb : ∀ y ∈ U, NormalChordBound S R (C y) (N y)) :
    Set.InjOn (normalMap C N) {z : ℝ × ℝ | z.2 ∈ U ∧ |z.1| < R} := by
  intro x hx y hy hxy
  have hnear : dist (normalOffset (C x.2) (N x.2) x.1) (C y.2) =
      Metric.infDist (normalOffset (C x.2) (N x.2) x.1) S := by
    change dist (normalMap C N x) (C y.2) = Metric.infDist (normalMap C N x) S
    rw [hxy]
    simpa [normalMap, normalOffset, dist_eq_norm, norm_smul, hN y.2 hy.1] using
      (infDist_normalOffset S R y.1 (C y.2) (N y.2) hR (hmem y.2 hy.1)
        (hN y.2 hy.1) (hb y.2 hy.1) hy.2.le).symm
  have hfoot := normalOffset_unique_nearest S R x.1 (C x.2) (N x.2) (C y.2) hR
    (hmem x.2 hx.1) (hN x.2 hx.1) (hb x.2 hx.1) hx.2 (hmem y.2 hy.1) hnear
  have hyx : y.2 = x.2 := hC hy.1 hx.1 hfoot
  have hsmul : x.1 • N x.2 = y.1 • N x.2 := by simpa [normalMap, normalOffset, hyx] using hxy
  have hnne : N x.2 ≠ 0 := norm_ne_zero_iff.mp ((hN x.2 hx.1).trans_ne one_ne_zero)
  exact Prod.ext (smul_left_injective ℝ hnne hsmul) hyx.symm

end FractalSync.TubularNeighborhood
