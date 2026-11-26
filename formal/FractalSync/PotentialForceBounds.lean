import FractalSync.MechanicalBounds
import FractalSync.PotentialForces
import Mathlib.Analysis.Real.Pi.Bounds

/-!
# Explicit regional bounds for the actual conservative force catalog

The Shore and secant-wall covectors retain their existing definitions. Their
norm and pairwise Lipschitz bounds are explicit arithmetic expressions, with
positive denominator floors derived from the physical domain where stated.
-/

noncomputable section

namespace FractalSync.PotentialForceBounds

open Set ScaleMetric.MechanicalBounds PotentialForces
open scoped RealInnerProductSpace

def shoreForceBudget (epsilon kappa : ℝ) (br bu : Budget) : Budget :=
  productBudget (productBudget (|kappa / Real.log 2|, 0) (inverseBudget epsilon br)) bu

def wallCoordinateBudget (bx : Budget) : Budget :=
  productBudget (1 / 4, 0) (productBudget bx bx)

def wallPhaseBudget (bx : Budget) : Budget :=
  let bs := wallCoordinateBudget bx
  let bs2 := productBudget bs bs
  productBudget (Real.pi / 2, 0) (productBudget bs2 bs2)

def wallForceBudget (mu c : ℝ) (bx : Budget) : Budget :=
  let bs := wallCoordinateBudget bx
  let bs3 := productBudget (productBudget bs bs) bs
  let bt : Budget := (1, (wallPhaseBudget bx).2)
  let bi := inverseBudget c bt
  productBudget (productBudget (productBudget
    (productBudget (productBudget (|mu * Real.pi|, 0) bs3) bt) bi) bi) bx

def totalForceBudget (epsilon kappa mu c : ℝ) (br bu bx : Budget) : Budget :=
  addBudget (shoreForceBudget epsilon kappa br bu) (wallForceBudget mu c bx)

variable {X : Type*} [PseudoMetricSpace X]

theorem const_real (s : Set X) (c : ℝ) :
    BoundedLipschitzOn (fun _ : X => c) s |c| 0 := by
  simpa only [Real.norm_eq_abs] using BoundedLipschitzOn.const s c

theorem sin_bound {f : X → ℝ} {s : Set X} {M L : ℝ}
    (hf : BoundedLipschitzOn f s M L) :
    BoundedLipschitzOn (fun x => Real.sin (f x)) s 1 L := by
  refine ⟨zero_le_one, hf.lip_nonneg, fun x _ => Real.abs_sin_le_one (f x), ?_⟩
  exact fun x hx y hy => (show ‖Real.sin (f x) - Real.sin (f y)‖ ≤ ‖f x - f y‖ from
    by simpa only [one_mul] using (Convex.norm_image_sub_le_of_norm_hasDerivWithin_le
      (fun t (_ : t ∈ (univ : Set ℝ)) => (Real.hasDerivAt_sin t).hasDerivWithinAt)
      (fun t _ => Real.abs_cos_le_one t) convex_univ (mem_univ (f y)) (mem_univ (f x)))).trans
    (hf.norm_sub_le x hx y hy)

theorem cos_bound {f : X → ℝ} {s : Set X} {M L : ℝ}
    (hf : BoundedLipschitzOn f s M L) :
    BoundedLipschitzOn (fun x => Real.cos (f x)) s 1 L := by
  exact ⟨zero_le_one, hf.lip_nonneg, fun x _ => Real.abs_cos_le_one (f x),
    fun x hx y hy => (show ‖Real.cos (f x) - Real.cos (f y)‖ ≤ ‖f x - f y‖ from
      by simpa only [one_mul] using (Convex.norm_image_sub_le_of_norm_hasDerivWithin_le (C := 1)
        (fun t (_ : t ∈ (univ : Set ℝ)) => (Real.hasDerivAt_cos t).hasDerivWithinAt)
        (fun t _ => by simpa only [norm_neg, Real.norm_eq_abs] using Real.abs_sin_le_one t)
        convex_univ (mem_univ (f y)) (mem_univ (f x)))).trans (hf.norm_sub_le x hx y hy)⟩

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- Actual Shore force from regional ruler and field-gradient certificates. -/
theorem shoreForce_bound (epsilon kappa : ℝ) (rho : X → ℝ) (u : X → E)
    (s : Set X) (br bu : Budget) (he : 0 < epsilon)
    (hr : BoundedLipschitzOn rho s br.1 br.2)
    (hu : BoundedLipschitzOn u s bu.1 bu.2)
    (hfloor : ∀ x ∈ s, epsilon ≤ rho x) :
    BoundedLipschitzOn (fun x => shoreForce kappa (rho x) (u x)) s
      (shoreForceBudget epsilon kappa br bu).1
      (shoreForceBudget epsilon kappa br bu).2 := by
  have hi := hr.inv he hfloor
  simpa only [shoreForceBudget, productBudget, inverseBudget, shoreForce,
    div_eq_mul_inv, mul_inv_rev, mul_assoc, mul_left_comm, mul_comm] using
    ((const_real s (kappa / Real.log 2)).mul hi).smul hu

/-- Choosing kappa = log 2 removes the Shore logarithm from the budget. -/
theorem shoreForce_log_two_bound (epsilon : ℝ) (rho : X → ℝ) (u : X → E)
    (s : Set X) (br bu : Budget) (he : 0 < epsilon)
    (hr : BoundedLipschitzOn rho s br.1 br.2)
    (hu : BoundedLipschitzOn u s bu.1 bu.2)
    (hfloor : ∀ x ∈ s, epsilon ≤ rho x) :
    BoundedLipschitzOn (fun x => shoreForce (Real.log 2) (rho x) (u x)) s
      (bu.1 / epsilon) (bu.2 / epsilon + bu.1 * br.2 / epsilon ^ 2) := by
  have hlog : Real.log 2 ≠ 0 := ne_of_gt (Real.log_pos (by norm_num))
  simpa [shoreForceBudget, productBudget, inverseBudget, hlog,
    div_eq_mul_inv, mul_assoc, mul_left_comm, mul_comm] using
    shoreForce_bound epsilon (Real.log 2) rho u s br bu he hr hu hfloor

theorem wallCoordinate_bound (x : X → E) (s : Set X) (bx : Budget)
    (hx : BoundedLipschitzOn x s bx.1 bx.2) :
    BoundedLipschitzOn (fun z => wallCoordinate (x z)) s
      (wallCoordinateBudget bx).1 (wallCoordinateBudget bx).2 := by
  simpa [wallCoordinateBudget, productBudget, wallCoordinate, real_inner_self_eq_norm_sq,
    div_eq_mul_inv, mul_comm] using (const_real s (1 / 4)).mul (hx.inner hx)

theorem wallPhase_bound (x : X → E) (s : Set X) (bx : Budget)
    (hx : BoundedLipschitzOn x s bx.1 bx.2) :
    BoundedLipschitzOn (fun z => wallPhase (x z)) s
      (wallPhaseBudget bx).1 (wallPhaseBudget bx).2 := by
  have hs := wallCoordinate_bound x s bx hx
  simpa only [wallPhaseBudget, productBudget, wallPhase, abs_of_pos (half_pos Real.pi_pos),
    pow_succ, pow_zero, mul_one, one_mul, mul_assoc] using
    (const_real s (Real.pi / 2)).mul ((hs.mul hs).mul (hs.mul hs))

/-- Explicit cosine denominator floor at a regional radius. -/
def wallCosFloor (r : ℝ) : ℝ :=
  Real.cos (Real.pi / 2 * (r ^ 2 / 4) ^ 4)

omit [InnerProductSpace ℝ E] in
theorem wallPhase_le_radius (r : ℝ) (x : E) (hx : ‖x‖ ≤ r) :
    wallPhase x ≤ Real.pi / 2 * (r ^ 2 / 4) ^ 4 := by
  apply mul_le_mul_of_nonneg_left _ (half_pos Real.pi_pos).le
  exact pow_le_pow_left₀ (wallCoordinate_nonneg x)
    (div_le_div_of_nonneg_right (pow_le_pow_left₀ (norm_nonneg x) hx 2) (by norm_num)) 4

theorem wallCosFloor_pos (r : ℝ) (hr0 : 0 ≤ r) (hr2 : r < 2) :
    0 < wallCosFloor r := by
  simpa only [wallCosFloor, wallPhase, wallCoordinate, Real.norm_eq_abs,
    abs_of_nonneg hr0] using cos_wallPhase_pos r
      (by simpa only [Real.norm_eq_abs, abs_of_nonneg hr0] using hr2)

omit [InnerProductSpace ℝ E] in
theorem wallCosFloor_le (r : ℝ) (x : E) (hr2 : r < 2) (hx : ‖x‖ ≤ r) :
    wallCosFloor r ≤ Real.cos (wallPhase x) := by
  have hr0 := (norm_nonneg x).trans hx
  have hphase : Real.pi / 2 * (r ^ 2 / 4) ^ 4 < Real.pi / 2 := by
    simpa only [wallPhase, wallCoordinate, Real.norm_eq_abs,
      abs_of_nonneg hr0] using
      (wallPhase_mem r (by simpa only [Real.norm_eq_abs, abs_of_nonneg hr0] using hr2)).2
  exact Real.cos_le_cos_of_nonneg_of_le_pi
    (wallPhase_mem x (hx.trans_lt hr2)).1 (by linarith [Real.pi_pos])
    (wallPhase_le_radius r x hx)

omit [InnerProductSpace ℝ E] in
/-- A rational denominator floor on the unit position ball. -/
theorem half_le_cos_wallPhase (x : E) (hx : ‖x‖ ≤ 1) :
    (1 : ℝ) / 2 ≤ Real.cos (wallPhase x) := by
  have hp := wallPhase_le_radius 1 x hx
  simpa only [Real.cos_pi_div_three] using
    Real.cos_le_cos_of_nonneg_of_le_pi
      (wallPhase_mem x (hx.trans_lt (by norm_num))).1
      (by linarith [Real.pi_pos] : Real.pi / 3 ≤ Real.pi)
      (by norm_num at hp; linarith [Real.pi_pos] : wallPhase x ≤ Real.pi / 3)

/-- Positive cosine-floor evidence gives explicit bounds for the actual
secant-wall force, including every pair of points in the supplied region. -/
theorem wallForce_bound (mu c : ℝ) (x : X → E) (s : Set X) (bx : Budget)
    (hc : 0 < c) (hx : BoundedLipschitzOn x s bx.1 bx.2)
    (hfloor : ∀ z ∈ s, c ≤ Real.cos (wallPhase (x z))) :
    BoundedLipschitzOn (fun z => wallForce mu (x z)) s
      (wallForceBudget mu c bx).1 (wallForceBudget mu c bx).2 := by
  have hs := wallCoordinate_bound x s bx hx
  have hp := wallPhase_bound x s bx hx
  have hi := (cos_bound hp).inv hc hfloor
  have hresult := (((((const_real s (-(mu * Real.pi))).mul ((hs.mul hs).mul hs)).mul
    (sin_bound hp)).mul hi).mul hi).smul hx
  simpa only [wallForceBudget, productBudget, inverseBudget, wallForce, wallCoefficient,
    Real.tan_eq_sin_div_cos, div_eq_mul_inv, pow_succ, pow_zero, one_mul, mul_one,
    abs_neg, neg_mul, mul_neg, mul_assoc, mul_left_comm, mul_comm] using hresult

/-- The position radius alone discharges all wall denominator hypotheses. -/
theorem wallForce_bound_of_radius (mu : ℝ) (x : X → E) (s : Set X) (bx : Budget)
    (hx : BoundedLipschitzOn x s bx.1 bx.2) (hr : bx.1 < 2) :
    BoundedLipschitzOn (fun z => wallForce mu (x z)) s
      (wallForceBudget mu (wallCosFloor bx.1) bx).1
      (wallForceBudget mu (wallCosFloor bx.1) bx).2 := by
  exact wallForce_bound mu (wallCosFloor bx.1) x s bx
    (wallCosFloor_pos bx.1 hx.bound_nonneg hr) hx
    (fun z hz => wallCosFloor_le bx.1 (x z) hr (hx.norm_le z hz))

/-- At the actual default wall parameter, the unit ball has rational force
bounds. The Lipschitz constant scales with the position-field certificate. -/
theorem wallForce_default_unit_bound (x : X → E) (s : Set X) (L : ℝ)
    (hx : BoundedLipschitzOn x s 1 L) :
    BoundedLipschitzOn (fun z => wallForce (1 / Real.pi) (x z)) s (1 / 16) (L / 2) := by
  have h := wallForce_bound (1 / Real.pi) (1 / 2) x s (1, L)
    (by norm_num) hx (fun z hz => half_le_cos_wallPhase (x z) (hx.norm_le z hz))
  apply h.enlarge
  all_goals norm_num [wallForceBudget, wallCoordinateBudget, wallPhaseBudget,
    productBudget, inverseBudget, Real.pi_ne_zero]
  nlinarith [hx.lip_nonneg, mul_nonneg (sub_nonneg.mpr Real.pi_lt_four.le) hx.lip_nonneg]

/-- The complete conservative catalog inherits explicit force and difference
budgets by addition of the actual Shore and wall force covectors. -/
theorem totalForce_bound (epsilon kappa mu c : ℝ) (rho : X → ℝ)
    (u x : X → E) (s : Set X) (br bu bx : Budget) (he : 0 < epsilon) (hc : 0 < c)
    (hr : BoundedLipschitzOn rho s br.1 br.2)
    (hu : BoundedLipschitzOn u s bu.1 bu.2)
    (hx : BoundedLipschitzOn x s bx.1 bx.2)
    (hfloor : ∀ z ∈ s, epsilon ≤ rho z)
    (hcos : ∀ z ∈ s, c ≤ Real.cos (wallPhase (x z))) :
    BoundedLipschitzOn (fun z => totalForce kappa mu (rho z) (u z) (x z)) s
      (totalForceBudget epsilon kappa mu c br bu bx).1
      (totalForceBudget epsilon kappa mu c br bu bx).2 := by
  simpa only [totalForceBudget, addBudget, totalForce] using
    (shoreForce_bound epsilon kappa rho u s br bu he hr hu hfloor).add
      (wallForce_bound mu c x s bx hc hx hcos)

/-- Fully rationalizable conservative bounds for kappa = log 2, the actual
wall default mu = 1/pi, and positions in the unit ball. -/
theorem totalForce_log_two_default_unit_bound (epsilon : ℝ) (rho : X → ℝ)
    (u x : X → E) (s : Set X) (br bu : Budget) (L : ℝ) (he : 0 < epsilon)
    (hr : BoundedLipschitzOn rho s br.1 br.2)
    (hu : BoundedLipschitzOn u s bu.1 bu.2)
    (hx : BoundedLipschitzOn x s 1 L)
    (hfloor : ∀ z ∈ s, epsilon ≤ rho z) :
    BoundedLipschitzOn
      (fun z => totalForce (Real.log 2) (1 / Real.pi) (rho z) (u z) (x z)) s
      (bu.1 / epsilon + 1 / 16)
      (bu.2 / epsilon + bu.1 * br.2 / epsilon ^ 2 + L / 2) := by
  exact (shoreForce_log_two_bound epsilon rho u s br bu he hr hu hfloor).add
    (wallForce_default_unit_bound x s L hx)

end FractalSync.PotentialForceBounds
