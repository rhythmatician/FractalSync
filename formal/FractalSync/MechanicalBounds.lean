import FractalSync.BoundedLipschitz

/-!
# Explicit coefficient bounds for mechanical acceleration

Each pair below consists of a norm bound and a Lipschitz constant. The
arithmetic expressions propagate those bounds through the existing exact
inverse and connection, without differentiating a separately supplied formula.
-/

noncomputable section

namespace FractalSync.ScaleMetric.MechanicalBounds

open Set
open scoped RealInnerProductSpace

abbrev Budget := ℝ × ℝ

def addBudget (p q : Budget) : Budget := (p.1 + q.1, p.2 + q.2)

def productBudget (p q : Budget) : Budget :=
  (p.1 * q.1, p.1 * q.2 + q.1 * p.2)

def inverseBudget (epsilon : ℝ) (p : Budget) : Budget :=
  (epsilon⁻¹, p.2 / epsilon ^ 2)

/-- Bounds for the actual inverse-metric force term. -/
def raisedForceBudget (a : ℝ) (rho u force : Budget) : Budget :=
  let A : Budget := (a ^ 2, 0)
  let W := addBudget (1, 0) (productBudget A (productBudget u u))
  let Wi := inverseBudget 1 W
  productBudget (productBudget rho rho)
    (addBudget force (productBudget (productBudget (productBudget A (productBudget u force)) Wi) u))

/-- Bounds for the actual connection evaluated twice on velocity. -/
def connectionBudget (epsilon a : ℝ) (rho u H velocity : Budget) : Budget :=
  let A : Budget := (a ^ 2, 0)
  let W := addBudget (1, 0) (productBudget A (productBudget u u))
  let Wi := inverseBudget 1 W
  let ri := inverseBudget epsilon rho
  let uv := productBudget u velocity
  let Hvv := productBudget (productBudget H velocity) velocity
  let first := productBudget (productBudget (productBudget A Hvv) Wi) u
  let second := productBudget (productBudget uv ri) velocity
  let numerator := addBudget (productBudget velocity velocity)
    (productBudget (productBudget A uv) uv)
  let last := productBudget (productBudget numerator (productBudget ri Wi)) u
  addBudget (addBudget (addBudget first second) second) last

/-- The two entries are an explicit acceleration norm bound and Lipschitz constant. -/
def accelerationBudget (epsilon a : ℝ) (rho u H force velocity : Budget) : Budget :=
  addBudget (raisedForceBudget a rho u force) (connectionBudget epsilon a rho u H velocity)

variable {E X : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E] [PseudoMetricSpace X]

/-- The actual mechanical expression obeys the constants computed above. -/
theorem acceleration_bound (epsilon a : ℝ) (s : Set X)
    (rho : X → ℝ) (u : X → E) (H : X → E →L[ℝ] E) (force velocity : X → E)
    (br bu bH bQ bv : Budget) (he : 0 < epsilon)
    (hr : BoundedLipschitzOn rho s br.1 br.2)
    (hu : BoundedLipschitzOn u s bu.1 bu.2)
    (hH : BoundedLipschitzOn H s bH.1 bH.2)
    (hQ : BoundedLipschitzOn force s bQ.1 bQ.2)
    (hv : BoundedLipschitzOn velocity s bv.1 bv.2)
    (hfloor : ∀ x ∈ s, epsilon ≤ rho x) :
    BoundedLipschitzOn
      (fun x => inverseAction (rho x) a (u x) (force x) -
        connection (rho x) a (u x) (H x) (velocity x) (velocity x)) s
      (accelerationBudget epsilon a br bu bH bQ bv).1
      (accelerationBudget epsilon a br bu bH bQ bv).2 := by
  have hA : BoundedLipschitzOn (fun _ : X => a ^ 2) s (a ^ 2) 0 := by
    exact ⟨sq_nonneg a, le_rfl, fun _ _ => (Real.norm_of_nonneg (sq_nonneg a)).le,
      fun _ _ _ _ => by simp⟩
  have hOne : BoundedLipschitzOn (fun _ : X => (1 : ℝ)) s 1 0 := by
    exact ⟨zero_le_one, le_rfl, fun _ _ => (norm_one : ‖(1 : ℝ)‖ = 1).le,
      fun _ _ _ _ => by simp⟩
  have hW := hOne.add (hA.mul (hu.inner hu))
  have hWi := hW.inv zero_lt_one (fun x (_ : x ∈ s) =>
    le_add_of_nonneg_right (mul_nonneg (sq_nonneg a) (real_inner_self_nonneg)))
  have hri := hr.inv he hfloor
  have hRaised := (hr.mul hr).smul (hQ.sub (((hA.mul (hu.inner hQ)).mul hWi).smul hu))
  have hFirst := ((hA.mul ((hH.clm_apply hv).inner hv)).mul hWi).smul hu
  have hSecond := ((hu.inner hv).mul hri).smul hv
  have hNum := (hv.inner hv).add ((hA.mul (hu.inner hv)).mul (hu.inner hv))
  have hLast := (hNum.mul (hri.mul hWi)).smul hu
  have hConn := ((hFirst.sub hSecond).sub hSecond).add hLast
  simpa only [accelerationBudget, raisedForceBudget, connectionBudget, addBudget,
    productBudget, inverseBudget, inverseAction, unstretch, connection, weight,
    div_eq_mul_inv, pow_two, mul_inv_rev, mul_assoc, mul_left_comm, mul_comm] using
      hRaised.sub hConn

end FractalSync.ScaleMetric.MechanicalBounds
