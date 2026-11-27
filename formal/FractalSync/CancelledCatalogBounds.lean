import FractalSync.RankOneInverseBounds
import FractalSync.OperationalCatalogBounds

/-!
# Mechanical estimates after exact metric cancellations

These regional bounds apply to the actual catalog and connection. Metric
raising is simplified before estimating Shore, normalized drive, and drag.
-/

noncomputable section

namespace FractalSync.OperationalDynamics

open Set ScaleMetric MechanicalBounds ControlMechanics PotentialForces PotentialForceBounds
open scoped InnerProductSpace

variable {E X : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
  [PseudoMetricSpace X]

/-- Exact raising of the unit drive/drag catalog with the logarithm-normalized Shore force. -/
theorem raised_catalog_eq (rho a mu : ℝ) (u x v e : E) (hrho : rho ≠ 0) :
    inverseAction rho a u (catalogForce rho a (Real.log 2) mu 1 1 u x v e) =
      (rho / weight a u) • u + inverseAction rho a u (wallForce mu x) +
        metricNormalize rho a u e - v := by
  simp only [catalogForce, totalForce, inverseAction_add, drive_acceleration rho a 1 u _ hrho,
    shore_acceleration (Real.log 2) rho a u hrho, one_smul]
  rw [mul_div_mul_left _ _ (Real.log_ne_zero_of_pos_of_ne_one (by norm_num) (by norm_num))]
  rw [← neg_one_smul ℝ (metricAction rho a u v), inverseAction_smul,
    inverse_metricAction rho a u v hrho, neg_one_smul, sub_eq_add_neg]

/-- The existing connection budget, without an accompanying force term. -/
theorem connection_boundedLipschitz (epsilon a : ℝ) (s : Set X)
    (rho : X → ℝ) (u : X → E) (H : X → E →L[ℝ] E) (v : X → E)
    (br bu bH bv : Budget) (he : 0 < epsilon)
    (hr : BoundedLipschitzOn rho s br.1 br.2)
    (hu : BoundedLipschitzOn u s bu.1 bu.2)
    (hH : BoundedLipschitzOn H s bH.1 bH.2)
    (hv : BoundedLipschitzOn v s bv.1 bv.2)
    (hfloor : ∀ z ∈ s, epsilon ≤ rho z) :
    BoundedLipschitzOn (fun z => connection (rho z) a (u z) (H z) (v z) (v z)) s
      (connectionBudget epsilon a br bu bH bv).1
      (connectionBudget epsilon a br bu bH bv).2 := by
  have hzero : BoundedLipschitzOn (fun _ : X => (0 : E)) s 0 0 := by
    simpa only [norm_zero] using BoundedLipschitzOn.const s (0 : E)
  have ha := MechanicalBounds.acceleration_bound epsilon a s rho u H (fun _ => 0) v
    br bu bH (0, 0) bv he hr hu hH hzero hv hfloor
  simpa [accelerationBudget, raisedForceBudget, addBudget, productBudget, inverseBudget,
    inverseAction, unstretch] using (BoundedLipschitzOn.const s (-1 : ℝ)).smul ha

/-- Budget after raising the logarithm-normalized Shore force, default wall,
unit normalized drive, and unit drag. The position norm is at most one. -/
def cancelledRaisedCatalogBudget (a : ℝ) (br bu bv : Budget) (Lx : ℝ) : Budget :=
  let W := addBudget (1, 0) (productBudget (a ^ 2, 0) (productBudget bu bu))
  let shore := productBudget (productBudget br (inverseBudget 1 W)) bu
  let wall : Budget := (br.1 ^ 2 * (1 / 16), br.1 ^ 2 * (Lx / 2) +
    (2 * br.1 * br.2 + 2 * br.1 ^ 2 * a ^ 2 * bu.1 * bu.2) * (1 / 16))
  addBudget (addBudget (addBudget shore wall) (normalizedHeadingBudget a br bu)) bv

/-- The cancelled catalog budget controls the actual raised force. -/
theorem raised_catalog_boundedLipschitz (a : ℝ) (s : Set X)
    (rho : X → ℝ) (u x v : X → E) (e : E) (br bu bv : Budget) (Lx : ℝ)
    (he : e ≠ 0) (hr : BoundedLipschitzOn rho s br.1 br.2)
    (hu : BoundedLipschitzOn u s bu.1 bu.2)
    (hx : BoundedLipschitzOn x s 1 Lx)
    (hv : BoundedLipschitzOn v s bv.1 bv.2)
    (hpos : ∀ z ∈ s, 0 < rho z) :
    BoundedLipschitzOn (fun z => inverseAction (rho z) a (u z)
      (catalogForce (rho z) a (Real.log 2) (1 / Real.pi) 1 1 (u z) (x z) (v z) e)) s
      (cancelledRaisedCatalogBudget a br bu bv Lx).1
      (cancelledRaisedCatalogBudget a br bu bv Lx).2 := by
  have hA : BoundedLipschitzOn (fun _ : X => a ^ 2) s (a ^ 2) 0 := by
    exact ⟨sq_nonneg a, le_rfl, fun _ _ => (Real.norm_of_nonneg (sq_nonneg a)).le,
      fun _ _ _ _ => by simp⟩
  have hOne : BoundedLipschitzOn (fun _ : X => (1 : ℝ)) s 1 0 := by
    exact ⟨zero_le_one, le_rfl, fun _ _ => (norm_one : ‖(1 : ℝ)‖ = 1).le,
      fun _ _ _ _ => by simp⟩
  have hW := hOne.add (hA.mul (hu.inner hu))
  have hWi := hW.inv zero_lt_one (fun z (_ : z ∈ s) =>
    le_add_of_nonneg_right (mul_nonneg (sq_nonneg a) (real_inner_self_nonneg)))
  have hshore := (hr.mul hWi).smul hu
  have hwall := inverseAction_boundedLipschitz a s rho u (fun z => wallForce (1 / Real.pi) (x z))
    br.1 br.2 bu.1 bu.2 (1 / 16) (Lx / 2) hr hu (wallForce_default_unit_bound x s Lx hx)
  have hd := metricNormalize_boundedLipschitz a s rho u e br bu hr hu hpos he
  have h := ((hshore.add hwall).add hd).sub hv
  refine ⟨h.bound_nonneg, h.lip_nonneg, ?_, ?_⟩
  · intro z hz
    rw [raised_catalog_eq (rho z) a (1 / Real.pi) _ _ _ _ (hpos z hz).ne']
    simpa only [weight, div_eq_mul_inv] using h.norm_le z hz
  · intro z hz w hw
    rw [raised_catalog_eq (rho z) a (1 / Real.pi) _ _ _ _ (hpos z hz).ne',
      raised_catalog_eq (rho w) a (1 / Real.pi) _ _ _ _ (hpos w hw).ne']
    simpa only [weight, div_eq_mul_inv] using h.norm_sub_le z hz w hw

/-- Combined acceleration budget after exact force cancellations. -/
def cancelledAccelerationBudget (epsilon a : ℝ) (br bu bH bv : Budget) (Lx : ℝ) : Budget :=
  addBudget (cancelledRaisedCatalogBudget a br bu bv Lx)
    (connectionBudget epsilon a br bu bH bv)

/-- Full actual acceleration with the cancellation-preserving regional budget. -/
theorem cancelled_catalog_acceleration_bound (epsilon a : ℝ) (s : Set X)
    (rho : X → ℝ) (u : X → E) (H : X → E →L[ℝ] E) (x v : X → E) (e : E)
    (br bu bH bv : Budget) (Lx : ℝ) (hepsilon : 0 < epsilon) (he : e ≠ 0)
    (hr : BoundedLipschitzOn rho s br.1 br.2)
    (hu : BoundedLipschitzOn u s bu.1 bu.2)
    (hH : BoundedLipschitzOn H s bH.1 bH.2)
    (hx : BoundedLipschitzOn x s 1 Lx)
    (hv : BoundedLipschitzOn v s bv.1 bv.2)
    (hfloor : ∀ z ∈ s, epsilon ≤ rho z) :
    BoundedLipschitzOn (fun z => inverseAction (rho z) a (u z)
      (catalogForce (rho z) a (Real.log 2) (1 / Real.pi) 1 1 (u z) (x z) (v z) e) -
      connection (rho z) a (u z) (H z) (v z) (v z)) s
      (cancelledAccelerationBudget epsilon a br bu bH bv Lx).1
      (cancelledAccelerationBudget epsilon a br bu bH bv Lx).2 := by
  exact (raised_catalog_boundedLipschitz a s rho u x v e br bu bv Lx he hr hu hx hv
    (fun z hz => hepsilon.trans_le (hfloor z hz))).sub
    (connection_boundedLipschitz epsilon a s rho u H v br bu bH bv hepsilon hr hu hH hv hfloor)

/-- Exact arithmetic for the local quadratic-fixture input budgets. -/
theorem cancelled_local_budget :
    cancelledRaisedCatalogBudget 1 (2, 1) (1, 20) (1 / 3, 1) 1 = (55 / 12, 701 / 4) ∧
    connectionBudget 1 1 (2, 1) (1, 20) (20, 200) (1 / 3, 1) = (8 / 3, 1748 / 9) ∧
    cancelledAccelerationBudget 1 1 (2, 1) (1, 20) (20, 200) (1 / 3, 1) 1 =
      (29 / 4, 13301 / 36) := by
  norm_num [cancelledRaisedCatalogBudget, cancelledAccelerationBudget, connectionBudget,
    normalizedHeadingBudget, addBudget, productBudget, inverseBudget]

/-- A norm-eight, Lipschitz-four-hundred certificate for the full actual acceleration. -/
theorem cancelled_catalog_acceleration_local_bound (s : Set X)
    (rho : X → ℝ) (u : X → E) (H : X → E →L[ℝ] E) (x v : X → E) (e : E)
    (he : e ≠ 0) (hr : BoundedLipschitzOn rho s 2 1)
    (hu : BoundedLipschitzOn u s 1 20) (hH : BoundedLipschitzOn H s 20 200)
    (hx : BoundedLipschitzOn x s 1 1) (hv : BoundedLipschitzOn v s (1 / 3) 1)
    (hfloor : ∀ z ∈ s, 1 ≤ rho z) :
    BoundedLipschitzOn (fun z => inverseAction (rho z) 1 (u z)
      (catalogForce (rho z) 1 (Real.log 2) (1 / Real.pi) 1 1 (u z) (x z) (v z) e) -
      connection (rho z) 1 (u z) (H z) (v z) (v z)) s 8 400 := by
  have h := cancelled_catalog_acceleration_bound 1 1 s rho u H x v e
    (2, 1) (1, 20) (20, 200) (1 / 3, 1) 1 (by norm_num) he hr hu hH hx hv hfloor
  exact h.enlarge (by rw [cancelled_local_budget.2.2]; norm_num)
    (by rw [cancelled_local_budget.2.2]; norm_num)

/-- The corresponding actual unraised catalog force also has small explicit constants. -/
theorem catalogForce_local_bound (s : Set X) (rho : X → ℝ) (u x v : X → E)
    (e : E) (he : e ≠ 0) (hr : BoundedLipschitzOn rho s 2 1)
    (hu : BoundedLipschitzOn u s 1 20) (hx : BoundedLipschitzOn x s 1 1)
    (hv : BoundedLipschitzOn v s (1 / 3) 1) (hfloor : ∀ z ∈ s, 1 ≤ rho z) :
    BoundedLipschitzOn (fun z => catalogForce (rho z) 1 (Real.log 2) (1 / Real.pi) 1 1
      (u z) (x z) (v z) e) s 6 210 := by
  have htotal := totalForce_log_two_default_unit_bound 1 rho u x s
    (2, 1) (1, 20) 1 (by norm_num) hr hu hx hfloor
  have hOne : BoundedLipschitzOn (fun _ : X => (1 : ℝ)) s 1 0 := by
    simpa only [abs_one] using const_real s (1 : ℝ)
  have hdrive := normalizedDriveForce_boundedLipschitz 1 1 s rho (fun _ => 1) u e
    (2, 1) (1, 20) (1, 0) (by norm_num) hr hu hOne hfloor he
  have hdrag := dragForce_boundedLipschitz 1 1 s rho (fun _ => 1) u v
    (2, 1) (1, 20) (1, 0) (1 / 3, 1) (by norm_num) hr hu hOne hv hfloor
  apply ((htotal.add hdrive).add hdrag).enlarge
  all_goals norm_num [normalizedDriveForceBudget, normalizedHeadingBudget, driveForceBudget,
    dragForceBudget, metricActionBudget, addBudget, productBudget, inverseBudget]

end FractalSync.OperationalDynamics
