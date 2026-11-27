import FractalSync.OperationalMechanicalBounds
import FractalSync.ControlForceBounds
import FractalSync.PotentialForceBounds

/-!
# Regional constants for the operational potential and control catalog

The constants in this module bound the actual ruler, its actual derivatives,
and the actual catalog forces. All coefficient and geometric assumptions are
stated explicitly; the concrete finite fixture discharges them separately.
-/

noncomputable section

namespace FractalSync.OperationalDynamics

open Set Metric OperationalField FlatShoreIntrinsic ScaleMetric MechanicalBounds
open ControlMechanics PotentialForces PotentialForceBounds
open scoped NNReal

local instance operationalCatalogPhaseNormedSpace : NormedSpace ℝ (Plane × Plane) :=
  Prod.normedSpace

/-- Simple rational majorants for a coefficient bound of seven and spacing
at least one tenth. They are intentionally valid without cancellation. -/
theorem rational_ruler_budget (h : ℝ) (hh : (1 : ℝ) / 10 ≤ h) :
    ruler 1 7 ≤ 8 ∧ splineGradientBound 7 h ≤ 200 ∧
      rulerHessianBound 7 h 1 ≤ 50000 ∧ rulerThirdBound 7 h 1 ≤ 30000000 := by
  have hpos : 0 < h := lt_of_lt_of_le (by norm_num) hh
  have hG : splineGradientBound 7 h ≤ 200 := by
    unfold splineGradientBound
    exact (div_le_iff₀ hpos).mpr (by linarith)
  have hH : splineHessianBound 7 h ≤ 6000 := by
    exact (div_le_iff₀ (sq_pos_of_pos hpos)).mpr (by nlinarith [sq_nonneg (h - 1 / 10)])
  have hT : splineThirdBound 7 h ≤ 250000 := by
    exact (show splineThirdBound 7 h ≤ 32 * 7 / ((1 : ℝ) / 10) ^ 3 from
      div_le_div_of_nonneg_left (by norm_num) (by norm_num)
        (pow_le_pow_left₀ (by norm_num) hh 3)).trans (by norm_num)
  have hG0 : 0 ≤ splineGradientBound 7 h := by
    unfold splineGradientBound
    positivity
  refine ⟨?_, hG, ?_, ?_⟩
  · norm_num [ruler, Real.sqrt_le_left]
  · dsimp only [rulerHessianBound, div_one]
    nlinarith [pow_le_pow_left₀ hG0 hG 2]
  · dsimp only [rulerThirdBound]
    nlinarith [pow_le_pow_left₀ hG0 hG 3,
      mul_le_mul_of_nonneg_left hH hG0]

/-- The actual conservative force, fixed-direction normalized drive, and
viscous covector at one phase state. All five parameters remain explicit. -/
def catalogForce {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]
    (rho a kappa mu F beta : ℝ) (u x v e : E) : E :=
  totalForce kappa mu rho u x +
    driveForce rho a F u (metricNormalize rho a u e) + -(beta • metricAction rho a u v)

def catalogForceBudget (epsilon a kappa mu F beta : ℝ)
    (br bu bx bv : Budget) : Budget :=
  addBudget (addBudget
    (totalForceBudget epsilon kappa mu (wallCosFloor bx.1) br bu bx)
    (normalizedDriveForceBudget epsilon a br bu (|F|, 0)))
    (dragForceBudget epsilon a br bu (|beta|, 0) bv)

/-- The region's position radius discharges the wall denominator floor;
the nonzero fixed heading discharges normalization regularity. -/
theorem catalogForce_boundedLipschitz {E X : Type*}
    [NormedAddCommGroup E] [InnerProductSpace ℝ E] [PseudoMetricSpace X]
    (epsilon a kappa mu F beta : ℝ) (s : Set X)
    (rho : X → ℝ) (u x v : X → E) (e : E) (br bu bx bv : Budget)
    (hepsilon : 0 < epsilon) (he : e ≠ 0)
    (hr : BoundedLipschitzOn rho s br.1 br.2)
    (hu : BoundedLipschitzOn u s bu.1 bu.2)
    (hx : BoundedLipschitzOn x s bx.1 bx.2)
    (hv : BoundedLipschitzOn v s bv.1 bv.2)
    (hfloor : ∀ z ∈ s, epsilon ≤ rho z) (hradius : bx.1 < 2) :
    BoundedLipschitzOn
      (fun z => catalogForce (rho z) a kappa mu F beta (u z) (x z) (v z) e) s
      (catalogForceBudget epsilon a kappa mu F beta br bu bx bv).1
      (catalogForceBudget epsilon a kappa mu F beta br bu bx bv).2 := by
  have htotal := totalForce_bound epsilon kappa mu (wallCosFloor bx.1) rho u x s br bu bx
    hepsilon (wallCosFloor_pos bx.1 hx.bound_nonneg hradius) hr hu hx hfloor
    (fun z hz => wallCosFloor_le bx.1 (x z) hradius (hx.norm_le z hz))
  have hdrive := normalizedDriveForce_boundedLipschitz epsilon a s rho (fun _ => F) u e
    br bu (|F|, 0) hepsilon hr hu (const_real s F) hfloor he
  have hdrag := dragForce_boundedLipschitz epsilon a s rho (fun _ => beta) u v
    br bu (|beta|, 0) bv hepsilon hr hu (const_real s beta) hv hfloor
  simpa only [catalogForce, catalogForceBudget, addBudget] using (htotal.add hdrive).add hdrag

/-- A fully numerical force certificate for a nonzero Shore potential,
the default wall, unit drive, and unit viscous damping. -/
theorem catalogForce_rational_bound {E X : Type*}
    [NormedAddCommGroup E] [InnerProductSpace ℝ E] [PseudoMetricSpace X]
    (s : Set X) (rho : X → ℝ) (u x v : X → E) (e : E) (he : e ≠ 0)
    (hr : BoundedLipschitzOn rho s 8 200)
    (hu : BoundedLipschitzOn u s 200 50000)
    (hx : BoundedLipschitzOn x s 1 1)
    (hv : BoundedLipschitzOn v s 1 1)
    (hfloor : ∀ z ∈ s, 1 ≤ rho z) :
    BoundedLipschitzOn
      (fun z => catalogForce (rho z) 1 (Real.log 2) (1 / Real.pi) 1 1
        (u z) (x z) (v z) e) s 400000 4000000000000 := by
  have htotal := totalForce_log_two_default_unit_bound 1 rho u x s
    (8, 200) (200, 50000) 1 (by norm_num) hr hu hx hfloor
  have hOne : BoundedLipschitzOn (fun _ : X => (1 : ℝ)) s 1 0 := by
    simpa only [abs_one] using const_real s (1 : ℝ)
  have hdrive := normalizedDriveForce_boundedLipschitz 1 1 s rho (fun _ => 1) u e
    (8, 200) (200, 50000) (1, 0) (by norm_num) hr hu hOne hfloor he
  have hdrag := dragForce_boundedLipschitz 1 1 s rho (fun _ => 1) u v
    (8, 200) (200, 50000) (1, 0) (1, 1) (by norm_num) hr hu hOne hv hfloor
  apply ((htotal.add hdrive).add hdrag).enlarge
  all_goals norm_num [normalizedDriveForceBudget, normalizedHeadingBudget, driveForceBudget,
    dragForceBudget, metricActionBudget, addBudget, productBudget, inverseBudget]

/-- The literal spline and its actual first two ruler derivatives satisfy
the rational budgets used by the concrete catalog certificate. -/
theorem rational_operational_jets (C : Lattice) (o : Coords) (h : ℝ)
    (hh : (1 : ℝ) / 10 ≤ h) (hC : ∀ i j, |C i j| ≤ 7) :
    BoundedLipschitzOn (rulerField C o h 1) univ 8 200 ∧
      BoundedLipschitzOn (gradient (rulerField C o h 1)) univ 200 50000 ∧
      BoundedLipschitzOn (fderiv ℝ (gradient (rulerField C o h 1))) univ 50000 30000000 := by
  have hpos : 0 < h := lt_of_lt_of_le (by norm_num) hh
  obtain ⟨hr, hG, hH, hT⟩ := rational_ruler_budget h hh
  exact ⟨(rulerField_boundedLipschitz C o h 1 7 hpos (by norm_num) (by norm_num) hC).enlarge hr hG,
    (gradient_rulerField_boundedLipschitz C o h 1 7 hpos (by norm_num) (by norm_num) hC).enlarge hG hH,
    (hessian_rulerField_boundedLipschitz C o h 1 7 hpos (by norm_num) (by norm_num) hC).enlarge hH hT⟩

end FractalSync.OperationalDynamics
