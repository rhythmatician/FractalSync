import FractalSync.OperationalHalo
import FractalSync.QuantitativeDynamics
import FractalSync.OperationalMechanicalBounds

/-!
# Quantitative trajectories for an actual finite operational asset

The actual finite-asset acceleration bounds supply a constructed reference
solution, explicit horizon and numerical tube. Interpolant errors and displaced
query addresses are certified throughout that interval over exact reals.
-/

noncomputable section

namespace FractalSync.OperationalDynamics

open OperationalField OperationalHalo FlatShoreIntrinsic ScaleMetric Set Metric Filter
open scoped Topology NNReal

local instance operationalTrajectoryPhaseNormedSpace : NormedSpace ℝ (Plane × Plane) :=
  Prod.normedSpace

/-- Phase speed from a concrete acceleration budget (norm bound, Lipschitz constant). -/
def phaseSpeedBound (p : Plane × Plane) (R : ℝ) (bA : MechanicalBounds.Budget) : ℝ :=
  max (‖p.2‖ + R) bA.1

/-- The explicit time horizon associated with the actual regional acceleration budget. -/
def trajectoryHorizon (p : Plane × Plane) (R : ℝ) (bA : MechanicalBounds.Budget) : ℝ :=
  R / (4 * (phaseSpeedBound p R bA + 1))

/-- An actual reported query lies in the open halo and all 36 mathematical
axis and row-major stencil addresses lie inside the finite asset. -/
def SafeReportedQuery (Nx Ny : ℕ) (o : Coords) (h : ℝ) (q : Coords) : Prop :=
  q ∈ openPhysicalHalo Nx Ny o h ∧
    let n := normalizedCoordinates o h q
    ∀ i j : Fin 6,
      (0 ≤ ⌊n.1⌋ + i.val - 2 ∧ ⌊n.1⌋ + i.val - 2 < (Nx : ℤ)) ∧
      (0 ≤ ⌊n.2⌋ + j.val - 2 ∧ ⌊n.2⌋ + j.val - 2 < (Ny : ℤ)) ∧
      (0 ≤ (⌊n.2⌋ + j.val - 2) * Nx + (⌊n.1⌋ + i.val - 2) ∧
        (⌊n.2⌋ + j.val - 2) * Nx + (⌊n.1⌋ + i.val - 2) < (Nx : ℤ) * Ny)

theorem safeReportedQuery_of_phaseBall (Nx Ny : ℕ) (o q : Coords)
    (p z : Plane × Plane) (h R delta : ℝ) (hh : 0 < h)
    (hz : z ∈ closedBall p R) (hquery : ‖q - chart z.1‖ ≤ delta)
    (hmargin : R + delta < haloMargin Nx Ny o h (chart p.1)) :
    SafeReportedQuery Nx Ny o h q := by
  have hq := reportedQuery_mem_of_phaseBall Nx Ny o q p z h R delta hh hz hquery hmargin
  exact ⟨hq, stencil_indices_of_mem_halo Nx Ny o q h
    ⟨⟨hq.1.1.le, hq.1.2⟩, ⟨hq.2.1.le, hq.2.2⟩⟩⟩

/-- Finite stored coefficients and a regional force certificate construct an
actual reference with explicit speed, Lipschitz, horizon, and tube constants. -/
theorem exists_finiteAsset_ODE_certificate (C : Lattice) (Nx Ny : ℕ) (o : Coords)
    (h epsilon a B : ℝ) (p : Plane × Plane) (R : ℝ) (force : Plane × Plane → Plane)
    (bQ : MechanicalBounds.Budget) (hh : 0 < h) (he : 0 < epsilon) (hB : 0 ≤ B) (hR : 0 < R)
    (hC : ∀ i j, 0 ≤ i → i < (Nx : ℤ) → 0 ≤ j → j < (Ny : ℤ) → |C i j| ≤ B)
    (hmargin : R < haloMargin Nx Ny o h (chart p.1))
    (hQ : BoundedLipschitzOn force (closedBall p R) bQ.1 bQ.2) :
    let bA := operationalAccelerationBudget epsilon a B h p R bQ
    let BV := phaseSpeedBound p R bA
    let T := trajectoryHorizon p R bA
    0 < T ∧ ∃ z : ℝ → Plane × Plane, z 0 = p ∧
      ∃ c : UniformODECertificate (acceleration C o h epsilon a force) z T (ball p R),
        (c.accelerationLip : ℝ) = bA.2 ∧ (c.phaseSpeed : ℝ) = BV ∧
        ∀ t ∈ Icc (0 : ℝ) T, ‖z t - p‖ ≤ BV * t ∧
          ∀ w : Plane × Plane, ‖w - z t‖ ≤ R / 2 → w ∈ ball p R := by
  have hacc := finiteAsset_acceleration_boundedLipschitz C Nx Ny o h epsilon a B p R force bQ
    hh he hB hR.le hC hmargin hQ
  let A : ℝ≥0 := ⟨(operationalAccelerationBudget epsilon a B h p R bQ).1, hacc.bound_nonneg⟩
  let L : ℝ≥0 := ⟨(operationalAccelerationBudget epsilon a B h p R bQ).2, hacc.lip_nonneg⟩
  let BV := phaseBallSpeed p R A
  have hspeed : ∀ x ∈ closedBall p R, ‖phaseField (acceleration C o h epsilon a force) x‖ ≤ BV :=
    fun x hx => phaseField_norm_le_phaseBallSpeed _ p x R A hx (hacc.norm_le x hx)
  obtain ⟨z, hz0, c, hcL, hcB, htube⟩ := exists_quantitative_ODE_certificate
    (acceleration C o h epsilon a force) p R L BV hR hacc.lipschitzOnWith hspeed
  exact ⟨(quantitativeHorizon_properties R BV hR).1, z, hz0, c,
    congrArg (fun r : ℝ≥0 => (r : ℝ)) hcL,
    congrArg (fun r : ℝ≥0 => (r : ℝ)) hcB, htube⟩

/-- One actual finite-asset reference controls every admitted numerical mesh,
including all off-grid phase states and all displaced reported stencil queries. -/
theorem exists_finiteAsset_trajectory_control (C : Lattice) (Nx Ny : ℕ) (o : Coords)
    (h epsilon a B : ℝ) (p : Plane × Plane) (R delta : ℝ)
    (force : Plane × Plane → Plane) (bQ : MechanicalBounds.Budget)
    (hh : 0 < h) (he : 0 < epsilon) (hB : 0 ≤ B) (hR : 0 < R) (hdelta : 0 ≤ delta)
    (hC : ∀ i j, 0 ≤ i → i < (Nx : ℤ) → 0 ≤ j → j < (Ny : ℤ) → |C i j| ≤ B)
    (hmargin : R + delta < haloMargin Nx Ny o h (chart p.1))
    (hQ : BoundedLipschitzOn force (closedBall p R) bQ.1 bQ.2)
    (residualCoeff roundingCoeff initialCoeff : ℝ≥0) :
    let acc := acceleration C o h epsilon a force
    let bA := operationalAccelerationBudget epsilon a B h p R bQ
    let BV := phaseSpeedBound p R bA
    let T := trajectoryHorizon p R bA
    let K := refinementCoefficient T bA.2 BV residualCoeff roundingCoeff initialCoeff + BV / 2
    0 < T ∧ ∃ z : ℝ → Plane × Plane, z 0 = p ∧
      (∀ t ∈ Icc (0 : ℝ) T, HasDerivAt z (phaseField acc (z t)) t) ∧
      (∀ t ∈ Icc (0 : ℝ) T, z t ∈ ball p R) ∧
      (∀ w : ℝ → Plane × Plane,
        (∀ t ∈ Icc (0 : ℝ) T, HasDerivAt w (phaseField acc (w t)) t) →
        (∀ t ∈ Icc (0 : ℝ) T, w t ∈ ball p R) → w 0 = p → EqOn z w (Icc (0 : ℝ) T)) ∧
      ∀ (N : ℕ) (y : ℕ → Plane × Plane) (Astep : ℕ → Plane),
        ‖y 0 - p‖ ≤ refinementStep T N * initialCoeff →
        (∀ i < N + 1, y i ∈ ball p R →
          ‖Astep i - acc (y i)‖ ≤ refinementStep T N * residualCoeff) →
        (∀ i < N + 1, y i ∈ ball p R →
          ‖y (i + 1) - positionVelocityStep (refinementStep T N) (y i).1 (y i).2 (Astep i)‖ ≤
            (refinementStep T N) ^ 2 * roundingCoeff) →
        2 * T * K ≤ R * ((N : ℝ) + 1) →
        Continuous (trajectoryInterpolant (refinementStep T N) y (N + 1)) ∧
        ∀ t ∈ Icc (0 : ℝ) T,
          ‖trajectoryInterpolant (refinementStep T N) y (N + 1) t - z t‖ ≤ refinementStep T N * K ∧
          trajectoryInterpolant (refinementStep T N) y (N + 1) t ∈ ball p R ∧
          ∀ q : Coords,
            ‖q - chart (trajectoryInterpolant (refinementStep T N) y (N + 1) t).1‖ ≤ delta →
            SafeReportedQuery Nx Ny o h q := by
  obtain ⟨hT, z, hz0, c, hcL, hcB, htube⟩ :=
    exists_finiteAsset_ODE_certificate C Nx Ny o h epsilon a B p R force bQ hh he hB hR hC
      (by linarith) hQ
  refine ⟨hT, z, hz0, c.equation, c.reference_mem, ?_, ?_⟩
  · intro w hw hmem hw0
    exact c.solution_unique w hw hmem (hz0.trans hw0.symm)
  · intro N y Astep hinit hres hround hcount
    let T := trajectoryHorizon p R (operationalAccelerationBudget epsilon a B h p R bQ)
    have hfit : refinementStep T N *
        (refinementCoefficient T c.accelerationLip c.phaseSpeed residualCoeff roundingCoeff initialCoeff +
          c.phaseSpeed / 2) ≤ R / 2 := by
      rw [hcL, hcB]
      unfold refinementStep
      rw [div_mul_eq_mul_div, div_le_iff₀ (by positivity)]
      push_cast
      dsimp only [T]
      nlinarith [hcount]
    have hbound := c.interpolation_error_bound hT N y Astep residualCoeff roundingCoeff initialCoeff
      (R / 2) (by simpa only [hz0] using hinit) hres hround (fun t ht => (htube t ht).2) hfit
    rw [hcL, hcB] at hbound
    refine ⟨continuous_trajectoryInterpolant _ y (div_pos hT (by positivity)) _,
      fun t ht => ⟨(hbound t ht).1, (hbound t ht).2, ?_⟩⟩
    intro q hquery
    exact safeReportedQuery_of_phaseBall Nx Ny o q p _ h R delta hh
      (ball_subset_closedBall (hbound t ht).2) hquery hmargin

/-- Mesh-scaled provider and stored-state budgets converge uniformly to a
constructed finite-asset mechanical solution, retaining all-time halo validity. -/
theorem exists_finiteAsset_interpolation_converges (C : Lattice) (Nx Ny : ℕ) (o : Coords)
    (h epsilon a B : ℝ) (p : Plane × Plane) (R delta : ℝ)
    (force : Plane × Plane → Plane) (bQ : MechanicalBounds.Budget)
    (hh : 0 < h) (he : 0 < epsilon) (hB : 0 ≤ B) (hR : 0 < R) (hdelta : 0 ≤ delta)
    (hC : ∀ i j, 0 ≤ i → i < (Nx : ℤ) → 0 ≤ j → j < (Ny : ℤ) → |C i j| ≤ B)
    (hmargin : R + delta < haloMargin Nx Ny o h (chart p.1))
    (hQ : BoundedLipschitzOn force (closedBall p R) bQ.1 bQ.2)
    (residualCoeff roundingCoeff initialCoeff : ℝ≥0) :
    let acc := acceleration C o h epsilon a force
    let T := trajectoryHorizon p R (operationalAccelerationBudget epsilon a B h p R bQ)
    0 < T ∧ ∃ z : ℝ → Plane × Plane, z 0 = p ∧
      (∀ t ∈ Icc (0 : ℝ) T, HasDerivAt z (phaseField acc (z t)) t) ∧
      (∀ t ∈ Icc (0 : ℝ) T, z t ∈ ball p R) ∧
      (∀ w : ℝ → Plane × Plane,
        (∀ t ∈ Icc (0 : ℝ) T, HasDerivAt w (phaseField acc (w t)) t) →
        (∀ t ∈ Icc (0 : ℝ) T, w t ∈ ball p R) → w 0 = p → EqOn z w (Icc (0 : ℝ) T)) ∧
      ∀ (y : ℕ → ℕ → Plane × Plane) (Astep : ℕ → ℕ → Plane),
        (∀ N, ‖y N 0 - p‖ ≤ refinementStep T N * initialCoeff) →
        (∀ N i, i < N + 1 → y N i ∈ ball p R →
          ‖Astep N i - acc (y N i)‖ ≤ refinementStep T N * residualCoeff) →
        (∀ N i, i < N + 1 → y N i ∈ ball p R →
          ‖y N (i + 1) - positionVelocityStep (refinementStep T N) (y N i).1 (y N i).2 (Astep N i)‖ ≤
            (refinementStep T N) ^ 2 * roundingCoeff) →
        ∀ tolerance > 0, ∀ᶠ N in atTop,
          Continuous (trajectoryInterpolant (refinementStep T N) (y N) (N + 1)) ∧
          ∀ t ∈ Icc (0 : ℝ) T,
            ‖trajectoryInterpolant (refinementStep T N) (y N) (N + 1) t - z t‖ < tolerance ∧
            trajectoryInterpolant (refinementStep T N) (y N) (N + 1) t ∈ ball p R ∧
            ∀ q : Coords,
              ‖q - chart (trajectoryInterpolant (refinementStep T N) (y N) (N + 1) t).1‖ ≤ delta →
              SafeReportedQuery Nx Ny o h q := by
  obtain ⟨hT, z, hz0, c, _, _, htube⟩ :=
    exists_finiteAsset_ODE_certificate C Nx Ny o h epsilon a B p R force bQ hh he hB hR hC
      (by linarith) hQ
  refine ⟨hT, z, hz0, c.equation, c.reference_mem, ?_, ?_⟩
  · intro w hw hmem hw0
    exact c.solution_unique w hw hmem (hz0.trans hw0.symm)
  · intro y Astep hinit hres hround tolerance htolerance
    have hconv := c.interpolation_converges hT y Astep residualCoeff roundingCoeff initialCoeff
      (R / 2) (half_pos hR) (by simpa only [hz0] using hinit) hres hround
      (fun t ht => (htube t ht).2) tolerance htolerance
    filter_upwards [hconv] with N hN
    refine ⟨hN.1, fun t ht => ⟨(hN.2 t ht).1, (hN.2 t ht).2, ?_⟩⟩
    intro q hquery
    exact safeReportedQuery_of_phaseBall Nx Ny o q p _ h R delta hh
      (ball_subset_closedBall (hN.2 t ht).2) hquery hmargin

end FractalSync.OperationalDynamics
