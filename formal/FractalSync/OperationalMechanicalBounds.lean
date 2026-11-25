import FractalSync.MechanicalBounds
import FractalSync.OperationalRulerBounds
import FractalSync.OperationalHalo

/-!
# Quantitative mechanics from finite operational assets

Literal coefficient bounds give explicit norm and Lipschitz bounds for the
actual mechanical acceleration on a supplied phase ball. Only the force
certificate remains an external coefficient assumption. A finite asset
suffices when the ball lies inside its physical halo; arbitrary lattice
values outside the stored rectangle have no effect on the conclusion.
-/

noncomputable section

namespace FractalSync.OperationalDynamics

open Set Metric OperationalField FlatShoreIntrinsic ScaleMetric
open MechanicalBounds OperationalHalo
open scoped NNReal

local instance operationalBoundsPhaseNormedSpace : NormedSpace ℝ (Plane × Plane) :=
  Prod.normedSpace

/-- The explicit pair is (acceleration norm bound, acceleration Lipschitz constant). -/
def operationalAccelerationBudget (epsilon a B h : ℝ) (p : Plane × Plane)
    (R : ℝ) (forceBudget : Budget) : Budget :=
  accelerationBudget epsilon a (ruler epsilon B, splineGradientBound B h)
    (splineGradientBound B h, rulerHessianBound B h epsilon)
    (rulerHessianBound B h epsilon, rulerThirdBound B h epsilon)
    forceBudget (‖p.2‖ + R, 1)

theorem velocity_boundedLipschitz (p : Plane × Plane) (R : ℝ) (hR : 0 ≤ R) :
    BoundedLipschitzOn (fun z : Plane × Plane => z.2) (closedBall p R) (‖p.2‖ + R) 1 := by
  refine ⟨add_nonneg (norm_nonneg _) hR, zero_le_one, ?_, ?_⟩
  · exact fun z hz => (norm_le_norm_sub_add z.2 p.2).trans
      (by linarith [show ‖z.2 - p.2‖ ≤ R from
        (norm_snd_le (z - p)).trans (mem_closedBall_iff_norm.mp hz)])
  · exact fun x _ y _ => by simpa only [one_mul, dist_eq_norm] using norm_snd_le (x - y)

/-- Uniform bounds for the literal ruler on the entire Euclidean plane. -/
theorem rulerField_boundedLipschitz (C : Lattice) (o : Coords) (h epsilon B : ℝ)
    (hh : 0 < h) (he : 0 < epsilon) (hB : 0 ≤ B) (hC : ∀ i j, |C i j| ≤ B) :
    BoundedLipschitzOn (rulerField C o h epsilon) univ
      (ruler epsilon B) (splineGradientBound B h) := by
  exact ⟨(ruler_pos epsilon B he).le, by unfold splineGradientBound; positivity,
    fun x _ => (Real.norm_of_nonneg (ruler_pos epsilon _ he).le).le.trans
      (rulerField_bounds C o h epsilon B hh he hB hC x).2,
    fun x _ y _ => by simpa only [Real.norm_eq_abs, dist_eq_norm] using
      rulerField_sub_le C o h epsilon B hh he hB hC y x⟩

theorem gradient_rulerField_boundedLipschitz (C : Lattice) (o : Coords) (h epsilon B : ℝ)
    (hh : 0 < h) (he : 0 < epsilon) (hB : 0 ≤ B) (hC : ∀ i j, |C i j| ≤ B) :
    BoundedLipschitzOn (gradient (rulerField C o h epsilon)) univ
      (splineGradientBound B h) (rulerHessianBound B h epsilon) := by
  exact ⟨by unfold splineGradientBound; positivity,
    by unfold rulerHessianBound splineHessianBound; positivity,
    fun x _ => norm_gradient_rulerField_le C o h epsilon B hh he hB hC x,
    fun x _ y _ => by simpa only [dist_eq_norm] using
      gradient_rulerField_sub_le C o h epsilon B hh he hB hC y x⟩

theorem hessian_rulerField_boundedLipschitz (C : Lattice) (o : Coords) (h epsilon B : ℝ)
    (hh : 0 < h) (he : 0 < epsilon) (hB : 0 ≤ B) (hC : ∀ i j, |C i j| ≤ B) :
    BoundedLipschitzOn (fderiv ℝ (gradient (rulerField C o h epsilon))) univ
      (rulerHessianBound B h epsilon) (rulerThirdBound B h epsilon) := by
  exact ⟨by unfold rulerHessianBound splineHessianBound; positivity,
    by unfold rulerThirdBound splineThirdBound splineGradientBound splineHessianBound; positivity,
    fun x _ => norm_hessian_rulerField_le C o h epsilon B hh he hB hC x,
    fun x _ y _ => by simpa only [dist_eq_norm] using
      hessian_rulerField_sub_le C o h epsilon B hh he hB hC y x⟩

/-- Explicit constants for the actual operational acceleration, including
the changing ruler and actual metric jets throughout the supplied phase ball. -/
theorem acceleration_boundedLipschitz (C : Lattice) (o : Coords) (h epsilon a B : ℝ)
    (p : Plane × Plane) (R : ℝ) (force : Plane × Plane → Plane) (bQ : Budget)
    (hh : 0 < h) (he : 0 < epsilon) (hB : 0 ≤ B) (hR : 0 ≤ R)
    (hC : ∀ i j, |C i j| ≤ B)
    (hQ : BoundedLipschitzOn force (closedBall p R) bQ.1 bQ.2) :
    BoundedLipschitzOn (acceleration C o h epsilon a force) (closedBall p R)
      (operationalAccelerationBudget epsilon a B h p R bQ).1
      (operationalAccelerationBudget epsilon a B h p R bQ).2 := by
  have hfst : LipschitzOnWith 1 (fun z : Plane × Plane => z.1) (closedBall p R) :=
    LipschitzOnWith.of_dist_le_mul (fun x _ y _ => by
      simpa only [NNReal.coe_one, one_mul, dist_eq_norm] using norm_fst_le (x - y))
  have hr : BoundedLipschitzOn (fun z : Plane × Plane => rulerField C o h epsilon z.1)
      (closedBall p R) (ruler epsilon B) (splineGradientBound B h) := by
    simpa only [NNReal.coe_one, mul_one] using
      (rulerField_boundedLipschitz C o h epsilon B hh he hB hC).comp hfst
        (fun _ _ => mem_univ _)
  have hu : BoundedLipschitzOn (fun z : Plane × Plane => gradient (rulerField C o h epsilon) z.1)
      (closedBall p R) (splineGradientBound B h) (rulerHessianBound B h epsilon) := by
    simpa only [NNReal.coe_one, mul_one] using
      (gradient_rulerField_boundedLipschitz C o h epsilon B hh he hB hC).comp hfst
        (fun _ _ => mem_univ _)
  have hH : BoundedLipschitzOn
      (fun z : Plane × Plane => fderiv ℝ (gradient (rulerField C o h epsilon)) z.1)
      (closedBall p R) (rulerHessianBound B h epsilon) (rulerThirdBound B h epsilon) := by
    simpa only [NNReal.coe_one, mul_one] using
      (hessian_rulerField_boundedLipschitz C o h epsilon B hh he hB hC).comp hfst
        (fun _ _ => mem_univ _)
  exact acceleration_bound epsilon a (closedBall p R)
    (fun z => rulerField C o h epsilon z.1)
    (fun z => gradient (rulerField C o h epsilon) z.1)
    (fun z => fderiv ℝ (gradient (rulerField C o h epsilon)) z.1)
    force (fun z => z.2)
    (ruler epsilon B, splineGradientBound B h)
    (splineGradientBound B h, rulerHessianBound B h epsilon)
    (rulerHessianBound B h epsilon, rulerThirdBound B h epsilon) bQ (‖p.2‖ + R, 1)
    he hr hu hH hQ (velocity_boundedLipschitz p R hR)
    (fun z _ => (rulerField_bounds C o h epsilon B hh he hB hC z.1).1)

/-- On a phase ball inside the finite halo, the zero extension and the
arbitrary original extension give exactly the same actual acceleration. -/
theorem acceleration_zeroExtension_eqOn (C : Lattice) (Nx Ny : ℕ) (o : Coords)
    (h epsilon a : ℝ) (p : Plane × Plane) (R : ℝ) (force : Plane × Plane → Plane)
    (hh : 0 < h) (hmargin : R < haloMargin Nx Ny o h (chart p.1)) :
    EqOn (acceleration (zeroExtension C Nx Ny) o h epsilon a force)
      (acceleration C o h epsilon a force) (closedBall p R) := by
  exact fun z hz => (acceleration_eventuallyEq (zeroExtension C Nx Ny) C Nx Ny o h epsilon a
    (zeroExtension_agrees C Nx Ny) force z
    (reportedQuery_mem_of_phaseBall Nx Ny o (chart z.1) p z h R 0 hh hz
      (by simp) (by simpa only [add_zero] using hmargin))).self_of_nhds

/-- Finite stored coefficient evidence suffices for the explicit actual
acceleration certificate. No values outside the asset are bounded. -/
theorem finiteAsset_acceleration_boundedLipschitz (C : Lattice) (Nx Ny : ℕ) (o : Coords)
    (h epsilon a B : ℝ) (p : Plane × Plane) (R : ℝ) (force : Plane × Plane → Plane)
    (bQ : Budget) (hh : 0 < h) (he : 0 < epsilon) (hB : 0 ≤ B) (hR : 0 ≤ R)
    (hC : ∀ i j, 0 ≤ i → i < (Nx : ℤ) → 0 ≤ j → j < (Ny : ℤ) → |C i j| ≤ B)
    (hmargin : R < haloMargin Nx Ny o h (chart p.1))
    (hQ : BoundedLipschitzOn force (closedBall p R) bQ.1 bQ.2) :
    BoundedLipschitzOn (acceleration C o h epsilon a force) (closedBall p R)
      (operationalAccelerationBudget epsilon a B h p R bQ).1
      (operationalAccelerationBudget epsilon a B h p R bQ).2 := by
  have hc := acceleration_boundedLipschitz (zeroExtension C Nx Ny) o h epsilon a B p R force bQ
    hh he hB hR (abs_zeroExtension_le C Nx Ny B hB hC) hQ
  have heq := acceleration_zeroExtension_eqOn C Nx Ny o h epsilon a p R force hh hmargin
  exact ⟨hc.bound_nonneg, hc.lip_nonneg,
    fun x hx => by rw [← heq hx]; exact hc.norm_le x hx,
    fun x hx y hy => by rw [← heq hx, ← heq hy]; exact hc.norm_sub_le x hx y hy⟩

end FractalSync.OperationalDynamics
