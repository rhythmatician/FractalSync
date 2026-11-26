import FractalSync.LocalDynamics
import FractalSync.OperationalFieldQueryCorrespondence
import FractalSync.FlatShoreIntrinsic
import Mathlib.Analysis.Calculus.Gradient.Basic
import Mathlib.Analysis.InnerProductSpace.ProdL2

/-!
# Local mechanics for the literal operational spline

The globally glued physical spline is pulled back to the Euclidean plane.
Positive regularization gives a C4 ruler, its actual Riesz gradient and
actual derivative Hessian, and an actual local mechanical ODE solution.
No signed-distance fidelity premise or arbitrary gradient/Hessian field is
used. Physical spacing is normally positive; the regularity statements also
hold for the totalized exact-real normalization at any spacing.
-/

noncomputable section

namespace FractalSync.OperationalDynamics

open OperationalField FlatShoreIntrinsic ScaleMetric Set Metric

local instance operationalDynamicsPhaseNormedSpace : NormedSpace ℝ (Plane × Plane) :=
  Prod.normedSpace

/-- The regularized literal global physical spline, on the Euclidean plane. -/
def rulerField (C : Lattice) (o : Coords) (h epsilon : ℝ) (x : Plane) : ℝ :=
  ruler epsilon (physicalField C o h (chart x))

/-- Acceleration uses the actual gradient and its actual Fréchet derivative. -/
def acceleration (C : Lattice) (o : Coords) (h epsilon a : ℝ)
    (force : Plane × Plane → Plane) : Plane × Plane → Plane :=
  mechanicalAcceleration a (rulerField C o h epsilon)
    (gradient (rulerField C o h epsilon))
    (fderiv ℝ (gradient (rulerField C o h epsilon))) force

/-- Positive regularization preserves C4 smoothness across every spline knot. -/
theorem contDiff_rulerField (C : Lattice) (o : Coords) (h epsilon : ℝ)
    (he : 0 < epsilon) : ContDiff ℝ 4 (rulerField C o h epsilon) := by
  unfold rulerField ruler
  apply (((contDiff_physicalField C o h).comp chart.contDiff).pow 2 |>.add contDiff_const).sqrt
  intro x
  positivity

/-- The actual Riesz gradient is C3 and its actual derivative Hessian is C2. -/
theorem actual_jet_regularity (C : Lattice) (o : Coords) (h epsilon : ℝ)
    (he : 0 < epsilon) :
    ContDiff ℝ 3 (gradient (rulerField C o h epsilon)) ∧
      ContDiff ℝ 2 (fderiv ℝ (gradient (rulerField C o h epsilon))) := by
  have hu : ContDiff ℝ 3 (gradient (rulerField C o h epsilon)) :=
    (InnerProductSpace.toDual ℝ Plane).symm.toContinuousLinearEquiv.contDiff.comp
      ((contDiff_rulerField C o h epsilon he).fderiv_right (m := 3) (by norm_num))
  exact ⟨hu, hu.fderiv_right (m := 2) (by norm_num)⟩

/-- The actual ruler derivative is supplied by the literal physical query jet,
transported through the Euclidean chart and the regularization chain rule. -/
theorem hasFDerivAt_rulerField (C : Lattice) (o : Coords) (h epsilon : ℝ)
    (he : 0 < epsilon) (x : Plane) :
    HasFDerivAt (rulerField C o h epsilon)
      ((physicalField C o h (chart x) / rulerField C o h epsilon x) •
        (physicalDifferential C o h (chart x)).comp chart.toContinuousLinearMap) x := by
  exact hasFDerivAt_ruler_comp epsilon he _ _ x
    ((hasFDerivAt_physicalField C o (chart x) h).comp x chart.hasFDerivAt)

/-- Actual operational metric jets and a C1 force give C1 mechanical acceleration.
Only the force regularity remains an input coefficient assumption. -/
theorem contDiffAt_acceleration (C : Lattice) (o : Coords) (h epsilon a : ℝ)
    (he : 0 < epsilon) (force : Plane × Plane → Plane) (p : Plane × Plane)
    (hF : ContDiffAt ℝ 1 force p) :
    ContDiffAt ℝ 1 (acceleration C o h epsilon a force) p := by
  obtain ⟨hu, hH⟩ := actual_jet_regularity C o h epsilon he
  exact contDiffAt_mechanicalAcceleration a _ _ _ force p
    ((contDiff_rulerField C o h epsilon he).contDiffAt.of_le (by norm_num))
    (hu.contDiffAt.of_le (by norm_num)) (hH.contDiffAt.of_le one_le_two) hF
    (ruler_pos epsilon _ he).ne'

/-- Every initial phase state admits a genuine local operational mechanical
trajectory, a uniform certificate, and a positive domain tube. -/
theorem exists_local_certificate (C : Lattice) (o : Coords) (h epsilon a : ℝ)
    (he : 0 < epsilon) (force : Plane × Plane → Plane) (p : Plane × Plane)
    (hF : ContDiffAt ℝ 1 force p) :
    ∃ T > (0 : ℝ), ∃ R > (0 : ℝ), ∃ z : ℝ → Plane × Plane,
      z 0 = p ∧
      Nonempty (UniformODECertificate (acceleration C o h epsilon a force)
        z T (ball p R)) ∧
      ∀ t ∈ Icc (0 : ℝ) T, ∀ w : Plane × Plane,
        ‖w - z t‖ ≤ R / 2 → w ∈ ball p R := by
  exact exists_local_ODE_certificate _ p (contDiffAt_acceleration C o h epsilon a he force p hF)

end FractalSync.OperationalDynamics
