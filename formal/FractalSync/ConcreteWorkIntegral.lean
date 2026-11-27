import FractalSync.ConcreteHeldPower
import Mathlib.MeasureTheory.Integral.IntervalIntegral.FundThmCalculus

/-!
# Actual integrated work of the concrete reference dynamics

The reference equation and membership in the certified phase ball discharge
the derivative and integrability hypotheses of the fundamental theorem of
calculus. The resulting integral is the actual signed drive-plus-drag work.
-/

noncomputable section

namespace FractalSync.ConcreteWorkIntegral

open Set Metric FlatShoreIntrinsic ScaleMetric ConcreteOperationalAsset
open ConcreteOperationalDynamics (acc)
open ConcreteHeldPower

local instance workIntegralPhaseNormedSpace : NormedSpace ℝ (Plane × Plane) :=
  Prod.normedSpace

/-- Signed work integrated along the actual continuously reevaluated reference field. -/
def referenceWork (z : ℝ → Plane × Plane) (t : ℝ) : ℝ :=
  ∫ s in (0 : ℝ)..t, targetPower (z s)

/-- The actual energy derivative along a reference solution equals its actual power. -/
theorem hasDerivAt_energy_trace (z : ℝ → Plane × Plane) (T : ℝ)
    (heq : ∀ s ∈ Icc (0 : ℝ) T, HasDerivAt z (phaseField acc (z s)) s)
    (hmem : ∀ s ∈ Icc (0 : ℝ) T, z s ∈ closedBall phase radius)
    (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) :
    HasDerivAt (fun s => energy (z s)) (targetPower (z t)) t := by
  simpa only [Function.comp_def, fderiv_energy_targetField (z t) (hmem t ht)] using
    (energy_differentiable (z t) (hmem t ht)).hasFDerivAt.comp_hasDerivAt t (heq t ht)

/-- The reference power trace is continuous on the entire certified time interval. -/
theorem continuousOn_power_trace (z : ℝ → Plane × Plane) (T : ℝ)
    (heq : ∀ s ∈ Icc (0 : ℝ) T, HasDerivAt z (phaseField acc (z s)) s)
    (hmem : ∀ s ∈ Icc (0 : ℝ) T, z s ∈ closedBall phase radius) :
    ContinuousOn (fun s => targetPower (z s)) (Icc (0 : ℝ) T) := by
  exact targetPower_continuousOn.comp
    (fun s hs => (heq s hs).continuousAt.continuousWithinAt) hmem

/-- Actual power is interval integrable up to every time in the certified interval. -/
theorem intervalIntegrable_power_trace (z : ℝ → Plane × Plane) (T : ℝ)
    (heq : ∀ s ∈ Icc (0 : ℝ) T, HasDerivAt z (phaseField acc (z s)) s)
    (hmem : ∀ s ∈ Icc (0 : ℝ) T, z s ∈ closedBall phase radius)
    (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) :
    IntervalIntegrable (fun s => targetPower (z s)) MeasureTheory.volume 0 t := by
  exact ContinuousOn.intervalIntegrable_of_Icc ht.1
    ((continuousOn_power_trace z T heq hmem).mono
      (fun _ hs => ⟨hs.1, hs.2.trans ht.2⟩))

/-- The actual power integral equals the actual total-energy change. -/
theorem referenceWork_eq_energy_sub (z : ℝ → Plane × Plane) (T : ℝ)
    (heq : ∀ s ∈ Icc (0 : ℝ) T, HasDerivAt z (phaseField acc (z s)) s)
    (hmem : ∀ s ∈ Icc (0 : ℝ) T, z s ∈ closedBall phase radius)
    (t : ℝ) (ht : t ∈ Icc (0 : ℝ) T) :
    referenceWork z t = energy (z t) - energy (z 0) := by
  exact intervalIntegral.integral_eq_sub_of_hasDerivAt_of_le ht.1
    (fun s hs => (hasDerivAt_energy_trace z T heq hmem s
      ⟨hs.1, hs.2.trans ht.2⟩).continuousAt.continuousWithinAt)
    (fun s hs => hasDerivAt_energy_trace z T heq hmem s
      ⟨hs.1.le, hs.2.le.trans ht.2⟩)
    (intervalIntegrable_power_trace z T heq hmem t ht)

/-- The actual cumulative work is 40-Lipschitz on the certified time interval. -/
theorem referenceWork_lipschitzOn (z : ℝ → Plane × Plane) (T : ℝ)
    (heq : ∀ s ∈ Icc (0 : ℝ) T, HasDerivAt z (phaseField acc (z s)) s)
    (hmem : ∀ s ∈ Icc (0 : ℝ) T, z s ∈ closedBall phase radius) :
    LipschitzOnWith 40 (referenceWork z) (Icc (0 : ℝ) T) := by
  have henergy : LipschitzOnWith 40 (fun s => energy (z s) - energy (z 0))
      (Icc (0 : ℝ) T) := by
    refine (convex_Icc (0 : ℝ) T).lipschitzOnWith_of_nnnorm_hasDerivWithin_le
      (fun s hs => ((hasDerivAt_energy_trace z T heq hmem s hs).sub_const
        (energy (z 0))).hasDerivWithinAt) ?_
    intro s hs
    exact_mod_cast targetPower_bound.norm_le (z s) (hmem s hs)
  apply LipschitzOnWith.of_dist_le_mul
  intro s hs t ht
  rw [referenceWork_eq_energy_sub z T heq hmem s hs,
    referenceWork_eq_energy_sub z T heq hmem t ht]
  exact henergy.dist_le_mul s hs t ht

end FractalSync.ConcreteWorkIntegral
