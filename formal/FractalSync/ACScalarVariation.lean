import FractalSync.Analysis.AbsolutelyContinuousFTC
import Mathlib.Analysis.Calculus.Deriv.Slope

/-!
# Scalar absolute continuity and equality in total variation

For an actual absolutely continuous real function on an ordered interval,
the integral of the absolute derivative equals the signed endpoint increment
exactly when the function is nondecreasing. Endpoint derivative values do
not affect the statement, including on a degenerate interval.
-/

open Set Filter MeasureTheory
open scoped Topology

namespace FractalSync.ACScalarVariation

/-- Monotonicity on a closed interval makes the actual derivative
nonnegative almost everywhere; no endpoint derivative is assumed. -/
theorem ae_deriv_nonneg_of_monotoneOn (f : ℝ → ℝ) (lo hi : ℝ) (horder : lo ≤ hi)
    (hmono : MonotoneOn f (uIcc lo hi)) :
    ∀ᵐ t, t ∈ uIcc lo hi → 0 ≤ deriv f t := by
  filter_upwards [show ∀ᵐ t : ℝ, t ≠ lo from by simp [ae_iff],
    show ∀ᵐ t : ℝ, t ≠ hi from by simp [ae_iff]] with t hlo hhi ht
  rw [uIcc_of_le horder] at hmono ht
  simpa only [derivWithin_of_mem_nhds
    (Icc_mem_nhds (lt_of_le_of_ne ht.1 hlo.symm) (lt_of_le_of_ne ht.2 hhi))] using
    hmono.derivWithin_nonneg (x := t)

/-- Almost-everywhere nonnegative actual derivative gives monotonicity for
an absolutely continuous function, by the FTC on every subinterval. -/
theorem monotoneOn_of_ae_deriv_nonneg (f : ℝ → ℝ) (lo hi : ℝ)
    (hf : AbsolutelyContinuousOnInterval f lo hi)
    (hderiv : ∀ᵐ t, t ∈ uIcc lo hi → 0 ≤ deriv f t) :
    MonotoneOn f (uIcc lo hi) := by
  intro x hx y hy hxy
  apply sub_nonneg.mp
  rw [← (hf.mono (uIcc_subset_uIcc hx hy)).integral_deriv_eq_sub]
  apply intervalIntegral.integral_nonneg_of_ae_restrict hxy
  rw [EventuallyLE, ae_restrict_iff' measurableSet_Icc]
  exact hderiv.mono (fun t ht hmem => ht ((uIcc_subset_uIcc hx hy)
    (by simpa only [uIcc_of_le hxy] using hmem)))

/-- Equality between total scalar variation and the signed increment
characterizes every nondecreasing absolutely continuous traversal. -/
theorem integral_abs_deriv_eq_sub_iff_monotoneOn (f : ℝ → ℝ) (lo hi : ℝ)
    (horder : lo ≤ hi) (hf : AbsolutelyContinuousOnInterval f lo hi) :
    (∫ t in lo..hi, |deriv f t|) = f hi - f lo ↔ MonotoneOn f (uIcc lo hi) := by
  constructor
  · intro heq
    have hsame : (∫ t in lo..hi, deriv f t) = ∫ t in lo..hi, |deriv f t| :=
      hf.integral_deriv_eq_sub.trans heq.symm
    have hae : deriv f =ᵐ[volume.restrict (Ioc lo hi)] (fun t => |deriv f t|) :=
      (integral_eq_iff_of_ae_le hf.intervalIntegrable_deriv.1 hf.intervalIntegrable_deriv.abs.1
        (Eventually.of_forall (fun t => le_abs_self (deriv f t)))).1
          (by simpa only [intervalIntegral.integral_of_le horder] using hsame)
    rw [Measure.restrict_congr_set Ioc_ae_eq_Icc, EventuallyEq,
      ae_restrict_iff' measurableSet_Icc] at hae
    apply monotoneOn_of_ae_deriv_nonneg f lo hi hf
    exact hae.mono (fun t ht hmem => (abs_nonneg (deriv f t)).trans_eq
      (ht (by simpa only [uIcc_of_le horder] using hmem)).symm)
  · intro hmono
    rw [← hf.integral_deriv_eq_sub]
    apply intervalIntegral.integral_congr_ae
    exact (ae_deriv_nonneg_of_monotoneOn f lo hi horder hmono).mono
      (fun t ht hmem => abs_of_nonneg (ht (uIoc_subset_uIcc hmem)))

end FractalSync.ACScalarVariation
