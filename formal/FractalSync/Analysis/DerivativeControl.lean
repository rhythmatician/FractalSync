import Mathlib.Analysis.Calculus.Deriv.Slope
import Mathlib.Analysis.SpecialFunctions.ExpDeriv

/-! A local chord control with a continuous exponential factor bounds the
actual derivative. Project support for the graph-length identification. -/

namespace FractalSync.Analysis

open Filter
open scoped Topology

theorem norm_deriv_le_exp_control {F : Type*} [NormedAddCommGroup F] [NormedSpace ℝ F]
    {f : ℝ → F} {A : ℝ → ℝ} {t C A' : ℝ} {f' : F}
    (hf : HasDerivAt f f' t) (hA : HasDerivAt A A' t)
    (hbound : ∀ᶠ s in 𝓝 t, ‖f s - f t‖ ≤ C * Real.exp (|A s - A t|) * |A s - A t|) :
    ‖f'‖ ≤ C * |A'| := by
  have hfn := (hasDerivAt_iff_tendsto_slope.mp hf).norm
  have hAn : Tendsto (fun s => |slope A t s|) (𝓝[≠] t) (𝓝 |A'|) := by
    simpa only [Real.norm_eq_abs] using (hasDerivAt_iff_tendsto_slope.mp hA).norm
  have hzero : Tendsto (fun s => |A s - A t|) (𝓝[≠] t) (𝓝 (0 : ℝ)) := by
    simpa using ((hA.continuousAt.tendsto.mono_left nhdsWithin_le_nhds).sub_const (A t)).abs
  have hright : Tendsto (fun s => C * Real.exp |A s - A t| * |slope A t s|)
      (𝓝[≠] t) (𝓝 (C * |A'|)) := by
    simpa using (((Real.continuous_exp.tendsto 0).comp hzero).const_mul C).mul hAn
  apply le_of_tendsto_of_tendsto hfn hright
  filter_upwards [hbound.filter_mono nhdsWithin_le_nhds] with s hs
  simpa only [slope_def_module, norm_smul, Real.norm_eq_abs, smul_eq_mul,
    abs_mul, abs_inv, mul_assoc, mul_left_comm, mul_comm] using
      mul_le_mul_of_nonneg_left hs (inv_nonneg.mpr (abs_nonneg (s - t)))

end FractalSync.Analysis
