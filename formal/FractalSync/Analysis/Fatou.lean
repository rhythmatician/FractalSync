import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.MeasureTheory.Integral.Lebesgue.Add

/-! Compatibility lemmas deriving the sequential integrability criterion from
the pinned Mathlib Fatou lemma. No new analytic assumption is introduced. -/

open MeasureTheory Filter
open scoped Topology

namespace FractalSync.Analysis

variable {α E : Type*} [MeasurableSpace α] [NormedAddCommGroup E]
  {μ : Measure α} {f : ℕ → α → E} {g : α → E}

theorem lintegral_enorm_le_liminf_of_tendsto
    (hlim : ∀ᵐ x ∂μ, Tendsto (fun n => f n x) atTop (𝓝 (g x)))
    (hmeas : ∀ n, AEMeasurable (fun x => ‖f n x‖ₑ) μ) :
    (∫⁻ x, ‖g x‖ₑ ∂μ) ≤ liminf (fun n => ∫⁻ x, ‖f n x‖ₑ ∂μ) atTop := by
  have hnorm : ∀ᵐ x ∂μ, liminf (fun n => ‖f n x‖ₑ) atTop = ‖g x‖ₑ :=
    hlim.mono fun x hx => ((continuous_enorm.tendsto (g x)).comp hx).liminf_eq
  exact (lintegral_congr_ae hnorm).symm.le.trans (lintegral_liminf_le' hmeas)

theorem integrable_of_tendsto
    (hlim : ∀ᵐ x ∂μ, Tendsto (fun n => f n x) atTop (𝓝 (g x)))
    (hmeas : ∀ n, AEStronglyMeasurable (f n) μ)
    (hfinite : liminf (fun n => ∫⁻ x, ‖f n x‖ₑ ∂μ) atTop ≠ ⊤) :
    Integrable g μ := by
  borelize E
  refine ⟨aestronglyMeasurable_of_tendsto_ae _ hmeas hlim, ?_⟩
  exact (lintegral_enorm_le_liminf_of_tendsto hlim
    (fun n => (hmeas n).aemeasurable.enorm)).trans_lt (lt_top_iff_ne_top.mpr hfinite)

end FractalSync.Analysis
