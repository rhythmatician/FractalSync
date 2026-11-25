import FractalSync.Analysis.ProperGeodesics
import Mathlib.Topology.ContinuousMap.Bounded.ArzelaAscoli
import Mathlib.Topology.Sequences

/-!
# Uniform sequential compactness of bounded-speed endpoint paths

The bounded-continuous-function form of Arzelà–Ascoli upgrades the existing
pointwise compact path family to an actual uniformly convergent subsequence.
-/

namespace FractalSync.MetricGeometry

open Set Filter
open scoped Topology NNReal BoundedContinuousFunction

variable {X : Type*} [MetricSpace X] [ProperSpace X]

/-- A sequence of endpoint paths with one Lipschitz bound has a uniformly
convergent subsequence in a proper metric space. -/
theorem exists_uniformly_convergent_subsequence (x y : X) (K : ℝ≥0)
    (f : ℕ → unitInterval → X)
    (hf : ∀ n, f n ∈ boundedSpeedPaths x y K) :
    ∃ g : unitInterval → X, g ∈ boundedSpeedPaths x y K ∧
      ∃ phi : ℕ → ℕ, StrictMono phi ∧ TendstoUniformly (f ∘ phi) g atTop := by
  let A : Set (unitInterval →ᵇ X) :=
    {g | (g : unitInterval → X) ∈ boundedSpeedPaths x y K}
  have hclosed : IsClosed A :=
    (isClosed_boundedSpeedPaths x y K).preimage BoundedContinuousFunction.continuous_coe
  have hequi : Equicontinuous (fun g : A => (g.1 : unitInterval → X)) :=
    Metric.equicontinuous_of_continuity_modulus (fun r : ℝ => K * r)
      (by simpa using (continuous_const.mul continuous_id).tendsto (0 : ℝ))
      _ (fun s t g => g.2.2.2 s t)
  have hrange : ∀ (g : unitInterval →ᵇ X) (t : unitInterval), g ∈ A →
      g t ∈ Metric.closedBall x K := by
    intro g t hg
    have hdist : dist t (0 : unitInterval) ≤ 1 := by
      simpa only [Subtype.dist_eq, Set.Icc.coe_zero, Real.dist_eq, sub_zero,
        abs_of_nonneg t.2.1] using t.2.2
    simpa only [Metric.mem_closedBall, hg.1, mul_one] using
      (hg.2.2 t 0).trans (mul_le_mul_of_nonneg_left hdist K.2)
  have hcompact : IsCompact A := BoundedContinuousFunction.arzela_ascoli₂
    (Metric.closedBall x K) (isCompact_closedBall x K) A hclosed hrange hequi
  let F : ℕ → unitInterval →ᵇ X := fun n => BoundedContinuousFunction.mkOfCompact
    ⟨f n, (LipschitzWith.of_dist_le_mul (K := K) (hf n).2.2).continuous⟩
  obtain ⟨g, hg, phi, hphi, hlim⟩ := hcompact.tendsto_subseq (fun n => show F n ∈ A from hf n)
  exact ⟨g, hg, phi, hphi, BoundedContinuousFunction.tendsto_iff_tendstoUniformly.mp hlim⟩

end FractalSync.MetricGeometry
