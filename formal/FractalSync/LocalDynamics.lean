import FractalSync.TrajectoryInterpolation
import Mathlib.Analysis.ODE.PicardLindelof
import Mathlib.Analysis.ODE.Gronwall

/-!
# Local reference trajectories from actual acceleration regularity

Actual C1 acceleration supplies a positive local time horizon, a phase ball,
an actual ODE solution, and the uniform certificate and interior tube used
by the numerical trajectory theorem. Existence of constants is separate
from numerical certification of their values.
-/

open Set Filter Metric
open scoped Topology NNReal InnerProductSpace

namespace FractalSync.ScaleMetric

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

noncomputable local instance localDynamicsNormedSpace : NormedSpace ℝ (E × E) := Prod.normedSpace

omit [InnerProductSpace ℝ E] in
/-- A ballwise acceleration Lipschitz constant bounds the actual phase field
using its value at the center. -/
theorem phaseField_norm_le_on_ball (acc : E × E → E) (p z : E × E)
    (L : ℝ≥0) (R : ℝ) (hR : 0 < R) (hz : z ∈ ball p R)
    (hLip : LipschitzOnWith L acc (ball p R)) :
    ‖phaseField acc z‖ ≤ max 1 (L : ℝ) * R + ‖phaseField acc p‖ := by
  have hacc : ‖acc z - acc p‖ ≤ (L : ℝ) * ‖z - p‖ := by
    simpa only [dist_eq_norm] using hLip.dist_le_mul z hz p (mem_ball_self hR)
  have hnorm : ‖z - p‖ ≤ R := (mem_ball_iff_norm.mp hz).le
  exact (norm_le_norm_sub_add (phaseField acc z) (phaseField acc p)).trans
    (add_le_add_right ((phaseField_sub_le acc z p L hacc).trans
      (mul_le_mul_of_nonneg_left hnorm (zero_le_one.trans (le_max_left 1 (L : ℝ))))) _)

/-- The actual phase vector field is C1 wherever the actual acceleration is. -/
theorem contDiffAt_phaseField (acc : E × E → E) (p : E × E)
    (hacc : ContDiffAt ℝ 1 acc p) : ContDiffAt ℝ 1 (phaseField acc) p := by
  exact contDiffAt_snd.prodMk hacc

/-- The certified actual solution is unique among actual solutions remaining
in the same phase region and starting at the same state. -/
theorem UniformODECertificate.solution_unique {acc : E × E → E} {z : ℝ → E × E}
    {T : ℝ} {K : Set (E × E)} (c : UniformODECertificate acc z T K)
    (w : ℝ → E × E)
    (hw : ∀ t ∈ Icc (0 : ℝ) T, HasDerivAt w (phaseField acc (w t)) t)
    (hwmem : ∀ t ∈ Icc (0 : ℝ) T, w t ∈ K) (hw0 : z 0 = w 0) :
    EqOn z w (Icc (0 : ℝ) T) := by
  have hF : LipschitzOnWith (max 1 c.accelerationLip) (phaseField acc) K := by
    apply LipschitzOnWith.of_dist_le_mul
    intro u hu v hv
    simpa only [dist_eq_norm, NNReal.coe_max, NNReal.coe_one] using
      phaseField_sub_le acc u v c.accelerationLip (c.acceleration_lipschitz u hu v hv)
  exact ODE_solution_unique_of_mem_Icc_right (v := fun _ => phaseField acc) (s := fun _ => K)
    (fun _ _ => hF) (fun t ht => (c.equation t ht).continuousAt.continuousWithinAt)
    (fun t ht => (c.equation t (Ico_subset_Icc_self ht)).hasDerivWithinAt)
    (fun t ht => c.reference_mem t (Ico_subset_Icc_self ht))
    (fun t ht => (hw t ht).continuousAt.continuousWithinAt)
    (fun t ht => (hw t (Ico_subset_Icc_self ht)).hasDerivWithinAt)
    (fun t ht => hwmem t (Ico_subset_Icc_self ht)) hw0

/-- Actual local C1 acceleration yields a reference solution together with
all the uniform hypotheses used by trajectory refinement, including a
positive tube inside a phase ball. No solution is supplied as an assumption. -/
theorem exists_local_ODE_certificate [CompleteSpace E] (acc : E × E → E) (p : E × E)
    (hacc : ContDiffAt ℝ 1 acc p) :
    ∃ T > (0 : ℝ), ∃ R > (0 : ℝ), ∃ z : ℝ → E × E,
      z 0 = p ∧ Nonempty (UniformODECertificate acc z T (ball p R)) ∧
      ∀ t ∈ Icc (0 : ℝ) T, ∀ w : E × E, ‖w - z t‖ ≤ R / 2 → w ∈ ball p R := by
  obtain ⟨L, U, hU, hLip⟩ := hacc.exists_lipschitzOnWith
  obtain ⟨R, hR, hRU⟩ := Metric.mem_nhds_iff.mp hU
  obtain ⟨z, hz0, e, he, hz⟩ :=
    (contDiffAt_phaseField acc p hacc).exists_forall_mem_closedBall_exists_eq_forall_mem_Ioo_hasDerivAt₀ 0
  have hzcont : ContinuousAt z 0 := (hz 0 (by constructor <;> linarith)).continuousAt
  have hzsmall : ∀ᶠ t in 𝓝 (0 : ℝ), z t ∈ ball p (R / 2) := by
    simpa only [hz0] using hzcont.tendsto.eventually (ball_mem_nhds (z 0) (half_pos hR))
  have hevent : ∀ᶠ t in 𝓝 (0 : ℝ), HasDerivAt z (phaseField acc (z t)) t ∧
      z t ∈ ball p (R / 2) := by
    filter_upwards [Ioo_mem_nhds (by linarith : 0 - e < 0) (by linarith : 0 < 0 + e),
      hzsmall] with t ht hzt using ⟨hz t ht, hzt⟩
  obtain ⟨d, hd, hdsub⟩ := Metric.mem_nhds_iff.mp hevent
  have hinterval : ∀ t ∈ Icc (0 : ℝ) (d / 2),
      HasDerivAt z (phaseField acc (z t)) t ∧ z t ∈ ball p (R / 2) := by
    intro t ht
    apply hdsub
    simpa only [mem_ball, dist_zero_right, Real.norm_eq_abs, abs_of_nonneg ht.1] using
      ht.2.trans_lt (half_lt_self hd)
  have hmem : ∀ t ∈ Icc (0 : ℝ) (d / 2), z t ∈ ball p R :=
    fun t ht => ball_subset_ball (half_le_self hR.le) (hinterval t ht).2
  let B : ℝ≥0 := ⟨max 1 (L : ℝ) * R + ‖phaseField acc p‖, by positivity⟩
  refine ⟨d / 2, half_pos hd, R, hR, z, hz0, ?_, ?_⟩
  · refine ⟨⟨L, B, fun t ht => (hinterval t ht).1, hmem,
      fun t ht => phaseField_norm_le_on_ball acc p (z t) L R hR
        (hmem t ht) (hLip.mono hRU), ?_⟩⟩
    intro u hu v hv
    simpa only [dist_eq_norm] using hLip.dist_le_mul u (hRU hu) v (hRU hv)
  · intro t ht w hw
    have hzt := mem_ball_iff_norm.mp (hinterval t ht).2
    exact mem_ball_iff_norm.mpr
      ((norm_sub_le_norm_sub_add_norm_sub w (z t) p).trans_lt (by linarith))

/-- The actual raised-force minus connection acceleration for specified
ruler, gradient, Hessian, and net force-covector fields. The ambient inner
product identifies force vectors with covectors in this expression. -/
noncomputable def mechanicalAcceleration (a : ℝ) (rho : E → ℝ) (gradient : E → E)
    (hessian : E → E →L[ℝ] E) (force : E × E → E) (z : E × E) : E :=
  inverseAction (rho z.1) a (gradient z.1) (force z) -
    connection (rho z.1) a (gradient z.1) (hessian z.1) z.2 z.2

/-- Actual C1 coefficient fields give actual C1 mechanical acceleration
where the ruler is nonzero. The weight denominator is positive automatically. -/
theorem contDiffAt_mechanicalAcceleration (a : ℝ) (rho : E → ℝ) (gradient : E → E)
    (hessian : E → E →L[ℝ] E) (force : E × E → E) (p : E × E)
    (hr : ContDiffAt ℝ 1 rho p.1) (hu : ContDiffAt ℝ 1 gradient p.1)
    (hH : ContDiffAt ℝ 1 hessian p.1) (hF : ContDiffAt ℝ 1 force p)
    (hrho : rho p.1 ≠ 0) :
    ContDiffAt ℝ 1 (mechanicalAcceleration a rho gradient hessian force) p := by
  have hrp := hr.comp p contDiffAt_fst
  have hup := hu.comp p contDiffAt_fst
  have hHp := hH.comp p contDiffAt_fst
  have hW : ContDiffAt ℝ 1 (fun z : E × E => weight a (gradient z.1)) p :=
    contDiffAt_const.add (contDiffAt_const.mul (hup.inner ℝ hup))
  have hWne : weight a (gradient p.1) ≠ 0 := (weight_pos a (gradient p.1)).ne'
  unfold mechanicalAcceleration inverseAction unstretch connection
  refine ((hrp.pow 2).smul
    (hF.sub (((contDiffAt_const.mul (hup.inner ℝ hF)).div hW hWne).smul hup))).sub ?_
  have huv := hup.inner ℝ (contDiffAt_snd : ContDiffAt ℝ 1 (Prod.snd : E × E → E) p)
  have hHv := (hHp.clm_apply contDiffAt_snd).inner ℝ contDiffAt_snd
  exact ((((contDiffAt_const.mul hHv).div hW hWne).smul hup).sub
    ((huv.div hrp hrho).smul contDiffAt_snd)).sub
      ((huv.div hrp hrho).smul contDiffAt_snd) |>.add
        ((((contDiffAt_snd.inner ℝ contDiffAt_snd).add
          ((contDiffAt_const.mul huv).mul huv)).div (hrp.mul hW)
            (mul_ne_zero hrho hWne)).smul hup)

/-- The same actual coefficient assumptions construct a local mechanical
solution and numerical-analysis region, rather than postulating a reference. -/
theorem exists_local_mechanical_certificate [CompleteSpace E]
    (a : ℝ) (rho : E → ℝ) (gradient : E → E)
    (hessian : E → E →L[ℝ] E) (force : E × E → E) (p : E × E)
    (hr : ContDiffAt ℝ 1 rho p.1) (hu : ContDiffAt ℝ 1 gradient p.1)
    (hH : ContDiffAt ℝ 1 hessian p.1) (hF : ContDiffAt ℝ 1 force p)
    (hrho : rho p.1 ≠ 0) :
    ∃ T > (0 : ℝ), ∃ R > (0 : ℝ), ∃ z : ℝ → E × E,
      z 0 = p ∧
      Nonempty (UniformODECertificate (mechanicalAcceleration a rho gradient hessian force)
        z T (ball p R)) ∧
      ∀ t ∈ Icc (0 : ℝ) T, ∀ w : E × E, ‖w - z t‖ ≤ R / 2 → w ∈ ball p R := by
  exact exists_local_ODE_certificate _ p
    (contDiffAt_mechanicalAcceleration a rho gradient hessian force p hr hu hH hF hrho)

end FractalSync.ScaleMetric
