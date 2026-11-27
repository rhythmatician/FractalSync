import FractalSync.IntrinsicLength
import FractalSync.ZeroFloorLimit
import FractalSync.Analysis.ACVectorFTC
import FractalSync.Analysis.DerivativeControl
import Mathlib.Analysis.Normed.Lp.ProdLp
import Mathlib.Analysis.Calculus.Deriv.Prod

/-!
# The lifted graph chord and the actual intrinsic metric

An ordinary L2 product lift retains both displacement and ruler change. Its
small-chord lower bound recovers the full graph density along differentiable
curves, including where the ambient distance field is nonsmooth.
-/

namespace FractalSync.DistanceGraph

open MeasureTheory Set Filter
open scoped Topology NNReal

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

noncomputable def rulerGraphLift (S : Set E) (epsilon a : ℝ) (x : E) : WithLp 2 (E × ℝ) :=
  WithLp.toLp 2 (x, a * ruler epsilon (Metric.infDist x S))

theorem ac_rulerGraphLift (S : Set E) (epsilon a lo hi : ℝ) (gamma : ℝ → E)
    (he : 0 < epsilon) (hgamma : AbsolutelyContinuousOnInterval gamma lo hi) :
    AbsolutelyContinuousOnInterval (fun t => rulerGraphLift S epsilon a (gamma t)) lo hi := by
  exact (WithLp.prodContinuousLinearEquiv 2 ℝ E ℝ).symm.lipschitz.comp_absolutelyContinuousOnInterval
    (hgamma.prodMk ((ac_ruler_trace S gamma epsilon lo hi he hgamma).const_mul a))

theorem hasDerivAt_rulerGraphLift (S : Set E) (epsilon a : ℝ) (gamma : ℝ → E)
    (t : ℝ) (v : E) (w : ℝ) (hgamma : HasDerivAt gamma v t)
    (hrho : HasDerivAt (fun s => ruler epsilon (Metric.infDist (gamma s) S)) w t) :
    HasDerivAt (fun s => rulerGraphLift S epsilon a (gamma s))
      (WithLp.toLp 2 (v, a * w)) t := by
  exact (WithLp.prodContinuousLinearEquiv 2 ℝ E ℝ).symm.hasFDerivAt.comp_hasDerivAt t
    (hgamma.prodMk (hrho.const_mul a))

theorem norm_deriv_rulerGraphLift (S : Set E) (epsilon a : ℝ) (gamma : ℝ → E)
    (t : ℝ) (v : E) (w : ℝ) (hgamma : HasDerivAt gamma v t)
    (hrho : HasDerivAt (fun s => ruler epsilon (Metric.infDist (gamma s) S)) w t) :
    ‖deriv (fun s => rulerGraphLift S epsilon a (gamma s)) t‖ =
      Real.sqrt (‖v‖ ^ 2 + a ^ 2 * w ^ 2) := by
  rw [(hasDerivAt_rulerGraphLift S epsilon a gamma t v w hgamma hrho).deriv,
    WithLp.prod_norm_eq_of_L2]
  simp [Real.norm_eq_abs, mul_pow]

/-- The lifted velocity retains exactly the two terms in the graph density. -/
theorem ae_norm_deriv_rulerGraphLift [FiniteDimensional ℝ E]
    (S : Set E) (epsilon a lo hi : ℝ) (gamma : ℝ → E) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma lo hi) :
    ∀ᵐ t, t ∈ uIcc lo hi →
      ‖deriv (fun s => rulerGraphLift S epsilon a (gamma s)) t‖ =
        ruler epsilon (Metric.infDist (gamma t) S) *
          graphSpeed a epsilon (Metric.infDist (gamma t) S) ‖deriv gamma t‖
            (deriv (fun s => Metric.infDist (gamma s) S) t) := by
  filter_upwards [hgamma.ae_differentiableAt,
    (ac_distance_trace S gamma lo hi hgamma).ae_differentiableAt] with t ht hr hmem
  have hchain := (hasDerivAt_ruler epsilon (Metric.infDist (gamma t) S) he).comp t
    (hr hmem).hasDerivAt
  rw [norm_deriv_rulerGraphLift S epsilon a gamma t _ _ (ht hmem).hasDerivAt hchain,
    graphSpeed_eq_rulerSpeed epsilon a _ _ _ he]
  exact (mul_div_cancel₀ _ (ruler_pos epsilon _ he).ne').symm

theorem intrinsicDistance_le_length_prefix [FiniteDimensional ℝ E]
    (S : Set E) (epsilon a : ℝ) (gamma : ℝ → E) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) (t : ℝ)
    (ht : t ∈ uIcc (0 : ℝ) 1) :
    intrinsicDistance S epsilon a (gamma 0) (gamma t) ≤ acGraphLength S epsilon a gamma := by
  have ht' : t ∈ Icc (0 : ℝ) 1 := by simpa using ht
  exact (intrinsicDistance_le_subpath_integral S gamma epsilon a 0 t ht'.1
    (hgamma.mono (uIcc_subset_uIcc_left ht))).trans
      (intervalIntegral.integral_mono_interval le_rfl ht'.1 ht'.2
        (Eventually.of_forall fun s => graphSpeed_nonneg a epsilon
          (Metric.infDist (gamma s) S) ‖deriv gamma s‖ (deriv (fun u => Metric.infDist (gamma u) S) s))
        (ac_graphSpeed_intervalIntegrable S gamma epsilon a 0 1 he hgamma))

theorem ac_ruler_upper_of_length [FiniteDimensional ℝ E]
    (S : Set E) (epsilon a : ℝ) (gamma : ℝ → E) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) (t : ℝ)
    (ht : t ∈ uIcc (0 : ℝ) 1) :
    ruler epsilon (Metric.infDist (gamma t) S) ≤
      ruler epsilon (Metric.infDist (gamma 0) S) * Real.exp (acGraphLength S epsilon a gamma) := by
  exact (ruler_le_exp_intrinsicDistance S epsilon a (gamma 0) (gamma t) he).trans
    (mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr
      (intrinsicDistance_le_length_prefix S epsilon a gamma he hgamma t ht))
        (ruler_pos epsilon _ he).le)

/-- Every AC candidate bounds the ordinary chord of its lifted graph. -/
theorem rulerGraphChord_le_length [FiniteDimensional ℝ E]
    (S : Set E) (epsilon a : ℝ) (gamma : ℝ → E) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) :
    dist (rulerGraphLift S epsilon a (gamma 0)) (rulerGraphLift S epsilon a (gamma 1)) ≤
      ruler epsilon (Metric.infDist (gamma 0) S) * Real.exp (acGraphLength S epsilon a gamma) *
        acGraphLength S epsilon a gamma := by
  have hlift := ac_rulerGraphLift S epsilon a 0 1 gamma he hgamma
  rw [dist_comm, dist_eq_norm, ← hlift.integral_deriv_eq_sub_vector]
  apply (intervalIntegral.norm_integral_le_integral_norm (by norm_num : (0 : ℝ) ≤ 1)).trans
  change _ ≤ (ruler epsilon (Metric.infDist (gamma 0) S) * Real.exp (acGraphLength S epsilon a gamma)) *
    ∫ t in (0 : ℝ)..1, graphSpeed a epsilon (Metric.infDist (gamma t) S)
      ‖deriv gamma t‖ (deriv (fun s => Metric.infDist (gamma s) S) t)
  rw [← intervalIntegral.integral_const_mul]
  apply intervalIntegral.integral_mono_ae_restrict (by norm_num)
    hlift.intervalIntegrable_deriv_vector.norm
    ((ac_graphSpeed_intervalIntegrable S gamma epsilon a 0 1 he hgamma).const_mul _)
  exact (ae_restrict_iff' measurableSet_Icc).2
    ((ae_norm_deriv_rulerGraphLift S epsilon a 0 1 gamma he hgamma).mono fun t h ht =>
      (h (by simpa using ht)).le.trans (mul_le_mul_of_nonneg_right
        (ac_ruler_upper_of_length S epsilon a gamma he hgamma t (by simpa using ht))
        (graphSpeed_nonneg a epsilon _ _ _)))

/-- The full lifted chord is controlled by the actual path infimum. The
coefficient tends to the local ruler as the endpoint distance tends to zero. -/
theorem rulerGraphChord_le_intrinsicDistance [FiniteDimensional ℝ E]
    (S : Set E) (epsilon a : ℝ) (x y : E) (he : 0 < epsilon) :
    dist (rulerGraphLift S epsilon a x) (rulerGraphLift S epsilon a y) ≤
      ruler epsilon (Metric.infDist x S) * Real.exp (intrinsicDistance S epsilon a x y) *
        intrinsicDistance S epsilon a x y := by
  have hclosed : IsClosed {L : ℝ |
      dist (rulerGraphLift S epsilon a x) (rulerGraphLift S epsilon a y) ≤
        ruler epsilon (Metric.infDist x S) * Real.exp L * L} :=
    isClosed_le continuous_const (by fun_prop)
  apply hclosed.closure_subset_iff.mpr _
    (csInf_mem_closure (pairwisePathLengths_nonempty S epsilon a x y)
      (pairwisePathLengths_bddBelow S epsilon a x y))
  rintro L ⟨gamma, hgamma, hstart, hend, rfl⟩
  simpa only [mem_setOf_eq, hstart, hend] using
    rulerGraphChord_le_length S epsilon a gamma he hgamma

/-- A differentiable scalar control on intrinsic increments controls the
full lifted derivative with the exact local ruler coefficient. -/
theorem norm_deriv_rulerGraphLift_le_of_control [FiniteDimensional ℝ E]
    (S : Set E) (epsilon a : ℝ) (gamma : ℝ → E) (A : ℝ → ℝ) (t : ℝ)
    (v : WithLp 2 (E × ℝ)) (w : ℝ) (he : 0 < epsilon)
    (hgamma : HasDerivAt (fun s => rulerGraphLift S epsilon a (gamma s)) v t)
    (hA : HasDerivAt A w t)
    (hcontrol : ∀ᶠ s in 𝓝 t, intrinsicDistance S epsilon a (gamma t) (gamma s) ≤ |A s - A t|) :
    ‖v‖ ≤ ruler epsilon (Metric.infDist (gamma t) S) * |w| := by
  apply FractalSync.Analysis.norm_deriv_le_exp_control hgamma hA
  filter_upwards [hcontrol] with s hs
  have hc := rulerGraphChord_le_intrinsicDistance S epsilon a (gamma t) (gamma s) he
  rw [dist_comm, dist_eq_norm] at hc
  exact hc.trans (mul_le_mul
    (mul_le_mul_of_nonneg_left (Real.exp_le_exp.mpr hs) (ruler_pos epsilon _ he).le) hs
      (intrinsicDistance_nonneg S epsilon a (gamma t) (gamma s))
      (mul_nonneg (ruler_pos epsilon _ he).le (Real.exp_pos _).le))

/-- An AC scalar control of all intrinsic increments bounds the actual graph
speed almost everywhere. No ambient differentiability of the Shore field is used. -/
theorem ac_graphSpeed_le_control_deriv [FiniteDimensional ℝ E]
    (S : Set E) (epsilon a : ℝ) (gamma : ℝ → E) (A : ℝ → ℝ) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1)
    (hA : AbsolutelyContinuousOnInterval A 0 1)
    (hcontrol : ∀ s ∈ uIcc (0 : ℝ) 1, ∀ t ∈ uIcc (0 : ℝ) 1,
      intrinsicDistance S epsilon a (gamma s) (gamma t) ≤ |A s - A t|) :
    ∀ᵐ t, t ∈ uIcc (0 : ℝ) 1 →
      graphSpeed a epsilon (Metric.infDist (gamma t) S) ‖deriv gamma t‖
        (deriv (fun s => Metric.infDist (gamma s) S) t) ≤ |deriv A t| := by
  filter_upwards [ae_norm_deriv_rulerGraphLift S epsilon a 0 1 gamma he hgamma,
    (ac_rulerGraphLift S epsilon a 0 1 gamma he hgamma).ae_differentiableAt,
    hA.ae_differentiableAt, show ∀ᵐ t : ℝ, t ≠ 0 from by simp [ae_iff],
    show ∀ᵐ t : ℝ, t ≠ 1 from by simp [ae_iff]] with t hn hg hAd h0 h1 ht
  have ht' : t ∈ Icc (0 : ℝ) 1 := by simpa using ht
  have hti : t ∈ Ioo (0 : ℝ) 1 :=
    ⟨lt_of_le_of_ne ht'.1 h0.symm, lt_of_le_of_ne ht'.2 h1⟩
  have hlocal : ∀ᶠ s in 𝓝 t,
      intrinsicDistance S epsilon a (gamma t) (gamma s) ≤ |A s - A t| := by
    filter_upwards [isOpen_Ioo.mem_nhds hti] with s hs
    simpa only [abs_sub_comm] using hcontrol t ht s (by simpa using Ioo_subset_Icc_self hs)
  have hb := norm_deriv_rulerGraphLift_le_of_control S epsilon a gamma A t _ _ he
    (hg ht).hasDerivAt (hAd ht).hasDerivAt hlocal
  rw [hn ht] at hb
  exact (mul_le_mul_iff_right₀ (ruler_pos epsilon _ he)).mp hb

end FractalSync.DistanceGraph
