import FractalSync.MetricConnectionUniqueness
import FractalSync.EmbeddedNormalPatch
import FractalSync.NonlinearCoordinates
import Mathlib.Analysis.Calculus.Gradient.Basic

/-!
# Ambient graph connection in actual normal coordinates

Metric equality on a neighborhood is differentiated before applying
Levi-Civita uniqueness. The ambient gradient and its derivative are actual
Fréchet jets; no connection transformation identity is supplied as a premise.
-/

namespace FractalSync.EmbeddedShoreConnection

open EmbeddedShoreMetric EmbeddedNormalPatch TubularNeighborhood
open scoped Topology RealInnerProductSpace

variable {E : Type*} [NormedAddCommGroup E] [InnerProductSpace ℝ E]

/-- The compact graph connection uses the actual derivative of its gradient field. -/
noncomputable def graphConnection (rho : E → ℝ) (a : ℝ) (u : E → E)
    (x v w : E) : E :=
  ScaleMetric.connection (rho x) a (u x) (fderiv ℝ u x) v w

/-- The corresponding rank-one metric, written in the same ambient frame. -/
noncomputable def graphMetric (rho : E → ℝ) (a : ℝ) (u : E → E)
    (x v w : E) : ℝ := ScaleMetric.metric (rho x) a (u x) v w

/-- An actual C¹ gradient on a neighborhood makes the scalar field C². -/
theorem contDiffAt_of_actual_gradient (rho : E → ℝ) (u : E → E) (x : E)
    (hgrad : ∀ᶠ y in 𝓝 x, HasFDerivAt rho (innerSL ℝ (u y)) y)
    (hu : ContDiffAt ℝ 1 u x) : ContDiffAt ℝ 2 rho x := by
  exact (contDiffAt_succ_iff_hasFDerivAt (n := 1)).2
    ⟨fun y => innerSL ℝ (u y), Filter.eventually_iff_exists_mem.mp hgrad,
      (innerSL ℝ).contDiff.contDiffAt.comp x hu⟩

/-- The derivative of an actual C¹ gradient is symmetric; Hessian symmetry
is derived from calculus instead of being supplied as a provider contract. -/
theorem gradient_derivative_symm (rho : E → ℝ) (u : E → E) (x p q : E)
    (hgrad : ∀ᶠ y in 𝓝 x, HasFDerivAt rho (innerSL ℝ (u y)) y)
    (hu : ContDiffAt ℝ 1 u x) :
    inner ℝ (fderiv ℝ u x p) q = inner ℝ (fderiv ℝ u x q) p := by
  have heq : fderiv ℝ rho =ᶠ[𝓝 x] (fun y => innerSL ℝ (u y)) :=
    hgrad.mono fun _ hy => hy.fderiv
  have hd : fderiv ℝ (fderiv ℝ rho) x = (innerSL ℝ).comp (fderiv ℝ u x) :=
    heq.fderiv_eq.trans ((innerSL ℝ).hasFDerivAt.comp x
      (hu.differentiableAt le_rfl).hasFDerivAt).fderiv
  simpa only [hd, ContinuousLinearMap.comp_apply, innerSL_apply] using
    ((contDiffAt_of_actual_gradient rho u x hgrad hu).isSymmSndFDerivAt (by norm_num)).eq p q

/-- Actual C¹ regularity of the compact connection evaluated on varying
vector fields follows from a C² gradient and a nonzero C¹ ruler. -/
theorem contDiffAt_graphConnection_fields (rho : E → ℝ) (a : ℝ)
    (u X Y : E → E) (x : E) (hr : ContDiffAt ℝ 1 rho x)
    (hu : ContDiffAt ℝ 2 u x) (hX : ContDiffAt ℝ 1 X x) (hY : ContDiffAt ℝ 1 Y x)
    (hrho : rho x ≠ 0) :
    ContDiffAt ℝ 1 (fun y => graphConnection rho a u y (X y) (Y y)) x := by
  have hu1 : ContDiffAt ℝ 1 u x := hu.of_le one_le_two
  have hW : ContDiffAt ℝ 1 (fun y => ScaleMetric.weight a (u y)) x :=
    contDiffAt_const.add (contDiffAt_const.mul (hu1.inner ℝ hu1))
  have hHXY := ((hu.fderiv_right (m := 1) (by norm_num)).clm_apply hX).inner ℝ hY
  exact ((((contDiffAt_const.mul hHXY).div hW (ScaleMetric.weight_pos a (u x)).ne').smul hu1).sub
    (((hu1.inner ℝ hY).div hr hrho).smul hX)).sub
      (((hu1.inner ℝ hX).div hr hrho).smul hY) |>.add
        ((((hX.inner ℝ hY).add ((contDiffAt_const.mul (hu1.inner ℝ hX)).mul
          (hu1.inner ℝ hY))).div (hr.mul hW)
            (mul_ne_zero hrho (ScaleMetric.weight_pos a (u x)).ne')).smul hu1)

/-- Actual differentiation of a pulled-back metric pairing retains both
coordinate-Hessian contributions. -/
theorem hasDerivAt_pullback_pairing (f : Coord → E) (rho : E → ℝ)
    (u : E → E) (a t : ℝ) (gamma : ℝ → Coord) (d v w : Coord)
    (H : E →L[ℝ] E) (hf : ContDiffAt ℝ 2 f (gamma t))
    (hg : HasDerivAt gamma d t)
    (hr : HasFDerivAt rho (innerSL ℝ (u (f (gamma t)))) (f (gamma t)))
    (hu : HasFDerivAt u H (f (gamma t))) (hrho : rho (f (gamma t)) ≠ 0) :
    HasDerivAt
      (fun s => NonlinearCoordinates.pullbackMetric f (graphMetric rho a u) (gamma s) v w)
      (ScaleMetric.metricDifferential (rho (f (gamma t))) a (u (f (gamma t))) H
          (fderiv ℝ f (gamma t) d) (fderiv ℝ f (gamma t) v) (fderiv ℝ f (gamma t) w) +
        ScaleMetric.metric (rho (f (gamma t))) a (u (f (gamma t)))
          (fderiv ℝ (fderiv ℝ f) (gamma t) d v) (fderiv ℝ f (gamma t) w) +
        ScaleMetric.metric (rho (f (gamma t))) a (u (f (gamma t)))
          (fderiv ℝ f (gamma t) v) (fderiv ℝ (fderiv ℝ f) (gamma t) d w)) t := by
  have hfline := (hf.differentiableAt one_le_two).hasFDerivAt.comp_hasDerivAt t hg
  have hJline := ((hf.fderiv_right (m := 1) (by norm_num)).differentiableAt le_rfl).hasFDerivAt.comp_hasDerivAt t hg
  simpa only [NonlinearCoordinates.pullbackMetric, graphMetric, Function.comp_apply, map_zero, add_zero] using
    ScaleMetric.hasDerivAt_metric_pairing a t (rho ∘ f ∘ gamma) (u ∘ f ∘ gamma)
      (fun s => fderiv ℝ f (gamma s) v) (fun s => fderiv ℝ f (gamma s) w)
      (fderiv ℝ f (gamma t) d) (fderiv ℝ (fderiv ℝ f) (gamma t) d v)
      (fderiv ℝ (fderiv ℝ f) (gamma t) d w) H
      (hr.comp_hasDerivAt t hfline) (hu.comp_hasDerivAt t hfline)
      (by simpa using hJline.clm_apply (hasDerivAt_const t v))
      (by simpa using hJline.clm_apply (hasDerivAt_const t w)) hrho

/-- The chart-Hessian contributions combine with the compatible ambient
connection into the pulled-back metric pairing. -/
theorem pullback_rate_eq (f : Coord → E) (rho : E → ℝ) (u : E → E)
    (a : ℝ) (x d v w : Coord) (hi : (fderiv ℝ f x).IsInvertible)
    (hr : rho (f x) ≠ 0) :
    ScaleMetric.metricDifferential (rho (f x)) a (u (f x)) (fderiv ℝ u (f x))
          (fderiv ℝ f x d) (fderiv ℝ f x v) (fderiv ℝ f x w) +
        ScaleMetric.metric (rho (f x)) a (u (f x))
          (fderiv ℝ (fderiv ℝ f) x d v) (fderiv ℝ f x w) +
        ScaleMetric.metric (rho (f x)) a (u (f x))
          (fderiv ℝ f x v) (fderiv ℝ (fderiv ℝ f) x d w) =
      NonlinearCoordinates.pullbackMetric f (graphMetric rho a u) x
        (NonlinearCoordinates.pullbackConnection f (graphConnection rho a u) x d v) w +
      NonlinearCoordinates.pullbackMetric f (graphMetric rho a u) x v
        (NonlinearCoordinates.pullbackConnection f (graphConnection rho a u) x d w) := by
  rcases hi with ⟨M, hM⟩
  simp only [NonlinearCoordinates.pullbackMetric, NonlinearCoordinates.pullbackConnection, graphMetric, graphConnection, ← hM, ContinuousLinearMap.inverse_equiv, ContinuousLinearEquiv.coe_coe, ContinuousLinearEquiv.apply_symm_apply]
  rw [ScaleMetric.connection_metric_compatible _ _ _ _ _ _ _ hr, ScaleMetric.metric_add_left, ScaleMetric.metric_add_right]
  ring

/-- Torsion freedom of the actual pullback uses symmetry of the coordinate
Hessian and of the actual ambient gradient derivative. -/
theorem pullbackConnection_symm (f : Coord → E) (rho : E → ℝ)
    (u : E → E) (a : ℝ) (x v w : Coord) (hf : ContDiffAt ℝ 2 f x)
    (hH : ∀ p q, inner ℝ (fderiv ℝ u (f x) p) q = inner ℝ (fderiv ℝ u (f x) q) p) :
    NonlinearCoordinates.pullbackConnection f (graphConnection rho a u) x v w =
      NonlinearCoordinates.pullbackConnection f (graphConnection rho a u) x w v := by
  simp only [NonlinearCoordinates.pullbackConnection, graphConnection,
    (hf.isSymmSndFDerivAt (by norm_num)).eq v w,
    ScaleMetric.connection_symm (rho (f x)) a (u (f x)) (fderiv ℝ u (f x)) hH
      (fderiv ℝ f x v) (fderiv ℝ f x w)]

/-- A local metric identification determines the connection transformation.
The metric equality is a germ, so its directional derivatives may be compared. -/
theorem pullbackConnection_eq_curved_of_metric_germ
    (f : Coord → E) (rho : E → ℝ) (u : E → E) (epsilon a : ℝ) (k : ℝ → ℝ)
    (x v w : Coord) (hf : ContDiffAt ℝ 2 f x)
    (hi : (fderiv ℝ f x).IsInvertible)
    (hr : HasFDerivAt rho (innerSL ℝ (u (f x))) (f x))
    (hu : DifferentiableAt ℝ u (f x)) (hrho : rho (f x) ≠ 0)
    (hH : ∀ p q, inner ℝ (fderiv ℝ u (f x) p) q = inner ℝ (fderiv ℝ u (f x) q) p)
    (he : 0 < epsilon) (hh : 1 - k x.2 * x.1 ≠ 0)
    (hk : DifferentiableAt ℝ k x.2)
    (hmetric : ∀ᶠ y in 𝓝 x, ∀ p q,
      NonlinearCoordinates.pullbackMetric f (graphMetric rho a u) y p q =
        CurvedShoreGeometry.metric epsilon a (k y.2) y.1 p q) :
    NonlinearCoordinates.pullbackConnection f (graphConnection rho a u) x v w =
      CurvedShoreGeometry.christoffel epsilon a k x v w := by
  refine CurvedShoreGeometry.christoffel_eq_of_compatible epsilon a k x
    (NonlinearCoordinates.pullbackConnection f (graphConnection rho a u) x) he hh
    (fun p q => pullbackConnection_symm f rho u a x p q hf hH) ?_ v w
  intro d p q
  have hline : HasDerivAt (fun t : ℝ => x + t • d) d 0 := by
    simpa only [one_smul, zero_add, Pi.add_apply] using (hasDerivAt_const (0 : ℝ) x).add ((hasDerivAt_id (0 : ℝ)).smul_const d)
  have hactual := hasDerivAt_pullback_pairing f rho u a 0 (fun t : ℝ => x + t • d) d p q
    (fderiv ℝ u (f x)) (by simpa only [zero_smul, add_zero] using hf) hline
    (by simpa only [zero_smul, add_zero] using hr)
    (by simpa only [zero_smul, add_zero] using hu.hasFDerivAt)
    (by simpa only [zero_smul, add_zero] using hrho)
  have hcoord := CurvedShoreGeometry.hasDerivAt_metric_pairing epsilon a 0 k
    (fun t : ℝ => x + t • d) (fun _ => p) (fun _ => q) d 0 0 he hline
    (hasDerivAt_const 0 p) (hasDerivAt_const 0 q)
    (by simpa only [zero_smul, add_zero] using hk)
  have heq : (fun t : ℝ => NonlinearCoordinates.pullbackMetric f (graphMetric rho a u) (x + t • d) p q) =ᶠ[𝓝 (0 : ℝ)]
      (fun t : ℝ => CurvedShoreGeometry.metric epsilon a (k (x + t • d).2) (x + t • d).1 p q) :=
    ((show Filter.Tendsto (fun t : ℝ => x + t • d) (𝓝 0) (𝓝 x) from
      by simpa only [zero_smul, add_zero] using hline.continuousAt.tendsto).eventually hmetric).mono
        (fun t ht => ht p q)
  rw [← hmetric.self_of_nhds _ q, ← hmetric.self_of_nhds p _, ← pullback_rate_eq f rho u a x d p q hi hrho]
  simpa only [zero_smul, add_zero, CurvedShoreGeometry.metric, Prod.fst_zero, Prod.snd_zero, mul_zero, zero_mul] using
    hcoord.unique (hactual.congr_of_eventuallyEq heq.symm)

/-- The ambient actual-gradient metric pulls back to the already-defined
normal graph metric by the actual chain rule. -/
theorem pullbackMetric_eq_normalGraphMetric (S : Set E) (epsilon a : ℝ)
    (C N : ℝ → E) (u : E → E) (x v w : Coord)
    (hf : DifferentiableAt ℝ (normalMap C N) x)
    (hr : HasFDerivAt (fun y => ruler epsilon (Metric.infDist y S))
      (innerSL ℝ (u (normalMap C N x))) (normalMap C N x)) :
    NonlinearCoordinates.pullbackMetric (normalMap C N)
        (graphMetric (fun y => ruler epsilon (Metric.infDist y S)) a u) x v w =
      normalGraphMetric S epsilon a C N x v w := by
  have hchain := (hr.comp x hf.hasFDerivAt).fderiv
  change fderiv ℝ (normalRuler S epsilon C N) x = _ at hchain
  simp only [NonlinearCoordinates.pullbackMetric, graphMetric, ScaleMetric.metric,
    normalGraphMetric, hchain, ContinuousLinearMap.comp_apply, innerSL_apply, normalRuler]

/-- The local geometric certificate and actual Frenet data identify the
pulled-back ambient metric as a germ, not merely at one point. -/
theorem eventually_pullbackMetric_eq_curved (S : Set E) (epsilon a R : ℝ)
    (C N T : ℝ → E) (k : ℝ → ℝ) (V : Set ℝ) (u : E → E) (x : Coord)
    (hpatch : NormalPatch S R C N V) (hV : V ∈ 𝓝 x.2)
    (he : 0 < epsilon) (hr : |x.1| < R)
    (hf : ContinuousAt (normalMap C N) x)
    (hgrad : ∀ᶠ y in 𝓝 (normalMap C N x),
      HasFDerivAt (fun z => ruler epsilon (Metric.infDist z S)) (innerSL ℝ (u y)) y)
    (hF : ∀ᶠ s in 𝓝 x.2,
      HasDerivAt C (T s) s ∧ HasDerivAt N (-(k s) • T s) s ∧
        ‖T s‖ = 1 ∧ inner ℝ (N s) (T s) = 0) :
    ∀ᶠ y in 𝓝 x, ∀ v w,
      NonlinearCoordinates.pullbackMetric (normalMap C N)
        (graphMetric (fun z => ruler epsilon (Metric.infDist z S)) a u) y v w =
          CurvedShoreGeometry.metric epsilon a (k y.2) y.1 v w := by
  filter_upwards [(continuous_snd.tendsto x).eventually (eventually_mem_nhds_iff.2 hV),
    (continuous_fst.abs.continuousAt).eventually_lt_const hr,
    (continuous_snd.tendsto x).eventually hF, hf.tendsto.eventually hgrad]
      with y hyV hyr hyF hyg
  intro v w
  rw [pullbackMetric_eq_normalGraphMetric S epsilon a C N u y v w
    ((hasFDerivAt_normalMap C N y (T y.2) (k y.2) hyF.1 hyF.2.1).differentiableAt) hyg]
  exact EmbeddedNormalPatch.normalGraphMetric_eq S epsilon a R C N V hpatch y v w
    (T y.2) (k y.2) hyV he hyr hyF.1 hyF.2.1 hyF.2.2.1 hyF.2.2.2

/-- Actual closed-set geometry and local Frenet calculus identify the
ambient compact connection with the curved normal-coordinate coefficients. -/
theorem normalMap_connection (S : Set E) (epsilon a R : ℝ)
    (C N T : ℝ → E) (k : ℝ → ℝ) (V : Set ℝ) (u : E → E) (x v w : Coord)
    (hpatch : NormalPatch S R C N V) (hV : V ∈ 𝓝 x.2)
    (he : 0 < epsilon) (hr : |x.1| < R) (hh : 1 - k x.2 * x.1 ≠ 0)
    (hf : ContDiffAt ℝ 2 (normalMap C N) x)
    (hi : (fderiv ℝ (normalMap C N) x).IsInvertible)
    (hk : DifferentiableAt ℝ k x.2)
    (hgrad : ∀ᶠ y in 𝓝 (normalMap C N x),
      HasFDerivAt (fun z => ruler epsilon (Metric.infDist z S)) (innerSL ℝ (u y)) y)
    (hu : DifferentiableAt ℝ u (normalMap C N x))
    (hH : ∀ p q, inner ℝ (fderiv ℝ u (normalMap C N x) p) q =
      inner ℝ (fderiv ℝ u (normalMap C N x) q) p)
    (hF : ∀ᶠ s in 𝓝 x.2,
      HasDerivAt C (T s) s ∧ HasDerivAt N (-(k s) • T s) s ∧
        ‖T s‖ = 1 ∧ inner ℝ (N s) (T s) = 0) :
    NonlinearCoordinates.pullbackConnection (normalMap C N)
      (graphConnection (fun y => ruler epsilon (Metric.infDist y S)) a u) x v w =
        CurvedShoreGeometry.christoffel epsilon a k x v w := by
  exact pullbackConnection_eq_curved_of_metric_germ (normalMap C N)
    (fun y => ruler epsilon (Metric.infDist y S)) u epsilon a k x v w hf hi
    hgrad.self_of_nhds hu (ruler_pos epsilon _ he).ne' hH he hh hk
    (eventually_pullbackMetric_eq_curved S epsilon a R C N T k V u x hpatch hV
      he hr hf.continuousAt hgrad hF)

/-- A smooth inverse normal chart gives the actual regularized set distance the same regularity,
including at the unsigned-distance cusp on Shore. -/
theorem contDiffAt_ambientRuler {n : WithTop ℕ∞} (S : Set E) (epsilon R : ℝ)
    (C N : ℝ → E) (V : Set ℝ) (hpatch : NormalPatch S R C N V)
    (e : OpenPartialHomeomorph Coord E) (heq : (e : Coord → E) = normalMap C N)
    (x : Coord) (hx : x ∈ e.source) (hV : V ∈ 𝓝 x.2)
    (hr : |x.1| < R) (he : 0 < epsilon) (hInv : ContDiffAt ℝ n e.symm (e x)) :
    ContDiffAt ℝ n (fun y => ruler epsilon (Metric.infDist y S)) (e x) := by
  have hsigned : ContDiffAt ℝ n (fun y => ruler epsilon (e.symm y).1) (e x) :=
    ((hInv.fst.pow 2).add contDiffAt_const).sqrt
      (by dsimp; nlinarith [sq_nonneg (e.symm (e x)).1])
  apply hsigned.congr_of_eventuallyEq
  filter_upwards [EmbeddedNormalPatch.eventually_infDist_normalChart S R C N V hpatch e heq x hx hV hr] with y hy
  simp only [hy, ruler, sq_abs]

/-- The inverse function theorem supplies the smooth inverse chart needed for
actual ambient ruler regularity. No differentiability of distance is assumed. -/
theorem contDiffAt_ambientRuler_of_normalMap {n : WithTop ℕ∞} (S : Set E) (epsilon R : ℝ)
    (C N : ℝ → E) (V : Set ℝ) (hpatch : NormalPatch S R C N V)
    (x : Coord) (hV : V ∈ 𝓝 x.2) (hr : |x.1| < R) (he : 0 < epsilon)
    (hn : 1 ≤ n) (hf : ContDiffAt ℝ n (normalMap C N) x)
    (hi : (fderiv ℝ (normalMap C N) x).IsInvertible) :
    ContDiffAt ℝ n (fun y => ruler epsilon (Metric.infDist y S)) (normalMap C N x) := by
  rcases hi with ⟨B, hB⟩
  have hd : HasFDerivAt (normalMap C N) (B : Coord →L[ℝ] E) x :=
    hB ▸ (hf.differentiableAt hn).hasFDerivAt
  exact contDiffAt_ambientRuler S epsilon R C N V hpatch
    (hf.toOpenPartialHomeomorph (normalMap C N) hd hn) rfl x
    (hf.mem_toOpenPartialHomeomorph_source hd hn) hV hr he
    (hf.to_localInverse hd hn)

section ActualGradient

variable [CompleteSpace E]

/-- Mathlib's actual Riesz gradient is C¹ wherever the scalar field is C². -/
theorem contDiffAt_gradient (rho : E → ℝ) (x : E)
    (hrho : ContDiffAt ℝ 2 rho x) : ContDiffAt ℝ 1 (gradient rho) x := by
  exact (InnerProductSpace.toDual ℝ E).symm.toContinuousLinearEquiv.contDiff.contDiffAt.comp x
    (hrho.fderiv_right (m := 1) (by norm_num))

/-- The actual gradient supplies the exact Fréchet derivative on the scalar
field's differentiability region. -/
theorem hasFDerivAt_inner_gradient (rho : E → ℝ) (x : E)
    (hrho : DifferentiableAt ℝ rho x) :
    HasFDerivAt rho (innerSL ℝ (gradient rho x)) x := by
  exact hrho.hasGradientAt.hasFDerivAt

/-- A C³ scalar ruler gives an actually C¹ covariant derivative for a C¹
direction field and a C² differentiated field. -/
theorem contDiffAt_gradientConnection_covariantDerivative (rho : E → ℝ) (a : ℝ)
    (X Y : E → E) (x : E) (hr : ContDiffAt ℝ 3 rho x)
    (hX : ContDiffAt ℝ 1 X x) (hY : ContDiffAt ℝ 2 Y x) (hrho : rho x ≠ 0) :
    ContDiffAt ℝ 1
      (CoordinateCovariance.covariantDerivative (graphConnection rho a (gradient rho)) X Y) x := by
  have hgrad : ContDiffAt ℝ 2 (gradient rho) x :=
    (InnerProductSpace.toDual ℝ E).symm.toContinuousLinearEquiv.contDiff.contDiffAt.comp x
      (hr.fderiv_right (m := 2) (by norm_num))
  exact ((hY.fderiv_right (m := 1) (by norm_num)).clm_apply hX).add
    (contDiffAt_graphConnection_fields rho a (gradient rho) X Y x (hr.of_le (by norm_num))
      hgrad hX (hY.of_le one_le_two) hrho)

/-- The ambient metric connection of actual regularized distance, using
Mathlib's gradient and its actual Fréchet derivative. -/
noncomputable def ambientConnection (S : Set E) (epsilon a : ℝ) : E → E → E → E :=
  graphConnection (fun y => ruler epsilon (Metric.infDist y S)) a
    (gradient (fun y => ruler epsilon (Metric.infDist y S)))

/-- The actual ambient connection has the curved normal-coordinate
coefficients. Actual ruler smoothness, gradient regularity, and Hessian
symmetry are derived from the normal geometry and inverse function theorem. -/
theorem ambientConnection_normalMap (S : Set E) (epsilon a R : ℝ)
    (C N T : ℝ → E) (k : ℝ → ℝ) (V : Set ℝ) (x v w : Coord)
    (hpatch : NormalPatch S R C N V) (hV : V ∈ 𝓝 x.2)
    (he : 0 < epsilon) (hr : |x.1| < R) (hh : 1 - k x.2 * x.1 ≠ 0)
    (hf : ContDiffAt ℝ 2 (normalMap C N) x)
    (hi : (fderiv ℝ (normalMap C N) x).IsInvertible)
    (hk : DifferentiableAt ℝ k x.2)
    (hF : ∀ᶠ s in 𝓝 x.2,
      HasDerivAt C (T s) s ∧ HasDerivAt N (-(k s) • T s) s ∧
        ‖T s‖ = 1 ∧ inner ℝ (N s) (T s) = 0) :
    NonlinearCoordinates.pullbackConnection (normalMap C N) (ambientConnection S epsilon a)
      x v w = CurvedShoreGeometry.christoffel epsilon a k x v w := by
  have hR := contDiffAt_ambientRuler_of_normalMap S epsilon R C N V hpatch x hV hr he one_le_two hf hi
  have hgrad : ∀ᶠ y in 𝓝 (normalMap C N x),
      HasFDerivAt (fun z => ruler epsilon (Metric.infDist z S))
        (innerSL ℝ (gradient (fun z => ruler epsilon (Metric.infDist z S)) y)) y :=
    (hR.eventually (by norm_num)).mono fun y hy =>
      hasFDerivAt_inner_gradient _ y (hy.differentiableAt one_le_two)
  have hG := contDiffAt_gradient (fun z => ruler epsilon (Metric.infDist z S)) (normalMap C N x) hR
  exact normalMap_connection S epsilon a R C N T k V
    (gradient (fun z => ruler epsilon (Metric.infDist z S))) x v w hpatch hV he hr hh hf hi hk
    hgrad (hG.differentiableAt le_rfl)
    (fun p q => gradient_derivative_symm _ _ _ p q hgrad hG) hF

/-- An actual tangent/normal frame and the nonfocal factor discharge the
Jacobian-invertibility premise of the ambient connection theorem. -/
theorem ambientConnection_normalFrame (S : Set E) (epsilon a R : ℝ)
    (C N T : ℝ → E) (k : ℝ → ℝ) (V : Set ℝ) (x v w : Coord)
    (hpatch : NormalPatch S R C N V) (hV : V ∈ 𝓝 x.2)
    (he : 0 < epsilon) (hr : |x.1| < R) (hh : 1 - k x.2 * x.1 ≠ 0)
    (hC : ContDiffAt ℝ 2 C x.2) (hN : ContDiffAt ℝ 2 N x.2)
    (B : Coord ≃L[ℝ] E) (hn : N x.2 = B (1, 0)) (ht : T x.2 = B (0, 1))
    (hk : DifferentiableAt ℝ k x.2)
    (hF : ∀ᶠ s in 𝓝 x.2,
      HasDerivAt C (T s) s ∧ HasDerivAt N (-(k s) • T s) s ∧
        ‖T s‖ = 1 ∧ inner ℝ (N s) (T s) = 0) :
    NonlinearCoordinates.pullbackConnection (normalMap C N) (ambientConnection S epsilon a)
      x v w = CurvedShoreGeometry.christoffel epsilon a k x v w := by
  have hf : ContDiffAt ℝ 2 (normalMap C N) x :=
    (hC.comp x contDiffAt_snd).add (contDiffAt_fst.smul (hN.comp x contDiffAt_snd))
  have hd := hasFDerivAt_normalMap_frame C N x B (k x.2) hh hn
    (by simpa only [ht] using hF.self_of_nhds.1)
    (by simpa only [ht] using hF.self_of_nhds.2.1)
  exact ambientConnection_normalMap S epsilon a R C N T k V x v w hpatch hV he hr hh hf
    ⟨normalFrameEquiv B (1 - k x.2 * x.1) hh, hd.fderiv.symm⟩ hk hF

end ActualGradient

end FractalSync.EmbeddedShoreConnection
