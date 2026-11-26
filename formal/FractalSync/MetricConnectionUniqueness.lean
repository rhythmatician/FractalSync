import FractalSync.ContinuousMechanics
import FractalSync.CurvedShoreGeometry

/-!
# Local uniqueness of the metric connection

The uniqueness argument needs only pointwise bilinearity, positive definiteness,
symmetry, torsion freedom, and a shared actual metric differential. It does not
assume a connection transformation law.
-/

namespace FractalSync.MetricConnectionUniqueness

variable {E : Type*} [AddCommGroup E] [Module ℝ E]

/-- Torsion-free metric compatibility implies the first-kind Koszul identity. -/
theorem koszul (g : LinearMap.BilinForm ℝ E) (D : E → E → E → ℝ)
    (Gamma : E → E → E) (hs : ∀ v w, g v w = g w v)
    (hT : ∀ v w, Gamma v w = Gamma w v)
    (hM : ∀ d v w, D d v w = g (Gamma d v) w + g v (Gamma d w))
    (v w z : E) :
    2 * g (Gamma v w) z = D v w z + D w v z - D z v w := by
  rw [hM v w z, hM w v z, hM z v w]
  rw [hT w v, hT z v, hT z w, hs w (Gamma v z)]
  ring

/-- A positive symmetric metric has at most one torsion-free compatible
connection value. Both candidates use the same metric derivative. -/
theorem eq_of_compatible (g : LinearMap.BilinForm ℝ E) (D : E → E → E → ℝ)
    (Gamma Lambda : E → E → E) (hs : ∀ v w, g v w = g w v)
    (hp : ∀ v, v ≠ 0 → 0 < g v v)
    (hTG : ∀ v w, Gamma v w = Gamma w v)
    (hTL : ∀ v w, Lambda v w = Lambda w v)
    (hMG : ∀ d v w, D d v w = g (Gamma d v) w + g v (Gamma d w))
    (hML : ∀ d v w, D d v w = g (Lambda d v) w + g v (Lambda d w))
    (v w : E) : Gamma v w = Lambda v w := by
  have hpair : ∀ z, g (Gamma v w) z = g (Lambda v w) z := fun z => by
    linarith [koszul g D Gamma hs hTG hMG v w z, koszul g D Lambda hs hTL hML v w z]
  by_contra hne
  have hz : g (Gamma v w - Lambda v w) (Gamma v w - Lambda v w) = 0 := by
    simp only [map_sub, LinearMap.sub_apply, hpair, sub_self]
  exact (ne_of_gt (hp _ (sub_ne_zero.mpr hne))) hz

end FractalSync.MetricConnectionUniqueness

namespace FractalSync.CurvedShoreGeometry

/-- The existing curved normal metric bundled as a bilinear form. -/
noncomputable def metricBilinForm (epsilon a k r : ℝ) : LinearMap.BilinForm ℝ Coord := by
  refine LinearMap.mk₂ ℝ (metric epsilon a k r) ?_ ?_ ?_ ?_
  all_goals intros
  all_goals dsimp [metric]
  all_goals ring

/-- The checked curved coefficients are the unique torsion-free connection
compatible with their actual metric-rate expression at a nonfocal point. -/
theorem christoffel_eq_of_compatible (epsilon a : ℝ) (k : ℝ → ℝ)
    (x : Coord) (Gamma : Coord → Coord → Coord) (he : 0 < epsilon)
    (hh : 1 - k x.2 * x.1 ≠ 0)
    (hT : ∀ v w, Gamma v w = Gamma w v)
    (hM : ∀ d v w,
      metricRate epsilon a (k x.2) (deriv k x.2) x.1 d v w =
        metric epsilon a (k x.2) x.1 (Gamma d v) w +
          metric epsilon a (k x.2) x.1 v (Gamma d w))
    (v w : Coord) : Gamma v w = christoffel epsilon a k x v w := by
  exact MetricConnectionUniqueness.eq_of_compatible
    (metricBilinForm epsilon a (k x.2) x.1)
    (metricRate epsilon a (k x.2) (deriv k x.2) x.1)
    Gamma (christoffel epsilon a k x)
    (metric_symm epsilon a (k x.2) x.1)
    (fun z hz => metric_self_pos epsilon a (k x.2) x.1 z he hh hz)
    hT (christoffel_symm epsilon a k x) hM
    (fun d v w => christoffel_metric_compatible epsilon a k x d v w he hh) v w

end FractalSync.CurvedShoreGeometry
