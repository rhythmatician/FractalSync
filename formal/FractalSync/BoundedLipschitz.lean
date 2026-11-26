import FractalSync.LocalDynamics

/-!
# Explicit bounds for coefficient expressions

A norm and Lipschitz certificate is propagated through the actual bilinear
operations used by mechanical acceleration. All bounds are exact real
inequalities on the specified region.
-/

namespace FractalSync

open Set
open scoped NNReal

variable {X F G H : Type*} [PseudoMetricSpace X]
  [NormedAddCommGroup F] [NormedSpace ℝ F]
  [NormedAddCommGroup G] [NormedSpace ℝ G]
  [NormedAddCommGroup H] [NormedSpace ℝ H]

/-- A uniform norm bound and a Lipschitz constant on the same actual region. -/
structure BoundedLipschitzOn (f : X → F) (s : Set X) (M L : ℝ) : Prop where
  bound_nonneg : 0 ≤ M
  lip_nonneg : 0 ≤ L
  norm_le : ∀ x ∈ s, ‖f x‖ ≤ M
  norm_sub_le : ∀ x ∈ s, ∀ y ∈ s, ‖f x - f y‖ ≤ L * dist x y

namespace BoundedLipschitzOn

/-- A norm-one bilinear operation propagates both concrete constants. -/
theorem bilinear {f : X → F} {g : X → G} {s : Set X} {M L N K : ℝ}
    (hf : BoundedLipschitzOn f s M L) (hg : BoundedLipschitzOn g s N K)
    (op : F →L[ℝ] G →L[ℝ] H)
    (hop : ∀ u v, ‖op u v‖ ≤ ‖u‖ * ‖v‖) :
    BoundedLipschitzOn (fun x => op (f x) (g x)) s (M * N) (M * K + N * L) := by
  refine ⟨mul_nonneg hf.bound_nonneg hg.bound_nonneg,
    add_nonneg (mul_nonneg hf.bound_nonneg hg.lip_nonneg)
      (mul_nonneg hg.bound_nonneg hf.lip_nonneg), ?_, ?_⟩
  case refine_1 =>
    exact fun x hx => (hop _ _).trans
      (mul_le_mul (hf.norm_le x hx) (hg.norm_le x hx) (norm_nonneg _) hf.bound_nonneg)
  case refine_2 =>
    intro x hx y hy
    have heq : op (f x) (g x) - op (f y) (g y) =
        op (f x) (g x - g y) + op (f x - f y) (g y) := by
      simp [map_sub, ContinuousLinearMap.sub_apply]
    calc
      _ = ‖op (f x) (g x - g y) + op (f x - f y) (g y)‖ := congrArg norm heq
      _ ≤ ‖op (f x) (g x - g y)‖ + ‖op (f x - f y) (g y)‖ := norm_add_le _ _
      _ ≤ M * (K * dist x y) + (L * dist x y) * N := add_le_add
        ((hop _ _).trans (mul_le_mul (hf.norm_le x hx) (hg.norm_sub_le x hx y hy)
          (norm_nonneg _) hf.bound_nonneg))
        ((hop _ _).trans (mul_le_mul (hf.norm_sub_le x hx y hy) (hg.norm_le y hy)
          (norm_nonneg _) (mul_nonneg hf.lip_nonneg dist_nonneg)))
      _ = (M * K + N * L) * dist x y := by ring

/-- A positive denominator has a bounded Lipschitz reciprocal on the region. -/
theorem inv {f : X → ℝ} {s : Set X} {M L epsilon : ℝ}
    (hf : BoundedLipschitzOn f s M L) (he : 0 < epsilon)
    (hfloor : ∀ x ∈ s, epsilon ≤ f x) :
    BoundedLipschitzOn (fun x => (f x)⁻¹) s epsilon⁻¹ (L / epsilon ^ 2) := by
  refine ⟨inv_nonneg.mpr he.le, div_nonneg hf.lip_nonneg (sq_nonneg _), ?_, ?_⟩
  case refine_1 =>
    exact fun x hx => by
      simpa only [Real.norm_eq_abs, abs_of_pos (inv_pos.mpr (he.trans_le (hfloor x hx)))] using
        (inv_le_inv₀ (he.trans_le (hfloor x hx)) he).mpr (hfloor x hx)
  case refine_2 =>
    intro x hx y hy
    calc
      ‖(f x)⁻¹ - (f y)⁻¹‖ = |f x - f y| / (f x * f y) := by
        simp only [Real.norm_eq_abs, inv_sub_inv (he.trans_le (hfloor x hx)).ne'
          (he.trans_le (hfloor y hy)).ne', abs_div, abs_mul,
          abs_of_pos (he.trans_le (hfloor x hx)), abs_of_pos (he.trans_le (hfloor y hy)),
          abs_sub_comm]
      _ ≤ (L * dist x y) / epsilon ^ 2 := div_le_div₀
        (mul_nonneg hf.lip_nonneg dist_nonneg)
        (by simpa only [Real.norm_eq_abs] using hf.norm_sub_le x hx y hy)
        (sq_pos_of_pos he)
        (by simpa only [pow_two] using (mul_le_mul (hfloor x hx) (hfloor y hy) he.le (he.le.trans (hfloor x hx))))
      _ = L / epsilon ^ 2 * dist x y := by ring

omit [NormedSpace ℝ F] in
theorem lipschitzOnWith {f : X → F} {s : Set X} {M L : ℝ}
    (hf : BoundedLipschitzOn f s M L) : LipschitzOnWith ⟨L, hf.lip_nonneg⟩ f s := by
  exact LipschitzOnWith.of_dist_le_mul fun x hx y hy => by
    simpa only [dist_eq_norm, NNReal.coe_mk] using hf.norm_sub_le x hx y hy

omit [NormedSpace ℝ F] in
theorem const (s : Set X) (c : F) : BoundedLipschitzOn (fun _ : X => c) s ‖c‖ 0 := by
  exact ⟨norm_nonneg _, le_rfl, fun _ _ => le_rfl, fun _ _ _ _ => by simp⟩

omit [NormedSpace ℝ F] in
theorem mono {f : X → F} {s t : Set X} {M L : ℝ}
    (hf : BoundedLipschitzOn f s M L) (ht : t ⊆ s) : BoundedLipschitzOn f t M L := by
  exact ⟨hf.bound_nonneg, hf.lip_nonneg, fun x hx => hf.norm_le x (ht hx),
    fun x hx y hy => hf.norm_sub_le x (ht hx) y (ht hy)⟩

omit [NormedSpace ℝ F] in
theorem enlarge {f : X → F} {s : Set X} {M L N K : ℝ}
    (hf : BoundedLipschitzOn f s M L) (hM : M ≤ N) (hL : L ≤ K) :
    BoundedLipschitzOn f s N K := by
  exact ⟨hf.bound_nonneg.trans hM, hf.lip_nonneg.trans hL,
    fun x hx => (hf.norm_le x hx).trans hM,
    fun x hx y hy => (hf.norm_sub_le x hx y hy).trans (mul_le_mul_of_nonneg_right hL dist_nonneg)⟩

omit [NormedSpace ℝ F] in
theorem add {f g : X → F} {s : Set X} {M L N K : ℝ}
    (hf : BoundedLipschitzOn f s M L) (hg : BoundedLipschitzOn g s N K) :
    BoundedLipschitzOn (fun x => f x + g x) s (M + N) (L + K) := by
  exact ⟨add_nonneg hf.bound_nonneg hg.bound_nonneg, add_nonneg hf.lip_nonneg hg.lip_nonneg,
    fun x hx => (norm_add_le _ _).trans (add_le_add (hf.norm_le x hx) (hg.norm_le x hx)),
    fun x hx y hy => by
      simpa only [dist_eq_norm, NNReal.coe_add, NNReal.coe_mk] using
        (hf.lipschitzOnWith.add hg.lipschitzOnWith).dist_le_mul x hx y hy⟩

omit [NormedSpace ℝ F] in
theorem sub {f g : X → F} {s : Set X} {M L N K : ℝ}
    (hf : BoundedLipschitzOn f s M L) (hg : BoundedLipschitzOn g s N K) :
    BoundedLipschitzOn (fun x => f x - g x) s (M + N) (L + K) := by
  exact ⟨add_nonneg hf.bound_nonneg hg.bound_nonneg, add_nonneg hf.lip_nonneg hg.lip_nonneg,
    fun x hx => (_root_.norm_sub_le _ _).trans (add_le_add (hf.norm_le x hx) (hg.norm_le x hx)),
    fun x hx y hy => by
      simpa only [dist_eq_norm, NNReal.coe_add, NNReal.coe_mk] using
        (hf.lipschitzOnWith.sub hg.lipschitzOnWith).dist_le_mul x hx y hy⟩

theorem smul {f : X → ℝ} {g : X → F} {s : Set X} {M L N K : ℝ}
    (hf : BoundedLipschitzOn f s M L) (hg : BoundedLipschitzOn g s N K) :
    BoundedLipschitzOn (fun x => f x • g x) s (M * N) (M * K + N * L) := by
  exact hf.bilinear hg (ContinuousLinearMap.lsmul ℝ ℝ) (fun u v => (norm_smul_le u v))

theorem mul {f g : X → ℝ} {s : Set X} {M L N K : ℝ}
    (hf : BoundedLipschitzOn f s M L) (hg : BoundedLipschitzOn g s N K) :
    BoundedLipschitzOn (fun x => f x * g x) s (M * N) (M * K + N * L) := by
  exact hf.smul hg

omit [NormedSpace ℝ F] in
theorem inner [InnerProductSpace ℝ F] {f g : X → F} {s : Set X} {M L N K : ℝ}
    (hf : BoundedLipschitzOn f s M L) (hg : BoundedLipschitzOn g s N K) :
    BoundedLipschitzOn (fun x => inner ℝ (f x) (g x)) s (M * N) (M * K + N * L) := by
  exact hf.bilinear hg (innerSL ℝ) (fun u v => norm_inner_le_norm u v)

theorem clm_apply {f : X → F →L[ℝ] G} {g : X → F} {s : Set X} {M L N K : ℝ}
    (hf : BoundedLipschitzOn f s M L) (hg : BoundedLipschitzOn g s N K) :
    BoundedLipschitzOn (fun x => f x (g x)) s (M * N) (M * K + N * L) := by
  exact hf.bilinear hg (ContinuousLinearMap.id ℝ (F →L[ℝ] G)) (fun u v => u.le_opNorm v)

omit [NormedSpace ℝ F] in
theorem of_lipschitz {f : X → F} {s : Set X} {M : ℝ} {L : ℝ≥0}
    (hM : 0 ≤ M) (hLip : LipschitzOnWith L f s) (hbound : ∀ x ∈ s, ‖f x‖ ≤ M) :
    BoundedLipschitzOn f s M L := by
  exact ⟨hM, L.coe_nonneg, hbound, fun x hx y hy => by
    simpa only [dist_eq_norm] using hLip.dist_le_mul x hx y hy⟩

omit [NormedSpace ℝ F] in
theorem comp {Y : Type*} [PseudoMetricSpace Y] {f : Y → F} {g : X → Y}
    {s : Set X} {t : Set Y} {M L : ℝ} {K : ℝ≥0}
    (hf : BoundedLipschitzOn f t M L) (hg : LipschitzOnWith K g s)
    (hmap : MapsTo g s t) :
    BoundedLipschitzOn (fun x => f (g x)) s M (L * K) := by
  exact ⟨hf.bound_nonneg, mul_nonneg hf.lip_nonneg K.coe_nonneg,
    fun x hx => hf.norm_le (g x) (hmap hx),
    fun x hx y hy => by
      simpa only [mul_assoc] using (hf.norm_sub_le (g x) (hmap hx) (g y) (hmap hy)).trans
        (mul_le_mul_of_nonneg_left (hg.dist_le_mul x hx y hy) hf.lip_nonneg)⟩

end BoundedLipschitzOn
end FractalSync
