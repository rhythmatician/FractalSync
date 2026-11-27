import FractalSync.IntrinsicTopology

/-!
# Target sets and search envelopes for the positive-floor intrinsic metric

These lower bounds use whole-set coverage and one-sided real certificates.
They refer to the actual unrestricted path infimum.
-/

namespace FractalSync.DistanceGraph

open Set

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]

/-- Infimum of the intrinsic distances to every acceptable endpoint. -/
noncomputable def intrinsicTargetDistance (S : Set E) (epsilon a : ℝ)
    (T : Set E) (x : E) : ℝ :=
  sInf (intrinsicDistance S epsilon a x '' T)

private theorem bddBelow_intrinsicTargetImage (S : Set E) (epsilon a : ℝ)
    (T : Set E) (x : E) :
    BddBelow (intrinsicDistance S epsilon a x '' T) :=
  ⟨0, fun _ ⟨y, _, hy⟩ => hy ▸ intrinsicDistance_nonneg S epsilon a x y⟩

/-- A target lower bound must cover every accepted endpoint. -/
theorem le_intrinsicTargetDistance_iff (S : Set E) (epsilon a : ℝ)
    (T : Set E) (x : E) (b : ℝ) (hne : T.Nonempty) :
    b ≤ intrinsicTargetDistance S epsilon a T x ↔
      ∀ y ∈ T, b ≤ intrinsicDistance S epsilon a x y := by
  constructor
  · intro h y hy
    exact h.trans (csInf_le (bddBelow_intrinsicTargetImage S epsilon a T x) ⟨y, hy, rfl⟩)
  · intro h
    change b ≤ sInf (intrinsicDistance S epsilon a x '' T)
    refine le_csInf (hne.image _) ?_
    rintro _ ⟨y, hy, rfl⟩
    exact h y hy

theorem intrinsicTargetDistance_nonneg (S : Set E) (epsilon a : ℝ)
    (T : Set E) (x : E) (hne : T.Nonempty) :
    0 ≤ intrinsicTargetDistance S epsilon a T x :=
  (le_intrinsicTargetDistance_iff S epsilon a T x 0 hne).mpr
    (fun y _ => intrinsicDistance_nonneg S epsilon a x y)

theorem intrinsicTargetDistance_le_of_mem (S : Set E) (epsilon a : ℝ)
    (T : Set E) (x y : E) (hy : y ∈ T) :
    intrinsicTargetDistance S epsilon a T x ≤ intrinsicDistance S epsilon a x y :=
  csInf_le (bddBelow_intrinsicTargetImage S epsilon a T x) ⟨y, hy, rfl⟩

/-- An accepted state has zero remaining intrinsic target distance. -/
theorem intrinsicTargetDistance_eq_zero_of_mem (S : Set E) (epsilon a : ℝ)
    (T : Set E) (x : E) (hx : x ∈ T) :
    intrinsicTargetDistance S epsilon a T x = 0 :=
  le_antisymm
    ((intrinsicTargetDistance_le_of_mem S epsilon a T x x hx).trans_eq
      (intrinsicDistance_self S epsilon a x))
    (intrinsicTargetDistance_nonneg S epsilon a T x ⟨x, hx⟩)

/-- Replacing a whole target by a nonempty sample can only raise its infimum. -/
theorem intrinsicTargetDistance_le_of_subset (S : Set E) (epsilon a : ℝ)
    (T U : Set E) (x : E) (hne : U.Nonempty) (hsubset : U ⊆ T) :
    intrinsicTargetDistance S epsilon a T x ≤ intrinsicTargetDistance S epsilon a U x :=
  (le_intrinsicTargetDistance_iff S epsilon a U x _ hne).mpr
    (fun y hy => intrinsicTargetDistance_le_of_mem S epsilon a T x y (hsubset hy))

private theorem intervalGap_le_abs_sub_add (lo hi u v : ℝ) :
    max 0 (max (u - hi) (lo - u)) ≤ |u - v| + max 0 (max (v - hi) (lo - v)) := by
  have hnonneg : 0 ≤ max 0 (max (v - hi) (lo - v)) := le_max_left _ _
  have hhi : v - hi ≤ max 0 (max (v - hi) (lo - v)) :=
    (le_max_left _ _).trans (le_max_right _ _)
  have hlo : lo - v ≤ max 0 (max (v - hi) (lo - v)) :=
    (le_max_right _ _).trans (le_max_right _ _)
  have huv : u - v ≤ |u - v| := le_abs_self _
  have hvu : v - u ≤ |u - v| := by
    simpa only [neg_sub] using neg_le_abs (u - v)
  exact max_le (add_nonneg (abs_nonneg _) hnonneg)
    (max_le (by linarith only [huv, hhi]) (by linarith only [hvu, hlo]))

variable [FiniteDimensional ℝ E]

/-- Disjoint radial-profile intervals give a sound endpoint lower bound. -/
theorem radialIntervalGap_le_intrinsicDistance (S : Set E) (epsilon a : ℝ)
    (x y : E) (he : 0 < epsilon) {lx ux ly uy : ℝ}
    (hx : radialProfile epsilon a (Metric.infDist x S) ∈ Icc lx ux)
    (hy : radialProfile epsilon a (Metric.infDist y S) ∈ Icc ly uy) :
    max 0 (max (lx - uy) (ly - ux)) ≤ intrinsicDistance S epsilon a x y := by
  have h := radialProfile_difference_le_intrinsicDistance S epsilon a x y he
  refine max_le (intrinsicDistance_nonneg S epsilon a x y) (max_le ?_ ?_)
  · calc
      lx - uy ≤ radialProfile epsilon a (Metric.infDist x S) -
          radialProfile epsilon a (Metric.infDist y S) := sub_le_sub hx.1 hy.2
      _ ≤ |radialProfile epsilon a (Metric.infDist y S) -
          radialProfile epsilon a (Metric.infDist x S)| := by
        rw [abs_sub_comm]
        exact le_abs_self _
      _ ≤ _ := h
  · exact ((sub_le_sub hy.1 hx.2).trans (le_abs_self _)).trans h

/-- Clip a coordinate lower bound at zero and use an upper bound on the ruler. -/
theorem log_displacement_lower_le_intrinsicDistance (S : Set E) (epsilon a : ℝ)
    (x y : E) (he : 0 < epsilon) {lower upperRuler : ℝ}
    (hdisplacement : lower ≤ dist x y)
    (hruler : ruler epsilon (Metric.infDist x S) ≤ upperRuler) :
    Real.log (1 + max 0 lower / upperRuler) ≤ intrinsicDistance S epsilon a x y := by
  have hp := ruler_pos epsilon (Metric.infDist x S) he
  have hu := hp.trans_le hruler
  have hd : max 0 lower / upperRuler ≤ dist x y / ruler epsilon (Metric.infDist x S) :=
    (div_le_div_of_nonneg_right (max_le dist_nonneg hdisplacement) hu.le).trans
      (div_le_div_of_nonneg_left dist_nonneg hp hruler)
  exact (Real.log_le_log (by positivity) (add_le_add_left hd 1)).trans
    (log_displacement_le_intrinsicDistance S epsilon a x y he)

/-- Combine exact one-sided certificates that cover an entire target set. -/
theorem max_log_radialInterval_le_intrinsicTargetDistance
    (S : Set E) (epsilon a : ℝ) (T : Set E) (x : E) (he : 0 < epsilon)
    (hne : T.Nonempty) {lower upperRuler lx ux ly uy : ℝ}
    (hruler : ruler epsilon (Metric.infDist x S) ≤ upperRuler)
    (hx : radialProfile epsilon a (Metric.infDist x S) ∈ Icc lx ux)
    (hcover : ∀ y ∈ T,
      lower ≤ dist x y ∧ radialProfile epsilon a (Metric.infDist y S) ∈ Icc ly uy) :
    max (Real.log (1 + max 0 lower / upperRuler)) (max 0 (max (lx - uy) (ly - ux))) ≤
      intrinsicTargetDistance S epsilon a T x := by
  refine (le_intrinsicTargetDistance_iff S epsilon a T x _ hne).mpr (fun y hy => ?_)
  exact max_le
    (log_displacement_lower_le_intrinsicDistance S epsilon a x y he (hcover y hy).1 hruler)
    (radialIntervalGap_le_intrinsicDistance S epsilon a x y he hx (hcover y hy).2)

omit [NormedSpace ℝ E] [FiniteDimensional ℝ E] in
/-- Translate an ambient tolerance to a coordinate displacement lower bound. -/
theorem dist_sub_radius_le_dist_of_mem_closedBall
    (x p y : E) (tau : ℝ) (hy : y ∈ Metric.closedBall p tau) :
    dist x p - tau ≤ dist x y := by
  exact sub_le_iff_le_add.mpr ((dist_triangle x y p).trans
    (add_le_add_left (Metric.mem_closedBall.mp hy) _))

/-- An ambient tolerance reduces coordinate displacement before taking the log. -/
theorem log_ambientBall_le_intrinsicTargetDistance
    (S : Set E) (epsilon a : ℝ) (T : Set E) (x p : E) (he : 0 < epsilon)
    (hne : T.Nonempty) (tau : ℝ) {upperRuler : ℝ}
    (hruler : ruler epsilon (Metric.infDist x S) ≤ upperRuler)
    (hcover : T ⊆ Metric.closedBall p tau) :
    Real.log (1 + max 0 (dist x p - tau) / upperRuler) ≤
      intrinsicTargetDistance S epsilon a T x :=
  (le_intrinsicTargetDistance_iff S epsilon a T x _ hne).mpr (fun y hy =>
    log_displacement_lower_le_intrinsicDistance S epsilon a x y he
      (dist_sub_radius_le_dist_of_mem_closedBall x p y tau (hcover hy)) hruler)

/-- Translate an intrinsic tolerance by the intrinsic triangle inequality. -/
theorem center_lower_sub_tolerance_le_intrinsicTargetDistance
    (S : Set E) (epsilon a : ℝ) (T : Set E) (x p : E) (he : 0 < epsilon)
    (hne : T.Nonempty) (lower tau : ℝ) (hcenter : lower ≤ intrinsicDistance S epsilon a x p)
    (hcover : ∀ y ∈ T, intrinsicDistance S epsilon a y p ≤ tau) :
    max 0 (lower - tau) ≤ intrinsicTargetDistance S epsilon a T x := by
  refine (le_intrinsicTargetDistance_iff S epsilon a T x _ hne).mpr (fun y hy => ?_)
  have hxy := (intrinsicDistance_triangle S epsilon a x y p he).trans
    (add_le_add_left (hcover y hy) _)
  exact max_le (intrinsicDistance_nonneg S epsilon a x y)
    (sub_le_iff_le_add.mpr (hcenter.trans hxy))

/-- Exact target distance obeys the directed inequality for any dominating edge cost. -/
theorem intrinsicTargetDistance_edge_consistent (S : Set E) (epsilon a : ℝ)
    (T : Set E) (x y : E) (he : 0 < epsilon) (hne : T.Nonempty) (cost : ℝ)
    (hcost : intrinsicDistance S epsilon a x y ≤ cost) :
    intrinsicTargetDistance S epsilon a T x ≤ cost + intrinsicTargetDistance S epsilon a T y := by
  apply sub_le_iff_le_add'.mp
  refine (le_intrinsicTargetDistance_iff S epsilon a T y _ hne).mpr (fun z hz => ?_)
  apply sub_le_iff_le_add'.mpr
  calc
    intrinsicTargetDistance S epsilon a T x ≤ intrinsicDistance S epsilon a x z :=
      intrinsicTargetDistance_le_of_mem S epsilon a T x z hz
    _ ≤ intrinsicDistance S epsilon a x y + intrinsicDistance S epsilon a y z :=
      intrinsicDistance_triangle S epsilon a x y z he
    _ ≤ cost + intrinsicDistance S epsilon a y z := add_le_add_right hcost _

/-- Distance from the exact radial profile to a fixed target profile interval. -/
noncomputable def radialTargetIntervalGap (S : Set E) (epsilon a lo hi : ℝ) (x : E) : ℝ :=
  let profile := radialProfile epsilon a (Metric.infDist x S)
  max 0 (max (profile - hi) (lo - profile))

theorem radialTargetIntervalGap_le_intrinsicTargetDistance
    (S : Set E) (epsilon a : ℝ) (T : Set E) (x : E) (he : 0 < epsilon)
    (hne : T.Nonempty) (lo hi : ℝ)
    (hcover : ∀ y ∈ T, radialProfile epsilon a (Metric.infDist y S) ∈ Icc lo hi) :
    radialTargetIntervalGap S epsilon a lo hi x ≤ intrinsicTargetDistance S epsilon a T x := by
  refine (le_intrinsicTargetDistance_iff S epsilon a T x _ hne).mpr (fun y hy => ?_)
  exact radialIntervalGap_le_intrinsicDistance S epsilon a x y he
    ⟨le_rfl, le_rfl⟩ (hcover y hy)

/-- The fixed-interval radial lower bound is edge-consistent with exact profiles. -/
theorem radialTargetIntervalGap_edge_consistent (S : Set E) (epsilon a lo hi : ℝ)
    (x y : E) (he : 0 < epsilon) (cost : ℝ)
    (hcost : intrinsicDistance S epsilon a x y ≤ cost) :
    radialTargetIntervalGap S epsilon a lo hi x ≤ cost + radialTargetIntervalGap S epsilon a lo hi y := by
  apply (intervalGap_le_abs_sub_add lo hi _ _).trans
  apply add_le_add_right _ _
  rw [abs_sub_comm]
  exact (radialProfile_difference_le_intrinsicDistance S epsilon a x y he).trans hcost

omit [FiniteDimensional ℝ E] in
/-- A path split with certified prefix and suffix costs lies in the budget sublevel. -/
theorem mem_intrinsicTargetBudget_of_split (S : Set E) (epsilon a : ℝ)
    (T : Set E) {x y z : E} {prefixCost suffixCost budget : ℝ} (hz : z ∈ T)
    (hprefix : intrinsicDistance S epsilon a x y ≤ prefixCost)
    (hsuffix : intrinsicDistance S epsilon a y z ≤ suffixCost)
    (hbudget : prefixCost + suffixCost ≤ budget) :
    y ∈ {w | intrinsicDistance S epsilon a x w +
      intrinsicTargetDistance S epsilon a T w ≤ budget} :=
  (add_le_add hprefix
    ((intrinsicTargetDistance_le_of_mem S epsilon a T y z hz).trans hsuffix)).trans hbudget

/-- A combined start and remaining target budget fits in an ambient outer ball. -/
theorem intrinsicTargetBudget_subset_closedBall (S : Set E) (epsilon a : ℝ)
    (T : Set E) (x : E) (he : 0 < epsilon) (hne : T.Nonempty) {budget upperRuler : ℝ}
    (hruler : ruler epsilon (Metric.infDist x S) ≤ upperRuler) :
    {y | intrinsicDistance S epsilon a x y +
      intrinsicTargetDistance S epsilon a T y ≤ budget} ⊆
      Metric.closedBall x (upperRuler * (Real.exp budget - 1)) := by
  intro y hy
  have hdist : intrinsicDistance S epsilon a x y ≤ budget :=
    (le_add_of_nonneg_right (intrinsicTargetDistance_nonneg S epsilon a T y hne)).trans hy
  have hbudget := (intrinsicDistance_nonneg S epsilon a x y).trans hdist
  have h := intrinsicSublevel_subset_closedBall S epsilon a x budget he hdist
  exact Metric.closedBall_subset_closedBall
    (mul_le_mul_of_nonneg_right hruler (sub_nonneg.mpr (by simpa using Real.exp_le_exp.mpr hbudget))) h

end FractalSync.DistanceGraph
