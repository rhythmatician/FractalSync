import FractalSync.ZeroFloorLimit
import FractalSync.IntrinsicCurveLength
import FractalSync.Analysis.BoundedSpeedCompactness

/-!
# Attained zero-floor geodesics and compactness of regularized minimizers

All zero-floor lengths below are the existing actual integrals on ordinary
absolutely continuous paths avoiding the closed Shore.
-/

namespace FractalSync.DistanceGraph

open Set Filter MeasureTheory
open scoped Topology NNReal

variable {E : Type*} [NormedAddCommGroup E] [NormedSpace ℝ E]
  [FiniteDimensional ℝ E]

/-- A bound on every regularized length gives an actual uniform clearance,
with no smoothness or derivative premise on distance to the set. -/
theorem infDist_lower_of_positive_floor_lengths (S : Set E) (a B : ℝ)
    (gamma : ℝ → E) (hgamma : AbsolutelyContinuousOnInterval gamma 0 1)
    (hB : ∀ epsilon > (0 : ℝ), acGraphLength S epsilon a gamma ≤ B)
    (t : ℝ) (ht : t ∈ uIcc (0 : ℝ) 1) :
    Metric.infDist (gamma 0) S * Real.exp (-B) ≤ Metric.infDist (gamma t) S := by
  have hcont : Continuous (fun epsilon : ℝ =>
      Real.sqrt ((Metric.infDist (gamma t) S) ^ 2 + epsilon ^ 2)) :=
    Real.continuous_sqrt.comp (continuous_const.add (continuous_id.pow 2))
  have hlim : Tendsto (fun epsilon : ℝ => ruler epsilon (Metric.infDist (gamma t) S))
      (𝓝[>] (0 : ℝ)) (𝓝 (Metric.infDist (gamma t) S)) := by
    simpa only [ruler, zero_pow (by norm_num : (2 : ℕ) ≠ 0), add_zero,
      Real.sqrt_sq (Metric.infDist_nonneg)] using
        (hcont.tendsto 0).mono_left nhdsWithin_le_nhds
  exact ge_of_tendsto hlim (Filter.Eventually.mono self_mem_nhdsWithin fun epsilon he =>
    ac_ruler_lower_of_length_le S gamma epsilon a B t he hgamma (hB epsilon he) ht)

/-- Uniform positive-floor length bounds pass to the actual zero-floor
integral when the initial point is off a nonempty closed Shore. -/
theorem acGraphLength_zero_le_of_positive_floor_lengths (S : Set E)
    (hclosed : IsClosed S) (hne : S.Nonempty) (a B : ℝ) (gamma : ℝ → E)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) (hx : gamma 0 ∉ S)
    (hB : ∀ epsilon > (0 : ℝ), acGraphLength S epsilon a gamma ≤ B) :
    acGraphLength S 0 a gamma ≤ B := by
  have hm : 0 < Metric.infDist (gamma 0) S * Real.exp (-B) :=
    mul_pos ((hclosed.notMem_iff_infDist_pos hne).mp hx) (Real.exp_pos _)
  have hcont : Continuous (fun epsilon : ℝ =>
      (1 - epsilon ^ 2 / (Metric.infDist (gamma 0) S * Real.exp (-B)) ^ 2) *
        acGraphLength S 0 a gamma) := by fun_prop
  have hlim : Tendsto (fun epsilon : ℝ =>
      (1 - epsilon ^ 2 / (Metric.infDist (gamma 0) S * Real.exp (-B)) ^ 2) *
        acGraphLength S 0 a gamma) (𝓝[>] (0 : ℝ)) (𝓝 (acGraphLength S 0 a gamma)) := by
    simpa using (hcont.tendsto 0).mono_left nhdsWithin_le_nhds
  apply le_of_tendsto hlim
  filter_upwards [self_mem_nhdsWithin,
    (eventually_lt_nhds hm).filter_mono nhdsWithin_le_nhds] with epsilon he hsmall
  exact (zeroFloorLength_mul_le_of_length_le S hclosed hne epsilon a B gamma he hgamma
    (hB epsilon he) hsmall).trans (hB epsilon he)

/-- The positive-floor distance is jointly continuous in the original ambient
topology, as well as in its equal intrinsic topology. -/
theorem continuous_intrinsicDistance_pair (S : Set E) (epsilon a : ℝ)
    (he : 0 < epsilon) :
    Continuous (fun p : E × E => intrinsicDistance S epsilon a p.1 p.2) := by
  have h := @continuous_dist E (intrinsicMetricSpace S epsilon a he).toPseudoMetricSpace
  simpa only [intrinsicMetricSpace_dist, intrinsicMetricSpace_topology S epsilon a he] using h

/-- Any selected sequence of constant-speed regularized geodesics with floors
in `(0,1]` tending to zero has a uniformly convergent subsequence. Its limit
obeys the zero-floor speed bound in every fixed positive-floor metric. -/
theorem exists_uniform_limit_of_small_floor_geodesics (S : Set E)
    (hclosed : IsClosed S) (hne : S.Nonempty) (a : ℝ) (x y : E)
    (hpaths : (zeroFloorPathLengths S a x y).Nonempty)
    (epsilon : ℕ → ℝ) (he : ∀ n, 0 < epsilon n) (hsmall : ∀ n, epsilon n ≤ 1)
    (he0 : Tendsto epsilon atTop (𝓝 0)) (f : ℕ → unitInterval → E)
    (hf : ∀ n, f n 0 = x ∧ f n 1 = y ∧ ∀ s t,
      intrinsicDistance S (epsilon n) a (f n s) (f n t) =
        intrinsicDistance S (epsilon n) a x y * dist s t) :
    ∃ g : unitInterval → E, ∃ phi : ℕ → ℕ,
      StrictMono phi ∧ TendstoUniformly (f ∘ phi) g atTop ∧ g 0 = x ∧ g 1 = y ∧
        ∀ delta > (0 : ℝ), ∀ s t,
          intrinsicDistance S delta a (g s) (g t) ≤ zeroFloorDistance S a x y * dist s t := by
  have hD : 0 ≤ zeroFloorDistance S a x y := zeroFloorDistance_nonneg S a x y hpaths
  have hspeed : ∀ n s t, intrinsicDistance S 1 a (f n s) (f n t) ≤
      zeroFloorDistance S a x y * dist s t := fun n s t =>
    (intrinsicDistance_antitone_floor S (epsilon n) 1 a (f n s) (f n t) (he n) (hsmall n)).trans
      (((hf n).2.2 s t).le.trans (mul_le_mul_of_nonneg_right
        (intrinsicDistance_le_zeroFloorDistance S hclosed hne (epsilon n) a x y (he n) hpaths)
        dist_nonneg))
  let K : ℝ≥0 := ⟨ruler 1 (Metric.infDist x S) * (Real.exp (zeroFloorDistance S a x y)) ^ 2 *
      zeroFloorDistance S a x y,
    mul_nonneg (mul_nonneg (ruler_pos 1 _ zero_lt_one).le (sq_nonneg _)) hD⟩
  have hLip : ∀ n, LipschitzWith K (f n) := fun n => by
    simpa only [(hf n).1] using
      lipschitz_of_intrinsic_boundedSpeed S 1 a (f n) (zeroFloorDistance S a x y)
        zero_lt_one hD (hspeed n)
  obtain ⟨g, hg, phi, hphi, hconv⟩ := MetricGeometry.exists_uniformly_convergent_subsequence
    x y K f (fun n => ⟨(hf n).1, (hf n).2.1, (hLip n).dist_le_mul⟩)
  refine ⟨g, phi, hphi, hconv, hg.1, hg.2.1, fun delta hdelta s t => ?_⟩
  apply le_of_tendsto (((continuous_intrinsicDistance_pair S delta a hdelta).tendsto
    (g s, g t)).comp ((hconv.tendsto_at s).prodMk_nhds (hconv.tendsto_at t)))
  filter_upwards [(he0.comp hphi.tendsto_atTop).eventually (eventually_lt_nhds hdelta)]
    with n hn
  exact (intrinsicDistance_antitone_floor S (epsilon (phi n)) delta a
    (f (phi n) s) (f (phi n) t) (he (phi n)) hn.le).trans
      (((hf (phi n)).2.2 s t).le.trans (mul_le_mul_of_nonneg_right
        (intrinsicDistance_le_zeroFloorDistance S hclosed hne (epsilon (phi n)) a x y
          (he (phi n)) hpaths) dist_nonneg))

/-- Arbitrary positive floors tending to zero, without a monotonicity or
global upper-bound assumption, have the same selected-geodesic compactness. -/
theorem exists_uniform_limit_of_geodesics (S : Set E)
    (hclosed : IsClosed S) (hne : S.Nonempty) (a : ℝ) (x y : E)
    (hpaths : (zeroFloorPathLengths S a x y).Nonempty)
    (epsilon : ℕ → ℝ) (he : ∀ n, 0 < epsilon n)
    (he0 : Tendsto epsilon atTop (𝓝 0)) (f : ℕ → unitInterval → E)
    (hf : ∀ n, f n 0 = x ∧ f n 1 = y ∧ ∀ s t,
      intrinsicDistance S (epsilon n) a (f n s) (f n t) =
        intrinsicDistance S (epsilon n) a x y * dist s t) :
    ∃ g : unitInterval → E, ∃ phi : ℕ → ℕ,
      StrictMono phi ∧ TendstoUniformly (f ∘ phi) g atTop ∧ g 0 = x ∧ g 1 = y ∧
        ∀ delta > (0 : ℝ), ∀ s t,
          intrinsicDistance S delta a (g s) (g t) ≤ zeroFloorDistance S a x y * dist s t := by
  obtain ⟨N, hN⟩ := eventually_atTop.mp (he0.eventually (eventually_lt_nhds zero_lt_one))
  obtain ⟨g, phi, hphi, hconv, hg0, hg1, hspeed⟩ :=
    exists_uniform_limit_of_small_floor_geodesics S hclosed hne a x y hpaths
      (fun n => epsilon (n + N)) (fun n => he (n + N))
      (fun n => (hN (n + N) (Nat.le_add_left N n)).le)
      (he0.comp (tendsto_add_atTop_nat N)) (fun n => f (n + N)) (fun n => hf (n + N))
  exact ⟨g, fun n => phi n + N, fun i j hij => Nat.add_lt_add_right (hphi hij) N,
    hconv, hg0, hg1, hspeed⟩

/-- A limiting path controlled at every positive floor is an ordinary AC
minimizer of the actual zero-floor integral, with explicit Shore clearance. -/
theorem zeroFloor_minimizer_of_positive_floor_speed (S : Set E)
    (hclosed : IsClosed S) (hne : S.Nonempty) (a : ℝ) (x y : E) (hx : x ∉ S)
    (hpaths : (zeroFloorPathLengths S a x y).Nonempty)
    (g : unitInterval → E) (hg0 : g 0 = x) (hg1 : g 1 = y)
    (hspeed : ∀ epsilon > (0 : ℝ), ∀ s t,
      intrinsicDistance S epsilon a (g s) (g t) ≤ zeroFloorDistance S a x y * dist s t) :
    ∃ gamma : ℝ → E, AbsolutelyContinuousOnInterval gamma 0 1 ∧
      gamma 0 = x ∧ gamma 1 = y ∧ (∀ t : unitInterval, gamma t = g t) ∧
      (∀ t ∈ uIcc (0 : ℝ) 1, Metric.infDist x S * Real.exp (-zeroFloorDistance S a x y) ≤
        Metric.infDist (gamma t) S) ∧
      (∀ t ∈ uIcc (0 : ℝ) 1, gamma t ∉ S) ∧
      acGraphLength S 0 a gamma = zeroFloorDistance S a x y := by
  have hD : 0 ≤ zeroFloorDistance S a x y := zeroFloorDistance_nonneg S a x y hpaths
  have hLip := lipschitz_of_intrinsic_boundedSpeed S 1 a g (zeroFloorDistance S a x y)
    zero_lt_one hD (hspeed 1 zero_lt_one)
  let gamma : ℝ → E := g ∘ projIcc (0 : ℝ) 1 zero_le_one
  have hAC : AbsolutelyContinuousOnInterval gamma 0 1 :=
    (hLip.comp (LipschitzWith.projIcc zero_le_one)).lipschitzOnWith.absolutelyContinuousOnInterval
  have hends : gamma 0 = x ∧ gamma 1 = y := by
    simpa only [gamma, Function.comp_apply, projIcc_left, projIcc_right] using And.intro hg0 hg1
  have hagree : ∀ t : unitInterval, gamma t = g t := fun t => by
    simp only [gamma, Function.comp_apply, projIcc_of_mem zero_le_one t.2]
  have hlength : ∀ epsilon > (0 : ℝ), acGraphLength S epsilon a gamma ≤ zeroFloorDistance S a x y :=
    fun epsilon he => acGraphLength_le_of_intrinsic_boundedSpeed S epsilon a gamma
      (zeroFloorDistance S a x y) he hD hAC (fun s hs t ht => by
        simpa only [← hagree, Subtype.dist_eq, Real.dist_eq] using
          hspeed epsilon he ⟨s, by simpa only [uIcc_of_le zero_le_one] using hs⟩
            ⟨t, by simpa only [uIcc_of_le zero_le_one] using ht⟩)
  have hclear : ∀ t ∈ uIcc (0 : ℝ) 1,
      Metric.infDist x S * Real.exp (-zeroFloorDistance S a x y) ≤ Metric.infDist (gamma t) S :=
    fun t ht => by simpa only [hends.1] using
      infDist_lower_of_positive_floor_lengths S a (zeroFloorDistance S a x y) gamma hAC hlength t ht
  have havoid : ∀ t ∈ uIcc (0 : ℝ) 1, gamma t ∉ S := fun t ht =>
    (hclosed.notMem_iff_infDist_pos hne).mpr
      ((mul_pos ((hclosed.notMem_iff_infDist_pos hne).mp hx) (Real.exp_pos _)).trans_le (hclear t ht))
  exact ⟨gamma, hAC, hends.1, hends.2, hagree, hclear, havoid,
    le_antisymm
      (acGraphLength_zero_le_of_positive_floor_lengths S hclosed hne a
        (zeroFloorDistance S a x y) gamma hAC (havoid 0 left_mem_uIcc) hlength)
      (zeroFloorDistance_le_length S a x y gamma hAC hends.1 hends.2 havoid)⟩

/-- The zero-floor increment bound is sharp on every subinterval. Only the
actual fixed-pair floor limit and the positive-floor triangle inequality are
used; no zero-floor metric or minimizing-curve structure is assumed. -/
theorem zeroFloor_geodesic_of_positive_floor_speed (S : Set E)
    (hclosed : IsClosed S) (hne : S.Nonempty) (a : ℝ) (x y : E)
    (g : unitInterval → E) (hg0 : g 0 = x) (hg1 : g 1 = y)
    (hcont : Continuous g) (havoid : ∀ t, g t ∉ S)
    (hspeed : ∀ epsilon > (0 : ℝ), ∀ s t,
      intrinsicDistance S epsilon a (g s) (g t) ≤ zeroFloorDistance S a x y * dist s t) :
    ∀ s t, zeroFloorDistance S a (g s) (g t) = zeroFloorDistance S a x y * dist s t := by
  have hpaths : ∀ s t, (zeroFloorPathLengths S a (g s) (g t)).Nonempty := fun s t =>
    zeroFloorPathLengths_nonempty_of_mem_connectedComponentIn S hclosed a (g s) (g t)
      (havoid s) ((isPreconnected_range hcont).subset_connectedComponentIn
        (mem_range_self s) (range_subset_iff.mpr havoid) (mem_range_self t))
  have hlim := fun s t => intrinsicDistance_tendsto_zero_floor_of_path_nonempty
    S hclosed hne a (g s) (g t) (hpaths s t)
  have hupper : ∀ s t, zeroFloorDistance S a (g s) (g t) ≤
      zeroFloorDistance S a x y * dist s t := fun s t =>
    le_of_tendsto (hlim s t) (Filter.Eventually.mono self_mem_nhdsWithin
      fun epsilon he => hspeed epsilon he s t)
  have hsymm : ∀ s t, zeroFloorDistance S a (g s) (g t) = zeroFloorDistance S a (g t) (g s) :=
    fun s t => tendsto_nhds_unique (hlim s t)
      ((hlim t s).congr (fun epsilon => intrinsicDistance_symm S epsilon a (g t) (g s)))
  have hfour : ∀ s t, zeroFloorDistance S a (g 0) (g 1) ≤
      zeroFloorDistance S a (g 0) (g s) + zeroFloorDistance S a (g s) (g t) +
        zeroFloorDistance S a (g t) (g 1) := fun s t =>
    le_of_tendsto_of_tendsto (hlim 0 1) (((hlim 0 s).add (hlim s t)).add (hlim t 1))
      (Filter.Eventually.mono self_mem_nhdsWithin fun epsilon he =>
        @dist_triangle4 E (intrinsicMetricSpace S epsilon a he).toPseudoMetricSpace
          (g 0) (g s) (g t) (g 1))
  intro s t
  wlog hst : s ≤ t generalizing s t
  · simpa only [hsymm t s, dist_comm t s] using this t s (le_of_not_ge hst)
  refine le_antisymm (hupper s t) ?_
  have hb := (hfour s t).trans
    (add_le_add (add_le_add_right (hupper 0 s) _) (hupper t 1))
  simp only [hg0, hg1, Subtype.dist_eq, Set.Icc.coe_zero, Set.Icc.coe_one,
    Real.dist_eq, zero_sub, abs_neg, abs_of_nonneg s.2.1,
    abs_of_nonpos (sub_nonpos.mpr t.2.2),
    abs_of_nonpos (sub_nonpos.mpr (show (s : ℝ) ≤ t from hst))] at hb ⊢
  nlinarith

/-- Every selected sequence of regularized constant-speed minimizers along
positive floors tending to zero has an ambient-uniformly convergent
subsequence. Its limit is an actual constant-speed zero-floor AC minimizer
inside the original complement component. Uniqueness is not assumed. -/
theorem selected_geodesics_subsequence_zero_floor (S : Set E)
    (hclosed : IsClosed S) (hne : S.Nonempty) (a : ℝ) (x y : E) (hx : x ∉ S)
    (hy : y ∈ connectedComponentIn Sᶜ x)
    (epsilon : ℕ → ℝ) (he : ∀ n, 0 < epsilon n)
    (he0 : Tendsto epsilon atTop (𝓝 0)) (f : ℕ → unitInterval → E)
    (hf : ∀ n, f n 0 = x ∧ f n 1 = y ∧ ∀ s t,
      intrinsicDistance S (epsilon n) a (f n s) (f n t) =
        intrinsicDistance S (epsilon n) a x y * dist s t) :
    ∃ phi : ℕ → ℕ, ∃ gamma : ℝ → E,
      StrictMono phi ∧ TendstoUniformly (f ∘ phi) (fun t : unitInterval => gamma t) atTop ∧
      AbsolutelyContinuousOnInterval gamma 0 1 ∧ gamma 0 = x ∧ gamma 1 = y ∧
      (∀ t ∈ uIcc (0 : ℝ) 1, gamma t ∈ connectedComponentIn Sᶜ x) ∧
      (∀ t ∈ uIcc (0 : ℝ) 1, Metric.infDist x S * Real.exp (-zeroFloorDistance S a x y) ≤
        Metric.infDist (gamma t) S) ∧
      acGraphLength S 0 a gamma = zeroFloorDistance S a x y ∧
      ∀ s ∈ uIcc (0 : ℝ) 1, ∀ t ∈ uIcc (0 : ℝ) 1,
        zeroFloorDistance S a (gamma s) (gamma t) = zeroFloorDistance S a x y * |s - t| := by
  have hpaths := zeroFloorPathLengths_nonempty_of_mem_connectedComponentIn S hclosed a x y hx hy
  obtain ⟨g, phi, hphi, hconv, hg0, hg1, hspeed⟩ :=
    exists_uniform_limit_of_geodesics S hclosed hne a x y hpaths epsilon he he0 f hf
  obtain ⟨gamma, hAC, hstart, hend, hagree, hclear, havoid, hlength⟩ :=
    zeroFloor_minimizer_of_positive_floor_speed S hclosed hne a x y hx hpaths g hg0 hg1 hspeed
  have hcont : Continuous g := (lipschitz_of_intrinsic_boundedSpeed S 1 a g
    (zeroFloorDistance S a x y) zero_lt_one (zeroFloorDistance_nonneg S a x y hpaths)
    (hspeed 1 zero_lt_one)).continuous
  have havoid_unit : ∀ t, g t ∉ S := fun t => by
    simpa only [hagree] using havoid t (by simpa only [uIcc_of_le zero_le_one] using t.2)
  refine ⟨phi, gamma, hphi, ?_, hAC, hstart, hend, ?_, hclear, hlength, ?_⟩
  · simpa only [hagree] using hconv
  · exact fun t ht => by
      simpa only [← hagree] using
        (isPreconnected_range hcont).subset_connectedComponentIn ⟨0, hg0⟩
          (range_subset_iff.mpr havoid_unit)
          (mem_range_self (f := g) ⟨t, by simpa only [uIcc_of_le zero_le_one] using ht⟩)
  exact fun s hs t ht => by
    simpa only [← hagree, Subtype.dist_eq, Real.dist_eq] using
      zeroFloor_geodesic_of_positive_floor_speed S hclosed hne a x y g hg0 hg1 hcont
        havoid_unit hspeed ⟨s, by simpa only [uIcc_of_le zero_le_one] using hs⟩
          ⟨t, by simpa only [uIcc_of_le zero_le_one] using ht⟩

/-- Same-component endpoints admit an ordinary AC path attaining the actual
zero-floor integral infimum, with constant zero-floor metric speed. -/
theorem exists_zeroFloor_minimizer (S : Set E) (hclosed : IsClosed S) (hne : S.Nonempty)
    (a : ℝ) (x y : E) (hx : x ∉ S) (hy : y ∈ connectedComponentIn Sᶜ x) :
    ∃ gamma : ℝ → E, AbsolutelyContinuousOnInterval gamma 0 1 ∧ gamma 0 = x ∧ gamma 1 = y ∧
      (∀ t ∈ uIcc (0 : ℝ) 1, gamma t ∈ connectedComponentIn Sᶜ x) ∧
      (∀ t ∈ uIcc (0 : ℝ) 1, Metric.infDist x S * Real.exp (-zeroFloorDistance S a x y) ≤
        Metric.infDist (gamma t) S) ∧
      acGraphLength S 0 a gamma = zeroFloorDistance S a x y ∧
      ∀ s ∈ uIcc (0 : ℝ) 1, ∀ t ∈ uIcc (0 : ℝ) 1,
        zeroFloorDistance S a (gamma s) (gamma t) = zeroFloorDistance S a x y * |s - t| := by
  have he : ∀ n : ℕ, 0 < 1 / ((n : ℝ) + 1) := fun n => by positivity
  choose f hf using fun n : ℕ => exists_intrinsic_metricGeodesic S (1 / ((n : ℝ) + 1)) a x y (he n)
  obtain ⟨_, gamma, _, _, hgamma⟩ := selected_geodesics_subsequence_zero_floor
    S hclosed hne a x y hx hy (fun n => 1 / ((n : ℝ) + 1)) he
      tendsto_one_div_add_atTop_nhds_zero_nat f hf
  exact ⟨gamma, hgamma⟩

/-- The existing zero-floor path infimum is a minimum of its original
off-Shore AC length class for same-component endpoints. -/
theorem zeroFloorDistance_isLeast (S : Set E) (hclosed : IsClosed S) (hne : S.Nonempty)
    (a : ℝ) (x y : E) (hx : x ∉ S) (hy : y ∈ connectedComponentIn Sᶜ x) :
    IsLeast (zeroFloorPathLengths S a x y) (zeroFloorDistance S a x y) := by
  obtain ⟨gamma, hAC, h0, h1, hcomponent, _, hlength, _⟩ :=
    exists_zeroFloor_minimizer S hclosed hne a x y hx hy
  exact ⟨⟨gamma, hAC, h0, h1,
    fun t ht => connectedComponentIn_subset Sᶜ x (hcomponent t ht), hlength⟩,
    fun _ hL => csInf_le (zeroFloorPathLengths_bddBelow S a x y) hL⟩

/-- Every selected family of actual regularized AC minimizers eventually
avoids Shore, and its unregularized integral lengths tend to the zero-floor
minimum. No constant-speed parameterization is required for this statement. -/
theorem selected_minimizer_lengths_tendsto_zero_floor (S : Set E)
    (hclosed : IsClosed S) (hne : S.Nonempty) (a : ℝ) (x y : E) (hx : x ∉ S)
    (hy : y ∈ connectedComponentIn Sᶜ x) (epsilon : ℕ → ℝ)
    (he : ∀ n, 0 < epsilon n) (he0 : Tendsto epsilon atTop (𝓝 0))
    (gamma : ℕ → ℝ → E) (hAC : ∀ n, AbsolutelyContinuousOnInterval (gamma n) 0 1)
    (hstart : ∀ n, gamma n 0 = x) (hend : ∀ n, gamma n 1 = y)
    (hlength : ∀ n, acGraphLength S (epsilon n) a (gamma n) = intrinsicDistance S (epsilon n) a x y) :
    (∀ᶠ n in atTop, ∀ t ∈ uIcc (0 : ℝ) 1, gamma n t ∉ S) ∧
      Tendsto (fun n => acGraphLength S 0 a (gamma n)) atTop (𝓝 (zeroFloorDistance S a x y)) := by
  have hpaths := zeroFloorPathLengths_nonempty_of_mem_connectedComponentIn S hclosed a x y hx hy
  have hbound : ∀ n, acGraphLength S (epsilon n) a (gamma n) ≤ zeroFloorDistance S a x y :=
    fun n => (hlength n).le.trans
      (intrinsicDistance_le_zeroFloorDistance S hclosed hne (epsilon n) a x y (he n) hpaths)
  let m := Metric.infDist x S * Real.exp (-zeroFloorDistance S a x y)
  have hm : 0 < m := mul_pos ((hclosed.notMem_iff_infDist_pos hne).mp hx) (Real.exp_pos _)
  have hsmall : ∀ᶠ n in atTop, epsilon n < m := he0.eventually (eventually_lt_nhds hm)
  have havoid : ∀ᶠ n in atTop, ∀ t ∈ uIcc (0 : ℝ) 1, gamma n t ∉ S :=
    hsmall.mono fun n hn => ac_path_avoids_of_length_le S (gamma n) (epsilon n) a
      (zeroFloorDistance S a x y) (he n) (hAC n) (hbound n)
        (by simpa only [hstart n] using hn)
  have hupper : ∀ᶠ n in atTop, acGraphLength S 0 a (gamma n) ≤
      zeroFloorDistance S a x y / (1 - (epsilon n) ^ 2 / m ^ 2) := by
    filter_upwards [hsmall] with n hn
    have hfactor : 0 < 1 - (epsilon n) ^ 2 / m ^ 2 := sub_pos.mpr
      ((div_lt_one (sq_pos_of_pos hm)).mpr ((sq_lt_sq₀ (he n).le hm.le).mpr hn))
    apply (le_div_iff₀ hfactor).mpr
    simpa only [hstart n, mul_comm] using
      (zeroFloorLength_mul_le_of_length_le S hclosed hne (epsilon n) a
        (zeroFloorDistance S a x y) (gamma n) (he n) (hAC n) (hbound n)
          (by simpa only [hstart n] using hn)).trans (hbound n)
  have hlimupper : Tendsto (fun n => zeroFloorDistance S a x y /
      (1 - (epsilon n) ^ 2 / m ^ 2)) atTop (𝓝 (zeroFloorDistance S a x y)) := by
    simpa using (tendsto_const_nhds (x := zeroFloorDistance S a x y)).div
      ((tendsto_const_nhds (x := (1 : ℝ))).sub ((he0.pow 2).div_const (m ^ 2))) (by simp)
  exact ⟨havoid, tendsto_of_tendsto_of_tendsto_of_le_of_le' tendsto_const_nhds hlimupper
    (havoid.mono fun n hn => zeroFloorDistance_le_length S a x y (gamma n)
      (hAC n) (hstart n) (hend n) hn) hupper⟩

end FractalSync.DistanceGraph
