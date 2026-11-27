import FractalSync.IntrinsicShoreDistance
import FractalSync.Analysis.ACComposition
import Mathlib.Analysis.Calculus.Deriv.Shift

/-!
# Reversal and concatenation of absolutely continuous graph paths

The admissible path operations preserve actual absolute continuity and the
length computed from actual derivatives. Affine derivative formulas and
Lebesgue interval changes of variables establish the length identities.
-/

namespace FractalSync.DistanceGraph

open MeasureTheory Set Filter
open scoped Topology NNReal

/-- Reverse the unit-interval parameter of a path. -/
def reversePath {E : Type*} (gamma : ℝ → E) (t : ℝ) : E := gamma (1 - t)

@[simp] theorem reversePath_zero {E : Type*} (gamma : ℝ → E) :
    reversePath gamma 0 = gamma 1 := by simp [reversePath]

@[simp] theorem reversePath_one {E : Type*} (gamma : ℝ → E) :
    reversePath gamma 1 = gamma 0 := by simp [reversePath]

@[simp] theorem reversePath_reversePath {E : Type*} (gamma : ℝ → E) :
    reversePath (reversePath gamma) = gamma := by
  funext t
  simp [reversePath]

/-- Reversal preserves absolute continuity, including nonsmooth paths. -/
theorem ac_reversePath {E : Type*} [PseudoMetricSpace E] (gamma : ℝ → E)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1) :
    AbsolutelyContinuousOnInterval (reversePath gamma) 0 1 := by
  apply ACComposition.comp_lipschitz_mono hgamma (K := 1)
  · exact LipschitzWith.of_dist_le_mul fun x y => by
      simpa only [NNReal.coe_one, one_mul] using (dist_sub_left (1 : ℝ) x y).le
  · exact Or.inr fun x y h => sub_le_sub_left h 1
  · intro t ht
    simp only [uIcc_of_le (by norm_num : (0 : ℝ) ≤ 1), mem_Icc] at ht ⊢
    constructor <;> linarith

/-- The actual derivative-based graph length is invariant under reversal.
The affine derivative identity is valid even at nondifferentiable points,
where Lean's derivative has its usual zero value. -/
theorem acGraphLength_reversePath {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] (S : Set E) (epsilon a : ℝ) (gamma : ℝ → E) :
    acGraphLength S epsilon a (reversePath gamma) = acGraphLength S epsilon a gamma := by
  have hd (t : ℝ) : deriv (reversePath gamma) t = -deriv gamma (1 - t) :=
    deriv_comp_const_sub gamma 1 t
  have hr (t : ℝ) : deriv (fun s => Metric.infDist (gamma (1 - s)) S) t =
      -deriv (fun s => Metric.infDist (gamma s) S) (1 - t) :=
    deriv_comp_const_sub (fun s => Metric.infDist (gamma s) S) 1 t
  simpa only [acGraphLength, hd, hr, reversePath, norm_neg,
    graphSpeed, graphSpeedSq, neg_sq, sub_self, sub_zero] using
      (intervalIntegral.integral_comp_sub_left
        (fun t => graphSpeed a epsilon (Metric.infDist (gamma t) S)
          ‖deriv gamma t‖ (deriv (fun s => Metric.infDist (gamma s) S) t))
        (a := (0 : ℝ)) (b := 1) 1)

lemma lipschitz_double_sub (b : ℝ) :
    LipschitzWith 2 (fun t : ℝ => 2 * t - b) := by
  apply LipschitzWith.of_dist_le_mul
  intro x y
  change |(2 * x - b) - (2 * y - b)| ≤ 2 * |x - y|
  rw [show (2 * x - b) - (2 * y - b) = 2 * (x - y) by ring, abs_mul]
  norm_num

/-- Traverse each input path in one half of the unit interval. -/
noncomputable def concatPath {E : Type*} (gamma eta : ℝ → E) : ℝ → E :=
  ACComposition.gluePath (fun t => gamma (2 * t))
    (fun t => eta (2 * t - 1)) (1 / 2)

@[simp] theorem concatPath_zero {E : Type*} (gamma eta : ℝ → E) :
    concatPath gamma eta 0 = gamma 0 := by
  norm_num [concatPath, ACComposition.gluePath]

@[simp] theorem concatPath_one {E : Type*} (gamma eta : ℝ → E) :
    concatPath gamma eta 1 = eta 1 := by
  norm_num [concatPath, ACComposition.gluePath]

/-- Matching AC paths concatenate without a differentiability requirement at
one half. -/
theorem ac_concatPath {E : Type*} [SeminormedAddCommGroup E] (gamma eta : ℝ → E)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1)
    (heta : AbsolutelyContinuousOnInterval eta 0 1) (hmatch : gamma 1 = eta 0) :
    AbsolutelyContinuousOnInterval (concatPath gamma eta) 0 1 := by
  apply ACComposition.absolutelyContinuous_gluePath (by norm_num) (by norm_num)
  · apply ACComposition.comp_lipschitz_mono hgamma
      (show LipschitzWith 2 (fun t : ℝ => 2 * t) from by
        simpa only [sub_zero] using lipschitz_double_sub 0)
      (Or.inl fun _ _ h => mul_le_mul_of_nonneg_left h (by norm_num))
    intro t ht
    simp only [uIcc_of_le (by norm_num : (0 : ℝ) ≤ 1 / 2),
      uIcc_of_le (by norm_num : (0 : ℝ) ≤ 1), mem_Icc] at ht ⊢
    constructor <;> linarith
  · apply ACComposition.comp_lipschitz_mono heta (lipschitz_double_sub 1)
      (Or.inl fun _ _ h => sub_le_sub_right
        (mul_le_mul_of_nonneg_left h (by norm_num)) 1)
    intro t ht
    simp only [uIcc_of_le (by norm_num : (1 / 2 : ℝ) ≤ 1),
      uIcc_of_le (by norm_num : (0 : ℝ) ≤ 1), mem_Icc] at ht ⊢
    constructor <;> linarith
  · simpa using hmatch

/-- Applying a function pointwise commutes with the two-branch concatenation. -/
theorem map_concatPath {E F : Type*} (f : E → F) (gamma eta : ℝ → E) :
    (fun t => f (concatPath gamma eta t)) =
      concatPath (fun t => f (gamma t)) (fun t => f (eta t)) := by
  funext t
  simp only [concatPath, ACComposition.gluePath]
  split_ifs <;> rfl

/-- Actual derivative on the first open half, with the affine speed factor. -/
theorem deriv_concatPath_of_lt {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] (gamma eta : ℝ → E) (t : ℝ) (ht : t < 1 / 2) :
    deriv (concatPath gamma eta) t = (2 : ℝ) • deriv gamma (2 * t) := by
  have hEq : concatPath gamma eta =ᶠ[𝓝 t] fun s => gamma (2 * s) := by
    filter_upwards [isOpen_Iio.mem_nhds ht] with s hs
    exact if_pos hs.le
  rw [hEq.deriv_eq]
  exact deriv_comp_mul_left 2 gamma t

/-- Actual derivative on the second open half. -/
theorem deriv_concatPath_of_gt {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] (gamma eta : ℝ → E) (t : ℝ) (ht : 1 / 2 < t) :
    deriv (concatPath gamma eta) t = (2 : ℝ) • deriv eta (2 * t - 1) := by
  have hEq : concatPath gamma eta =ᶠ[𝓝 t] fun s => eta (2 * s - 1) := by
    filter_upwards [isOpen_Ioi.mem_nhds ht] with s hs
    exact if_neg (not_le_of_gt hs)
  rw [hEq.deriv_eq]
  simpa only [deriv_comp_sub_const] using
    (deriv_comp_mul_left (2 : ℝ) (fun s => eta (s - 1)) t)

/-- Positive homogeneity of the graph speed in its two velocity inputs. -/
theorem graphSpeed_mul_nonneg (a epsilon r speed w c : ℝ) (hc : 0 ≤ c) :
    graphSpeed a epsilon r (c * speed) (c * w) =
      c * graphSpeed a epsilon r speed w := by
  have hsq : graphSpeedSq a epsilon r (c * speed) (c * w) =
      c ^ 2 * graphSpeedSq a epsilon r speed w := by
    unfold graphSpeedSq
    ring
  rw [graphSpeed, hsq, Real.sqrt_mul (sq_nonneg c), Real.sqrt_sq hc]
  rfl

/-- The graph density doubles on the first half of a concatenated path. -/
theorem graphSpeed_concatPath_of_lt {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] (S : Set E) (epsilon a : ℝ) (gamma eta : ℝ → E)
    (t : ℝ) (ht : t < 1 / 2) :
    graphSpeed a epsilon (Metric.infDist (concatPath gamma eta t) S)
      ‖deriv (concatPath gamma eta) t‖
      (deriv (fun s => Metric.infDist (concatPath gamma eta s) S) t) =
    2 * graphSpeed a epsilon (Metric.infDist (gamma (2 * t)) S)
      ‖deriv gamma (2 * t)‖
      (deriv (fun s => Metric.infDist (gamma s) S) (2 * t)) := by
  rw [map_concatPath (fun x => Metric.infDist x S),
    deriv_concatPath_of_lt gamma eta t ht,
    deriv_concatPath_of_lt (fun s => Metric.infDist (gamma s) S)
      (fun s => Metric.infDist (eta s) S) t ht]
  simp only [concatPath, ACComposition.gluePath, if_pos ht.le,
    norm_smul, Real.norm_eq_abs, abs_of_pos (by norm_num : (0 : ℝ) < 2), smul_eq_mul]
  exact graphSpeed_mul_nonneg _ _ _ _ _ _ (by norm_num)

/-- The graph density doubles on the second half of a concatenated path. -/
theorem graphSpeed_concatPath_of_gt {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] (S : Set E) (epsilon a : ℝ) (gamma eta : ℝ → E)
    (t : ℝ) (ht : 1 / 2 < t) :
    graphSpeed a epsilon (Metric.infDist (concatPath gamma eta t) S)
      ‖deriv (concatPath gamma eta) t‖
      (deriv (fun s => Metric.infDist (concatPath gamma eta s) S) t) =
    2 * graphSpeed a epsilon (Metric.infDist (eta (2 * t - 1)) S)
      ‖deriv eta (2 * t - 1)‖
      (deriv (fun s => Metric.infDist (eta s) S) (2 * t - 1)) := by
  rw [map_concatPath (fun x => Metric.infDist x S),
    deriv_concatPath_of_gt gamma eta t ht,
    deriv_concatPath_of_gt (fun s => Metric.infDist (gamma s) S)
      (fun s => Metric.infDist (eta s) S) t ht]
  simp only [concatPath, ACComposition.gluePath, if_neg (not_le_of_gt ht),
    norm_smul, Real.norm_eq_abs, abs_of_pos (by norm_num : (0 : ℝ) < 2), smul_eq_mul]
  exact graphSpeed_mul_nonneg _ _ _ _ _ _ (by norm_num)

/-- Concatenation adds the actual graph lengths. The single joining point is
null; all integrability follows from absolute continuity and the positive floor. -/
theorem acGraphLength_concatPath {E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] [FiniteDimensional ℝ E]
    (S : Set E) (epsilon a : ℝ) (gamma eta : ℝ → E) (he : 0 < epsilon)
    (hgamma : AbsolutelyContinuousOnInterval gamma 0 1)
    (heta : AbsolutelyContinuousOnInterval eta 0 1) (hmatch : gamma 1 = eta 0) :
    acGraphLength S epsilon a (concatPath gamma eta) =
      acGraphLength S epsilon a gamma + acGraphLength S epsilon a eta := by
  let f : ℝ → ℝ := fun t => graphSpeed a epsilon
    (Metric.infDist (concatPath gamma eta t) S) ‖deriv (concatPath gamma eta) t‖
    (deriv (fun s => Metric.infDist (concatPath gamma eta s) S) t)
  let fg : ℝ → ℝ := fun t => graphSpeed a epsilon (Metric.infDist (gamma t) S)
    ‖deriv gamma t‖ (deriv (fun s => Metric.infDist (gamma s) S) t)
  let fe : ℝ → ℝ := fun t => graphSpeed a epsilon (Metric.infDist (eta t) S)
    ‖deriv eta t‖ (deriv (fun s => Metric.infDist (eta s) S) t)
  have hi : IntervalIntegrable f volume 0 1 :=
    ac_graphSpeed_intervalIntegrable S (concatPath gamma eta) epsilon a 0 1 he
      (ac_concatPath gamma eta hgamma heta hmatch)
  have hleft : (∫ t in (0 : ℝ)..(1 / 2), f t) = ∫ t in (0 : ℝ)..1, fg t := by
    calc
      _ = ∫ t in (0 : ℝ)..(1 / 2), 2 * fg (2 * t) := by
        apply intervalIntegral.integral_congr_ae
        filter_upwards [show ∀ᵐ t : ℝ, t ≠ 1 / 2 from by simp [ae_iff]] with t hne ht
        have ht' : t ∈ Ioc (0 : ℝ) (1 / 2) := by
          simpa only [uIoc_of_le (by norm_num : (0 : ℝ) ≤ 1 / 2)] using ht
        exact graphSpeed_concatPath_of_lt S epsilon a gamma eta t
          (lt_of_le_of_ne ht'.2 hne)
      _ = ∫ t in (0 : ℝ)..1, fg t := by
        rw [intervalIntegral.integral_const_mul]
        norm_num
        ring
  have hright : (∫ t in (1 / 2 : ℝ)..1, f t) = ∫ t in (0 : ℝ)..1, fe t := by
    calc
      _ = ∫ t in (1 / 2 : ℝ)..1, 2 * fe (2 * t - 1) := by
        apply intervalIntegral.integral_congr_ae
        filter_upwards with t ht
        have ht' : t ∈ Ioc (1 / 2 : ℝ) 1 := by
          simpa only [uIoc_of_le (by norm_num : (1 / 2 : ℝ) ≤ 1)] using ht
        exact graphSpeed_concatPath_of_gt S epsilon a gamma eta t ht'.1
      _ = ∫ t in (0 : ℝ)..1, fe t := by
        rw [intervalIntegral.integral_const_mul]
        norm_num
        ring
  have hsubleft : uIcc (0 : ℝ) (1 / 2) ⊆ uIcc (0 : ℝ) 1 := by
    simp only [uIcc_of_le (by norm_num : (0 : ℝ) ≤ 1 / 2),
      uIcc_of_le (by norm_num : (0 : ℝ) ≤ 1)]
    exact Icc_subset_Icc le_rfl (by norm_num)
  have hsubright : uIcc (1 / 2 : ℝ) 1 ⊆ uIcc (0 : ℝ) 1 := by
    simp only [uIcc_of_le (by norm_num : (1 / 2 : ℝ) ≤ 1),
      uIcc_of_le (by norm_num : (0 : ℝ) ≤ 1)]
    exact Icc_subset_Icc (by norm_num) le_rfl
  change (∫ t in (0 : ℝ)..1, f t) =
    (∫ t in (0 : ℝ)..1, fg t) + ∫ t in (0 : ℝ)..1, fe t
  rw [← intervalIntegral.integral_add_adjacent_intervals
    (hi.mono_set hsubleft) (hi.mono_set hsubright), hleft, hright]

end FractalSync.DistanceGraph
