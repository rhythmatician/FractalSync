import Mathlib.Analysis.Calculus.ContDiff.Defs
import Mathlib.Analysis.Calculus.Deriv.Abs
import Mathlib.Dynamics.PeriodicPts.Defs
import Mathlib.Order.Filter.Finite
import Mathlib.Tactic.Linarith

/-!
# Certified finite trap selection

Only the scores of a fixed candidate set participate in the certificate.
Orbit evaluation, escape and contour tracking remain separate obligations.
-/

open Filter
open scoped Topology

namespace FractalSync.OrbitTrapSelection

theorem recoverTrueGap {measuredI measuredJ trueI trueJ errorI errorJ : ℝ}
    (hi : |measuredI - trueI| ≤ errorI) (hj : |measuredJ - trueJ| ≤ errorJ) :
    measuredJ - measuredI - (errorI + errorJ) ≤ trueJ - trueI := by
  linarith [(abs_le.mp hi).1, (abs_le.mp hj).2]

/-- The measurement score gap must strictly exceed both error bounds. -/
theorem certifiedStrictMinimum {ι : Type*} {s : Finset ι} {i : ι}
    {measured truth error : ι → ℝ} (hi : i ∈ s)
    (herr : ∀ j ∈ s, |measured j - truth j| ≤ error j)
    (hgap : ∀ j ∈ s, j ≠ i → error i + error j < measured j - measured i) :
    ∀ j ∈ s, j ≠ i → truth i < truth j := by
  exact fun j hj hji => sub_pos.mp <|
    (sub_pos.mpr (hgap j hj hji)).trans_le (recoverTrueGap (herr i hi) (herr j hj))

/-- The bound is attained with errors at the two opposite endpoints. -/
theorem recoverTrueGap_sharp (errorI errorJ : ℝ) (hi : 0 ≤ errorI) (hj : 0 ≤ errorJ) :
    ∃ measuredI measuredJ trueI trueJ : ℝ,
      |measuredI - trueI| ≤ errorI ∧ |measuredJ - trueJ| ≤ errorJ ∧
      measuredJ - measuredI = errorI + errorJ ∧ trueJ - trueI = 0 := by
  exact ⟨-errorI, errorJ, 0, 0, by simp [abs_of_nonneg hi, abs_of_nonneg hj, add_comm]⟩

/-- An open strict margin of a finite family of continuous scores persists locally. -/
theorem eventually_preservesGap {ι X : Type*} [TopologicalSpace X] {s : Finset ι}
    {score : ι → X → ℝ} {i : ι} {x : X} {margin : ℝ} (hi : i ∈ s)
    (hcont : ∀ j ∈ s, ContinuousAt (score j) x)
    (hgap : ∀ j ∈ s, j ≠ i → score i x + margin < score j x) :
    ∀ᶠ y in 𝓝 x, ∀ j ∈ s, j ≠ i → score i y + margin < score j y := by
  classical
  exact s.eventually_all.mpr fun j hj =>
    if hji : j = i then .of_forall (fun _ hj' => (hj' hji).elim)
    else (((hcont i hi).add continuousAt_const).eventually_lt (hcont j hj)
      (hgap j hj hji)).mono (fun _ h _ => h)

/-- The finite lower envelope is locally the single strict minimizing branch. -/
theorem lowerEnvelope_eventuallyEq {ι X : Type*} [TopologicalSpace X] {s : Finset ι}
    {score : ι → X → ℝ} {i : ι} {x : X} (hi : i ∈ s)
    (hcont : ∀ j ∈ s, ContinuousAt (score j) x)
    (hmin : ∀ j ∈ s, j ≠ i → score i x < score j x) :
    (fun y => s.inf' ⟨i, hi⟩ (fun j => score j y)) =ᶠ[𝓝 x] score i := by
  classical
  exact (eventually_preservesGap (margin := 0) hi hcont
    (fun j hj hji => by simpa using hmin j hj hji)).mono fun y hy =>
    le_antisymm (Finset.inf'_le _ hi) (Finset.le_inf' ⟨i, hi⟩ _ fun j hj =>
      if eq : j = i then le_of_eq (congrArg (fun k => score k y) eq.symm)
      else by simpa using (hy j hj eq).le)

/-- A certified measured gap transfers the selected branch's full local regularity. -/
theorem certifiedLowerEnvelope_contDiffAt {ι E : Type*} [NormedAddCommGroup E]
    [NormedSpace ℝ E] {s : Finset ι} {score : ι → E → ℝ} {i : ι} {x : E}
    {n : WithTop ℕ∞} {measured error : ι → ℝ} (hi : i ∈ s)
    (hcont : ∀ j ∈ s, ContinuousAt (score j) x) (hsmooth : ContDiffAt ℝ n (score i) x)
    (herr : ∀ j ∈ s, |measured j - score j x| ≤ error j)
    (hgap : ∀ j ∈ s, j ≠ i → error i + error j < measured j - measured i) :
    ContDiffAt ℝ n (fun y => s.inf' ⟨i, hi⟩ (fun j => score j y)) x := by
  exact hsmooth.congr_of_eventuallyEq
    (lowerEnvelope_eventuallyEq hi hcont (certifiedStrictMinimum hi herr hgap))

theorem tieEnvelope_eq (a x : ℝ) : min (a + x) (a - x) = a - |x| := by
  by_cases h : 0 ≤ x
  · rw [abs_of_nonneg h, min_eq_right (by linarith)]
  · rw [abs_of_neg (lt_of_not_ge h), min_eq_left (by linarith), sub_neg_eq_add]

/-- A tie of two analytic branches can make their minimum nondifferentiable. -/
theorem tieEnvelope_not_differentiable (a : ℝ) :
    ¬ DifferentiableAt ℝ (fun x : ℝ => min (a + x) (a - x)) 0 := by
  intro h
  have hc : DifferentiableAt ℝ (fun _ : ℝ => a) 0 := differentiableAt_const a
  exact not_differentiableAt_abs_zero (by simpa [tieEnvelope_eq, sub_sub_cancel] using hc.sub h)

/-- An exact positive iterate return to zero certifies a positive period dividing that index. -/
theorem criticalReturn_period {α : Type*} [Zero α] {f : α → α} {n : ℕ}
    (hn : 0 < n) (h : f^[n] 0 = 0) :
    0 < Function.minimalPeriod f 0 ∧ Function.minimalPeriod f 0 ∣ n := by
  exact ⟨(show Function.IsPeriodicPt f n 0 from h).minimalPeriod_pos hn,
    (show Function.IsPeriodicPt f n 0 from h).minimalPeriod_dvd⟩

end FractalSync.OrbitTrapSelection
