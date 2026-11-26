import FractalSync.OperationalFieldLattice

/-!
# Actual global real jets of the floor-selected operational field

An arbitrary integer coefficient lattice determines the exact 36-term query at
every real point. Its first and second derivatives remain the selected tensor
jets across grid lines and vertices. Integer floors here are mathematical floors;
no Rust floating-point indexing theorem is asserted.
-/

noncomputable section

namespace FractalSync.OperationalField

def xSelectedJet (n m : ℕ) (C : Lattice) (l : ℤ) : Coords → ℝ :=
  floorSelect Prod.fst (fun k => latticePatchJet n m C k l)

def globalJet (n m : ℕ) (C : Lattice) (q : Coords) : ℝ :=
  latticePatchJet n m C ⌊q.1⌋ ⌊q.2⌋ q

theorem hasFDerivAt_xSelectedJet (n m : ℕ) (hn : n < 4) (C : Lattice)
    (l : ℤ) (q : Coords) :
    HasFDerivAt (xSelectedJet n m C l)
      (linearForm (xSelectedJet (n+1) m C l q) (xSelectedJet n (m+1) C l q)) q := by
  simpa only [xSelectedJet, floorSelect] using
    hasFDerivAt_floorSelect Prod.fst continuous_fst
      (fun k => latticePatchJet n m C k l)
      (fun k z => linearForm (latticePatchJet (n+1) m C k l z)
        (latticePatchJet n (m+1) C k l z))
      (fun k z => hasFDerivAt_latticePatchJet n m C k l z)
      (fun k z hz => latticePatchJet_join_x ⟨n, by omega⟩ m C k l z hz)
      (fun k z hz => congrArg₂ linearForm
        (latticePatchJet_join_x ⟨n+1, by omega⟩ m C k l z hz)
        (latticePatchJet_join_x ⟨n, by omega⟩ (m+1) C k l z hz)) q

/-- Full derivatives of actual floor-selected mixed jets, including at grid vertices. -/
theorem hasFDerivAt_globalJet (n m : ℕ) (hn : n < 4) (hm : m < 4)
    (C : Lattice) (q : Coords) :
    HasFDerivAt (globalJet n m C)
      (linearForm (globalJet (n+1) m C q) (globalJet n (m+1) C q)) q := by
  simpa only [globalJet, xSelectedJet, floorSelect] using
    hasFDerivAt_floorSelect Prod.snd continuous_snd
      (fun l => xSelectedJet n m C l)
      (fun l z => linearForm (xSelectedJet (n+1) m C l z)
        (xSelectedJet n (m+1) C l z))
      (fun l z => hasFDerivAt_xSelectedJet n m hn C l z)
      (fun l z hz => latticePatchJet_join_y n ⟨m, by omega⟩ C ⌊z.1⌋ l z hz)
      (fun l z hz => congrArg₂ linearForm
        (latticePatchJet_join_y (n+1) ⟨m, by omega⟩ C ⌊z.1⌋ l z hz)
        (latticePatchJet_join_y n ⟨m+1, by omega⟩ C ⌊z.1⌋ l z hz)) q


theorem continuous_xSelectedJet (n m : ℕ) (hn : n ≤ 4) (C : Lattice) (l : ℤ) :
    Continuous (xSelectedJet n m C l) := by
  exact continuous_iff_continuousAt.mpr (fun q =>
    continuousAt_floorSelect Prod.fst continuous_fst
      (fun k => latticePatchJet n m C k l)
      (fun k z => (hasFDerivAt_latticePatchJet n m C k l z).continuousAt)
      (fun k z hz => latticePatchJet_join_x ⟨n, by omega⟩ m C k l z hz) q)

theorem continuous_globalJet (n m : ℕ) (hn : n ≤ 4) (hm : m ≤ 4) (C : Lattice) :
    Continuous (globalJet n m C) := by
  exact continuous_iff_continuousAt.mpr (fun q =>
    continuousAt_floorSelect Prod.snd continuous_snd
      (fun l => xSelectedJet n m C l)
      (fun l _ => (continuous_xSelectedJet n m hn C l).continuousAt)
      (fun l z hz => latticePatchJet_join_y n ⟨m, by omega⟩ C ⌊z.1⌋ l z hz) q)

/-- Mixed jets have the remaining differentiability promised by the quintic joins. -/
theorem contDiff_globalJet (r n m : ℕ) (hn : n+r ≤ 4) (hm : m+r ≤ 4) (C : Lattice) :
    ContDiff ℝ r (globalJet n m C) := by
  induction r generalizing n m with
  | zero => exact contDiff_zero.mpr (continuous_globalJet n m hn hm C)
  | succ r ih =>
    apply contDiff_succ_iff_hasFDerivAt.mpr
    refine ⟨fun q => linearForm (globalJet (n+1) m C q) (globalJet n (m+1) C q), ?_,
      fun q => hasFDerivAt_globalJet n m (by omega) (by omega) C q⟩
    exact ((ih (n+1) m (by omega) (by omega)).smul
      (contDiff_const (c := ContinuousLinearMap.fst ℝ ℝ ℝ))).add
      ((ih n (m+1) (by omega) (by omega)).smul
        (contDiff_const (c := ContinuousLinearMap.snd ℝ ℝ ℝ)))


def globalField (C : Lattice) : Coords → ℝ := globalJet 0 0 C

def globalDifferential (C : Lattice) (q : Coords) : Coords →L[ℝ] ℝ :=
  linearForm (globalJet 1 0 C q) (globalJet 0 1 C q)

def globalHessian (C : Lattice) (q : Coords) : Coords →L[ℝ] (Coords →L[ℝ] ℝ) :=
  quadraticForm (globalJet 2 0 C q) (globalJet 1 1 C q) (globalJet 0 2 C q)

theorem hasFDerivAt_globalField (C : Lattice) (q : Coords) :
    HasFDerivAt (globalField C) (globalDifferential C q) q :=
  hasFDerivAt_globalJet 0 0 (by norm_num) (by norm_num) C q

theorem hasFDerivAt_globalDifferential (C : Lattice) (q : Coords) :
    HasFDerivAt (globalDifferential C) (globalHessian C q) q :=
  hasFDerivAt_linearForm _ _ _ _ q
    (hasFDerivAt_globalJet 1 0 (by norm_num) (by norm_num) C q)
    (hasFDerivAt_globalJet 0 1 (by norm_num) (by norm_num) C q)

theorem fderiv_globalField (C : Lattice) (q : Coords) :
    fderiv ℝ (globalField C) q = globalDifferential C q :=
  (hasFDerivAt_globalField C q).fderiv

theorem fderiv_fderiv_globalField (C : Lattice) (q : Coords) :
    fderiv ℝ (fderiv ℝ (globalField C)) q = globalHessian C q := by
  simpa only [show fderiv ℝ (globalField C) = globalDifferential C from
    funext (fun z => fderiv_globalField C z)] using
      (hasFDerivAt_globalDifferential C q).fderiv

/-- The actual globally selected real field is C4 for every coefficient lattice. -/
theorem contDiff_globalField (C : Lattice) : ContDiff ℝ 4 (globalField C) :=
  contDiff_globalJet 4 0 0 (by norm_num) (by norm_num) C

theorem globalField_eq_selected_patch (C : Lattice) (q : Coords) :
    globalField C q =
      field (latticeCoefficients C ⌊q.1⌋ ⌊q.2⌋) ((⌊q.1⌋ : ℝ), (⌊q.2⌋ : ℝ)) 1 q := by
  unfold globalField globalJet latticePatchJet field weights
  simp

end FractalSync.OperationalField
