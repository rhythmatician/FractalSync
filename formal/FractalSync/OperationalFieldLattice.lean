import FractalSync.OperationalFieldJoins

/-!
# Coherently indexed tensor patches and their actual boundary jets
-/

open scoped BigOperators

noncomputable section

namespace FractalSync.OperationalField

abbrev Lattice := ℤ → ℤ → ℝ

def latticeCoefficients (C : Lattice) (k l : ℤ) : Coefficients :=
  fun i j => C (k + i.val - 2) (l + j.val - 2)

def latticePatchJet (n m : ℕ) (C : Lattice) (k l : ℤ) (q : Coords) : ℝ :=
  tensorPatch (latticeCoefficients C k l)
    (fun x => basisJet n (x - k)) (fun y => basisJet m (y - l)) q

theorem hasFDerivAt_latticePatchJet (n m : ℕ) (C : Lattice) (k l : ℤ) (q : Coords) :
    HasFDerivAt (latticePatchJet n m C k l)
      (linearForm (latticePatchJet (n+1) m C k l q)
        (latticePatchJet n (m+1) C k l q)) q := by
  unfold latticePatchJet
  apply hasFDerivAt_tensorPatch
  · intro i
    simpa using ((hasDerivAt_basisJet n (q.1-k) i).comp q.1
      ((hasDerivAt_id q.1).sub_const (k : ℝ)))
  · intro i
    simpa using ((hasDerivAt_basisJet m (q.2-l) i).comp q.2
      ((hasDerivAt_id q.2).sub_const (l : ℝ)))


theorem latticePatchJet_join_x (n : Fin 5) (m : ℕ) (C : Lattice) (k l : ℤ)
    (q : Coords) (hq : q.1 = (k : ℝ)+1) :
    latticePatchJet n m C k l q = latticePatchJet n m C (k+1) l q := by
  simp only [latticePatchJet, tensorPatch, latticeCoefficients, hq, Int.cast_add,
    Int.cast_one, add_sub_cancel_left, sub_self, ← Finset.sum_mul]
  exact Finset.sum_congr rfl (fun j _ => congrArg
    (fun x => x * basisJet m (q.2-l) j)
    (latticeLineJet_join n (fun z => C z (l+j.val-2)) k))


theorem latticePatchJet_swap (n m : ℕ) (C : Lattice) (k l : ℤ) (q : Coords) :
    latticePatchJet n m C k l q =
      latticePatchJet m n (fun i j => C j i) l k (q.2, q.1) := by
  unfold latticePatchJet tensorPatch latticeCoefficients
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl (fun _ _ => Finset.sum_congr rfl (fun _ _ => by ring))

theorem latticePatchJet_join_y (n : ℕ) (m : Fin 5) (C : Lattice) (k l : ℤ)
    (q : Coords) (hq : q.2 = (l : ℝ)+1) :
    latticePatchJet n m C k l q = latticePatchJet n m C k (l+1) q := by
  simpa only [latticePatchJet_swap n m C k l q,
    latticePatchJet_swap n m C k (l+1) q] using
      latticePatchJet_join_x m n (fun i j => C j i) l k (q.2, q.1) hq

end FractalSync.OperationalField
