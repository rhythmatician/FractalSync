import FractalSync.OperationalFieldGlobalError

/-!
# Global Hessian query stability of the concrete quintic field

An explicit third-derivative Bernstein certificate supplies the missing
cross-cell Hessian query bound. The constants concern exact real evaluation.
-/

open Set
open scoped BigOperators

noncomputable section

namespace FractalSync.OperationalField

/-- Degree-two Bernstein numerators of the actual third basis derivatives. -/
def thirdControls : Fin 6 → Fin 3 → ℤ :=
  ![![-60, 0, 0], ![120, -120, -60], ![0, 360, 120],
    ![-120, -360, 0], ![60, 120, -120], ![0, 0, 60]]

theorem basisThird_eq_bernstein (t : ℝ) (i : Fin 6) :
    basisJet 3 t i = (∑ k : Fin 3, (thirdControls i k : ℝ) * bernsteinOfDegree 2 t k) / 120 := by
  fin_cases i
  all_goals norm_num [basisJet, coefficients, hornerDerivative, thirdControls,
    bernsteinOfDegree, Fin.sum_univ_succ, Nat.choose]
  all_goals ring


theorem thirdControls_column_bound (k : Fin 3) :
    (∑ i : Fin 6, |(thirdControls i k : ℝ)|) ≤ 960 := by
  fin_cases k <;> norm_num [thirdControls, Fin.sum_univ_succ]

theorem basisThird_abs_sum_le (t : ℝ) (ht : t ∈ Icc (0 : ℝ) 1) :
    (∑ i : Fin 6, |basisJet 3 t i|) ≤ 8 := by
  calc
    _ = (∑ i : Fin 6, |∑ k : Fin 3,
        (thirdControls i k : ℝ) * bernsteinOfDegree 2 t k|) / 120 := by
      simp only [basisThird_eq_bernstein, abs_div,
        abs_of_pos (show (0 : ℝ) < 120 by norm_num)]
      rw [Finset.sum_div]
    _ ≤ 960/120 := div_le_div_of_nonneg_right
      (sum_abs_weighted_columns_bound (fun i k => (thirdControls i k : ℝ))
        (bernsteinOfDegree 2 t) 960 (bernsteinOfDegree_nonneg 2 t ht)
        (bernsteinOfDegree_partition 2 t) thirdControls_column_bound) (by norm_num)
    _ = 8 := by norm_num


def jetBound : ℕ → ℝ
  | 0 => 1
  | 1 => 4/3
  | 2 => 2
  | _ => 8

theorem basisJet_abs_sum_le (n : ℕ) (hn : n ≤ 3) (t : ℝ) (ht : t ∈ Icc (0 : ℝ) 1) :
    (∑ i : Fin 6, |basisJet n t i|) ≤ jetBound n := by
  interval_cases n
  · simpa [jetBound] using (basis_abs_sum t ht).le
  · simpa [jetBound] using basisFirst_abs_sum_le t ht
  · simpa [jetBound] using basisSecond_abs_sum_le t ht
  · exact basisThird_abs_sum_le t ht


theorem globalJet_abs_le (n m : ℕ) (hn : n ≤ 3) (hm : m ≤ 3)
    (C : Lattice) (q : Coords) (B : ℝ) (hB : 0 ≤ B) (hC : ∀ i j, |C i j| ≤ B) :
    |globalJet n m C q| ≤ B * jetBound n * jetBound m := by
  exact tensorPatch_abs_bound (latticeCoefficients C ⌊q.1⌋ ⌊q.2⌋)
    (fun x => basisJet n (x-⌊q.1⌋)) (fun y => basisJet m (y-⌊q.2⌋)) q B
    (jetBound n) (jetBound m) hB (fun _ _ => hC _ _)
    (basisJet_abs_sum_le n hn _ (Ico_subset_Icc_self (floor_fraction_mem q.1)))
    (basisJet_abs_sum_le m hm _ (Ico_subset_Icc_self (floor_fraction_mem q.2)))

/-- Coefficient-derived physical mixed-jet bounds, including every third-order term. -/
theorem physicalJet_abs_le (n m : ℕ) (hn : n ≤ 3) (hm : m ≤ 3)
    (C : Lattice) (o q : Coords) (h B : ℝ) (hh : 0 < h) (hB : 0 ≤ B)
    (hC : ∀ i j, |C i j| ≤ B) :
    |physicalJet n m C o h q| ≤ B * jetBound n * jetBound m / h^(n+m) := by
  simpa only [physicalJet, abs_div, abs_of_pos (pow_pos hh (n+m))] using
    div_le_div_of_nonneg_right (globalJet_abs_le n m hn hm C (normalizedCoordinates o h q) B hB hC)
      (pow_pos hh (n+m)).le


/-- Every second-order component has global query stability through cell boundaries. -/
theorem physicalJet_query_error_bound (n m : ℕ) (hn : n ≤ 2) (hm : m ≤ 2)
    (C : Lattice) (o q q' : Coords) (h B : ℝ) (hh : 0 < h) (hB : 0 ≤ B)
    (hC : ∀ i j, |C i j| ≤ B) :
    |physicalJet n m C o h q' - physicalJet n m C o h q| ≤
      B * (jetBound (n+1)*jetBound m + jetBound n*jetBound (m+1)) / h^(n+m+1) * ‖q'-q‖ := by
  have hb (z : Coords) :
      ‖linearForm (physicalJet (n+1) m C o h z) (physicalJet n (m+1) C o h z)‖ ≤
        B * (jetBound (n+1)*jetBound m + jetBound n*jetBound (m+1)) / h^(n+m+1) := by
    calc
      _ ≤ |physicalJet (n+1) m C o h z| + |physicalJet n (m+1) C o h z| := norm_linearForm_le _ _
      _ ≤ B*jetBound (n+1)*jetBound m/h^(n+1+m) + B*jetBound n*jetBound (m+1)/h^(n+(m+1)) :=
        add_le_add (physicalJet_abs_le (n+1) m (by omega) (by omega) C o z h B hh hB hC)
          (physicalJet_abs_le n (m+1) (by omega) (by omega) C o z h B hh hB hC)
      _ = _ := by
        rw [show n+1+m = n+m+1 by omega, show n+(m+1) = n+m+1 by omega]
        ring
  simpa only [Real.norm_eq_abs] using
    Convex.norm_image_sub_le_of_norm_hasFDerivWithin_le
      (fun z (_ : z ∈ (univ : Set Coords)) =>
        (hasFDerivAt_physicalJet n m (by omega) (by omega) C o z h).hasFDerivWithinAt)
      (fun z _ => hb z) convex_univ (mem_univ q) (mem_univ q')


theorem quadraticForm_sub (xx xy yy xx' xy' yy' : ℝ) :
    quadraticForm xx xy yy - quadraticForm xx' xy' yy' =
      quadraticForm (xx-xx') (xy-xy') (yy-yy') := by
  apply ContinuousLinearMap.ext
  intro v
  apply ContinuousLinearMap.ext
  intro w
  simp only [quadraticForm, linearForm, ContinuousLinearMap.sub_apply,
    ContinuousLinearMap.add_apply, ContinuousLinearMap.smulRight_apply,
    ContinuousLinearMap.smul_apply, smul_eq_mul]
  ring


/-- Global Hessian query stability, with an explicit coefficient-derived constant. -/
theorem physicalHessian_query_error_bound (C : Lattice) (o q q' : Coords) (h B : ℝ)
    (hh : 0 < h) (hB : 0 ≤ B) (hC : ∀ i j, |C i j| ≤ B) :
    ‖physicalHessian C o h q' - physicalHessian C o h q‖ ≤ 32*B/h^3 * ‖q'-q‖ := by
  have hxx : |physicalJet 2 0 C o h q' - physicalJet 2 0 C o h q| ≤
      (32/3 : ℝ)*B/h^3 * ‖q'-q‖ := by
    have ht := physicalJet_query_error_bound 2 0 (by norm_num) (by norm_num) C o q q' h B hh hB hC
    norm_num [jetBound] at ht
    simpa [mul_comm] using ht
  have hxy : |physicalJet 1 1 C o h q' - physicalJet 1 1 C o h q| ≤
      (16/3 : ℝ)*B/h^3 * ‖q'-q‖ := by
    have ht := physicalJet_query_error_bound 1 1 (by norm_num) (by norm_num) C o q q' h B hh hB hC
    norm_num [jetBound] at ht
    simpa [mul_comm] using ht
  have hyy : |physicalJet 0 2 C o h q' - physicalJet 0 2 C o h q| ≤
      (32/3 : ℝ)*B/h^3 * ‖q'-q‖ := by
    have ht := physicalJet_query_error_bound 0 2 (by norm_num) (by norm_num) C o q q' h B hh hB hC
    norm_num [jetBound] at ht
    simpa [mul_comm] using ht
  rw [physicalHessian, physicalHessian, quadraticForm_sub]
  refine (norm_quadraticForm_le _ _ _).trans ?_
  convert add_le_add (add_le_add hxx (mul_le_mul_of_nonneg_left hxy (show (0 : ℝ) ≤ 2 by norm_num))) hyy using 1
  ring

end FractalSync.OperationalField
