import FractalSync.ConcretePerturbedRungeKuttaWork

/-!
# Four observed RK4 substeps with one held drive

An outer step retains the drive evaluated at its starting phase through
all four substeps. The phase defect compares with the outer anchor's
continuous field, and the signed work defect telescopes through the
internal endpoints before the outer work accumulator is rounded.
-/

noncomputable section

namespace FractalSync.ConcreteFourSubstepRungeKutta

open Set Metric FlatShoreIntrinsic ScaleMetric OperationalDynamics ConcreteOperationalAsset
open ConcreteOperationalDynamics (acc)
open ConcreteRungeKutta (heldField exists_inner_reference)
open ConcreteHeldControl ConcreteHeldPower ConcreteWorkIntegral
open ImprovedConcreteDynamics (horizon horizon_pos)
open RungeKutta.Perturbed

local instance : NormedSpace ℝ (Plane × Plane) := Prod.normedSpace

local notation "outerDt" => refinementStep horizon

/-- The exact real time assigned to one of the four internal substeps. -/
def substepDt (N : ℕ) : ℝ := outerDt N / 4

/-- Four recorded substeps with a shared outer-anchor drive. -/
structure OuterStep (N : ℕ) (anchor endpoint : Plane × Plane) where
  states : ℕ → Plane × Plane
  stages : ℕ → Stages (Plane × Plane)
  initial : states 0 = anchor
  final : states 4 = endpoint
  next : ∀ j < 4, (stages j).endpoint = states (j + 1)
  errors : ∀ j < 4,
    Residuals (heldField anchor) (substepDt N) (states j) (stages j)
      (substepDt N) ((substepDt N) ^ 2) ((substepDt N) ^ 2)

private theorem outerDt_pos (N : ℕ) : 0 < outerDt N :=
  div_pos horizon_pos (by positivity)

private theorem substepDt_pos (N : ℕ) : 0 < substepDt N :=
  div_pos (outerDt_pos N) (by norm_num)

private theorem inner_room (N : ℕ) (hN : 15 ≤ N) :
    substepDt N * (10 + substepDt N) + (substepDt N) ^ 2 + (substepDt N) ^ 2 ≤
      12 * substepDt N := by
  have hsmall : substepDt N ≤ 2 / 3 := by
    unfold substepDt
    linarith [ConcretePerturbedRungeKutta.mesh_small N hN]
  nlinarith [mul_nonneg (substepDt_pos N).le (sub_nonneg.mpr hsmall)]

private theorem substep_certificate {N : ℕ} (hN : 15 ≤ N)
    {anchor endpoint : Plane × Plane} (r : OuterStep N anchor endpoint)
    (ha : anchor ∈ ball phase (radius / 2)) :
    ∀ j < 4, StageBounds (r.stages j) (substepDt N) (r.states j) 10
      (substepDt N) ((substepDt N) ^ 2) ((substepDt N) ^ 2) ∧
      StageRegion (r.stages j) (r.states j) anchor (12 * outerDt N) := by
  have hh := (substepDt_pos N).le
  have hR : 12 * outerDt N ≤ 18 * outerDt N := by
    linarith [(outerDt_pos N).le]
  have hf (y : Plane × Plane) (hy : y ∈ closedBall anchor (12 * outerDt N)) :
      ‖heldField anchor y‖ ≤ 10 := by
    have hya := mem_closedBall.mp hy
    have hy' := ConcretePerturbedRungeKutta.local_ball_subset N hN anchor ha
      (closedBall_subset_closedBall hR hy)
    exact (ConcreteRungeKutta.heldField_norm_le anchor y
      (ball_subset_closedBall (ball_subset_ball (by norm_num [radius]) ha))
      (ball_subset_closedBall hy')).trans
        (by linarith [ConcretePerturbedRungeKutta.mesh_small N hN])
  have hlocal (j : ℕ) (hj : j < 4)
      (hd : ‖r.states j - anchor‖ ≤ (j : ℝ) * (12 * substepDt N)) :
      StageBounds (r.stages j) (substepDt N) (r.states j) 10
        (substepDt N) ((substepDt N) ^ 2) ((substepDt N) ^ 2) ∧
      StageRegion (r.stages j) (r.states j) anchor (12 * outerDt N) := by
    have hj' : (j : ℝ) ≤ 3 := by exact_mod_cast (show j ≤ 3 by omega)
    have hmargin := mul_le_mul_of_nonneg_right hj' (mul_nonneg (by norm_num : (0 : ℝ) ≤ 12) hh)
    have hroom : ‖r.states j - anchor‖ + substepDt N * (10 + substepDt N) +
        (substepDt N) ^ 2 + (substepDt N) ^ 2 ≤ 12 * outerDt N := by
      linarith [inner_room N hN, show 4 * substepDt N = outerDt N by unfold substepDt; ring]
    have hb := stageBounds (r.errors j hj) hh (by norm_num) hh (sq_nonneg _)
      (sq_nonneg _) hf hroom
    exact ⟨hb, hb.region hh (by norm_num) hh (sq_nonneg _) (sq_nonneg _) hroom⟩
  have hprefix (j : ℕ) (hj : j ≤ 4) :
      ‖r.states j - anchor‖ ≤ (j : ℝ) * (12 * substepDt N) := by
    induction j with
    | zero => simp [r.initial]
    | succ j ih =>
      have hlt : j < 4 := hj
      have hprev := ih hlt.le
      have hb := (hlocal j hlt hprev).1
      rw [← r.next j hlt]
      exact (norm_sub_le_norm_sub_add_norm_sub (r.stages j).endpoint (r.states j) anchor).trans
        (by rw [Nat.cast_succ]; linarith [hb.endpoint_displacement, inner_room N hN,
          sq_nonneg (substepDt N)])
  exact fun j hj => hlocal j hj (hprefix j hj.le)

/-- The four endpoints compose to a single ordered-step defect at the outer anchor. -/
theorem OuterStep.ordered_defect {N : ℕ} (hN : 15 ≤ N)
    {anchor endpoint : Plane × Plane} (r : OuterStep N anchor endpoint)
    (ha : anchor ∈ ball phase (radius / 2)) :
    ‖endpoint - orderedStep acc (outerDt N) anchor‖ ≤ (outerDt N) ^ 2 * 18000 := by
  let h := substepDt N
  let H := outerDt N
  let v := phaseField acc anchor
  have hh : 0 ≤ h := (substepDt_pos N).le
  have hH : 0 ≤ H := (outerDt_pos N).le
  have hfour : 4 * h = H := by dsimp [h, H, substepDt]; ring
  have hR : 12 * H ≤ 18 * H := by linarith
  have ha' : anchor ∈ closedBall phase radius :=
    ball_subset_closedBall (ball_subset_ball (by norm_num [radius]) ha)
  have hslope (y k : Plane × Plane) (hy : y ∈ closedBall anchor (12 * H))
      (he : ‖k - heldField anchor y‖ ≤ h) :
      ‖k - v‖ ≤ h + 1080 * (12 * H) := by
    have hy' := ConcretePerturbedRungeKutta.local_ball_subset N hN anchor ha
      (closedBall_subset_closedBall hR hy)
    exact (norm_sub_le_norm_sub_add_norm_sub k (heldField anchor y) v).trans
      (add_le_add he (((ConcreteRungeKutta.heldField_from_anchor anchor y ha'
        (ball_subset_closedBall hy')).trans
        (mul_le_mul_of_nonneg_left (mem_closedBall.mp hy) (by norm_num)))))
  have hlocal (j : ℕ) (hj : j < 4) :
      ‖r.states (j + 1) - (r.states j + h • v)‖ ≤ h * (h + 1080 * (12 * H)) + h ^ 2 := by
    have hr := (substep_certificate hN r ha j hj).2
    have he := r.errors j hj
    have hm : ‖(r.stages j).meanSlope - v‖ ≤ h + 1080 * (12 * H) :=
      RungeKutta.weighted_sub_norm_le
        (hslope _ _ hr.start_mem he.slope0_error)
        (hslope _ _ hr.stage2_mem he.slope2_error)
        (hslope _ _ hr.stage3_mem he.slope3_error)
        (hslope _ _ hr.stage4_mem he.slope4_error)
    have hi : ‖(r.states j + h • (r.stages j).meanSlope) - (r.states j + h • v)‖ ≤
        h * (h + 1080 * (12 * H)) := by
      simpa only [add_sub_add_left_eq_sub, ← smul_sub, norm_smul, Real.norm_eq_abs,
        abs_of_nonneg hh] using mul_le_mul_of_nonneg_left hm hh
    rw [← r.next j hj]
    exact ((norm_sub_le_norm_sub_add_norm_sub (r.stages j).endpoint
      (r.states j + h • (r.stages j).meanSlope) (r.states j + h • v)).trans
      (add_le_add he.endpoint_error hi)).trans_eq (add_comm _ _)
  have hprefix (j : ℕ) (hj : j ≤ 4) :
      ‖r.states j - (anchor + ((j : ℝ) * h) • v)‖ ≤
        (j : ℝ) * (h * (h + 1080 * (12 * H)) + h ^ 2) := by
    induction j with
    | zero => simp [r.initial]
    | succ j ih =>
      have hj' : j < 4 := hj
      have hc :
          (r.states j + h • v) - (anchor + ((↑(j + 1) : ℝ) * h) • v) =
            r.states j - (anchor + ((j : ℝ) * h) • v) := by
        rw [Nat.cast_succ, add_mul, one_mul, add_smul, ← add_assoc, add_sub_add_right_eq_sub]
      refine (norm_sub_le_norm_sub_add_norm_sub (r.states (j + 1))
        (r.states j + h • v) (anchor + ((↑(j + 1) : ℝ) * h) • v)).trans ?_
      rw [hc]
      exact (add_le_add (hlocal j hj') (ih hj'.le)).trans_eq (by rw [Nat.cast_succ]; ring)
  have heuler := hprefix 4 le_rfl
  rw [r.final, Nat.cast_ofNat, hfour] at heuler
  have hgap : ‖(anchor + H • v) - orderedStep acc H anchor‖ = H ^ 2 * ‖acc anchor‖ := by
    simp [orderedStep_eq, v, norm_smul, Real.norm_eq_abs]
    positivity
  have htotal := (norm_sub_le_norm_sub_add_norm_sub endpoint
    (anchor + H • v) (orderedStep acc H anchor)).trans (add_le_add heuler hgap.le)
  have ha8 := mul_le_mul_of_nonneg_left (ImprovedConcreteDynamics.acc_bound.norm_le anchor ha')
    (sq_nonneg H)
  have hhexact : h = H / 4 := by dsimp [h, H, substepDt]
  rw [hhexact] at htotal
  exact htotal.trans (by nlinarith)

/-- The stored outer work increment is compared with the sum of the four
weighted power samples, including all machine accumulation errors. -/
structure OuterWork {N : ℕ} {anchor endpoint : Plane × Plane}
    (r : OuterStep N anchor endpoint) (increment : ℝ) where
  powers : ℕ → Powers
  evaluations : ∀ j < 4,
    PowerResiduals (heldPower (drive anchor)) (r.states j) (r.stages j)
      (powers j) (substepDt N)
  accumulation : |increment - ∑ j ∈ Finset.range 4, (powers j).work (substepDt N)| ≤
    (outerDt N) ^ 2

/-- The inner defects telescope through the actual substep endpoints. -/
theorem OuterWork.energy_defect {N : ℕ} (hN : 15 ≤ N)
    {anchor endpoint : Plane × Plane} {r : OuterStep N anchor endpoint}
    {increment : ℝ} (w : OuterWork r increment)
    (ha : anchor ∈ ball phase (radius / 2)) :
    |energy endpoint - energy anchor - increment| ≤ 65536 * (outerDt N) ^ 2 := by
  let h := substepDt N
  let H := outerDt N
  have hh : 0 ≤ h := (substepDt_pos N).le
  have hR : 12 * H ≤ 18 * H := by linarith [(outerDt_pos N).le]
  have hmem : closedBall anchor (12 * H) ⊆ closedBall phase radius :=
    fun y hy => ball_subset_closedBall (ConcretePerturbedRungeKutta.local_ball_subset N hN
      anchor ha (closedBall_subset_closedBall hR hy))
  have hf (y : Plane × Plane) (hy : y ∈ closedBall anchor (12 * H)) :
      ‖heldField anchor y‖ ≤ 10 :=
    (ConcreteRungeKutta.heldField_norm_le anchor y
      (ball_subset_closedBall (ball_subset_ball (by norm_num [radius]) ha)) (hmem hy)).trans
      (by linarith [mem_closedBall.mp hy, ConcretePerturbedRungeKutta.mesh_small N hN])
  have hD : h * (10 + h) + h ^ 2 ≤ 12 * h := by
    linarith [inner_room N hN, sq_nonneg h]
  have hlocal (j : Fin 4) : |energy (r.states (j.val + 1)) - energy (r.states j.val) -
      (w.powers j.val).work h| ≤ 65536 * h ^ 2 := by
    obtain ⟨hb, hr⟩ := substep_certificate hN r ha j j.isLt
    rw [← r.next j j.isLt]
    have he := energy_work_defect hb hr (r.errors j j.isLt) (w.evaluations j j.isLt)
      hh (by norm_num) hh (show 0 ≤ substepDt N ^ 2 from sq_nonneg _)
      (show (0 : ℝ) ≤ 134 by norm_num) hf
      (fun y hy => energy_differentiable y (hmem hy))
      (energy_differential_bound.norm_le _ (hmem hr.start_mem))
      (fun y hy z hz => by simpa only [dist_eq_norm] using
        energy_differential_bound.norm_sub_le y (hmem hy) z (hmem hz))
      (fun y hy => (fderiv_energy_heldField (drive anchor) y (hmem hy)).symm)
      (show |(w.powers j.val).work h - (w.powers j.val).work h| ≤ 0 by simp)
    have hsquare := pow_le_pow_left₀ (show 0 ≤ h * (10 + h) + h ^ 2 by positivity) hD 2
    exact he.trans (by nlinarith [hsquare, mul_le_mul_of_nonneg_left hD hh])
  have hsum := ScaleMetric.accumulated_energy_defect 4 (fun j => energy (r.states j))
    (fun j => (w.powers j.val).work h) (fun _ => 65536 * h ^ 2) hlocal
  simp only [Fin.sum_univ_eq_sum_range (fun j => (w.powers j).work h) 4,
    Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, Nat.cast_ofNat,
    r.initial, r.final] at hsum
  have htotal := (abs_sub_le (energy endpoint - energy anchor)
    (∑ j ∈ Finset.range 4, (w.powers j).work h) increment).trans
      (add_le_add hsum (by simpa only [abs_sub_comm] using w.accumulation))
  have hhexact : h = H / 4 := by dsimp [h, H, substepDt]
  rw [hhexact] at htotal
  exact htotal.trans (by nlinarith [sq_nonneg H])

/-- Observed outer anchors and stored work on the same exact-time mesh. -/
structure History (N : ℕ) where
  states : ℕ → Plane × Plane
  work : ℕ → ℝ
  initial : states 0 = phase
  initial_work : work 0 = 0
  step : ∀ i < N + 1, OuterStep N (states i) (states (i + 1))
  step_work : ∀ i (hi : i < N + 1), OuterWork (step i hi) (work (i + 1) - work i)

/-- One reference bounds both outer-mesh interpolants on the entire horizon. -/
theorem exists_phase_work_certificate : ∃ z : ℝ → Plane × Plane, z 0 = phase ∧
    (∀ t ∈ Icc (0 : ℝ) horizon, HasDerivAt z (phaseField acc (z t)) t) ∧
    (∀ t ∈ Icc (0 : ℝ) horizon, z t ∈ closedBall phase radius) ∧
    ∀ N : ℕ, 15 ≤ N → ∀ r : History N,
      ∀ t ∈ Icc (0 : ℝ) horizon,
        ‖trajectoryInterpolant (outerDt N) r.states (N + 1) t - z t‖ ≤
          3 / (16 * ((N : ℝ) + 1)) ∧
        |trajectoryInterpolant (outerDt N) r.work (N + 1) t - referenceWork z t| ≤
          11 / (9 * ((N : ℝ) + 1)) := by
  obtain ⟨z, hz0, c, hcL, hcB, htube⟩ := exists_inner_reference
  have hzmem (t : ℝ) (ht : t ∈ Icc (0 : ℝ) horizon) : z t ∈ closedBall phase radius :=
    ball_subset_closedBall (ball_subset_ball (by norm_num [radius]) (c.reference_mem t ht))
  refine ⟨z, hz0, c.equation, hzmem, ?_⟩
  intro N hN r
  have hfit : outerDt N *
      (refinementCoefficient horizon c.accelerationLip c.phaseSpeed 0 18000 0 +
        (c.phaseSpeed : ℝ) / 2) ≤ radius / 4 := by
    simpa only [hcL, hcB, NNReal.coe_ofNat, ConcretePerturbedRungeKutta.coefficient]
      using ConcretePerturbedRungeKutta.mesh_fits N hN
  have hbound := c.interpolation_error_bound horizon_pos N r.states (fun i => acc (r.states i))
    0 18000 0 (radius / 4) (by simp [r.initial, hz0]) (fun i _ _ => by simp)
    (fun i hi ha => by simpa only [NNReal.coe_ofNat] using (r.step i hi).ordered_defect hN ha)
    (fun t ht => (htube t ht).2) hfit
  simp only [hcL, hcB, NNReal.coe_ofNat, NNReal.coe_zero] at hbound
  have hphase (t : ℝ) (ht : t ∈ Icc (0 : ℝ) horizon) :
      ‖trajectoryInterpolant (outerDt N) r.states (N + 1) t - z t‖ ≤ 3 / (16 * ((N : ℝ) + 1)) :=
    ((hbound t ht).1).trans (ConcretePerturbedRungeKutta.mesh_error_le N)
  have hh := outerDt_pos N
  have hspan := (refinementStep_properties horizon N horizon_pos.le).2.2
  have htime (i : ℕ) (hi : i ≤ N + 1) : (i : ℝ) * outerDt N ∈ Icc (0 : ℝ) horizon :=
    ⟨mul_nonneg (Nat.cast_nonneg i) hh.le,
      (mul_le_mul_of_nonneg_right (Nat.cast_le.mpr hi) hh.le).trans_eq hspan⟩
  have hregion (i : ℕ) (hi : i ≤ N + 1) : r.states i ∈ ball phase (radius / 2) := by
    simpa only [trajectoryInterpolant_grid _ r.states hh (N + 1) i hi] using
      ((hbound _ (htime i hi)).2)
  have hwork_grid (i : ℕ) (hi : i ≤ N + 1) :
      ‖r.work i - referenceWork z ((i : ℝ) * outerDt N)‖ ≤ 19 / (16 * ((N : ℝ) + 1)) := by
    have hlocal (j : Fin i) :
        |energy (r.states (j.val + 1)) - energy (r.states j.val) -
          (r.work (j.val + 1) - r.work j.val)| ≤ 65536 * (outerDt N) ^ 2 :=
      (r.step_work j (j.isLt.trans_le hi)).energy_defect hN (hregion j (j.isLt.le.trans hi))
    have hsum := ScaleMetric.accumulated_energy_defect i (fun j => energy (r.states j))
      (fun j => r.work (j.val + 1) - r.work j.val) (fun _ => 65536 * (outerDt N) ^ 2) hlocal
    simp only [Fin.sum_univ_eq_sum_range (fun j => r.work (j + 1) - r.work j) i,
      Finset.sum_range_sub, Finset.sum_const, Finset.card_univ, Fintype.card_fin,
      nsmul_eq_mul, r.initial, r.initial_work, sub_zero] at hsum
    have hledger : |energy (r.states i) - energy phase - r.work i| ≤
        1 / (4 * ((N : ℝ) + 1)) := by
      refine hsum.trans ((mul_le_mul_of_nonneg_right
        (show (i : ℝ) ≤ (N : ℝ) + 1 by exact_mod_cast hi)
        (show 0 ≤ 65536 * (outerDt N) ^ 2 by positivity)).trans ?_)
      norm_num [refinementStep, horizon, localTrajectoryHorizon]
      field_simp
      norm_num
    have hphase_grid : ‖r.states i - z ((i : ℝ) * outerDt N)‖ ≤ 3 / (16 * ((N : ℝ) + 1)) := by
      simpa only [trajectoryInterpolant_grid _ r.states hh (N + 1) i hi] using hphase _ (htime i hi)
    have hE : |energy (r.states i) - energy (z ((i : ℝ) * outerDt N))| ≤
        5 * ‖r.states i - z ((i : ℝ) * outerDt N)‖ := by
      simpa only [dist_eq_norm] using ImprovedConcreteDynamics.energy_sub_le _ _
        (ball_subset_closedBall (ball_subset_ball (by norm_num [radius]) (hregion i hi)))
        (hzmem _ (htime i hi))
    have hE' := hE.trans (mul_le_mul_of_nonneg_left hphase_grid (by norm_num : (0 : ℝ) ≤ 5))
    rw [Real.norm_eq_abs, referenceWork_eq_energy_sub z horizon c.equation hzmem _ (htime i hi), hz0]
    refine ((abs_sub_le (r.work i) (energy (r.states i) - energy phase)
      (energy (z ((i : ℝ) * outerDt N)) - energy phase)).trans
      (add_le_add (by simpa only [abs_sub_comm] using hledger)
        (by simpa only [sub_sub_sub_cancel_right] using hE'))).trans_eq ?_
    field_simp
    norm_num
  have hwork := trajectoryInterpolant_error (outerDt N)
    r.work (referenceWork z) (19 / (16 * ((N : ℝ) + 1))) 40 hh (N + 1) hwork_grid
    (by simpa only [hspan] using referenceWork_lipschitzOn z horizon c.equation hzmem)
  refine fun t ht => ⟨hphase t ht, (hwork t (by simpa only [hspan] using ht)).trans_eq ?_⟩
  norm_num [refinementStep, horizon, localTrajectoryHorizon, div_eq_mul_inv, mul_inv_rev]
  ring

/-- The sixteen-outer-step mesh used by the recorded quadratic fixture. -/
theorem exists_fixture_certificate : ∃ z : ℝ → Plane × Plane, z 0 = phase ∧
    (∀ t ∈ Icc (0 : ℝ) horizon, HasDerivAt z (phaseField acc (z t)) t) ∧
    ∀ r : History 15, ∀ t ∈ Icc (0 : ℝ) horizon,
      ‖trajectoryInterpolant (outerDt 15) r.states 16 t - z t‖ ≤ 3 / 256 ∧
      |trajectoryInterpolant (outerDt 15) r.work 16 t - referenceWork z t| ≤ 11 / 144 := by
  obtain ⟨z, hz0, heq, _, hc⟩ := exists_phase_work_certificate
  refine ⟨z, hz0, heq, fun r t ht => ?_⟩
  convert hc 15 le_rfl r t ht <;> norm_num

end FractalSync.ConcreteFourSubstepRungeKutta
