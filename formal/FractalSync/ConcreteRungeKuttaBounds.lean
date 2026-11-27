import FractalSync.ConcreteHeldControl

/-!
# Concrete enclosing balls for held-control Runge--Kutta stages

The anchor is the beginning of the complete outer step. The drive covector
remains fixed at this anchor, including across internal RK substeps. The
bounds use local smooth-field estimates and no distance-to-set identity.
-/

noncomputable section

namespace FractalSync.ConcreteRungeKutta

open Set Metric FlatShoreIntrinsic ScaleMetric OperationalDynamics ConcreteOperationalAsset
open ConcreteOperationalDynamics (acc)
open ConcreteHeldControl
open ImprovedConcreteDynamics (horizon horizon_pos)

local instance concreteRKPhaseNormedSpace : NormedSpace ℝ (Plane × Plane) :=
  Prod.normedSpace

/-- The exact-real phase field with the drive held at the outer-step anchor. -/
def heldField (sample : Plane × Plane) : Plane × Plane → Plane × Plane :=
  phaseField (frozenAcceleration (drive sample))

/-- Variation from the reevaluated acceleration at the outer-step anchor. -/
theorem acceleration_from_anchor (sample current : Plane × Plane)
    (hsample : sample ∈ closedBall phase radius) (hcurrent : current ∈ closedBall phase radius) :
    ‖frozenAcceleration (drive sample) current - acc sample‖ ≤ 1080 * dist current sample := by
  calc
    _ ≤ ‖frozenAcceleration (drive sample) current - acc current‖ +
        ‖acc current - acc sample‖ := norm_sub_le_norm_sub_add_norm_sub _ _ _
    _ ≤ 680 * dist sample current + 400 * dist current sample :=
      add_le_add (held_acceleration_sub_le sample current hsample hcurrent)
        (ImprovedConcreteDynamics.acc_bound.norm_sub_le current hcurrent sample hsample)
    _ = 1080 * dist current sample := by rw [dist_comm sample current]; ring

/-- The same anchor bound controls the complete phase vector in its maximum norm. -/
theorem heldField_from_anchor (sample current : Plane × Plane)
    (hsample : sample ∈ closedBall phase radius) (hcurrent : current ∈ closedBall phase radius) :
    ‖heldField sample current - phaseField acc sample‖ ≤ 1080 * dist current sample := by
  exact max_le ((norm_snd_le (current - sample)).trans (by
    rw [dist_eq_norm]
    nlinarith [norm_nonneg (current - sample)]))
    (acceleration_from_anchor sample current hsample hcurrent)

/-- The held field is bounded by its actual value at the anchor and local variation. -/
theorem heldField_norm_le (sample current : Plane × Plane)
    (hsample : sample ∈ closedBall phase radius) (hcurrent : current ∈ closedBall phase radius) :
    ‖heldField sample current‖ ≤ 8 + 1080 * dist current sample := by
  calc
    _ = ‖(heldField sample current - phaseField acc sample) + phaseField acc sample‖ :=
      congrArg norm (sub_add_cancel _ _).symm
    _ ≤ ‖heldField sample current - phaseField acc sample‖ + ‖phaseField acc sample‖ :=
      norm_add_le _ _
    _ ≤ 1080 * dist current sample + 8 := add_le_add
      (heldField_from_anchor sample current hsample hcurrent)
      (ImprovedConcreteDynamics.phaseField_bound sample hsample)
    _ = 8 + 1080 * dist current sample := add_comm _ _

/-- An inner anchor leaves room for every stage in the complete outer step. -/
theorem stageBall_subset (sample : Plane × Plane) (h : ℝ)
    (hsample : sample ∈ ball phase (radius / 2)) (hfit : 16 * h ≤ radius / 2) :
    closedBall sample (16 * h) ⊆ ball phase radius := by
  exact fun current hcurrent => mem_ball_iff_norm.mpr
    ((norm_sub_le_norm_sub_add_norm_sub current sample phase).trans_lt (by
      have hc := mem_closedBall_iff_norm.mp hcurrent
      have hs := mem_ball_iff_norm.mp hsample
      linarith))

/-- The enclosing-ball speed bound closes from the actual held field itself. -/
theorem heldField_stageBall_bound (sample : Plane × Plane) (h : ℝ)
    (hsample : sample ∈ ball phase (radius / 2)) (hfit : 16 * h ≤ radius / 2)
    (hsmall : h ≤ 1 / 2160) :
    ∀ current ∈ closedBall sample (16 * h), ‖heldField sample current‖ ≤ 16 := by
  exact fun current hcurrent =>
    (heldField_norm_le sample current
      (ball_subset_closedBall (ball_subset_ball (by norm_num [radius]) hsample))
      (ball_subset_closedBall (stageBall_subset sample h hsample hfit hcurrent))).trans (by
        have hc := mem_closedBall.mp hcurrent
        linarith)

/-- The existing actual solution has a smaller tube, leaving an outer stage margin. -/
theorem exists_inner_reference : ∃ z : ℝ → Plane × Plane, z 0 = phase ∧
    ∃ c : UniformODECertificate acc z horizon (ball phase (radius / 2)),
      c.accelerationLip = 400 ∧ c.phaseSpeed = 8 ∧
      ∀ t ∈ Icc (0 : ℝ) horizon, ‖z t - phase‖ ≤ 8 * t ∧
        ∀ w : Plane × Plane, ‖w - z t‖ ≤ radius / 4 → w ∈ ball phase (radius / 2) := by
  obtain ⟨z, hz0, c, hcL, hcB, htube⟩ := ImprovedConcreteDynamics.exists_reference
  have hmove : ∀ t ∈ Icc (0 : ℝ) horizon, ‖z t - phase‖ < radius / 4 := by
    intro t ht
    have hd := (htube t ht).1
    have htT := ht.2
    norm_num [ImprovedConcreteDynamics.horizon, localTrajectoryHorizon, radius] at htT ⊢
    linarith
  let c' : UniformODECertificate acc z horizon (ball phase (radius / 2)) := {
    accelerationLip := 400
    phaseSpeed := 8
    equation := c.equation
    reference_mem := fun t ht => mem_ball_iff_norm.mpr
      ((hmove t ht).trans_le (by norm_num [radius]))
    speed_bound := by simpa only [hcB] using c.speed_bound
    acceleration_lipschitz := fun p hp q hq => by
      simpa only [NNReal.coe_ofNat, dist_eq_norm] using
        ImprovedConcreteDynamics.acc_bound.norm_sub_le p
          (ball_subset_closedBall (ball_subset_ball (by norm_num [radius]) hp)) q
          (ball_subset_closedBall (ball_subset_ball (by norm_num [radius]) hq)) }
  exact ⟨z, hz0, c', rfl, rfl, fun t ht => ⟨(htube t ht).1, fun w hw =>
    mem_ball_iff_norm.mpr ((norm_sub_le_norm_sub_add_norm_sub w (z t) phase).trans_lt (by
      have hm := hmove t ht
      linarith))⟩⟩

end FractalSync.ConcreteRungeKutta
