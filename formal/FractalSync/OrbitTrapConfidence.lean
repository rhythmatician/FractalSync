import FractalSync.OrbitTrapSteering

/-!
# Sharp gradient-error gate for oriented trap steering

The error budget is an input. These results concern exact real arithmetic.
-/

namespace FractalSync.OrbitTrapConfidence

open OrbitTrapSteering

noncomputable section

def errorSq (estimate actual : Point) : ℝ :=
  (estimate.1 - actual.1) ^ 2 + (estimate.2 - actual.2) ^ 2

def determinant (estimate actual : Point) : ℝ :=
  estimate.1 * actual.2 - estimate.2 * actual.1

private theorem errorSq_eq (u v : Point) :
    errorSq u v = pairing u u - 2 * pairing u v + pairing v v := by
  unfold errorSq pairing
  ring

theorem dot_det_sq (u v : Point) :
    pairing u v ^ 2 + determinant u v ^ 2 = pairing u u * pairing v v := by
  unfold pairing determinant
  ring

theorem error_projection_sq (u v : Point) :
    pairing v v * errorSq u v =
      (pairing v v - pairing u v) ^ 2 + determinant u v ^ 2 := by
  unfold pairing errorSq determinant
  ring

/-- An error ball missing the origin lies in the forward half-plane.
The squared determinant bound then describes its sharp angular cone. -/
theorem gradient_cone (u v : Point) (E : ℝ)
    (hE : E < pairing u u) (herr : errorSq u v ≤ E) :
    0 < pairing u v ∧ determinant u v ^ 2 ≤ E * pairing v v := by
  have hnorm : 0 ≤ pairing v v := by
    unfold pairing
    nlinarith [sq_nonneg v.1, sq_nonneg v.2]
  constructor
  · by_contra! hd
    linarith [errorSq_eq u v]
  · have hprod := mul_le_mul_of_nonneg_left herr hnorm
    nlinarith only [hprod, error_projection_sq u v, sq_nonneg (pairing v v - pairing u v)]

/-- This boundary gradient has both error exactly `E` and squared
determinant exactly `E` times its squared norm. -/
theorem gradient_cone_attained (u : Point) (E : ℝ)
    (hE0 : 0 ≤ E) (hE : E < pairing u u) :
    ∃ v : Point, errorSq u v = E ∧
      pairing u v = pairing u u - E ∧ pairing v v = pairing u u - E ∧
      determinant u v ^ 2 = E * pairing v v := by
  let M := pairing u u
  let A := M - E
  let B := Real.sqrt (E * A)
  have hm : 0 < M := lt_of_le_of_lt hE0 hE
  have ha : 0 < A := sub_pos.mpr hE
  have hb : B ^ 2 = E * A := Real.sq_sqrt (mul_nonneg hE0 ha.le)
  let v : Point := ((A * u.1 - B * u.2) / M, (A * u.2 + B * u.1) / M)
  have hdot : pairing u v = A := by
    change u.1 * ((A * u.1 - B * u.2) / M) +
      u.2 * ((A * u.2 + B * u.1) / M) = A
    field_simp [hm.ne']
    dsimp [M, pairing]
    ring
  have hdet : determinant u v = B := by
    change u.1 * ((A * u.2 + B * u.1) / M) -
      u.2 * ((A * u.1 - B * u.2) / M) = B
    field_simp [hm.ne']
    dsimp [M, pairing]
    ring
  have hnorm : pairing v v = A := by
    have h := dot_det_sq u v
    rw [hdot, hdet, hb] at h
    change A ^ 2 + E * A = M * pairing v v at h
    have hmul : M * pairing v v = M * A := by
      rw [← h]
      dsimp [A]
      ring
    exact mul_left_cancel₀ hm.ne' hmul
  refine ⟨v, ?_, hdot, hnorm, ?_⟩
  · rw [errorSq_eq, hdot, hnorm]
    change M - 2 * A + A = E
    dsimp [A]
    ring
  · rw [hdet, hb, hnorm]

/-- Squared cosine target `k` is guaranteed for every gradient in the
closed error ball exactly when this square-root-free threshold passes. -/
theorem gradient_cosine_iff (u : Point) (E k : ℝ)
    (hE0 : 0 ≤ E) (hE : E < pairing u u) :
    (∀ v : Point, errorSq u v ≤ E →
      0 < pairing u v ∧ k * (pairing u u * pairing v v) ≤ pairing u v ^ 2) ↔
      E ≤ (1 - k) * pairing u u := by
  constructor
  · intro h
    obtain ⟨v, herr, _, hnorm, hdet⟩ := gradient_cone_attained u E hE0 hE
    have hcos := (h v herr.le).2
    rw [← sub_eq_iff_eq_add.mpr (dot_det_sq u v).symm, hdet] at hcos
    have hmul : E * pairing v v ≤ ((1 - k) * pairing u u) * pairing v v := by
      nlinarith only [hcos]
    have hn : 0 < pairing v v := by
      rw [hnorm]
      exact sub_pos.mpr hE
    exact (mul_le_mul_iff_left₀ hn).mp hmul
  · intro hthreshold v herr
    obtain ⟨hpos, hdet⟩ := gradient_cone u v E hE herr
    have hn : 0 ≤ pairing v v := by
      unfold pairing
      nlinarith [sq_nonneg v.1, sq_nonneg v.2]
    have hbound := mul_le_mul_of_nonneg_right hthreshold hn
    have h := dot_det_sq u v
    exact ⟨hpos, by nlinarith only [hdet, hbound, h]⟩

/-- Once the error ball contains the origin, it includes a critical candidate. -/
theorem gradient_gate_contains_zero (u : Point) (E : ℝ)
    (hE : pairing u u ≤ E) : errorSq u (0, 0) ≤ E := by
  unfold errorSq pairing at *
  nlinarith

/-- The two runtime comparisons characterize the certificate over the entire
error ball, including the strict guard that excludes the zero candidate. -/
theorem gradient_gate_iff (u : Point) (E k : ℝ) (hE0 : 0 ≤ E) :
    (∀ v : Point, errorSq u v ≤ E →
      0 < pairing u v ∧ k * (pairing u u * pairing v v) ≤ pairing u v ^ 2) ↔
      E < pairing u u ∧ E ≤ (1 - k) * pairing u u := by
  constructor
  · intro h
    have hE : E < pairing u u := by
      by_contra! hbad
      have hdot := (h (0, 0) (gradient_gate_contains_zero u E hbad)).1
      simp [pairing] at hdot
    exact ⟨hE, (gradient_cosine_iff u E k hE0 hE).mp h⟩
  · rintro ⟨hE, hthreshold⟩
    exact (gradient_cosine_iff u E k hE0 hE).mpr hthreshold

end

end FractalSync.OrbitTrapConfidence
