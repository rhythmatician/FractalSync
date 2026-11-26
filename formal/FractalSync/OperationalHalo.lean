import FractalSync.OperationalDynamics

/-!
# Physical halo margins and independence of lattice extensions

Exact-real query displacements are controlled by the actual four-sided
physical halo. Agreement on a finite asset gives equality of all selected
query jets. On the open halo it also gives equality of actual ruler germs,
Riesz gradients, derivative Hessians, and mechanical accelerations.
-/

noncomputable section

namespace FractalSync.OperationalHalo

open OperationalField OperationalDynamics FlatShoreIntrinsic ScaleMetric Set Filter Metric
open scoped Topology

/-- Two arbitrary full-lattice extensions contain exactly the same finite asset. -/
def AgreeOnAsset (C D : Lattice) (Nx Ny : ℕ) : Prop :=
  ∀ i j, 0 ≤ i → i < (Nx : ℤ) → 0 ≤ j → j < (Ny : ℤ) → C i j = D i j

/-- One globally bounded representative of a finite asset, without any
condition on the original lattice outside the stored rectangle. -/
def zeroExtension (C : Lattice) (Nx Ny : ℕ) : Lattice :=
  fun i j => if 0 ≤ i ∧ i < (Nx : ℤ) ∧ 0 ≤ j ∧ j < (Ny : ℤ) then C i j else 0

theorem zeroExtension_agrees (C : Lattice) (Nx Ny : ℕ) :
    AgreeOnAsset (zeroExtension C Nx Ny) C Nx Ny := by
  intro i j hi hix hj hjy
  simp only [zeroExtension, hi, hix, hj, hjy, and_self, ↓reduceIte]

/-- A magnitude bound required only on the finite asset bounds the actual
zero extension at every integer lattice point. -/
theorem abs_zeroExtension_le (C : Lattice) (Nx Ny : ℕ) (B : ℝ) (hB : 0 ≤ B)
    (hC : ∀ i j, 0 ≤ i → i < (Nx : ℤ) → 0 ≤ j → j < (Ny : ℤ) → |C i j| ≤ B) :
    ∀ i j, |zeroExtension C Nx Ny i j| ≤ B := by
  intro i j
  unfold zeroExtension
  split_ifs with hij
  · exact hC i j hij.1 hij.2.1 hij.2.2.1 hij.2.2.2
  · simpa only [abs_zero] using hB

/-- The exact-real full-stencil query domain, including its lower faces. -/
def physicalHalo (Nx Ny : ℕ) (o : Coords) (h : ℝ) : Set Coords :=
  normalizedCoordinates o h ⁻¹' (Ico (2 : ℝ) ((Nx : ℝ) - 3) ×ˢ Ico (2 : ℝ) ((Ny : ℝ) - 3))

/-- The strict interior used for neighborhood and actual-derivative arguments. -/
def openPhysicalHalo (Nx Ny : ℕ) (o : Coords) (h : ℝ) : Set Coords :=
  normalizedCoordinates o h ⁻¹' (Ioo (2 : ℝ) ((Nx : ℝ) - 3) ×ˢ Ioo (2 : ℝ) ((Ny : ℝ) - 3))

/-- The smallest of the four signed physical distances to a halo face. -/
def haloMargin (Nx Ny : ℕ) (o : Coords) (h : ℝ) (q : Coords) : ℝ :=
  min (min (q.1 - (o.1 + 2 * h)) (o.1 + h * ((Nx : ℝ) - 3) - q.1))
    (min (q.2 - (o.2 + 2 * h)) (o.2 + h * ((Ny : ℝ) - 3) - q.2))

/-- A closed coordinate ball fits in the open physical halo whenever its
radius is smaller than each of the four physical face margins. -/
theorem closedBall_subset_openPhysicalHalo (Nx Ny : ℕ) (o q : Coords) (h R : ℝ)
    (hh : 0 < h) (hmargin : R < haloMargin Nx Ny o h q) :
    closedBall q R ⊆ openPhysicalHalo Nx Ny o h := by
  intro y hy
  have hxy : ‖y - q‖ ≤ R := mem_closedBall_iff_norm.mp hy
  have hdx : |y.1 - q.1| ≤ R := (norm_fst_le (y - q)).trans hxy
  have hdy : |y.2 - q.2| ≤ R := (norm_snd_le (y - q)).trans hxy
  rcases (show (R < q.1 - (o.1 + 2 * h) ∧ R < o.1 + h * ((Nx : ℝ) - 3) - q.1) ∧
      (R < q.2 - (o.2 + 2 * h) ∧ R < o.2 + h * ((Ny : ℝ) - 3) - q.2) from
    by simpa only [haloMargin, lt_min_iff] using hmargin) with ⟨⟨hxl, hxr⟩, ⟨hyl, hyr⟩⟩
  change (2 < (y.1 - o.1) / h ∧ (y.1 - o.1) / h < (Nx : ℝ) - 3) ∧
    (2 < (y.2 - o.2) / h ∧ (y.2 - o.2) / h < (Ny : ℝ) - 3)
  rw [lt_div_iff₀ hh, div_lt_iff₀ hh, lt_div_iff₀ hh, div_lt_iff₀ hh]
  exact ⟨⟨by linarith [(abs_le.mp hdx).1], by nlinarith [(abs_le.mp hdx).2]⟩,
    ⟨by linarith [(abs_le.mp hdy).1], by nlinarith [(abs_le.mp hdy).2]⟩⟩

/-- The same physical margin contains a closed Euclidean configuration ball.
The proof uses the L2-to-coordinate-max norm inequality explicitly. -/
theorem euclidean_closedBall_subset_openPhysicalHalo (Nx Ny : ℕ) (o : Coords)
    (x : Plane) (h R : ℝ) (hh : 0 < h)
    (hmargin : R < haloMargin Nx Ny o h (chart x)) :
    closedBall x R ⊆ chart ⁻¹' openPhysicalHalo Nx Ny o h := by
  intro y hy
  change chart y ∈ openPhysicalHalo Nx Ny o h
  apply closedBall_subset_openPhysicalHalo Nx Ny o (chart x) h R hh hmargin
  rw [mem_closedBall_iff_norm, ← map_sub]
  exact max_le ((WithLp.norm_fst_le ℝ (y - x)).trans (mem_closedBall_iff_norm.mp hy))
    ((WithLp.norm_snd_le ℝ (y - x)).trans (mem_closedBall_iff_norm.mp hy))

/-- A phase-ball position radius and a reported coordinate-query displacement
add before comparison with the physical margin. Cell crossings are allowed. -/
theorem reportedQuery_mem_of_phaseBall (Nx Ny : ℕ) (o q : Coords)
    (p z : Plane × Plane) (h R delta : ℝ) (hh : 0 < h)
    (hz : z ∈ closedBall p R) (hquery : ‖q - chart z.1‖ ≤ delta)
    (hmargin : R + delta < haloMargin Nx Ny o h (chart p.1)) :
    q ∈ openPhysicalHalo Nx Ny o h := by
  have hpos : ‖z.1 - p.1‖ ≤ R :=
    (norm_fst_le (z - p)).trans (mem_closedBall_iff_norm.mp hz)
  have hchart : ‖chart z.1 - chart p.1‖ ≤ R := by
    rw [← map_sub]
    exact max_le ((WithLp.norm_fst_le ℝ (z.1 - p.1)).trans hpos)
      ((WithLp.norm_snd_le ℝ (z.1 - p.1)).trans hpos)
  apply closedBall_subset_openPhysicalHalo Nx Ny o (chart p.1) h (R + delta) hh hmargin
  rw [mem_closedBall_iff_norm]
  exact (norm_sub_le_norm_sub_add_norm_sub q (chart z.1) (chart p.1)).trans
    (by linarith)

/-- All 36 mathematical stencil addresses, including the row-major address,
are inside the finite asset at every halo query. -/
theorem stencil_indices_of_mem_halo (Nx Ny : ℕ) (o q : Coords) (h : ℝ)
    (hq : q ∈ physicalHalo Nx Ny o h) :
    let n := normalizedCoordinates o h q
    ∀ i j : Fin 6,
      (0 ≤ ⌊n.1⌋ + i.val - 2 ∧ ⌊n.1⌋ + i.val - 2 < (Nx : ℤ)) ∧
      (0 ≤ ⌊n.2⌋ + j.val - 2 ∧ ⌊n.2⌋ + j.val - 2 < (Ny : ℤ)) ∧
      (0 ≤ (⌊n.2⌋ + j.val - 2) * Nx + (⌊n.1⌋ + i.val - 2) ∧
        (⌊n.2⌋ + j.val - 2) * Nx + (⌊n.1⌋ + i.val - 2) < (Nx : ℤ) * Ny) := by
  dsimp only
  intro i j
  exact ⟨halo_index_bounds Nx _ hq.1.1 hq.1.2 i,
    halo_index_bounds Ny _ hq.2.1 hq.2.2 j,
    halo_flattened_index_bounds Nx Ny _ hq.1.1 hq.1.2 hq.2.1 hq.2.2 i j⟩

/-- Every physical query jet is independent of the extension outside the asset,
including when the selected integer cell changes. -/
theorem physicalJet_eq_of_mem_halo (n m : ℕ) (C D : Lattice) (Nx Ny : ℕ)
    (o q : Coords) (h : ℝ) (hCD : AgreeOnAsset C D Nx Ny)
    (hq : q ∈ physicalHalo Nx Ny o h) :
    physicalJet n m C o h q = physicalJet n m D o h q := by
  unfold physicalJet
  rw [globalJet_eq_of_halo n m C D Nx Ny (normalizedCoordinates o h q)
    hq.1.1 hq.1.2 hq.2.1 hq.2.2 hCD]

theorem isOpen_openPhysicalHalo (Nx Ny : ℕ) (o : Coords) (h : ℝ) :
    IsOpen (openPhysicalHalo Nx Ny o h) :=
  (isOpen_Ioo.prod isOpen_Ioo).preimage (contDiff_normalizedCoordinates o h).continuous

/-- Actual ruler fields agree on a neighborhood of every interior halo point. -/
theorem rulerField_eventuallyEq (C D : Lattice) (Nx Ny : ℕ) (o : Coords)
    (h epsilon : ℝ) (hCD : AgreeOnAsset C D Nx Ny) (x : Plane)
    (hx : chart x ∈ openPhysicalHalo Nx Ny o h) :
    rulerField C o h epsilon =ᶠ[𝓝 x] rulerField D o h epsilon := by
  have hnhds : chart ⁻¹' openPhysicalHalo Nx Ny o h ∈ 𝓝 x :=
    ((isOpen_openPhysicalHalo Nx Ny o h).preimage chart.continuous).mem_nhds hx
  filter_upwards [hnhds] with y hy
  exact congrArg (ruler epsilon) (physicalJet_eq_of_mem_halo 0 0 C D Nx Ny o (chart y) h hCD
    ⟨⟨hy.1.1.le, hy.1.2⟩, ⟨hy.2.1.le, hy.2.2⟩⟩)

/-- Neighborhood equality identifies the actual Riesz gradient and actual
derivative Hessian, rather than merely the provider's named query fields. -/
theorem actual_metricJets_eq (C D : Lattice) (Nx Ny : ℕ) (o : Coords)
    (h epsilon : ℝ) (hCD : AgreeOnAsset C D Nx Ny) (x : Plane)
    (hx : chart x ∈ openPhysicalHalo Nx Ny o h) :
    rulerField C o h epsilon x = rulerField D o h epsilon x ∧
      gradient (rulerField C o h epsilon) x = gradient (rulerField D o h epsilon) x ∧
      fderiv ℝ (gradient (rulerField C o h epsilon)) x =
        fderiv ℝ (gradient (rulerField D o h epsilon)) x := by
  have heq := rulerField_eventuallyEq C D Nx Ny o h epsilon hCD x hx
  exact ⟨heq.self_of_nhds, heq.gradient_eq, heq.gradient.fderiv_eq⟩

/-- The actual mechanical vector field has the same phase-space germ for
every extension of one finite asset, with the same supplied force field. -/
theorem acceleration_eventuallyEq (C D : Lattice) (Nx Ny : ℕ) (o : Coords)
    (h epsilon a : ℝ) (hCD : AgreeOnAsset C D Nx Ny)
    (force : Plane × Plane → Plane) (p : Plane × Plane)
    (hp : chart p.1 ∈ openPhysicalHalo Nx Ny o h) :
    acceleration C o h epsilon a force =ᶠ[𝓝 p] acceleration D o h epsilon a force := by
  have hnhds : (fun z : Plane × Plane => chart z.1) ⁻¹' openPhysicalHalo Nx Ny o h ∈ 𝓝 p :=
    ((isOpen_openPhysicalHalo Nx Ny o h).preimage
      (chart.continuous.comp continuous_fst)).mem_nhds hp
  filter_upwards [hnhds] with z hz
  obtain ⟨hr, hu, hH⟩ := actual_metricJets_eq C D Nx Ny o h epsilon hCD z.1 hz
  simp only [acceleration, mechanicalAcceleration, hr, hu, hH]

end FractalSync.OperationalHalo
