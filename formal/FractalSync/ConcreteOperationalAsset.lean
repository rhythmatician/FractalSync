import FractalSync.OperationalTrajectoryCertificate

/-!
# A literal finite operational-field fixture

This exact-real asset materializes the nonconstant quadratic fixture in
`runtime-core/src/operational_field.rs::tests::field`. The companion extraction
check records binary64 bit patterns and checks separately rounded, non-fused
source operations against exact rational rounding. That external evidence is
not Rust execution, compiler-contraction/FMA equivalence, or an IEEE semantics
proof. The Lean statements concern the literal dyadic table, the reconstructed
mathematical field, and its exact physical domain.
-/

noncomputable section

namespace FractalSync.ConcreteOperationalAsset

open OperationalField OperationalHalo OperationalDynamics FlatShoreIntrinsic Set Metric

local instance concreteOperationalPhaseNormedSpace : NormedSpace ℝ (Plane × Plane) :=
  Prod.normedSpace

/-- Exact real value of the runtime fixture's binary64 spacing `0.1`. -/
def spacing : ℝ := 3602879701896397 / 36028797018963968

def origin : Coords := (0, 0)

-- BEGIN GENERATED QUADRATIC NUMERATORS
def coefficientNumerators : Fin 16 → Fin 16 → ℤ :=
  ![
    ![0, 1441151880758559, 5764607523034236, 12970366926827032, 23058430092136944, 36028797018963968, 51881467707308128, 70616442157169392, 92233720368547776, 116733302341443264, 144115188075855872, 174379377571785632, 207525870829232512, 243554667848196448, 282465768628677568, 324259173170675712],
    ![-4323455642275677, 0, 7205759403792795, 17293822569102708, 30264189495929740, 46116860184273872, 64851834634135152, 86469112845513536, 110968694818409040, 138350580552821648, 168614770048751360, 201761263306198240, 237790060325162240, 276701161105643296, 318494565647641536, 363170273951156800],
    ![-17293822569102708, -10088063165309912, 0, 12970366926827032, 28823037615171180, 47558012065032432, 69175290276410832, 93674872249306336, 121056757983718960, 151320947479648672, 184467440737095488, 220496237756059520, 259407338536540608, 301200743078538752, 345876451382054144, 393434463447086528],
    ![-38911100780481096, -28823037615171184, -15852670688344152, 0, 18734974449861264, 40352252661239640, 64851834634135160, 92233720368547776, 122497909864477504, 155644403121924352, 191673200140888320, 230584300921369440, 272377705463367680, 317053413766882944, 364611425831915456, 415051741658464960],
    ![-69175290276410832, -56204923349583800, -40352252661239648, -21617278211378384, 0, 24499581972895488, 51881467707308128, 82145657203237872, 115292150460684720, 151320947479648704, 190232048260129728, 232025452802128000, 276701161105643328, 324259173170675776, 374699488997225344, 428022108585291968],
    ![-108086391056891904, -92233720368547760, -73498745918686496, -51881467707308104, -27381885734412608, 0, 30264189495929760, 63410682753376608, 99439479772340576, 138350580552821632, 180143985094819840, 224819693398335232, 272377705463367680, 322818021289917184, 376140640877983872, 432345564227567616],
    ![-155644403121924384, -136909428672063120, -115292150460684736, -90792568487789216, -63410682753376608, -33146493257446880, 0, 36028797018963968, 74939897799445056, 116733302341443264, 161409010644958560, 208967022709991008, 259407338536540640, 312729958124607232, 368934881474191104, 428022108585292032],
    ![-211849326471508160, -190232048260129792, -165732466287234272, -138350580552821648, -108086391056891920, -74939897799445072, -38911100780481088, 0, 41793404541998208, 86469112845513536, 134027124910545984, 184467440737095552, 237790060325162304, 293994983674746048, 353082210785846976, 415051741658464832],
    ![-276701161105643328, -252201579132747840, -224819693398335200, -194555503902405472, -161409010644958592, -125380213625994656, -86469112845513536, -44675708303515328, 0, 47558012065032384, 97998327891581952, 151320947479648704, 207525870829232512, 266613097940333376, 328582628812951488, 393434463447086528],
    ![-350199907024329792, -322818021289917184, -292553831793987456, -259407338536540576, -223378541517576608, -184467440737095552, -142674036195097280, -97998327891581952, -50440315826549568, 0, 53322619588066624, 109527542937650496, 168614770048751488, 230584300921369408, 295436135555504576, 363170273951156800],
    ![-432345564227567616, -402081374731637888, -368934881474191040, -332906084455227008, -293994983674745984, -252201579132747776, -207525870829232416, -159967858764200000, -109527542937650432, -56204923349583744, 0, 59087227111100928, 121056757983719040, 185908592617854080, 253642731013506432, 324259173170675712],
    ![-523138132715356928, -489991639457910080, -453962842438946112, -415051741658465024, -373258337116466816, -328582628812951488, -281024616747919008, -230584300921369472, -177261681333302784, -121056757983719040, -61969530872618112, 0, 64851834634135168, 132585973029787392, 203202415186956800, 276701161105643264],
    ![-622577612487697536, -586548815468733568, -547637714688252480, -505844310146254208, -461168601842738944, -413610589777706496, -363170273951156864, -309847654363090240, -253642731013506432, -194555503902405504, -132585973029787520, -67734138395652352, 0, 70616442157169280, 144115188075855872, 220496237756059392],
    ![-730664003544589312, -691752902764108288, -649959498222109952, -605283789918594688, -557725777853562240, -507285462027012736, -453962842438945984, -397757919089362176, -338670691978261312, -276701161105643264, -211849326471508160, -144115188075855872, -73498745918686464, 0, 76381049680203776, 155644403121924352],
    ![-847397305886032640, -805603901344034432, -760928193040519168, -713370180975486720, -662929865148937088, -609607245560870400, -553402322211286592, -494315095100185664, -432345564227567680, -367493729593432512, -299759591197780288, -229143149040610816, -155644403121924352, -79263353441720704, 0, 82145657203237760],
    ![-972777519512027136, -928101811208511744, -880543799143479296, -830103483316929792, -776780863728863104, -720575940379279360, -661488713268178432, -599519182395560448, -534667347761425216, -466933209365772992, -396316767208603648, -322818021289917056, -246436971609713408, -167173618167992704, -85027960964754944, 0]
  ]
-- END GENERATED QUADRATIC NUMERATORS

/-- The first index is the column and the second is the row. -/
def coefficient (i j : Fin 16) : ℝ :=
  (coefficientNumerators j i : ℝ) / 144115188075855872

/-- The zero extension of the stored finite table. Off-asset coefficients
do not affect any certified halo query. -/
def coefficients : Lattice :=
  zeroExtension (fun i j =>
    coefficient ⟨i.toNat % 16, Nat.mod_lt _ (by decide)⟩
      ⟨j.toNat % 16, Nat.mod_lt _ (by decide)⟩) 16 16

def position : Plane := WithLp.toLp 2 ((1 / 2 : ℝ), (1 / 2 : ℝ))

def velocity : Plane := WithLp.toLp 2 ((1 / 4 : ℝ), (0 : ℝ))

def phase : Plane × Plane := (position, velocity)

def radius : ℝ := 1 / 16

def queryError : ℝ := 1 / 64

/-- A kernel-reduced finite integer certificate for all 256 stored values. -/
theorem numerator_bound :
    ∀ i j : Fin 16, |coefficientNumerators j i| ≤ 1008806316530991104 := by
  decide

/-- The actual literal table has a proved coefficient budget of seven. -/
theorem coefficient_bound (i j : Fin 16) : |coefficient i j| ≤ 7 := by
  have hn : |(coefficientNumerators j i : ℝ)| ≤ 1008806316530991104 := by
    exact_mod_cast numerator_bound i j
  norm_num [coefficient, abs_div, div_le_iff₀]
  exact hn

/-- The selected zero extension satisfies the same global magnitude bound. -/
theorem coefficients_bound : ∀ i j, |coefficients i j| ≤ 7 := by
  exact abs_zeroExtension_le _ 16 16 7 (by norm_num)
    (fun _ _ _ _ _ _ => coefficient_bound _ _)

theorem finite_coefficients_bound (i j : ℤ) (_hi : 0 ≤ i) (_hix : i < 16)
    (_hj : 0 ≤ j) (_hjy : j < 16) : |coefficients i j| ≤ 7 :=
  coefficients_bound i j

/-- Tight rational spacing bounds retain the actual binary64 datum. -/
theorem spacing_bounds : (1 / 10 : ℝ) ≤ spacing ∧ spacing ≤ 1 / 8 ∧ 0 < spacing := by
  norm_num [spacing]

theorem spacing_pos : 0 < spacing := spacing_bounds.2.2

theorem spacing_lower : (1 / 10 : ℝ) ≤ spacing := spacing_bounds.1

theorem spacing_upper : spacing ≤ (1 / 8 : ℝ) := spacing_bounds.2.1

theorem norm_position_sq : ‖position‖ ^ 2 = (1 / 2 : ℝ) := by
  norm_num [position, WithLp.prod_norm_sq_eq_of_L2]

theorem norm_position_le : ‖position‖ ≤ (3 / 4 : ℝ) := by
  nlinarith [norm_position_sq, norm_nonneg position]

theorem norm_velocity : ‖velocity‖ = (1 / 4 : ℝ) := by
  norm_num [velocity, WithLp.prod_norm_eq_of_L2, Real.sqrt_div]

/-- The exact physical margin is attained at both lower halo faces. -/
theorem halo_margin :
    haloMargin 16 16 origin spacing (chart position) =
      (5404319552844595 / 18014398509481984 : ℝ) := by
  norm_num [haloMargin, origin, spacing, position, chart]

theorem halo_margin_gt_quarter :
    (1 / 4 : ℝ) < haloMargin 16 16 origin spacing (chart position) := by
  norm_num [halo_margin]

theorem phase_query_margin :
    radius + queryError < haloMargin 16 16 origin spacing (chart phase.1) := by
  norm_num [phase, radius, queryError, halo_margin]

/-- The complete phase ball stays in the unit configuration ball used by
the nonzero secant-wall force certificate. -/
theorem phaseBall_position_norm_le_one (z : Plane × Plane)
    (hz : z ∈ closedBall phase radius) : ‖z.1‖ ≤ 1 := by
  have hpos : ‖z.1 - position‖ ≤ radius :=
    (norm_fst_le (z - phase)).trans (mem_closedBall_iff_norm.mp hz)
  linarith [norm_le_norm_sub_add z.1 position, norm_position_le,
    show radius = (1 / 16 : ℝ) from rfl]

/-- Integer indexing agrees with the actual stored row/column entry. -/
theorem coefficients_of_fin (i j : Fin 16) :
    coefficients (i : ℤ) (j : ℤ) = coefficient i j := by
  simp [coefficients, zeroExtension, Nat.mod_eq_of_lt i.isLt, Nat.mod_eq_of_lt j.isLt]

/-- Explicit unequal values rule out a constant or zero coefficient fixture. -/
theorem coefficients_values : coefficients 0 0 = 0 ∧
    coefficients 15 0 = (9 / 4 : ℝ) ∧ coefficients 0 15 = (-27 / 4 : ℝ) := by
  change ((0 : ℤ) : ℝ) / 144115188075855872 = 0 ∧
    ((324259173170675712 : ℤ) : ℝ) / 144115188075855872 = 9 / 4 ∧
    ((-972777519512027136 : ℤ) : ℝ) / 144115188075855872 = -27 / 4
  norm_num

/-- A second finite integer certificate compares every stored value with
the quadratic at the exact physical lattice node, before reconstruction. -/
theorem sample_error_numerator_bound : ∀ i j : Fin 16,
    |coefficientNumerators j i * 9007199254740992 -
      ((i : ℤ) ^ 2 + 2 * (i : ℤ) * (j : ℤ) - 3 * (j : ℤ) ^ 2) *
        (3602879701896397 : ℤ) ^ 2| ≤ 4611686018427387904 := by
  decide

/-- Every literal stored coefficient is within `2⁻⁴⁸` of its quadratic
reference value at the exact physical node. -/
theorem coefficient_sample_error (i j : Fin 16) :
    |coefficient i j - (((i : ℝ) * spacing) ^ 2 +
      2 * ((i : ℝ) * spacing) * ((j : ℝ) * spacing) -
      3 * ((j : ℝ) * spacing) ^ 2)| ≤ (1 / 281474976710656 : ℝ) := by
  have hn : |(coefficientNumerators j i : ℝ) * 9007199254740992 -
      ((i : ℝ) ^ 2 + 2 * (i : ℝ) * (j : ℝ) - 3 * (j : ℝ) ^ 2) *
        (3602879701896397 : ℝ) ^ 2| ≤ 4611686018427387904 := by
    exact_mod_cast sample_error_numerator_bound i j
  rw [abs_le] at hn ⊢
  dsimp only [coefficient, spacing]
  constructor <;> nlinarith [hn.1, hn.2]

theorem radii_positive : 0 < radius ∧ 0 ≤ queryError := by
  norm_num [radius, queryError]

theorem radius_pos : 0 < radius := radii_positive.1

theorem queryError_nonneg : 0 ≤ queryError := radii_positive.2

/-- Every admitted reported coordinate query has a complete in-bounds
36-point stencil, including its mathematical row-major addresses. -/
theorem safe_reported_query (z : Plane × Plane) (q : Coords)
    (hz : z ∈ closedBall phase radius) (hquery : ‖q - chart z.1‖ ≤ queryError) :
    SafeReportedQuery 16 16 origin spacing q :=
  safeReportedQuery_of_phaseBall 16 16 origin q phase z spacing radius queryError
    spacing_pos hz hquery phase_query_margin

end FractalSync.ConcreteOperationalAsset
