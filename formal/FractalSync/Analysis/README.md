# Absolute-continuity support for the pinned proof environment

The package remains on Lean 4.24.0 and Mathlib
`f897ebcf72cd16f89ab4577d0c826cd14afaafc7`. That Mathlib snapshot defines
`AbsolutelyContinuousOnInterval`, but does not yet contain the absolutely
continuous fundamental theorem of calculus. The files here provide the
required analysis through attributed backports and project compatibility
proofs, without modifying `.lake/packages/mathlib` or changing the committed
toolchain or manifest.

## Provenance

The upstream source snapshot is Mathlib commit
[`aad247f928b1eb7c58478742119fdd15a37f0ffd`](https://github.com/leanprover-community/mathlib4/tree/aad247f928b1eb7c58478742119fdd15a37f0ffd).
The original files are copyright 2025 Yizheng Zhu and distributed under
Apache 2.0. Their headers are retained and the upstream license is included
in [LICENSE](LICENSE). These are attributed upstream results, not new
FractalSync research claims.

| Local file | Upstream file below `Mathlib/` |
| --- | --- |
| `AbsolutelyContinuous.lean` | `MeasureTheory/Function/AbsolutelyContinuous.lean` (selected additions) |
| `AbsolutelyContinuousFTC.lean` | `MeasureTheory/Integral/IntervalIntegral/AbsolutelyContinuousFun.lean` |
| `DerivIntegrable.lean` | `MeasureTheory/Integral/IntervalIntegral/DerivIntegrable.lean` (required subset) |
| `IntervalSlope.lean` | `MeasureTheory/Integral/IntervalIntegral/Slope.lean` |
| `OrderFinsetGaps.lean` | `Order/Interval/Finset/Gaps.lean` |
| `FinsetGaps.lean` | `Algebra/BigOperators/Group/Finset/Gaps.lean` |

Each retrieved source was checked against its blob at the pinned upstream
commit. Adaptations remove newer module-visibility syntax, use the older
import locations and theorem names, and supply proofs for APIs absent in the
older snapshot. The optional monotone derivative-integral range theorem was
omitted because the FTC does not need it. The AC definition remains the
actual pinned Mathlib definition throughout.

`Fatou.lean` proves the missing sequential integrability criterion directly
from the pinned Fatou lemma and almost-everywhere strong-measurability theorem.
`FiniteDimensionalAC.lean` extends scalar derivative integrability to finite
dimensional real normed spaces by continuous linear coordinates. The AC support
file also obtains almost-everywhere differentiability in finite dimension from
the pinned bounded-variation theorem.

## Project AC composition and gluing

`ACComposition.lean` adds four proved declarations in
`FractalSync.ACComposition`. They are project compatibility proofs and are not
copied from the upstream snapshot above:

- `disjoint_uIoc_map_monotone` and `disjoint_uIoc_map_antitone` preserve the
  disjoint-interval condition under monotone or antitone real maps.
- `comp_lipschitz_mono` proves AC preservation under a monotone or antitone
  Lipschitz domain map whose interval image lies in the original interval.
- `absolutelyContinuous_gluePath` glues two matching AC pieces by expressing
  them as constant extensions using `min` and `max`.

The proofs use the actual pinned Mathlib AC definition. The downstream
`PathOperations` module uses them for reversal and concatenation and then
proves the corresponding actual integral-length identities. AC preservation
alone is not a claim of length invariance under every nonlinear
reparameterization.

The nine support modules now contain 43 explicit theorem/lemma declarations:
39 in the earlier support files and 4 in `ACComposition`.

## Trust boundary

The backport proves that the derivative of an absolutely continuous real
function is integrable and integrates to the endpoint difference. The critical
zero-derivative constancy argument uses a Vitali covering and finite interval
gaps; absolute continuity controls the uncovered gaps. Almost-everywhere
differentiability alone would not suffice for this theorem.

The umbrella package imports all of this support through its result modules.
The axiom audit roots every declaration originating in a `FractalSync` module,
including declarations in upstream namespaces such as `Finset` and
`AbsolutelyContinuousOnInterval`. It then checks all transitive dependencies
against the unchanged allowlist: `propext`, `Classical.choice`, and `Quot.sound`.
Source adaptation introduces no axioms and does not replace a missing proof
with an extra theorem hypothesis.

When upgrading the pinned environment later, remove these compatibility files
once the corresponding upstream APIs are available and update the project
imports. Do not retain duplicate declarations alongside the newer Mathlib.
