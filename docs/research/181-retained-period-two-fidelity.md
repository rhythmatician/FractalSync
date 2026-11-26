# Retained prepared field versus the period-two Shore

The compiled prepared-field evaluator returned a finite jet on five of eight sampled directions at each of seven depths. The closest accepted samples showed large signed-distance and derivative errors against the exact period-two disk. This comparison addresses the independent reference-fidelity requirement in [#181](https://github.com/rhythmatician/FractalSync/issues/181). The existing arithmetic certificate instead compares compiled output to its own stored-coefficient polynomial. Both results can hold at once. [Operational polynomial correspondence](181-operational-patch.md).

## Native assets and exact reference

The earlier replay retained two prepared assets. Both have origin `(-1.002, -0.702)`, spacing `0.0005`, dimensions `3425 × 2825`, and no serialized availability mask. The fine asset's archived policy has one exact reference-cut rejection at `(-1, 0)` and leaves true-Shore fidelity unknown elsewhere. The probe decodes and re-encodes each exact asset using the compiled Rust path. It queries `PreparedField::query` on both and `FrozenRollout::probe` with the archived policy on the fine asset. The coarse asset needs its own policy for rollout claims. [Retained asset preparation](../validation/bounded-operational-engine.md), [archived reference policy](../validation/bounded-reference-policy.json).

| Retained asset | Native field ID | SHA-256 |
| --- | --- | --- |
| Fine | `9721bb7647a0b8e6` | `bc2604771c2cc48fb71cda6cb22c6e4338a776b6591fa923dd065977817ad9fb` |
| Coarse | `71eaa19d09e17bac` | `3bb6bbaca940bc98bbd4d735702d79209cda7bc862559a321491dbb990a62a63` |

The open period-two component is the disk $|c+1|<1/4$, and its circle lies on the Shore. For any interior point, every Shore point lies outside the open disk and the nearest point on the circle attains the distance to the Shore. Thus the signed interior distance is exactly $D(c)=|c+1|-1/4$. At noncentral points with $r=|c+1|$ and $n=(c+1)/r$, its gradient is $n$ and its Hessian is $(I-nn^{\mathsf T})/r$. At the center only the scalar distance has a reference value. The comparison evaluates these formulas in 70-digit Decimal arithmetic from the exact binary64 coordinates passed to Rust. [Exact period-two component and distance](184-period-two-attachment.md).

## Observations

Each depth uses four axis points and four directions from $(\pm3/5,\pm4/5)$, rounded to the nearest dyadic grid coordinate of spacing $2^{-30}$. All 56 points remain strictly inside the reference disk. Both assets return identical jets or unavailability at all 59 queried positions, including the three controls. The three westward directions in each depth band lie beyond the left side of the retained lattice and are unavailable. All 35 accepted depth-band samples are also accepted by the fine frozen probe with `reference_distance_accuracy = unknown`. All returned scalar signs in these interior samples agree with the reference.

| Nominal depth | Accepted / 8 | Maximum $|F-D|$ | Maximum $\|\nabla F-\nabla D\|_2$ | Maximum Hessian Frobenius error |
| ---: | ---: | ---: | ---: | ---: |
| $2^{-4}$ | 5 | $1.086\times10^{-5}$ | 0.0116 | 13.0 |
| $2^{-6}$ | 5 | $1.996\times10^{-5}$ | 0.0099 | 18.8 |
| $2^{-9}$ | 5 | $2.912\times10^{-5}$ | 0.0184 | 173.9 |
| $2^{-11}$ | 5 | $1.388\times10^{-4}$ | 0.2958 | 1109.9 |
| $2^{-13}$ | 5 | $3.230\times10^{-4}$ | 0.7832 | 2198.2 |
| $2^{-15}$ | 5 | $4.015\times10^{-4}$ | 0.9486 | 2318.6 |
| $2^{-17}$ | 5 | $4.235\times10^{-4}$ | 0.9909 | 2326.9 |

At the closest lower axis sample $(-1,-0.24999237060546875)$, the exact depth is $7.6294\times10^{-6}$. The prepared field reports $-4.3118\times10^{-4}$, about 56.5 times the actual magnitude, and gives gradient $(-0.11683,-0.01600)$ where the oracle gives $(0,-1)$. The largest Hessian error occurs on the right axis approaching the attachment. There the reported Hessian is approximately $\operatorname{diag}(-2065.26,1075.96)$ and the exact interior Hessian is approximately $\operatorname{diag}(0,4.00012)$. Errors in value and gradient are possible much farther inside, and the near-Shore results are enough to rule out treating this prepared asset as a true-distance oracle even where it returns a coherent jet.

For the left-halo controls, $x=-1-2^{-10}$ returns a jet and $x=-1-2^{-9}$ returns `GeometryUnavailable`. Both are inside the exact disk. At the exact center, the raw field returns a finite jet and scalar $-0.2495943$, but the archived fine policy returns `ReferenceSingularity: canonical cut or cusp`. No derivative error is reported at the center. Over these positions the raw evaluator accepts 37 of 59 queries per asset and the fine frozen policy accepts 36.

## Reproduce and remaining scope

From the repository root with the retained assets present and cached Cargo dependencies:

```powershell
rtk proxy .venv/Scripts/python.exe formal/fixtures/check_181_retained_period_two.py tmp/geometry145-retained-fine.bin tmp/geometry145-retained-coarse.bin > tmp/geometry145-retained-period-two.json
```

The [checker](../../formal/fixtures/check_181_retained_period_two.py) builds a temporary release-mode Rust consumer with `experimental-operational-field`, without loading the installed Python wheel. This run used Rust `1.98.0` on `x86_64-pc-windows-msvc`. The report retains binary query coordinates, asset and policy hashes, current source commit, both native dispositions, raw jet values and exact-reference estimates. No new asset or field policy is introduced.

These maxima describe sampled errors over two retained assets and one analytic Shore region. There is no interval enclosure or whole-patch bound. Keep the experimental provider's `reference-distance-unknown` contract and #181 open for regional certificates, true-Shore evidence on independent patches, and outstanding executable correspondence and adoption work. [Prior installed-provider sampling](120-sampled-metric-validation.md) is a separate experiment on a different field implementation.
