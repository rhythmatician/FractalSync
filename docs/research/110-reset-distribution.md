# Scale and kinetic-energy reset comparison

[Issue #110](https://github.com/rhythmatician/FractalSync/issues/110) asks how to expose the Player to deep geometry without losing ordinary-start competence. This offline comparison checks coverage and initialization using the installed native Rust provider. The sampled states are diagnostic inputs. No Player training or autonomous navigation is evaluated.

## Sampling

~~~powershell
rtk proxy .\.venv\Scripts\python.exe -u formal/checks/check_110_reset_distribution.py
~~~

The [check](../../formal/checks/check_110_reset_distribution.py) uses 640 seeded uniform-area proposals in the disk $|c|<1.9$. This radius leaves a research margin from the wall at $|c|=2$. It also generates 20,000 area proposals for cheap screening by the embedded unsigned distance raster. In each occupied 0.45-wide square it retains up to two close-to-Shore raster anchors and takes up to six short gradient-guided refinement queries. The proposed move subtracts $0.6D\nabla D/|\nabla D|^2$. This is a search proposal with a new provider query at the endpoint. The native provider supplies the sign and scale of each accepted site. The moves certify no path to the Shore.

The candidate pool uses the installed provider's regular, non-bridge results. The sign check requires $|D|>4$ times its estimated error. This threshold screens ambiguous signs empirically. It makes no certificate about exact Mandelbrot membership. Scale, the Gram matrix, kinetic energy and the energy ledger come from the Rust APIs. The report records sampled-region occupancy, provider refusals, selected profile and seed.

The bands are $\sigma<0$, $[0,3)$, $[3,6)$, $[6,9)$, and $\sigma\ge9$. A band participates in scale draws if each sampled side has at least two occupied research squares. Draw the band uniformly, then the sampled side, square and point uniformly at each level. This is equal mass per selected band. The points within a band follow the discrete validated proposal pool. It is not a continuous uniform-$\sigma$ sampler. The comparison draws 200 points per policy and reuses the exact mixed positions for both velocity policies.

| Policy | Position | Initial motion |
| --- | --- | --- |
| Area rest | Uniform draw from the validated uniform-area proposals | Rest |
| Scale rest | Equal selected band mass with side and region balance | Rest |
| Mixed rest | Half area draws and half scale draws | Rest |
| Mixed energy | Same positions as mixed rest | 50% rest, 25% each at $K_0=0.05$ and 0.25 in expectation |

The motion direction is drawn uniformly in a local $G$-orthonormal basis. Multiplying by $\sqrt{2K_0}$ gives the intended intrinsic speed, with actual energy rechecked through the binding. A fixed Euclidean speed of 0.01 in that same coordinate direction supplies a diagnostic comparison only. The report summarizes $U$, the wall potential contribution, and $E_0$. The Rust debug snapshot supplies the Shore-only crest threshold. The sign of $K_0+U_\sigma-U_{\mathrm{crest},\sigma}$ compares available energy to that threshold. The wall change, direction, dissipation and actual arrival are separate.

For jitter, take an affine displacement of length 0.05 in a uniformly sampled metric-unit direction. Verify the tangent norm at the origin and recheck the endpoint's band, side and provider status. This does not measure an exponential-map displacement or certify path regularity. A first-step screen then compares rest and both positive energy targets at the same eight or fewer sites per band. It uses the native Controls-v2 binding with default zero drive and impulse and a duration of 1024/48000 s. The endpoint must pass the same provider screen. This screen measures launch failures and side changes for one idle step.

## Coverage and launch evidence

The default seed 1102026 used geometry-provider/6, orbit-controller/10 and the embedded 1024² distance asset. The profile was $\epsilon=10^{-4}$, $d_\mathrm{ref}=0.1$, $\lambda^2=\kappa=1$ and $\mu=1/\pi$. The provider accepted 628 of 640 area centers and 132 additional guided queries. The area screen rejected ten unresolved queries and two cut-locus queries. The guided screen rejected fourteen unresolved queries. All five bands supported the sampling rule. The deepest band still had only four Inside candidates across three squares and 26 Outside candidates across 17 squares.

| Policy | $\sigma<0$ | $[0,3)$ | $[3,6)$ | $[6,9)$ | $\sigma\ge9$ | Regions | Unique positions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Area rest | 167 | 29 | 4 | 0 | 0 | 59 | 176 |
| Scale rest | 41 | 39 | 34 | 44 | 42 | 41 | 129 |
| Mixed rest and mixed energy | 104 | 38 | 16 | 25 | 17 | 56 | 160 |

Area-only left both deep bands empty. Scale-only reached them but revisited more sampled points and covered fewer coarse regions. Mixing produced 42 deep starts with $\sigma\ge6$, including 19 Inside starts, while preserving 104 starts with $\sigma<0$. These are selected starts and span counts in one finite proposal pool.

For the mixed-energy comparison, the actual draw counts were 104 rest, 43 at $K_0=0.05$ and 53 at $K_0=0.25$. The largest error in Rust-recomputed $K_0$ was $1.67\times10^{-16}$. Moving Euclidean speeds ranged from $3.13\times10^{-5}$ to 0.982. Fixing Euclidean speed at 0.01 instead gave median Rust kinetic energies of 0.000240 in the shallowest band and 4,877.9 in the deepest. At identical positions, median $E_0$ moved from −0.467 to −0.425 and its maximum from 9.966 to 10.216. The wall contribution reached 0.297 in the mixed sample, even with the 1.9 radius cap. Two mixed-energy starts exceeded the Shore-only crest reference, both in the deepest band. That is an initial ledger comparison with no predicted crossing outcome.

The jitter check preserved validity, band and sampled side in 78 of 80 draws. One endpoint was unresolved and one changed band or side. The maximum origin-norm error was $1.39\times10^{-17}$. A sampler needs an endpoint screen even at this small intrinsic displacement.

The one-step idle screen accepted every rest case and all lower-band moving cases. It refused one of eight moving trials at each positive energy in the deepest band. Every accepted endpoint stayed on its original sampled side. The deep failures therefore matter even when the initial center and target kinetic energy both passed validation. The ignored tmp/reset-110/results.json retains the full counts, per-band energy summaries, and metadata for the run.

## First trial distribution

Use the 50:50 area-plus-scale mixture as the first matched-training candidate. Keep ordinary area starts at rest. Within the scale half, use the first four bands, give each side and occupied square its balanced share, and use the tested 50% rest and 25% each positive energy mix. This confines positive launch energy to the portion where the first-step screen passed, while preserving deep local training starts at $\sigma\ge6$. The report tested the energy mix across both halves and all five bands. The restricted distribution is a proposal derived from its refusal evidence and requires a seeded rerun before a training job.

Keep $\sigma\ge9$ as a separate rest-first challenge until it has more independent Inside support and the moving failure is understood. Use constant neutral Julia presentation in the first comparison. Keep a separate evaluation suite for ordinary shallow-start discovery, stratified local handling, and crest tests with a full signed energy ledger and a native Controls schedule. Use held-out regions and seeds to measure both local competence and discovery. Defer the replay archive until the matched variants show which states are still missing. Future runs should stamp the versioned provider, sampler, seed, pool/region rules, band weights, energy and direction policy, jitter and independent presentation policy.

Keep #110 open for an independently regenerated pool, rollout/Player comparisons, standalone Rust reset contract, and matching #111/#121 replay provenance. The current data support an initial sampling choice and a deeper-band restriction. They give no sample-efficiency or autonomous navigation result.
