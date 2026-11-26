# Period-two Physics ablation

This [#183](https://github.com/rhythmatician/FractalSync/issues/183) pilot compares three metrics over the same short period-two interior fixture. It uses the existing Rust `FrozenRollout` for all three trajectories. The distance profiles share one field and differ only in the graph term. The conformal profiles share the same metric formula and differ in the scalar field. The experiment leaves the shipping provider and training unchanged.

## Reproduce the run

From the repository root, run the example with the retained field and refusal policy. The field used here has ID `9721bb7647a0b8e6` and SHA-256 `bc2604771c2cc48fb71cda6cb22c6e4338a776b6591fa923dd065977817ad9fb`. The [preparation recipe and hashes](../validation/bounded-operational-engine.md#recorded-local-integration-results) explain how to recover the retained asset. The example rejects a different field ID and requires a matching policy. [Rust example](../../runtime-core/examples/dynamical_ruler_ablation.rs), [frozen reference policy](../validation/bounded-reference-policy.json).

```powershell
cargo test --offline -p runtime_core --features experimental-operational-field --example dynamical_ruler_ablation
cargo run --offline -p runtime_core --features experimental-operational-field --example dynamical_ruler_ablation -- --retained tmp/geometry145-retained-fine.bin docs/validation/bounded-reference-policy.json
cargo run --offline -p runtime_core --features experimental-operational-field --example dynamical_ruler_ablation -- --circle-distance
```

The last command is a self-contained comparison using a spline sampled from exact bulb-circle distance. A requested retained run fails on unavailable coverage or an incorrect asset. Each successful run emits JSON with the field and Physics profile IDs, matched trajectory samples, energy ledger, coordinate and intrinsic speeds, geometry query count, and local step timings.

## Analytic check and controls

Inside the open bulb \(|c+1|<1/4\), the period-two multiplier is \(4(c+1)\) and the dynamical ruler is \(r=(1-16|c+1|^2)/8\). The experiment sends the signed scalar \(D=-r=2|c+1|^2-1/8\) through the existing preparation seam. In real coordinates its gradient is \(4(x+1,y)\) and its Hessian is \(4I\). The quintic spline with grid spacing \(h\) reproduces a quadratic up to the additive \(h^2\Delta D/4\). Setting its coefficients to \(D\) at the grid nodes minus \(2h^2\) removes that bias in exact arithmetic. The 49 by 49 coefficient patch, including the stencil halo, lies inside the open bulb. [Period-two chart](184-period-two-attachment.md), [quintic reconstruction and polynomial test](../../runtime-core/src/operational_field.rs#L246).

The compiled check compares values, gradient entries, and Hessian entries at 26 points, including both sides of a spline join and the initial position. For the quadratic, maximum absolute errors were \(4.16\times10^{-17}\), \(7.16\times10^{-15}\), and \(5.24\times10^{-12}\). The same points tested against the exact signed circle distance and its derivatives give the baseline discrepancies below. The example gates the baseline on complete jets and a maximum sampled scalar discrepancy of \(10^{-3}\). The derivative discrepancies remain diagnostics. In particular, the large retained Hessian discrepancy limits the interpretation of the graph profile. Neither sampled check supplies a uniform floating-point error enclosure.

| Prepared distance field | Scalar | Gradient entry | Hessian entry |
| --- | ---: | ---: | ---: |
| Retained field | \(1.40\times10^{-5}\) | \(7.01\times10^{-3}\) | 9.96 |
| Circle-distance spline | \(8.69\times10^{-6}\) | \(5.60\times10^{-5}\) | \(1.08\times10^{-3}\) |

All profiles start at \((-0.9,0.1)\) with coordinate velocity \((0.02,-0.01)\). The signed-distance graph has \(\lambda^2=1\). Both conformal profiles have \(\lambda^2=0\) and metric \(\rho^{-2}I\), which is the \(\alpha=1\) setting. They share the default regularization and wall with fixed drag \(\beta=0.2\). The scale potential is disabled with \(\kappa=0\) so the wall is the same potential for every run. The raw drive covector is \((2,-1)\) for 0.2 s and \((-0.5,1)\) for the next 0.2 s. Comparing equal coordinate starts and covectors allows the initial intrinsic speeds, energies, and delivered work to differ. Metric-normalized Controls is a separate comparison.

Each rollout uses four RK4 substeps per attempt. The coarse rollout attempts 40 steps at 0.01 s. The refinement uses 80 steps at 0.005 s and replays the same time-indexed covectors. All attempts succeeded. The example checks the energy-work residual at every committed attempt against 0.05, the magnitude used in the existing energy replay. It requires coarse and fine position and velocity differences below \(10^{-6}\) at shared sample times. A refinement difference measures consistency for this run. It gives no certified bound on trajectory error. [Existing energy replay](../../scripts/replay_bounded_energy.py), [frozen RK4 and ledger](../../runtime-core/src/operational_engine.rs#L356).

## Results

| Profile using the retained baseline | Final \(c\) | Final coordinate speed | Final intrinsic speed | Max position difference | Max velocity difference | Max coarse ledger residual |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Distance graph | \((-0.8915762,0.0955596)\) | 0.0217113 | 0.233882 | \(1.13\times10^{-11}\) | \(1.04\times10^{-10}\) | \(2.26\times10^{-10}\) |
| Distance conformal | \((-0.8910458,0.0961307)\) | 0.0228589 | 0.218310 | \(1.57\times10^{-14}\) | \(2.38\times10^{-13}\) | \(5.17\times10^{-13}\) |
| Dynamical conformal | \((-0.8915263,0.0961635)\) | 0.0219494 | 0.264540 | \(1.67\times10^{-15}\) | \(4.93\times10^{-17}\) | \(1.27\times10^{-15}\) |

Every coarse run made 3,680 geometry evaluations during its 40 attempts. The recorded debug runs spent 11–14 ms in the attempts per retained-baseline profile, excluding setup and reporting. These single-run timings do not establish a production latency target. Substituting the circle-distance spline moved the distance-graph endpoint by \(2.85\times10^{-5}\) and the distance-conformal endpoint by \(4.66\times10^{-7}\). With the same dynamical coefficients, the dynamical replay produced identical mechanical results.

The pilot establishes an end-to-end run of a local interior dynamical ruler and two distance profiles through the same Physics engine. Its scope is one short fixture. It leaves cardioid and finite-exterior jets, provider error contracts, seam and fallback policy, Shore crossings, visual measurements, and any geometry adoption decision open. [#183](https://github.com/rhythmatician/FractalSync/issues/183), [#182](https://github.com/rhythmatician/FractalSync/issues/182).
