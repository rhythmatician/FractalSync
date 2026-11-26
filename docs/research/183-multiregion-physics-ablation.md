# Main-cardioid and exterior ruler ablations

The [period-two pilot](183-period-two-physics-ablation.md) now also runs on a main-cardioid patch and a finite exterior patch for [#183](https://github.com/rhythmatician/FractalSync/issues/183). Both use the same `FrozenRollout`, retained prepared distance asset, refusal policy, and held covectors as the period-two run. In each region the two distance profiles share a field and differ in \(\lambda^2\). The two conformal profiles share \(\lambda^2=0\), \(\alpha=1\), and differ in the scalar used to derive the ruler. They share the wall potential and drag, with the scale potential disabled at \(\kappa=0\). [Rust example](../../runtime-core/examples/dynamical_ruler_ablation.rs).

## Patches and checks

The main-cardioid chart starts at \((-0.45,0.1)\). The stored scalar is the signed ruler \(-\tfrac14(1-|1-s|^2)|s|\), where \(s=\sqrt{1-4c}\) is the branch with positive real part. The initializer rejects a sample unless \(|1-s|<1\). The compiled tests check the exact scalar values \(-1/4\) at \(c=0\) and \(-9/32\) at \(c=-5/16\), and reject the cusp. [Main-cardioid ruler](184-main-cardioid-ruler.md).

The exterior chart starts at \((0.5,0.5)\). Starting from \(z=D=0\), advance \(D\gets2zD+1\) and \(z\gets z^2+c\) with the old iterate. At stage eight, use \(|z|\log|z|/|D|\), which cancels the common dyadic normalization of the Green function and its gradient. The sample must stay finite, meet \(|z|\ge4\) and \(|c|/|z|^2\le1/4\), and pass the positive denominator test from the exact-input tail formula. The implementation also rejects a sample if the binary64 evaluation of that tail formula gives a scalar allowance at or above \(10^{-7}\). The stage is fixed across the patch to avoid pointwise iteration-count joins. Stage nine is a separate diagnostic. None of these binary64 checks encloses floating-point error in the orbit, escape test, or tail formula. [Exterior escape, derivative, and ratio bounds](184-exterior-ruler-bounds.md).

Each patch is a \(49\times49\) local grid with spacing \(2^{-10}\) and a full stencil halo. Preparation evaluates the signed chart formula at every coefficient. Every committed state in the comparison must still admit a guarded ruler evaluation and a full stencil in that patch. The candidate uses the prepared field's own stencil checks at intermediate stages. The two distance profiles use the archived engine's stage checks and must agree with the chart's sign at the 26 common validation points. Those archive checks establish local coverage and sign for the experiment, with reference-distance accuracy still unknown.

For these smooth nonquadratic scalars, raw nodal samples define the prepared spline. An independent centered stencil numerically differentiates the scalar formula at 26 off-grid and join-adjacent points, including the start. The table compares the spline jet against that reference at step \(10^{-4}\). The reference stencil step change is its difference from the result at \(2\times10^{-4}\).

| Region and comparison | Max scalar difference | Max gradient entry difference | Max Hessian entry difference |
| --- | ---: | ---: | ---: |
| Cardioid, spline versus reference stencil | \(7.75\times10^{-7}\) | \(1.91\times10^{-7}\) | \(1.25\times10^{-7}\) |
| Cardioid, reference stencil step change | 0 | \(6.30\times10^{-9}\) | \(3.54\times10^{-8}\) |
| Exterior, spline versus stage-eight reference stencil | \(1.33\times10^{-6}\) | \(3.75\times10^{-5}\) | \(1.47\times10^{-3}\) |
| Exterior, stage-eight reference stencil step change | 0 | \(8.95\times10^{-7}\) | \(1.40\times10^{-5}\) |
| Exterior, stage eight versus nine at fixed stencil size | \(2.75\times10^{-9}\) | \(4.80\times10^{-7}\) | \(8.31\times10^{-5}\) |

The maximum exterior scalar tail diagnostic at the 26 comparison points was \(4.25\times10^{-9}\). The measured spline discrepancies refer to the finite stage-eight scalar in the exterior and the multiplier formula in the cardioid. The next-stage and stencil comparisons are diagnostics over the sampled points. In particular, they supply no uniform true exterior derivative bound or IEEE error certificate. [Exterior derivative tail scope](184-exterior-ruler-jets.md).

## Matched Physics results

All six profiles completed the coarse and refined runs. Every run used the same 0.4 s schedule as the period-two fixture. Each coarse result used 3,680 prepared geometry queries. This count covers the frozen engine's steps and excludes asset preparation, chart validation, and reporting.

| Region | Profile | Final \(c\) | Final coordinate speed | Final intrinsic speed | Max coarse energy-work residual |
| --- | --- | --- | ---: | ---: | ---: |
| Cardioid | Distance graph | \((-0.4393855,0.0944964)\) | 0.028556 | 0.167567 | \(3.36\times10^{-11}\) |
| Cardioid | Distance conformal | \((-0.4330139,0.0927331)\) | 0.044702 | 0.146618 | \(1.34\times10^{-13}\) |
| Cardioid | Dynamical conformal | \((-0.4368642,0.0941515)\) | 0.034372 | 0.150197 | \(3.85\times10^{-16}\) |
| Exterior | Distance graph | \((0.5088130,0.4952118)\) | 0.023707 | 0.208840 | \(5.19\times10^{-13}\) |
| Exterior | Distance conformal | \((0.5088561,0.4952883)\) | 0.023735 | 0.208906 | \(1.46\times10^{-14}\) |
| Exterior | Dynamical conformal | \((0.5088084,0.4954275)\) | 0.024235 | 0.237915 | \(1.93\times10^{-16}\) |

The largest step-halving differences across the new regions were \(4.02\times10^{-12}\) in position and \(2.92\times10^{-11}\) in velocity, both on the cardioid distance graph. The retained distance field ID is `9721bb7647a0b8e6`. The candidate field IDs are `3a5ec5fd614e67d3` for the cardioid and `e15f87eabf43cd9b` for the exterior.

## Reproduce

Use the asset and policy from the [period-two setup](183-period-two-physics-ablation.md#reproduce-the-run). The original CLI still selects the period-two case.

```powershell
cargo test --offline -p runtime_core --features experimental-operational-field --example dynamical_ruler_ablation
cargo run --offline -p runtime_core --features experimental-operational-field --example dynamical_ruler_ablation -- --retained tmp/geometry145-retained-fine.bin docs/validation/bounded-reference-policy.json --region cardioid
cargo run --offline -p runtime_core --features experimental-operational-field --example dynamical_ruler_ablation -- --retained tmp/geometry145-retained-fine.bin docs/validation/bounded-reference-policy.json --region exterior
```

These runs compare three matched prepared profiles within each local chart. The [attachment fixture](183-attachment-crossing-fixture.md) adds one analytic Shore crossing. The [geometry decision](182-dynamical-ruler-decision.md) compares rendered motion on its three paths and keeps the production baseline. Provider joins, independent true-Shore correspondence, general attracting-cycle coverage, and certified operational error remain open for #183.
