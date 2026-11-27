# Held-input RK work on a constant metric

This fixture exercises the actual `FrozenRollout` one- and four-substep RK profiles with a fixed nonzero covector and metric drag. It checks the stored work and endpoint energy at every accepted outer step against calculations for a constant prepared field. The earlier engine tests already cover drag-only work and rejected differential stencils. This comparison adds nonzero drive, independent signed stage sums, and a closed-form work integral. [Rust fixture](../../runtime-core/examples/held_rk_work.rs), [existing engine tests](../../runtime-core/src/operational_engine.rs#L618), [general held-input ledger analysis](159-rk-work-ledger.md).

The prepared field has all-zero coefficients, so its value and derivatives vanish. With \(\epsilon=1/2\), the metric is \(G=4I\). Both potentials are disabled. Start at \((0.3,0)\) with velocity \((0.3,0)\), constant covector \((1,0)\), and drag coefficient \(\beta=3/2\). The path stays on the horizontal axis, where \(\dot x=v\) and \(\dot v=1/4-\beta v\). In particular, the drive power is \(v\) and drag power is \(-6v^2\).

For a substep of size \(h\), let \(A=1/4-\beta v\) and \(b=\beta h/2\). The four affine stage velocities are \(v,\ v+hA/2,\ v+(hA/2)(1-b),\ v+hA(1-b+b^2)\). The example evaluates those polynomials separately from the engine and applies weights \(h(1,2,2,1)/6\) to each signed work contribution. The runtime exposes their combined accumulator. The two separately calculated contributions are positive drive and negative drag, and their sum agrees with the runtime. The same polynomial check covers committed position, velocity, and endpoint energy \(2v^2\).

For the continuum reference set \(v_\infty=1/6\) and \(u=v(0)-v_\infty\). Then \(v(t)=v_\infty+u e^{-\beta t}\). The work references integrate \(v(t)\) and \(-6v(t)^2\) explicitly. At 0.8 s they give drive \(+0.195449403386\) and drag \(-0.289895501765\). The physical-work integral comparison is independent of the observed energy-work residual.

| RK substeps | Outer steps over 0.8 s | Max stage-work discrepancy | Max discrepancy from exact work integral | Max absolute ledger residual |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | \(2.78\times10^{-17}\) | \(3.19\times10^{-6}\) | \(6.65\times10^{-6}\) |
| 1 | 8 | \(8.33\times10^{-17}\) | \(1.86\times10^{-7}\) | \(3.76\times10^{-7}\) |
| 4 | 4 | \(2.78\times10^{-17}\) | \(1.12\times10^{-8}\) | \(2.24\times10^{-8}\) |
| 4 | 8 | \(4.16\times10^{-17}\) | \(6.88\times10^{-10}\) | \(1.37\times10^{-9}\) |

The maximum endpoint energy discrepancy against the reconstructed discrete stage solution was \(2.78\times10^{-17}\). Halving the outer step reduced the work-integral discrepancy in both profiles. These observations provide fixture-level refinement evidence.

A synthetic refusal box covers the first weighted endpoint for a separate 0.4 s attempt. In the one-substep profile the refusal occurs at the final endpoint. In the four-substep profile it occurs at the first internal endpoint. The engine reports the endpoint phase, which confirms that the distinct fourth stage passed. The fixture verifies byte-for-byte rollback of the entire state, including preseeded time and work. A retry with the opposite held covector succeeds and adds exactly the independently calculated retry work within \(2\times10^{-14}\).

Run from the repository root.

```powershell
cargo run --offline -p runtime_core --features experimental-operational-field --example held_rk_work
```

This checks compiled work accumulation, endpoint energy, and refusal for one constant-metric example. Provider fidelity, changing-metric work, profile-wide numerical bounds, and certificate provenance for general accepted histories remain outside this result. The fixture has no impulses or reevaluated commands. It supplies empirical held-RK evidence for [#166](https://github.com/rhythmatician/FractalSync/issues/166) and [#178](https://github.com/rhythmatician/FractalSync/issues/178) without changing the Euler ledger or the engine.
