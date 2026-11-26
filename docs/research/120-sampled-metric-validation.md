# Sampled distance-graph metric validation

This experiment measures the installed Rust Python binding for [#120](https://github.com/rhythmatician/FractalSync/issues/120). It tests the canonical distance-graph metric on actual accepted jets and compares the period-two interior jets with an exact signed-distance reference. The runtime metric and integrator are called through their native bindings.

## Reproduce

From the repository root, with the installed `geometry-provider/6` binding and existing field assets:

```powershell
rtk proxy .venv/Scripts/python.exe -X utf8 formal/checks/check_sampled_metric.py --details
```

The run uses $d_{\rm ref}=0.1$, $\varepsilon=10^{-4}$, $\lambda^2=1$, $\kappa=1$ and the default wall coefficient $\mu=1/\pi$. The native binary measured here had SHA-256 `0e26f43b33b499a5da5346b412bea57605a64cb9dc8886976f585ea0ec99cac6`. The provider version exported by the installed binding differs from the `geometry-provider/1` source on the research checkout. Treat these measurements as evidence for that installed binary. The command prints timing and statuses for the actual queried provider. It rejects a binding with a different exported version. [Measurement script](../../formal/checks/check_sampled_metric.py), [canonical metric and inverse](../equations.md).

## Reference and coverage

The open period-two bulb is the Euclidean disk $|c+1|<1/4$. A closest point of its boundary is a Shore point. Because the whole open disk lies in the interior of the Mandelbrot set, the exact signed distance in the disk is $D(c)=|c+1|-1/4$. Away from the center, the exact gradient is the outward unit radial vector $n$ and the Hessian is $tt^{\mathsf T}/|c+1|$, where $t$ is its quarter turn. The center is a true cut locus. The adjacent control $c=-1.001$ still has a unique nearest Shore point. [Exact period-two component and distance](184-period-two-attachment.md).

The experiment takes twelve equally spaced angles at each of seven interior depths, and twelve positions on the exact reference Shore. Generic exterior samples use the provider's own $D$ for the anisotropy comparison. The left-axis control $c<-2$ has exact distance $-2-c$ because the Mandelbrot set is contained in the radius-two disk and $-2$ lies on its boundary. The provider domain guard takes precedence there. Failure rows retain their reason and remain outside metric/error aggregates.

For every accepted native symmetric matrix $G$, the script solves the two-by-two characteristic equation for both eigenvalues. It checks their finiteness and positivity and measures the condition $g_{\max}/g_{\min}$. For an ideal distance gradient of length one the predicted condition is

$$
1+\frac{\lambda^2}{(\ln 2)^2}\frac{D^2}{D^2+\varepsilon^2}.
$$

The same formula using $|\nabla D|^2$ in the numerator gives the prediction from the sampled jet. Comparing both predictions separates matrix construction from distance and derivative fidelity. The reported unit-intrinsic-speed Euclidean speed along a unit direction $u$ is $1/\sqrt{u^{\mathsf T}Gu}$. The interior oracle uses the exact radial and tangent directions for speed measurements and also reports speeds along the native principal axes. The detailed report retains point coordinates, resolved spacings, tile identities, and refusal diagnostics.

## Observed coverage and metric

The binding accepted 40 of 110 samples. All 40 native matrices were finite, symmetric and positive definite. The independently computed condition agreed with the formula using the sampled jet to within $1.34\times10^{-15}$, and the largest relative error in the minimum eigenvalue against that jet was $8.89\times10^{-16}$.

| Region or exact interior depth | Accepted / attempted | Refusal breakdown |
| --- | ---: | --- |
| Bulb $0.24$ | 0/12 | 12 singular/cut locus |
| Bulb $0.15$ | 12/12 | |
| Bulb $0.05$ | 12/12 | |
| Bulb $0.01$ | 6/12 | 6 unresolved |
| Bulb $4\varepsilon$ | 6/12 | 6 unresolved |
| Bulb $\varepsilon$ | 0/12 | 12 unresolved |
| Bulb $\varepsilon/4$ | 0/12 | 12 unresolved |
| Exact bulb Shore | 0/12 | 12 unresolved |
| Bulb center and adjacent control | 0/2 | 2 singular/cut locus |
| Offshore left axis and wall tip | 0/5 | 5 outside domain/boundary |
| Generic exterior | 4/7 | 2 singular/cut locus, 1 unresolved |

The entire $0.24$ band consists of smooth exact-distance points, $0.01$ from the bulb center. Its classifications measure refusals in the provider's estimated cut-locus neighborhood. The exact-Shore queries returned best-effort distances of both signs, with magnitudes as large as $3.36\times10^{-4}$. All twelve refused the metric. At the accepted exterior points, all sampled $D$ values were positive.

| Accepted region | $g_{\min}$ range | $g_{\max}$ range | Condition range | Median tangential / normal Euclidean speed |
| --- | ---: | ---: | ---: | ---: |
| Bulb $0.15$ | 44.44–44.54 | 136.35–136.89 | 3.0614–3.0789 | $0.14988$ / $0.08549$ |
| Bulb $0.05$ | 397.76–401.76 | 1215.44–1237.76 | 3.0480–3.0826 | $0.05001$ / $0.02851$ |
| Bulb $0.01$ | $[9.901,10.052]\times10^3$ | $[3.023,3.097]\times10^4$ | 3.0535–3.0804 | $0.01002$ / $0.00572$ |
| Bulb $4\varepsilon$ | $[5.452,5.754]\times10^6$ | $[1.613,1.704]\times10^7$ | 2.9583–2.9645 | $0.000404$ / $0.000249$ |
| Generic exterior | 0.425–10.059 | 1.302–30.836 | 3.0099–3.0655 | $0.87044$ / $0.50124$ |

The speed columns use the analytic directions in the bulb and the native principal directions in the exterior. At equal intrinsic speed, the median true tangential coordinate speed fell by a factor of about 371 between the $0.15$ and $4\varepsilon$ depth bands. At $4\varepsilon$, the median speed along the native tangent principal axis was $0.000423$, compared with $0.000404$ in the true tangent direction. The largest observed condition, $3.082562$, exceeds the ideal unit-gradient ceiling $3.081369$ by $0.001193$. It agrees with the sampled-gradient prediction. The observed range applies to these samples.

| Accepted bulb depth | Max $|D-D_{\rm exact}|$ | Max $\|\nabla D-n\|$ | Max Hessian Frobenius error | Max condition error vs exact $D$ |
| ---: | ---: | ---: | ---: | ---: |
| $0.15$ | $1.607\times10^{-4}$ | $0.00916$ | $3.15$ | $0.01995$ |
| $0.05$ | $1.407\times10^{-4}$ | $0.01510$ | $4.79$ | $0.03337$ |
| $0.01$ | $4.925\times10^{-5}$ | $0.01554$ | $448.67$ | $0.02769$ |
| $4\varepsilon$ | $1.643\times10^{-5}$ | $0.27396$ | $2474.17$ | $0.00560$ |

The $4\varepsilon$ band's exact Hessian has Frobenius norm about $4.01$. Its large observed Hessian error can coexist with a close condition estimate because the metric uses the first derivative and the condition does not track its direction. At depth $0.01$ the accepted jets reported two spacings, approximately $6.10\times10^{-5}$ and $4.88\times10^{-4}$. The two spacings arose within the same depth band. Their presence alone gives no estimate of the size or location of a discontinuity across a tile change.

On this run, the provider query median per accepted-band sample ranged from about 22 ms in the generic exterior band to 2.5 s in the $4\varepsilon$ band, with refusal sampling included in those band medians. The immediately repeated metric call at an accepted point had band medians of about 4–7 μs. These are single-process timings of this diagnostic call order on the installed binary. A frame budget also depends on reuse, branch, location, and client.

## Local derivative consistency

At the $60^\circ$ radial sample of depth $0.05$, both centered stencils kept the same reported tile and spacing. The finite differences compare $\rho$ values with the center $\nabla\rho$, adjacent $\nabla\rho$ values with the center Hessian from the jet, and adjacent metric matrices with the center native Christoffel symbols. The Christoffel estimate uses the usual inverse-metric contraction of the finite-differenced metric.

| Coordinate step | Gradient error | Hessian Frobenius error | Christoffel Frobenius error |
| ---: | ---: | ---: | ---: |
| $9.765625\times10^{-4}$ | $0.002060$ | $1.581$ | $2.455$ |
| $4.8828125\times10^{-4}$ | $0.001632$ | $1.410$ | $2.081$ |

Depths $0.01$ and $10^{-4}$ at the same angle were unresolved at the center. These two stencils only test how the exposed field and derivative values agree on the chosen points and steps. They leave the location and causes of tile-transition and raster artifacts unassigned.

## Conservative replay

The replay starts at $-1.15+0.085i$ with unit native metric speed in coordinate direction $(0.4,0.9)$. The applied generalized force and drag coefficient are zero. The native Shore and secant-wall potentials remain active throughout the common $0.125$ time horizon. The perturbation holds the initial velocity fixed and shifts the initial position by $10^{-6}i$.

| Step | Steps completed | Signed final $\Delta E$ | Max $|\Delta E|$ |
| ---: | ---: | ---: | ---: |
| $1/120$ | 15/15 | $-6.359\times10^{-4}$ | $1.264\times10^{-3}$ |
| $1/240$ | 30/30 | $-3.331\times10^{-4}$ | $1.131\times10^{-3}$ |
| $1/240$, perturbed | 30/30 | $-3.327\times10^{-4}$ | $1.130\times10^{-3}$ |

There were no geometry refusals or non-finite states in these runs. The largest norms of the geodesic and summed potential acceleration terms were $0.01673$ and $0.03636$. The two fine endpoints differed by approximately $9.90\times10^{-7}$ in position and $1.58\times10^{-7}$ in velocity. The endpoint energy error fell after halving the step while the peak drift fell by about 11%. A single interior replay and two time steps cannot identify which share of the remaining drift comes from integration or sampled-jet error.

## Current-source distance bridge

The Rust example builds directly from the checkout, independently of
the installed Python wheel. It uses provider `geometry-provider/1`.
All 84 bulb-circle queries produced positive-definite metrics in this
run. The table reports the largest error at each depth across 12 angles.

| True depth | Absolute distance error | Absolute condition-number error |
| ---: | ---: | ---: |
| 0.24 | 0.000171 | 1.745 |
| 0.15 | 0.000563 | 0.0315 |
| 0.06 | 0.00222 | 0.184 |
| 0.01 | 0.0107 | 1.654 |
| 0.001 | 0.0174 | 2.060 |
| 0.0001 | 0.0183 | 1.039 |
| 0.000025 | 0.0184 | 1.716 |

The example also observed `Singular`/`CutLocus` at the center
and attachment, and `OutsideDomain` at `c=-2.001`. Its metric
constructor accepted these bridge jets. That is evidence about the
metric constructor's return path. A valid rollout also has domain,
force, and state checks. The `/1` and `/6` results have distinct
provider and validity semantics.

```powershell
cargo run --offline -p runtime_core --example sample_bulb_metric
```

[Rust baseline example](../../runtime-core/examples/sample_bulb_metric.rs)

## Remaining acceptance

Keep #120 open. The interior replay supplies a finite-horizon observation with timestep and initial-state perturbations. The issue still needs independent derivative-fidelity and transition probes near varied Shore features, derivative/stencil or representation ablations for energy-drift attribution, and representative #82 crossing/reflection and mini-Mandelbrot stress replays with full work and failure diagnostics. It also needs wider launch/wall incidence and matched-force validation, runtime performance and Rust/training/WASM/schema parity in matching builds, and the eyes-on evidence tracked through #111. [Crossing stress cases](https://github.com/rhythmatician/FractalSync/issues/82), [spectator and eyes-on validation](https://github.com/rhythmatician/FractalSync/issues/111).
