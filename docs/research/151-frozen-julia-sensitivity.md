# Frozen Julia renderer sensitivity

[Issue #151](https://github.com/rhythmatician/FractalSync/issues/151) asks whether equal intrinsic displacements produce comparable changes in a declared Julia representation. This probe measures the two sensitivity eigenvalues of one fixed RGB framebuffer representation. It compares Euclidean coordinates, the scale-only ruler and the full graph metric on identical states. The comparison is offline.

## Reproduce

~~~powershell
rtk proxy node frontend/scripts/check_julia_sensitivity.mjs
~~~

The [probe](../../frontend/scripts/check_julia_sensitivity.mjs) uses the repository Python environment with the installed native Rust binding, existing Vite and Playwright Chromium. It serves the tracked [fragment shader](../../shared/shaders/julia.frag) at the same URL used by [JuliaRenderer](../../frontend/src/lib/juliaRenderer.ts). It calls the synchronous frozen renderer, reads the WebGL buffer immediately, and verifies exact equality of a frame repeated after an intervening seed. The default output is the ignored file tmp/julia-151/results.json. An optional positional output path selects another location, and FRACTALSYNC_PYTHON can point to the project environment on another machine.

The run fixes an 80×80 square viewport, unit zoom, center zero, zero rotation and shader time zero. It fixes hue, saturation, brightness and the renderer's default lighting and normal settings. The zoom-one span is 2.7 complex units. The shader uses no supersampling and the renderer disables multisampling. The experiment also disables WebGL dithering for the quantization comparison. The report records Chromium, WebGL and renderer identifiers, actual shader uniforms, context attributes, color-buffer bit depths, shader SHA-256, and native provider and asset information.

## Measurement

Use the actual RGB8 readback as the continuity observable $V(c)=\operatorname{vec}(\mathrm{RGB})/(255\sqrt{3WH})$. This computes mean squared RGB-code change, with alpha omitted. Treat it as a renderer- and presentation-specific descriptor. It has no perceptual calibration or separate morphology validation.

For each matrix $M$ in $I,G_0,G$, construct $E$ by a metric Gram–Schmidt normalization of the coordinate axes and check $E^\top M E=I$ numerically. The installed Rust binding returns $G$ and $G_0$. The latter uses the same profile and provider with $\lambda^2=0$. For each $h\in\{0.02,0.01,0.005,0.0025,0.00125\}$, form the two central-difference columns $[V(c+he_i)-V(c-he_i)]/(2h)$. This is an affine query in the tangent norm at the center. Each metric uses its own normalization, so equal $h$ does not mean equal Euclidean displacement. The JSON retains every endpoint and its provider record, each basis, the raw Gram matrix in that basis, both eigenvalues and directions, and the Gram and Jacobian condition numbers where the measured lower eigenvalue is positive.

The geometry screen requires a regular, non-bridge center and endpoints with finite signed distance. A measured stencil remains on the same sampled side as its center. This screen distinguishes refusal from image readback behavior. Continuous regularity of a segment between samples would need additional evidence. A second check records the Float32 seed coordinates and rejects colliding opposite endpoints or more than 5% relative error in their coordinate separation. The recorded provider error field is the provider's estimate.

For RGB8 rounding with dithering disabled, if each reported code is within half a code of its underlying color, a derivative column can differ by at most $1/(510h)$ in the chosen norm. The corresponding two-column operator error bound is $\sqrt{2}/(510h)$. Compare each measured singular value to that budget without raising a null direction. If both exceed the budget, the report also gives a rounding-only interval for the ratio of the two singular values. Shader thresholds, arithmetic and finite-difference truncation can add errors beyond this model. The probe logs the fraction of changed color components and RMS code difference for each pair. Rank above the RGB8 budget describes resolution under the rounding model.

For adjacent admissible spacings, compare both eigenvalues and the two derivative columns. The exploratory refinement rule requires at most 15% relative change in each and two singular values above the rounding budget on both scales. It also requires the major axes to agree within 10° when both eigenvalue gaps are at least 10% of their respective larger eigenvalues. The axis sign is ignored. Two successive passing comparisons are the criterion for a resolved three-spacing run. Keep all raw outcomes, including isolated passes, missing stencils and rank loss.

## First run

Chromium 153.0.8010.12 used software ANGLE/Vulkan SwiftShader and 8-bit RGB channels. Time zero, the 80×80 viewport, the renderer's 500-step cap and the 32-cycle palette parameter were confirmed from active uniforms. The shader uses an escape test of squared magnitude greater than $10^{10}$. Its SHA-256 was c446d0d42789a45ea928fd93ceed700b76f3afbe39d698c04c35d90265786e7b. Native geometry reported geometry-provider/6, orbit-controller/10 and the embedded mandelbrot_1024 asset. The full profile used $\epsilon=10^{-4}$, $d_\mathrm{ref}=0.1$ and $\lambda^2=1$.

The repeated frozen frame was byte-identical after rendering a different seed. Six centers were usable and a seventh was refused as a cut locus. At the usable centers the largest basis residual was $2.22\times10^{-16}$. Of 90 stencils, 82 were measured, six had invalid geometry and two crossed the sampled Shore. Every refusal was in the Euclidean stencil at a near-Shore state. The largest Float32 relative displacement error among measured stencils was 4.98%.

Each table cell gives $(\lambda_-,\lambda_+)$ and, after the slash, the count of singular values above the RGB8 rounding budget at the finest spacing $h=0.00125$. These raw finite differences have no convergence guarantee.

| Center $c$ | $\rho$ | $I$ | $G_0$ | $G$ |
| --- | ---: | ---: | ---: | ---: |
| $0+0i$ | 0.25049 | (5.23, 36.17) / 2 | (0.0295, 0.0405) / 0 | (0.0157, 0.0362) / 0 |
| $-0.17+0.09i$ | 0.43001 | (13.90, 88.42) / 2 | (9.32, 55.86) / 2 | (7.16, 38.26) / 2 |
| $-0.98+0.09i$ | 0.15778 | (96.30, 129.16) / 2 | (5.43, 16.04) / 2 | (5.06, 12.71) / 2 |
| $-0.10+0.85i$ | 0.005315 | refused | (103.38, 168.45) / 2 | (110.18, 168.89) / 2 |
| $-0.74+0.132i$ | 0.0004995 | (4316.41, 4443.09) / 2 | (1007.02, 1181.63) / 2 | (921.81, 1087.58) / 2 |
| $1.20+0.25i$ | 0.79271 | (0.273, 0.584) / 0 | (0.183, 0.458) / 0 | (0.096, 0.389) / 0 |
| $-1.001+0i$ | — | cut locus | cut locus | cut locus |

Ruler normalization allowed all five tested spacings at both near-Shore centers under the same provider screen. It reduced both raw sensitivity eigenvalues at the near-Shore exterior point while leaving them large relative to the wider samples. The difference from $G_0$ to $G$ generally moved the two finite-difference eigenvalues less than the initial scale-only change, with state- and spacing-dependent effects. In the period-two row the raw Jacobian condition numbers are 1.16, 1.72 and 1.58 respectively. The $G$ rounding-only interval there is approximately $[1,183]$, so the apparent improvement over $G_0$ does not resolve a condition-number advantage. Across the full run, 64 stencils had rank two above the RGB8 budget, five rank one and 13 rank zero. The finest outside-wide readings have rank zero for all metrics. Both ruler metrics also have rank zero at the real-axis center at that spacing.

No adjacent measured pair passed the refinement rule. For example, at the near-Shore exterior center the full metric's finest two raw eigenvalues changed by about 73% and 68%, even though both directions exceeded the rounding-only budget. The run therefore supports a stencil-locality benefit and identifies finite-size and readback limits. It supplies no resolved derivative-based ordering of the normalizations. The results retain the low-signal and unstable cases.

## Inspect a bookmark in the cockpit

Open `debugCockpit.html`, expand **DIAG · Frozen Julia bookmark**, and choose the saved `results.json`. Select a frozen state, metric, and step size. A measured step shows the saved raw matrix, eigenvalues and axes, and rank above the RGB8 budget. A refused step shows its saved status. A center refused by the geometry provider has no metric selector.

**Saved replay inputs and diagnostics** shows the shader and renderer provenance, center geometry, basis and metric, endpoint records, and the selected step's saved diagnostics. The viewer reads the file locally. It does not rerun the probe or change the current trajectory or renderer. Player View hides the panel and preserves the loaded bookmark when returning to diagnostic view. [Cockpit bookmark viewer](../../frontend/src/components/JuliaSensitivityBookmark.tsx).

## Disposition

The offline probe and the [#111](https://github.com/rhythmatician/FractalSync/issues/111) bookmark inspection cover #151's diagnostic scope. The unresolved and near-null results remain part of that evidence. Calibration, held-out resolved patches, and alternate observable or resolution comparisons belong to the [#171](https://github.com/rhythmatician/FractalSync/issues/171) follow-up. The first RGB8 run supplies no evidence for changing the metric, observations, or reward.
