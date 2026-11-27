# Separate pixel resolution from color quantization in the frozen Julia probe

The [first RGB8 panel](151-frozen-julia-sensitivity.md) and the [second RGB8 panel](171-julia-sensitivity-calibration.md) had no passing adjacent finite-difference steps. They also used different centers, which prevented a matched resolution comparison. This run repeats all 13 prior centers on the same three pixel grids and separates RGB8 rounding from the effect of keeping changes at pixel scale. The [frozen probe](../../frontend/scripts/check_julia_sensitivity.mjs) draws every frame using the production renderer and the tracked fragment shader.

## Fixed observables and matched stencils

The view is a square of side 2.7 centered at zero, with unit zoom, zero rotation and shader time zero. The three midpoint grids have 80, 160 and 320 samples per axis. The shader uses the same 500-iteration cap, squared bailout of $10^{10}$, palette and lighting on each run. The render reads the same three clipped RGB shader outputs through the default RGB8 readback and through a floating RGBA attachment. Both omit alpha. The float path checks the attachment component type and reads `FLOAT` samples. Dithering and antialiasing are disabled. Each target repeats a baseline exactly after rendering a different seed. The report records the browser, GL settings, uniforms, shader hash and provider profile. [Frozen renderer](../../frontend/src/lib/juliaRenderer.ts), [fragment shader](../../shared/shaders/julia.frag), [float attachment specification](https://registry.khronos.org/webgl/extensions/WEBGL_color_buffer_float/).

For each readback, measure the normalized pixel vector and the normalized vector of mean RGB values over an 8×8 lattice of equal, fixed viewport cells. The cell side is $2.7/8$ complex units. The grid is the coarsest scale declared for this diagnostic before measuring the panel. Its purpose is to ask whether the color change survives averaging across that spatial scale. The grid and normalization are independent of $c$ and the Map geometry. Averaging defines a renderer-specific continuity observable. The [morphology audit](140-julia-morphology-provenance.md) established presentation and pixel confounds for other descriptors. Human perception and pre-palette morphology remain separate questions.

The two readback modes and two spatial representations use the same views and admissible endpoints. Each $I$, $G_0$ and $G$ basis keeps the original Gram–Schmidt convention and intrinsic spacings $h=0.02,0.01,0.005,0.0025,0.00125$. The [original protocol](151-frozen-julia-sensitivity.md#measurement) gives the local provider checks and Float32 seed rules. All provider refusals and rejected stencils stay in the report. The installed `geometry-provider/6` native wheel and its `mandelbrot_1024` asset supply the geometry in this run. Current source declares [`geometry-provider/1`](../../runtime-core/src/geometry_provider.rs). All comparisons involving $G_0$ and $G$ are conditional on the recorded wheel and profile.

The analysis retains both eigenvalues and the direction of the larger eigenvalue. It applies the existing 15% shift criterion to each eigenvalue and to the two derivative columns, with the existing 10° cap where both eigenvalue gaps resolve the major axis. The fixed-cell vectors allow the same comparison between resolutions at matched $h$. A jointly stable window requires three consecutive $h$ on both 160² and 320², all four within-resolution comparisons passing, and all three cross-resolution comparisons passing. Raw pixel columns have a different dimension on each grid. Their reported refinement comparisons use adjacent $h$ at each fixed resolution.

The RGB8 screen keeps the operator budget $\sqrt{2}/(510h)$ at both spatial scales. Equal-area averaging contracts the normalized Euclidean norm, but all per-pixel errors in a cell could point in the same direction. The float screen requires positive measured eigenvalues and applies the same refinement rule. It has no certified arithmetic error bound. The report also measures the operator-norm difference of the RGB8 and float stencils in each metric basis, and checks that box averaging increases neither ordered eigenvalue. [Comparison and averaging code](../../frontend/scripts/julia_sensitivity_refinement.mjs).

## What changed under refinement

Neither raw representation passed a single adjacent step. The float readback removes RGB8 rounding as an explanation for that outcome on this panel. Pooling recovered some agreement, mostly on the two distant exterior controls. Every observable missed the joint step and resolution criterion.

| Observable | Step passes at 80² / 160² / 320² | Resolution passes at 80² → 160² / 160² → 320² |
| --- | ---: | ---: |
| Raw RGB8 | 0 / 0 / 0 | — |
| Raw float | 0 / 0 / 0 | — |
| Box RGB8 | 0 / 0 / 2 | 0 / 2 |
| Box float | 0 / 2 / 18 | 0 / 7 |

Each resolution measured 139 stencils. Geometry excluded three centers before differencing, with a singular provider reading at the cut locus and at the period-three center and an unresolved reading at the attachment. Seven further stencils met invalid endpoint geometry and four straddled the sampled Shore. All eleven endpoint refusals used the Euclidean basis. All three readback panels used Chromium `153.0.8010.12` and SwiftShader. The fragment shader SHA-256 was `c446d0d42789a45ea928fd93ceed700b76f3afbe39d698c04c35d90265786e7b`.

Every sampled endpoint RGB8 value was within 0.50001 codes of the paired float output. The maximum measured operator difference between their derivatives at 80², 160² and 320² was $0.431$, $0.432$ and $0.432$ for raw pixels. The corresponding differences for cell averages were $0.0440$, $0.0203$ and $0.0108$. These are observed errors in the respective normalized color units per intrinsic displacement, over all stencils. The 320² RGB8 cell means had nine rank-two, four rank-one and 126 rank-zero readings against the conservative rounding screen. All 139 float cell means had positive measured eigenvalues. The two results distinguish resolution under the RGB8 budget from an exact null direction.

The table shows ordered full-metric eigenvalues on the 320² float cells. The two distant centers passed both adjacent comparisons over the first three spacings at this resolution. The wider exterior center passed the resolution comparison at $h=0.02$ only. The held-out distant center failed every full-metric resolution comparison. These are finite-stencil measurements at single centers.

| Center | $\lvert D\rvert/\varepsilon$ | $h=0.02$ | $h=0.01$ | $h=0.005$ | $h=0.00125$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| $1.2+0.25i$ | 7,927 | $(0.000666,0.00165)$ | $(0.000707,0.00175)$ | $(0.000762,0.00178)$ | $(0.000857,0.00194)$ |
| $1.1-0.31i$ | 6,877 | $(0.000647,0.00153)$ | $(0.000628,0.00154)$ | $(0.000618,0.00158)$ | $(0.000806,0.00147)$ |
| $-0.74+0.132i$ | 4.89 | $(0.00203,0.00575)$ | $(0.0128,0.0189)$ | $(0.0407,0.0505)$ | $(0.412,0.858)$ |

At the last center, even the cell means have both eigenvalues grow by more than 100 times over the tested sequence. Both ruler bases admitted all five steps there. The same provider refused the three largest Euclidean steps. The result extends the previously observed stencil-locality benefit and places no derivative limit on first- or second-order behavior. The steep step dependence also occurred with the scale-only ruler. A fit of eigenvalues across Map scales would mix unresolved stencils with single-resolution agreement.

Keep the three-way metric comparison and the present geometry. [#171](https://github.com/rhythmatician/FractalSync/issues/171) still lacks resolved two-sided spectra across held-out patches and directional scale exponents. [#150](https://github.com/rhythmatician/FractalSync/issues/150) still lacks a renderer-derived matching basin or evidence that these measurements improve learning. The exact Gaussian and asymmetric template controls remain in the [calibration](171-julia-sensitivity-calibration.md).

## Reproduce

```powershell
rtk proxy node frontend/scripts/check_julia_sensitivity.mjs --calibrate
rtk proxy node frontend/scripts/check_julia_sensitivity.mjs --refinement
```

Use the project native Python environment, Playwright Chromium with SwiftShader and the frontend dependencies. `FRACTALSYNC_PYTHON` selects another project venv. The second command writes `tmp/julia-171/refinement.json` by default and accepts an alternate output path. For replay, each element of `panels` carries the frozen centers, stencil endpoints and renderer settings at one resolution.
