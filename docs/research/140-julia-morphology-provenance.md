# Julia morphology before palette and rim light

The existing Rust Sobel metric responds to recoloring and rim light on the RGB framebuffer. On a binary mask taken before shading, it is independent of the three presentation changes tested here. Its value still changes with image resolution, zoom, and iteration budget. This supports using a declared pre-palette representation for [#140](https://github.com/rhythmatician/FractalSync/issues/140) and keeps the roughness definition and perceptual validation with [#105](https://github.com/rhythmatician/FractalSync/issues/105).

## Which code owns which measurement

| Source | Actual output and interpretation |
| --- | --- |
| [Rust visual metrics](../../runtime-core/src/visual_metrics.rs) | Takes normalized grayscale or RGB input. Averages the three RGB channels, clamps to `[0,1]`, applies a 3×3 Sobel filter with clamped edge indexing, and counts gradient magnitudes of at least 50 in 8-bit intensity units. `edge_density` divides that count by all pixels. `color_uniformity` is a monotone transform of mean 5×5 local grayscale variance. The brightness fields describe grayscale code values. `mandelbrot_membership` is a finite critical-orbit test of the seed and is independent of the image. |
| [Python loss metrics](../../backend/src/visual_metrics.py) | `LossVisualMetrics.compute_all_metrics()` returns `temporal_change`. Its existence does not establish a current roughness objective. |
| [Training UI](../../frontend/src/components/TrainingPanel.tsx) and [earlier coherence plan](../../backend/docs/EMOTIONAL_COHERENCE.md) | The UI still names `distortion_roughness_loss`. The earlier plan proposes roughness alignment. The audit found no backing computation for the UI field in the active loss-metric interface. |
| [Julia fragment shader](../../shared/shaders/julia.frag) | `smoothIterJulia` computes the smooth escape count, stripe average, distance estimate, normal and derivative confidence. `shadePixel` tests `niter <= 0.0` before palette, distance shaping and lighting, then emits the final color. The production readback is RGB. |

The audited native descriptor API has no width, branch-count, multiscale roughness, or structural-order output. Reusing Sobel on the new input exposes what this existing descriptor can and cannot distinguish. It does not select a canonical roughness score for #105.

## Reproduce the fixed-view diagnostic

```powershell
rtk proxy node frontend/scripts/check_julia_morphology.mjs
```

The script reuses the [#151 frozen renderer entry point](151-frozen-julia-sensitivity.md). It serves the tracked shader for the RGB run and an instrumented copy for the mask run. The copy returns three binary channels at the existing pre-shading predicate, using the same recursion and escape radius with iteration caps 500, 128 and 64. A 1 means the shader would enter its bounded-pixel branch by that cap. The support is a finite-orbit filled-set approximation on the viewport. Zero occupancy means zero foreground pixels on that grid, with no thickness measurement available. It does not report exact Julia connectivity or boundary dimension.

The run fixes time zero, a square viewport centered at zero with rotation zero, unit zoom unless stated, and resolutions 64², 128² and 256². The zoom-one span is 2.7 complex units. Both WebGL1 programs disable dithering and antialiasing. The escape test uses squared magnitude greater than $10^{10}$ and the renderer supplies its default 500-iteration uniform. The script checks that a larger iteration budget can only shrink the mask. For $c=0$, it also checks each mask byte against the exact unit disk evaluated at those pixel centers. [Frozen renderer](../../frontend/src/lib/juliaRenderer.ts), [diagnostic script](../../frontend/scripts/check_julia_morphology.mjs).

The Rust descriptor receives final RGB divided by 255 or a single-channel binary mask of zeros and ones. Mask `brightness_mean` equals the filled-pixel fraction exactly in these tests. This statistic describes the filled area fraction in the declared crop. It cannot distinguish a single thick region from many thinner parts with the same area. The Sobel value on the mask is a pixel-scale boundary response per viewport area, with a threshold fixed in intensity units.

The first run used Chromium `153.0.8010.12`, WebGL1 with ANGLE/Vulkan SwiftShader, 8-bit RGB channels, and installed `runtime-core` `0.1.0` reporting `orbit-controller/10`. The shader SHA-256 was `c446d0d42789a45ea928fd93ceed700b76f3afbe39d698c04c35d90265786e7b`. The ignored JSON report `tmp/julia-140/results.json` includes both shader hashes, renderer information, the known seeds, view and presentation controls, readback hashes and raw metrics. An optional positional path selects another output.

## Within-modality results

All five RGB frames repeated exactly after rendering an intervening seed. At 128², all five seeds changed RGB bytes under each separate hue, brightness, and rim intervention. Every three-cap mask stayed byte-identical under all three. The rim intervention changes the renderer's existing `u_rimIntensity` from `0.08` to `0.75`. It tests this additive rim control, while #95 owns any separate edge-glow presentation contract.

| Fixed seed $c_J$ | Base RGB Sobel | Hue intervention | Brightness intervention | Rim intervention | Unchanged mask Sobel |
| --- | ---: | ---: | ---: | ---: | ---: |
| $0$ | 0.04645 | 0.03729 | 0.04791 | 0.15546 | 0.04590 |
| $-0.17+0.09i$ | 0.05774 | 0.04297 | 0.05762 | 0.15900 | 0.04980 |
| $-0.98+0.09i$ | 0.10315 | 0.09003 | 0.09393 | 0.20789 | 0.08313 |
| $-0.74+0.132i$ | 0.30951 | 0.30786 | 0.29010 | 0.37909 | 0.14209 |
| $1.2+0.25i$ | 0.02826 | 0.00043 | 0.01233 | 0.09430 | 0 |

The sampled wide-exterior mask was empty at every tested setting. The RGB still had edges from the shading of escaping orbits, so the native RGB descriptor can report edge density even where this particular foreground support contains no pixels. That is a practical counterexample to interpreting the raw RGB metric as an occupied-boundary statistic.

| Fixed seed $c_J$ | Mask occupancy at 64² / 128² / 256² | Mask Sobel at 64² / 128² / 256² | Occupancy at zoom 1.5, 128² |
| --- | --- | --- | ---: |
| $0$ | 0.43066 / 0.43018 / 0.43121 | 0.09375 / 0.04590 / 0.02319 | 0.89697 |
| $-0.17+0.09i$ | 0.42041 / 0.42139 / 0.42130 | 0.09766 / 0.04980 / 0.02484 | 0.85437 |
| $-0.98+0.09i$ | 0.18750 / 0.18774 / 0.18732 | 0.13770 / 0.08313 / 0.05084 | 0.32678 |
| $-0.74+0.132i$ | 0.03271 / 0.03210 / 0.03076 | 0.15039 / 0.14209 / 0.12985 | 0.05713 |
| $1.2+0.25i$ | 0 / 0 / 0 | 0 / 0 / 0 | 0 |

For the smooth disk, occupied fraction stays close to $\pi/2.7^2\approx0.43095$ while edge density roughly halves on each resolution doubling. This follows the behavior of an edge band only a few pixels wide. The selected near-Shore seed behaves differently. Its 128² occupied fractions at caps 64, 128 and 500 were `0.21509`, `0.15942`, and `0.03210`. The associated Sobel values were `0.17932`, `0.22900`, and `0.14209`. Changing the iteration budget greatly changed both measurements. The Sobel threshold and escape radius were held fixed. Their sensitivity still needs a separate test.

## Disposition

Use this mask and raw Rust measurements as fixed-view controls and audit evidence. Current RGB Sobel is a presentation-sensitive feedback metric. The fixed-view mask Sobel needs scale/threshold and human-visible roughness validation before it could support H1. The occupied fraction needs independent width and foreground-quality controls before it could support H2 or H4. These observations do not establish standalone morphology axes or brightness and glow correspondence. #140 and #105 remain open for perceptual within-modality descriptors and blinded fit, matched audio targets, pose/probe trajectories, and any held-out learning ablation. No runtime, observation, or loss changes were made.
