# Frozen Julia eye measurements

This is a first renderer check for #258, #259 and #260. Run these from the repository root with the frontend dependencies, Playwright Chromium and the optional Python packages in `formal/checks/requirements.txt` installed.

```powershell
node frontend/scripts/capture_julia_eyes.mjs
.venv/Scripts/python.exe formal/checks/check_julia_eyes.py
```

The experiment renders the canonical shader at 800 × 600, zoom 1, time zero, hue 0.5, saturation 0.8 and brightness 0.9. It captures the default anchor and central pairs along the real and imaginary axes at Cartesian steps 0.00025, 0.0005 and 0.001. The repeated baseline was pixel-identical in the software WebGL run. These axes are convenient probe directions. They are not a replay of the cockpit's intrinsic axes or its musical phases.

The two dark openings had an equivalent-area radius of 13.56 px and centroids at pixel indices (291.066, 278.249) and (507.934, 320.751). Their centroids map to approximately ±(−0.48795 + 0.09563i). The candidate fixed point is about 1.12 pixels from its dark-opening centroid. The visually picked winding center is closer, but no winding-center estimator was used. The dark pixels describe the 500-iteration renderer's finite escape cutoff rather than membership in a Fatou component.

The numerical check also encloses the critical orbit of the exact decimal parameter using 300-digit interval arithmetic. It exhausts the radius-two test through step 418 and encloses the squared modulus at step 419 near 4.10379370126381. With the standard quadratic escape criterion this places the decimal parameter outside the Mandelbrot set and rules out a Siegel disk there. This uses the interval library as numerical evidence, outside the Lean audit, and uses the exact decimal parameter rather than the rounded shader uniforms.

At all three step sizes and RGB thresholds in the probe, the aperture log-radius response was positive along the real axis and negative along the imaginary axis. Its central estimates ranged from +324 to +370 and −287 to −274, respectively, per unit change in the Cartesian parameter. The outer annulus fit returned roughly +104 to +112 on the real axis and −9 to 0 on the imaginary axis. This supports measuring aperture separately from a uniform change of size. The outer estimates come from image registration and depend on the texture and mask.

The middle annulus returned a positive turn estimate on both axes in these samples. The amount varied with step and annulus, and the innermost real-axis fit changed sign. The fit estimates a common rotation, translation and scale in smoothed luminance. Its post-fit error was often most of the pre-fit error. This does not establish a converged apparent rotation differential or identify which pixels have reliable motion correspondence. The fit error is a finite image diagnostic and does not measure the differential projection residual proposed in #263.

For comparison, the fixed-point multiplier angle is about 2.951 radians per iterate. Its angular sensitivity along the real and imaginary parameter axes is approximately −0.282 and −0.958 radians per unit parameter change, respectively, from the imaginary part of `2 h / (λ (1 − λ))`. Equating these sensitivities to the fitted visual turn would fail even in sign on the middle annulus in this sample. A model of visible arm correspondence would need another step connecting iterates or inverse branches to the rendered features.

For a stronger check, validate landmark correspondence or local registration against synthetic turn, translation, twist and aperture changes. Compare render resolution and iteration depth, then rerun actual solo and combined #251 replays using the logged parameter path. The optional navigation study #264 asks a different question about repeated morphology along a path.

## From parameter motion to Map motion

At a resolved anchor, let `G` be the metric induced on the position–scale surface and let `A` have two rows, the measured aperture and angle differentials with respect to the two coordinates of `c`. In a `G`-orthonormal tangent frame `E`, the columns of `A E` give the change per unit intrinsic motion in each frame direction. Compare the same measurements under the Euclidean, scale-only, and full Map metrics using #151. The metric changes how much motion each effect costs. Since the frame is invertible, it cannot change whether the two effects are locally independent. Any numerical conditioning comparison must specify a normalization for the two outputs, which have different meanings.

Where the derivatives are reliable and well separated, solve for equal-cost motion that mostly opens the eye, mostly turns it, or combines the two. Compare the resulting Julia-only clips at the same anchor, then repeat at an anchor where the diagnostic predicts poor separation. This narrows the human question to whether the predicted contrast reads in the image. Use finite replays to check that the contrast holds over a playable excursion, since the differential gives only a local prediction.
