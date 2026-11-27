# Interior navigation fields at matched speed

[Issue #185](https://github.com/rhythmatician/FractalSync/issues/185) asks whether natural coordinates and orbit traps offer useful navigation beyond existing Map features. This trial compares four prescribed paths in the main cardioid and period-two bulb. It uses the installed Rust signed-distance provider for the existing Map direction and holds metric, speed and Julia presentation fixed within each chart.

The exterior comparison and the trap-selection certificate are in the
companion notes. [Exterior paths](185-exterior-navigation.md),
[finite trap selection](185-trap-selection-validity.md)

## Comparison

The multiplier coordinate is $u$, with $c=F(u)=u(2-u)/4$ in the cardioid and $c=F(u)=-1+u/4$ in the bulb. On $|u|<1$ use the curvature $-1$ Poincaré density $2/(1-|u|^2)$. In the $c$ plane this gives $G=r^{-2}I$, where $r=(1-|u|^2)|F'(u)|/2$. Each path covers intrinsic length $0.8$ at speed one in this unregularized candidate metric. The Map query uses $\epsilon=0.01$ to select its derivative stencil. The chart supplies the metric for all four paths.

| Path | Definition |
| --- | --- |
| Geodesic | Start in the positive trap tangent and map $e^{i\phi}\tanh(s/2)$ through the disk automorphism sending zero to the start. This compares free motion with the trap from the same heading. |
| Multiplier | Keep $\lvert u\rvert^2$ constant and increase its angle. At radius $b$, the angular rate is $(1-b^2)/(2b)$. |
| Map iso-scale | Follow the positive tangent to $D$ in the sampled interior. There $D<0$, so this is also the positive tangent to $-\sqrt{D^2+\epsilon^2}$. The trace uses the Rust jet only while it is regular, non-bridge and has a usable gradient. |
| P2 trap | Follow the positive tangent to $\lvert c(c+1)\rvert^2$. Its gradient in complex notation is $2\overline{(2c+1)}c(c+1)$. |

Every contour takes the positive quarter turn $J$ of its declared normal, with $\dot c=rJ\nabla f/|\nabla f|$ for a regular field $f$. Their headings can differ at the same start because the normals are the comparison. The Map and trap paths solve the unit-speed tangent ODE with 256 RK4 steps and are rerun with 128 steps. The starts in both charts have the identical multiplier coordinates $0.25$, $0.48e^{i\pi/3}$, and $0.66e^{-2i\pi/3}$. These reference paths prescribe shape. The [exact trap lift](185-exact-trap-contours.md) and [steering budgets](185-trap-steering-budget.md) give the separate requirements for following that shape with Controls.

The gradient comparison samples radii $0.20,0.40,0.65$ at 12 equally spaced phases in each chart. It also includes the candidate $-r$ gradient. For the bulb, $r=(1-|u|^2)/8$, so multiplier and candidate ruler directions coincide exactly. The cardioid has a varying $|F'|$, which changes the ruler gradient direction. Angles compare oriented normals, excluding refused Rust samples. The $10^\circ$ bin is descriptive.

## Results

The installed binding reported geometry-provider/6 with the built-in mandelbrot_1024 field. All 36 direction samples per chart were valid at their query locations. A cardioid trajectory later hit a cut-locus refusal.

| Direction pair | Cardioid median / maximum | Cardioid within 10° | Bulb median / maximum | Bulb within 10° |
| --- | ---: | ---: | ---: | ---: |
| Trap / multiplier | 2.71° / 31.47° | 26/36 | 3.00° / 9.87° | 36/36 |
| Map / multiplier | 68.71° / 180.00° | 6/36 | 0.08° / 0.74° | 36/36 |
| Trap / Map | 80.48° / 180.00° | 6/36 | 2.97° / 9.90° | 36/36 |
| Candidate ruler / multiplier | 29.50° / 180.00° | 7/36 | 0° / 0° | 36/36 |

Ignoring the sign adds two cardioid near-alignments to each Map pairing and one to candidate ruler versus multiplier. Reversed orientation explains only a small part of the cardioid disagreement. Map and multiplier are nearly redundant on this bulb sample. The trap resembles the multiplier in both charts, especially near the centers where the two squared fields have proportional leading terms. Their separation grows enough along the tested arcs to give different paths.

| Chart and start | Geodesic / multiplier | Map / multiplier | Trap / multiplier | Trap / Map |
| --- | ---: | ---: | ---: | ---: |
| Cardioid, $0.25$ | 0.6496 | 0.7472 | 0.0381 | 0.7780 |
| Cardioid, $0.48e^{i\pi/3}$ | 0.2785 | refused | 0.0747 | refused |
| Cardioid, $0.66e^{-2i\pi/3}$ | 0.1840 | 0.1923 | 0.0967 | 0.0994 |
| Bulb, $0.25$ | 0.6496 | 0.0015 | 0.0390 | 0.0401 |
| Bulb, $0.48e^{i\pi/3}$ | 0.3159 | 0.0017 | 0.1018 | 0.1029 |
| Bulb, $0.66e^{-2i\pi/3}$ | 0.4481 | 0.0041 | 0.1125 | 0.1087 |

Each entry is the maximum disk distance over equal-time samples, in intrinsic units. Accepted fine traces have total chord lengths within $2\times10^{-6}$ of 0.8 at the displayed precision. The largest endpoint change on doubling the step count was $1.2\times10^{-10}$ for the trap and $3.4\times10^{-5}$ for the sampled Map. The trap kept its level within $3.9\times10^{-14}$. Accepted Map paths showed signed-distance drift up to $2.6\times10^{-4}$ in $c$ units. These data do not identify the share of that drift due to the sampled jet versus integration. A provider invariant requires its own error bounds.

## Keep or drop

Keep multiplier and fixed-$P_2$ trap paths as research diagnostics. Drop the case for adding this fixed trap to Player observation or default steering on the present evidence. In the bulb, the existing Map and natural directions track each other, and the trap's departures have no measured benefit. In the cardioid, both alternatives remained regular on the sampled trial where the Map trace refused. That makes the alternate paths worth testing. Geometric diversity and diagnostic coverage alone leave predictive motion value, controllability and visual benefit unresolved.

## Validity and remaining scope

Inside these known hyperbolic components, multiplier phase and magnitude describe the attracting cycle. Fixed multiplier magnitude gives circles in $u$, and an internal ray keeps multiplier phase constant. Phase fails at the periodic center, and finite nonescape alone cannot certify an attracting period. Outside, parameter Böttcher potential $g=\ln|\Phi|$ gives equipotentials and $\arg\Phi$ gives external rays where the angle is defined. Interior and exterior phases have distinct validity domains. Recurrence needs its own confidence and time support. The critical iterate index identifies a composition count. See [the exterior ruler notes](184-exterior-ruler-bounds.md).

For the fixed trap, reject gradients near the hits $0,-1$ and saddle $-1/2$. The saddle level has crossing branches and requires a branch choice for continuation. The analytic and Rust contour probes refuse at stage or endpoint when they lose their chart or gradient authority. The finite threshold is a numerical guard. A noisy derivative needs directional error bounds near a critical point. For configurable $h(P_n)$, the defining function needs a regular derivative, the orbit must remain valid for the chosen index, and switching the selected iterate requires an explicit validity boundary. A zero of $P_n$ gives a period dividing $n$. See the [fixed-trap derivation](185-orbit-trap-steering.md) and [trap curvature study](185-trap-saddle-curvature.md).

The [exterior study](185-exterior-navigation.md) adds a guarded Map comparison. The [Controls trial](185-contour-controls-trial.md) compares multiplier and Map guidance in two components and measures frozen Julia output. Keep #185 open for configurable recurrence, held-out route effects, and perceptual evidence. Coordinate periodic centers with [#141](https://github.com/rhythmatician/FractalSync/issues/141), renderer trap textures with [#180](https://github.com/rhythmatician/FractalSync/issues/180), and any supported observation change with [#108](https://github.com/rhythmatician/FractalSync/issues/108).

## Reproduce the trial

~~~powershell
rtk proxy .\.venv\Scripts\python.exe formal/checks/check_navigation_fields.py --images tmp/navigation-185
~~~

The [check](../../formal/checks/check_navigation_fields.py) records the provider and embedded asset, valid gradient pair counts, intrinsic chord sums, matched-time separation, level-set drift and endpoint refinement. It writes a Map PNG for all six starts and a 33-frame GIF with moving markers and Julia thumbnails for each mode at the real positive start. The thumbnails hold the $[-1.55,1.55]^2$ viewport, palette, 80 iterations and bailout radius 2 fixed. These are offline escape-time illustrations. The [#151 frozen-state renderer probe](https://github.com/rhythmatician/FractalSync/issues/151) and [#171 sensitivity measurement](https://github.com/rhythmatician/FractalSync/issues/171) are the relevant renderer interfaces.
