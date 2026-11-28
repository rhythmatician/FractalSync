# Candidate dynamical centers of the default Julia eyes

The left eye at the default two-source anchor lies close to the fixed point
with multiplier modulus 1.00409. The right eye lies close to its negative,
which maps to the fixed point in one step. The dark aperture centroid is a
different observable from the dynamical point and from the perceived winding
center.

## Frozen-frame evidence

We used the canonical Julia renderer at `c = -0.7269 + 0.1889i`, with
presentation fixed at zoom 1 and an 800 × 600 viewport. The shader maps PNG
pixel indices `(x,y)` to `(2.7/600) ((x+0.5-400) + i(300-y-0.5))`.
Image indices increase downward. Both candidate points come from
`p = (1 - sqrt(1-4c))/2`, using the principal square root near this anchor:

| Quantity | Left eye | Right eye |
| --- | --- | --- |
| Candidate in complex coordinates | `-0.492949 + 0.095121i` | `+0.492949 - 0.095121i` |
| Projected candidate pixel | `(289.956,278.362)` | `(509.044,320.638)` |
| Dark aperture centroid | `(291.066,278.249)` | `(507.934,320.751)` |
| Candidate to centroid offset | 1.116 px | 1.116 px |

The aperture is the 4-connected component containing the chosen eye seed
with maximum RGB channel at most 20. The two components each have 578 pixels
in the baseline, unchanged at thresholds 10 and 30. For centered
perturbations of `c` by 0.00025, 0.0005 and 0.001 along the real and imaginary
axes, the left component remained within 0.12–1.50 pixels of the continued
fixed point. The component seed was the fixed pixel `(291,278)` throughout.
The candidate position did not determine the seed at each perturbation.
Computed locations and distances are approximate, and the centroid can
shift because of discretization and changes in the component's shape.

The other fixed point is about `1.492949 - 0.095121i`, outside the two
apertures. The period-two candidates are about
`-0.789143 + 0.326655i` and `-0.210857 - 0.326655i`, with shared
multiplier modulus 1.32826. Neither sits near the measured centers. The
aperture depiction comes from a finite iteration renderer. Its dark pixels
do not certify a Fatou component, and the candidate has a numerically
repelling multiplier rather than the neutral multiplier of a Siegel center.
For the exact decimal anchor, an independent interval computation finds
that the critical orbit has squared radius above 4 at iterate 419. This
places the parameter outside the Mandelbrot set using the quadratic
escape criterion. See [the shared measurement record](258-260-julia-eye-measurements.md)
for the interval computation and reproduction command.

## Exact statements and next tests

The fixed point has exact displacement law
`f_c(p+w)-p = (2p)w+w²`. Its multiplier at this anchor has argument about
169.08° per iterate. A local inverse branch has the reciprocal multiplier
and can generate contracted and rotated copies of local structure. A change
in the multiplier changes that local dynamical rule. The turn of an
identified rendered arm under parameter motion still depends on the arm's
continuation and on the renderer.

[The Lean module](../../formal/FractalSync/JuliaEyeDynamics.lean) checks the
half-turn identity, the fixed-point displacement law, and the parameter
derivative for an already-differentiable local fixed-point branch. It checks
that every positive iterate of the negative preimage lands at the fixed
point and derives the derivative of the multiplier logarithm on its stated
chart. The numerical center estimate and the identification of similarity
in images are outside those proofs.

A falsifiable next test is to define a winding observable on annuli around
the candidate and check it across radii, resolutions, iteration counts and
small moves in `c`. Compare it with an independently fitted winding center
and with the aperture centroid. In a larger survey, include parameters
with lightly and strongly repelling branches and controls such as
`c = 0`, whose Julia set is a circle despite its repelling fixed point at
`z = 1`. The presence of a repelling fixed point alone cannot classify the
two-eye appearance.
