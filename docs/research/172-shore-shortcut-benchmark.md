# Shore shortcut benchmark

This fixture compares Shore arc lengths, minimizing flat excursions, and stationary paths on a circular Shore patch. It gives navigation and probe tests reference values for route ambiguity, validity restrictions, and the difference between traveled length and endpoint distance. [#172](https://github.com/rhythmatician/FractalSync/issues/172), [benchmark checker](../../formal/checks/check_shore_shortcuts.py).

## Exact reference and numerical comparison

Use Shore coordinate \(y\), signed normal \(r\), regularization \(\epsilon>0\), and \(B=1+a^2\). The flat metric is

\[
g=\frac{\epsilon^2+Br^2}{(\epsilon^2+r^2)^2}\,dr^2
  +\frac{dy^2}{\epsilon^2+r^2}.
\]

For physical endpoint span \(L\), put \(\ell=L/\epsilon\). The checked threshold is \(\ell=\pi\). At and below it the endpoint distance is \(\ell\). Every minimizing absolutely continuous path remains on Shore and traverses the segment monotonically, with pauses allowed. Strictly shorter paths exist above \(\pi\). The prior conventional classification identifies two reflected minimizing excursions in this unrestricted flat model. The numerical check measures the two first-integral candidates against the Shore length. [Checked minimum and equality case](160-flat-global-minimum.md), [conventional excursion and cut-locus classification](160-flat-shore.md#3-flat-shore-geodesics-and-an-exact-shortcut-threshold).

For excursion height \(A=\epsilon H\), write \(r=\pm\epsilon H\sin u\), \(z=H^2\sin^2u\), and \(W=\sqrt{1+Bz}\). The tangential derivative is \(dy/du=\epsilon W\). The half-excursion spans \(0\le u\le\pi/2\), giving

\[
\ell(H)=2\int_0^{\pi/2} W\,du,\qquad
S(H)=2\sqrt{1+H^2}\int_0^{\pi/2}\frac{W}{1+z}\,du .
\]

The solver brackets \(H\) by increasing the upper endpoint and then bisects the strictly increasing span. It repeats the calculation with 256 and 512 Simpson panels. An independent integrand evaluates the metric speed directly along each full excursion with \(0\le u\le\pi\). It reproduces the reduced arc length for both signs and for \(\epsilon=0.007,1,250\). At Shore the two initial bearings relative to the positive tangent are \(\pm\arctan H\). Reflection preserves length while changing the initial bearing.

| \(a\) | \(\ell\) | Height \(A/\epsilon\) | Arc length of each excursion | Saving from Shore | Initial bearings, radians |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 4 | 1.144218606 | 3.830764130 | 0.169235870 | \(\pm0.852556298\) |
| 1 | 4 | 0.809084735 | 3.897669221 | 0.102330779 | \(\pm0.680255919\) |
| 3 | 4 | 0.361833694 | 3.974996960 | 0.025003040 | \(\pm0.347177943\) |

All nine combinations of \(a=0,1,3\) and \(\ell=3.2,4,6\) gave shorter arcs. The resolved cases require a gap greater than both \(10^{-10}\) and 100 times the observed panel-doubling change in arc length. This is a numerical regression margin. The exact guarantees come from the separate mathematics.

Below and at the threshold, the fixture checks translated and nonuniform monotone Shore traversals with a pause, explicit backtracking, a sine excursion, and the zero-span constant and moving paths. It also checks the independent metric quadrature's centered amplitude second derivative on both sides of \(\pi\) and at equality for all three widths and graph weights. Its analytic target is \(\ell((\pi/\ell)^2-1)/2\). The exact minimum at equality uses the global argument. [Checked sine-family variation and strict shortening](160-flat-second-variation.md).

## Near-threshold cancellation

Put \(s=\sqrt{1+H^2}\). Direct subtraction of the displayed lengths loses the shortening near \(\pi\). The identity

\[
\int_0^{\pi/2}\frac{du}{1+H^2\sin^2u}=\frac{\pi}{2s}
\]

follows by substituting \(\tan u\). The integral of \(1-s/(1+z)\) is therefore zero. The comparison evaluates the two small quantities using the rationalized terms

\[
W-1=\frac{Bz}{W+1},\qquad
1-\frac{s}{1+z}=\frac{z-H^2/(s+1)}{1+z},
\]

\[
\ell(H)-\pi=2\int_0^{\pi/2}(W-1)\,du,\qquad
\ell(H)-S(H)=2\int_0^{\pi/2}(W-1)\left(1-\frac{s}{1+z}\right)\,du.
\]

Expanding in \(H\) gives \(\ell-\pi\sim B\pi H^2/4\) and \(\ell-S\sim B\pi H^4/16\). Thus \(H/\sqrt{\ell-\pi}\) tends to \(2/\sqrt{B\pi}\), and the shortening divided by \((\ell-\pi)^2\) tends to \(1/(B\pi)\). The checker uses excesses \(10^{-4},10^{-8},10^{-12}\) directly in the bracket solve. For \(a=1\) and excess \(10^{-12}\), the stable gap was \(1.592\times10^{-25}\). Direct subtraction returned \(-4.44\times10^{-16}\) on this run. The result from direct subtraction is diagnostic and its sign is unasserted.

An independent strict comparator uses the sine path \(r/\epsilon=m\sin(\pi y/L)\). With \(y/\epsilon\) as parameter, \(w=r/\epsilon\), and the inequality \(\sqrt{1+x}\le1+x/2\), its length satisfies

\[
S_{\sin}-\ell
\le \frac{m^2\ell}{4}
\left[(1+Bm^2)(\pi/\ell)^2-\frac1{1+m^2}\right].
\]

Here the squared speed minus one is bounded above by \((1+Bm^2)(dw/d(y/\epsilon))^2-w^2/(1+m^2)\). For \(m=1/4\), \(a=1\), and \(\ell=4\), replacing \(\pi\) by \(22/7\) gives the exact rational upper bound \(-6575/426496\). Direct integration gave \(S_{\sin}=3.978162027178\). This comparison has a strict margin independent of the amplitude solver and binary64 cancellation.

## Flat validity and geodesic continuation

For \(a=1\) and \(\ell=4\), restricting paths to \(r\ge 0\) keeps the positive minimizing excursion and removes its reflection. The minimum stays \(3.897669220614\) to the reported precision. Restricting paths to the closed strip \(|r|\le \epsilon/20\) removes both minimizing excursions. Every admissible path has length at least \(4/\sqrt{1+1/400}=3.995009355511\) because the tangential speed alone pays this much. The unrestricted rational sine bound above is strictly below that lower bound. Within the strip, the sine path of normalized amplitude \(1/20\) has length \(3.999045669887\), strictly below the Shore length by the same rational estimate. These inequalities give a restricted distance interval independent of any computed optimal path in the strip. [Flat all-minimizer classification](160-flat-shore.md#3-flat-shore-geodesics-and-an-exact-shortcut-threshold).

The unit-speed Shore geodesic from \(y=0\) to \(y=3\epsilon\) realizes endpoint distance 3. Continuing it to \(4\epsilon\) gives traveled length 4 and endpoint distance \(3.897669220614\). A local geodesic continuation can therefore remain valid while its arc length ceases to be a shortest-distance oracle. The checker also compares six pairs on Shore with the lower bound \(2\log(1+\ell/2)\), reverses the excursion quadrature with a finer panel count, and checks the triangle inequalities for all ordered triples. [Checked Shore geodesic](160-flat-geodesics.md), [two-ended displacement bound](174-two-ended-displacement.md).

## Circular local fixture

Use a circle of radius \(R\) with outer normal \(r\) and base arclength \(y=R\theta\). The map \((r,y)\mapsto(R+r)(\cos(y/R),\sin(y/R))\) has signed distance \(r\) and unique radial foot in the chosen arc and annulus. The metric has the same normal coefficient and replaces \(dy^2\) by \((1+r/R)^2dy^2\). This shows how curvature enters the length integral of the same two flat trial paths. Their domain has \(|r|<R/10\) and \(0\le y/R<\pi\).

At \(a=1\), \(\ell=4\), the sign of \(r\) makes the inward trial shorter and the outward trial longer than the flat trial. The triangle inequality for the speed norm bounds the absolute change by \((\epsilon/R)H\ell\), since the change in the tangential speed component is \(|r/R|\,|\dot y|/\rho\).

| \(\epsilon/R\) | Inward trial | Flat trial | Outward trial |
| ---: | ---: | ---: | ---: |
| 0.001 | 3.895958029 | 3.897669221 | 3.899380513 |
| 0.010 | 3.880561886 | 3.897669221 | 3.914786675 |

These are lengths of prescribed paths in the circular metric. The curved minimizing branches and curved cut locus remain separate questions. [Curved normal-coordinate metric](160-curved-coordinate-geometry.md), [local curved-Shore bifurcation analysis](160-flat-shore.md#5-curved-shore-a-rigorous-local-cubic-root-branch).

## Circular geodesic branches

For the boundary-value solve, switch to the Frenet sign used in the curved-Shore analysis. Write \(x=y/\epsilon\), \(q=r/\epsilon\), \(\delta=\epsilon\kappa\), with \(\delta>0\) denoting the inward normal and \(|\delta|=\epsilon/R\). The normalized diagonal metric coefficients are

\[
M(q)=\frac{1+Bq^2}{(1+q^2)^2},\qquad
N(q)=\frac{(1-\delta q)^2}{1+q^2}.
\]

The length Euler–Lagrange equation with \(x\) as parameter gives

\[
q''=\frac{N'}{2M}
  +\left(\frac{N'}{N}-\frac{M'}{2M}\right)(q')^2,
\]

where primes on \(M,N\) denote derivatives with respect to \(q\). The checker bisects a bracket for \(q(\ell/2)\), integrates from \(q'(\ell/2)=0\) to \(q(\ell)=0\) by RK4, and reflects the resulting half-curve in \(x\). It repeats the solve with 128, 256, and 512 steps per half. Every RK stage stays in \(|\delta q|<0.1\) while \(|\delta|\ell<\pi\). Independent length quadrature uses ambient Cartesian velocity and the lifted graph velocity \(a\,d\sqrt{1+q^2}/dx\). The conserved tangential momentum \(N/\sqrt{M(q')^2+N}\) provides a residual check. From the middle to the finest grid, the length changes by less than \(3\times10^{-9}\) in every shot and by less than 0.5% of the saving at the flat threshold. The checker also verifies convergence of the midpoint heights. [Curved normal-coordinate metric](160-curved-coordinate-geometry.md), [local bifurcation analysis](160-flat-shore.md#5-curved-shore-a-rigorous-local-cubic-root-branch).

At \(\ell=\pi\), the measured sine coefficient \(A\) approaches the signed prediction \(A_0=(8\delta/(\pi B))^{1/3}\). The reduced length expansion also gives \(\pi-S\sim 3\delta A_0/2\) by substituting the branch relation \(\delta\sim \pi B A^3/8\) into the leading terms \(-2\delta A+\pi B A^4/16\). At \(|\delta|=10^{-7}\), the amplitude ratios differed from 1 by at most \(7.3\times10^{-6}\) across \(a=0,1,3\). At that curvature the final refinement changed \(A\) by at most \(2.4\times10^{-9}\). Reversing the normal reverses the sign of the measured coefficient and preserves the length.

| \(\delta\), with \(a=1\) | \(A\) | \(A/A_0\) | \(\pi-S\) |
| ---: | ---: | ---: | ---: |
| \(10^{-3}\) | 0.108594303 | 1.001929126 | \(1.622260468\times10^{-4}\) |
| \(10^{-5}\) | 0.023353007 | 1.000090814 | \(3.502282304\times10^{-7}\) |
| \(10^{-7}\) | 0.005030817 | 1.000004199 | \(7.546159253\times10^{-10}\) |

At \(\ell=2.5\) and \(a=1\), the midpoint agrees with the linear prediction \(\delta[\sec(\ell/2)-1]\) within \(1.4\times10^{-5}\) relative error at \(\delta=10^{-3}\), and within \(1.4\times10^{-7}\) at \(\delta=10^{-4}\). At \(\ell=4\) both supercritical shoots are shorter than their prescribed circular trials above. Their lengths differ, and both remain below the Shore arc length 4.

| \(\delta\) | Inward shot | Flat excursion | Outward shot |
| ---: | ---: | ---: | ---: |
| 0.001 | 3.895956792 | 3.897669221 | 3.899379268 |
| 0.010 | 3.880441251 | 3.897669221 | 3.914658799 |

The shoots give reproducible stationary-path comparisons. The conventional local result gives strict local minimality at \(\ell=\pi\) for sufficiently small nonzero \(\delta\). The supercritical shoots give candidate route lengths. Establishing the global curved minimum and finding all competing paths would require additional analysis.

## Reproduce and use

Run from the repository root with the existing environment.

```powershell
.venv/Scripts/python.exe formal/checks/check_shore_shortcuts.py
```

The checker uses only the Python standard library. The competing bearings and strip restriction support reporting candidate lengths and branch validity wherever guidance or probes expose a route. Comparing those values with production GeometryProvider and cockpit behavior requires tests at the Rust entry points. Any special move or reward decision needs separate evidence from controlled-motion replays, including timing and work. [#172 acceptance and integration scope](https://github.com/rhythmatician/FractalSync/issues/172).
