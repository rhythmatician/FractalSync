# Julia sensitivity calibration and second panel

The frozen probe now checks its spectrum calculation against analytic
location-scale templates and a fold. A separate six-center run at 160×160
extends the first renderer measurement. The analytic comparisons converge.
The rendered results still support a stencil-locality benefit near the sampled
Shore, with no resolved two-sided spectrum across the tested step sizes.
[First panel and validity protocol](151-frozen-julia-sensitivity.md).

## Analytic checks with the same spectrum calculation

For $\beta\in\{0,0.7\}$ use the normalized amplitude

$$
f_\beta(x,y)=\frac{(1+\beta x)e^{-(x^2+y^2)/2}}{\sqrt{\pi(1+\beta^2/2)}}.
$$

The location-scale family is $\rho^{-1}f_\beta((z-c)/\rho)$ with
$\rho=\sqrt{c_x^2+0.9^2}$. The translation columns at unit scale are
$-\partial_x f_\beta$ and $-\partial_y f_\beta$. The scale column is
$-f_\beta-(x,y)\cdot\nabla f_\beta$. Integrating their products against the
Gaussian gives the constant Gram matrix

$$
K_\beta=\frac{1}{1+\beta^2/2}
\begin{pmatrix}
1/2+3\beta^2/4 & 0 & \beta/2\\
0 & 1/2+\beta^2/4 & 0\\
\beta/2 & 0 & 1+\beta^2
\end{pmatrix}.
$$

The mixed term survives for the asymmetric template. Pulling this matrix back
by $c\mapsto(c,\rho(c))$ supplies an independent reference for each chart basis.
The Gaussian at $a^2=2$ gives both full-metric eigenvalues exactly $1/2$.
For $\beta=0.7$ the full-metric eigenvalues at the fixture center are
$0.5$ and $0.8482785524$.

The probe represents each translated/scaled template by 96² Float64 midpoint
samples at spacing $1/8$, with weights for $L^2$ comparison. It evaluates
central differences at $h=0.02,0.01,0.005$ in the three metric bases and in a
rotated copy of each basis. The largest relative eigenvalue error at the three
steps was $9.06\times10^{-4}$, $2.27\times10^{-4}$, and $5.66\times10^{-5}$.
The rotated measured Gram has an off-diagonal entry for the asymmetric case
and preserves the reference spectrum.
[Location-scale formulas](161-sensitivity-preconditioning.md),
[calibration code](../../frontend/scripts/julia_sensitivity_calibration.mjs).

For $V(t,y)=(\sqrt t,y)$ and $G_0=t^{-2}I$, the small and large eigenvalues
scale as $t^2$ and $t/4$. The probe recovers exponents $2$ and $1$ on
$t=2^{-6},2^{-8},\ldots,2^{-14}$. A rank-one linear control measured
$(-1.39\times10^{-17},1.00)$ in a rotated basis. The reported small signed
roundoff residual is retained. These controls validate the differencing and
spectrum machinery with known targets.

## Second frozen-renderer panel

The installed native wheel again reported `geometry-provider/6`. Four of six
new centers were regular, from $|D|/\varepsilon=54.3$ on the exterior collar
to $6,877$ in the distant exterior. The attachment was unresolved and the
period-three center was singular. Both were excluded before differencing.
Presentation and the five spacings stayed fixed. The doubled per-axis
resolution accompanied a different panel, so the two runs do not isolate
the effect of resolution.

| Center | Euclidean / scale-only / graph stencils measured | Observed rank and refinement |
| --- | --- | --- |
| Cardioid | 5 / 5 / 5 | All above the RGB8 rank-two budget. No refinement pass. |
| Period two | 5 / 5 / 5 | All above the RGB8 rank-two budget. No refinement pass. |
| Exterior collar | 2 / 5 / 5 | Two Euclidean stencils span sampled Shore and one hits invalid geometry. The finest scale-only and graph probes fall to ranks one and zero. |
| Distant exterior | 5 / 5 / 5 | All three normalizations fall below rank two at the finest spacings. |

The script keeps all 57 measured spectra and excluded endpoint records.
No adjacent pair passed the refinement criterion. Per-stencil rounding-only
intervals remain in the report where both singular values clear that budget.
Those intervals precede a derivative convergence check. There is no empirical
two-sided derivative bound or direction-specific Julia exponent to report
from these panels. Retain the three-way diagnostic and the current production
geometry. A revised representation or resolution study needs to give a stable
measurement before comparing cross-scale sensitivity. The morphology study
reports additional effects of presentation, iteration budget, and zoom.
[Morphology study](140-julia-morphology-provenance.md),
[geometry decision](182-dynamical-ruler-decision.md),
[remaining #171 acceptance](https://github.com/rhythmatician/FractalSync/issues/171).

```sh
node frontend/scripts/check_julia_sensitivity.mjs --calibrate
node frontend/scripts/check_julia_sensitivity.mjs --held-out
```

The second command saves `tmp/julia-151/held-out.json` by default. The first
requires the frontend's Node dependencies. The second also uses the same
native Python and Chromium prerequisites as the original probe.
