# Fixed point-trap steering on the second critical iterate

This slice uses a point trap at the origin and a fixed orbit index. For
\(f_c(z)=z^2+c\) and \(z_0=0\), the second iterate is \(P_2(c)=c^2+c\).
Use the squared distance \(F(c)=|P_2(c)|^2\), which gives the same level sets
as the distance and is differentiable at trap hits. The two hits are the
period-one center \(c=0\) and the period-two center \(c=-1\).
[Issue #185](https://github.com/rhythmatician/FractalSync/issues/185)

## Tangency and critical points

Write \(g=\nabla F\) in Euclidean coordinates and let \(J\) be the positive
quarter turn. At a regular point, the conformal metric \(G=\rho^{-2}I\)
with \(\rho>0\) gives \(\nabla_G F=\rho^2 g\). Thus

\[
T=\frac{J\nabla_G F}{\|\nabla_G F\|_G}=\rho\frac{Jg}{|g|},
\qquad \|T\|_G=1,\qquad dF(T)=g\cdot T=0 .
\]

A differentiable trajectory solving \(\dot c=T(c)\) on the regular set
therefore keeps \(F\) constant by the chain rule. The argument is pointwise
in \(\rho\). Solving for a trajectory requires regularity of the field, and
a finite time step can drift off the level set.

In complex coordinates \(g=2\overline{(2c+1)}(c^2+c)\), so
\(|g|^2=4|2c+1|^2F\). The critical points are exactly \(0,-1,-1/2\).
The trap hits are strict minima, with each at the center of a family of
small closed contours. At \(c=-1/2\), write \(w=c+1/2=re^{i\theta}\).
Then

\[
F(c)-1/16=|w|^4-\tfrac12\operatorname{Re}(w^2).
\]

This point is a saddle. The nonzero points on its level satisfy
\(r^2=\cos(2\theta)/2\). The four arcs approaching \(r=0\) have tangents
along the two diagonals, so the contour alone cannot select a continuation
through the crossing. The leading terms of \(g\) are \(-\bar w\) at the
saddle and \(2(c-c_*)\) at either minimum \(c_*\). Approaches along the
real and imaginary axes give different limits for the unit tangent when
\(\rho\) has a positive continuous extension there.

The [Lean module](../../formal/FractalSync/OrbitTrapSteering.lean) checks
the real Fréchet derivative, the unit-speed and tangency equations, the
exact critical-set classification, and the polynomial identity at the
saddle. The level-set interpretation and the limiting-direction argument
above are conventional proofs.

For this fixture, a contour direction is available only where \(\rho>0\)
and \(g\ne0\). A finite-precision comparison also needs a confidence
rule tied to position or gradient error. For smooth \(g\), differentiating
\(g/|g|\) projects \(Dg\) perpendicular to \(g\) and divides it by \(|g|\).
If a gradient estimate \(\hat g\) has error at most \(e<|\hat g|\), the
disk of possible gradients gives an angular error bound
\(\arcsin(e/|\hat g|)\). If \(e\ge|\hat g|\), this bound offers no
certificate of a regular point. The fixed iteration and smooth squared
distance remove closest-iteration switching and the distance cusp from
this fixture.

## Numerical comparison

The experiment uses \(\rho=1\), double precision, and the positive
orientation. The regular start is on \(F=0.16^2\) near \(c=0\). It compares
forward Euler trap steering with the straight geodesic in the same metric.
Both start in the same direction and travel polygon length \(0.5\).

| Steps | Maximum trap-steering drift in \(F\) | Maximum geodesic drift in \(F\) |
| ---: | ---: | ---: |
| 128 | 0.00184515 | 0.3925 |
| 256 | 0.000924605 | 0.3925 |
| 512 | 0.000462816 | 0.3925 |

Halving the Euler step nearly halves its drift. The geodesic's endpoint
stays fixed as the step is refined. Along the geodesic \(c=x_0+iy\),
\(F(c)-F(x_0)=y^4+(1+2\cdot0.16)y^2\), so its maximum drift occurs at
\(y=0.5\).
The second probe starts at \(c_*+r\) for each critical point and applies a
transverse perturbation of \(0.01r\). It reports the change in tangent
angle divided by the perturbation size.

| \(r\) | Saddle, radians per unit | Minimum at \(0\) | Minimum at \(-1\) |
| ---: | ---: | ---: | ---: |
| \(10^{-2}\) | 100.077 | 99.026 | 101.027 |
| \(10^{-3}\) | 999.975 | 998.97 | 1000.97 |
| \(10^{-4}\) | 9999.67 | 9998.67 | 10000.7 |
| \(10^{-5}\) | 99996.7 | 99995.7 | 99997.7 |

Copy the block into the project Python interpreter. It uses the standard
library.

~~~python
from math import atan2, sqrt


def trap(c: complex) -> float:
    return abs(c * (c + 1)) ** 2


def tangent(c: complex) -> complex:
    gradient = 2 * (2 * c + 1).conjugate() * c * (c + 1)
    return 1j * gradient / abs(gradient)


def drift(start: complex, steps: int, steer: bool) -> float:
    c = start
    value = trap(start)
    direction = tangent(start)
    maximum = 0.0
    for _ in range(steps):
        if steer:
            direction = tangent(c)
        c += (0.5 / steps) * direction
        maximum = max(maximum, abs(trap(c) - value))
    return maximum


start = complex((sqrt(1 + 4 * 0.16) - 1) / 2)
for steps in (128, 256, 512):
    print(steps, f"{drift(start, steps, True):.6g}", f"{drift(start, steps, False):.6g}")

for radius in (1e-2, 1e-3, 1e-4, 1e-5):
    epsilon = 0.01 * radius
    sensitivities = []
    for critical in (-0.5, 0.0, -1.0):
        c = critical + radius
        relative = tangent(c).conjugate() * tangent(c + 1j * epsilon)
        sensitivities.append(abs(atan2(relative.imag, relative.real)) / epsilon)
    print(f"{radius:.0e}", *(f"{value:.6g}" for value in sensitivities))
~~~

The leading terms above predict a transverse angular sensitivity of order
\(1/r\) at each critical point, matching the tenfold increases in the table.
The fixed-point multiplier branch \(\Lambda_1=1-\sqrt{1-4c}\) has a
simple zero at \(0\). The period-two multiplier \(\Lambda_2=4(c+1)\)
has a simple zero at \(-1\). Near either trap hit, squared multiplier
magnitude and \(F\) are proportional to leading order. At the saddle,
\(\Lambda_1\) and its derivative are both nonzero, so the multiplier
magnitude has a regular contour there. These local facts leave the
predictive value of the trap feature open.

Keep the fixed-index trap as a research diagnostic. An adoption decision
for the Player still needs the issue's cross-component Map-feature
ablation, a comparison using the candidate nonconstant ruler, and
trajectory presentation with the Julia settings held fixed. Escape and
minimum-distance selection across orbit indices need separate validity
rules before using configurable traps.
