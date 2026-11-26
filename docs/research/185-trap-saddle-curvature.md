# Turning near the orbit-trap saddle

The [exact trap contour](185-exact-trap-contours.md) keeps
$F(c)=|P(c)|^2$ fixed, where $P(c)=c^2+c$. Close to the saddle at
$c_s=-1/2$, following the contour at unit intrinsic speed can demand
arbitrarily large steering acceleration even with a smooth positive ruler.

## Turning per unit phase

Use phase $\theta$ measured from the positive real $P$-axis and follow a
regular lift with $p=P(c(\theta))=re^{i\theta}$, $r>0$, and $A=2c+1\ne0$.
The local lift formula, with its phase anchored at the starting point,
applies to either choice of $A$ with $A^2=1+4p$. With primes denoting
phase derivatives,

$$
p'=ip,\qquad c'=\frac{ip}{A},\qquad A'=2c',\qquad
\frac{c''}{c'}=i\left(1-\frac{2p}{A^2}\right).
$$

Write the nonzero tangent locally as $c'=|c'|e^{i\alpha}$. Then
$c''/c'=(\log|c'|)'+i\alpha'$. The signed Euclidean curvature for
increasing phase is the heading derivative per Euclidean arc length.
Using $A^2=1+4p$ gives

$$
\alpha'=\frac12\left(1+\operatorname{Re}\frac{1}{1+4p}\right),
\qquad
\kappa_E=\frac{\alpha'}{|c'|}
=\frac{|A|}{2r}\left(1+\operatorname{Re}\frac{1}{1+4p}\right).
$$

Both lifts have the same signed curvature at a given phase. The map
$c\mapsto -1-c$ exchanges them by a Euclidean half turn. Reversing the
direction of traversal reverses the sign of curvature.

## Closest phase and intrinsic steering

For $r\ne1/4$, the closest phase to $c_s$ is $\theta=\pi$, where
$p=-r$ and $|c-c_s|=\tfrac12\sqrt{|1-4r|}$. Put $d=1-4r$.
The exact curvature there is

$$
\kappa_E(\pi)=\frac{(1+d)\operatorname{sgn}(d)}{2r\sqrt{|d|}}
\sim \frac{\operatorname{sgn}(d)}{2r\sqrt{|d|}}
\quad (r\to1/4).
$$

The sign is positive for the inner loops and negative for the outer
loop near the saddle. Restrict the conformal metric $G=\rho^{-2}I$
to a neighborhood of $c_s$ with $\rho$ positive and $C^2$. The
[main cardioid ruler](184-main-cardioid-ruler.md) meets these
conditions there since its multiplier at $c_s$ is $1-\sqrt{3}$.
Let $T$ be the Euclidean unit tangent and $N=iT$ its left normal.
Writing $u=-\log\rho$, the conformal connection gives signed
geodesic curvature $e^{-u}(\kappa_E-\partial_N u)$. Indeed,
for the $G$-unit tangent $e^{-u}T$ its covariant derivative
in its own direction is $e^{-2u}(\kappa_E-\partial_Nu)N$.
Thus

$$
k_G=\rho\kappa_E+\partial_N\rho .
$$

The second term stays bounded as $c$ tends to $c_s$, and $\rho$
tends to the positive value $\rho(c_s)$. The intrinsic curvature at
closest phase therefore has the same signed divergence, with leading
coefficient multiplied by $\rho(c_s)$.

For positive intrinsic speed $v=ds_G/dt$, elapsed time $t$ has
$d\theta/dt=v\rho|A|/r$. The covariant normal acceleration along the
contour is $v^2k_G$, and the tangent turns at signed rate $vk_G$
relative to parallel transport. At unit intrinsic speed these both
equal $k_G$. The Euclidean heading rate in elapsed time is
$v\rho\kappa_E$. A fixed positive speed through the closest phase
cannot satisfy a uniform bound on normal acceleration across these
regular levels. A bound $a_{\max}>0$ would require
$v\le\sqrt{a_{\max}/|k_G|}$ where $k_G\ne0$. This local constraint
can be tested against a proposed contour-following policy and ruler.

## Reaching the critical point

At $r=1/4$, $|1+4p|\sim|\theta-\pi|$, so
$ds_G/d\theta\sim(4\rho(c_s)\sqrt{|\theta-\pi|})^{-1}$ on either
incident regular arc. The singularity is integrable, and at unit
intrinsic speed the saddle is reached in finite elapsed time.
The square root has two choices on the outgoing side of the critical
phase. Near that phase the leading term is
$c-c_s\sim\tfrac12\sqrt{-i(\theta-\pi)}$ on the chosen branch, so the
two outgoing rays are perpendicular to the incoming tangent line.
An exact-trap continuation with increasing phase needs a choice of
outgoing branch. The regular-lift derivative and its normalized
steering direction are undefined at the saddle itself.

## Where the turn concentrates

Put $b=4r$. Away from the singular point, the sign of
$\kappa_E$ is the sign of $2+b^2+3b\cos\theta$, obtained by
multiplying $1+\operatorname{Re}(1/(1+4p))$ by $|1+4p|^2$.
Its minimum over phase is $(b-1)(b-2)$. The only regular contours
with negative Euclidean curvature therefore have $1/4<r<1/2$.
At $r=1/2$ the closest point has zero Euclidean curvature.
The conformal normal-gradient term makes this sign test specific to
Euclidean curvature.

To resolve the turn near the saddle, let
$\delta=|1-4r|$, $\sigma=\operatorname{sgn}(1-4r)$, and
$\theta=\pi+\delta\xi$. At bounded $\xi$, the discriminant is
$\delta(\sigma-i\xi)+O(\delta^2)$, uniformly on compact intervals.
Since $|A|=\sqrt{|1+4p|}$, the curvature formula gives

$$
\sqrt{\delta}\,\kappa_E\longrightarrow
   \frac{2\sigma}{(1+\xi^2)^{3/4}},
\qquad
\sqrt{\delta}\,k_G\longrightarrow
   \frac{2\sigma\,\rho(c_s)}{(1+\xi^2)^{3/4}}.
$$

Across $\xi\in[-T,T]$, arc length is $O(\sqrt{\delta})$ in either
metric for fixed $T$. The heading formula gives
$\delta\alpha'\to \sigma/(2(1+\xi^2))$. Its integral on that interval
tends to $\sigma\arctan T$. The additional conformal turn
$\int\partial_N\rho\,ds_G$ tends to zero since the derivative
is bounded and the arc shrinks. Letting $T$ grow after taking
$\delta\to0$ recovers a signed quarter turn. The turning rate
can therefore grow without bound even as the region of rapid turn
shrinks.

## Sample with the quartic cardioid ruler

At each closest phase, evaluate the sign and scale of $k_G$
using the [quartic main-cardioid floor](184-quartic-ruler-floor.md)
with $\varepsilon=10^{-4}$. The columns are scaled by
$\sqrt{\delta}$. Their predicted limits are
$\pm 2\rho(c_s)=\pm 0.401923789$, rounded. These are
binary64 evaluations of the analytic formulas.

| $\delta$ | Inner $\sqrt{\delta}\,k_G$ | Outer $\sqrt{\delta}\,k_G$ |
| ---: | ---: | ---: |
| $10^{-2}$ | $+0.517589367$ | $-0.383050644$ |
| $10^{-4}$ | $+0.413836796$ | $-0.401733611$ |
| $10^{-6}$ | $+0.403119447$ | $-0.401921887$ |

~~~python
from cmath import sqrt as complex_sqrt
from math import sqrt

EPSILON = 1e-4


def ruler_jet(c: complex) -> tuple[float, complex]:
    u = 1 - complex_sqrt(1 - 4 * c)
    s = abs(1 - u)
    a = 1 - abs(u) ** 2
    assert a > 0
    rho = a * s / 4
    gradient = (-u + a * (u - 1) / (2 * s**2)) * (1 - u) / s
    floored = (rho**4 + EPSILON**4)**0.25
    return floored, (rho / floored)**3 * gradient


def closest_curvature(delta: float, side: int) -> float:
    radius = (1 - side * delta) / 4
    a = complex_sqrt(side * delta)
    c = (a - 1) / 2
    velocity = -1j * radius / a
    normal = 1j * velocity / abs(velocity)
    euclidean = sqrt(delta) * (1 + side / delta) / (2 * radius)
    ruler, gradient = ruler_jet(c)
    return ruler * euclidean + (normal.conjugate() * gradient).real


for delta in (1e-2, 1e-4, 1e-6):
    inner, outer = (sqrt(delta) * closest_curvature(delta, side) for side in (1, -1))
    print(f"{delta:.0e} {inner:+.9f} {outer:+.9f}")
print(f"{2 * ruler_jet(-.5)[0]:.9f}")
~~~

## Machine-checked scope

[OrbitTrapCurvature.lean](../../formal/FractalSync/OrbitTrapCurvature.lean)
checks the second real-phase derivative of the branch-anchored lift
on its regular domain and computes the imaginary part of its ratio
to the first derivative for $r>0$. The heading interpretation, signed
curvature, asymptotics, and time and steering consequences above use
conventional differential geometry and calculus.
