# Exact contours of the second-iterate orbit trap

The [fixed trap](185-orbit-trap-steering.md) measures $F(c)=|P(c)|^2$ for
$P(c)=c^2+c$. The [finite-step bound](185-finite-step-trap-drift.md)
quantifies drift when a straight step follows a tangent. Rotating the
trap value and lifting that rotation to the Map gives an exact contour
reference for the same fixture.

## A branch anchored at the starting point

Write $p=P(c)\ne0$ and $A=2c+1\ne0$. The identity $A^2=1+4p$
gives a lift of $p_\theta=p e^{i\theta}$

$$
u_\theta=\frac{4p(e^{i\theta}-1)}{A^2},\qquad
A_\theta=A\sqrt{1+u_\theta},\qquad
c_\theta=\frac{A_\theta-1}{2}.
$$

Here the square root is the principal root near $1$. If
$|4p(e^{it}-1)|<|A|^2$ at every $t$ between $0$ and $\theta$,
then $1+u_t$ stays in the open right half-plane. This branch starts
at $A_0=A$, is smooth, and avoids $A_t=0$. A sufficient step test is
$4|p||\theta|<|A|^2$, since $|e^{it}-1|\le |t|$.
Squaring $A_\theta$ gives $1+4p_\theta$, so $P(c_\theta)=p_\theta$
and $F(c_\theta)=F(c)$ in exact arithmetic. Re-anchor at each accepted
point to trace a longer arc. On a regular contour of radius $r=|p|$,
$\min_\theta |1+4p e^{i\theta}|=|1-4r|$, so the uniform bound
$4r|\Delta\theta|<|1-4r|$ suffices for subdivision.

For a small floating-point increment, the same branch gives

$$
c_\theta-c=\frac{2p(e^{i\theta}-1)}{A(1+\sqrt{1+u_\theta})},
\qquad e^{i\theta}-1=-2\sin^2(\theta/2)+i\sin\theta.
$$

This avoids subtracting nearby roots and evaluating $\cos\theta-1$
directly. It still uses the same noncritical branch conditions, and a
rounded result needs its own error budget.

## Angle, time, and Euler drift

Differentiating $P(c_\theta)=p_\theta$ gives

$$
\frac{dc_\theta}{d\theta}=\frac{i p_\theta}{A_\theta}
=\frac{J\nabla F(c_\theta)}{2|A_\theta|^2},
\qquad
\left\|\frac{dc_\theta}{d\theta}\right\|_G
=\frac{|p|}{\rho(c_\theta)|A_\theta|}.
$$

$J$ is the positive quarter turn and $G=\rho^{-2}I$, with $\rho>0$.
Increasing $\theta$ follows the positive trap tangent. For positive
unit conformal speed in time, use $d\theta/dt=\rho(c_\theta)|A_\theta|/|p|$.
On an arc with a positive continuous ruler, the equivalent elapsed time
is the integral of $|p|/(\rho(c_\theta)|A_\theta|)$ over the angle arc.
Reverse the sign of the angular rate for the opposite orientation.

For an angular step $h$, compare the lift with Euler's
$d=ihP(c)/(2c+1)$. They have the same initial angle derivative.
Euler's straight chord can change $F$ by order $h^2$ in a single
regular step. The lift supplies a zero-drift reference and a pointwise
position reference for algorithms compared at the same trap phase.
Comparisons at the same elapsed time require the ruler-dependent
reparameterization.

## What happens after a turn

The discriminant $1+4p e^{i\theta}$ traverses a circle centered at
$1$. For $0<r<1/4$ the circle has winding number zero about the origin.
Its two square-root lifts close separately after $2\pi$, so the level
has two loops. For $r>1/4$ it winds once. The continuously tracked root
changes sign after $2\pi$, placing $c_\theta$ at $-1-c$. The second
turn returns to $c$ and completes the single outer loop. Each phase
has exactly two preimages, distinguished by the square-root sign.
Equal lift points must have equal trap phase, so no earlier point on
either loop repeats before its stated period. At $r=1/4$
the circle hits zero and the lift meets the saddle $c=-1/2$. The regular
angle derivative fails there. At $r=0$ the level consists of the two
trap hits $0$ and $-1$.

## Numerical benchmark

Start near the origin on $|P_2|=0.16$. March through one turn with the
same phase increments in two schemes. Euler uses
$c\gets c+ihP_2(c)/(2c+1)$ for phase step $h$. The lifted step uses the
rationalized formula below. Each scheme starts its next step from the
computed point. This compares integration in phase coordinates. Euclidean
step lengths need not match.

| Steps per turn | Euler max drift in $F$ | Lift max drift in $F$ | Lift closure error in $c$ |
| ---: | ---: | ---: | ---: |
| 128 | 0.00923976 | $3.46945\times10^{-17}$ | $1.38778\times10^{-17}$ |
| 512 | 0.00205190 | $5.20417\times10^{-17}$ | $3.68629\times10^{-17}$ |
| 4096 | 0.000247933 | $2.60209\times10^{-16}$ | $5.11602\times10^{-16}$ |

With $|P_2|=0.35$ and 512 lifted steps per turn, the first turn ends
within $6.07\times10^{-16}$ of $-1-c_0$, and the second ends within
$1.42\times10^{-15}$ of $c_0$. These are binary64 observations. The
exact level preservation and branch behavior follow from the lift.

~~~python
from cmath import sqrt as complex_sqrt
from math import sin, sqrt, tau


def phase_trace(start: complex, steps: int, use_lift: bool,
                turns: int = 1) -> tuple[complex, float]:
    phase = tau / steps
    delta = complex(-2 * sin(phase / 2) ** 2, sin(phase))
    c = start
    initial = abs(c * (c + 1)) ** 2
    maximum = 0.0
    for _ in range(turns * steps):
        p, a = c * (c + 1), 2 * c + 1
        if use_lift:
            assert p and 4 * abs(p) * abs(phase) < abs(a) ** 2
            root = complex_sqrt(1 + 4 * p * delta / (a * a))
            c += 2 * p * delta / (a * (1 + root))
        else:
            c += 1j * phase * p / a
        maximum = max(maximum, abs(abs(c * (c + 1)) ** 2 - initial))
    return c, maximum


start = complex((sqrt(1 + 4 * 0.16) - 1) / 2)
for steps in (128, 512, 4096):
    _, euler_drift = phase_trace(start, steps, False)
    end, lift_drift = phase_trace(start, steps, True)
    print(steps, *(f"{v:.6g}" for v in (euler_drift, lift_drift, abs(end - start))))

outer = complex((sqrt(1 + 4 * 0.35) - 1) / 2)
one, _ = phase_trace(outer, 512, True)
two, _ = phase_trace(outer, 512, True, turns=2)
print("Outer closure", f"{abs(one + 1 + outer):.6g}", f"{abs(two - outer):.6g}")
~~~

## Verification scope

The [Lean module](../../formal/FractalSync/OrbitTrapContours.lean) checks
the exact trap identity, the discriminant step guard, the actual
real-angle derivative, tangency to the existing trap differential, and
squared conformal speed. The angular step bound, elapsed-time conversion,
and full-turn classification use the conventional arguments above.
This diagnostic provides a reference for the fixed trap. Choosing a
Player constraint still needs evidence from the Map-feature and
trajectory comparisons in #185.
