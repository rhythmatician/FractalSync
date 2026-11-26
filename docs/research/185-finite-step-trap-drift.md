# Finite-step drift of the second-iterate trap

The [fixed trap](185-orbit-trap-steering.md) is \(F(c)=|c^2+c|^2\). Its
[gradient-error gate](185-steering-confidence.md) certifies an oriented tangent
at the current point. The finite step needs a separate budget. Even an exact
tangent can change \(F\) over a straight step.

## One step

Write \(P=c^2+c\), \(A=2c+1\), \(g=\nabla F(c)=2\bar A P\), and let \(d\)
be the displacement in the C-plane with \(s=|d|\). For real pairing
\(\langle u,v\rangle=\mathrm{Re}(\bar u v)\), the substitution
\(P(c+d)=P+Ad+d^2\) gives the exact quartic increment

\[
F(c+d)-F(c)
=\langle g,d\rangle+2\,\mathrm{Re}(\bar P d^2)+s^2|A+d|^2.
\]

The last two terms have combined magnitude at most \(C(c,s)s^2\), where

\[
C(c,s)=2|P|+(|A|+s)^2.
\]

This coefficient uses the starting point and the proposed step length.
A segment Hessian estimate gives an alternative coefficient
\(\sup_{0\le t\le1}\{2|P(c+td)|+|2(c+td)+1|^2\}\).
Indeed the real derivative of \(g\) along \(v\) is
\(4\bar v P+2|A|^2v\). The integrated Taylor remainder multiplies the
segment bound on \(\|Dg\|\) by \(s^2/2\).

For an Euler time step \(h\ge0\) under the conformal ruler
\(G=\rho^{-2}I\), take \(d=h\rho J\hat g/|\hat g|\) after the gate passes.
Here \(J\) is the positive quarter turn and \(s=h\rho\) at the starting
point. If the gate certifies an angle at most \(\theta\), the linear
change satisfies \(|\langle g,d\rangle|\le |g|s\sin\theta\).
A Euclidean gradient error budget \(|g-\hat g|\le e\) also gives
\(|\langle g,d\rangle|=|\langle g-\hat g,d\rangle|\le es\), since \(d\)
is perpendicular to \(\hat g\). Thus

\[
|F(c+d)-F(c)|\le es+C(c,s)s^2.
\]

For a regular example with exact tangent, take a real starting point \(x>0\)
and \(d=is\). The increment is \((2x^2+2x+1)s^2+s^4>0\) for \(s>0\).

## Choosing a step

For a trial cap \(s_*>0\), set \(C_*=C(c,s_*)\). For all \(s\le s_*\) the
same bound holds with \(C_*\). To limit a single change in \(F\) to
\(\tau>0\), use

\[
s\le\min\left(s_*,\frac{2\tau}{e+\sqrt{e^2+4C_*\tau}}\right).
\]

For a path of accepted steps with lengths \(s_i\), budgets \(e_i\), and
coefficients \(C_i=C(c_i,s_i)\), every prefix satisfies

\[
|F(c_n)-F(c_0)|\le\sum_{i<n}(e_i s_i+C_i s_i^2).
\]

With uniform bounds \(e_i\le e\), \(C_i\le C\), and \(s_i\le s_*\), a
polygon of length \(L\) has drift at most \(L(e+Cs_*)\). Refinement at
fixed \(L\) reduces the curvature contribution linearly in \(s_*\). A
persistent angular bias can still accumulate a first-order contribution.
For a total tolerance \(B>eL\) with \(L>0\), a cap
\(s_*\le (B/L-e)/C\) suffices when \(C>0\) is valid for all those steps.

The [Lean module](../../formal/FractalSync/OrbitTrapDrift.lean) checks the
quartic identity, the endpoint remainder, the error-budget bound for a
displacement with or without a perpendicularity defect, and the finite sum
bound.
The segment Hessian estimate and step-size formulas above are conventional
derivations. The checked bounds concern exact real displacements for this
fixed trap. For a displacement with measured perpendicularity defect,
add \(|\langle\hat g,d\rangle|\) to the one-step bound. Budgeting a computed
update requires enclosing that term and the other quantities using its
actual arithmetic and coordinate displacement.

## Step refinement with a fixed steering bias

Start on the regular contour $F=0.16^2$ near the origin, as in the
[earlier steering study](185-orbit-trap-steering.md). In the constant
metric `ρ=1`, travel polygon length `0.5` with equal Euler steps. In the
biased case rotate the gradient estimate by `-0.001` radians at every
step. The table compares maximum drift over all prefixes with the sum
of their step budgets over the full path.

| Steps | Unbiased drift | Unbiased budget | Biased drift | Biased budget |
| ---: | ---: | ---: | ---: | ---: |
| 128 | `0.00184515` | `0.00266902` | `0.00200547` | `0.00283292` |
| 512 | `0.000462816` | `0.000657354` | `0.000621118` | `0.000816519` |
| 4096 | `0.0000579085` | `0.0000818135` | `0.000215620` | `0.000239632` |

This is a binary64 demonstration. It evaluates the analytic budgets
without adding bounds for the arithmetic of the demonstration itself.

```python
from cmath import exp
from math import sqrt


def run(steps: int, angle: float) -> tuple[float, float]:
    c = complex((sqrt(1 + 4 * 0.16) - 1) / 2)
    initial = abs(c * (c + 1)) ** 2
    length = 0.5 / steps
    rotation = exp(-1j * angle)
    budget = maximum = 0.0
    for _ in range(steps):
        p, a = c * (c + 1), 2 * c + 1
        gradient = 2 * a.conjugate() * p
        estimate = rotation * gradient
        error = abs(estimate - gradient)
        budget += error * length + (2 * abs(p) + (abs(a) + length) ** 2) * length**2
        c += 1j * length * estimate / abs(estimate)
        maximum = max(maximum, abs(abs(c * (c + 1)) ** 2 - initial))
    return maximum, budget


for steps in (128, 512, 4096):
    print(steps, *(f"{value:.8g}" for angle in (0.0, 0.001) for value in run(steps, angle)))
```
