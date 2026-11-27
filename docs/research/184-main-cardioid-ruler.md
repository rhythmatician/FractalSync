# The main-cardioid ruler from the attracting multiplier

The period-one multiplier gives an explicit conformal ruler on the open main cardioid. With curvature $-1$ normalization its ratio to true Shore distance tends to $1$ at every smooth point of the cardioid boundary. At the cusp the limit depends on the approach. On the real axis the ratio tends to $2$. The current runtime ruler has a fixed positive floor, so it has different limiting behavior in both cases. [Research #182](https://github.com/rhythmatician/FractalSync/issues/182), [Lean scope #184](https://github.com/rhythmatician/FractalSync/issues/184).

## Derivation

For $f_c(p)=p^2+c$, the fixed-point equation gives $c=p-p^2$ and the multiplier is $u=2p$. The resulting parameter map is

$$
F(u)=\frac{u(2-u)}4=\frac14-\frac{(1-u)^2}4,\qquad F'(u)=\frac{1-u}2.
$$

It maps the open unit disk onto the main cardioid $H$. The chord identity $F(u)-F(v)=(u-v)(2-u-v)/4$ proves injectivity on the closed disk, since $u+v=2$ there forces $u=v=1$. The derivative is nonzero on the open disk. The unit circle maps onto $\partial H$. The orbit theorem puts every parameter with an attracting fixed point in $\operatorname{int} M$, and the boundary of this hyperbolic component is in $\partial M$. For $c\in H$, the distance to $\partial H$ therefore equals $d_S(c)=\operatorname{dist}(c,\partial M)$. A segment from $c$ to any point of $\partial M$ must leave $H$, and $\partial H$ itself is part of $\partial M$. The relevant smoothness is that of $\partial H$ even at the attachment of a satellite component. [Milnor, *Periodic Orbits, External Rays and the Mandelbrot Set*, §6](https://arxiv.org/pdf/math/9905169), [Douady and Hubbard, *Exploring the Mandelbrot Set*, chapters 3 and 14](https://pi.math.cornell.edu/~hubbard/OrsayEnglish.pdf).

The disk has curvature $-1$ length density $2/(1-|u|^2)$. Transport through $F$ gives the parameter-plane length element $|dc|/\rho_H(c)$ with

$$
\rho_H(F(u))=\frac{(1-|u|^2)|1-u|}{4}.
$$

This fixes the multiplicative constant. The squared speed on a differentiable path $c(t)=F(u(t))$ is $|\dot c|^2/\rho_H^2$, and its square root is exactly $2|\dot u|/(1-|u|^2)$. [Beardon and Minda, *The Hyperbolic Metric and Geometric Function Theory*, §2 and Theorem 6.3](https://www.math.stonybrook.edu/~bishop/classes/math401.F09/Beardon-Minda.pdf).

## Smooth boundary and cusp

Write $r=|u|$ and $v=u/r$ near the boundary. Pick a minimizing unit multiplier $a$. The chord identity and $|u-a|\ge 1-r$ give

$$
\frac{(1-r)|2-u-a|}{4}\ \le\ d_S(F(u))\ \le\ \frac{(1-r)|2-u-v|}{4}.
$$

As $u\to b$ with $|b|=1$, the radial candidate $v\to b$ gives $d_S(F(u))\to0$. The minimizing multipliers $a\to b$ because $F$ is a homeomorphism on the closed disk. For $b\ne1$ the bounds squeeze $d_S/(1-r)$ to $|1-b|/2$, which is also the limit of $\rho_H/(1-r)$. Thus $\rho_H/d_S\to1$ for any interior approach to $F(b)$.

On the cusp axis $u=r\in[0,1)$, every unit multiplier $a$ has $|r-a|\ge1-r$ and $|2-r-a|\ge 2-r-\operatorname{Re}a\ge1-r$. Both bounds attain equality at $a=1$. Hence $d_S=(1-r)^2/4$ and $\rho_H/d_S=1+r$.

For the straight multiplier approaches $u=1-t(1+i\alpha)$, with $t\downarrow0$ and fixed real $\alpha$, put $q=-(1+i\alpha)^2/4$. The rescaled parameter point $(F(u)-1/4)/t^2$ equals $q$. A boundary candidate within $O(t^2)$ of the cusp comes from $a=e^{is}$ with $s=O(t)$, by $|F(a)-1/4|=|1-a|^2/4$. Under this rescaling the limits of these boundary candidates form the nonnegative real ray since $-(1-e^{is})^2/(4t^2)\to k^2/4$ when $s/t\to k$. The cusp bounds the minimizing distance by $t^2|q|$, so the minimizing candidates are among these bounded rescaled points. It follows that $d_S/t^2\to\operatorname{dist}(q,[0,\infty))$. Evaluating that distance gives

$$
\lim_{t\downarrow0}\frac{\rho_H(F(u))}{d_S(F(u))}
=
\begin{cases}
2/\sqrt{1+\alpha^2},& |\alpha|\le1,\\
\sqrt{1+\alpha^2}/|\alpha|,& |\alpha|\ge1.
\end{cases}
$$

For instance, the oblique approach $\alpha=1$ gives $\sqrt2$.

## Comparison on the same paths

The current runtime derives $\rho_{\mathrm{run}}=\sqrt{D_{\mathrm{provider}}^2+\varepsilon^2}$ with default $\varepsilon=10^{-4}$. The table includes its ideal-distance counterpart $\sqrt{d_S^2+\varepsilon^2}$ to show where the physical floor begins to matter. At each $F(u)$, the last column uses the Rust binding installed in the project venv, package `runtime-core` 0.1.0 with `geometry-provider/6`, via `geometry_provider_query(c, 1e-4)`. Values are rounded. [Rust ruler and floor](../../runtime-core/src/manifold.rs), [provider conversion](../../runtime-core/src/geometry_provider.rs).

| Multiplier $u$ | True Shore distance $d_S$ | $\rho_H$ | Ideal regularized Shore ruler | Runtime ruler and validity |
| --- | ---: | ---: | ---: | --- |
| $-0.5$ | 0.4375 | 0.28125 | 0.4375 | 0.4370 regular |
| $-0.9$ | 0.0975 | 0.09025 | 0.0975 | 0.09728 regular |
| $-0.99$ | 0.009975 | 0.009900 | 0.009976 | 0.009949 regular |
| $-0.9999$ | 0.0001000 | 0.00009999 | 0.0001414 | 0.0009565 unresolved |
| $0.5$ | 0.0625 | 0.09375 | 0.06250 | 0.06262 regular |
| $0.9$ | 0.0025 | 0.00475 | 0.002502 | 0.002510 regular |
| $0.99$ | 0.000025 | 0.00004975 | 0.0001031 | 0.0001031 regular |
| $0.9999$ | $2.5\times10^{-9}$ | $5.0\times10^{-9}$ | 0.0001000 | 0.0001000 regular |

For the negative real path $u=-r$, the chord distance squared is the product of two nonnegative affine factors in $\operatorname{Re}a$, with opposite slopes, divided by $16$. It is concave on $[-1,1]$, so the minimum occurs at $a=-1$ or $a=1$. The former wins for $r\ge\sqrt2-1$, giving the table's exact $d_S=(1-r)(3+r)/4$ and ratio $\rho_H/d_S=(1+r)^2/(3+r)$. For the positive path the cusp-axis formula gives the exact $d_S$. These are two parameter paths and eight provider observations, with one unresolved query.

Run this in the project venv to reproduce the table. Each row prints the multiplier, true distance, multiplier ruler, ideal regularized ruler, and the runtime ruler with its validity.

```python
import json
from math import hypot

from runtime_core import geometry_provider_query, geometry_provider_version

epsilon = 1e-4
print(geometry_provider_version())
for direction in (-1, 1):
    for r in (0.5, 0.9, 0.99, 0.9999):
        u = direction * r
        c = complex(u * (2 - u) / 4)
        distance = (1 - r) * ((3 + r) if direction < 0 else (1 - r)) / 4
        rho = (1 - r * r) * abs(1 - u) / 4
        jet = json.loads(geometry_provider_query(c, epsilon))
        print(u, distance, rho, hypot(distance, epsilon),
              hypot(jet["D"], epsilon), jet["validity"])
```

## Intrinsic speed and steering

At a fixed Euclidean velocity the conformal component of intrinsic speed is inversely proportional to the ruler. Conversely, at equal conformal speed the Euclidean speeds are in the ratio $\rho_H/\rho_{\mathrm{run}}$. The two rulers are close on the smooth path in the samples above the regularization floor, but the cusp-axis Euclidean speed can approach twice the ideal unregularized Shore-based speed in that regime. At smaller distances $\rho_H/\rho_{\mathrm{run}}\to0$ because of the floor. The unregularized conformal metric is complete and applies only inside $H$.

For the conformal metric $g_H=\rho_H^{-2} I$, write $L=\nabla_c\log\rho_H$. The connection term is $\Gamma(v,v)=|v|^2 L-2(v\cdot L)v$. Thus steering depends on the transverse component of $L$, even when two rulers give similar instantaneous speeds. In multiplier coordinates,

$$
\nabla_u\log\rho_H=-\frac{2u}{1-|u|^2}+\frac{u-1}{|u-1|^2}.
$$

Treat the complex expressions as real vectors and use the inverse transpose of $DF$ to obtain $L$. The current full metric also has a scale-gradient term and the controller has forces and a timebase. A conformal Physics ablation can compare the same parameters and velocities for speed, then the same initial conditions and controls for steering, with floor and boundary behavior stated explicitly. The data above cover ruler values and provider flags. [Current metric](../../runtime-core/src/manifold.rs), [prototype #183](https://github.com/rhythmatician/FractalSync/issues/183).

## Checked scope

[MainCardioidRuler.lean](../../formal/FractalSync/MainCardioidRuler.lean) checks the fixed-point equation, polynomial derivative, chord identity, tangent-speed identity, and the finite distance bracket given a minimizing unit multiplier. It proves the exact cusp-axis distance to the image of the unit circle and the ruler factor $1+r$. The identification with the true Shore, the smooth squeeze limit, and the oblique cusp limit are conventional arguments above. The provider samples are floating-point observations.
