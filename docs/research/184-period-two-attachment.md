# Period-two ruler and the attachment to the main cardioid

The period-two bulb has an exact conformal ruler in the same normalization as the [main-cardioid ruler](184-main-cardioid-ruler.md). At the attachment $c_0=-3/4$, the two rulers have smooth signed extensions. Their zero-extended fourth powers give a positive $C^2$ metric for every fixed floor $\varepsilon>0$, with Gaussian curvature zero at the attachment. [Ruler research #182](https://github.com/rhythmatician/FractalSync/issues/182), [Lean scope #184](https://github.com/rhythmatician/FractalSync/issues/184).

## Exact bulb and Shore distance

For $f_c(z)=z^2+c$ the second iterate factors as

$$
f_c^2(z)-z=(f_c(z)-z)(z^2+z+c+1).
$$

The roots $z_\pm=(-1\pm\sqrt{-4c-3})/2$ of the second factor form a period-two orbit for $c\ne-3/4$. Indeed, $f_c(z)=-1-z$ on its roots, and the roots coincide only at $-3/4$. The cycle multiplier is $(2z_+)(2z_-)=4(c+1)$ by their product. Thus the open disk $B=\{c:|c+1|<1/4\}$ consists exactly of the parameters with an attracting period-two cycle. The multiplier map $w=4(c+1)$ is a conformal bijection onto the open unit disk. Pulling back the curvature $-1$ disk length density $2/(1-|w|^2)$ yields $|dc|/r_2(c)$, where

$$
r_2(c)=\frac{1-|4(c+1)|^2}{8}=\frac18-2|c+1|^2.
$$

At $c_0$, the period-two multiplier tends to $+1$ and the fixed point has multiplier $-1$. This is the period-doubling attachment. [Milnor, *Periodic Orbits, External Rays and the Mandelbrot Set*, §6.1 and Theorem 6.5](https://arxiv.org/pdf/math/9905169), [Beardon and Minda, *The Hyperbolic Metric and Geometric Function Theory*, §2 and Theorem 6.3](https://www.math.stonybrook.edu/~bishop/classes/math401.F09/Beardon-Minda.pdf).

The bulb lies in the interior of the Map and its boundary is part of the Shore $\partial M$. For any $c\in B$, a segment to any point of the Shore first meets $\partial B$. Radial projection onto $\partial B$ attains the distance to that circle, with any direction valid at $c=-1$. The unsigned Shore distance satisfies

$$
d=\operatorname{dist}(c,\partial M)=\tfrac14-|c+1|,
\qquad r_2=d-2d^2.
$$

The ratio $r_2/d=1-2d$ tends to $1$ at every point of the circle and equals $1/2$ at the center. The identity also describes the distance from the real attachment along the inward axis $c=c_0-d$ for $0<d\le1/4$. On the cardioid side, let $c=c_0+x$ for $x>0$ small and $s=\sqrt{1-x}$. The multiplier is $u=1-2s$, so $r_1=2s^2(1-s)=2s^2 x/(1+s)=x-3x^2/4+O(x^3)$. The two unfloored rulers vanish linearly on their respective axis approaches. [Douady and Hubbard, Exploring the Mandelbrot Set, chapter 14](https://pi.math.cornell.edu/~hubbard/OrsayEnglish.pdf), [cardioid Shore distance on the real axis](184-main-cardioid-ruler.md#comparison-on-the-same-paths).

## Gluing through the attachment

The cardioid parameter map $F(u)=u(2-u)/4$ has $F(-1)=c_0$ and $F'(-1)=1$. Its local inverse $u(c)$ is holomorphic and takes values near $-1$. The formula

$$
r_1(c)=\frac{(1-|u(c)|^2)|1-u(c)|}{4}
$$

therefore defines a smooth signed function near $c_0$. Its positive set there is the cardioid. The polynomial $r_2$ is already smooth and its positive set is $B$. The two interiors are disjoint. In fact, for $u=a+ib$ with $|u|<1$, $4\operatorname{Re}(F(u)-c_0)=(a+1)(3-a)+b^2>0$. Every point in $B$ lies strictly to the left of $c_0$. The common normal at contact is horizontal and both boundary arcs are smooth there.

Write $q(t)=(\max\{t,0\})^4$ and set

$$
S(c)=\varepsilon^4+q(r_1(c))+q(r_2(c)),\qquad
R(c)=S(c)^{1/4},\qquad g=R(c)^{-2}I.
$$

For $t>0$ the derivatives of $q$ are $4t^3$ and $12t^2$. For $t\le0$ both are zero, including at the join by the difference quotient. Thus $S$ is $C^2$ and $S\ge\varepsilon^4>0$, so $R=S^{1/4}$ and the metric coefficient $S^{-1/2}$ are $C^2$. On either component the formula agrees with its quartic floor. Where both rulers are nonpositive it gives $R=\varepsilon$. At $c_0$, $\nabla S$ and $D^2S$ vanish since each signed ruler vanishes there. The Christoffel symbols vanish in these coordinates at $c_0$, and $K=R^2\Delta\log R$ is continuous through $c_0$ with $K(c_0)=0$. The same argument applies at each nearby boundary point of the two components.

The construction is a local analytical fixture for the two components. It assigns the constant floor to every remaining point in the neighborhood. A ruler for other hyperbolic components, the exterior, or a general Shore neighborhood requires a separate construction. The smooth signed extension used here comes from the specific multiplier inverses at the attachment.

## Curvature and Physics

With $s=|c+1|$, the bulb has $\Delta r_2=-8$ and $|\nabla r_2|^2=16s^2=1-8r_2$. The unfloored curvature $r_2\Delta r_2-|\nabla r_2|^2=-1$ agrees with the disk normalization. For the quartic floor, the [weighted-trace formula](184-quartic-ruler-floor.md#curvature-identity) gives the exact bulb profile

$$
K_R=\frac{r_2^2}{\sqrt{r_2^4+\varepsilon^4}}
\left[-1+\frac{4\varepsilon^4(1-8r_2)}{r_2^4+\varepsilon^4}\right].
$$

It is independent of polar angle, so it also gives the full real-axis curvature. At the center it is $-r_2^2/\sqrt{r_2^4+\varepsilon^4}$ with $r_2=1/8$. At the circle it tends to zero for fixed $\varepsilon$. In the joint limit $\varepsilon\downarrow0$ and $d/\varepsilon\to\lambda<\infty$, substitution of $r_2/d=1-2d$ gives the smooth profile shared with the cardioid side,

$$
\frac{\lambda^2}{\sqrt{1+\lambda^4}}\left[-1+\frac{4}{1+\lambda^4}\right].
$$

This is positive for $0<\lambda<3^{1/4}$ and peaks at $1$ when $\lambda=3^{-1/4}$. At a fixed floor, equal conformal speed gives Euclidean speed $R$ times that speed, so the attachment admits finite nonzero crossing speed in the fixture. The connection term from the conformal factor vanishes there. For the unfloored rulers, $r_i\sim d$ on the smooth boundaries gives logarithmically divergent length along the axis, consistent with completeness of each open component. The existing Physics metric includes scale-gradient terms, so the calculation concerns the candidate conformal term in isolation. [Current metric](../../runtime-core/src/manifold.rs).

## Crossing paths at the attachment

Write $c=-3/4+x+iy$. For $u=-e^{-it}$ on the main cardioid boundary,
$$
y=\frac{\sin t(1+\cos t)}{2},
\qquad
x_{\rm card}=\frac{1-\cos t}{2}+\frac{\sin^2 t}{2}.
$$
Consequently $x_{\rm card}/y^2=2(2+\cos t)/(1+\cos t)^3\to 3/4$.
The right edge of the bulb at the same height has
$$
x_{\rm bulb}=-\frac14+\sqrt{\frac1{16}-y^2}
            =-\frac{y^2}{1/4+\sqrt{1/16-y^2}},
$$
so $x_{\rm bulb}/y^2\to-2$. Both arcs are tangent at the attachment, and
the horizontal gap opens as $(11/4)y^2+O(y^4)$.

The two-component extension assigns ruler $\varepsilon$ across this
gap. Its horizontal length there is exactly
$(x_{\rm card}-x_{\rm bulb})/\varepsilon$. When $y=k\sqrt{\varepsilon}$
with fixed $k$, that length tends to $11k^2/4$. This is the contribution
from the gap alone for the specified path. It makes the transverse scale
$\sqrt{\varepsilon}$ relevant when comparing offset crossings with the
real-axis crossing. The formula uses the extension in this fixture.
The gap may contain other dynamics that the fixture does not model.

On the real axis the gap has zero width. Put $x=\varepsilon\xi$.
The axis formulas above give $r_i/\varepsilon\to|\xi|$ for fixed $\xi$.
The length of the crossing from $x=-a\varepsilon$ to $x=b\varepsilon$
converges to
$$
  \int_{-a}^b (1+\xi^4)^{-1/4}\,d\xi .
$$
The integrands are bounded by one, so the pointwise limit suffices.
For $a=b=2$, quadrature gives lengths 3.242836, 3.219312, and
3.219080 at floors $10^{-2},10^{-4},10^{-6}$, approaching 3.219078.
These are lengths along the specified paths. Global shortest paths and
Physics rollout behavior remain open.

The installed Rust `geometry-provider/6` binding returned `unresolved`
at the attachment, with `D ≈ −3.15e−4` when queried with floor input `1e−4`.
It also returned `unresolved` at `c = −3/4 ± 1e−3`. These are sampled
provider readings. The constructed metric gives an analytic crossing
fixture at those coordinates. [Provider comparison method](184-main-cardioid-ruler.md#comparison-on-the-same-paths)

## Checked scope

Lean checks the second-iterate factorization, exact-period condition,
multiplier and disk threshold, circle distance, ruler identity, attachment
multipliers, and scalar curvature identity. It checks $C^2$ regularity of
the glued metric coefficient given a positive floor and two $C^2$ signed
rulers. The identification with the Shore, smooth local inverse, vanishing
derivatives at the attachment, geometric curvature, and path limits follow
from the conventional arguments above. [PeriodTwoAttachment.lean](../../formal/FractalSync/PeriodTwoAttachment.lean)
