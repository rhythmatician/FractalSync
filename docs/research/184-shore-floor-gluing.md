# Quartic floor across the finite Shore

Use the reciprocal of the curvature $-1$ Poincaré density on the exterior
and on each component of the Map's interior. Give the ruler value zero on
the Shore. Its fourth power is $C^3$ on the finite parameter plane, and its
first three derivatives vanish on the Shore. For every fixed floor
$\varepsilon>0$, the metric $(\rho^4+\varepsilon^4)^{-1/2}|dc|^2$ is
positive and $C^3$ with zero Gaussian curvature on the Shore.
[Ruler research #182](https://github.com/rhythmatician/FractalSync/issues/182).

## Ruler versus Shore distance

Let $M$ denote the Mandelbrot set, $S=\partial M$, $\Omega=\mathbb C\setminus M$,
and $d(c)=\operatorname{dist}(c,S)$. The exterior uniformization
$\Phi:\Omega\to\{|w|>1\}$ is normalized at infinity. Thus $\Omega$ is connected
and $M$ is full. Each interior component $U$ is simply connected. Indeed,
the bounded region enclosed by a Jordan curve in $U$ lies in $M$, since
the connected exterior reaches infinity without meeting the curve. The
enclosed region is open, connected and meets $U$ near the curve. It
therefore lies in $U$, which gives simple connectivity by the Jordan-curve
criterion. This argument applies to every interior component independently
of its dynamical classification.
[Douady and Hubbard, *Exploring the Mandelbrot Set*, Chapter 8, Theorem 8.1](https://pi.math.cornell.edu/~hubbard/OrsayEnglish.pdf#page=64),
[Chapter 2, proof of Proposition 2.3(a)](https://pi.math.cornell.edu/~hubbard/OrsayEnglish.pdf#page=17).

The boundary of each $U$ lies in $S$. A boundary point in $M^\circ$ would
lie in an open disk in $M^\circ$ meeting $U$, and thus belong to $U$.
Since $S$ is also disjoint from $U$, $d(c)=\operatorname{dist}(c,\partial U)$
when $c\in U$. At an exterior point, $\partial\Omega=S$.

For $c\in U$, take a conformal map $f:U\to\mathbb D$ with $f(c)=0$. Then
$\rho(c)=1/(2|f'(c)|)$. Applying Schwarz's lemma on $B(c,d(c))$ gives the
lower bound below. Applying the Koebe quarter theorem to $f^{-1}$ gives
$d(c)\ge 1/(4|f'(c)|)$ and hence the upper bound.

$$
\frac{d(c)}2\le\rho(c)\le 2d(c) \qquad (c\in U).
$$

On $\Omega$ put $g=\log|\Phi|$. Then $\rho=g/|\Phi'/\Phi|$.
There is a single collar $d(c)<\delta$ in which $g(c)<\pi$ for all exterior
points. Otherwise there would be a sequence approaching $S$ with
$|\Phi|\ge e^\pi$. Its images are bounded because $\Phi^{-1}$ tends to infinity
at infinity. A subsequential limit would lie in $\{|w|>1\}$, so its inverse
image would be exterior, contradicting approach to $S$.
[Exterior ruler and its normalization](184-exterior-ruler-bounds.md#exterior-coordinate).

For $c$ in this exterior collar choose $t_0$ with $e^{t_0}=\Phi(c)$
and $\operatorname{Re}t_0=g(c)$. The map
$t\mapsto\Phi^{-1}(e^t)$ is univalent on $B(t_0,g(c))$ because the ball
lies in the right half-plane and has imaginary width less than $2\pi$.
Its derivative at $t_0$ has modulus $1/|\Phi'(c)/\Phi(c)|$. Applying the
quarter theorem to this ball gives $d(c)\ge g(c)/(4|\Phi'(c)/\Phi(c)|)$.
For the lower bound, take a holomorphic logarithm $t(z)$ of $\Phi(z)$ on
$B(c,d(c))$ with $t(c)=t_0$. The function
$f(z)=(t(z)-t_0)/(t(z)+\overline{t_0})$ maps the ball into the unit disk,
vanishes at $c$ and satisfies $|f'(c)|=1/(2\rho(c))$. Schwarz's lemma gives
the lower bound as before.

$$
\frac{d(c)}2\le\rho(c)\le 4d(c) \qquad (c\in\Omega,\ d(c)<\delta).
$$

The univalent-function input is the quarter theorem with the image disk
centered at the image of the source center.
[Bieberbach, *Über die Koeffizienten derjenigen Potenzreihen, welche eine schlichte Abbildung des Einheitskreises vermitteln* (1916), pp. 940–955](https://upload.wikimedia.org/wikipedia/commons/4/49/Sitzungsberichte_der_K%C3%B6niglich_Preussischen_Akademie_der_Wissenschaften_zu_Berlin_%28IA_sitzungsberichte1916deut%29.pdf).

## Uniform derivative bounds

On $B(c,d)$, with $d=d(c)$, use the normalized disk map $f$ just
constructed on the appropriate side. Both constructions give
$\rho(z)=(1-|f(z)|^2)/(2|f'(z)|)$ throughout this ball. On the exterior
this follows from the Cayley map and the locally pulled-back half-plane
density. Define $F(\zeta)=f(c+d\zeta)$ for $|\zeta|<1$.
The distance comparisons give $|F'(0)|=d/(2\rho(c))\ge 1/8$ at every
point in the collar off $S$. We have

$$
d^{-1}\rho(c+d\zeta)=\frac{1-|F(\zeta)|^2}{2|F'(\zeta)|}.
$$

Cauchy's estimates on $|\zeta|=1/2$ give $|F^{(k)}(0)|\le k!2^k$ for
$k\le4$. Each real derivative through order three of the right-hand
side at $0$ is a continuous expression in those bounded derivatives
on the closed set $|F'(0)|\ge1/8$. The denominator is bounded away from
zero there. Rescaling gives $\|D^j\rho(c)\|\le C_j d^{1-j}$ for
$0\le j\le3$, with constants independent of the component and of the
center. The product rule then gives

$$
\|D^j(\rho^4)(c)\|\le A_j d(c)^{4-j},\qquad 0\le j\le3.
$$

In each product-rule term the four derivative orders sum to $j$, so
the distance powers sum to $4-j$.

## Gluing and curvature

Extend $Q=\rho^4$ and its three derivative fields by zero on $S$. For each
$s\in S$, $d(c)\le |c-s|$. The bounds with powers $4$, $3$ and $2$ make the
successive difference quotients for $Q$, $DQ$ and $D^2Q$ tend to zero at
$s$. The power-$1$ bound gives continuity of $D^3Q$ at $s$. Away from
$S$ the fields are smooth. This argument covers cusps, attachment points
and accumulation of interior components without a boundary chart.

With $R=(Q+\varepsilon^4)^{1/4}$, the curvature of $R^{-2}|dc|^2$ is
$K=R^2\Delta\log R$. The first and second derivatives of $Q$ vanish at
each Shore point, so $\Delta\log R=0$ there. This proves the fixed-floor
extension independently of the way a point approaches the Shore.
At fixed $\varepsilon$, the collar bounds make the Christoffel symbols
$O(d^3)$ in parameter coordinates, since they are linear in the gradient
of $-\log R$. The curvature is $O(d^2)$ from the Hessian bound on $Q$.
The constants in these estimates depend on the floor.
[Conformal curvature convention](184-conformal-floor-curvature.md),
[quartic curvature away from the Shore](184-quartic-ruler-floor.md).

## Sharpness at the cardioid cusp

The quartic construction has no $C^4$ extension at the cusp
$c_*=1/4$ that agrees with the interior ruler. Let
$u=1-t(1+i a)$ with fixed real $a$ and $t\downarrow0$.
These multipliers lie inside the disk for small positive $t$.
The parameter and ruler formulas give
$$
c-c_*=-\tfrac14t^2(1+i a)^2,\qquad
\frac{\rho(c)}{|c-c_*|}\longrightarrow\frac{2}{\sqrt{1+a^2}} .
$$
Thus along the unit direction $d_a=-(1+i a)^2/(1+a^2)$, the normalized
fourth power $Q(c)/|c-c_*|^4$ tends to $16/(1+a^2)^2$.
[Main-cardioid ruler](184-main-cardioid-ruler.md)

If $Q$ were $C^4$ through the cusp, its zero lower-order
derivatives would give a homogeneous quartic Taylor term $P$.
Along the ray with $a=0$, $P(-1)=16$. Along the family of rays
with $a\to+\infty$, $d_a\to 1$ and $P(d_a)\to0$, so continuity
forces $P(1)=0$. A homogeneous quartic satisfies $P(-1)=P(1)$.
The same obstruction holds for the metric coefficient
$f=(Q+\varepsilon^4)^{-1/2}$, since $Q=f^{-2}-\varepsilon^4$ is a
smooth function of a positive $f$. This argument uses only the
interior data.

## Checked scope

[ShoreFloorGluing.lean](../../formal/FractalSync/ShoreFloorGluing.lean)
checks the general $C^3$ gluing step in real normed spaces, given
distance-power bounds for a field and its three derivative fields and
the derivative identities off the set. It checks that the fields vanish
on the set and that the glued fields are the actual Fréchet derivatives.
The Map-specific uniformization, Cauchy estimates for the ruler, and
curvature identification use the conventional arguments above.
