# Quartic floor for the main-cardioid ruler

The quartic floor gives zero Gaussian curvature at the Shore for each fixed positive floor, including at the main-cardioid cusp. Positive curvature still occurs inside the cardioid at distances comparable to the floor. This note uses the candidate $\rho$ from the [main-cardioid ruler study](184-main-cardioid-ruler.md) and the curvature convention from the [quadratic-floor calculation](184-conformal-floor-curvature.md).

## Curvature identity

For any positive $C^2$ planar ruler $\rho$, the metric $g=\rho^{-2}I$ has Gaussian curvature $K=\rho\Delta\rho-|\nabla\rho|^2$. Both derivatives use the Euclidean parameter coordinates. Fix $\varepsilon>0$ and set
$$
R=(\rho^4+\varepsilon^4)^{1/4},
\qquad
f(r)=\tfrac14\log(r^4+\varepsilon^4).
$$
The scalar derivatives are
$$
f'(r)=\frac{r^3}{r^4+\varepsilon^4},
\qquad
f''(r)=\frac{r^2(3\varepsilon^4-r^4)}{(r^4+\varepsilon^4)^2}.
$$
Taking the Euclidean trace of $D^2(f\circ\rho)$ and multiplying by $R^2$ gives the Gaussian curvature of $R^{-2}I$.
$$
\begin{aligned}
K_R
&=R^2\left(f'(\rho)\Delta\rho+f''(\rho)|\nabla\rho|^2\right)\\
&=\left(\frac{\rho}{R}\right)^2
\left[K+\frac{4\varepsilon^4}{\rho^4+\varepsilon^4}|\nabla\rho|^2\right].
\end{aligned}
$$

On the open main cardioid, $K=-1$ and $0\le |\nabla\rho|^2\le4$. Set $y=(\rho/R)^2\in(0,1)$. Then
$$
-y\le K_R=y[-1+4(1-y^2)|\nabla\rho|^2]\le15y-16y^3.
$$
In particular $|K_R|\le15y\le 15(\rho/\varepsilon)^2$. The ruler tends to zero at every Shore point, so the curvature has a continuous zero limit for fixed $\varepsilon$, on every interior approach to the cusp as well.

The global bounds over all positive floors are $-1\le K_R\le 5\sqrt5/2$. For the upper bound set $a=\sqrt5/4$ and use
$$
\frac{5\sqrt5}{2}-(15y-16y^3)=16(y-a)^2(y+2a).
$$
The lower bound is approached at a fixed interior point as $\varepsilon\to0$. The cusp-axis profile below approaches the upper bound.

## A fixed-floor metric extension

Extend $Q=\rho^4$ by zero on and outside the cardioid boundary. At each smooth arc the local multiplier inverse gives a smooth signed $\rho$ with positive sign inside. There $Q=(\max\{\rho,0\})^4$, which is $C^2$ across the arc.

At the cusp $c_*=1/4$, write $v=1-u$ and $s=|v|$ in the interior, so $c=c_*-v^2/4$ and $\rho=(2\operatorname{Re}v-s^2)s/4$. As $s\to0$ in the interior, this product gives
$$
\rho=O(s^2),\qquad D_v\rho=O(s),\qquad D_v^2\rho=O(1).
$$
Here $D_vs$ is bounded and $D_v^2s=O(s^{-1})$. The local inverse of $v\mapsto c_*-v^2/4$ has first and second derivative $O(s^{-1})$ and $O(s^{-3})$ respectively, by differentiating that relation. The chain rule gives $D_c\rho=O(1)$ and $D_c^2\rho=O(s^{-2})$. With $\delta=|c-c_*|=s^2/4$ this yields
$$
Q=O(\delta^4),\qquad D_cQ=O(\delta^3),\qquad D_c^2Q=O(\delta^2).
$$
The estimates also hold on the nearby smooth Shore and outside, where these quantities are zero. The difference quotient at $c_*$ for $Q$ tends to zero by the first estimate. The difference quotient for the resulting gradient tends to zero by the second estimate. The third estimate gives continuity of the Hessian at $c_*$. Thus $Q$ is $C^2$, and composing it with $z\mapsto(z+\varepsilon^4)^{-1/2}$ gives a positive $C^2$ conformal metric through the cusp.

## The sign change in floor units

Let $\lambda$ be the limiting value of $d_S/\varepsilon$ as $\varepsilon\to0$, with $d_S$ the Euclidean Shore distance. At a smooth Shore point, $\rho/d_S\to1$ and $|\nabla\rho|^2\to1$. Along the real cusp axis the two limits are $2$ and $4$. These limits substitute into the exact curvature identity. [Shore-distance ratios](184-main-cardioid-ruler.md#smooth-boundary-and-cusp), [gradient limits](184-conformal-floor-curvature.md#main-cardioid).

| Approach | Quartic limiting curvature | Positive for | Quadratic is positive for |
| --- | --- | --- | --- |
| Smooth Shore | $\dfrac{\lambda^2}{\sqrt{1+\lambda^4}}\left[-1+\dfrac{4}{1+\lambda^4}\right]$ | $0<\lambda<3^{1/4}$ | $0\le\lambda<1$ |
| Cusp axis | $\dfrac{4\lambda^2}{\sqrt{1+16\lambda^4}}\left[-1+\dfrac{16}{1+16\lambda^4}\right]$ | $0<\lambda<15^{1/4}/2$ | $0\le\lambda<\sqrt7/2$ |

The quartic smooth profile reaches $1$ at $\lambda=3^{-1/4}$, where $y=1/2$ maximizes $3y-4y^3$. The cusp-axis profile reaches $5\sqrt5/2$ at $\lambda=(5/11)^{1/4}/2$, where $y=\sqrt5/4$. The positive interval extends farther in floor units along smooth arcs and ends sooner along the cusp axis than with the quadratic floor.

For the same $\rho$ and $\varepsilon$, the quartic and quadratic rulers satisfy $2^{-1/4}\le R/R_2\le1$, where $R_2=(\rho^2+\varepsilon^2)^{1/2}$. The fourth power of this ratio is $(\rho^4+\varepsilon^4)/(\rho^2+\varepsilon^2)^2$. Its lower bound follows from $(\rho^2-\varepsilon^2)^2\ge0$ and is attained at $\rho=\varepsilon$. At a fixed conformal speed the same ratio applies to Euclidean speed because $|\dot c|=R\|\dot c\|_{R^{-2}I}$.

## Checked scope

[QuarticRulerFloor.lean](../../formal/FractalSync/QuarticRulerFloor.lean) checks the scalar derivatives, the second Fréchet derivative of the composite log floor, the weighted-trace identity, and the two scalar envelope bounds. The curvature identification, boundary limits, metric extension, and floor-scaled profiles use the conventional derivations above.
