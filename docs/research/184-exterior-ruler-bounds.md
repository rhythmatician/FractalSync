# Exterior conformal ruler and escape-tail bounds

On the finite exterior of the Map, the reciprocal hyperbolic density is $\rho=g/|\nabla g|$, where $g$ is the parameter Green function. An escaped critical orbit gives a finite approximation and explicit truncation bounds for both terms in this quotient. [Ruler research #182](https://github.com/rhythmatician/FractalSync/issues/182), [Lean scope #184](https://github.com/rhythmatician/FractalSync/issues/184).

## Exterior coordinate

Let $M$ be the Mandelbrot set and $\Omega=\mathbb C\setminus M$. Douady and Hubbard's parameter coordinate $\Phi(c)=\phi_c(c)$ maps $\Omega$ conformally onto $\{|w|>1\}$ and is tangent to the identity at infinity. Here $\phi_c$ is the dynamical Böttcher coordinate, normalized at infinity, and $\phi_c(c)$ is evaluated at the critical value. With $z_n=f_c^{\circ n}(0)$, the Green function $g(c)=\log|\Phi(c)|$ is the limit of $2^{1-n}\log|z_n|$. Starting the limit at the critical value accounts for the factor of two relative to the dynamical potential at $0$. [Douady and Hubbard, *Exploring the Mandelbrot Set*, Chapter 8, §1.1 and Theorem 8.1](https://pi.math.cornell.edu/~hubbard/OrsayEnglish.pdf#page=61).

In the curvature $-1$ normalization used by the [main-cardioid ruler](184-main-cardioid-ruler.md), the right half-plane has density $1/\operatorname{Re}t$. The covering $u=e^{-t}$ sends it onto the punctured disk $0<|u|<1$ with $|du/dt|=|u|$, giving density $1/(|u|\log(1/|u|))$. Pulling that density back through $u=1/\Phi(c)$ gives

$$
\lambda_\Omega(c)=\frac{1}{g(c)}\left|\frac{\Phi'(c)}{\Phi(c)}\right|,\qquad
\rho(c)=\frac{g(c)}{|\nabla g(c)|}.
$$

Locally $g$ is the real part of a holomorphic logarithm of $\Phi$. Hence $g_x-i g_y=\Phi'/\Phi$ and this complex logarithmic derivative is nonzero on $\Omega$. The puncture represents infinity in the finite parameter plane. [Beardon and Minda, *The Hyperbolic Metric and Geometric Function Theory*, §§7 and 12.1](https://www.math.stonybrook.edu/~bishop/classes/math401.F09/Beardon-Minda.pdf#page=39).

## Exact-input truncation bounds

Write $D_n=dz_n/dc$, with $z_1=c$ and $D_1=1$. At any finite stage with $z_n\ne0$, use

$$
z_{n+1}=z_n^2+c,\qquad D_{n+1}=2z_nD_n+1,\qquad
G_n=2^{1-n}\log|z_n|,\qquad L_n=2^{1-n}D_n/z_n.
$$

At consecutive stages with nonzero $z_k,z_{k+1}$, taking norms in the orbit recurrence and substituting the derivative recurrence give

$$
G_{k+1}-G_k=2^{-k}\log|1+c/z_k^2|,\qquad
L_{k+1}-L_k=\frac{2^{-k}-cL_k}{z_k^2+c}.
$$

Suppose $n\ge1$, $R=|z_n|\ge4$ and $q=|c|/R^2\le1/4$. The reverse triangle inequality gives $|z_{n+1}|\ge(3/4)|z_n|^2\ge3|z_n|$. Both guard inequalities persist, so $|z_{n+j}|\ge3^jR$ for $j\ge0$ and $c\in\Omega$. For $|a|\le1/4$, the modulus of $1+a$ lies in $[1-|a|,1+|a|]$. On this real interval the derivative of the logarithm is at most $4/3$, which gives $|\log|1+a||\le(4/3)|a|$. Thus

$$
|g-G_n|\le \frac43\,2^{-n}q\sum_{j=0}^\infty 18^{-j}
=\frac{24}{17}\,2^{-n}q=:E_G.
$$

For the derivative, set $a_j=(4/3)q\,9^{-j}$ and $b_j=(4/3)2^{-n}R^{-2}\,18^{-j}$. The recurrence and $|z_{k+1}|\ge (3/4)|z_k|^2$ give $|L_{n+j+1}-L_{n+j}|\le a_j|L_{n+j}|+b_j$. The coefficient sums are

$$
A=\sum_{j=0}^\infty a_j=\frac32q\le\frac38,\qquad
B=\sum_{j=0}^\infty b_j=\frac{24}{17}\frac{2^{-n}}{R^2}.
$$

Take $S_m=\max_{0\le j\le m}|L_{n+j}|$. Telescoping the recurrence to any index in this prefix gives $S_m\le |L_n|+A S_m+B$, hence $S_m\le (|L_n|+B)/(1-A)$. Inserting this uniform bound into the increment inequality gives absolute summability and

$$
|L-L_n|\le\frac{A|L_n|+B}{1-A}=:E_L,\qquad L=\lim_{k\to\infty}L_k.
$$

To identify $L$, take a small closed parameter disk in $\Omega$ about any given point and choose a common stage where the guard inequalities hold strictly throughout the disk. Such a stage exists after sufficiently many escapes at the center and by continuity of $z_n$. The geometric increment estimates then hold uniformly, with a positive lower bound for $R$, a uniform upper bound $q<1/4$, and bounded $L_n$ on the disk. The functions $G_k$ converge uniformly there to $g$ and their real gradients, represented as $(G_k)_x-i(G_k)_y=L_k$, converge uniformly to $L$. Integrating on horizontal and vertical segments identifies the limit as $g_x-i g_y=\Phi'/\Phi$.

## Ruler interval and Shore comparison

When $|L_n|>E_L$, the triangle inequality gives the enclosure

$$
\frac{\max(0,G_n-E_G)}{|L_n|+E_L}
\;\le\;\rho(c)\;\le\;
\frac{G_n+E_G}{|L_n|-E_L}.
$$

The finite estimate $G_n/|L_n|=|z_n|\log|z_n|/|D_n|$ applies when $D_n\ne0$. The Rust analytic distance-estimator branch uses twice this finite estimate. The interval is for the exterior conformal ruler. A certification of IEEE evaluation would also need to enclose represented inputs, the orbit and derivative recurrence, the escape guard, and all norm, logarithm, and division evaluations. [Rust distance estimator](../../runtime-core/src/distance_field.rs).

The common factor $2^{1-n}$ cancels from both interval endpoints.
An evaluator can use $\log R$ and $S=|D_n/z_n|$ with the rescaled radii
$(12/17)q$ and $(AS+12/(17R^2))/(1-A)$, respectively. This avoids
underflow from the iteration weight alone. The orbit and derivative
recurrences still need range and rounding bounds.

Already at $n=1$, large $|c|$ gives $R=|c|$, $q=1/|c|$, $G_1=\log|c|$ and $L_1=1/c$. The bounds give $g=\log|c|+O(|c|^{-1})$ and $L=1/c+O(|c|^{-2})$, so $\rho\sim |c|\log|c|$. The Shore is nonempty and bounded, so $d_S(c)=\operatorname{dist}(c,\partial M)=|c|+O(1)$. Therefore $\rho/d_S\sim\log|c|$. A uniform upper comparison by a constant times Shore distance fails on the whole exterior.

## Checked scope

The [Lean module](../../formal/FractalSync/ExteriorRulerBounds.lean) checks persistence of the escape guard, the two increment identities and their guarded bounds, the geometric series constants, the finite derivative envelope under cumulative coefficient bounds, and the positive-denominator ratio enclosure. The conformal coordinate and density, the identification of the limit with the parameter gradient, and the assembly of the infinite-tail and offshore bounds use the conventional arguments above.

## Interval examples

The table evaluates the enclosures at the first escape guard and two
iterates later. The midpoint and width come from the later interval.
The example uses the rescaled formula with 60-digit interval arithmetic in
`mpmath` 1.3.0. It
checks that each later interval is inside the first for these inputs.
The displayed values are rounded.

| $c$ | First $n$ | First interval | Midpoint at $n+2$ | Width at $n+2$ |
| ---: | ---: | ---: | ---: | ---: |
| $4$ | 1 | $[2.5708755,53.134008]$ | $6.7162649$ | $5.975\times10^{-4}$ |
| $1$ | 3 | $[0.56552412,0.68122704]$ | $0.64765352$ | $4.744\times10^{-6}$ |
| $-2.01$ | 6 | $[0.019762285,0.020521553]$ | $0.019883602$ | $9.731\times10^{-11}$ |
| $0.5+0.5i$ | 6 | $[0.095140557,0.096533199]$ | $0.096156915$ | $2.159\times10^{-10}$ |
| $0.2501$ | 313 | $[8.976858,9.2249577]\times10^{-7}$ | $9.1847896\times10^{-7}$ | $4.703\times10^{-13}$ |

The first valid escape guard can give a wide interval. Extra iterates
quickly reduce its width in these examples.
These examples exercise the exact-real error formula with a separate
interval library. A Rust binary64 implementation would need enclosures
for its own orbit and derivative arithmetic.

~~~python
from mpmath import iv, mp

iv.dps = mp.dps = 60


def advance(c, z, derivative):
    return z * z + c, 2 * z * derivative + 1


def enclosure(c, z, derivative):
    radius = abs(z)
    log_radius = iv.ln(radius)
    slope = abs(derivative / z)
    a = 3 * abs(c) / (2 * radius**2)
    b = 12 / (17 * radius**2)
    green_error = abs(c) * b
    slope_error = (a * slope + b) / (1 - a)
    assert slope > slope_error and log_radius > green_error
    lower = (log_radius - green_error) / (slope + slope_error)
    upper = (log_radius + green_error) / (slope - slope_error)
    return mp.mpf(lower.a), mp.mpf(upper.b)


for c in (iv.mpc(4), iv.mpc(1), iv.mpc("-2.01"),
          iv.mpc(".5", ".5"), iv.mpc(".2501")):
    z, derivative = c, iv.mpc(1)
    for n in range(1, 500):
        radius = abs(z)
        if radius >= 4 and 4 * abs(c) <= radius**2:
            break
        z, derivative = advance(c, z, derivative)
    else:
        raise ArithmeticError("Escape guard unresolved")

    lo, hi = enclosure(c, z, derivative)
    for _ in range(2):
        z, derivative = advance(c, z, derivative)
    lo2, hi2 = enclosure(c, z, derivative)
    assert lo <= lo2 <= hi2 <= hi
    print(n, mp.nstr(lo, 8), mp.nstr(hi, 8),
          mp.nstr((lo2 + hi2) / 2, 8), mp.nstr(hi2 - lo2, 4))
~~~
