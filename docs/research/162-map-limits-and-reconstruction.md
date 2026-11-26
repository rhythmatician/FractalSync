# Recovering sets and affine fields from normalized distance data

The [normalized distance-field study](162-normalized-distance-fields.md)
gives the chart transition, scale and center bounds, provider error
propagation, and examples of nonunique blow-ups. The results below close
two remaining questions in
[#162](https://github.com/rhythmatician/FractalSync/issues/162).

## A pointwise unsigned distance limit still represents a set

Let $S_n$ be closed subsets of a proper metric space, each containing
a fixed anchor $a$. Suppose $\operatorname{dist}(x,S_n)\to f(x)$ for
every $x$. Then $f=\operatorname{dist}(\cdot,\{f=0\})$.
For each $x$, take a nearest point $p_n\in S_n$. It lies in the
closed ball about $a$ of radius $2d(x,a)$. Along a convergent
subsequence $p_{n_j}\to p$, pointwise convergence at $x$ gives
$f(x)=d(x,p)$. Pointwise convergence at $p$ and
$d(p,S_{n_j})\le d(p,p_{n_j})$ give $f(p)=0$.
Passing the distance-function Lipschitz bound to the limit shows
$f(x)\le d(x,z)$ for every zero $z$ of $f$. Together these
give equality. Lean checks the nearest-point and limit argument.
[NormalizedMapLimits.lean](../../formal/FractalSync/NormalizedMapLimits.lean)

At a center on the Shore, the unsigned normalized fields are the
distance functions of translated and dilated sets that all contain
zero. Arzelà–Ascoli on finite-dimensional balls supplies locally
uniform subsequences, and the checked theorem identifies each limit's
set. The equi-Lipschitz bound also upgrades pointwise convergence to
uniform convergence on each compact ball by a finite net argument.
Signed fields need side information, and the allowed blow-up sets
can be nonunique and nonconical. Those conventional arguments and
examples are in the [blow-up analysis](162-normalized-distance-fields.md#6-blow-ups-useful-limits-with-two-essential-qualifications).
Uniform convergence of distance functions on bounded sets is the
Attouch–Wets topology. [Beer, *Uniform continuity on bounded sets and the Attouch–Wets topology*](https://doi.org/10.1090/S0002-9939-1991-1033956-1)

## Finite reconstruction on a smooth window

Let $M(0)=0$, $\nabla M(0)=g$, and suppose the segment from $0$
to every query and stencil point satisfies a Hessian norm bound $L$.
Use $2n$ measurements $\hat M(\pm s e_i)$ with component error
at most $e_i$ and $s>0$. Then $ \hat g_i=(\hat M(s e_i)-\hat M(-s e_i))/(2s)$
satisfies
$$
|\hat g_i-g_i|\le Ls/2+e_i/s .
$$
The reconstructed affine field at $u$ has error
$$
\left|M(u)-\sum_i\hat g_i u_i\right|
\le \tfrac12 L\|u\|^2+\sum_i (Ls/2+e_i/s)|u_i|.
$$
The bound follows from the quadratic Taylor remainder at the query
and on the axes, followed by the triangle inequality. An exactly
affine field with exact samples is recovered at every query. In two
dimensions, $M(u)=g\cdot u+\tfrac12 L\,u_1^2$ gives a simple smooth
case. The symmetric slopes recover $g$ exactly and the affine
error at $u$ is $\tfrac12 L u_1^2$. A crease or competing nearest
branch can invalidate the local Hessian premise.

Lean checks the symmetric slope bound and the finite reconstruction
inequality. The actual Taylor remainder is available in the
[local field proof](159-differential-mechanics.md). The stencil error
can include a separately certified query-scale and arithmetic error.
The reconstruction is an estimate of field values on the supported
window. The [connectivity counterexample](162-normalized-distance-fields.md#exact-samples-do-not-certify-connectivity)
still applies outside the smooth class.

## Decision

Keep the chart, limit, and reconstruction results as an engineering
theorem package. The approximation uses Taylor and distance-function
arguments with established prior art. A provider still must supply the
derivative, query, and numerical evidence used in the bounds. Map
channel selection and empirical validation remain with
[#108](https://github.com/rhythmatician/FractalSync/issues/108) and
[#167](https://github.com/rhythmatician/FractalSync/issues/167).
