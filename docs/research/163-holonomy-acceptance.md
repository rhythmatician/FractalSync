# Finite transport kernels and closed loops

This note completes the holonomy examples and finite-kernel argument in the [phase-harmonic study for #163](163-covariant-phase-harmonics.md).

## Exact finite criterion

Let $K$ be a complex matrix on a nonempty finite set of samples, with $K_{ii}=1$, $|K_{ij}|=1$, and $K_{ji}=\overline{K_{ij}}$. For samples $a,b,c$, let $h=K_{ab}K_{bc}K_{ca}$. Expanding the three-by-three determinant gives

$$
\det K_{(a,b,c)} = h+\bar h-2=-|h-1|^2.
$$

Positive semidefiniteness forces every principal determinant to be nonnegative, so every triangle has $h=1$. Conversely, fix a reference sample $a$ and let $q_i=K_{ai}$. Trivial triangle products and unit entries give $K_{ij}=\overline{q_i}q_j$. This is a positive semidefinite outer product of rank one. It telescopes around any cycle. The criterion applies to complete kernels under the stated unit-entry hypotheses.

For pairwise transports $U_{ij}$ in oriented dimension two, use $K_{ij}=U_{ij}^m$ at any integer harmonic $m$. The triangle obstruction becomes $(U_{ab}U_{bc}U_{ca})^m \ne 1$. A half-turn gives determinant $-4$ for odd $m$ and zero for even $m$, including zero and negative orders. The [original note's half-turn matrix](163-covariant-phase-harmonics.md#4-a-pairwise-formula-and-its-precise-obstruction) also supplies a concrete negative squared-coherence witness.

[CovariantHolonomy.lean](../../formal/FractalSync/CovariantHolonomy.lean) checks the triangle determinant, the PSD equivalence, factor recovery from any reference row, rank one, and the integer-harmonic specialization. Its inputs are unit complex transports. The circle connection calculations below use conventional differential geometry.

## Signed examples on three surfaces

Use polar coordinates with oriented metric $dr^2+f(r)^2d\phi^2$, at $r>0$ with $f(r)>0$. The frame is $e_r=\partial_r$, $e_\phi=f^{-1}\partial_\phi$, in that order. The connection coefficient $\Gamma^\phi_{\phi r}=f'/f$ gives $\nabla_{\partial_\phi}e_r=f'e_\phi$ and metric compatibility gives $\nabla_{\partial_\phi}e_\phi=-f'e_r$. Thus along the increasing circle $\phi=t$ for $0\le t\le2\pi$, the covariant derivative of a complex coordinate $b$ is $\dot b+i f'(r)b$. The velocity coordinates are the constant $i f(r)$ in the periodic polar frame. Transport to time zero multiplies them by $\exp(i f'(r)t)$, so the integrated intrinsic turning is $2\pi f'(r)$. Forward holonomy has lifted angle $-2\pi f'(r)$, represented modulo $2\pi$ by $2\pi(1-f'(r))$.

| Surface | $f(r)$ | Radius used | Intrinsic turning | Forward holonomy modulo $2\pi$ |
| --- | --- | --- | --- | --- |
| Euclidean plane | $r$ | $1$ | $2\pi$ | $0$ |
| Unit sphere | $\sin r$ | $\pi/3$ | $\pi$ | $\pi$ |
| Hyperbolic plane of curvature $-1$ | $\sinh r$ | $\operatorname{arcosh}(3/2)$ | $3\pi$ | $-\pi$ |

These values can be reproduced by substituting $f'(r)=1,\cos r,\cosh r$ in the same connection formula. The changing signs also follow by integrating Gaussian curvature $-f''/f$ over the positively oriented disk with area element $f(r)\,dr\,d\phi$ and $f'(0)=1$. A regular closed curve can have nonzero intrinsic turning even when its holonomy vanishes, as the flat circle shows.

Cut either curved circle at $t=0$ to work with the time interval $[0,2\pi]$. The common-reference factors $p(t)=\exp(i f'(r)t)$ continue through the final sample. On the time interval, the kernel $\overline{p(s)^m}p(t)^m$ remains positive semidefinite and the transported phase advances by $2\pi m f'(r)$. If the endpoints are identified and the pairwise paths include the wraparound arc, take three samples at $0,2\pi/3,4\pi/3$ and connect successive samples around the circle. The product of forward transports around those three arcs equals the loop holonomy. At odd $m$ it is $-1$ and the corresponding Hermitian kernel has determinant $-4$. Reversing triangle orientation conjugates the product and leaves the determinant unchanged.

## Scope and originality

The original note gives the complex gauge law, the common-reference statistic, normalization and perturbation bounds, the exact epoch-pooling range, and the signed-score requirement. Its numerical examples cover support loss and incompatible pairwise transports. The circle calculations above give reproducible analytic checks for flat and curved closed loops. For live use, the declared lag and mask must use only samples available at the evaluation time. Unknown phase across a reset or unavailable observation remains missing data. The gesture-stream contract owns runtime validity, including rest and impulses. [#124](https://github.com/rhythmatician/FractalSync/issues/124).

Mallat, Zhang, and Rochette give the exact phase-harmonic Lipschitz bound in Lemma 2.1. Singer treats recovery of a common phase reference from pairwise offsets, including inconsistent measurements. Singer and Wu build a connection-Laplacian diffusion statistic. Shao, Lin, and Yao define frame-independent covariance for manifold-valued functional data using a covariance bundle and parallel transport. These constructions clarify the distinction between a transported scalar harmonic moment, phase recovery, and an intrinsic covariance operator. [Mallat, Zhang, and Rochette](https://arxiv.org/abs/1810.12136), [Singer](https://arxiv.org/abs/0905.3174), [Singer and Wu](https://arxiv.org/abs/1102.0075), [Shao, Lin, and Yao](https://arxiv.org/abs/2009.07427).

Kepler and Kagan establish geometric phase shifts for limit-cycle oscillators under adiabatic parameter variation. Here the phase shift is ordinary tangent-bundle parallel transport. It is present in the synthetic circles without any audio input, so the shift alone supplies no evidence of audiovisual synchronization. [Kepler and Kagan](https://doi.org/10.1103/PhysRevLett.66.847).

The finite criterion, bounds, and examples provide reusable contracts for FractalSync. The comparison found no independent novelty claim. Along one specified path the covariant formula exactly reproduces the transported-frame statistic with the same samples, weights, and lag. The empirical question of fit belongs to [#126](https://github.com/rhythmatician/FractalSync/issues/126).
