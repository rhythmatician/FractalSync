# Validity of finite orbit-trap selection

[Issue #185](https://github.com/rhythmatician/FractalSync/issues/185) considers selecting a closest orbit iterate for a configurable trap. Let $P_n(c)=f_c^{\,n}(0)$ with $f_c(z)=z^2+c$. For a fixed finite candidate set $S$, the score of index $j$ is $F_j(c)=h(P_j(c))$, with lower scores closer to the trap. A smooth squared distance to a fixed point is one example of $h$. Certifying the closest index and following its contour are separate problems.

## A gap certificate

At one parameter point, let $\widehat F_j$ be the measured score with certified absolute error $e_j\ge0$. For the proposed winner $i\in S$ and each other $j\in S$, set

$$
\delta_j=\widehat F_j-\widehat F_i-e_i-e_j.
$$

Subtract the upper endpoint of the interval for $F_i$ from the lower endpoint for $F_j$. The true gap $F_j-F_i$ is at least $\delta_j$. The condition $\delta_j>0$ for every competitor gives the same unique measured and true winner. If there is at least one competitor, the minimum $\delta=\min_{j\ne i}\delta_j$ is a certified lower bound on the true gap to the runner-up. A singleton needs no score-gap comparison. An empty candidate set has no winner.

The strict inequality matters. Take equal true scores $B$ and measured scores $B-e_i$ and $B+e_j$. The measured difference is exactly $e_i+e_j$ and the true gap is zero. Taking $B$ large enough makes even the measured scores nonnegative. A deterministic tie breaker can stabilize a display label while leaving the differentiable contour uncertified.

For a fixed finite $S$, continuity of every true $F_j$ preserves any strict gap margin $\gamma$ in a neighborhood of the query point. For the certificate above, choose $0\le\gamma<\delta$. The finite minimum equals $F_i$ throughout that neighborhood. In particular, it inherits all local differentiability of $F_i$ there. If the score functions have local Lipschitz constants $L_i,L_j$, the ordinary triangle inequality gives the more explicit lower bound $\delta_j-(L_i+L_j)|c'-c|$. A positive bound for every competitor certifies the same winner throughout the corresponding ball.

The explicit Lipschitz radius is a conventional consequence of the gap bound. The error assumptions concern the score values. Certifying gradient direction requires a separate derivative budget. For a fixed regular branch and conformal ruler $\rho>0$, the unit contour tangent is $\rho J\nabla F_i/|\nabla F_i|$. Use the [fixed-trap analysis](185-orbit-trap-steering.md) for the checked tangent identity and the [gradient confidence margin](185-orbit-trap-steering.md#tangency-and-critical-points) for the directional guard.

## Ties and soft scores

Two regular scores can be $F_1(x)=a+x$ and $F_2(x)=a-x$ near zero, with $a>0$. The lower envelope is $a-|x|$. It switches from derivative $+1$ to $-1$ at zero. Actual nonnegative squared scores can behave the same way. The pair $(x-1)^2,(x+1)^2$ gives $1+x^2-2|x|$. Both gradients are nonzero and opposite at the tie. A chosen index resolves which gradient to report, but the minimum has a kink there.

For constant $\tau>0$ the soft minimum $-\tau\log\sum_{j\in S}\exp(-F_j/\tau)$ has gradient $\sum_j w_j\nabla F_j$ where $w_j=\exp(-F_j/\tau)/\sum_k\exp(-F_k/\tau)$. This is the chain rule for a fixed nonempty set of differentiable scores. In the two-branch tie above the weights are each $1/2$, and the gradient cancels. Smoothing the value can remove a kink while producing a critical point. It also changes which contours are followed, and its output depends on candidate membership. The smooth-min formula and squared-score example are conventional results. The affine tie and its failure of differentiability are checked.

## Escape, the index budget, and period

Apply the certificate to a declared set of eligible indices and a fixed score definition. Reject missing, overflowed, or unsupported entries rather than silently changing the set along a certified arc. A valid prefix ending at escape or at the maximum index can support a minimum **over that prefix**. If the compared set changes, the old gap certificate expires. A claim about excluded indices needs a separate tail bound. For a trap at the origin, for instance, a certified magnitude beyond $R\ge1$ with $R^2-|c|\ge R$ forces all later magnitudes to grow by the reverse triangle inequality. This can exclude the tail when its lower score bound lies above the winning interval. A configurable trap score has no such tail bound merely from escape.

A return $P_n(c)=0$ with $n>0$ is exactly a periodic-point equation for $0$ under $f_c$. Its positive least period divides $n$. For instance, at $c=0$ the second critical iterate is zero but the least period is one. A hit on a general trap need not be a return to $0$. An approximate near-return needs a numerical bound and periodicity evidence before it can certify a center. The index counts compositions of $f_c$. For trajectories and for recurrence confidence, compare it under a stable eligibility and measurement rule.

The certificate controls index stability on the specified scores. The selected branch still needs a nonzero, accurately evaluated gradient, a valid metric and a trajectory segment on which all assumptions persist. The [navigation comparison](185-navigation-field-comparison.md) uses the fixed second iterate. It measures no configurable-trap benefit or success through Controls and the official renderer.

## Checked statements

The [Lean module](../../formal/FractalSync/OrbitTrapSelection.lean) uses the namespace *FractalSync.OrbitTrapSelection*.

| Result | Declaration |
| --- | --- |
| Lower bound on the true pairwise gap, endpoint example at equality, strict measured-to-true winner | recoverTrueGap, recoverTrueGap_sharp, certifiedStrictMinimum |
| Local preservation of any strict margin for fixed finite continuous scores, local equality of the envelope and limiting branch, inheritance of branch regularity | eventually_preservesGap, lowerEnvelope_eventuallyEq, certifiedLowerEnvelope_contDiffAt |
| Explicit affine tie envelope and its nondifferentiability | tieEnvelope_eq, tieEnvelope_not_differentiable |
| Exact positive return implies positive least period dividing the return index | criticalReturn_period |
