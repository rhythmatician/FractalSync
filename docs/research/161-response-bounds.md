# Residual curvature and finite-horizon sensitivity

The [sensitivity study](161-sensitivity-preconditioning.md) gives an
analytic location-scale control family, an anisotropic fold, and a
derivative-error transfer bound. The statements here complete the
least-squares and control-response comparisons in
[#161](https://github.com/rhythmatician/FractalSync/issues/161).

## Curvature away from an exact fit

Let $r(t)$ be the residual along a smooth real parameter path with
first and second derivatives $j(t)$ and $b(t)$ in a real Hilbert
space. At the query the loss $\ell(t)=\|r(t)\|^2/2$ satisfies
$$
\ell''=\|j\|^2+\langle r,b\rangle.
$$
For a state-space geodesic with initial velocity $v$, this is the
intrinsic Hessian on $v$. For a straight coordinate path it is the
coordinate Hessian. At an exact fit the residual vanishes, and the
two yield the same quadratic form.

Write $s=\|v\|_G$. If $l s\le\|j\|\le u s$ with $l\ge0$, and
$\|b\|\le m s^2$ along the chosen path at the query, then
$$
(l^2-m\|r\|)s^2\ \le\ \ell''\ \le\ (u^2+m\|r\|)s^2.
$$
The margin for positive curvature is $l^2>m\|r\|$. The residual
term matters even for an identifiable observable. For example
$r(t)=t+bt^2/2-a$ at $t=0$ has $\|j\|=1$ and
$\ell''(0)=1-ab$. The curvature is negative when $ab>1$.
The Hessian decomposition is the standard least-squares identity.
[Nocedal and Wright, *Numerical Optimization*, §10.1](https://users.iems.northwestern.edu/~nocedal/book/toc.html)

## Conditioning through Controls

Along a fixed finite-horizon trajectory, let $B$ be the actual
linearized response from a control variation $u_c$ to the stacked
state displacement, and let $V'$ be the stacked observable derivative.
Suppose
$$
\beta_-\|u_c\|_H\le \|Bu_c\|_G\le\beta_+\|u_c\|_H,\qquad
\lambda_-\|w\|_G^2\le\|V'w\|^2\le\lambda_+\|w\|_G^2
$$
on the evaluated action subspace and its reachable state directions.
With nonnegative margins the composed squared response is between
$\lambda_-\beta_-^2\|u_c\|_H^2$ and
$\lambda_+\beta_+^2\|u_c\|_H^2$. At an exact fit this bounds the
action Hessian. When the lower constants are positive, the condition
number relative to $H$ is at most
$(\lambda_+/\lambda_-)(\beta_+/\beta_-)^2$.
For example, a nonzero action in $\ker B$ has zero response even for
a perfectly conditioned observable. Restricting the space of actions
or changing the Controls is needed to obtain a positive lower margin.
[Linearized reachability background, Sontag, *Mathematical Control Theory*, chapter 3](https://www.sontaglab.org/FTPDIR/sontag_mathematical_control_theory_springer98.pdf)

## Verification and decision

Lean checks the actual second derivative of the residual loss along
a real path with the given derivatives, the residual correction bound,
and the composition bound on reachable action directions. The
directional hypotheses require evaluation for the renderer and Controls.
[SensitivityResponse.lean](../../formal/FractalSync/SensitivityResponse.lean)

The analytic template, fold and branched periodic-point examples
support a reusable engineering comparison. The affine-group and
least-squares structure is classical, and the derived bounds give
no separate publication claim. The Julia sensitivity measurements
and finite-horizon Controls estimates remain in
[#150](https://github.com/rhythmatician/FractalSync/issues/150) and
[#151](https://github.com/rhythmatician/FractalSync/issues/151).
