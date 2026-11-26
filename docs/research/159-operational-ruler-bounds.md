# Explicit bounds for the actual operational ruler jets

Issue: #159 Phase A, with the concrete provider from #162 and the shared metric
from #157.

The regularized operational spline now supplies explicit global bounds for its
actual Euclidean gradient, its actual derivative Hessian, and the derivative of
that Hessian. These bounds depend only on a uniform bound for the literal
lattice coefficients, positive grid spacing, and positive regularization. They
do not require an assumed derivative-error budget or an assumed signed-distance
identity.

| Lean module | New theorem/lemma declarations | Scope |
| --- | ---: | --- |
| [RulerJetBounds.lean](../../formal/FractalSync/RulerJetBounds.lean) | 10 | Scalar regularizer derivatives, coefficient variation, and bilinear chain-rule norm bounds |
| [OperationalRulerBounds.lean](../../formal/FractalSync/OperationalRulerBounds.lean) | 23 | Euclidean transport of actual spline jets, actual ruler derivatives, explicit global bounds and variation |

These 33 declarations include supporting lemmas. They extend the existing
[actual operational dynamics](159-local-trajectories.md#7-local-mechanics-for-the-literal-operational-spline)
and the [coefficient-derived spline jet bounds](162-operational-reference-fidelity.md).

## 1. Exact field and hypotheses

Let \(C:\mathbb Z^2\to\mathbb R\) be the coefficient lattice used by the
literal tensor-product quintic field, \(o\in\mathbb R^2\) its origin, and
\(h>0\) its physical grid spacing. Assume

\[
B\ge0,\qquad |C_{ij}|\le B\quad\text{for every }i,j\in\mathbb Z.
\]

The field is the actual floor-selected function
`OperationalField.physicalField C o h`; its global C4 regularity through all
grid boundaries was already proved. On the Euclidean plane, write

\[
P(x)=P_{C,o,h}(\operatorname{chart}x),\qquad
\rho(x)=\sqrt{P(x)^2+\epsilon^2},\qquad \epsilon>0,
\]

\[
u(x)=\nabla\rho(x),\qquad H(x)=D(\nabla\rho)(x).
\]

Here \(\rho\) is exactly `OperationalDynamics.rulerField`, \(u\) is Mathlib's
actual `gradient`, and \(H\) is its actual `fderiv`. The latter two fields are
not separate input coefficients. The origin is arbitrary, and all query points
are allowed; none of the new estimates require two queries to share a cell.

The lattice premise is explicit data evidence still required of an application.
For a finite asset, one can bound the stored coefficients and choose a bounded
extension, but correspondence with its finite query is only available on the
valid support halo. The finite-domain results address that separate issue.
This note does not infer global validity of a finite provider from the global
existence of a lattice extension.

## 2. Constants and checked conclusions

Define the existing spline bounds

\[
G=\frac{8B}{3h},\qquad
K=\frac{68B}{9h^2},\qquad
T=\frac{32B}{h^3},
\]

and the regularized bounds

\[
K_\rho=K+\frac{G^2}{\epsilon},\qquad
J_\rho=T+\frac{3GK}{\epsilon}+\frac{3G^3}{\epsilon^2}.
\]

The Lean definitions are `splineGradientBound`, `splineHessianBound`,
`splineThirdBound`, `rulerHessianBound`, and `rulerThirdBound`, all in
`FractalSync.OperationalDynamics`. Expanded algebraically, the last two are

\[
K_\rho=\frac{68B}{9h^2}+\frac{64B^2}{9h^2\epsilon},
\qquad
J_\rho=\frac{32B}{h^3}+
\frac{544B^2}{9h^3\epsilon}+\frac{512B^3}{9h^3\epsilon^2}.
\]

For every Euclidean point \(x\), Lean proves

\[
|P(x)|\le B,\qquad
\boxed{\epsilon\le\rho(x)\le\sqrt{B^2+\epsilon^2}},
\]

\[
\boxed{\|u(x)\|_2\le G},\qquad
\boxed{\|H(x)\|_{2\to2}\le K_\rho},\qquad
\boxed{\|DH(x)\|\le J_\rho}.
\]

The norm in the last bound is the operator norm for a map from the Euclidean
plane into its continuous linear endomorphisms. The theorem
`actual_third_rulerField_bound` also states a `HasFDerivAt` assertion for that
specific derivative. The bound therefore concerns an existing derivative;
it does not rely on the convention that `fderiv` is zero where a function is
not differentiable.

For every pair of Euclidean points \(x,y\), the checked global variation bounds
are

\[
|\rho(y)-\rho(x)|\le G\|y-x\|_2,
\]

\[
\|u(y)-u(x)\|_2\le K_\rho\|y-x\|_2,
\qquad
\|H(y)-H(x)\|_{2\to2}\le J_\rho\|y-x\|_2.
\]

| Actual-field theorem | Conclusion |
| --- | --- |
| `rulerField_bounds` | Lower and upper ruler bounds |
| `norm_gradient_rulerField_le` | Uniform actual Riesz-gradient norm |
| `norm_hessian_rulerField_le` | Uniform actual derivative-Hessian norm |
| `rulerField_sub_le` | Global ruler variation |
| `gradient_rulerField_sub_le` | Global actual-gradient variation |
| `hessian_rulerField_sub_le` | Global actual-Hessian variation |
| `actual_third_rulerField_bound` | Actual Hessian differentiability and derivative norm |

All seven use the same coefficient, spacing, and regularization hypotheses.
The constants are conservative bounds, not optimal constants. In particular,
the spline is not assumed to be a distance function, and this result does not
assert \(\|u\|\le1\). Any downstream estimate using that stronger inequality
must establish it separately or use the checked bound \(G\).

## 3. Euclidean transport and the actual second derivative

The existing spline calculations take place in `Coords = ℝ × ℝ`, whose norm is
the maximum of the two absolute coordinates. The dynamics use
`Plane = WithLp 2 (ℝ × ℝ)`, with its Euclidean norm. The chart from `Plane` into
`Coords` is a continuous linear equivalence, and the new proof establishes

\[
\|\operatorname{chart}(v)\|_\infty\le\|v\|_2,
\qquad \|\operatorname{chart}\|\le1.
\]

Pulling a coordinate covector or bilinear form back along this contractive map
cannot increase its operator norm. This justifies using \(G,K,T\) as Euclidean
upper bounds without equating the max norm with the Euclidean norm.
Riesz dualization then preserves the actual Euclidean dual norm. In particular,
the Hessian estimate applies to an operator on the Euclidean plane, not merely
to a list of coordinate entries.

The code packages the transported physical differential and Hessian as
`euclideanDifferential` and `euclideanHessianForm` and proves that the latter
actually differentiates the former. With \(R=\sqrt{P^2+\epsilon^2}\), it then
identifies the derivative of the actual ruler differential with

\[
D^2\rho
=\frac{P}{R}\,D^2P+
\frac{\epsilon^2}{R^3}\,DP\otimes DP.
\]

`hasFDerivAt_fderiv_rulerField` proves that identity as a continuous bilinear
map. `hasFDerivAt_gradient_rulerField` transports it through the Riesz map to
the actual derivative of the gradient. The global Hessian-norm and
gradient-variation bounds use these derivative identifications directly.

## 4. Why the Hessian constant has these terms

For the scalar regularizer \(R(r)=\sqrt{r^2+\epsilon^2}\), the checked calculus
gives

\[
R'(r)=\frac r{R(r)},\qquad
R''(r)=\frac{\epsilon^2}{R(r)^3},\qquad
R'''(r)=-\frac{3\epsilon^2r}{R(r)^5},
\]

with global bounds

\[
|R'|\le1,\qquad 0\le R''\le\epsilon^{-1},\qquad
|R'''|\le3\epsilon^{-2}.
\]

The mean value theorem supplies corresponding variation bounds for \(R'\)
and \(R''\). The second-chain-rule expression has two terms. The variation
of \(R'(P)D^2P\) contributes \(T+GK/\epsilon\). The variation of
\(R''(P)DP\otimes DP\) contributes
\(2GK/\epsilon+3G^3/\epsilon^2\). Their sum is \(J_\rho\).

The underlying \(T\) is already derived from the literal quintic basis's
third derivatives and the global spline joins. It is not a new smoothness or
Hessian-Lipschitz assumption. The new proof transports that existing bound,
proves the regularized bilinear variation, and applies Riesz dualization.
Finally, the ruler's existing C4 regularity makes the Hessian genuinely C2;
the converse mean value estimate bounds its actual derivative by
\(J_\rho\).

## 5. Use in quantitative dynamics and verification boundary

These results supply the metric-coefficient part of a quantitative mechanical
certificate: a positive ruler floor, a finite ruler upper bound, uniform
gradient and Hessian norms, and explicit global Lipschitz constants for all
three fields. They remove the need to posit these constants independently
when the metric comes from the bounded operational lattice.

Force bounds and force variation remain separate inputs. A useful phase
region also needs a velocity bound, and a finite provider needs its support
halo. Converting those inputs to an explicit trajectory horizon and numerical
error contract is handled by the corresponding dynamics results.

The exact spline correspondence still uses the unchanged tensor sums and
spacing factors in
[`runtime-core/src/operational_field.rs`](../../runtime-core/src/operational_field.rs):
the physical gradient carries \(h^{-1}\), and its Hessian carries \(h^{-2}\).
This increment does not prove an IEEE rounding model for those calculations,
certify concrete asset values, prove a signed-distance identity, or establish
a force or integrator implementation contract. It proves bounds for the
literal provider's exact-real field and the actual regularized metric jets.

The targeted command `lake build FractalSync.OperationalRulerBounds --wfail`
passed with 2,635 Lake jobs and no warnings or errors; it includes the generic
helper transitively. The complete package build and unchanged transitive
axiom audit are recorded in [verification.md](verification.md) when this
increment is integrated. No runtime code or dependency pin changed.
