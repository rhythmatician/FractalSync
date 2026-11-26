# Explicit mechanical bounds from finite operational assets

The actual regularized operational metric now supplies an explicit acceleration
norm bound and an explicit acceleration Lipschitz constant on a specified phase
ball. The metric inputs come from literal spline coefficients and actual
Euclidean derivatives. A bounded Lipschitz certificate for the supplied force
remains an input. Finite stored coefficient bounds suffice when the phase ball
lies strictly inside the physical query halo.

| Lean module | New theorem/lemma declarations | Scope |
| --- | ---: | --- |
| [BoundedLipschitz.lean](../../formal/FractalSync/BoundedLipschitz.lean) | 14 | Uniform norm and variation certificates, propagated through actual operations |
| [MechanicalBounds.lean](../../formal/FractalSync/MechanicalBounds.lean) | 1 | Explicit pair of bounds for the existing inverse-metric force and connection expression |
| [OperationalMechanicalBounds.lean](../../formal/FractalSync/OperationalMechanicalBounds.lean) | 7 | Actual operational metric specialization, phase-ball velocity bounds, and finite-asset transfer |

These 22 declarations include supporting lemmas. The metric constants are
proved separately in [operational ruler bounds](159-operational-ruler-bounds.md).
The resulting acceleration certificate supplies the hypotheses of the
[quantitative trajectory construction](159-quantitative-dynamics.md).

## 1. The mechanical expression being bounded

For the actual Euclidean ruler \(\rho(x)\), put

\[
u(x)=\nabla\rho(x),\qquad H(x)=D(\nabla\rho)(x),\qquad
W(x)=1+a^2\|u(x)\|_2^2.
\]

The parameter \(a\) is real; all estimates depend on it through \(a^2\).
For a phase state \(z=(x,v)\) and supplied force \(Q(z)\), the existing
`OperationalDynamics.acceleration` is exactly

\[
\mathcal A(z)=
\rho(x)^2\left(Q(z)-\frac{a^2\langle u(x),Q(z)\rangle}{W(x)}u(x)\right)
-\Gamma_x(v,v),
\]

where

\[
\Gamma_x(v,v)=
\frac{a^2\langle H(x)v,v\rangle}{W(x)}u(x)
-\frac{2\langle u(x),v\rangle}{\rho(x)}v
+\frac{\|v\|_2^2+a^2\langle u(x),v\rangle^2}{\rho(x)W(x)}u(x).
\]

The ambient Euclidean inner product identifies the force covector with the
vector appearing in this formula. The new proof bounds this existing
expression by unfolding `inverseAction` and `connection`; it does not introduce
an alternative mechanical field.

The generic `MechanicalBounds.acceleration_bound` accepts arbitrary coefficient
fields \(\rho,u,H\), together with their norm and variation certificates. It
does not infer derivative identities for arbitrary inputs. The operational
specialization supplies exactly `gradient (rulerField ...)` and its actual
`fderiv`, so those identities are already part of the concrete field being
bounded.

## 2. Uniform norm and Lipschitz pairs

`BoundedLipschitzOn f s M L` asserts

\[
M\ge0,\quad L\ge0,\quad
\|f(z)\|\le M\quad(z\in s),
\]

\[
\|f(z)-f(w)\|\le L\,d(z,w)\quad(z,w\in s).
\]

The same region is used for both bounds. For vector fields the norm is the
ambient norm; for operator fields it is the continuous-linear-map operator
norm. No differentiability or convexity premise is required to propagate an
already supplied pair.

Write a pair as \(p=(M,L)\), and define

\[
p\oplus q=(M+N,L+K),
\qquad
p\otimes q=(MN,MK+NL),
\qquad q=(N,K).
\]

Both addition and subtraction use \(\oplus\). Scalar multiplication, scalar
products, inner products, and application of an operator to a vector use
\(\otimes\). The proof is the actual bilinear difference identity together
with the relevant norm inequality.

For a real field with a positive floor \(f(z)\ge\delta>0\), reciprocal
propagation is

\[
\operatorname{Inv}_\delta(M,L)
=\left(\frac1\delta,\frac{L}{\delta^2}\right).
\]

The reciprocal operation requires its floor evidence; a pair alone does not
provide it. In mechanics, the checked floors are \(\rho\ge\epsilon>0\)
and \(W\ge1\). The latter follows from \(a^2\|u\|^2\ge0\), without a
unit-gradient hypothesis.

## 3. A direct acceleration norm bound

Suppose the region has bounds

\[
\epsilon\le\rho\le\overline\rho,\qquad
\|u\|_2\le U,\qquad \|H\|\le H_0,
\qquad \|Q\|_2\le Q_0,\qquad \|v\|_2\le V.
\]

Simplifying the first entry of `accelerationBudget` gives

\[
\boxed{
A_0=
\overline\rho^{\,2}Q_0(1+a^2U^2)
+a^2H_0UV^2
+\frac{U(3+a^2U^2)V^2}{\epsilon}.
}
\]

The actual conclusion is \(\|\mathcal A(z)\|_2\le A_0\) on the entire
specified region. The three contributions come from the inverse-metric force,
the Hessian term in the connection, and the remaining connection terms.

This is a conservative estimate. For example, the pair propagation bounds the
two terms inside the inverse-metric action by their norms separately, and it
uses \(W^{-1}\le1\). It does not claim an optimal operator or acceleration
bound. Its purpose is to give a finite, explicit constant that can be inserted
into the checked local trajectory theorem.

## 4. The explicit Lipschitz arithmetic recipe

Let the five supplied pairs be

\[
p_\rho=(\overline\rho,L_\rho),\quad
p_u=(U,L_u),\quad p_H=(H_0,L_H),\quad
p_Q=(Q_0,L_Q),\quad p_v=(V,L_v).
\]

The following finite arithmetic recipe is the definition of the bound pair.
All operations refer to the pair rules in section 2:

\[
p_a=(a^2,0),\qquad
p_W=(1,0)\oplus(p_a\otimes p_u\otimes p_u),
\]

\[
p_{W^{-1}}=\operatorname{Inv}_1(p_W),\qquad
p_{\rho^{-1}}=\operatorname{Inv}_\epsilon(p_\rho),
\]

\[
p_{uv}=p_u\otimes p_v,\qquad
p_{Hvv}=p_H\otimes p_v\otimes p_v.
\]

For clarity, \(p_{W^{-1}}=(1,2a^2UL_u)\) and
\(p_{\rho^{-1}}=(\epsilon^{-1},L_\rho/\epsilon^2)\).
The inverse-metric force pair is

\[
p_{\rm raised}=
p_\rho\otimes p_\rho\otimes
\left[p_Q\oplus
\left(p_a\otimes p_u\otimes p_Q\otimes p_{W^{-1}}\otimes p_u\right)\right].
\]

The connection pairs are

\[
p_1=p_a\otimes p_{Hvv}\otimes p_{W^{-1}}\otimes p_u,
\]

\[
p_2=p_{uv}\otimes p_{\rho^{-1}}\otimes p_v,
\]

\[
p_3=\left[(p_v\otimes p_v)\oplus
(p_a\otimes p_{uv}\otimes p_{uv})\right]
\otimes p_{\rho^{-1}}\otimes p_{W^{-1}}\otimes p_u.
\]

The final pair is

\[
\boxed{(A_0,L_0)=p_{\rm raised}\oplus p_1\oplus p_2\oplus p_2\oplus p_3.}
\]

The repeated \(p_2\) accounts for the two equal mixed terms in
\(\Gamma(v,v)\). Minus signs are handled by the subtraction bound. Pair
multiplication is associative, so the grouping above gives the same arithmetic
as the nested Lean definitions `raisedForceBudget`, `connectionBudget`, and
`accelerationBudget`.

`MechanicalBounds.acceleration_bound` proves both

\[
\|\mathcal A(z)\|_2\le A_0,
\qquad
\|\mathcal A(z)-\mathcal A(w)\|_2\le L_0\,d(z,w)
\]

on the supplied region. In particular, \(L_0\) is an explicit expression in
the supplied numbers, not a constant obtained by an existence argument.

## 5. Substitution of the actual operational metric

Let \(B\ge0\) bound the absolute lattice coefficients, with physical spacing
\(h>0\) and regularization \(\epsilon>0\). Retain

\[
G=\frac{8B}{3h},\qquad
K_\rho=\frac{68B}{9h^2}+\frac{G^2}{\epsilon},
\]

\[
J_\rho=\frac{32B}{h^3}+
\frac{3G}{\epsilon}\frac{68B}{9h^2}+
\frac{3G^3}{\epsilon^2}.
\]

The actual ruler results supply

\[
p_\rho=(\sqrt{B^2+\epsilon^2},G),\qquad
p_u=(G,K_\rho),\qquad p_H=(K_\rho,J_\rho).
\]

For the phase ball centered at \(p=(x_0,v_0)\), the phase norm is

\[
\|(x,v)\|=\max(\|x\|_2,\|v\|_2).
\]

The position and velocity projections are therefore one-Lipschitz. Composing
the Euclidean coefficient fields with the position projection preserves the
three pairs above. Throughout \(\overline B(p,R)\), with \(R\ge0\),

\[
\|v\|_2\le V:=\|v_0\|_2+R,
\qquad p_v=(V,1)
\]

The code supplies this velocity bound pair directly in `operationalAccelerationBudget`, avoiding an independent
velocity-bound premise.

Given a force certificate

\[
\|Q(z)\|_2\le Q_0,\qquad
\|Q(z)-Q(w)\|_2\le L_Q\|z-w\|
\quad(z,w\in\overline B(p,R)),
\]

`acceleration_boundedLipschitz` returns the full acceleration certificate with
the exact pair produced by `operationalAccelerationBudget`. The force may
depend on both position and velocity. No C1 force premise is needed for this
certificate; its actual bounded Lipschitz premise is the input used by the
quantitative ODE theorem. A force implementation must still supply evidence
for that premise.

## 6. Finite stored data and extension independence

For an asset of dimensions \(N_x,N_y\), suppose \(|C_{ij}|\le B\) only on
the stored integer rectangle. No bound on the original lattice outside that
rectangle is required. Also assume the strict physical margin condition

\[
R<m(\operatorname{chart}(x_0)),
\]

where \(m\) is the four-sided `haloMargin` from
[the finite-asset halo results](162-operational-halo.md).

The proof forms the zero extension, whose entire lattice is bounded by \(B\).
It derives the global coefficient bounds and the phase-ball acceleration
certificate for that extension. Every configuration in the closed phase ball
lies in the open halo, where agreeing extensions have equal ruler germs,
actual gradients, actual derivative Hessians, and actual acceleration fields
for the same force. `acceleration_zeroExtension_eqOn` records the equality on
the full closed phase ball.

`finiteAsset_acceleration_boundedLipschitz` transfers both the norm and pairwise
Lipschitz inequalities back to the original field. Thus its output is a
certificate for the original actual acceleration, with constants determined
by finite stored coefficient evidence. It does not impose boundedness on
arbitrary off-asset values or differentiate across a halo boundary.

The ball certificate allows \(R=0\). A positive trajectory horizon additionally
requires \(R>0\), as stated in the quantitative dynamics theorem. A reported
coordinate query with displacement budget \(\delta\) needs the stronger
domain condition \(R+\delta<m(\operatorname{chart}(x_0))\). That controls its
mathematical support domain; field errors at the displaced query remain a
separate contract.

## 7. Verification and remaining boundary

The new statements were reviewed against the existing inverse metric,
connection, actual operational gradient/Hessian definitions, and finite halo
germ transfer. The module target is
`lake build FractalSync.OperationalMechanicalBounds --wfail`; final build and
unchanged transitive axiom audit results are recorded in
[verification.md](verification.md).

These are exact-real certificates for actual coefficient functions. They do
not certify stored coefficient values for a particular asset, force-budget
values for a particular implementation, machine division or floor, arithmetic
rounding, query displacement, availability masks, or numerical update errors.
The proof does not identify the spline with ideal signed Shore distance.
Concrete force/potential fidelity and discrete control or steering contracts
remain separate. No runtime code, canonical parameter, provider contract,
training path, or dependency pin is changed by these bounds.
