# #159: regional mechanical bounds that retain metric cancellations

[RankOneInverseBounds.lean](../../formal/FractalSync/RankOneInverseBounds.lean)
and [CancelledCatalogBounds.lean](../../formal/FractalSync/CancelledCatalogBounds.lean)
prove smaller bounds for the existing exact-real mechanical expression. The
underlying metric, conservative forces, normalized drive, viscous covector,
and connection are unchanged. The improvement comes from simplifying the
actual inverse action before estimating its norm and variation.

## 1. The inverse is a contraction before ruler scaling

Write

\[
S_u=I+a^2u\otimes u,\qquad U_u=S_u^{-1},\qquad
G^{-1}=\rho^2U_u.
\]

The formal `stretch` and `unstretch` definitions already had checked
left- and right-inverse identities. The new proof establishes

\[
\|v\|\le\|S_uv\|,\qquad
\|U_uv\|\le\|v\|,\qquad
\boxed{\|G^{-1}v\|\le\rho^2\|v\|.}
\tag{1}
\]

For the first inequality, positivity gives
\(\langle v,S_uv\rangle=\|v\|^2+a^2\langle u,v\rangle^2\),
and Cauchy–Schwarz completes the argument. Applying it to \(U_uv\)
proves the contraction. This holds in every real inner-product space,
without a nonzero-gradient hypothesis or a unit-gradient assumption.
The last inequality itself does not require \(\rho>0\); positivity is
required when interpreting the original metric and cancelling its inverse.

The parameter variation uses the checked resolvent identity

\[
U_uv-U_wv=U_u\bigl((S_w-S_u)U_wv\bigr).
\]

Contraction at both ends and the rank-one difference estimate give

\[
\boxed{
\|U_uv-U_wv\|
\le a^2(\|u\|+\|w\|)\|u-w\|\|v\|.
}
\tag{2}
\]

This avoids treating the two terms of the Sherman–Morrison inverse as
independent contributions. No denominator derivative needs to be estimated
separately in (2).

If the regional bounded-Lipschitz data are

\[
\rho:(R,L_\rho),\qquad u:(U,L_u),\qquad Q:(V,L_Q),
\]

then `inverseAction_boundedLipschitz` proves the actual composed field has

\[
\boxed{
G^{-1}Q:
\left(R^2V,\quad
R^2L_Q+(2RL_\rho+2R^2a^2UL_u)V\right).
}
\tag{3}
\]

The domain can be an arbitrary pseudometric region. No differentiability
or convexity is needed for these algebraic estimates. At the local inputs
\(\rho:(2,1)\), \(u:(1,20)\), \(a=1\), the inverse operator has norm
bound 4 and parameter Lipschitz bound 164. These are sufficient uniform
bounds, not an optimality claim for a particular field.

## 2. Raise the catalog before combining its bounds

For the research-instance parameters

\[
\kappa=\log2,\qquad \mu=1/\pi,\qquad F=\beta=1,
\]

and fixed nonzero direction \(e\), let
\(d_g=\operatorname{metricNormalize}(\rho,a,u,e)\). The actual catalog is

\[
Q_{\rm tot}=Q_{\rm Shore}+Q_{\rm wall}+Gd_g-Gv.
\]

`raised_catalog_eq` checks the identity

\[
\boxed{
G^{-1}Q_{\rm tot}
=\frac{\rho}{W}u+G^{-1}Q_{\rm wall}+d_g-v,
\qquad W=1+a^2\|u\|^2.
}
\tag{4}
\]

The \(G^{-1}G\) cancellations in drive and drag are exact. The Shore
term cancels its inverse-ruler factor before its bound is propagated.
The wall term uses (3). The formal heading remains reevaluated at the
current phase point, as in the existing fixed-command model. A held drive
covector is a separate residual estimate, documented in
[the held-control note](159-held-control-residual.md).

The general theorem `raised_catalog_boundedLipschitz` proves the budget
`cancelledRaisedCatalogBudget`. It assumes a region with position norm at
most one, uses the already checked default-wall bound, and accepts arbitrary
regional ruler, gradient, and velocity budgets. The theorem
`cancelled_catalog_acceleration_bound` combines it with the existing
`connectionBudget`. No new connection formula is introduced.

## 3. Exact local constants

Assume the regional fields satisfy

\[
\rho\ge1,\quad \rho:(2,1),\quad u:(1,20),\quad
H:(20,200),\quad x:(1,1),\quad v:(1/3,1),\quad a=1.
\]

Here each pair is a norm bound and Lipschitz constant for that actual field
on the same region. The algebraic theorem accepts \(u,H\) as fields; their
identification with \(\nabla\rho,Du\) is supplied by the separate local-ruler
certificate in the mechanical application.

| Actual term | Norm bound | Lipschitz bound |
| --- | ---: | ---: |
| Raised Shore force \(\rho u/W\) | 2 | 121 |
| Raised default wall force | \(1/4\) | \(49/4\) |
| Raised normalized drive \(d_g\) | 2 | 41 |
| Raised viscous force \(-v\) | \(1/3\) | 1 |
| Complete raised catalog | \(55/12\) | \(701/4\) |
| Connection \(\Gamma(v,v)\) | \(8/3\) | \(1748/9\) |
| Actual acceleration \(G^{-1}Q_{\rm tot}-\Gamma(v,v)\) | \(29/4\) | \(13301/36\) |

`cancelled_local_budget` checks the exact rational evaluations.
`cancelled_catalog_acceleration_local_bound` then proves the convenient
majorants

\[
\boxed{\|A(z)\|\le8,\qquad
\|A(z)-A(w)\|\le400\,\operatorname{dist}(z,w).}
\tag{5}
\]

For comparison and use by interfaces that need the covector itself,
`catalogForce_local_bound` checks

\[
\boxed{Q_{\rm tot}:(6,210).}
\tag{6}
\]

Before enlargement, the unraised force budget is
\((275/48,1249/6)\). Its component budgets are Shore \((1,21)\),
wall \((1/16,1/2)\), normalized drive \((4,170)\), and drag
\((2/3,50/3)\). Applying a crude inverse estimate to their sum would
lose the cancellations in (4), so (5) is proved from the raised terms.

The resulting concrete horizon and mesh theorem is described in
[the local-horizon note](159-local-horizon.md). These bounds concern
exact real functions. They do not certify compiled provider evaluation,
machine arithmetic, the runtime Runge–Kutta update, changing commands,
impulses, or continuation outside the stated region.

## 4. Concrete held-drive comparison

[ConcreteHeldControl.lean](../../formal/FractalSync/ConcreteHeldControl.lean)
instantiates the holding residual on this same literal quadratic fixture. Its
`drive` is the actual normalized drive covector for the fixed command
\(e=(1/4,0)\). Its `frozenAcceleration(q,z)` reevaluates the conservative
forces, viscous drag, ruler, inverse metric, and connection at \(z\), while
using the supplied drive covector \(q\). It checks

\[
A_{\rm hold}(q,z)-A(z)=G_z^{-1}(q-Q_d(z))
\]

and equality at the sample state when \(q=Q_d(z)\).

The fixture's local jet certificate discharges the ruler and gradient
premises of the generic holding theorem. For sample and current states
\(z_0,z\in\overline B((x_0,v_0),1/16)\), the concrete theorem proves

\[
\|A_{\rm hold}(Q_d(z_0),z)-A(z)\|
\le680\,\operatorname{dist}(z_0,z).
\]

For an explicit construction-error premise
\(\|q-Q_d(z_0)\|\le\delta_Q\), it proves

\[
\|A_{\rm hold}(q,z)-A(z)\|
\le4\delta_Q+680\,\operatorname{dist}(z_0,z).
\]

A further displacement premise
\(\operatorname{dist}(z_0,z)\le8\tau\) yields the checked numerical
bounds \(5440\tau\) and \(4\delta_Q+5440\tau\), respectively. No generic
asset, reference-fidelity, force-regularity, or ruler-jet premise remains in
these concrete statements. Region membership, construction error, and the
compared states' displacement remain explicit premises. In particular, the
speed bound of the reference curve is not used to assert that an actual
Runge–Kutta stage has the required displacement, and the coefficient 5440
is not inserted into a theorem that permits only a unit residual coefficient.

## 5. Verification scope

The inverse and catalog modules add seventeen proved theorem declarations:
ten in `RankOneInverseBounds` and seven in `CancelledCatalogBounds`, including
supporting algebraic lemmas and the exact budget evaluation.
`ConcreteHeldControl` adds seven further proved declarations. Targeted Lean
checks pass using the existing Lean and Mathlib pins. No placeholders,
new axioms, native-evaluation proofs, or modifications to the audit are used.
Runtime behavior, canonical parameters, and provider contracts are unchanged.
