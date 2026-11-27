# #160: the actual ambient Shore connection and curvature

The normal-coordinate calculation is now connected to the **actual ambient
metric connection of regularized distance**. The proof derives ambient ruler
regularity from the inverse normal chart, uses Mathlib's actual gradient, and
identifies the transformed compact connection by torsion freedom and metric
compatibility. A neighborhood identity then transfers the actual curvature
commutator. The resulting curvature ratio is `+1` at every represented Shore
point under the stated smooth normal-patch hypotheses.

The new sources are
[MetricConnectionUniqueness.lean](../../formal/FractalSync/MetricConnectionUniqueness.lean),
[EmbeddedShoreConnection.lean](../../formal/FractalSync/EmbeddedShoreConnection.lean),
and [EmbeddedShoreCurvature.lean](../../formal/FractalSync/EmbeddedShoreCurvature.lean).
They extend [derived normal radii](160-derived-normal-radii.md),
[embedded normal neighborhoods](160-embedded-normal-neighborhoods.md), and
[the curved coordinate calculation](160-curved-coordinate-geometry.md).

## 1. What the final connection theorem assumes

Let `E` be a complete real inner-product space and let

\[
\Phi(r,s)=C(s)+rN(s),\qquad
\rho(x)=\sqrt{\epsilon^2+d(x,S)^2},\qquad \epsilon>0.
\]

The geometric input is the existing `NormalPatch S R C N V` certificate. It
states `R>0`, that each foot `C(s)` belongs to the actual set `S` for `s` in
`V`, that `N(s)` has unit norm there, and that the full-set normal chord
inequality holds. The earlier jet/separation theorems can construct this
certificate; it is not a new distance or nearest-point assumption.

At the coordinate point `(r,s)`, the connection theorem requires:

* `V` contains a neighborhood of `s`, and `|r|<R`;
* `Phi` is `C²` locally and its actual derivative is invertible;
* the nonfocal factor `1-kappa(s) r` is nonzero;
* `kappa` is differentiable at `s`;
* on a parameter neighborhood, the actual Frenet identities
  `C'=T`, `N'=-kappa T`, `norm(T)=1`, and `inner(N,T)=0` hold.

The wrapper `ambientConnection_normalFrame` replaces the abstract derivative
invertibility premise by an actual continuous linear frame equivalence
`B : (R x R) ≃L E` with `B(1,0)=N(s)`, `B(0,1)=T(s)`, and the nonfocal factor.
It derives the normal-map differential and its inverse. In that wrapper,
local `C²` regularity of `C` and `N` derives `C²` regularity of `Phi`.

No ambient gradient, Hessian, metric-pullback identity, or connection
transformation identity is assumed by either final theorem. The normal patch
is local in `s`; no global injective parameterization, convexity of `S`, or
global smooth distance function is required. A supplied frame equivalence
identifies the ambient space with the plane. The core theorem instead states
invertibility of the actual normal-map derivative directly.

## 2. Actual ambient regularity is derived

The inverse function theorem constructs an open partial homeomorphism with
forward map `Phi` and a smooth local inverse. The normal-patch result already
identifies the actual set distance on an ambient neighborhood as

\[
d(x,S)=\left|\operatorname{fst}(\Phi^{-1}(x))\right|.
\]

Consequently the actual regularized distance agrees on that neighborhood with

\[
\sqrt{\epsilon^2+\operatorname{fst}(\Phi^{-1}(x))^2}.
\]

`contDiffAt_ambientRuler_of_normalMap` now proves that this actual ruler has the
same regularity as the locally invertible normal map, for every stated order
`n>=1`. The scalar square root is differentiated only where its argument is
strictly positive. This proves smoothness through `r=0`; it never differentiates
the absolute-value cusp as if unsigned distance itself were differentiable.

The final connection is the concrete function

\[
\Gamma^{\mathrm{amb}}_x(v,w)
=\operatorname{connection}
  \bigl(\rho(x),a,\nabla\rho(x),D(\nabla\rho)_x\bigr)(v,w),
\]

using the existing compact expression from
[ContinuousMechanics.lean](../../formal/FractalSync/ContinuousMechanics.lean).
Here `gradient` is Mathlib's Riesz gradient of the actual scalar function;
the Hessian operator is its actual `fderiv`.

The new proofs establish the exact gradient derivative identity and `C¹`
gradient regularity from `C²` ruler regularity. Conversely, an actual `C¹`
gradient on a neighborhood makes its scalar field `C²`.
`gradient_derivative_symm` derives symmetry of the gradient derivative using
the actual second-Fréchet-derivative symmetry theorem. Symmetry is not an
additional numerical-provider contract in the final embedded result.

For the curvature stage, `C³` ruler regularity gives a `C²` gradient. The
compact connection evaluated on `C¹` vector fields is actually `C¹` wherever
the ruler is nonzero. A `C¹` direction field and a `C²` differentiated field
therefore have a `C¹` covariant derivative. These statements discharge the
intermediate differentiability premises in the ambient curvature transport.

## 3. Connection identification uses differentiated metric equality

The ambient metric is the actual derivative-based graph metric

\[
g_x(v,w)=\frac{\langle v,w\rangle
  +a^2D\rho_x[v]D\rho_x[w]}{\rho(x)^2}.
\]

The normal-patch and Frenet identities first prove the metric pullback on a
whole coordinate neighborhood:

\[
\Phi^*g=
\frac{\epsilon^2+(1+a^2)r^2}{(\epsilon^2+r^2)^2}\,dr^2
+\frac{(1-\kappa(s)r)^2}{\epsilon^2+r^2}\,ds^2.
\tag{1}
\]

The neighborhood condition matters. Equality of metrics at one point alone
would not justify equating their derivatives.

For a coordinate direction `d`, the proof differentiates the actual pulled-back
pairing along `x+t d`. Both derivatives of `D Phi` remain in the calculation:

\[
\begin{aligned}
D(\Phi^*g)_x[d](v,w)
={}&Dg_{\Phi(x)}[D\Phi_x d](D\Phi_xv,D\Phi_xw)\\
&+g_{\Phi(x)}(D^2\Phi_x[d,v],D\Phi_xw)\\
&+g_{\Phi(x)}(D\Phi_xv,D^2\Phi_x[d,w]).
\end{aligned}
\tag{2}
\]

The already-checked ambient compact connection is metric compatible. Combining
it with (2) proves compatibility of the actual nonlinear pullback

\[
\widetilde\Gamma_x(v,w)
=D\Phi_x^{-1}\left(D^2\Phi_x[v,w]
  +\Gamma^{\mathrm{amb}}_{\Phi(x)}(D\Phi_xv,D\Phi_xw)\right).
\tag{3}
\]

Actual Hessian symmetry proves torsion freedom. The new generic uniqueness
theorem derives the Koszul identity from symmetry and compatibility, then uses
positive definiteness to show that two compatible torsion-free candidates
coincide. Applied to (1), it proves

\[
\boxed{\widetilde\Gamma
  =\Gamma^{\mathrm{curved}}_{\epsilon,a,\kappa}.}
\tag{4}
\]

The right side is the existing explicit `CurvedShoreGeometry.christoffel`, with
its actual `deriv kappa` term. Thus (4) identifies the connection, rather than
assuming that the prior scalar coordinate formulas apply to an embedded Shore.

## 4. Actual ambient curvature is transferred

With `kappa` locally `C¹`, the sufficient assumptions for (4) persist on a
neighborhood. `eventually_ambientConnection_normalMap` packages that germ.
`curvature_congr_connection` proves that the actual commutator depends only on
the germ of its connection coefficients. It compares the inner covariant
fields as germs before taking their Fréchet derivatives.

For the curvature convention

\[
R(X,Y)Z=\nabla_X\nabla_YZ-\nabla_Y\nabla_XZ-\nabla_{[X,Y]}Z,
\]

`curvature_normalMap` therefore equates the actual commutator computed from
the pulled-back ambient connection with the existing curved-coordinate
commutator. This equality holds for arbitrary coordinate fields as an equality
of the actual definitions.

The ambient-coordinate comparison is also checked. For a `C³` normal map,
`C¹` ambient direction fields `X,Y`, and a `C²` ambient field `Z`,
`ambient_curvature_pullback` proves

\[
R^{\mathrm{curved}}(\Phi^*X,\Phi^*Y)(\Phi^*Z)
=\Phi^*\bigl(R^{\mathrm{amb}}(X,Y)Z\bigr).
\tag{5}
\]

Here `Phi*X` is Mathlib's actual inverse-Jacobian vector-field pullback.
The proof derives the intermediate ambient covariant-field regularity from
the actual `C³` ruler, then applies the existing nonlinear covariance theorem.
`ambient_gramRatio_pullback` similarly transports curvature/metric-Gram ratios
for a `C¹` first field and `C²` second field. Its raw real equality uses totalized
division; an arbitrary pair of fields still needs a nonzero Gram denominator
for a sectional-curvature interpretation.

For the actual normal/tangential coordinate frame, `normalGram_pos` supplies
that denominator condition. It is

\[
\frac{\epsilon^2+(1+a^2)r^2}{(\epsilon^2+r^2)^2}
\frac{(1-\kappa(s)r)^2}{\epsilon^2+r^2}>0.
\]

The definition `ambientNormalCurvature` forms the ratio from the actual
ambient metric and actual ambient connection transported by `Phi`. For a
`C³` normal map and locally `C²` curvature profile, the checked result is

\[
\boxed{
K(\Phi(r,s))=
\frac{\epsilon^4-(1+a^2)r^4}
     {(\epsilon^2+(1+a^2)r^2)^2}
-\frac{\kappa(s)(1+a^2)r(\epsilon^2+r^2)^2}
 {(1-\kappa(s)r)(\epsilon^2+(1+a^2)r^2)^2}.
}
\tag{6}
\]

The derivatives of `kappa` cancel in the earlier coordinate computation; the
new theorems justify its application to this actual embedded metric connection.
At Shore, (6) gives

\[
\boxed{K(C(s))=1,}
\tag{7}
\]

independently of `epsilon`, `a`, and the local Euclidean Shore curvature.
The sufficient regularity hypotheses are stated on the normal map and curvature
profile. No minimal-regularity theorem for a bare class of Shore embeddings is
claimed.

## 5. Inventory, verification, and remaining boundary

| New module | Proved theorem/lemma declarations |
| --- | ---: |
| `MetricConnectionUniqueness` | 3 |
| `EmbeddedShoreConnection` | 17 |
| `EmbeddedShoreCurvature` | 10 |
| Total | 30 |

These counts include supporting lemmas. The targeted build passes with no
warnings under the existing Lean 4.24.0 and Mathlib pins:

```text
lake build --wfail FractalSync.MetricConnectionUniqueness \
  FractalSync.EmbeddedShoreConnection FractalSync.EmbeddedShoreCurvature
```

The actual-gradient proof reuses Mathlib's `Gradient.Basic` and Riesz duality;
the normal inverse reuses its `InverseFunctionTheorem.ContDiff`. Their existing
pinned precompiled cache files were fetched. No dependency version changed.
Combined package verification and the transitive axiom audit are recorded in
[verification.md](verification.md).

The ambient connection-identification and local curvature-transfer gap is now
closed under the stated normal-patch and smoothness hypotheses. The following
remain separate:

* Constructing and gluing Mathlib manifold/vector-bundle connection and curvature
  objects over a general atlas. The checked objects here are actual coordinate
  functions, derivatives, commutators, and bilinear forms.
* Deriving numerical normal-jet and separation constants for a concrete Shore
  representation, broader reach theorems, and weaker regularity hypotheses.
* Regularity and the geodesic ODE for arbitrary attained AC minimizers, general
  conjugacy/second variation, and global cut-locus results.
* The curved-Shore functional-analytic bifurcation and local-minimum argument.
* Numerical provider fidelity and IEEE arithmetic. A smooth abstract normal
  patch does not certify a runtime field or assert smoothness at a fractal Shore
  point or a distance cut locus.

The runtime, training, canonical parameters, provider contracts, audit allowlist,
and dependency pins are unchanged. The research issues remain open and PR #164
remains a draft.
