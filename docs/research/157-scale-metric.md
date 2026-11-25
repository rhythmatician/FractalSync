# Local-ruler metric foundations

This note supports #157 and Phase A of #159, and supplies the local algebra used
by #160. It concerns exact real arithmetic at one configuration. It does not
assert differentiability of a numerical GeometryProvider, or verify Rust's
floating-point operations.

## One rank-one operator

Let E be a real inner-product space, let rho>0, let a be real, and let u in E.
Define

\[
H v=v+a^2\langle u,v\rangle u,\qquad
W=1+a^2\lVert u\rVert^2,\qquad G=\rho^{-2}H.
\]

For FractalSync, u is the locally defined gradient of rho and
a=lambda/log(2). These pointwise results do not need u to be a gradient. In
particular, they do not assume that regularizing a distance function removes
its cut locus.

The corresponding bilinear form is

\[
g(v,w)=\frac{\langle v,w\rangle+
 a^2\langle u,v\rangle\langle u,w\rangle}{\rho^2}.
\]

It is symmetric, bilinear, and positive definite: for v nonzero, the first term
in its numerator is positive and the second is nonnegative. Cauchy--Schwarz
also gives the two-sided bound

\[
\frac{\lVert v\rVert^2}{\rho^2}\le g(v,v)
\le\frac{W\lVert v\rVert^2}{\rho^2}.
\]

The exact inverse is

\[
G^{-1}w=\rho^2\left(w-
 \frac{a^2\langle u,w\rangle}{W}u\right).
\]

To check it, first observe W>0 and
\(\langle u,Hv\rangle=W\langle u,v\rangle\).
Substituting this identity into the proposed inverse cancels the rank-one term.
The opposite composition follows from
\(\langle u,H^{-1}w\rangle=\langle u,w\rangle/W\).
No finite-dimensional assumption is needed for either identity.

Vectors perpendicular to u are eigenvectors with eigenvalue rho^-2, and u
itself satisfies the eigenvector equation with eigenvalue W/rho^2. When u=0,
the metric is isotropic; the vector equation still holds, but zero is not an
eigenvector in the usual definition. In dimension one there need not be a
nonzero perpendicular direction, so W must not automatically be called the
spectral condition number there. When both eigen-directions occur, their ratio
is W. Under the explicit ideal-field hypothesis norm(u)<=1,

\[
1\le W\le1+a^2.
\]

In an orthonormal coordinate system of finite dimension n, the matrix
determinant lemma gives

\[
\det G=\rho^{-2n}\left(1+a^2\sum_{i=1}^n u_i^2\right).
\]

The two-dimensional formula rho^-4 W in the equation catalog is the n=2
instance. The result remains valid for n=0, with determinant and empty product
both equal to one.

## Kinetic energy

At a fixed configuration,

\[
K(v)=\tfrac12 g(v,v)
=\frac{\lVert v\rVert^2+a^2\langle u,v\rangle^2}{2\rho^2}.
\]

If u=grad(rho) and a differentiable trajectory has velocity v, the chain rule
identifies the inner product with rho-dot. Combining that additional analytic
fact with a=lambda/log(2) and
sigma-dot=-rho-dot/(rho log(2)) yields the catalog's other kinetic-energy
expressions. The static algebra alone does not prove that a sampled provider
derivative is the derivative of its reported field.

## Metric rotations in an oriented plane

For a symmetric positive-definite matrix
\(G=\begin{pmatrix}A&B\\B&C\end{pmatrix}\), put Delta=AC-B^2>0. Define

\[
J_G(x,y)=\frac{1}{\sqrt\Delta}(-Bx-Cy,Ax+By).
\]

Direct multiplication shows J_G^2=-I. The bilinear form satisfies
g(v,J_Gv)=0 and g(J_Gv,J_Gv)=g(v,v). More generally, for real c,s with
c^2+s^2=1,

\[
g(cv+sJ_Gv,cv+sJ_Gv)=g(v,v).
\]

Taking c=cos(theta), s=sin(theta) proves the metric rotation identity. These
algebraic identities require only Delta>0. Positive definiteness is a separate
condition when interpreting g as a Riemannian metric; a negative-definite
matrix also has Delta>0 and satisfies the same identities.

## Fixed-configuration impulses

Let J be an impulse covector, represented by the ambient inner product, and let
the instantaneous velocity update be

\[
v_+=v+G^{-1}J.
\]

The exact kinetic-energy jump is

\[
\boxed{K(v_+)-K(v)=\langle v,J\rangle
 +\frac12\langle J,G^{-1}J\rangle.}
\]

This follows by expanding the bilinear form and using the proved inverse:
\(g(v,G^{-1}J)=\langle v,J\rangle\) and
\(g(G^{-1}J,G^{-1}J)=\langle J,G^{-1}J\rangle\).
Consequently the dual squared norm is nonnegative. Applying Cauchy--Schwarz
to the same bilinear form gives

\[
|\langle v,J\rangle|\le\sqrt{2K(v)}\,\lVert J\rVert_{G^{-1}}.
\]

For a dual-metric impulse budget \(\lVert J\rVert_{G^{-1}}\le J_{\max}\),

\[
\boxed{K(v_+)-K(v)\le
 \sqrt{2K(v)}J_{\max}+\frac12J_{\max}^2.}
\]

There is no explicit ruler factor in this bound. It uses an intrinsic impulse
budget, not a bound on the Euclidean coordinates of J. The premise itself
implies J_max>=0. Configuration, metric, and therefore potential are fixed
throughout the impulse. These statements do not assert continuous work-energy
balance, a numerical integrator's conservation, or that an arbitrary control
can realize the impulse.

## Checked proof boundary

`formal/FractalSync/ScaleMetric.lean` contains 31 checked theorem declarations;
`formal/FractalSync/MetricImpulse.lean` adds 14. The second module bundles the
first module's existing metric into Mathlib's `LinearMap.BilinForm`, then reuses
Mathlib's general Cauchy--Schwarz theorem. Both compiled with warnings treated
as errors on the repository's pinned Lean and Mathlib versions.

All names below have prefix `FractalSync.ScaleMetric.`; the plane names have
the additional prefix `Plane.`.

| Declarations | Checked content |
| --- | --- |
| `weight_pos`, `weight_norm`, `weight_bounds` | Strictly positive inverse denominator; its norm expression; the explicit conditional anisotropy bound. |
| `unstretch_stretch`, `stretch_unstretch`, `stretch_smul`, `unstretch_smul`, `inverse_metricAction`, `metric_inverseAction` | Two-sided Sherman--Morrison inverse and scaling identities in arbitrary real inner-product spaces. |
| `metric_self_formula`, `metric_self_pos`, `metric_self_nonneg`, `metric_symm`, `metric_add_left`, `metric_smul_left`, `metric_add_right`, `metric_smul_right`, `inner_metricAction` | Positive symmetric bilinear metric, zero-velocity case, and correspondence with its operator. |
| `metric_norm_lower`, `metric_norm_upper`, `metricAction_tangent`, `metricAction_normal` | Conformal comparison and the tangent/normal eigenvalue equations. |
| `det_metricMatrix`, `det_metricMatrix_pos`, `metricMatrix_apply` | Exact coordinate entries, determinant in every finite dimension, and positive determinant. |
| `kinetic_formula`, `two_kinetic`, `kinetic_nonneg`, `kinetic_add` | Equivalent kinetic energy and complete quadratic expansion. |
| `Plane.quarter_square`, `Plane.quarter_orthogonal`, `Plane.quarter_isometry`, `Plane.quarter_skew`, `Plane.form_mix`, `Plane.circle_rotation_isometry`, `Plane.rotation_isometry` | The quarter-turn and rotations preserve all bilinear pairings; only the determinant sign needed by each identity is assumed. |
| `metric_cauchy_schwarz_sq`, `metric_inverse_pairing`, `dualSq_eq_metric`, `dualSq_nonneg`, `dualNorm_sq`, `impulse_pairing_sq`, `impulse_pairing_bound`, `impulse_energy_jump`, `impulse_energy_bound` | Primal/dual pairing, exact impulse energy jump, and the scale-independent budget bound. |

The proofs contain no `sorry`, `admit`, or new axioms. The repository's umbrella
audit checks transitive dependencies after integration. The subsequent
[differential mechanics extension](159-differential-mechanics.md) now checks
the full ruler/scale Hessians, local connection and its Koszul uniqueness,
actual continuous work-energy derivative and signed integral ledger. Those
analytic results use explicit field/trajectory regularity assumptions in
addition to the static identities recorded here.

The [potential/control continuation](159-potential-controls.md) now checks
the actual Shore and secant-wall derivatives, their metric-raised forces,
full continuous energy and crest budget, momentum differentiation, intrinsic
drive/drag and additional candidate control formulas. The
[discrete note](159-discrete-mechanics.md) proves frozen-metric step identities,
including the exact explicit-drag energy factor and its sharp timestep
condition. Those statements preserve the changing-configuration numerical
boundary rather than treating continuous identities as integrator proofs.
The [moving-metric continuation](159-moving-metric-dynamics.md) now proves
a conditional quadratic local energy defect for the full exact-real ordered
step, the impulse ledger at the new position, a sufficient whole-segment
wall-domain condition, and continuous covariant steering norm preservation.
Actual provider/phase-Hessian certificates, floating execution, global rollout
control and a discrete steering realization remain separate.

## Relation to implementation

The definitions correspond to `induced_metric`, `kinetic_energy`, and the
inverse-metric operations in `runtime-core/src/manifold.rs`; the ruler and its
derivatives come from `runtime-core/src/geometry_provider.rs`. Rust remains
the runtime authority. In particular, the runtime's absolute determinant
threshold in `inverse_2x2` is a numerical acceptance rule, not a consequence of
exact positive definiteness. These theorems do not certify that every valid
exact metric passes that threshold.

The accepted Lean declarations and their precise verification boundary are
listed in [the formal inventory](../../formal/README.md).

For the authoring setup in #157, Lean and the pinned optional LSP/MCP server
were installed and the real stdio diagnostics/theorem smoke test passed.
The official `lean-proof` and `mathlib-build` skills were installed in the
personal catalog. A persistent Codex MCP registration was attempted but the
active client configuration is read-only; that concrete client-registration
step remains blocked. The formal README provides pinned Codex and VS Code
configuration templates and records the scope of the successful checks.

### Existing numerical seams and proposed follow-up checks

The highest existing seam for the local metric is the Rust unit-test module in
`runtime-core/src/manifold.rs`: `test_metric_positive_definite`,
`scale_relative_metric_inverse_is_identity`, and
`embedding_jacobian_and_qdot_consistency` already test the implemented metric,
inverse and constrained velocity relationship. The new proofs complement those
tests; they do not establish implementation correspondence. No runtime tests
were changed or rerun for this mathematical increment.

For a later runtime-focused change, extend those existing tests with a small
set of well-conditioned analytic jets, including zero gradient and an oblique
gradient. Compare the computed inverse and determinant with the proved
rank-one formulas, using relative residuals across several ruler scales. Keep
cases near the absolute determinant cutoff in a separate numerical-contract
test so exact SPD is not confused with the implementation's acceptance rule.

The narrow regression for an implemented impulse operation is to compute both
kinetic energies using the production metric, apply its production inverse to
the impulse, and check the jump and dual-budget bound at fixed configuration.
That tests the integration of existing operations rather than introducing a
second mechanics implementation. For ruler derivatives, the appropriate seam
is `rho_from_jet` and `grad_rho_from_jet` in `geometry_provider.rs` with analytic
jet fixtures. The full Hessian, local connection and continuous energy
identities now have the mathematical proofs in
[the differential extension](159-differential-mechanics.md); their numerical
implementation correspondence remains a separate validation obligation.
