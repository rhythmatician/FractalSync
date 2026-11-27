# Issue 159: complete exact-real held-control RK histories

This increment closes the stage and method-residual bridge for the existing
quadratic fixture. It uses the actual classical four-stage Runge–Kutta
recurrence, including any positive number of equal internal substeps with one
drive covector held across the entire outer step. In particular, it covers the
exact-real mechanical formulas corresponding to `Rk4` and
`Rk4FourSubsteps` in
[`operational_engine.rs`](../../runtime-core/src/operational_engine.rs).

The proof uses local field estimates. It adds no distance-to-set geometry and
does not select a replacement for the present SDF. The
[field-replacement boundary](157-field-replacement-boundary.md) identifies
which existing results can be reused if the field changes.

## 1. The actual recurrence and its caller contract

Let \(A(z)\) be the existing fixture acceleration with its fixed nonzero
command reevaluated at \(z=(x,v)\), and let \(Q_d(z)\) be that command's drive
covector. For an outer-step anchor \(y\), define

\[
f_y(z)=\bigl(v,A_{\mathrm{hold}}(z;Q_d(y))\bigr).
\]

All metric, potential, wall, drag and connection expressions are reevaluated
at the stage state. Only the supplied drive covector is held. The exact
identity between the held and reevaluated accelerations is proved in
[`ConcreteHeldControl`](../../formal/FractalSync/ConcreteHeldControl.lean).

For one substep of length \(s\),

\[
\begin{aligned}
k_1&=f_y(z),&k_2&=f_y(z+\tfrac{s}{2}k_1),\\
k_3&=f_y(z+\tfrac{s}{2}k_2),&k_4&=f_y(z+s k_3),\\
\operatorname{RK}_s^{f_y}(z)
&=z+\frac{s}{6}(k_1+2k_2+2k_3+k_4).
\end{aligned}
\]

The complete outer update composes \(n>0\) such substeps at \(s=h/n\),
starting from \(y\). The field \(f_y\) and its held covector remain unchanged
through all \(n\) substeps. A new covector is sampled only at the next outer
anchor. These are literal definitions in
[`RungeKuttaStages`](../../formal/FractalSync/RungeKuttaStages.lean) and
[`RungeKuttaPhase`](../../formal/FractalSync/RungeKuttaPhase.lean).

This specifies an exact-real caller policy. `FrozenRollout::attempt_step`
accepts a supplied covector; it does not itself guarantee that the caller
constructed \(Q_d(y)\). The fixture uses
\(\epsilon=a=\beta=1\), \(\kappa=\log 2\), \(\mu=1/\pi\), and unit
intrinsic drive along the first axis. In Rust's parameterization, \(a=1\)
corresponds to `lambda_sq = (log 2)^2`. Since `MAX_DRIVE_FORCE` is 2, unit
intrinsic drive corresponds to command magnitude \(1/2\). These are the
fixture's research parameters, not an assertion about the default runtime
profile, arbitrary supplied controls, or rounded command construction.

## 2. Closing the stage-domain estimate

Position and velocity carry Euclidean norms. Phase space uses

\[
\|(x,v)\|_{\mathrm{phase}}=\max\{\|x\|_2,\|v\|_2\}.
\]

The [existing local certificate](159-local-horizon.md) gives, on the closed
phase ball of radius \(R=1/16\) about the original fixture state \(z_0\),

\[
\|F(z)\|\le8,\qquad
\|A(z)-A(w)\|\le400\|z-w\|,
\qquad F(z)=(v,A(z)).
\]

The held-drive comparison contributes another \(680\|z-y\|\). Thus
[`ConcreteRungeKuttaBounds`](../../formal/FractalSync/ConcreteRungeKuttaBounds.lean)
proves the anchored bounds

\[
\|A_{\mathrm{hold}}(z;Q_d(y))-A(y)\|
\le1080\|z-y\|,
\qquad
\|f_y(z)\|\le8+1080\|z-y\|.
\]

These compare each stage with the outer anchor; no unproved two-point
Lipschitz estimate for the frozen field is inserted.

If \(y\in B(z_0,R/2)\), \(16h\le R/2\), and \(h\le1/2160\), then
the entire closed ball \(\overline B(y,16h)\) lies in the original region,
and throughout that ball

\[
\|f_y(z)\|\le8+1080(16h)\le16.
\]

The generic stage theorem then proves containment and slope bounds for the
actual stages in sequence. Induction over internal substeps keeps every
stage, every internal endpoint, and the final weighted endpoint in the same
enclosing ball. This avoids assuming the stage membership that the bound is
intended to establish. The weighted endpoint is included separately because
it generally differs from the fourth stage.

## 3. Exact residuals passed to the existing trajectory theorem

Let \(\bar f\) denote the average of all actual RK stage slopes over the
outer step, including the four RK weights and all \(n\) substeps, and let
\(\bar A=\bar f_2\). Lean proves the exact telescoping identity

\[
y_+=y+h\bar f.
\]

Positive averaging and the enclosing-ball estimates give

\[
\|\bar A-A(y)\|\le17280h,
\qquad
\|\bar f_1-v\|\le16h,
\qquad \|\bar A\|\le16.
\]

The existing trajectory machinery uses a velocity-first comparison step
\(v_+=v+h\bar A\), \(x_+=x+h v_+\). Its velocity agrees exactly with the
RK endpoint. The remaining position discrepancy satisfies

\[
\boxed{\|y_+-\operatorname{positionVelocityStep}(h,y,\bar A)\|
\le32h^2.}
\]

The coefficient \(17280\) is an acceleration residual; \(32h^2\) is a
deterministic comparison between two exact-real update formulas. It occupies
the existing certificate's stored-step-error slot. It is not a certificate
for machine rounding. No residual or stage-membership premise is left
uninstantiated in the resulting fixture recurrence.

## 4. The finite-run certificate

The reference remains the actual solution of the reevaluated-command ODE on

\[
T=1/576.
\]

Its displacement is at most \(8T=1/72<R/4\). Consequently a closed tube of
radius \(R/4\) around the reference fits inside \(B(z_0,R/2)\), leaving the
outer half-radius for stage evaluations.

Write \(m=N+1\) for the number of outer steps and \(h=T/m\). The
[mesh arithmetic](../../formal/FractalSync/RungeKuttaMesh.lean) checks the
existing interpolation coefficient with acceleration residual \(17280h\),
stored-step discrepancy \(32h^2\), and exact initial state. The coefficient
is at most 108. For every \(m\ge16\), all domain and step-size inequalities
hold and the actual outer-state piecewise-affine interpolant \(Y_m\) obeys

\[
\boxed{
\sup_{0\le t\le T}\|Y_m(t)-z(t)\|_{\mathrm{phase}}
\le\frac{3}{16m}.
}
\]

The same reference and coarse error bound apply for every positive internal
substep count, including one and four. The quantified uniform-convergence
corollary is also checked, uniformly over all these internal counts.
At \(m=16384\), the bound is
\(3/262144<2^{-16}\). The interpolant is defined through the stored outer
states; it is not RK dense output or an interpolation of the internal stages.

[`ConcreteRungeKuttaQueries`](../../formal/FractalSync/ConcreteRungeKuttaQueries.lean)
proves stage and endpoint query validity. A reported query whose maximum
coordinate error from any certified stage position is at most \(1/64\)
retains all 36 valid mathematical
stencil addresses, using the existing literal asset and halo certificate.
The same statement holds at every time of the outer-state interpolant.
The displacement \(1/64\) is an admitted query budget, not a derived
floating-point rounding bound.

This is a first-order convergence estimate against the continuously
reevaluated-command reference. Holding the covector creates an additional
policy discrepancy, so a classical fourth-order RK result for one fixed
smooth field would not alone establish a fourth-order result for this
comparison. The proof makes no claim that four internal substeps improve
the displayed coarse bound.

## 5. Formal and implementation boundary

The stage algebra, finite averaging, method comparison and error propagation
are independent of distance-to-set properties. A replacement field can use
them after supplying its own local norm, variation and domain evidence.
The concrete constants above certify the existing literal quadratic asset.

Rust separately checks state finiteness, the operational disk, provider
regularity, frozen identity, intermediate endpoints and energy/ledger data.
The theorem does not certify IEEE evaluations, storage/mask/index operations,
sample construction, refusal branches, or arbitrary caller behavior.

The fifth RHS component is power \(v\cdot(q+Q_{\rm drag})\); its RK
quadrature does not feed back into the mechanical stages. The separate
[signed-work extension](159-rk-work-ledger.md) now identifies this actual
stage power with \(DE[F_q]\) and bounds the exact-real cumulative
energy/work-ledger discrepancy by \(1/(8mn)\) at every outer prefix, for
\(m\ge16\) outer steps and \(n>0\) equal internal substeps. That argument
uses the stage domains proved here and the actual energy differential's
Lipschitz bound. Its \(1/n\) work-ledger improvement is separate from this
note's coarse phase-error rate.

The same extension identifies the reference work with its actual signed
power integral. One common reference solution controls both interpolants:
the affine work interpolant has all-time error at most
\(1/(8mn)+35/(36m)\le9/(8m)\), uniformly over positive internal counts.
The phase and work interpolants therefore converge uniformly together on
the same local interval.

IEEE power evaluation and ledger summation remain outside those exact-real
results. Runtime acceptance includes finiteness of the proposed outer ledger;
`attempt_step` has no energy/work-residual tolerance or drift-based refusal
rule. Changing commands, impulses, combined steering and continuation beyond
this local interval remain separate work.

No runtime implementation, parameter, provider contract, dependency pin or
audit rule changes in this increment. The geometric obligations and draft
status of PR #164 remain unchanged.

## Executable fixture addresses

The release-mode Rust regression decodes the pinned 256 binary64
coefficients, verifies the asset byte round trip, and runs 16 outer steps
for each of the one- and four-substep profiles on the horizon $1/576$.
It uses the runtime Controls drive at each outer anchor and holds the
resulting covector during that step.
[Literal asset](../../formal/fixtures/operational-quadratic.json)

A test-only recorder at the operational provider entry measures every
attempted stage and endpoint query, including repeated queries for
forces and energy. The recorder count agrees with the reported query
work. The test separately checks three interpolated positions per outer
step. For each tested coordinate, an independently indexed sparse asset
with only the predicted 36 addresses produces the same jet as the full
asset. On one query per visited cell, removing any one of those addresses
refuses the query. A frozen reference refusal at the actual second-stage
position rolls back the full state in both profiles.
[Compiled regression](../../runtime-core/src/operational_engine/fixture_queries.rs)

The one-substep run recorded 368 provider calls and the four-substep run
recorded 1,472. Each run added 48 interpolation probes. All of those
coordinates selected the same cell in this fixture. The separate
constant-field rollback tests use missing support at intermediate stages
and at a weighted outer endpoint.

```powershell
rtk cargo test --release -p runtime_core --features experimental-operational-field --lib literal_quadratic_rk_queries_require_exactly_the_predicted_stencil -- --nocapture
```

This exercises the actual machine-selected stencil on a fixed finite run.
The rounded-stage checker also bounds the local mechanics and work residuals
for both profiles. Its one-substep trace meets the local-error theorem's
numerical premises. Uniform rounded-stage bounds remain open in #181.
[Measured binary64 residuals](159-rk-operation-errors.md#measured-binary64-residuals-for-the-literal-fixture)
[#181](https://github.com/rhythmatician/FractalSync/issues/181)

## Composing heterogeneous accepted intervals

Issue #178 needs a phase-error budget when accepted step lengths vary. For
accepted states compared with reference states at matching times, let $e_j$ be
the error at the start of interval $j$. Give each accepted step its own
nonnegative gain $\kappa_j$ and local defect $\Delta_j$ satisfying
$e_{j+1}\le\kappa_j e_j+\Delta_j$. Starting from $B_0\ge e_0$, the Lean
recurrence $B_{j+1}=\kappa_j B_j+\Delta_j$ bounds every accepted prefix.
Each defect enters once and is amplified only by subsequent gains. The empty
prefix has the initial budget. This applies to unequal dyadic lengths when the
local contracts hold on their actual reference intervals.

If the gains are bounded by $1+Lh_j$ with $L,h_j\ge0$, the defects are
nonnegative, and $B_0\ge0$, a second theorem gives

\[
e_n\le \exp\!\left(L\sum_{j<n}h_j\right)
          \left(B_0+\sum_{j<n}\Delta_j\right).
\]

This uses actual accepted time and actual local defects. A fixed elapsed time
alone does not ensure convergence because the total local defect may fail to
shrink with the mesh.

The local stage and provider obligations still apply on each interval. The
recurrence covers the accepted history supplied to it. The executable caller
must provide evidence that rejected trials leave accepted phase, work, and
impulse accounting intact, and that accepted intervals exactly fill the
scheduled outer step. The budget is over real numbers and requires separate
rounding evidence for a stored implementation.

[Accepted-step composition proof](../../formal/FractalSync/AdaptiveStepComposition.lean)
