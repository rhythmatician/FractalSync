# Issue 159: signed RK work and the mechanical-energy ledger

The [held-RK extension](159-held-runge-kutta.md) established the actual
mechanical recurrence, stage and endpoint domains, and convergence to the
reevaluated-command reference. This extension connects the same numerical
history to the runtime's signed control-plus-drag work quadrature.

The energy is the existing kinetic-plus-conservative-potential function. The
work samples are the actual stage powers, not the earlier Taylor increments
\(DE(y_i)[y_{i+1}-y_i]\). The argument uses a Lipschitz energy differential
and bounded stage slopes; it introduces no distance-to-set assumption or new
field formulation.

The checked result covers every prefix of the actual finite work ledger and
uniform convergence of its affine interpolant to the reference solution's
actual power integral. It uses the same reference solution for all outer
meshes and positive internal substep counts.

## 1. Actual power and the runtime work rule

Write \(E(z)\) for the fixture's mechanical energy at \(z=(x,v)\). For a
supplied drive covector \(q\), define

\[
P_q(z)=\langle v,q+Q_{\mathrm{drag}}(z)\rangle,
\qquad Q_{\mathrm{drag}}(z)=-G(x)v
\]

for the fixture's unit damping coefficient. Its held mechanical field is

\[
F_q(z)=\bigl(v,A_{\mathrm{hold}}(z;q)\bigr).
\]

`fderiv_energy_heldField` in
[`ConcreteHeldPower`](../../formal/FractalSync/ConcreteHeldPower.lean)
identifies the actual differential with this power on the certified phase
ball:

\[
\boxed{DE(z)[F_q(z)]=P_q(z).}
\]

The proof uses the actual ruler, gradient and potential jets. The conservative
forces cancel the potential derivative, while the metric-compatible connection
cancels the changing-metric kinetic term. The supplied \(q\) is arbitrary in
this pointwise identity. The later numerical-domain certificate specializes
it to \(q=Q_d(y_i)\), sampled at the outer anchor \(y_i\).

In [`operational_engine.rs`](../../runtime-core/src/operational_engine.rs),
the fifth component of `rhs` is precisely the coordinate pairing
\(v\cdot(q+Q_{\mathrm{drag}})\). `advance` uses only the first four
components to construct mechanical stages. For a substep of length \(s\),
`rk4_advance` returns the signed work increment

\[
\mathcal W_s(y)=\frac{s}{6}
\bigl(P_q(y)+2P_q(y_2)+2P_q(y_3)+P_q(y_4)\bigr),
\]

where \(y_2,y_3,y_4\) are the actual classical RK stages. The fifth component
never feeds back into those stage states. `Rk4FourSubsteps` sums four such
increments at \(s=h/4\), retaining the same \(q\) throughout the outer
step. `initial_state` initializes integrated work to zero, and `attempt_step`
adds the completed outer work increment to that ledger.

The definitions `meanPower`, `stepWork`, and `substepsWork` in
[`RungeKuttaWork`](../../formal/FractalSync/RungeKuttaWork.lean) use these
actual stage samples and weights. The power may have either sign; the
accumulator does not replace signed work by absolute work or by drive-only
input energy.

## 2. One substep: the two error contributions

For \(s\ge0\), suppose the energy is differentiable on a closed ball,
its differential is \(K\)-Lipschitz there, and the ball contains the source,
every stage, and the weighted endpoint. Suppose also that the stage power
is \(P=DE[F]\) and all mechanical stage slopes have norm at most \(M\).
The existing RK stage certificate gives

\[
\|y_2-y\|\le\frac{sM}{2},\qquad
\|y_3-y\|\le\frac{sM}{2},\qquad
\|y_4-y\|\le sM,\qquad
\|y_+-y\|\le sM.
\]

First, the endpoint Taylor remainder is bounded by

\[
|E(y_+)-E(y)-DE(y)[y_+-y]|
\le\frac K2\|y_+-y\|^2
\le\frac K2M^2s^2.
\]

Second, linearity of \(DE(y)\) moves it through the actual weighted RK
increment. At each stage, replacing \(DE(y_j)\) by \(DE(y)\) costs at most
\(K\|y_j-y\|M\). The weighted stage displacement is at most

\[
\frac16\left(0+2\frac{sM}{2}+2\frac{sM}{2}+sM\right)
=\frac{sM}{2}.
\]

Multiplication by the substep length therefore gives

\[
|\mathcal W_s(y)-DE(y)[y_+-y]|
\le\frac K2M^2s^2.
\]

Combining the two estimates yields

\[
\boxed{|E(y_+)-E(y)-\mathcal W_s(y)|\le KM^2s^2.}
\]

This estimate retains the classical RK weights. It requires no fourth-order
truncation theorem or high-order derivative bounds for the vector field.
The formal statement is `RungeKutta.step_energy_work_defect`; the two
contributions are `energy_taylor_remainder_on_ball` and
`StageBounds.meanPower_sub_le`.
Stage and endpoint membership are supplied by the already checked enclosing
ball construction, rather than assumed for the complete numerical history.

## 3. Internal substeps and the complete finite ledger

The fixture retains

\[
R=\frac1{16},\qquad T=\frac1{576},\qquad
K=134,\qquad M=16.
\]

The energy differential bound comes from the
[local-energy extension](159-local-energy-bounds.md). The held-field slope
bound and all stage domains come from the
[complete held-RK certificate](159-held-runge-kutta.md). Position and velocity
use Euclidean norms; phase space uses their maximum.

Let \(m=N+1\ge16\) be the outer-step count, \(h=T/m\), and let \(n>0\)
be the number of equal internal RK substeps. The actual field within outer
step \(i\) holds \(q_i=Q_d(y_i)\), even at later internal substep sources.
The work increment \(W_i\) is the sum of their actual stage-power
quadratures. Since \(134\cdot16^2=34304\), telescoping within the outer
step gives

\[
|E(y_{i+1})-E(y_i)-W_i|
\le n\,34304(h/n)^2
=\frac{34304h^2}{n}.
\]

Define the stored signed ledger at a prefix \(k\le m\) by
\(W^{(k)}=\sum_{i<k}W_i\), with \(W^{(0)}=0\). The same energy function
is used throughout the frozen field/profile. Thus a second telescoping
argument yields, for every such prefix,

\[
\boxed{
\left|E(y_k)-E(y_0)-W^{(k)}\right|
\le\frac1{8mn}.
}
\]

The exact full-horizon coefficient before enlargement is

\[
34304T^2=\frac{67}{648}<\frac18.
\]

The rational arithmetic is recorded in
[`RungeKuttaWorkMesh`](../../formal/FractalSync/RungeKuttaWorkMesh.lean),
and the actual recurrence/ledger integration is in
[`ConcreteRungeKuttaWork`](../../formal/FractalSync/ConcreteRungeKuttaWork.lean).
Its `workTrajectory_energy_defect_prefix` retains the prefix-dependent bound
\(k\,34304h^2/n\), and `workTrajectory_energy_defect` supplies the displayed
uniform budget for every \(k\le m\). These theorems derive numerical-state
membership from the complete held-RK certificate; they do not leave it as a
premise for the caller.

At the existing \(m=16384\) mesh, the coarse prefix and final bounds are:

| Internal substeps | Energy/work-ledger defect bound |
| --- | ---: |
| \(n=1\), the one-step RK4 formula | \(2^{-17}\) |
| \(n=4\), four equal RK4 substeps | \(2^{-19}\) |

These are specializations of `workTrajectory_16384_energy_defect`.

The certified upper bound improves by \(1/n\). This is a bound on the
energy/work discrepancy, not a measured error ratio between runs. The
previous coarse phase-trajectory estimate remains independent of \(n\).
The ledger estimate itself does not require comparison with the reference
solution or a sign assumption on the work.

## 4. Comparison with the reference energy change

Let \(z(t)\) be the common reference solution that continuously reevaluates
the fixed command. The existing actual trajectory theorem gives

\[
\|y_k-z(kh)\|_{\mathrm{phase}}\le\frac3{16m}.
\]

The energy is five-Lipschitz on the certified region, and the initial states
agree exactly. Therefore

\[
\boxed{
\left|W^{(k)}-\bigl(E(z(kh))-E(z(0))\bigr)\right|
\le\frac1{8mn}+\frac{15}{16m}.
}
\]

For \(n=1\) this is \(17/(16m)\); for \(n=4\) it is \(31/(32m)\).
The second term comes from comparing the held-command numerical history
with the reevaluated-command reference. Increasing the internal substep
count alone does not remove that outer-step policy discrepancy.
The checked grid comparison is `workTrajectory_reference_error` in
[`ConcreteRungeKuttaWorkConvergence`](../../formal/FractalSync/ConcreteRungeKuttaWorkConvergence.lean).

## 5. Continuous physical work and all-time ledger interpolation

Define the reference power by

\[
P(z)=P_{Q_d(z)}(z).
\]

`ConcreteHeldPower` identifies \(P(z)=DE(z)[F(z)]\) for the actual
reference field. Its regional bounds are \(\|P\|\le40\) and Lipschitz
constant \(3072\), obtained from \(DE:(5,134)\) and \(F:(8,400)\).
The actual chain rule and interval FTC then identify

\[
\mathscr W(t):=\int_0^t P(z(s))\,ds
=E(z(t))-E(z(0)),\qquad 0\le t\le T.
\]

This is `referenceWork_eq_energy_sub` in
[`ConcreteWorkIntegral`](../../formal/FractalSync/ConcreteWorkIntegral.lean).
The same module proves continuity and interval integrability of the actual
power trace, as well as `referenceWork_lipschitzOn` with constant 40.
The grid estimate in Section 4 consequently compares the numerical ledger
with an actual signed physical-work integral.

For an all-time comparison, define \(\widetilde W_{m,n}\) by affine interpolation
through the stored outer-grid values \(W^{(k)}\). This is an analysis
construction, not an additional runtime integration rule or RK dense-output
formula. The reference energy trace is 40-Lipschitz. Interpolating its
endpoint values adds at most \(40h/2=5/(144m)\) to the grid error. Hence

\[
\boxed{
\sup_{0\le t\le T}
|\widetilde W_{m,n}(t)-\mathscr W(t)|
\le\frac1{8mn}+\frac{35}{36m}
\le\frac9{8m}.
}
\]

`exists_phase_work_certificate` in
[`ConcreteRungeKuttaWorkConvergence`](../../formal/FractalSync/ConcreteRungeKuttaWorkConvergence.lean)
supplies one reference solution, the phase estimate \(3/(16m)\), and the
displayed work estimate for every \(m\ge16\), \(n>0\), and \(t\in[0,T]\).
It also proves continuity of both interpolants and containment of the phase
interpolant in the half-radius ball. The reference is quantified before the
mesh and internal count.

`exists_uniformly_convergent_phase_work` packages simultaneous uniform
convergence of the phase and work interpolants: for every positive error
budget, a single outer-mesh threshold controls both errors at every time and
for every \(n>0\). The reference work remains the actual interval integral
above. These statements do not assert fourth-order convergence for the
held-command comparison.

## 6. What the work bound does and does not certify

The result controls
\(E(y_k)-E(y_0)-W^{(k)}\). The driven and damped system may change energy:
the ledger includes both drive work and negative drag work. A small residual
does not assert that \(E(y_k)\) is close to \(E(y_0)\).

There is no normalized energy-residual threshold or work-defect refusal rule
in `FrozenRollout::attempt_step`. Its relevant checks are frozen identity,
finite inputs, stage/provider conditions, checked endpoints, and finiteness
of the proposed outer energy/time/work ledger. Internal `checked_endpoint`
calls compute energy and propagate provider/energy-function errors; the
explicit final ledger-finiteness check occurs at the outer commit.

The synthetic drag unit test in `operational_engine.rs` checks an absolute
energy/work residual below \(10^{-7}\) for a different constant-field
profile. The separate
[`replay_bounded_energy.py`](../../scripts/replay_bounded_energy.py) checks
absolute total-energy excursion and drift below \(0.05\) in undriven,
zero-drag frozen-asset runs. Neither is an engine acceptance threshold or a
normalized residual specification. The present theorem does not claim to
certify those different fixtures or parameter choices.

The correspondence retains the exact fixture caller policy: supplied drive
\(q_i=Q_d(y_i)\) at each outer anchor, with the existing research parameters.
IEEE power evaluation, rounded control construction, summation error,
compiled provider fidelity, masks/indexing, frozen refusal policy and
operational acceptance remain separate evidence obligations. The theorem
does not infer these from mathematical query-address validity.

The generic quadrature proof is reusable for another smooth field after its
own energy differential, regional field bounds and domain evidence are
established. The concrete constants belong to the current literal quadratic
fixture. The proposed field replacement remains future work under the
[existing reuse boundary](157-field-replacement-boundary.md).
