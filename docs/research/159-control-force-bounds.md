# #159: regional constants for actual drive and viscous drag

[ControlForceBounds.lean](../../formal/FractalSync/ControlForceBounds.lean)
derives norm and Lipschitz constants for the actual metric-lowering operator,
intrinsic drive covector, and isotropic viscous covector. In particular, the
metric normalization of a fixed nonzero command now has a derived regional
bound. It is no longer necessary to assume a Lipschitz certificate for that
normalized heading independently of the ruler and gradient field.

The module uses the budget algebra from
[MechanicalBounds.lean](../../formal/FractalSync/MechanicalBounds.lean) and
the actual control definitions from
[ControlMechanics.lean](../../formal/FractalSync/ControlMechanics.lean).
The earlier [potential and control note](159-potential-controls.md) records
the work, impulse, and dissipation identities. This increment supplies the
field regularity estimates needed to insert those controls into regional
trajectory certificates.

## 1. Regional data and metric lowering

Let the region be a subset of a pseudometric space, and let the output
vectors belong to a real inner-product space. A certificate
`BoundedLipschitzOn f s M L` proves both

\[
\|f(x)\|\le M,\qquad
\|f(x)-f(y)\|\le L\,\operatorname{dist}(x,y)
\quad(x,y\in s),
\]

as well as the nonnegativity of \(M,L\). The domain can therefore be a
configuration region or a phase-space region with its specified distance.
No convexity, differentiability, or finite-dimensionality is required for
these particular estimates.

Suppose the regional input budgets are

\[
\rho:(R,L_\rho),\qquad u:(U,L_u),\qquad v:(V,L_v),
\qquad \rho(x)\ge\epsilon>0.
\]

Here \(u\) is the vector field used by the metric. Identifying it with an
actual ruler gradient is a separate input to the differential-mechanics
application; the algebraic metric bound itself does not assume that
identification. For fixed \(a\in\mathbb R\), set

\[
A=1+a^2U^2,
\qquad
G(x)v(x)=\rho(x)^{-2}
\bigl(v(x)+a^2\langle u(x),v(x)\rangle u(x)\bigr).
\]

`metricAction_boundedLipschitz` checks the budget produced by
`metricActionBudget`. Algebraically expanding that budget gives

\[
M_{Gv}=\frac{AV}{\epsilon^2},
\qquad
L_{Gv}=\frac{A L_v+2a^2UVL_u}{\epsilon^2}
       +\frac{2L_\rho AV}{\epsilon^3}.
\tag{1}
\]

The Lean interface stores the unsimplified budget operations, so subsequent
concrete instances can evaluate rational inputs directly with kernel-checked
arithmetic. The intermediate operations are

\[
(M,L)+(N,K)=(M+N,L+K),\qquad
(M,L)\cdot(N,K)=(MN,MK+NL),
\]

and the reciprocal budget above a positive floor is
\((\epsilon^{-1},L_\rho/\epsilon^2)\). Equation (1) includes variation of
both the ruler and the gradient field. It requires no unit-gradient bound.

## 2. The actual metric-normalized fixed direction

For a fixed Euclidean-unit vector \(e\), the checked identity is

\[
d(x)=\operatorname{metricNormalize}(\rho(x),a,u(x),e)
 =\frac{\rho(x)}{\sqrt{1+a^2\langle u(x),e\rangle^2}}\,e.
\tag{2}
\]

`metricNormalize_unit_formula` derives (2) from the original metric
quadratic form and its square root. The denominator is at least one. The
proof of the regional bound checks the scalar estimate

\[
|\sqrt{z}-\sqrt{w}|\le\tfrac12|z-w|\qquad(z,w\ge1)
\]

using the actual derivative and the mean value inequality. It then applies
the reciprocal estimate and the product budget. The resulting
`normalizedHeadingBudget` is

\[
\boxed{M_d=R,\qquad L_d=L_\rho+Ra^2UL_u.}
\tag{3}
\]

Only pointwise positivity of \(\rho\) is needed for this normalization
bound; its constants introduce no reciprocal regularization floor. The
uniform floor enters when applying \(G\) to the heading.

The input direction need not initially have Euclidean norm one.
`metricNormalize_pos_smul` proves invariance under any positive scalar
rescaling of the command. Applying it to \(\|e\|^{-1}e\) gives
`metricNormalize_boundedLipschitz` for every fixed \(e\ne0\), with exactly
the same budget (3). Thus the bound does not deteriorate when a fixed
command is represented by a small nonzero multiple of itself. A varying
command that approaches zero is a different regularity question and is
not inferred from this fixed-direction theorem.

## 3. Drive and drag budgets

For scalar amplitude \(F\) and heading field \(d\), the existing definition is

\[
Q_{\mathrm{drive}}=F\,Gd.
\]

If \(F\) has budget \((B_F,L_F)\), then
`driveForce_boundedLipschitz` proves

\[
M_{\mathrm{drive}}=B_FM_{Gd},\qquad
L_{\mathrm{drive}}=B_FL_{Gd}+M_{Gd}L_F.
\tag{4}
\]

The generic theorem accepts a certified heading field.
`normalizedDriveForce_boundedLipschitz` instead takes a fixed nonzero
direction and constructs that certificate from \(\rho,u\) using (3).
Its conclusion concerns the actual expression

\[
F(x)G(x)\operatorname{metricNormalize}(\rho(x),a,u(x),e),
\]

and its budget is `normalizedDriveForceBudget`. Both the heading and the
metric operator vary with configuration in this theorem.

For a viscous coefficient \(\beta\) with budget \((B_\beta,L_\beta)\),
`dragForce_boundedLipschitz` treats the actual isotropic covector

\[
Q_{\mathrm{drag}}=-\beta\,Gv
\]

and proves

\[
M_{\mathrm{drag}}=B_\beta M_{Gv},\qquad
L_{\mathrm{drag}}=B_\beta L_{Gv}+M_{Gv}L_\beta.
\tag{5}
\]

The norm estimates allow signed coefficients. Physical non-injection of
energy additionally uses \(\beta\ge0\), as in the earlier checked
dissipation contracts. The constants in (1), (3), (4), and (5) are sufficient
regional bounds; no optimality is claimed.

The existing `BoundedLipschitzOn.add` operation adds these certified
covectors to the actual potential-force bounds. Consequently the regional
\(Q_0,L_Q\) inputs can be computed from field and control data instead of
being assumed for the assembled force.

## 4. Correspondence to the current runtime

The current [controls module](../../runtime-core/src/controls.rs) uses a
world-aligned Cartesian direction, with intrinsic drive normalization.
On the nonzero direction branch, its documented formula is

\[
Q_{\mathrm{drive}}
 = \mathrm{throttle}\,\mathrm{MAX\_DRIVE\_FORCE}
   \frac{G e}{\sqrt{e^TGe}}.
\]

This agrees algebraically with (4) using the actual normalized heading.
The current friction coefficient combines base, grip, and brake terms;
the [manifold module](../../runtime-core/src/manifold.rs) implements
`drag_force` as \(-\beta Gv\). These are the control formulas addressed
here. Candidate anisotropic board friction and persistent heading/carve
controls are not promoted to the current runtime contract.

The formal results are exact-real field estimates. They do not certify
the executable's clamping, finite-value checks, direction deadzone,
floating-point normalization, coefficient rounding, or provider queries.
They also do not establish an evolution law for changing commands. A
time-dependent application must supply the corresponding regularity data
and use an appropriate time-dependent trajectory theorem; impulses and
switching require their own interfaces. Runtime parameters, behavior, and
provider contracts are unchanged.

## 5. Checked declarations

The module adds ten theorem declarations:

- `metricAction_boundedLipschitz`
- `driveForce_boundedLipschitz`
- `dragForce_boundedLipschitz`
- `sqrt_sub_le_of_one_le`
- `sqrt_boundedLipschitz_of_one_le`
- `metricNormalize_unit_formula`
- `metricNormalize_unit_boundedLipschitz`
- `metricNormalize_pos_smul`
- `metricNormalize_boundedLipschitz`
- `normalizedDriveForce_boundedLipschitz`

The target build `lake build FractalSync.ControlForceBounds --wfail`
passes with the repository's existing Lean and Mathlib pins. The module
introduces no placeholders, project axioms, or native-evaluation proofs.
