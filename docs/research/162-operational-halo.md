# #162: physical halo margins and finite-asset mechanics

[OperationalHalo.lean](../../formal/FractalSync/OperationalHalo.lean) connects
the actual physical query domain to explicit displacement margins and finite
coefficient assets. It proves valid mathematical stencil indices, equality of
query jets for arbitrary lattice extensions, and equality of the actual
regularized metric jets and mechanical field inside the open halo.

The module contains eleven theorem declarations. It builds on the
[global operational field](162-global-operational-field.md) and the actual
operational dynamics defined in the
[local-trajectory package](159-local-trajectories.md).

## 1. The actual half-open halo and its physical margin

For an asset of dimensions \(N_x,N_y\), origin \(o\), and spacing \(h>0\),
write

\[
n(q)=\left(\frac{q_x-o_x}{h},\frac{q_y-o_y}{h}\right).
\]

The definition `physicalHalo` is exactly

\[
2\le n_x(q)<N_x-3,
\qquad 2\le n_y(q)<N_y-3.
\tag{1}
\]

The lower faces are included, while the upper faces are excluded. The
definition `openPhysicalHalo` replaces the two lower inequalities with strict
inequalities. It is the open set used for neighborhood and derivative
arguments. A query on a lower face can satisfy (1) without belonging to this
open set.

The four-sided physical margin is the actual minimum

\[
m(q)=\min\left\{
q_x-(o_x+2h),\;
o_x+h(N_x-3)-q_x,\;
q_y-(o_y+2h),\;
o_y+h(N_y-3)-q_y
\right\}.
\tag{2}
\]

`closedBall_subset_openPhysicalHalo` proves

\[
R<m(q_0)
\quad\Longrightarrow\quad
\overline B_\infty(q_0,R)\subset\operatorname{openPhysicalHalo}.
\tag{3}
\]

`euclidean_closedBall_subset_openPhysicalHalo` proves the same inclusion for
an actual Euclidean configuration ball, transported through the global chart.
The proof explicitly uses
\(\|\operatorname{chart}(x)\|_\infty\le\|x\|_2\); it does not identify the
coordinate maximum norm with the Euclidean norm.

For a phase state \(z=(x,v)\), the phase norm is
\(\|z\|=\max(\|x\|_2,\|v\|_2)\). Suppose

\[
z\in\overline B(p,R),
\qquad \|q_{\rm reported}-\operatorname{chart}(x)\|_\infty\le\delta.
\]

`reportedQuery_mem_of_phaseBall` proves

\[
\boxed{
R+\delta<m(\operatorname{chart}(p_x))
\quad\Longrightarrow\quad
q_{\rm reported}\in\operatorname{openPhysicalHalo}.
}
\tag{4}
\]

The phase displacement and reported-coordinate displacement add before the
comparison with the physical margin. The reported query may cross one or
many spline-cell boundaries. The theorem uses its actual final coordinates,
without requiring it to remain in the initial cell.

## 2. The complete stencil stays inside the finite asset

For each \(i,j\in\{0,\ldots,5\}\), the mathematical query addresses are

\[
k_x=\lfloor n_x(q)\rfloor+i-2,
\qquad k_y=\lfloor n_y(q)\rfloor+j-2.
\]

`stencil_indices_of_mem_halo` proves, under the actual half-open domain (1),

\[
0\le k_x<N_x,\qquad 0\le k_y<N_y,
\]

\[
\boxed{0\le k_yN_x+k_x<N_xN_y.}
\tag{5}
\]

This is an exact-real normalization followed by mathematical integer floor.
It covers all 36 selected coefficients. It does not certify machine division,
machine floor, casts, integer overflow, masks, or a particular memory layout
implementation.

## 3. A bounded extension requires bounds only on stored coefficients

`AgreeOnAsset C D Nx Ny` states equality of the two coefficient lattices only
on the stored integer rectangle. The explicit `zeroExtension` is

\[
\widetilde C_{ij}=
\begin{cases}
C_{ij},&0\le i<N_x,\quad0\le j<N_y,\\
0,&\text{otherwise}.
\end{cases}
\]

`zeroExtension_agrees` proves finite-asset agreement. If \(B\ge0\) and only
the finite asset satisfies \(|C_{ij}|\le B\), `abs_zeroExtension_le` proves
the global bound

\[
|\widetilde C_{ij}|\le B\qquad(i,j\in\mathbb Z).
\tag{6}
\]

No bound is imposed on the original lattice outside the asset. This supplies
one bounded representative to which global spline regularity and magnitude
estimates can be applied. The independence results below transfer those
actual local fields back to any other extension of the same asset.

## 4. Query jets and actual metric jets are extension independent

`physicalJet_eq_of_mem_halo` proves equality of every selected physical mixed
jet of two agreeing finite assets at every query satisfying (1). Its inputs
do not fix the selected cell. The theorem therefore remains applicable after
a cell crossing.

On the open halo, the equality of actual scalar fields holds on a neighborhood.
For the Euclidean ruler

\[
\rho_C(x)=\sqrt{\epsilon^2+P_C(\operatorname{chart}x)^2},
\]

`rulerField_eventuallyEq` proves equality of the actual ruler germs for
\(C,D\) that agree on the finite asset. `actual_metricJets_eq` then proves

\[
\boxed{
\rho_C(x)=\rho_D(x),\qquad
\nabla\rho_C(x)=\nabla\rho_D(x),\qquad
D(\nabla\rho_C)(x)=D(\nabla\rho_D)(x).
}
\tag{7}
\]

The gradient is Mathlib's actual Riesz gradient, and its derivative is the
actual Fréchet derivative. The proof differentiates neighborhood equality;
it does not assume that nominal provider gradient/Hessian fields have the
desired identities. The equality theorem itself does not need positive
\(\epsilon\); positive regularization remains the hypothesis used for the
smooth mechanical existence and quantitative bound results.

For the same supplied force field, `acceleration_eventuallyEq` proves that
the actual operational mechanical accelerations also agree on a neighborhood
of every phase point whose configuration belongs to the open halo. Thus
arbitrary off-asset coefficients cannot change the local mechanical field
there. If the force field is itself changed between the two models, this
theorem does not infer its equality.

The query-displacement theorem (4) establishes domain validity of a reported
query. It does not equate a field value at the reported point with its value
at the exact point. That difference belongs to the separate field-fidelity
and rounding-error estimates.

## 5. Integration and remaining boundary

The [quantitative dynamics theorem](159-quantitative-dynamics.md) supplies an
actual local solution and continuous-interpolant domain control on a specified
phase ball. This module supplies the independent finite-asset domain condition
for that ball and displaced queries. It also removes the need to bound an
arbitrary outside-asset extension when deriving local operational constants.

The following remain separate from this module: machine arithmetic and address
evaluation, a certified numerical value for a chosen query-displacement
budget, runtime force fidelity, and a concrete runtime integrator's error
contracts. These exact-real theorems do not claim that the operational spline
equals ideal signed Shore distance.

`lake build --wfail FractalSync.OperationalHalo` passed with 2,631 jobs and no
warnings or errors. The complete build and unchanged transitive axiom audit
are recorded in [verification.md](verification.md). No runtime behavior,
provider contract, canonical parameter, training path, or dependency pin is
changed.
