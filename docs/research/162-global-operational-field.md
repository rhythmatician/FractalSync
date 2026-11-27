# #162/#159: global real operational-field correspondence

The real evaluator selected by the operational provider's two integer floors
now has a checked global \(C^4\) theorem. Its actual first and second Fréchet
derivatives agree with the selected value/gradient/Hessian formulas across
cell edges and grid vertices. The proof calculates the adjacent-cell joins
from the six runtime polynomial coefficient lists and the coherent shift of
the underlying coefficient lattice. It does not assume those joins.

The same increment verifies the ideal full-halo index inequalities and removes
the previous same-cell restriction from real-query stability bounds under a
uniform coefficient bound. Machine floor, casts, masks, and IEEE arithmetic
remain outside these statements.

The [reference-fidelity continuation](162-operational-reference-fidelity.md)
now adds local reference value/gradient/Hessian error bounds, an explicit global
Hessian query gain, and combined reference/query/reported-evaluation budgets.
Its sample-accuracy and reference-regularity premises require separate evidence
for a concrete asset.

## 1. Sources and verification

The source algorithm is
[`runtime-core/src/operational_field.rs`](../../runtime-core/src/operational_field.rs).
This work extends the
[fixed-cell operational-field certificates](162-concrete-provider-certificates.md).

| New module | Explicit proved declarations | Main content |
| --- | ---: | --- |
| [OperationalFieldGluing.lean](../../formal/FractalSync/OperationalFieldGluing.lean) | 5 | Actual continuity and Fréchet derivative gluing for floor-selected cells |
| [OperationalFieldJoins.lean](../../formal/FractalSync/OperationalFieldJoins.lean) | 7 | Generic higher Horner derivatives, existing-jet identification, concrete endpoint joins through order four |
| [OperationalFieldLattice.lean](../../formal/FractalSync/OperationalFieldLattice.lean) | 4 | Coherent lattice patches, their actual derivatives, and both coordinate joins |
| [GlobalOperationalField.lean](../../formal/FractalSync/GlobalOperationalField.lean) | 11 | Literal global floor selection, actual mixed jets, actual Hessian and global C4 |
| [OperationalFieldQueryCorrespondence.lean](../../formal/FractalSync/OperationalFieldQueryCorrespondence.lean) | 17 | Physical normalization, query-patch correspondence, halo addresses and finite-asset extension independence |
| [OperationalFieldGlobalError.lean](../../formal/FractalSync/OperationalFieldGlobalError.lean) | 7 | Global operator bounds and real-query/coefficient errors across cell boundaries |

There are 51 explicit proved declarations, including supporting lemmas. All six
modules pass the pinned Lean package's targeted `lake build --wfail`. The full
package build and transitive axiom audit belong to the central verification
inventory. No runtime, asset-format, provider-contract, training, or dependency
pin changes are required.

## 2. The actual coherent lattice evaluator

Let \(C:\mathbb Z\times\mathbb Z\to\mathbb R\) be any real coefficient lattice.
No boundedness or decay assumption is needed for regularity: each query reads
only 36 entries. Let \(B_0,\ldots,B_5\) be the six quintic polynomials already
formalized from the runtime's coefficient rows, including their division by
120.

For a fixed unit cell \((k,l)\in\mathbb Z^2\), define the actual mixed patch
jet

\[
J^{n,m}_{k,l}(x,y)=
\sum_{j=0}^{5}\sum_{i=0}^{5}
C_{k+i-2,l+j-2}\,B_i^{(n)}(x-k)\,B_j^{(m)}(y-l).
\tag{1}
\]

The integer offsets in (1) reproduce the implementation's six-by-six stencil.
`latticeCoefficients` and `latticePatchJet` are these definitions. The global
selection is the literal mathematical floor evaluator

\[
G_{n,m}(x,y)=J^{n,m}_{\lfloor x\rfloor,\lfloor y\rfloor}(x,y),
\qquad \Phi=G_{0,0}.
\tag{2}
\]

There is no replacement by an unspecified smooth interpolant. The final
regularity theorem concerns (2) itself.

### Actual higher derivatives of the existing Horner polynomials

`hornerDerivative` extends the previous simultaneous Horner evaluator to an
arbitrary derivative order. If \(H_{p,n}(t)\) denotes order \(n\) for a remaining
coefficient list \(p\), prepending coefficient \(a\) gives

\[
H_{a::p,0}(t)=a+tH_{p,0}(t),
\]

\[
H_{a::p,n+1}(t)=(n+1)H_{p,n}(t)+tH_{p,n+1}(t).
\tag{3}
\]

The checked theorem `hasDerivAt_hornerDerivative` proves
\(H_{p,n}'=H_{p,n+1}\) for every finite list and every natural order. The
separate `hornerDerivative_low` theorem identifies orders zero, one and two
with the previous actual runtime-style Horner accumulators. Consequently the
higher-order proof is attached to the same six polynomials used by the
existing value and jet certificates.

## 3. Concrete joins through derivative order four

For every integer row \(c:\mathbb Z\to\mathbb R\), every integer \(k\), and
\(0\le n\le4\), `latticeLineJet_join` proves

\[
\boxed{
\sum_{i=0}^{5}c_{k+i-2}B_i^{(n)}(1)
=
\sum_{i=0}^{5}c_{k+1+i-2}B_i^{(n)}(0).
}
\tag{4}
\]

Both sides are evaluated from the actual coefficient lists by exact
arithmetic. The shift in the coefficient indices is essential. Two arbitrary,
unrelated 36-coefficient patches need not match.

Equation (4) lifts to the tensor patch. At a vertical cell boundary,

\[
J^{n,m}_{k,l}(k+1,y)=J^{n,m}_{k+1,l}(k+1,y),
\qquad 0\le n\le4,
\tag{5}
\]

for every natural \(m\). At a horizontal cell boundary,

\[
J^{n,m}_{k,l}(x,l+1)=J^{n,m}_{k,l+1}(x,l+1),
\qquad 0\le m\le4,
\tag{6}
\]

for every natural \(n\). These are `latticePatchJet_join_x` and
`latticePatchJet_join_y`. They include the simultaneous boundary values at
grid vertices.

### Why the floor does not introduce a derivative defect

The supporting `floorSelect` theorem considers an actual selection
\(x\mapsto f_{\lfloor p(x)\rfloor}(x)\), where \(p\) is a continuous real cell
coordinate. It proves continuity from adjacent value matching and proves an
actual Fréchet derivative from adjacent value and derivative matching.

At an integer boundary, the proof identifies the selected function on each
closed neighboring cell, proves its derivative within each cell, and joins
those two derivative statements on a neighborhood. It therefore covers the
boundary itself. Differentiability of floor is neither assumed nor needed.
The concrete application discharges all matching hypotheses using (4)–(6).

Applying this construction first in the first coordinate and then in the
second proves, for \(n<4\) and \(m<4\),

\[
\boxed{
dG_{n,m}(x,y)[u,v]
=G_{n+1,m}(x,y)u+G_{n,m+1}(x,y)v.
}
\tag{7}
\]

The mixed jets are continuous for \(n,m\le4\). Induction using Mathlib's
actual `ContDiff` derivative characterization then proves

\[
\boxed{\Phi\in C^4(\mathbb R^2).}
\tag{8}
\]

More generally, `contDiff_globalJet` states that \(G_{n,m}\) is \(C^r\) when
\(n+r\le4\) and \(m+r\le4\). This is the standard regularity expected of a
quintic cardinal spline, now checked for these concrete coefficient rows and
this exact selection formula. No global \(C^5\) claim is made.

## 4. Physical origin, spacing, and actual returned jets

For physical origin \(o\), positive spacing \(h\), and query \(z\), put

\[
q=\frac{z-o}{h},
\qquad
P_{n,m}(z)=h^{-(n+m)}G_{n,m}(q),
\qquad P=P_{0,0}.
\tag{9}
\]

`physicalJet` is (9). The coordinate chain rule is checked, yielding

\[
dP(z)[u,v]=P_{1,0}(z)u+P_{0,1}(z)v,
\tag{10}
\]

and the actual Hessian

\[
\boxed{
d^2P(z)=
\begin{pmatrix}
P_{2,0}(z)&P_{1,1}(z)\\
P_{1,1}(z)&P_{0,2}(z)
\end{pmatrix}.
}
\tag{11}
\]

The formal representation is a curried continuous bilinear map. Theorems
`fderiv_physicalField` and `fderiv_fderiv_physicalField` identify the actual
first and iterated Fréchet derivatives. `contDiff_physicalField` proves the
physical evaluator is \(C^4\).

For \(k=\lfloor q_x\rfloor\), \(l=\lfloor q_y\rfloor\), the selected physical
cell origin is

\[
o_{k,l}=o+h(k,l).
\]

Three checked correspondence theorems identify the global value, differential
and Hessian at \(z\) with the previous `field`, `fieldDifferential`, and
`fieldHessian` of the selected 36 coefficients at origin \(o_{k,l}\) and
spacing \(h\). The algebraic identity

\[
\frac{z_x-o_x}{h}-k
=\frac{z_x-(o_x+hk)}h
\]

is proved with \(h\ne0\). This connects the global theorem to the earlier
physical query formulas, including their first- and second-derivative spacing
factors.

## 5. Ideal finite-asset halo and address bounds

The source accepts an axis only when its normalized coordinate satisfies

\[
2\le q_x<N_x-3,
\qquad 2\le q_y<N_y-3.
\tag{12}
\]

The ideal fraction property is checked:

\[
q_x-\lfloor q_x\rfloor\in[0,1),
\]

and similarly for the second coordinate. `halo_index_bounds` proves from
(12), for every \(i,j\in\{0,\ldots,5\}\),

\[
0\le\lfloor q_x\rfloor+i-2<N_x,
\qquad
0\le\lfloor q_y\rfloor+j-2<N_y.
\tag{13}
\]

`halo_flattened_index_bounds` also checks the actual row-major integer address

\[
a=(\lfloor q_y\rfloor+j-2)N_x+\lfloor q_x\rfloor+i-2
\]

satisfies

\[
\boxed{0\le a<N_xN_y.}
\tag{14}
\]

These are mathematical integer bounds derived from the actual ideal halo
conditions. They do not certify a rounded coordinate, a machine cast or
machine integer multiplication.

The global lattice is a convenient mathematical representation of a finite
asset. `latticeCoefficients_eq_of_halo` proves that any two lattices agreeing
on the finite rectangle select the same 36 coefficients under (12).
`globalJet_eq_of_halo` then proves equality of every selected mixed jet,
independently of how the two lattices were extended beyond the asset. Thus
the infinite-lattice regularity proof does not require meaningful off-asset
reference-distance values.

## 6. Global query-error bounds across cell boundaries

The earlier coefficient-derived operator estimates now hold globally. Suppose

\[
|C_{ij}|\le B\quad\text{for all }i,j\in\mathbb Z,
\qquad B\ge0,
\qquad h>0.
\]

Coordinates use the product maximum norm, and differential norms use the
associated operator norm. The selected query always belongs to its physical
closed cell, so the earlier concrete polynomial estimates give

\[
\|dP(z)\|\le\frac{8B}{3h},
\qquad
\|d^2P(z)\|\le\frac{68B}{9h^2}
\qquad\text{for every real }z.
\tag{15}
\]

Since these are now actual global derivatives, the mean value theorem proves,
for arbitrary real queries \(z,z'\),

\[
\boxed{
|P(z')-P(z)|\le\frac{8B}{3h}\|z'-z\|_\infty,
}
\tag{16}
\]

\[
\boxed{
\|dP(z')-dP(z)\|\le\frac{68B}{9h^2}\|z'-z\|_\infty.
}
\tag{17}
\]

The two queries may lie in different cells, and their connecting segment may
cross any number of cell edges or vertices. The constants are certified upper
bounds; their optimality is not asserted.

For two lattices with \(|C_{ij}-D_{ij}|\le\eta\) everywhere and
\(|D_{ij}|\le B\), the new global combined bound is

\[
\boxed{
|P_C(z')-P_D(z)|
\le\eta+\frac{8B}{3h}\|z'-z\|_\infty.
}
\tag{18}
\]

This removes the same-cell premise of the prior combined real-query theorem.
For a finite asset, a bounded real extension can supply the global hypothesis;
the checked halo-independence theorem identifies its values and jets with
those of the finite coefficient table at valid queries. The runtime's
availability mask remains a separate condition for query success.

## 7. Remaining numerical boundary

| Obligation | Status |
| --- | --- |
| Exact coherent adjacent-cell joins through order four | Checked from the six runtime coefficient rows |
| Literal floor-selected real evaluator and physical C4 regularity | Checked, including edges and vertices |
| Actual first/second derivatives equal the selected real jet formulas | Checked |
| Ideal halo, coordinate indices and row-major address range | Checked under (12) |
| Finite-table independence from off-asset extension | Checked |
| Global real value/differential query stability under a uniform coefficient bound | Checked |
| Global Hessian query stability | Checked with gain `32 B/h^3` in the reference-fidelity continuation |
| Concrete reference value/gradient/Hessian bounds | Checked in that continuation from local sampled-value accuracy and actual reference jet regularity |
| Rust floating division, floor and casts agree with ideal selection | Not established |
| IEEE Horner, accumulation, spacing division and finite-value checks | Not established |
| Availability-mask and asset-decoding correctness | Not established |
| Stored coefficient fidelity to true signed Shore distance | Not established |

The operational field is still the field represented by its stored
coefficients. Its `reference-distance-unknown` contract remains appropriate.
The results verify the smoothness and exact-real calculus of that field.
Conditional reference-error theorems are now available, but numerical evidence
for true-distance sample fidelity and IEEE evaluation remains separate. Research issues
remain open, and runtime behavior and provider contracts are unchanged.
