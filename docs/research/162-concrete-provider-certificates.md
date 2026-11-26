# #162/#159: concrete operational-field jets and error gains

The experimental operational provider now has checked exact-real value and jet
formulas for the quintic tensor patch actually represented by its 36 stored
coefficients. The proofs use the six coefficient lists and simultaneous Horner
recurrence in
[`runtime-core/src/operational_field.rs`](../../runtime-core/src/operational_field.rs).
They identify the patch's actual first and second Fréchet derivatives, quantify
its quadratic sampling bias, and derive coefficient and same-cell query error
bounds from the concrete basis. This closes a concrete analytic provider
obligation without asserting that the operational field is true signed Shore
distance or that Rust floating-point arithmetic is exact.

## 1. Formal sources and scope

| Source | Explicit proved theorem declarations | Main content |
| --- | ---: | --- |
| [OperationalField.lean](../../formal/FractalSync/OperationalField.lean) | 22 | Exact Horner derivatives, physical tensor-patch jets, basis moments and positivity |
| [OperationalFieldBounds.lean](../../formal/FractalSync/OperationalFieldBounds.lean) | 9 | Quadratic sampling bias, exact quadratic derivatives, monotonicity and value stability |
| [OperationalFieldJetError.lean](../../formal/FractalSync/OperationalFieldJetError.lean) | 16 | Derivative Bernstein tables and coefficient-to-jet gains |
| [OperationalFieldQueryError.lean](../../formal/FractalSync/OperationalFieldQueryError.lean) | 11 | Coefficient-derived operator bounds, real-query displacement and combined value errors |

These 58 declarations include supporting lemmas. All four modules pass the
pinned package's targeted `lake build --wfail`. The package-wide axiom audit
and complete build are recorded in the central verification inventory.

The object formalized here is one fixed cell's real polynomial, with an
arbitrary real table \(C=(C_{ij})_{0\le i,j<6}\). When comparing to the Rust
query, this table is the 36 stored values read by that query. The quadratic
sampling theorem has the additional, explicit premise that those values are
samples of a stated quadratic. It does not assert that every operational asset
contains quadratic samples.

## 2. The actual polynomial and simultaneous Horner jets

The runtime's ascending coefficient rows are reproduced exactly:

| Weight index | Coefficients before division by 120 |
| ---: | --- |
| 0 | \(1,-5,10,-10,5,-1\) |
| 1 | \(26,-50,20,20,-20,5\) |
| 2 | \(66,0,-60,0,30,-10\) |
| 3 | \(26,50,20,-20,-20,10\) |
| 4 | \(1,5,10,10,5,-5\) |
| 5 | \(0,0,0,0,0,1\) |

Let \(B_i(t)\) be the polynomial in row \(i\), divided by 120. The Lean
function `hornerJet` recursively evaluates a list of coefficients in the same
high-to-low order as the runtime loop. If the accumulators for the remaining
coefficients are \((v,d,e)\), prepending coefficient \(a\) gives

\[
(v,d,e)\longmapsto(a+tv,\ v+td,\ 2d+te).
\]

`hasDerivAt_hornerJet` proves, for every finite coefficient list, that the
second accumulator is the derivative of the first and the third is the
derivative of the second. Thus `basisFirst` and `basisSecond` are proved to be
\(B_i'\) and \(B_i''\), rather than assumed derivative annotations.

For physical spacing \(h>0\), a chosen cell origin \(o=(o_x,o_y)\), and real
query \(z=(x,y)\), set

\[
t_x=\frac{x-o_x}{h},\qquad t_y=\frac{y-o_y}{h},
\qquad
\Phi_C(z)=\sum_{j=0}^{5}\sum_{i=0}^{5}C_{ij}B_i(t_x)B_j(t_y).
\tag{1}
\]

`field` is this actual finite sum. The checked derivative identifications are

\[
\begin{aligned}
\partial_x\Phi_C(z)&=\frac1h\sum_{j,i}C_{ij}B_i'(t_x)B_j(t_y),\\
\partial_y\Phi_C(z)&=\frac1h\sum_{j,i}C_{ij}B_i(t_x)B_j'(t_y),\\
\partial_{xx}\Phi_C(z)&=\frac1{h^2}\sum_{j,i}C_{ij}B_i''(t_x)B_j(t_y),\\
\partial_{xy}\Phi_C(z)&=\frac1{h^2}\sum_{j,i}C_{ij}B_i'(t_x)B_j'(t_y),\\
\partial_{yy}\Phi_C(z)&=\frac1{h^2}\sum_{j,i}C_{ij}B_i(t_x)B_j''(t_y).
\end{aligned}
\tag{2}
\]

The main statements are `hasFDerivAt_field`,
`hasFDerivAt_fieldDifferential`, `fderiv_field`, and
`fderiv_fderiv_field`. They identify the actual differential and actual
iterated Fréchet derivative. `fieldDifferential_eq_query` and
`fieldHessian_eq_query` additionally put the spacing divisions outside the
finite sums, as in the runtime query. The Hessian's two mixed entries agree
because they are the computed derivative of the same real polynomial.

The real polynomial and these derivative identities are defined for all real
queries. The stability bounds below use the closed physical cell

\[
\mathcal C(o,h)=[o_x,o_x+h]\times[o_y,o_y+h],
\]

where \(t_x,t_y\in[0,1]\). Actual successful runtime queries use floor-selected
fractions in a half-open cell. The formal results include the closed-cell
endpoints, but do not yet prove how the global floor-selected patches join.

## 3. Positive weights, moments, and the exact quadratic bias

The basis has checked partition and moment identities for every real \(t\):

\[
\sum_i B_i(t)=1,\qquad
\sum_i(i-2)B_i(t)=t,\qquad
\sum_i(i-2)^2B_i(t)=t^2+\frac12.
\tag{3}
\]

The proof of nonnegativity on \([0,1]\) rewrites each weight in the degree-five
Bernstein basis. All the checked Bernstein control coefficients are
nonnegative. Consequently, \(B_i(t)\ge0\) on that interval, and the 36 tensor
weights form a convex combination on a cell.

The physical sample nodes are
\(x_i=o_x+h(i-2)\) and \(y_j=o_y+h(j-2)\). The physical second moment is

\[
\sum_i x_i^2B_i(t_x)=x^2+\frac{h^2}{2}.
\tag{4}
\]

For an arbitrary quadratic

\[
Q(x,y)=a x^2+bxy+c y^2+d x+e y+f,
\]

set the stored coefficient table to the raw samples
\(C_{ij}=Q(x_i,y_j)\). `field_quadratic` proves exactly

\[
\boxed{\Phi_C(z)=Q(z)+\frac{h^2}{2}(a+c).}
\tag{5}
\]

In particular, the operator reproduces affine fields exactly. For quadratics,
the value bias is spatially constant. The checked statements
`fieldDifferential_quadratic` and `fieldHessian_quadratic` prove

\[
\boxed{d\Phi_C=dQ,\qquad d^2\Phi_C=d^2Q.}
\tag{6}
\]

For the runtime test polynomial \(Q=x^2+2xy-3y^2\), (5) gives the observed
\(-h^2\) value bias. The general formula explains the source test and states
its exact derivative consequence. Raw sample coefficients are therefore not,
in general, interpolation coefficients: exact agreement at each sample node
is not a property of this construction.

## 4. Concrete coefficient-to-value and coefficient-to-jet gains

Let \(C,D\) be two real coefficient tables with

\[
|C_{ij}-D_{ij}|\le\eta\qquad\text{for all 36 entries}.
\tag{7}
\]

On the closed cell, positivity and partition imply the checked value bound

\[
\boxed{|\Phi_C(z)-\Phi_D(z)|\le\eta.}
\tag{8}
\]

The derivative bounds come from actual derivative polynomial coefficients.
`basisFirst_eq_bernstein` and `basisSecond_eq_bernstein` express the first and
second derivative weights in degree-four and degree-three Bernstein bases.
Their absolute control-column sums are at most 160 and 240 before division by
120. Bernstein nonnegativity and partition then give the checked estimates

\[
\sum_i|B_i'(t)|\le\frac43,\qquad
\sum_i|B_i''(t)|\le2,\qquad t\in[0,1].
\tag{9}
\]

Applying these inequalities to the tensor sums in (2) proves:

| Actual component | Uniform coefficient perturbation bound |
| --- | --- |
| Value | \(\eta\) |
| Each first coordinate derivative | \(4\eta/(3h)\) |
| Each pure second coordinate derivative | \(2\eta/h^2\) |
| Mixed second coordinate derivative | \(16\eta/(9h^2)\) |

The theorem names are `field_coefficient_error_bound`,
`fieldDifferential_coefficient_error_bound`, and
`fieldHessian_coefficient_error_bound`. These gains are derived from the
concrete six weight polynomials. They are certified upper bounds; optimality
of the derivative gains is not asserted.

Equation (7) can represent coefficient quantization, a perturbation between
assets, or a separately certified discrepancy between actual stored
coefficients and a reference coefficient table. Theorems (8) and (9) do not
supply that discrepancy automatically. In particular, they do not establish a
reference table of true signed-distance samples.

## 5. Same-cell query errors from coefficient magnitudes

There is also a concrete query-sensitivity certificate. Suppose
\(|C_{ij}|\le B\), with \(B\ge0\), for all stored coefficients. The new module
uses `Coords = ℝ × ℝ` with its product maximum norm,

\[
\|(x,y)\|_\infty=\max(|x|,|y|).
\]

Differential norms are the associated operator norms. The checked estimates
`norm_fieldDifferential_le` and `norm_fieldHessian_le` are

\[
\boxed{
\|d\Phi_C(z)\|\le\frac{8B}{3h},\qquad
\|d^2\Phi_C(z)\|\le\frac{68B}{9h^2},\qquad z\in\mathcal C(o,h).
}
\tag{10}
\]

The first coefficient is the sum of the two coordinate derivative bounds.
The Hessian coefficient is
\(2+2(16/9)+2=68/9\), from its four entries. These explicit bounds are useful
without an assumed provider Lipschitz constant.

Since the cell is convex and the patch has actual derivatives, Mathlib's mean
value theorem gives checked same-cell query bounds:

\[
\boxed{
|\Phi_C(z')-\Phi_C(z)|
\le\frac{8B}{3h}\|z'-z\|_\infty,
}
\tag{11}
\]

\[
\boxed{
\|d\Phi_C(z')-d\Phi_C(z)\|
\le\frac{68B}{9h^2}\|z'-z\|_\infty.
}
\tag{12}
\]

Finally, if \(C\) is a perturbed coefficient table, \(D\) is a reference
table satisfying \(|D_{ij}|\le B\), (7) holds, and both real queries remain
in this same cell, `field_combined_error_bound` proves

\[
\boxed{
|\Phi_C(z')-\Phi_D(z)|
\le\eta+\frac{8B}{3h}\|z'-z\|_\infty.
}
\tag{13}
\]

This gives the earlier
[provider stencil query budget](162-provider-stencil-certificates.md) a
coefficient-derived analytic gain for this concrete operational field. An
IEEE query-rounding analysis must still establish the real displacement
\(\|z'-z\|_\infty\), preserve the relevant cell and halo assumptions, and
add arithmetic evaluation error. Neither a displacement nor a successful
runtime query is inferred from the floating type alone.

## 6. Runtime mapping and remaining obligations

The source query takes a floor-selected cell, loads the 36 coefficients at
indices \((k_x+i-2,k_y+j-2)\), evaluates the two six-weight jets, and accumulates
the value and five derivative components. For comparison with (1), the fixed
patch origin is the global grid origin plus \(h(k_x,k_y)\), and \(C_{ij}\) is
the corresponding stored coefficient. These source expressions were inspected
against the Lean definitions. A kernel-checked refinement of Rust execution,
including integer indexing and floating arithmetic, has not been constructed.

The inspected provider bridge in
[`geometry_provider.rs`](../../runtime-core/src/geometry_provider.rs) consumes
the operational jets directly when that provider is selected. Its
`reference-distance-unknown` contract remains appropriate. The proofs here do
not change that contract.

| Obligation | Status after this increment |
| --- | --- |
| Fixed-cell real Horner value/first/second derivative coherence | Checked |
| Actual physical first and second Fréchet derivatives | Checked |
| Raw quadratic sample bias and exact quadratic jets | Checked under the explicit sampling premise |
| Coefficient perturbation gains for values and jets | Checked from the actual basis |
| Real-query value and differential variation | Checked here on a fixed cell; the [global continuation](162-global-operational-field.md) removes that restriction under a uniform lattice bound |
| Mathematical floor selection, coherent cell joins and global smoothness | Checked as global C4, with actual physical jets, in the continuation |
| Ideal full-halo integer indices, row-major bounds and off-asset extension independence | Checked in the continuation |
| Asset decoding, masks, machine indices, availability and query success | Not formalized |
| Numerical coefficient fidelity to true signed Shore distance | Not established |
| Query-rounding, subnormal, overflow and IEEE evaluation certificates | Not established |
| Hessian variation under query displacement | Checked globally with gain `32 B/h^3` in the [reference-fidelity continuation](162-operational-reference-fidelity.md) |
| Value/gradient/Hessian comparison with a reference field | Checked in that continuation from actual local reference jets, Hessian variation, and local sampled-value accuracy |

The [global join and indexing interface](162-global-operational-field.md) is
now checked for the literal mathematical floor-selected evaluator. It derives
the joins from these concrete coefficients and yields cross-cell query bounds.
The [reference-fidelity continuation](162-operational-reference-fidelity.md)
now derives concrete jet-error bounds from local sampled-value accuracy and
actual reference-field regularity, retaining the raw quadratic bias. The next
numerical obligations are evidence for those sample/regularity bounds for an
actual asset, actual query rounding, finite-asset/mask semantics and IEEE evaluation. Runtime behavior,
asset formats, training, canonical parameters and provider contracts are unchanged.
