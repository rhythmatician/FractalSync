# Operational-field reference fidelity and complete real-query jet budgets

This increment adds **40 proved theorem/lemma declarations in four modules**:

| Module | Declarations | Scope |
| --- | ---: | --- |
| [OperationalFieldFidelity](../../formal/FractalSync/OperationalFieldFidelity.lean) | 21 | Actual Taylor polynomial, sample remainders, concrete value/differential/Hessian reference bounds |
| [OperationalFieldReference](../../formal/FractalSync/OperationalFieldReference.lean) | 9 | Physical lattice samples, selected-node accuracy, literal global first/second derivative accuracy |
| [OperationalFieldHessianError](../../formal/FractalSync/OperationalFieldHessianError.lean) | 9 | Concrete third-basis certificate, mixed-jet bounds and global Hessian query stability |
| [OperationalFieldComputedError](../../formal/FractalSync/OperationalFieldComputedError.lean) | 1 | Combined reference, actual-query displacement and reported-result error |

The earlier [global evaluator](162-global-operational-field.md) proved that the
actual mathematical floor-selected field is C4, with the query's exact-real
gradient and Hessian. The new result connects those objects to a reference
field under explicit local analytic and sample-fidelity certificates. It also
supplies the missing coefficient-derived Hessian query bound.

The runtime source remains
[`runtime-core/src/operational_field.rs`](../../runtime-core/src/operational_field.rs).
Its stored values are samples, and its six quintic polynomial rows, 36-term
tensor sum, floor selection and spacing factors are unchanged. The proof uses
the same real-arithmetic definitions as the existing provider modules. It does
not identify the Rust floating operations with those real operations.

## 1. Exact hypotheses and norms

Let \(P\) be the global operational field, with positive physical spacing
\(h\), lattice origin \(o\), and stored real coefficients \(C_{ij}\). At a
physical query \(q\), the mathematical floor selects one 6-by-6 coefficient
table. Its physical nodes are

\[
z_{ij}=o_q+h(i-2,j-2),\qquad 0\le i,j\le5,
\]

where \(o_q\) is the selected cell's physical origin. The proof verifies

\[
\|z_{ij}-q\|_\infty\le3h.
\tag{1}
\]

All coordinate vectors use the product maximum norm. Differential and Hessian
bounds use the corresponding induced operator norms. The trace below is
\(H(e_1,e_1)+H(e_2,e_2)\), the ordinary Cartesian Hessian trace. It is not the
Laplace--Beltrami operator of the scale metric.

For a reference field \(f\), the certificate is the existing
`ProviderCertificates.LocalJetControl f q S L`. It supplies:

- A convex region \(S\) containing \(q\), and actual ambient Fréchet first and
  second derivatives at every point of \(S\).
- The anchored Hessian variation bound
  \(\|H_f(z)-H_f(q)\|\le L\|z-q\|_\infty\) for \(z\in S\), with \(L\ge0\).
- Membership in \(S\) for all 36 selected physical nodes.

The differential and Hessian accuracy statements additionally require
\(S\) to be a neighborhood of \(q\). This lets the proof derive Hessian
symmetry and identify the supplied derivative fields with the literal
`fderiv` and iterated `fderiv` of \(f\). The value remainder theorem does not
need that neighborhood condition.

`SampleAccuracyAt C f o h q eta` requires only the **36 coefficients selected
at \(q\)** to satisfy

\[
|C_{ij}-f(z_{ij})|\le\eta.
\tag{2}
\]

It does not assume accuracy at every point of an infinite lattice. The
optional `sampleAccuracyAt_of_lattice` wrapper derives this local certificate
from a global sample bound. `sampledLattice` is the actual physical sampling
function, and `latticeCoefficients_sampledLattice` checks that its selected table
is exactly the pre-existing physical `sampledCoefficients` table.

These are sufficient certificates, not conclusions inferred from a provider's
regularity flags. For a signed-distance reference, they apply only in a region
where the stated actual derivatives and variation bounds have been established.
They do not assert smoothness across a fractal Shore or a distance cut locus.

## 2. From actual calculus to sample errors

The proof uses the literal Taylor polynomial

\[
T_q(z)=f(q)+df_q(z-q)+\tfrac12H_f(q)[z-q,z-q].
\tag{3}
\]

The existing actual-derivative Taylor theorem gives

\[
|f(z)-T_q(z)|\le\frac L6\|z-q\|_\infty^3.
\tag{4}
\]

Combining (1) and (4) proves every selected sample remainder is bounded by
\(\tfrac92 Lh^3\). With (2), the coefficient table differs from the sampled
Taylor polynomial by at most

\[
E=\eta+\frac92 Lh^3.
\tag{5}
\]

The exact quintic moment identities then identify what the operator does to
(3). For every evaluation point \(z\), it returns

\[
P_{T_q}(z)=T_q(z)+\frac{h^2}{4}\operatorname{tr}H_f(q).
\tag{6}
\]

Its differential and Hessian are exactly those of \(T_q\). Equation (6)
accounts for the raw-sample quadratic bias already observed in the earlier
[concrete provider note](162-concrete-provider-certificates.md). The proof
does not treat this quasi-interpolating spline as an interpolant.

The new code checks the Taylor polynomial's actual derivative, its expression
as the previously formalized general quadratic, and its exact value and jet
reproduction. The reference error follows by applying the concrete coefficient
sensitivity bounds to (5).

## 3. Checked reference accuracy

At the actual physical query, the following inequalities hold:

\[
\boxed{
\left|P(q)-f(q)-\frac{h^2}{4}\operatorname{tr}H_f(q)\right|
\le\eta+\frac92Lh^3.
}
\tag{7}
\]

\[
\boxed{
\|dP_q-df_q\|\le\frac{8\eta}{3h}+12Lh^2.
}
\tag{8}
\]

\[
\boxed{
\|H_P(q)-H_f(q)\|\le\frac{68\eta}{9h^2}+34Lh.
}
\tag{9}
\]

For a value bound without subtracting the bias, the proved trace estimate
\(|\operatorname{tr}H|\le2\|H\|\) yields

\[
\boxed{
|P(q)-f(q)|
\le\eta+\frac{h^2}{2}\|H_f(q)\|+\frac92Lh^3.
}
\tag{10}
\]

`physicalField_reference_actual_jet_error` packages (8)--(10) using the literal
`fderiv` and iterated `fderiv` of both \(P\) and \(f\). Thus the theorem is
about actual derivatives of the selected global field and reference function.
The derivative fields in the certificate are identified with these derivatives
in `reference_actual_hessian`; they are not free jet labels.

The hypotheses refer to the selected cell at the query. The query can lie on a
grid edge or vertex, because the earlier global C4 theorem already identifies
the selected jets there. No same-cell neighborhood restriction is imposed on
the global evaluator.

For exact samples, these estimates give a second-order uncorrected value
bound, second-order differential error and first-order Hessian error. The
bias-corrected value remainder is third order. These are direct readings of
the proved inequalities; a separate asymptotic convergence theorem is not
packaged in this increment. The constants are certified upper bounds, with
no optimality claim.

The noise amplification is explicit. A fixed sample error \(\eta\) does not
become harmless as \(h\) decreases: the differential term grows as
\(\eta/h\), and the Hessian term grows as \(\eta/h^2\). For example, substituting
\(\eta=\eta_0h^3\) preserves second-order differential and first-order Hessian
bounds. That substitution is a consequence of (8)--(9), not an assertion about
how a particular asset was generated.

## 4. Explicit global Hessian query stability

The new third-derivative certificate is computed from the actual six quintic
basis rows. Their degree-two Bernstein numerator table is

\[
\begin{pmatrix}
-60&0&0\\
120&-120&-60\\
0&360&120\\
-120&-360&0\\
60&120&-120\\
0&0&60
\end{pmatrix}.
\]

Each basis third derivative is its row's Bernstein polynomial divided by
120. The absolute column sums are at most 960, giving the checked bound

\[
\sum_{i=0}^5|B_i'''(t)|\le8,\qquad 0\le t\le1.
\tag{11}
\]

Combined with the previous constants
\(\kappa_0=1\), \(\kappa_1=4/3\), \(\kappa_2=2\), the new module proves

\[
|\partial_x^n\partial_y^mP(q)|
\le\frac{B\kappa_n\kappa_m}{h^{n+m}},
\qquad 0\le n,m\le3,
\tag{12}
\]

assuming \(|C_{ij}|\le B\) everywhere, \(B\ge0\), with \(\kappa_3=8\).
Here the mixed jets are the concrete `physicalJet` functions, whose actual
calculus was already established in the global evaluator. The `jetBound`
function is used only within the theorem's stated order range.

The actual derivatives of the second-order component functions, followed by
the mean value theorem, give

\[
\boxed{
\|H_P(q')-H_P(q)\|
\le\frac{32B}{h^3}\|q'-q\|_\infty.
}
\tag{13}
\]

This is `physicalHessian_query_error_bound`. It allows arbitrary real queries
and any number of crossed grid lines. Together with the previous value and
differential query estimates, the three gains are now

\[
K_0=\frac{8B}{3h},\qquad
K_1=\frac{68B}{9h^2},\qquad
K_2=\frac{32B}{h^3}.
\tag{14}
\]

For a finite asset, a bounded extension of its coefficients supplies the
infinite-lattice hypothesis. The existing halo-independence results identify
its values and jets with the finite asset at valid queries. These mathematical
extensions do not claim that the runtime can evaluate outside its declared
halo or through an unavailable coefficient.

## 5. Combined reference, actual-query and reported-result contract

Let \(q'\) be the real query actually supplied to the mathematical evaluator,
while \(q\) is the intended reference point. Let reported values
\(\widehat P,\widehat D,\widehat H\) satisfy the explicit arithmetic discrepancy
certificates

\[
|\widehat P-P(q')|\le e_0,\quad
\|\widehat D-dP_{q'}\|\le e_1,\quad
\|\widehat H-H_P(q')\|\le e_2.
\tag{15}
\]

Define \(\delta_q=\|q'-q\|_\infty\). The final theorem
`reported_reference_jet_error` proves all three bounds simultaneously:

\[
\boxed{
|\widehat P-f(q)|\le
 e_0+K_0\delta_q+
 \eta+\frac{h^2}{2}\|H_f(q)\|+\frac92Lh^3,
}
\tag{16}
\]

\[
\boxed{
\|\widehat D-df_q\|\le
 e_1+K_1\delta_q+\frac{8\eta}{3h}+12Lh^2,
}
\tag{17}
\]

\[
\boxed{
\|\widehat H-H_f(q)\|\le
 e_2+K_2\delta_q+\frac{68\eta}{9h^2}+34Lh.
}
\tag{18}
\]

The reference smoothness and sample certificate are needed only around
\(q\) and its 36 nodes. The global coefficient bound controls movement of the
operational field from \(q\) to \(q'\). In particular, the theorem does not
require \(q'\) to select the same cell.

The quantities in (15) remain hypotheses. A finite-value check is not a bound
on them. The exact-real query displacement in (16)--(18) likewise does not
certify the runtime's floating normalization, floor or casts. This combined
statement specifies what numerical evidence would suffice and how it would
propagate; it does not invent that evidence.

## 6. Remaining provider boundary

| Obligation | Status after this increment |
| --- | --- |
| Concrete raw samples to actual reference value/differential/Hessian errors | Checked under local derivative, variation and selected-node sample certificates |
| Known raw-sample quadratic bias | Included exactly in (7), and bounded in (10) |
| Literal floor-selected physical evaluator and actual `fderiv` comparison | Checked, including cell boundaries |
| Coefficient-derived global Hessian query stability | Checked with gain \(32B/h^3\) |
| Combined reference, displaced-query and reported-result budgets | Checked under the separate stated certificates |
| Actual asset's sample fidelity to signed Shore distance | Still requires evidence; no such asset certificate was supplied here |
| Numerical values of reference derivative/variation constants | Still require geometric or provider-specific evidence |
| Rust floating normalization, floor/casts, Horner and accumulation errors | Not proved by these real-arithmetic theorems |
| Availability mask, asset decoding and machine index semantics | Separate from this increment |

The `reference-distance-unknown` runtime contract remains appropriate for an
asset that lacks the necessary reference certificate. Runtime behavior,
training, canonical parameters and provider contracts are unchanged. Research
issues remain open and PR #164 remains a draft.
