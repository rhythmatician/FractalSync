# #145 full-stencil reconstruction experiment

PR #148 remains unready to merge. The experimental oracle now supplies all 81
nodes of the archived coherent field's 9-by-9 stencil at both difficult
high-period fixtures. A Rust quintic kernel differentiates one polynomial into
value, gradient and Hessian, with directed propagation of nodal uncertainty.
Production geometry, thresholds and original crossing tests are unchanged.

## From nine nodes to the actual stencil

The new `stencil` mode uses floor-aligned dyadic coordinates and offsets -4
through +4, matching the archived reconstruction's input layout. It retains
`2^-20` spacing and a `5e-8` request at native, and `2^-27` spacing and a `5e-10`
request at Python. The initial native run refuses at its first numerical
projection. A bounded set of twelve starting angles recovers that proposal
without changing the original local step, iteration, orientation or curvature
guards. An analytic period-two control exercises recovery while retaining the
cut-locus refusal.

The native root-radius-0.99 cover then refuses a demanded node clearance at its
depth limit. Radius 0.999 succeeds under the same proof and work limits. Python
retains root radius 0.9. Every final node still requires an independent winding,
separation, interval-width and proposed-value error check. Proposal search cannot
issue a certificate. Full-stencil mode constructs the shared batch cover once;
the older diagnostic's redundant anchor cover is omitted.

## One immutable nodal assignment

Two independently certified values for the same node can differ, so independent
cell construction alone does not establish a coherent field. `NodalField` owns
one immutable assignment and rejects duplicate coordinates with conflicting
values or uncertainty windows. Cells can only be obtained from that assignment.
Missing nodes refuse construction.

After checking an actual distance bracket, `Sample` stores the declared
`value +/- error` interval. It does not retain whichever narrower bracket happened
to certify the node. This makes the assigned value and uncertainty independent of
witness selection. A test certifies the same assignment using different exterior
witnesses, and separate tests reject shared-value and shared-error conflicts.
The numerical proposal policy still includes the fixture period and seed; this
is not yet a general runtime component-discovery rule.

## Coherent polynomial and its error scope

The kernel uses the archived quintic basis and the same center-minus-quarter-
Laplacian coefficient correction. Its six output components are analytic
derivatives of one scalar polynomial. Adjacent cells taken from the same nodal
assignment agree in value, gradient and Hessian at their shared knot.

Directed multiprecision intervals cover coordinate normalization, coefficient
construction, basis derivatives, spatial scaling and accumulation. Reported
uncertainty includes conversion to the returned binary64 components. An
independent 512-bit reconstruction uses a truncated-power spline formula and
nonquadratic nodal data with small uncertainties. It checks all six components
for multiple choices of enclosed node values.

These bounds describe the reconstruction from true **nodal** distances. They do
not bound its interpolation remainder against the true distance between nodes
or certify true-distance derivatives. The kernel has no authority to classify
geometry as Regular or to hide a cut locus. That classification, adaptive
cross-level error, and refinement-weight integration remain required.

## Recorded outcomes and reproduction

Both fixtures pass their full scalar stencil and the reconstructed polynomial's
neighboring value/gradient/Hessian checks. The first completed native kernel run
takes 443 seconds cold, including a 2,746-edge shared cover. Its derivative
errors are about `1.14e-10` for the gradient and `4.96e-11` for the
distance-scaled Hessian. Python completes in about 100 seconds in the final
immutable-assignment variant, with corresponding errors `2.79e-11` and
`1.89e-11`. Workloads were concurrent; these are observed costs, not controlled
speedup ratios. The final native immutable-assignment run completes in 478.81
seconds. Kernel-only evaluation takes 0.750 seconds for 1,000 native queries
and 0.871 seconds for 1,000 Python queries, excluding proposal search, escape
proofs and cover construction.

```text
component_probe tight-bracket-native .999999 128 .999 .05 5e-8 9.5367431640625e-7 stencil
component_probe tight-bracket-python .999999 128 .9 .05 5e-10 7.450580596923828e-9 stencil
```

The isolated crate passes 77 tests and catches all 68 mutations by assertion
failures. New mutations protect stencil identity, bounded proposal recovery,
sample/query and accuracy binding, immutable shared assignments, prefilter sign,
derivative factors, spatial scaling, signed interval multiplication and nodal
uncertainty propagation. Reviews found and prompted the shared-assignment fix;
subsequent specification, health and standards reviews found no material defect
in the intended bounded change. Terminal failures and successes remain distinct
in the accompanying JSON, with exact source and binary hashes.

The remaining merge work includes general discovery and cache-stable node policy,
true-distance and canonical finite-orbit acceptance, singularity semantics,
cross-level field integration, acceptable cold cost, and the original energy,
crossing, training, version/golden/parity, browser/WASM and CI gates. Full-stencil
success at these two interior fixtures does not satisfy those broader gates.
