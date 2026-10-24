# Coherent-field enclosure follow-up for #145

These are unshipped experiments on PR #148. Production remains at
geometry-provider/6, orbit-controller/10 and debug-snapshot/5. All quality
thresholds and both original crossing tests remain unchanged.

## Analytic uncertainty reaches the output gate

The previous candidate enclosed the global cardioid distance, but did not carry
that enclosure through the scalar quintic reconstruction and refinement weights.
The new candidate propagates value, gradient and Hessian intervals through those
same operations. The normalized analytic error contributes to the existing
output error gate. An uncertain retained dependency can disappear only when its
whole interval proves the refinement weight is on a constant plateau. A numerical
zero alone cannot close an uncertain tail.

Invalid intervals and nonfinite comparison jets cannot produce finite errors or
be cancelled by a zero multiplier. The positive reciprocal and square-root
operations reject underflowed denominators. Analytic component sampling returns
the computed value/enclosure pair directly, so optional cache insertion or
eviction cannot change enclosure availability.

The actual integrated candidate passes 53 targeted tests. These include forced
cache eviction, cache-disabled comparisons, non-affine scalar/gradient/Hessian
enclosure checks at several scales and cell boundaries, and the final output
error gate. Disconnecting the final analytic error contribution makes its
regression fail. Removing either nonfinite-value guard also makes its regression
fail. The standalone interval and automatic-differentiation module passes ten
tests. These checks establish the propagation mechanism; outside-component
samples still depend on the existing contour and fit certificates.

All twelve public derivative checks pass in 10.88 seconds. Center values,
gradients, Hessians and resolved scales match the preceding adaptive-prefix
candidate bit for bit. Estimated errors now include the analytic uncertainty.
Timings were measured with other jobs running and are not isolated benchmarks.

The affine interior proof tries longer bounded prefixes at 1024, 2048 and 4096
updates after the original 512-update attempt. Each accepted interior box still
requires a strict periodic disk trap. Exterior certification retains the original
pre-update escape cutoff below 512. Bounded prefixes alone do not certify an
interior box.

## Energy and crossing evidence

The earlier combined affine/field candidate now passes all twelve energy cases
across separate low-curvature, high-curvature, near-Shore and inward sweeps. The
full combined run had timed out with eleven completed cases; the separate sweeps
complete that earlier candidate's evidence. The unchanged energy limit is 0.05.
These results do not validate the newer interval-propagating candidate.

The newer candidate passes all twelve energy checks across the low-curvature,
near-Shore, high-curvature and inward sweeps. High-curvature completes in
1020.60 seconds and inward in 1169.48 seconds under concurrent load. Their
excursions and largest step changes exactly match the earlier affine candidate.
Terminal crossing-query results are recorded separately in the accompanying
JSON. Missing terminal results are not passes. The recorded training
point returns Regular and passes its scale/error gates in 79.14 seconds cold.
The recorded Python crossing query times out at 180 seconds. The next recorded
native point times out at 1200 seconds with profiling enabled.

Profiling the next recorded native-trajectory point at
`(0.32049988639807619, 0.04733531447355364)` identifies repeated full contour
searches for individual retained scalar nodes. For one level-21 node, finer
searches at levels 24, 25 and 26 take about 80, 147 and 452 seconds respectively.
The expensive search is followed by a membership-stability refusal, so repeated
refinement magnifies the cost.

An optional smaller-node experiment tries centered 17-, 33- and 57-node windows
before the existing search. It retains the four-cell uncertainty band and rejects
uncontained or unstable samples. Three independent review axes found no concrete
unsafe acceptance. However, smaller windows can stop membership refinement at a
different budget and change coefficients. This is a new field definition whose
crossing and energy behavior needs validation. It passes 54 targeted tests and
all twelve derivative checks; neither cost improvement nor crossing completion
has been established. Its isolated recorded native-point query also times out
at 1200 seconds.

The production diagnostic accepts `--dt` to select one of the existing case time
steps, for example `--case inward --dt 0.00125 --check`. With no filter it runs
the same full sweep. The selector changes neither the step definitions nor the
0.05 limit. The selected low-curvature row exactly matches the unfiltered report;
seven missing, invalid or unmatched requests are rejected.

## Reproduction and remaining acceptance

Start at `d8ec37a` and apply the eight patches listed in
[the certified-feature report](geometry-145-certified-features.md). Then apply
`geometry-145-adaptive-affine.patch` and `geometry-145-ranged-field.patch` from
`experiments/`. The optional `geometry-145-centered-nodes.patch` adds the smaller
node search. Both patch sequences were reconstructed in an isolated directory
and compared with their exact captured source snapshots.

Alternatively, apply `geometry-145-box-fifo.patch` after the ranged-field patch
instead of the centered-node patch. This memoizes the unchanged uniform-box
proof by all three exact input bit patterns. A retained `None` remains uncertain;
eviction or lock failure only causes recomputation. Its 4096-entry FIFO avoids
the linear hit-maintenance cost measured in the superseded optional
`geometry-145-box-cache.patch` LRU experiment. Both memo variants pass 55 targeted
tests. Their derivative checks preserve every serialized row from the uncached
candidate; full-query cost improvement is not yet established.
The FIFO candidate's original native crossing test subsequently times out at
1200 seconds, as does the LRU candidate's isolated recorded-point query.
The recorded training jet is also bit-for-bit unchanged with FIFO caching; its
cold query takes 70.91 seconds under concurrent load.

The standalone `experiments/geometry-145-box-fifo-bench.py` harness compiles the
actual applied proof and cache implementation. It checks repeated-update queue
bounds and FIFO eviction, verifies enabled/disabled result equality, and compares
cheap and expensive repeated-box timings. This diagnostic uses standard-library
`LazyLock` in place of `once_cell::Lazy`. In the first sample, 20,000 cheap hits
take 0.00067 seconds with FIFO versus 0.00097 without caching. Ten expensive
uncertain queries take 0.00017 versus 0.00165 seconds. These microbenchmarks do
not predict the full crossing-query cost. All alternative patch sequences were
also reconstructed and compared with their captured snapshots.

[The JSON report](geometry-145-ranged-field.json) preserves variant ownership,
terminal outcomes and mutation logs. Original native and Python crossings,
general training validity and live query cost remain merge blockers. Adoption
also requires the version, golden and binding updates and full verification of
the exact chosen field. No experimental field has been promoted to production.

The subsequent [budget-aware proof experiment](geometry-145-budget-boxes.md)
preserves the ordinary tile budget semantics but establishes no cost benefit
in its sampled difficult region.
