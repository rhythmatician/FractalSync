# Work-energy ledger diagnostics

The ledger describes the existing semi-implicit Euler step. Power is sampled at
the start of the continuous step, and work is that power multiplied by the
step duration. The drive covector and metric drag are accounted separately.
Shore and wall forces are conservative: both potentials are included in
mechanical energy and neither is also counted as work. An impulse is applied
after the continuous step at its final position; its work is the kinetic-energy
change at that fixed position. The signed residual is the total energy change
less drive, friction, and impulse work. Time is seconds and work and energy
use the dimensionless mechanical units of the manifold model; power is those
units per second.

`valid` means both sampled endpoint geometry jets are regular and the sampled
quantities are finite. It does not certify the geometry between endpoints or
the accuracy of the raster bridge. A failed step invalidates the accumulated
ledger; a geometry-invalid step also invalidates it. A standalone snapshot
without a recorded step has no ledger. The accumulated residual is the signed
sum of step residuals, so cancellation is possible.

For diagnostic regression only, one resolved bridge sample at c=(0,0), with
v=(0.005,0.003), Q=(0.1,0.03), beta=0.2 and duration 0.1, produced residuals
-3.38038e-4, -1.68995e-4 and -8.44271e-5 for 10, 20 and 40 steps. The test
uses an absolute coarse bound of 5e-4 and requires each refinement to reduce
absolute residual by at least 40%. A scale-aware inspection threshold may use
`5e-4 + 1e-3 * max(|E_start|, |E_end|, |W_control| + |W_friction| + |W_impulse|)`;
this is a diagnostic threshold, not an acceptance gate for the Physics.

The bridge does not provide convergence everywhere: at c=(0.5,0.5) with the
same rollout, signed residuals for 10, 20, 40 and 80 steps were approximately
-1.46e-5, 1.61e-4, 2.48e-4 and 2.92e-4. The endpoint validity flag therefore
must not be interpreted as proof of convergence. General anisotropic friction
and broader numerical acceptance remain open; this measurement does not close
issue #120.

An independent test-only analytic plane D=x-0.25 supplies exact gradient and
Hessian to the same Physics step. With c=(1.4,0.2), v=(0.08,-0.03),
Q=(0.3,-0.1), beta=0.2, duration 0.4, and both Shore and wall potentials
active and changing, absolute accumulated residuals for 20, 40, and 80 steps
were 6.61613e-4, 3.33404e-4, and 1.67355e-4. The test requires a coarse
absolute bound of 0.01 and at least 40% reduction at each refinement. This
isolates integrator accounting from raster interpolation for this smooth field;
it does not establish fidelity of the bridge or acceptance across the Map.


The accumulated summary retains the first step (including its initial state,
configuration, and provider sample), start clock, signed sums and a persistent
configuration/provider consistency flag. A change and later reversion still
marks it inconsistent and invalid. A summary alone does not preserve each
step's controls or geometry; retain the snapshot stream for replay.

Raster ledger records carry the in-process installation generation. Each
replacement or clear of a loaded raster updates the stamp under the field's
write lock. A change across the continuous step or the Controls wrapper
invalidates that ledger. Different generations between accepted steps
invalidate accumulated provenance even if the sampled values at the splice
agree. A single initial installation from an empty slot is allowed. The stamp
identifies an installation within one process and is not a raster content hash.

A local release-mode microbenchmark on Windows at a repeated fixed point
(`cargo run --release -p runtime_core --example ledger_cost`) measured 1,000
pairs of provider queries at 6.80 ms and 1,000 complete steps at 14.11 ms before
reuse. This is a warm microbenchmark, not a production profile, and times vary.
The start ledger jet now also supplies the existing geodesic calculation,
eliminating one identical query. The end jet remains an additional query to
record endpoint validity and provenance; further reuse requires changing the
energy evaluation API while preserving provider and error semantics.
