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
must not be interpreted as proof of convergence. Analytic-provider rollout
evidence, general anisotropic friction, and broader numerical acceptance remain
open; this measurement does not close issue #120.
