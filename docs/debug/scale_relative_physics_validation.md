# Scale-relative physics validation

## Adopted mechanics

The runtime now implements the destination geometry settled in
`docs/equations.md`:

\[
G(c)=\rho(c)^{-2}I+\lambda^2\nabla\sigma(c)\nabla\sigma(c)^{\mathsf T}.
\]

Kinetic energy, metric-unit controls, drag, and generalized-force conversion
all consume this same metric. Conservative, control, friction, and impulse
inputs remain covectors until the single `G^{-1}` conversion at the dynamics
seam.

The connection uses the full scale-relative expression from equation 66. The
older graph-only formula differentiated only the
`lambda^2 grad(sigma) grad(sigma)^T` term and omitted derivatives of the
`rho^-2 I` conformal factor. Those omitted terms are large precisely where the
local ruler is small, so retaining the old connection with the new metric would
not be Levi-Civita compatible.

The debug snapshot preserves `potential` and `potentialForce` as Shore-specific
`U_sigma` and `Q_sigma` fields because the cockpit uses them to decide whether
the trajectory crested the Shore ridge. Its `total` field includes
`K + U_sigma + U_wall`, and its reconstructed acceleration includes the wall
force, so both the energy ledger and dynamics still match integration.

## Verification evidence

- The Rust runtime library suite passes: 49 tests.
- Direct regression tests verify `G G^{-1} = I` and compare the compact
  connection with the defining metric-derivative equation component by
  component.
- The native `tour_antenna_mini` replay completed all 550 frames with finite
  state, no anomalous acceleration, and no hard-guard rejection. Its largest
  coordinate acceleration magnitude was 0.1120 at tick 51.
- A native Shore launch with kinetic energy 25 times the scalar ridge height
  crosses at timesteps 0.002, 0.001, and 0.0005. First-crossing times are
  4.466, 4.383, and 4.3435 seconds respectively; the refinement gap decreases
  from 0.083 to 0.0395 seconds. A launch at half the ridge height reflects.
- The controller contract version is `orbit-controller/5`, forcing consumers
  to reject stale golden vectors or mirrors.
- A compiled WASM-to-Python probe agreed exactly at `c = (0.4, 0.3)`,
  `(-0.7, 0.2)`, and `(0, 0)` for all four metric entries, all eight connection
  components, and total energy at `v = (0.002, -0.003)` under the shared test
  configuration. The full generated-vector preflight passed with maximum mirror
  error `3.725e-9`, and the frontend production build passed.

## Limits and remaining acceptance

Distance, gradient, and Hessian values come from a finite-resolution sampled
field. The regularized ruler keeps `rho` positive, but it does not remove
signed-distance cut loci or guarantee a globally smooth Hessian. Connection
diagnostics and trajectories near those locations still inherit the sampled
field's resolution and smoothness limits.

The current signed-distance field is a bilinear sample of a 1024 by 1024 grid
covering `x = [-2, 0.4711]` and `y = [-1.122, 1.122]`, with cells about
0.0022 to 0.0024 units wide and a derivative step near `9.14e-5`. Near the
crossing cusp, samples at `x = 0.2550`, `0.2555`, and `0.2560` produce scale
gradients about `4043`, `-3965`, and `-1376`, and Hessian components about
`6.37e6`, `-2.32e6`, and `3.30e6` under the crossing configuration
(`epsilon = 0.001`). These jumps are consistent with a field that is only
continuous across bilinear cell boundaries; ruler regularization does not make
that source field globally twice differentiable.

Conservative-energy error for the crossing launch does not converge under the
same timestep refinement. Relative error at first crossing is 6.24%, 11.79%,
and 14.68% for timesteps 0.002, 0.001, and 0.0005. The crossing-time evidence
therefore establishes that native crossing survives timestep refinement, but
it does not establish energy convergence through the sampled Shore Hessian.
That numerical defect remains open and should not be hidden behind a wider
energy tolerance. A smoother derivative authority is separate necessary work
before conservation through the Shore can become an acceptance criterion. No
local smoothing heuristic is adopted here because it would introduce another
physics authority without settling that model.

The native replay establishes numerical stability for the previously failing
antenna preset. It does not replace the eyes-on acceptance requested by issue
#120 for scale-relative travel or by issue #82 for the rendered shore-crossing
experience. Those checks remain after rebuilt Python and WASM artifacts consume
the new controller version.
