# Issue #120 validation: open limitations

The historical experiment in `scale_relative_physics_validation.md` reports
first-crossing relative energy errors of 6.24%, 11.79%, and 14.68% at timesteps
0.002, 0.001, and 0.0005. Crossing-time refinement does not establish energy
convergence. These values were not reproduced in this correction.

Sampled-field derivative quality is the leading identified source; integrator
contribution remains to be isolated with a smooth analytic-field control using
the same integration kernel. Historical evidence describes a bilinear 1024x1024
grid and Hessian jumps of 6.37e6, -2.32e6, and 3.30e6 at x=0.2550, 0.2555,
and 0.2560. Current source uses bicubic sampling for fields at least 4x4.
Record sampler, field, and full configuration on rerun rather than treating
historical results as current measurements. No smoothing heuristic is adopted.

Phase 1-5 files are explicitly skipped pending tests, not measurement harnesses.
They compute no spectrum, anisotropy, energy rollouts, cross-scale results,
replay diagnostics, or runtime costs. Duplicate SPD checks were removed.
Existing evidence remains in `backend/tests/test_manifold_physics.py`, including
SPD, connection behavior, conservative rollouts, Shore crossing, refined crossing
times, and Rust/Python parity.

The project Python 3.13 venv imports the compiled manifold bindings successfully.
The next experiment needs a field-injection seam in the production Rust integrator
to compare a nonconstant smooth analytic field against the sampled Shore field
under timestep refinement. That experiment is not implemented. Issue #120 remains open.