# Issue #120 validation — open limitations (preserved, not hidden)

Per `docs/debug/scale_relative_physics_validation.md` and the existing
validation doc, the following numerical limitation remains OPEN and must
NOT be masked by a wider energy tolerance:

- Conservative-energy error for the Shore-crossing launch does NOT converge
  under timestep refinement. Relative error at first crossing: 6.24% (dt=0.002),
  11.79% (dt=0.001), 14.68% (dt=0.0005). Crossing-time evidence establishes
  native crossing survives refinement; it does NOT establish energy
  convergence through the sampled Shore Hessian.
- Source: sampled-field derivative quality near Shore / cut loci. The
  bilinear 1024x1024 grid (cell ~0.0022-0.0024, FD step ~9.14e-5) produces
  Hessian jumps (e.g. 6.37e6 / -2.32e6 / 3.30e6 at x=0.2550/0.2555/0.2560)
  consistent with a field only continuous across cell boundaries.
- Layer: geometry / Map derivative authority (NOT integrator). A smoother
  derivative authority is separate necessary work before conservation
  through the Shore can become an acceptance criterion.
- No local smoothing heuristic adopted — would introduce another physics
  authority without settling the model.

Phase 1 harness (`test_hyperbolic_validation_phase1.py`) measures SPD,
eigenvalues, condition number, and anisotropy but does NOT assert
convergence of energy through the Shore. Phase 2 (pending) will quantify
drift and identify dominant error sources explicitly.
