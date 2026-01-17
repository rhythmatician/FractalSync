We already have Mandelbrot lobe geometry code and a curriculum/orbit engine, but the current implementation is incomplete: angular_velocity is stored but not actually used, and curriculum velocities are computed via finite differences in a way that does NOT reliably respect direction (cw vs ccw) or speed scaling.

GOAL
Implement a mathematically concrete Mandelbrot-orbit curriculum for Julia parameter c(t), where:
1) Orbits are defined by MandelbrotGeometry.lobe_point_at_angle(lobe, theta, s, sub_lobe)
2) Time parameterization uses theta(t) = theta0 + omega * t, with omega = angular_velocity (rad/s)
3) Target velocities are analytic: v(t) = dc/dt = omega * d/dtheta[lobe_point_at_angle(...)]
4) Clockwise orbits are truly clockwise (omega < 0 flips direction)
5) s can be <1 (interior), =1 (boundary), >1 (exterior). For "just beyond the edge" we use s ≈ 1.02 (or small exterior values like 1.15 for more dramatic motion).

FILES TO EDIT
- backend/src/mandelbrot_orbits.py (core implementation)
- backend/src/orbit_engine.py (if needed to use improved API)
- any tests that need updating (backend/tests/test_orbit_engine.py, backend/test_curriculum_coverage.py, etc.)

NON-NEGOTIABLE: Do NOT change the validated center/radius logic. Use the existing MandelbrotGeometry.period_n_bulb_center(...) and period_n_bulb_radius(...) exactly as implemented. If your changes break tests, fix your code, not the geometry.

CURRENT PROBLEM SYMPTOMS
In backend/src/mandelbrot_orbits.py, MandelbrotOrbit has an angular_velocity field, and presets create "..._cw" variants by passing angular_velocity=-1.0, but the sampling still walks theta from 0..2pi in increasing order, so cw is not actually honored. Also velocity magnitude does not scale reliably with |omega|.

WHAT TO IMPLEMENT (MANDATORY)

A) Add an analytic derivative helper:
Implement MandelbrotGeometry.lobe_tangent_at_angle(lobe, theta, s=1.0, sub_lobe=0) -> complex
This returns d/dtheta of the lobe point (NOT multiplied by omega).

Define it exactly:

1) For main cardioid (lobe == 1):
We use the exact parametric form:
  c(theta) = 0.5*exp(i*theta) - 0.25*exp(2*i*theta)
The derivative:
  c'(theta) = 0.5*i*exp(i*theta) - 0.5*i*exp(2*i*theta)
Because lobe_point_at_angle for lobe==1 returns s * c(theta), the derivative must also be scaled:
  d/dtheta [ s*c(theta) ] = s * c'(theta)

Implement using cmath.exp and complex arithmetic.

2) For circular bulbs (lobe >= 2):
We define:
  center = period_n_bulb_center(lobe, sub_lobe)
  radius = period_n_bulb_radius(lobe, sub_lobe)
  point(theta) = center + s * radius * (cos theta + i sin theta)
Derivative:
  point'(theta) = s * radius * (-sin theta + i cos theta)
Equivalently point'(theta) = i * s * radius * exp(i*theta)
Implement with math.sin/cos or cmath.exp; must match exactly.

B) Fix MandelbrotOrbit so omega actually matters:
Modify MandelbrotOrbit so it is a TIME-PARAMETRIZED orbit.

1) Store:
- lobe, sub_lobe, s (choose a fixed s for the orbit instance; do NOT vary s per-point if you want purely tangential velocities)
- angular_velocity omega (rad/s)
- optional theta0 offset (default 0)

2) Implement MandelbrotOrbit.sample(n_samples: int, t0: float = 0.0, dt: float = 1/60) -> np.ndarray shape (n_samples, 2)
This should sample positions at times:
  t_k = t0 + k*dt
  theta_k = theta0 + omega * t_k
and return [real(c(theta_k)), imag(c(theta_k))].

3) Implement MandelbrotOrbit.compute_velocities(n_samples: int, t0: float = 0.0, dt: float = 1/60) -> np.ndarray shape (n_samples, 2)
This MUST be analytic, not finite differences:
  v_k = dc/dt = omega * lobe_tangent_at_angle(lobe, theta_k, s, sub_lobe)
Return [real(v_k), imag(v_k)].

This guarantees:
- cw works (omega < 0)
- speed scales with |omega|
- velocities are smooth regardless of discretization

NOTE: If other code depends on the old API signature compute_velocities(n_samples, time_step=1.0), keep it as a wrapper:
- interpret time_step as dt
- keep defaults consistent (prefer dt=1/60 for stable physics training)

C) Keep “orbit point clouds” if you still need them:
If any part of the UI/debugging expects a static closed loop of n_points around 0..2pi, keep that as a separate method, e.g.:
  generate_loop_points(n_points: int) -> np.ndarray
but do NOT use it for curriculum velocity targets.

D) Update preset generation:
In generate_dynamic_curriculum_orbits() keep the orbit set, but ensure s values match these intentions:
- boundary: s = 1.02 (just outside)
- interior: s = 0.7–0.8
- exterior: s = 1.15 (more outside)
Also keep both ccw and cw variants by choosing omega positive vs negative.
Make sure sub_lobe enumeration stays correct by using MandelbrotGeometry.get_max_sub_lobe_for_period(period) and generating orbits for each sub_lobe where applicable.

E) Update curriculum sequencing:
In generate_curriculum_sequence(n_samples):
- choose a stable dt (1/60 or 1/120)
- concatenate positions and velocities from multiple orbits
- velocities must come from the analytic method above

F) Tests:
Update/add tests to prove:
1) cw vs ccw are opposites:
For the same lobe/sub_lobe/s and same |omega|, the velocity vectors should be negations:
  v_cw(t) ≈ -v_ccw(t)  (within numerical tolerance)
2) speed scales with |omega|:
Doubling |omega| doubles |v| at same theta (within tolerance)
3) for circular bulbs:
The velocity is tangential:
For bulbs, (c - center) dot v ≈ 0 (in R2 dot product sense)
4) for cardioid:
No NaNs, stable magnitude, continuous.

Where possible, place tests in backend/tests/test_orbit_engine.py or add a new test file, but keep the suite fast.

IMPORTANT IMPLEMENTATION CONSTRAINTS
- Do NOT rewrite the validated center/radius calculations.
- Do NOT change the mathematical meaning of lobe_point_at_angle other than adding sub_lobe passthrough and using the existing exact cardioid equation.
- The only structural change is: make orbit sampling time-based and velocities analytic.

DELIVERABLE
A clean implementation that:
- preserves the validated geometry
- makes angular_velocity actually control direction and speed
- produces curriculum velocities that are physically meaningful for the physics/training model
- passes the full backend test suite

After code changes, run all tests and fix failures.