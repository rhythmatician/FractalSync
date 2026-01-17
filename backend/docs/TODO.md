You are GitHub Copilot working inside the FractalSync repository (this repo only). Your job is to implement the *exact* Mandelbrot-lobe orbital system I describe below, and wire it into the existing FractalSync backend so the training curriculum and synthetic-orbit tooling can use it immediately.

⚠️ NON-NEGOTIABLE CONSTRAINTS
- Do NOT “improve”, “simplify”, or “approximate” the geometry. The centers/radii and special-case tangencies below are already tested/validated elsewhere. Copy them *exactly*.
- Keep existing public APIs in FractalSync working (OrbitEngine, get_preset_orbit/list_preset_names, generate_curriculum_sequence, PhysicsTrainer usage).
- Any refactor must preserve behavior for existing tests (pytest).
- If you need to add new tests, do it, but do not break the suite.

HIGH-LEVEL GOAL
FractalSync currently has a simplified/partial Mandelbrot geometry implementation in:
- backend/src/mandelbrot_orbits.py

Replace/upgrade it so:
1) Orbits are defined by **(lobe, sub_lobe, angular_velocity, s)** where:
   - lobe = period number (1=main cardioid, 2=period-2 bulb, 3=period-3 bulbs, etc.)
   - sub_lobe selects among multiple bulbs for a given period (0-based)
   - angular_velocity is radians/sec (positive ccw, negative cw)
   - s scales radial distance relative to the lobe boundary:
       s = 1.0 => boundary
       s < 1.0 => inside
       s > 1.0 => outside (often slightly outside is preferred)
2) Geometry functions provide **exact** centers + radii for bulbs and special tangency rules for the cascade.
3) Curriculum generation produces a rich set of orbits that include:
   - cardioid interior/boundary/exterior
   - period-2 boundary/interior/exterior
   - period-3 upper/lower/mini-mandelbrot
   - period-4 cascade bulb and the two primary bulbs
   - period-8 cascade + two primaries
   - higher periods using Euler totient-based substructure enumeration
4) OrbitEngine can list and sample these orbits (for synthetic trajectories).
5) PhysicsTrainer curriculum (backend/src/physics_trainer.py) gets more varied, more interesting motion targets.

======================================================================
PART A — IMPLEMENT THE EXACT GEOMETRY + PARAMETERIZATION
======================================================================

FILE TO MODIFY:
- backend/src/mandelbrot_orbits.py

Right now it contains a partial MandelbrotGeometry, plus MandelbrotOrbit, plus preset helpers and curriculum generator. You must upgrade MandelbrotGeometry and the orbit definition system to match the specification below.

You MUST add these EXACT definitions (copy verbatim) into backend/src/mandelbrot_orbits.py.
You can reorganize file structure, but keep exports working.

----------------------------------------------------------------------
A1) Add this dataclass exactly (verbatim)
----------------------------------------------------------------------

from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class LobeOrbitParams:
    """Parameters for orbiting around a Mandelbrot set lobe."""
    lobe: int  # Lobe number (1=main cardioid, 2=period-2 circle, etc.)
    sub_lobe: int  # Sub-lobe index for periods with multiple bulbs (0=primary)
    angular_velocity: float  # Radians per second (positive = counterclockwise)
    s: float  # Radius scaling factor (1.0 = edge, <1 = inside, >1 = outside)

    def __post_init__(self):
        """Validate parameters."""
        if self.lobe < 1:
            raise ValueError("Lobe number must be >= 1")
        if self.sub_lobe < 0:
            raise ValueError("Sub-lobe index must be >= 0")
        if self.s <= 0:
            raise ValueError("Radius scaling factor s must be positive")

----------------------------------------------------------------------
A2) Replace/implement MandelbrotGeometry EXACTLY as below (verbatim)
----------------------------------------------------------------------

IMPORTANT:
- Copy the functions and logic exactly.
- Do NOT change numeric constants.
- Do NOT “simplify”.
- This includes special-case tangency for period-4 and period-8 cascade bulbs.
- This includes Euler totient logic to enumerate sub-lobes for general n.

Paste the following class definition exactly:

class MandelbrotGeometry:
    """
    Mathematical functions for Mandelbrot set lobe geometry.
    Implements precise parametric equations for bulbs and cardioid.
    """

    @staticmethod
    def get_max_sub_lobe_for_period(period: int) -> int:
        """
        Get the maximum valid sub-lobe index for a given period.
        Returns the actual number of bulbs minus 1 (since we use 0-based indexing).
        Uses Euler's totient function φ(n) to count primary bulbs.
        """
        if period == 1:
            return 0
        elif period == 2:
            return 0
        elif period == 3:
            return 2  # Three period-3 bulbs (2 primaries + 1 mini-Mandelbrot)
        elif period == 4:
            return 2  # Three period-4 bulbs (1 cascade + 2 primaries)
        elif period == 8:
            return 2  # Three period-8 bulbs (1 cascade + 2 satellites)
        else:
            phi_n = MandelbrotGeometry._euler_totient(period)
            return phi_n - 1 if phi_n > 0 else 0

    @staticmethod
    def _euler_totient(n: int) -> int:
        """Compute Euler's totient function φ(n)."""
        if n <= 0:
            return 0
        result = n
        p = 2
        while p * p <= n:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                result -= result // p
            p += 1
        if n > 1:
            result -= result // n
        return result

    @staticmethod
    def _cardioid_boundary(theta: float) -> complex:
        """
        Cardioid boundary parametrization.
        C(θ) = (1/2)e^(iθ) - (1/4)e^(i2θ)
        """
        import cmath
        return 0.5 * cmath.exp(1j * theta) - 0.25 * cmath.exp(1j * 2 * theta)

    @staticmethod
    def _cardioid_outward_normal(theta: float) -> complex:
        """
        Outward unit normal to cardioid at angle theta.
        N(θ) = -i * T(θ) / |T(θ)| where T(θ) = C'(θ)
        """
        import cmath
        tangent = (1j / 2) * cmath.exp(1j * theta) - (1j / 2) * cmath.exp(1j * 2 * theta)
        if abs(tangent) < 1e-12:
            return complex(1, 0)
        return -1j * tangent / abs(tangent)

    @staticmethod
    def _iterate_g_and_dg(c: complex, n: int) -> Tuple[complex, complex]:
        """
        Compute g_n(c) and g'_n(c) using coupled recurrence.
        z_{k+1} = z_k^2 + c, z_0 = 0
        w_{k+1} = 2*z_k*w_k + 1, w_0 = 0
        """
        z = complex(0, 0)
        w = complex(0, 0)
        for _ in range(n):
            z_new = z * z + c
            w_new = 2 * z * w + 1
            z, w = z_new, w_new
        return z, w

    @staticmethod
    def _has_primitive_period(c: complex, n: int) -> bool:
        """Check if c gives primitive period n (f_c^(k)(0) ≠ 0 for all k < n)."""
        z = complex(0, 0)
        for k in range(1, n):
            z = z * z + c
            if abs(z) < 1e-12:
                return False
        return True

    @staticmethod
    def _newton_solve_g(
        c_initial: complex,
        period: int,
        max_iterations: int = 50,
        tolerance: float = 1e-14,
    ) -> Optional[complex]:
        """Newton solver for g_n(c) = 0."""
        c = c_initial
        for _ in range(max_iterations):
            g_n, g_prime_n = MandelbrotGeometry._iterate_g_and_dg(c, period)
            if abs(g_n) < tolerance:
                if MandelbrotGeometry._has_primitive_period(c, period):
                    return c
                else:
                    return None
            if abs(g_prime_n) < 1e-12:
                return None
            c = c - g_n / g_prime_n
        return None

    @staticmethod
    def _primary_center(p: int, q: int) -> complex:
        """
        Compute center of primary bulb with rotation number p/q using Newton solver.
        """
        import math
        theta = 2 * math.pi * p / q
        root = MandelbrotGeometry._cardioid_boundary(theta)
        normal = MandelbrotGeometry._cardioid_outward_normal(theta)
        delta = 1e-3
        c_initial = root + delta * normal
        center = MandelbrotGeometry._newton_solve_g(c_initial, q)
        if center is None:
            return root
        return center

    @staticmethod
    def _cardioid_root(p: int, q: int) -> complex:
        import math
        theta = 2 * math.pi * p / q
        return MandelbrotGeometry._cardioid_boundary(theta)

    @staticmethod
    def _primary_radius(p: int, q: int) -> float:
        center = MandelbrotGeometry._primary_center(p, q)
        root = MandelbrotGeometry._cardioid_root(p, q)
        return abs(center - root)

    @staticmethod
    def _period2_root(p: int, q: int) -> complex:
        import math
        phi = 2 * math.pi * p / q
        return complex(-1.0, 0.0) + 0.25 * complex(math.cos(phi), math.sin(phi))

    @staticmethod
    def _satellite2_center(p: int, q: int) -> complex:
        """
        Compute center for satellite bulb off the period-2 circle.
        Special case: For period-8 cascade (q=4), position tangent to period-4 cascade.
        """
        import math
        if q == 4:
            return MandelbrotGeometry._cascade8_tangent_center(p)

        root = MandelbrotGeometry._period2_root(p, q)
        nvec = complex(math.cos(2 * math.pi * p / q), math.sin(2 * math.pi * p / q))
        c_initial = root + 1e-3 * nvec
        center = MandelbrotGeometry._newton_solve_g(c_initial, 2 * q)
        if center is None:
            return root
        return center

    @staticmethod
    def _satellite2_radius(p: int, q: int) -> float:
        import math
        if q == 4:
            root = MandelbrotGeometry._period2_root(p, q)
            nvec = complex(math.cos(2 * math.pi * p / q), math.sin(2 * math.pi * p / q))
            c_initial = root + 1e-3 * nvec
            center = MandelbrotGeometry._newton_solve_g(c_initial, 2 * q)
            if center is None:
                center = root
            return abs(center - root)

        center = MandelbrotGeometry._satellite2_center(p, q)
        root = MandelbrotGeometry._period2_root(p, q)
        return abs(center - root)

    @staticmethod
    def _cascade8_tangent_center(p: int) -> complex:
        import math
        # Get period-4 cascade info (direct calculation to avoid recursion)
        p4_root = MandelbrotGeometry._period2_root(1, 2)
        p4_nvec = complex(math.cos(2 * math.pi * 1 / 2), math.sin(2 * math.pi * 1 / 2))
        p4_initial = p4_root + 1e-3 * p4_nvec
        p4_center = MandelbrotGeometry._newton_solve_g(p4_initial, 4)
        if p4_center is None:
            p4_center = p4_root
        p4_radius = abs(p4_center - p4_root)

        # Get original period-8 radius (from its proper root)
        p8_root = MandelbrotGeometry._period2_root(p, 4)
        p8_nvec = complex(math.cos(2 * math.pi * p / 4), math.sin(2 * math.pi * p / 4))
        p8_initial = p8_root + 1e-3 * p8_nvec
        p8_original_center = MandelbrotGeometry._newton_solve_g(p8_initial, 8)
        if p8_original_center is None:
            p8_original_center = p8_root
        p8_radius = abs(p8_original_center - p8_root)

        # Position P8 tangent to P4: center distance = sum of radii
        period2_center = complex(-1.0, 0.0)
        direction_vector = period2_center - p4_center
        direction_toward_p2 = direction_vector / abs(direction_vector)
        tangent_distance = p4_radius + p8_radius
        p8_center = p4_center + tangent_distance * direction_toward_p2
        return p8_center

    @staticmethod
    def main_cardioid_point(theta: float) -> complex:
        import math
        term1 = 0.5 * complex(math.cos(theta), math.sin(theta))
        term2 = 0.25 * complex(math.cos(2 * theta), math.sin(2 * theta))
        return term1 - term2

    @staticmethod
    def period_n_bulb_center(n: int, sub_lobe: int = 0) -> complex:
        import math
        if n == 1:
            return complex(0.0, 0.0)
        elif n == 2:
            return complex(-1.0, 0)
        elif n == 3:
            if sub_lobe == 0:
                return complex(-0.122, 0.745)
            elif sub_lobe == 1:
                return complex(-0.122, -0.745)
            elif sub_lobe == 2:
                return complex(-1.75, 0)
            else:
                return complex(-0.122, 0.745)
        elif n == 4:
            if sub_lobe == 0:
                r4 = MandelbrotGeometry._satellite2_radius(1, 2)
                c4_real = -1.0 - (0.25 + r4)
                return complex(c4_real, 0.0)
            elif sub_lobe == 1:
                return MandelbrotGeometry._primary_center(1, 4)
            elif sub_lobe == 2:
                return MandelbrotGeometry._primary_center(3, 4)
            else:
                r4 = MandelbrotGeometry._satellite2_radius(1, 2)
                c4_real = -1.0 - (0.25 + r4)
                return complex(c4_real, 0.0)
        elif n == 8:
            if sub_lobe == 0:
                r4 = MandelbrotGeometry._satellite2_radius(1, 2)
                r8 = MandelbrotGeometry._satellite2_radius(1, 4)
                c4_real = -1.0 - (0.25 + r4)
                c8_real = c4_real - (r4 + r8)
                return complex(c8_real, 0.0)
            elif sub_lobe == 1:
                return MandelbrotGeometry._primary_center(3, 8)
            elif sub_lobe == 2:
                return MandelbrotGeometry._primary_center(5, 8)
            else:
                r4 = MandelbrotGeometry._satellite2_radius(1, 2)
                r8 = MandelbrotGeometry._satellite2_radius(1, 4)
                c4_real = -1.0 - (0.25 + r4)
                c8_real = c4_real - (r4 + r8)
                return complex(c8_real, 0.0)
        else:
            valid_p = [p for p in range(1, n) if math.gcd(p, n) == 1]
            if sub_lobe < len(valid_p):
                p = valid_p[sub_lobe]
                return MandelbrotGeometry._primary_center(p, n)
            else:
                p = valid_p[0] if valid_p else 1
                return MandelbrotGeometry._primary_center(p, n)

    @staticmethod
    def period_n_bulb_radius(n: int, sub_lobe: int = 0) -> float:
        import math
        if n == 1:
            return 0.5
        elif n == 2:
            return 0.25

        if n == 4 and sub_lobe == 0:
            return MandelbrotGeometry._satellite2_radius(1, 2)
        elif n == 8 and sub_lobe == 0:
            return MandelbrotGeometry._satellite2_radius(1, 4)

        if n not in (2,) and sub_lobe >= 0:
            p_vals = [p for p in range(1, n) if math.gcd(p, n) == 1]
            p_vals.sort()
            if p_vals:
                p = p_vals[sub_lobe % len(p_vals)]
                return MandelbrotGeometry._primary_radius(p, n)

        return 0.25 / (n * n * 0.8)

    @staticmethod
    def lobe_point_at_angle(
        lobe: int, theta: float, s: float = 1.0, sub_lobe: int = 0
    ) -> complex:
        import math
        if lobe == 1:
            boundary_point = MandelbrotGeometry.main_cardioid_point(theta)
            return s * boundary_point
        else:
            center = MandelbrotGeometry.period_n_bulb_center(lobe, sub_lobe)
            radius = MandelbrotGeometry.period_n_bulb_radius(lobe, sub_lobe)
            boundary_offset = radius * complex(math.cos(theta), math.sin(theta))
            return center + s * boundary_offset

----------------------------------------------------------------------
A3) Keep the current FractalSync orbit sampling interface, but upgrade it
----------------------------------------------------------------------

FractalSync currently has a MandelbrotOrbit class used by:
- backend/src/orbit_engine.py
- backend/src/physics_trainer.py (via generate_curriculum_sequence)

You can keep MandelbrotOrbit, but it MUST be able to represent:
- lobe
- sub_lobe
- angular_velocity (or implicitly one revolution per orbit; your choice, but support preset speeds)
- s behavior (constant s OR range with modulation)

Strong preference:
- add fields: lobe:int, sub_lobe:int
- add a deterministic way to sample theta progression (0..2π) and s progression.

CRITICAL: “Orbits aren’t on the edge of lobes, but just beyond it”
So most *default/preset* orbits should use s slightly > 1 (example: 1.02–1.15),
but ALSO include some s<1 presets because those are desirable too.

Implement MandelbrotOrbit.sample(n_samples) so it uses:
MandelbrotGeometry.lobe_point_at_angle(lobe, theta, s, sub_lobe)

If your MandelbrotOrbit currently stores center+radius directly, you can keep that for convenience, but the source of truth is MandelbrotGeometry.

Velocities:
- Existing code uses finite differences. That’s acceptable, but if you can do analytic velocities cleanly, do it.
- If analytic:
    For circle bulbs: c = center + s*r*(cosθ + i sinθ), θ = ωt
      dc/dt = s*r*ω*(-sinθ + i cosθ)  (+ optional ds/dt term if s varies)
    For cardioid: c(θ)=0.5e^{iθ}-0.25e^{i2θ}
      dc/dθ = 0.5 i e^{iθ} - 0.5 i e^{i2θ}
      dc/dt = dc/dθ * ω  (+ optional scaling by s if you apply s multiplicatively)
If you keep finite differences:
- ensure it’s smooth (use central differences, and use the same theta schedule as sampling)

======================================================================
PART B — PRESETS: MAKE THE ORBIT LIBRARY RICH AND EXPLICIT
======================================================================

FractalSync currently exposes:
- list_preset_names()
- get_preset_orbit(name)

Upgrade these so presets cover lobes/sub-lobes and s choices.

Implement a preset list similar to this (names can be slightly different, but keep existing ones working):
- "cardioid_boundary"  -> lobe=1 sub=0 s ~ 1.02 (slightly outside), medium speed
- "cardioid_interior"  -> lobe=1 sub=0 s ~ 0.8
- "cardioid_exterior"  -> lobe=1 sub=0 s ~ 1.15
- "period2_boundary"   -> lobe=2 sub=0 s ~ 1.02
- "period2_interior"   -> lobe=2 sub=0 s ~ 0.7
- "period2_exterior"   -> lobe=2 sub=0 s ~ 1.15

Period-3 must include:
- "period3_upper"      -> lobe=3 sub=0
- "period3_lower"      -> lobe=3 sub=1
- "period3_mini"       -> lobe=3 sub=2

Period-4 must include:
- "period4_cascade"    -> lobe=4 sub=0  (this uses the tangency center definition above)
- "period4_primary_1"  -> lobe=4 sub=1  (p/q=1/4)
- "period4_primary_2"  -> lobe=4 sub=2  (p/q=3/4)

Period-8 must include:
- "period8_cascade"    -> lobe=8 sub=0  (tangency chain)
- "period8_primary_1"  -> lobe=8 sub=1
- "period8_primary_2"  -> lobe=8 sub=2

Higher periods:
- Provide at least period 5 and 6 (or 7) presets using totient enumeration:
  For period n, sub_lobe indexes correspond to the sorted list of p with gcd(p,n)=1.
- Add both clockwise and counterclockwise variants for at least cardioid and period-2.

Also:
- Keep backward compatibility: if current tests reference "period3_boundary" etc, keep that name mapped to something sensible (ex: period3_upper boundary).

======================================================================
PART C — CURRICULUM GENERATION: USE THE FULL GEOMETRY
======================================================================

FUNCTION TO UPGRADE:
- generate_curriculum_sequence(...) in backend/src/mandelbrot_orbits.py

Currently it produces a small set of orbit_defs and phases with s_range ~ (0.95, 1.05).
Upgrade it so:
1) Default curriculum is more interesting and matches the stated “just beyond the edge” preference.
2) It includes sub-lobes and multiple periods.
3) It includes some interior and some exterior.

Concrete requirements:
- Use the new MandelbrotGeometry.get_max_sub_lobe_for_period(period) to enumerate sub-lobes for each period you include.
- Curriculum should contain a blend like:
  Phase 0 (stability / easy):
    - cardioid_interior (s=0.8)
    - cardioid_boundary-ish (s=1.02)
    - period2_boundary-ish (s=1.02)
  Phase 1 (variety):
    - cardioid_exterior (s=1.15)
    - period2_interior (s=0.7)
    - period2_exterior (s=1.15)
    - period3_upper/lower (s=1.02)
  Phase 2+ (complex):
    - period3_mini (s=1.02)
    - period4_cascade + primaries (s=1.02)
    - period8_cascade + primaries (s=1.02)
    - a couple higher periods (5, 6, 7) using totient enumeration; pick a small subset to keep dataset size reasonable.
- Ensure both positive and negative angular velocities appear in the curriculum.

Return format must remain compatible with existing uses in:
- backend/src/physics_trainer.py
It currently expects:
curriculum_positions, curriculum_velocities, curriculum_phases = generate_curriculum_sequence(...)
Make sure those still work, and metadata is still meaningful.

======================================================================
PART D — WIRING / INTEGRATION POINTS
======================================================================

D1) OrbitEngine (backend/src/orbit_engine.py)
- It imports MandelbrotOrbit, MandelbrotGeometry, get_preset_orbit, list_preset_names.
- After your changes, ensure:
  - list_preset_names returns the expanded set
  - get_preset_orbit supports existing names used in tests + the new names
  - orbit.sample and orbit.compute_velocities work and return (n,2) float32 arrays.

D2) PhysicsTrainer (backend/src/physics_trainer.py)
- It uses generate_curriculum_sequence() to create targets.
- Your upgraded curriculum should increase diversity without breaking training loop.
- Keep sequence lengths reasonable; do not blow memory.

======================================================================
PART E — TESTS: ADD “GEOMETRY SANITY” WITHOUT BREAKING SUITE
======================================================================

Add a new test file:
- backend/tests/test_mandelbrot_geometry.py

Write tests that validate *properties* (not the entire Newton internals), e.g.:

1) Period-2 exactness:
   - center == (-1+0j)
   - radius == 0.25

2) Period-3 hardcoded centers:
   - sub0 == (-0.122 + 0.745j)
   - sub1 == (-0.122 - 0.745j)
   - sub2 == (-1.75 + 0j)

3) Period-4 cascade tangency property:
   - r4 = period_n_bulb_radius(4,0)
   - c4 = period_n_bulb_center(4,0)
   - Assert abs((c4 - (-1+0j))) is approximately (0.25 + r4) (within small tolerance)

4) lobe_point_at_angle for bulb is on the boundary at s=1:
   - pick lobe=2, theta=0, sub=0:
     expected = center + radius*(1+0j)
     assert equality within tolerance

5) cardioid function is consistent:
   - main_cardioid_point(0) should equal 0.25 + 0j (since 0.5 - 0.25)

Also add a test ensuring get_max_sub_lobe_for_period matches the specified special cases.

Then run the full test suite.

======================================================================
PART F — DELIVERABLES CHECKLIST
======================================================================

When done, make sure:
- backend/src/mandelbrot_orbits.py contains:
  - LobeOrbitParams dataclass
  - MandelbrotGeometry exactly as specified above
  - Presets expanded and backward compatible
  - generate_curriculum_sequence expanded
- OrbitEngine tests still pass.
- New geometry tests pass.
- pytest passes.

If you need to rename/reorganize internal helpers, do it, but preserve imports used elsewhere.
No silent behavior changes that reduce orbit diversity or revert to “toy” geometry.

GO DO IT.