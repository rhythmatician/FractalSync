"""
Mandelbrot set orbital trajectories for curriculum learning.

Uses mathematically precise geometry to generate orbits around the lobes and bulbs
of the Mandelbrot set, ensuring coverage of both real and imaginary axes.

This module implements exact bulb centers, radii, and tangency rules for:
- Main cardioid (period-1)
- Period-2 bulb
- Period-3 bulbs (upper, lower, mini-Mandelbrot)
- Period-4 cascade + primary bulbs
- Period-8 cascade + satellites
- Higher periods using Euler totient enumeration
"""

import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional


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


class MandelbrotOrbit:
    """Defines a trajectory through complex parameter space based on Mandelbrot geometry."""

    def __init__(
        self,
        name: str,
        lobe: int,
        sub_lobe: int = 0,
        n_points: int = 50,
        s_range: Tuple[float, float] = (1.02, 1.02),
        angular_velocity: float = 1.0,
    ):
        """
        Initialize an orbit around a specific Mandelbrot lobe.

        Args:
            name: Orbit name/identifier
            lobe: Period number (1=main cardioid, 2=period-2 bulb, etc.)
            sub_lobe: Sub-lobe index for periods with multiple bulbs
            n_points: Number of points to sample around the orbit
            s_range: (s_min, s_max) for radius scaling variations
            angular_velocity: Radians per second (positive=ccw, negative=cw)
        """
        self.name = name
        self.lobe = lobe
        self.sub_lobe = sub_lobe
        self.n_points = n_points
        self.s_range = s_range
        self.angular_velocity = angular_velocity

        # Generate points dynamically
        self.points = self._generate_points()

    def _generate_points(self) -> np.ndarray:
        """Generate orbit points using mathematical geometry."""
        angles = np.linspace(0, 2 * np.pi, self.n_points, endpoint=False)
        s_values = np.linspace(self.s_range[0], self.s_range[1], self.n_points)

        points: List[List[float]] = []
        for angle, s in zip(angles, s_values):
            c = MandelbrotGeometry.lobe_point_at_angle(
                self.lobe, float(angle), float(s), self.sub_lobe
            )
            points.append([c.real, c.imag])

        return np.array(points, dtype=np.float32)

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Sample points along the orbit.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Array of shape (n_samples, 2) with [real, imag] coordinates
        """
        if len(self.points) == 0:
            return np.zeros((n_samples, 2), dtype=np.float32)

        # Create interpolation parameter
        t = np.linspace(0, len(self.points) - 1, n_samples)

        # Interpolate along orbit
        real = np.interp(t, np.arange(len(self.points)), self.points[:, 0])
        imag = np.interp(t, np.arange(len(self.points)), self.points[:, 1])

        # Close the loop: ensure first and last match
        real[-1] = self.points[0, 0]
        imag[-1] = self.points[0, 1]

        return np.stack([real, imag], axis=1).astype(np.float32)

    def compute_velocities(self, n_samples: int, time_step: float = 1.0) -> np.ndarray:
        """
        Compute velocities along the orbit.

        Args:
            n_samples: Number of samples
            time_step: Time step between samples

        Returns:
            Array of shape (n_samples, 2) with [v_real, v_imag] velocities
        """
        positions = self.sample(n_samples)

        # Compute finite differences
        velocities = np.zeros_like(positions, dtype=np.float32)
        if n_samples > 1:
            # Forward differences for all but last
            velocities[:-1] = (positions[1:] - positions[:-1]) / time_step
            # Wrap around for closed orbits
            velocities[-1] = (positions[0] - positions[-1]) / time_step

        return velocities


def generate_dynamic_curriculum_orbits() -> Dict[str, MandelbrotOrbit]:
    """
    Generate a comprehensive curriculum of orbits covering all major lobes.
    Implements the full geometry with sub-lobes and appropriate s-values.

    Returns:
        Dictionary mapping orbit names to MandelbrotOrbit instances
    """
    orbits = {}

    # Main cardioid (period 1) - use slightly outside boundary (s ~ 1.02-1.15)
    orbits["cardioid_boundary"] = MandelbrotOrbit(
        "cardioid_boundary", lobe=1, sub_lobe=0, n_points=100, s_range=(1.02, 1.02), angular_velocity=1.0
    )
    orbits["cardioid_interior"] = MandelbrotOrbit(
        "cardioid_interior", lobe=1, sub_lobe=0, n_points=100, s_range=(0.8, 0.8), angular_velocity=1.0
    )
    orbits["cardioid_exterior"] = MandelbrotOrbit(
        "cardioid_exterior", lobe=1, sub_lobe=0, n_points=100, s_range=(1.15, 1.15), angular_velocity=1.0
    )
    orbits["cardioid_boundary_cw"] = MandelbrotOrbit(
        "cardioid_boundary_cw", lobe=1, sub_lobe=0, n_points=100, s_range=(1.02, 1.02), angular_velocity=-1.0
    )

    # Period-2 bulb (left circle)
    orbits["period2_boundary"] = MandelbrotOrbit(
        "period2_boundary", lobe=2, sub_lobe=0, n_points=80, s_range=(1.02, 1.02), angular_velocity=1.0
    )
    orbits["period2_interior"] = MandelbrotOrbit(
        "period2_interior", lobe=2, sub_lobe=0, n_points=80, s_range=(0.7, 0.7), angular_velocity=1.0
    )
    orbits["period2_exterior"] = MandelbrotOrbit(
        "period2_exterior", lobe=2, sub_lobe=0, n_points=80, s_range=(1.15, 1.15), angular_velocity=1.0
    )
    orbits["period2_boundary_cw"] = MandelbrotOrbit(
        "period2_boundary_cw", lobe=2, sub_lobe=0, n_points=80, s_range=(1.02, 1.02), angular_velocity=-1.0
    )

    # Period-3 bulbs (upper, lower, mini-Mandelbrot)
    orbits["period3_upper"] = MandelbrotOrbit(
        "period3_upper", lobe=3, sub_lobe=0, n_points=80, s_range=(1.02, 1.02), angular_velocity=1.0
    )
    orbits["period3_lower"] = MandelbrotOrbit(
        "period3_lower", lobe=3, sub_lobe=1, n_points=80, s_range=(1.02, 1.02), angular_velocity=1.0
    )
    orbits["period3_mini"] = MandelbrotOrbit(
        "period3_mini", lobe=3, sub_lobe=2, n_points=80, s_range=(1.02, 1.02), angular_velocity=1.0
    )
    # Backward compatibility
    orbits["period3_boundary"] = orbits["period3_upper"]
    orbits["period3_interior"] = MandelbrotOrbit(
        "period3_interior", lobe=3, sub_lobe=0, n_points=80, s_range=(0.6, 0.6), angular_velocity=1.0
    )

    # Period-4 bulbs (cascade + 2 primaries)
    orbits["period4_cascade"] = MandelbrotOrbit(
        "period4_cascade", lobe=4, sub_lobe=0, n_points=80, s_range=(1.02, 1.02), angular_velocity=1.0
    )
    orbits["period4_primary_1"] = MandelbrotOrbit(
        "period4_primary_1", lobe=4, sub_lobe=1, n_points=80, s_range=(1.02, 1.02), angular_velocity=1.0
    )
    orbits["period4_primary_2"] = MandelbrotOrbit(
        "period4_primary_2", lobe=4, sub_lobe=2, n_points=80, s_range=(1.02, 1.02), angular_velocity=1.0
    )
    # Backward compatibility
    orbits["period4_boundary"] = orbits["period4_cascade"]
    orbits["period4_interior"] = MandelbrotOrbit(
        "period4_interior", lobe=4, sub_lobe=0, n_points=80, s_range=(0.6, 0.6), angular_velocity=1.0
    )

    # Period-8 bulbs (cascade + 2 satellites)
    orbits["period8_cascade"] = MandelbrotOrbit(
        "period8_cascade", lobe=8, sub_lobe=0, n_points=60, s_range=(1.02, 1.02), angular_velocity=1.0
    )
    orbits["period8_primary_1"] = MandelbrotOrbit(
        "period8_primary_1", lobe=8, sub_lobe=1, n_points=60, s_range=(1.02, 1.02), angular_velocity=1.0
    )
    orbits["period8_primary_2"] = MandelbrotOrbit(
        "period8_primary_2", lobe=8, sub_lobe=2, n_points=60, s_range=(1.02, 1.02), angular_velocity=1.0
    )
    orbits["period8_boundary"] = orbits["period8_cascade"]
    orbits["period8_interior"] = MandelbrotOrbit(
        "period8_interior", lobe=8, sub_lobe=0, n_points=60, s_range=(0.6, 0.6), angular_velocity=1.0
    )

    # Higher periods (5, 6, 7) using totient enumeration
    for period in [5, 6, 7]:
        max_sub = MandelbrotGeometry.get_max_sub_lobe_for_period(period)
        for sub in range(min(3, max_sub + 1)):  # Limit to 3 sub-lobes to keep dataset manageable
            orbits[f"period{period}_sub{sub}"] = MandelbrotOrbit(
                f"period{period}_sub{sub}", lobe=period, sub_lobe=sub, n_points=60, s_range=(1.02, 1.02), angular_velocity=1.0
            )
        # Backward compatibility
        orbits[f"period{period}_boundary"] = MandelbrotOrbit(
            f"period{period}_boundary", lobe=period, sub_lobe=0, n_points=60, s_range=(1.02, 1.02), angular_velocity=1.0
        )
        orbits[f"period{period}_interior"] = MandelbrotOrbit(
            f"period{period}_interior", lobe=period, sub_lobe=0, n_points=60, s_range=(0.6, 0.6), angular_velocity=1.0
        )

    return orbits


# Global cache of orbits
_ORBIT_CACHE = generate_dynamic_curriculum_orbits()


def get_preset_orbit(name: str) -> MandelbrotOrbit:
    """
    Get a preset orbit by name.

    Args:
        name: Orbit name (see _ORBIT_CACHE keys)

    Returns:
        MandelbrotOrbit instance

    Raises:
        KeyError: If orbit name not found
    """
    return _ORBIT_CACHE[name]


def list_preset_names() -> List[str]:
    """
    Get list of available preset orbit names.

    Returns:
        List of preset names
    """
    return list(_ORBIT_CACHE.keys())


def generate_curriculum_sequence(
    n_samples: int = 1000, use_curriculum: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a curriculum learning sequence of positions and velocities.

    Orbits are ordered by increasing difficulty:
    Phase 0 (stability/easy): cardioid interior/boundary, period2 boundary
    Phase 1 (variety): cardioid exterior, period2 interior/exterior, period3 upper/lower
    Phase 2+ (complex): period3 mini, period4 cascade+primaries, period8, higher periods

    Args:
        n_samples: Total number of samples to generate
        use_curriculum: If True, order by difficulty; if False, sample uniformly

    Returns:
        Tuple of (positions, velocities) arrays of shape (n_samples, 2)
    """
    if use_curriculum:
        # Phase 0: Easy/stable orbits
        phase0_orbits = [
            "cardioid_interior",
            "cardioid_boundary",
            "period2_boundary",
        ]
        # Phase 1: More variety
        phase1_orbits = [
            "cardioid_exterior",
            "period2_interior",
            "period2_exterior",
            "period3_upper",
            "period3_lower",
        ]
        # Phase 2+: Complex orbits
        phase2_orbits = [
            "period3_mini",
            "period4_cascade",
            "period4_primary_1",
            "period4_primary_2",
            "period8_cascade",
            "period8_primary_1",
            "period5_sub0",
            "period6_sub0",
        ]
        
        # Mix with some clockwise orbits for variety
        difficulty_order = phase0_orbits + phase1_orbits + phase2_orbits + [
            "cardioid_boundary_cw",
            "period2_boundary_cw",
        ]
    else:
        # Use all available orbits equally
        difficulty_order = list(_ORBIT_CACHE.keys())

    # Distribute samples across orbits
    samples_per_orbit = n_samples // len(difficulty_order)
    remainder = n_samples % len(difficulty_order)

    all_positions = []
    all_velocities = []

    for i, orbit_name in enumerate(difficulty_order):
        if orbit_name not in _ORBIT_CACHE:
            continue

        orbit = get_preset_orbit(orbit_name)
        n = samples_per_orbit + (1 if i < remainder else 0)

        if n > 0:
            positions = orbit.sample(n)
            velocities = orbit.compute_velocities(n)

            all_positions.append(positions)
            all_velocities.append(velocities)

    if not all_positions:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    return np.concatenate(all_positions, axis=0), np.concatenate(all_velocities, axis=0)


# Legacy compatibility functions
def is_in_mandelbrot_set(c_real: float, c_imag: float, max_iter: int = 100) -> bool:
    """
    Check if a complex number is in the Mandelbrot set.

    Args:
        c_real: Real part of c
        c_imag: Imaginary part of c
        max_iter: Maximum iterations to test

    Returns:
        True if point is in the set (bounded)
    """
    c = complex(c_real, c_imag)
    z = 0 + 0j

    for _ in range(max_iter):
        if abs(z) > 2.0:
            return False
        z = z * z + c

    return True


def generate_random_mandelbrot_points(
    n_points: int, region: str = "cardioid", max_attempts: int = 10
) -> np.ndarray:
    """
    Generate random points near the Mandelbrot set boundary.

    Args:
        n_points: Number of points to generate
        region: Region to sample from ('cardioid', 'bulb', 'all')
        max_attempts: Max attempts per point to find valid location

    Returns:
        Array of shape (n_points, 2) with [real, imag] coordinates
    """
    points: List[List[float]] = []

    # Define sampling bounds based on region
    bounds = {
        "cardioid": (
            (-0.75, 0.5),
            (-0.75, 0.75),
        ),  # (real_min, real_max), (imag_min, imag_max)
        "bulb": ((-1.5, -0.5), (-0.5, 0.5)),
        "all": ((-2.0, 0.5), (-1.25, 1.25)),
    }

    real_bounds, imag_bounds = bounds.get(region, bounds["all"])

    while len(points) < n_points:
        for _ in range(max_attempts):
            # Sample point
            real = np.random.uniform(*real_bounds)
            imag = np.random.uniform(*imag_bounds)

            # Check if near boundary (in set but close to escaping)
            if is_in_mandelbrot_set(real, imag, max_iter=50):
                # Add small perturbation to be near boundary
                real += np.random.normal(0, 0.01)
                imag += np.random.normal(0, 0.01)

            points.append([real, imag])
            break

        if len(points) >= n_points:
            break

    return np.array(points[:n_points], dtype=np.float32)
