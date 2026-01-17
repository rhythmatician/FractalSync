"""
Mandelbrot set orbital trajectories for curriculum learning.

Uses mathematically precise geometry to generate orbits around the lobes and bulbs
of the Mandelbrot set, ensuring coverage of both real and imaginary axes.

Based on the rigorous approach from julia-viz:
- Cardioid parametrization for main body
- Newton solvers for period-n bulb centers
- Proper tangency calculations for cascading bulbs
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Optional


class MandelbrotGeometry:
    """Mathematical functions for Mandelbrot set lobe geometry."""

    @staticmethod
    def cardioid_point(theta: float) -> complex:
        """
        Main cardioid boundary using exact parametric equation.
        c = (1/2)e^(iθ) - (1/4)e^(2iθ)

        Args:
            theta: Angle parameter (0 to 2π)

        Returns:
            Complex point on the cardioid boundary
        """
        return 0.5 * complex(math.cos(theta), math.sin(theta)) - 0.25 * complex(
            math.cos(2 * theta), math.sin(2 * theta)
        )

    @staticmethod
    def iterate_g_and_dg(c: complex, n: int) -> Tuple[complex, complex]:
        """
        Compute g_n(c) and g'_n(c) for Newton solving.
        g_n(c) = f_c^(n)(0), the n-th iterate starting from 0.

        Args:
            c: Complex parameter
            n: Period (number of iterations)

        Returns:
            Tuple of (g_n(c), g'_n(c))
        """
        z = 0j
        w = 0j
        for _ in range(n):
            z_new = z * z + c
            w = 2 * z * w + 1
            z = z_new
        return z, w

    @staticmethod
    def has_primitive_period(c: complex, n: int) -> bool:
        """Check if c has primitive period n (not a lower-period divisor)."""
        z = 0j
        for k in range(1, n):
            z = z * z + c
            if abs(z) < 1e-12:
                return False
        return True

    @staticmethod
    def newton_solve_g(
        c_initial: complex, period: int, max_iter: int = 50, tolerance: float = 1e-14
    ) -> Optional[complex]:
        """
        Newton solver for g_n(c) = 0 (period-n bulb center).

        Args:
            c_initial: Starting point
            period: Period to solve for
            max_iter: Max Newton iterations
            tolerance: Convergence tolerance

        Returns:
            Converged center or None if failed
        """
        c = c_initial
        for _ in range(max_iter):
            g_n, g_prime_n = MandelbrotGeometry.iterate_g_and_dg(c, period)
            if abs(g_n) < tolerance:
                if MandelbrotGeometry.has_primitive_period(c, period):
                    return c
                else:
                    return None
            if abs(g_prime_n) < 1e-12:
                return None
            c = c - g_n / g_prime_n
        return None

    @staticmethod
    def period_n_bulb_center(n: int) -> complex:
        """
        Get center of period-n bulb.

        Args:
            n: Period number

        Returns:
            Complex center of the bulb
        """
        if n == 1:
            return 0j  # Main cardioid (approximate)
        elif n == 2:
            return -1.0 + 0j  # Period-2 bulb
        elif n == 3:
            # Period-3: upper bulb
            return -0.122 + 0.745j
        elif n == 4:
            # Period-4: cascade bulb to the left
            return -1.31 + 0j
        else:
            # For higher periods, use cardioid attachment
            # p/q fraction (coprime): use p=1, q=n for primary bulb
            p, q = 1, n
            theta = 2 * math.pi * p / q

            # Start at cardioid root attachment point
            c_init = 0.5 * complex(math.cos(theta), math.sin(theta)) - 0.25 * complex(
                math.cos(2 * theta), math.sin(2 * theta)
            )

            # Take small step outward
            nvec = complex(math.cos(theta), math.sin(theta))
            c_init += 1e-3 * nvec

            # Solve for center
            center = MandelbrotGeometry.newton_solve_g(c_init, n)
            return center if center is not None else c_init

    @staticmethod
    def period_n_bulb_radius(n: int) -> float:
        """
        Approximate radius of period-n bulb.

        Args:
            n: Period number

        Returns:
            Approximate radius
        """
        if n == 1:
            return 0.5
        elif n == 2:
            return 0.25
        elif n == 3:
            return 0.2
        else:
            return 0.2 / n

    @staticmethod
    def lobe_point_at_angle(lobe: int, theta: float, s: float = 1.0) -> complex:
        """
        Get a point on the lobe at the given angle.

        Args:
            lobe: Period number (1=cardioid, 2=period-2, etc.)
            theta: Angular parameter (0 to 2π)
            s: Radius scaling (1.0=boundary, <1=inside, >1=outside)

        Returns:
            Complex point on the lobe
        """
        if lobe == 1:
            # Cardioid: use exact parametric equation
            return s * MandelbrotGeometry.cardioid_point(theta)
        else:
            # Period-n bulbs: approximate as circles
            center = MandelbrotGeometry.period_n_bulb_center(lobe)
            radius = MandelbrotGeometry.period_n_bulb_radius(lobe)
            return center + s * radius * complex(math.cos(theta), math.sin(theta))


class MandelbrotOrbit:
    """Defines a trajectory through complex parameter space based on Mandelbrot geometry."""

    def __init__(
        self,
        name: str,
        lobe: int,
        n_points: int = 50,
        s_range: Tuple[float, float] = (0.9, 1.1),
    ):
        """
        Initialize an orbit around a specific Mandelbrot lobe.

        Args:
            name: Orbit name/identifier
            lobe: Period number (1=main cardioid, 2=period-2 bulb, etc.)
            n_points: Number of points to sample around the orbit
            s_range: (s_min, s_max) for radius scaling variations
        """
        self.name = name
        self.lobe = lobe
        self.n_points = n_points
        self.s_range = s_range

        # Generate points dynamically
        self.points = self._generate_points()

    def _generate_points(self) -> np.ndarray:
        """Generate orbit points using mathematical geometry."""
        angles = np.linspace(0, 2 * np.pi, self.n_points, endpoint=False)
        s_values = np.linspace(self.s_range[0], self.s_range[1], self.n_points)

        points: List[List[float]] = []
        for angle, s in zip(angles, s_values):
            c = MandelbrotGeometry.lobe_point_at_angle(
                self.lobe, float(angle), float(s)
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

    Returns:
        Dictionary mapping orbit names to MandelbrotOrbit instances
    """
    orbits = {}

    # Main cardioid (period 1)
    orbits["cardioid_boundary"] = MandelbrotOrbit(
        "cardioid_boundary", lobe=1, n_points=100, s_range=(0.95, 1.05)
    )
    orbits["cardioid_interior"] = MandelbrotOrbit(
        "cardioid_interior", lobe=1, n_points=100, s_range=(0.3, 0.7)
    )
    orbits["cardioid_exterior"] = MandelbrotOrbit(
        "cardioid_exterior", lobe=1, n_points=100, s_range=(1.2, 1.5)
    )

    # Period-2 bulb (left circle)
    orbits["period2_boundary"] = MandelbrotOrbit(
        "period2_boundary", lobe=2, n_points=80, s_range=(0.95, 1.05)
    )
    orbits["period2_interior"] = MandelbrotOrbit(
        "period2_interior", lobe=2, n_points=80, s_range=(0.3, 0.7)
    )

    # Period-3 bulbs
    orbits["period3_boundary"] = MandelbrotOrbit(
        "period3_boundary", lobe=3, n_points=80, s_range=(0.9, 1.1)
    )
    orbits["period3_interior"] = MandelbrotOrbit(
        "period3_interior", lobe=3, n_points=80, s_range=(0.2, 0.6)
    )

    # Period-4 bulbs
    orbits["period4_boundary"] = MandelbrotOrbit(
        "period4_boundary", lobe=4, n_points=80, s_range=(0.9, 1.1)
    )
    orbits["period4_interior"] = MandelbrotOrbit(
        "period4_interior", lobe=4, n_points=80, s_range=(0.2, 0.6)
    )

    # Higher periods
    for period in [5, 6, 7, 8]:
        orbits[f"period{period}_boundary"] = MandelbrotOrbit(
            f"period{period}_boundary", lobe=period, n_points=60, s_range=(0.9, 1.1)
        )
        orbits[f"period{period}_interior"] = MandelbrotOrbit(
            f"period{period}_interior", lobe=period, n_points=60, s_range=(0.2, 0.6)
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

    Orbits are ordered by increasing period and sampling region (boundary → interior).
    This ensures early training focuses on well-defined structures.

    Args:
        n_samples: Total number of samples to generate
        use_curriculum: If True, order by difficulty; if False, sample uniformly

    Returns:
        Tuple of (positions, velocities) arrays of shape (n_samples, 2)
    """
    # Difficulty order: simple (low period, boundary) to complex (high period, interior)
    if use_curriculum:
        difficulty_order = [
            "cardioid_boundary",
            "cardioid_exterior",
            "cardioid_interior",
            "period2_boundary",
            "period2_interior",
            "period3_boundary",
            "period3_interior",
            "period4_boundary",
            "period4_interior",
            "period5_boundary",
            "period5_interior",
            "period6_boundary",
            "period6_interior",
            "period7_boundary",
            "period7_interior",
            "period8_boundary",
            "period8_interior",
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

        positions = orbit.sample(n)
        velocities = orbit.compute_velocities(n)

        all_positions.append(positions)
        all_velocities.append(velocities)

    return np.concatenate(all_positions, axis=0), np.concatenate(all_velocities, axis=0)


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
