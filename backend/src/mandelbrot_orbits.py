"""
Mandelbrot set orbital trajectories for curriculum learning.

Defines preset paths along the Mandelbrot set lobes that can be used
to initialize training with known good trajectories for the Julia parameter c.
"""

import numpy as np
from typing import List, Tuple, Dict


class MandelbrotOrbit:
    """Defines a trajectory through complex parameter space based on Mandelbrot geometry."""

    def __init__(self, name: str, points: List[Tuple[float, float]], closed: bool = True):
        """
        Initialize an orbit.

        Args:
            name: Orbit name/identifier
            points: List of (real, imag) complex number coordinates
            closed: Whether the orbit forms a closed loop
        """
        self.name = name
        self.points = np.array(points, dtype=np.float32)
        self.closed = closed

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

        if self.closed and n_samples > 1:
            # Ensure loop closure by connecting last point to first
            real[-1] = self.points[0, 0]
            imag[-1] = self.points[0, 1]

        return np.stack([real, imag], axis=1)

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
        velocities = np.zeros_like(positions)
        if n_samples > 1:
            velocities[:-1] = (positions[1:] - positions[:-1]) / time_step
            if self.closed:
                # Wrap around for closed orbits
                velocities[-1] = (positions[0] - positions[-1]) / time_step
            else:
                # Repeat last velocity for open orbits
                velocities[-1] = velocities[-2]

        return velocities


# Preset orbits based on Mandelbrot set geometry
MANDELBROT_PRESETS: Dict[str, MandelbrotOrbit] = {
    # Main cardioid orbit (heart-shaped main body)
    "cardioid": MandelbrotOrbit(
        name="cardioid",
        points=[
            (0.3, 0.0),  # Right side
            (0.25, 0.25),  # Upper right
            (0.0, 0.5),  # Top
            (-0.25, 0.25),  # Upper left
            (-0.5, 0.0),  # Left bulb connection
            (-0.25, -0.25),  # Lower left
            (0.0, -0.5),  # Bottom
            (0.25, -0.25),  # Lower right
        ],
        closed=True,
    ),
    # Circular bulb (left circular region)
    "bulb": MandelbrotOrbit(
        name="bulb",
        points=[
            (-1.0, 0.0),  # Center of bulb
            (-0.75, 0.25),  # Upper
            (-0.5, 0.0),  # Connection to cardioid
            (-0.75, -0.25),  # Lower
        ],
        closed=True,
    ),
    # Period-3 bulb (upper decorative region)
    "period3_upper": MandelbrotOrbit(
        name="period3_upper",
        points=[
            (-0.125, 0.65),  # Center of period-3 bulb
            (0.0, 0.75),  # Top point
            (0.1, 0.65),  # Right
            (0.0, 0.55),  # Bottom
            (-0.1, 0.65),  # Left
        ],
        closed=True,
    ),
    # Period-3 bulb (lower decorative region)
    "period3_lower": MandelbrotOrbit(
        name="period3_lower",
        points=[
            (-0.125, -0.65),  # Center of period-3 bulb
            (0.0, -0.75),  # Bottom point
            (0.1, -0.65),  # Right
            (0.0, -0.55),  # Top
            (-0.1, -0.65),  # Left
        ],
        closed=True,
    ),
    # Horizontal sweep across interesting regions
    "horizontal_sweep": MandelbrotOrbit(
        name="horizontal_sweep",
        points=[
            (-1.5, 0.0),  # Far left
            (-1.0, 0.0),  # Left bulb
            (-0.5, 0.0),  # Connection
            (0.0, 0.0),  # Origin
            (0.3, 0.0),  # Right cardioid
        ],
        closed=False,
    ),
    # Vertical sweep through upper regions
    "vertical_sweep": MandelbrotOrbit(
        name="vertical_sweep",
        points=[
            (0.0, 0.0),  # Origin
            (0.0, 0.3),  # Upper cardioid
            (0.0, 0.6),  # Approaching period-3
            (0.0, 0.8),  # Near boundary
        ],
        closed=False,
    ),
    # Spiral inward toward main antenna
    "spiral_antenna": MandelbrotOrbit(
        name="spiral_antenna",
        points=[
            (0.3, 0.1),  # Outer start
            (0.25, 0.05),  # Spiral in
            (0.2, 0.02),
            (0.15, 0.01),
            (0.1, 0.005),
            (0.05, 0.0025),
            (0.0, 0.0),  # Antenna point
        ],
        closed=False,
    ),
    # Figure-8 through both bulb and cardioid
    "figure8": MandelbrotOrbit(
        name="figure8",
        points=[
            (-1.0, 0.0),  # Left bulb
            (-0.75, 0.15),
            (-0.5, 0.0),  # Connection
            (-0.25, 0.15),
            (0.0, 0.0),  # Origin
            (0.15, 0.15),
            (0.3, 0.0),  # Right cardioid
            (0.15, -0.15),
            (0.0, 0.0),  # Back through origin
            (-0.25, -0.15),
            (-0.5, 0.0),
            (-0.75, -0.15),
        ],
        closed=True,
    ),
}


def get_preset_orbit(name: str) -> MandelbrotOrbit:
    """
    Get a preset orbit by name.

    Args:
        name: Orbit name (see MANDELBROT_PRESETS keys)

    Returns:
        MandelbrotOrbit instance

    Raises:
        KeyError: If orbit name not found
    """
    return MANDELBROT_PRESETS[name]


def list_preset_names() -> List[str]:
    """
    Get list of available preset orbit names.

    Returns:
        List of preset names
    """
    return list(MANDELBROT_PRESETS.keys())


def generate_curriculum_sequence(
    n_samples: int = 1000, difficulty_order: List[str] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a curriculum learning sequence of positions and velocities.

    Starts with simple orbits and progresses to more complex ones.

    Args:
        n_samples: Total number of samples to generate
        difficulty_order: Order of orbit names by difficulty (defaults to preset order)

    Returns:
        Tuple of (positions, velocities) arrays of shape (n_samples, 2)
    """
    if difficulty_order is None:
        # Default curriculum: simple to complex
        difficulty_order = [
            "horizontal_sweep",  # Simple 1D motion
            "vertical_sweep",  # Simple 1D motion
            "cardioid",  # Smooth closed curve
            "bulb",  # Simple circle
            "figure8",  # More complex crossing path
            "spiral_antenna",  # Accelerating motion
            "period3_upper",  # Smaller features
            "period3_lower",  # Smaller features
        ]

    # Distribute samples across orbits
    samples_per_orbit = n_samples // len(difficulty_order)
    remainder = n_samples % len(difficulty_order)

    all_positions = []
    all_velocities = []

    for i, orbit_name in enumerate(difficulty_order):
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
    points = []

    # Define sampling bounds based on region
    bounds = {
        "cardioid": ((-0.75, 0.5), (-0.75, 0.75)),  # (real_min, real_max), (imag_min, imag_max)
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
