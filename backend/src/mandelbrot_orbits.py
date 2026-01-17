"""
Mandelbrot set orbital trajectories for curriculum learning.

Defines preset paths along the Mandelbrot set lobes that can be used
to initialize training with known good trajectories for the Julia parameter c.
"""

import numpy as np
from typing import List, Tuple, Dict


class MandelbrotOrbit:
    """Defines a trajectory through complex parameter space based on Mandelbrot geometry."""

    def __init__(
        self, name: str, points: List[Tuple[float, float]], closed: bool = True
    ):
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


def compute_boundary_distance(
    c_real: float, c_imag: float, max_iter: int = 100, threshold: float = 2.0
) -> float:
    """
    Compute approximate distance to Mandelbrot set boundary.

    Uses the escape time algorithm to estimate distance. Points that
    escape quickly are far from the boundary, points that never escape
    are inside, and points near the boundary take intermediate iterations.

    Args:
        c_real: Real part of c
        c_imag: Imaginary part of c
        max_iter: Maximum iterations to test
        threshold: Escape threshold (typically 2.0)

    Returns:
        Normalized distance [0, 1] where 0 = on boundary, 1 = far from boundary
    """
    c = complex(c_real, c_imag)
    z = 0 + 0j

    for i in range(max_iter):
        if abs(z) > threshold:
            # Escaped - compute normalized distance
            # Points that escape early are far from boundary
            escape_speed = 1.0 - (i / max_iter)
            return escape_speed

        z = z * z + c

    # Did not escape - inside the set
    # Points deep inside have small magnitude, points near boundary approach threshold
    final_magnitude = abs(z)
    
    # Normalize to [0, 1] where:
    # - magnitude near 0 (deep inside) -> distance near 1 (far from boundary)
    # - magnitude near threshold (edge) -> distance near 0 (close to boundary)
    # Formula: higher magnitude = closer to boundary = lower distance
    distance = (threshold - final_magnitude) / threshold
    
    return max(0.0, min(1.0, distance))


def detect_boundary_crossing(
    prev_real: float,
    prev_imag: float,
    curr_real: float,
    curr_imag: float,
    max_iter: int = 100,
    crossing_threshold: float = 0.3,
) -> bool:
    """
    Detect if trajectory crossed the Mandelbrot set boundary.

    A crossing occurs when one point is inside/near the set and
    the next point is outside, or vice versa.

    Args:
        prev_real: Previous real coordinate
        prev_imag: Previous imaginary coordinate
        curr_real: Current real coordinate
        curr_imag: Current imaginary coordinate
        max_iter: Maximum iterations for boundary check
        crossing_threshold: Distance threshold for boundary proximity

    Returns:
        True if boundary was crossed
    """
    prev_dist = compute_boundary_distance(prev_real, prev_imag, max_iter)
    curr_dist = compute_boundary_distance(curr_real, curr_imag, max_iter)

    # Check if we crossed from inside to outside or vice versa
    # Low distance = near boundary, high distance = far from boundary
    # Point is "near boundary" when distance < crossing_threshold
    prev_near_boundary = prev_dist < crossing_threshold
    curr_near_boundary = curr_dist < crossing_threshold

    # Crossing detected if one is near boundary and one is not
    crossed = prev_near_boundary != curr_near_boundary

    return crossed


def compute_crossing_score(
    c_real: float, c_imag: float, max_iter: int = 100
) -> float:
    """
    Compute a score for how close a point is to the Mandelbrot boundary.

    Higher scores indicate the point is near the boundary, which is
    desirable for interesting Julia set visualizations.

    The scoring function peaks at mid-range distances, rewarding points
    that are transitioning between inside and outside the set.

    Args:
        c_real: Real part of c
        c_imag: Imaginary part of c
        max_iter: Maximum iterations

    Returns:
        Crossing score [0, 1] where 1 = on boundary, 0 = far from boundary
    """
    distance = compute_boundary_distance(c_real, c_imag, max_iter)

    # Convert distance to score (inverse relationship)
    # We want high scores for points near the boundary
    # Optimal distance is around 0.5 (mid-range between inside and outside)
    OPTIMAL_DISTANCE = 0.5
    DISTANCE_SCALE = 2.0
    
    score = 1.0 - abs(distance - OPTIMAL_DISTANCE) * DISTANCE_SCALE
    score = max(0.0, min(1.0, score))  # Clamp to [0, 1]

    return score


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


def generate_boundary_crossing_trajectory(
    n_samples: int,
    n_crossings: int = 10,
    orbit_names: List[str] = None,
    crossing_intensity: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic trajectory with intentional boundary crossings.

    Creates a trajectory that crosses the Mandelbrot set boundary at
    regular intervals, useful for training with boundary crossing rewards.

    Args:
        n_samples: Total number of samples to generate
        n_crossings: Approximate number of boundary crossings
        orbit_names: List of orbit names to use (defaults to all presets)
        crossing_intensity: How close to boundary to get (0-1, higher = closer)

    Returns:
        Tuple of (positions, velocities, crossing_events) arrays
        - positions: (n_samples, 2) [real, imag]
        - velocities: (n_samples, 2) [v_real, v_imag]
        - crossing_events: (n_samples,) binary array indicating crossings
    """
    if orbit_names is None:
        orbit_names = list(MANDELBROT_PRESETS.keys())

    # Calculate samples per crossing segment
    samples_per_segment = n_samples // (n_crossings + 1)

    all_positions = []
    all_velocities = []
    all_crossings = []

    # Generate trajectory segments
    for i in range(n_crossings + 1):
        # Alternate between different orbits
        orbit_name = orbit_names[i % len(orbit_names)]
        orbit = get_preset_orbit(orbit_name)

        # Sample positions from orbit
        positions = orbit.sample(samples_per_segment)
        velocities = orbit.compute_velocities(samples_per_segment)

        # Add boundary-crossing perturbation
        if i > 0:
            # Create a crossing at the start of this segment
            # Move from inside to outside (or vice versa) Mandelbrot set
            for j in range(min(5, samples_per_segment)):
                # Gradually move across boundary
                factor = j / 5.0
                perturbation = crossing_intensity * 0.1 * factor

                # Add perturbation to move across boundary
                if is_in_mandelbrot_set(
                    positions[j, 0], positions[j, 1], max_iter=50
                ):
                    # Move outward
                    positions[j, 0] += perturbation
                    positions[j, 1] += perturbation
                else:
                    # Move inward
                    positions[j, 0] -= perturbation
                    positions[j, 1] -= perturbation

        # Detect crossings in this segment
        crossings = np.zeros(samples_per_segment, dtype=np.float32)
        for j in range(1, samples_per_segment):
            if detect_boundary_crossing(
                positions[j - 1, 0],
                positions[j - 1, 1],
                positions[j, 0],
                positions[j, 1],
            ):
                crossings[j] = 1.0

        all_positions.append(positions)
        all_velocities.append(velocities)
        all_crossings.append(crossings)

    # Concatenate all segments
    final_positions = np.concatenate(all_positions, axis=0)[:n_samples]
    final_velocities = np.concatenate(all_velocities, axis=0)[:n_samples]
    final_crossings = np.concatenate(all_crossings, axis=0)[:n_samples]

    return final_positions, final_velocities, final_crossings
