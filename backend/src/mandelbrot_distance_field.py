"""
Mandelbrot distance field generator for velocity scaling.

Pre-computes a high-resolution distance estimation map of the Mandelbrot set
that can be used to slow down orbit synthesis near the boundary.
"""

import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def mandelbrot_escape_time(c: complex, max_iter: int = 256) -> float:
    """
    Compute escape time for Mandelbrot iteration.

    Returns normalized escape time:
    - 1.0 = escaped immediately (far from boundary)
    - 0.0 = didn't escape (inside/on boundary)

    This is much simpler than distance estimation and works well
    for velocity scaling: slow down near the boundary where escape
    time is high.
    """
    z = 0j

    for i in range(max_iter):
        if abs(z) > 2.0:
            # Escaped - normalize iteration count to [0, 1]
            return i / max_iter
        z = z * z + c

    # Didn't escape (inside or on boundary): use 1.0 to indicate maximum proximity
    return 1.0


def generate_distance_field(
    resolution: int = 2048,
    real_range: tuple[float, float] = (-2.5, 1.0),
    imag_range: tuple[float, float] = (-1.5, 1.5),
    max_iter: int = 1024,
    max_distance: float = 0.5,
    slowdown_threshold: float = 0.02,
) -> tuple[np.ndarray, dict]:
    """
    Generate an escape-time field for the Mandelbrot set.

    Uses simple escape time iteration - no complex derivatives needed.
    Points that escape quickly are far from boundary, points that escape
    slowly are near boundary.

    Args:
        resolution: Image resolution (will be resolution × resolution)
        real_range: (min, max) range for real axis
        imag_range: (min, max) range for imaginary axis
        max_iter: Maximum iterations
        max_distance: Unused (kept for API compatibility)

    Returns:
        (distance_field, metadata) where:
        - distance_field: 2D array of normalized escape times [0, 1]
        - metadata: dict with generation parameters
    """
    logger.info(f"Generating {resolution}×{resolution} Mandelbrot escape-time field...")

    real_min, real_max = real_range
    imag_min, imag_max = imag_range

    # Create coordinate grids
    real_vals = np.linspace(real_min, real_max, resolution)
    imag_vals = np.linspace(imag_min, imag_max, resolution)

    distance_field = np.zeros((resolution, resolution), dtype=np.float32)
    non_escape_count = 0

    for i, imag in enumerate(imag_vals):
        if i % 100 == 0:
            logger.info(f"  Row {i}/{resolution}")
        for j, real in enumerate(real_vals):
            c = complex(real, imag)
            # Escape time already normalized to [0, 1]
            et = mandelbrot_escape_time(c, max_iter)
            if et >= 1.0:
                non_escape_count += 1
            distance_field[i, j] = et

    # Already normalized by escape time function

    metadata = {
        "resolution": resolution,
        "real_range": real_range,
        "imag_range": imag_range,
        "max_iter": max_iter,
        "max_distance": max_distance,
        "slowdown_threshold": slowdown_threshold,
    }

    # Diagnostic logging
    frac_non_escape = float((distance_field == 1.0).sum()) / float(
        resolution * resolution
    )
    logger.info(
        (
            "Escape-time field generation complete! stats: "
            f"min={distance_field.min():.3f}, max={distance_field.max():.3f}, "
            f"non_escape_fraction={frac_non_escape:.3f}"
        )
    )

    # Sanity check: field shouldn't be constant
    assert not np.all(
        np.abs(distance_field - distance_field[0, 0]) < 1e-6
    ), "Distance field is constant!"

    return distance_field, metadata


def save_distance_field(
    distance_field: np.ndarray,
    metadata: dict,
    output_path: str,
    save_png: bool = True,
):
    """
    Save distance field to disk.

    Args:
        distance_field: 2D array of distances
        metadata: Generation parameters
        output_path: Base path (will save .npy and optionally .png)
        save_png: If True, also save a visualization as PNG
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save numpy array
    npy_path = out_path.with_suffix(".npy")
    np.save(npy_path, distance_field)
    logger.info(f"Saved distance field to {npy_path}")

    # Save metadata
    import json

    metadata_path = out_path.with_suffix(".json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {metadata_path}")

    # Save PNG visualization
    if save_png:
        try:
            from PIL import Image

            # Convert to uint8 for PNG
            # For visualization clarity, invert so far outside is bright.
            # escape_time in [0,1]: 0 = boundary/inside, 1 = far outside
            img_data = ((1.0 - distance_field) * 255).astype(np.uint8)
            img = Image.fromarray(img_data, mode="L")
            png_path = out_path.with_suffix(".png")
            img.save(png_path)
            logger.info(f"Saved visualization to {png_path}")
        except ImportError:
            logger.warning("PIL not available, skipping PNG visualization")


def load_distance_field(path: str) -> tuple[np.ndarray, dict]:
    """
    Load distance field from disk.

    Returns:
        (distance_field, metadata)
    """
    base = Path(path)

    # Load numpy array
    npy_path = base.with_suffix(".npy")
    distance_field = np.load(npy_path)

    # Load metadata
    import json

    metadata_path = base.with_suffix(".json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    logger.info(
        f"Loaded {metadata['resolution']}×{metadata['resolution']} distance field"
    )
    return distance_field, metadata


class MandelbrotDistanceField:
    """Lookup table for Mandelbrot distance at any point in the complex plane."""

    def __init__(self, distance_field: np.ndarray, metadata: dict):
        self.field = distance_field
        self.resolution = metadata["resolution"]
        self.real_min, self.real_max = metadata["real_range"]
        self.imag_min, self.imag_max = metadata["imag_range"]
        self.max_distance = metadata["max_distance"]

        # Precompute for fast lookup
        self.real_scale = (self.real_max - self.real_min) / self.resolution
        self.imag_scale = (self.imag_max - self.imag_min) / self.resolution

    def lookup(self, c: complex) -> float:
        """
        Look up distance at complex point c.

        Returns:
            Distance estimate (0.0 to 1.0, normalized by max_distance)
            Returns 1.0 if c is outside the field bounds.
        """
        # Convert complex to pixel coordinates
        real_idx = int((c.real - self.real_min) / self.real_scale)
        imag_idx = int((c.imag - self.imag_min) / self.imag_scale)

        # Check bounds
        if (
            real_idx < 0
            or real_idx >= self.resolution
            or imag_idx < 0
            or imag_idx >= self.resolution
        ):
            return 1.0  # Far outside

        return self.field[imag_idx, real_idx]

    def lookup_array(self, c_array: np.ndarray) -> np.ndarray:
        """
        Look up distances for array of complex values.

        Args:
            c_array: Array of complex values (any shape)

        Returns:
            Array of distances (same shape as input)
        """
        # Convert to pixel coordinates
        real_idx = ((c_array.real - self.real_min) / self.real_scale).astype(int)
        imag_idx = ((c_array.imag - self.imag_min) / self.imag_scale).astype(int)

        # Initialize with max distance (for out-of-bounds)
        distances = np.ones_like(c_array, dtype=np.float32)

        # Find valid indices
        valid_mask = (
            (real_idx >= 0)
            & (real_idx < self.resolution)
            & (imag_idx >= 0)
            & (imag_idx < self.resolution)
        )

        # Look up valid values
        distances[valid_mask] = self.field[imag_idx[valid_mask], real_idx[valid_mask]]

        return distances


def generate_and_save_default_field(output_dir: str = "data"):
    """Generate and save the default distance field used by the trainer."""
    output_path = Path(output_dir) / "mandelbrot_distance_field"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    logger.info("Generating default Mandelbrot escape-time field...")
    logger.info("This may take a few minutes...")

    distance_field, metadata = generate_distance_field(
        resolution=2048,
        real_range=(-2.5, 1.0),
        imag_range=(-1.5, 1.5),
        max_iter=128,  # Lower is fine for escape time
        max_distance=0.5,  # Unused but kept for compatibility
    )

    save_distance_field(distance_field, metadata, str(output_path), save_png=True)

    logger.info("Done! Distance field ready for use.")


if __name__ == "__main__":
    # Generate default field when run as script
    generate_and_save_default_field()
