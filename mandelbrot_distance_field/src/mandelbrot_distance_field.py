"""
Mandelbrot distance field generator (GPU-accelerated via OpenGL compute shaders).

Pre-computes a high-resolution escape-time map of the Mandelbrot set that can
be used to slow down orbit synthesis near the boundary.

Semantics of the escape-time field `E`:
- `E = i / max_iter` for points that escape at iteration `i` (fast escape → small E)
- `E = 1.0` for points that never escape within `max_iter` (inside/near boundary)

This convention (1.0 near/inside, ~0 far outside) matches the runtime-core
velocity scaling that slows down near the boundary.
"""

import numpy as np
from pathlib import Path
import logging
from typing import Tuple

try:
    import moderngl
except ImportError:  # pragma: no cover
    moderngl = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def mandelbrot_distance_hybrid(c: complex, max_iter: int = 256) -> float:
    """
    Compute distance metric for Mandelbrot set using hybrid approach.

    For points outside: normalized escape time (fast escape → 0, slow escape → ~0.2)
    For points inside: distance estimation using derivatives (near boundary → 0, deep inside → higher)

    Returns normalized value in [0, 1]:
    - 0.0 = far from boundary (either fast escape or deep interior)
    - 1.0 = on/very near the boundary

    This allows velocity scaling to work both inside and outside the set.
    """
    z = 0j
    dz = 0j  # Derivative for distance estimation

    for i in range(max_iter):
        if abs(z) > 2.0:
            # Escaped - use normalized iteration as distance proxy
            # Lower iterations = farther from boundary
            escape_time = i / max_iter
            # Invert so boundary (slow escape) has high value
            # Map escape times to emphasize boundary region
            # Most points escape quickly (small i), boundary points escape slowly (large i)
            return escape_time

        # Update derivative: dz = 2*z*dz + 1
        dz = 2.0 * z * dz + 1.0
        # Mandelbrot iteration
        z = z * z + c

    # Didn't escape - use distance estimation
    # Distance = |z| * ln(|z|) / |dz|
    z_mag = abs(z)
    dz_mag = abs(dz)

    if z_mag < 1e-10 or dz_mag < 1e-10:
        # On boundary or very close
        return 1.0

    # Distance estimation (smaller = closer to boundary)
    distance = z_mag * np.log(z_mag) / dz_mag

    # Normalize: clamp and map to [0, 1]
    # Typical interior distances range from ~0 (boundary) to ~0.5 (deep inside)
    # We want boundary (distance ≈ 0) → 1.0, deep inside (distance > 0.2) → 0
    max_interior_dist = 0.3
    normalized_dist = min(distance / max_interior_dist, 1.0)

    # Invert: small distance (near boundary) → high value
    return 1.0 - normalized_dist


def mandelbrot_escape_time(c: complex, max_iter: int = 256) -> float:
    """
    Compute escape time for Mandelbrot iteration (deprecated - use hybrid version).

    Returns normalized escape time E in [0, 1]:
    - E = i / max_iter if the point escapes at iteration i (fast escape → small E)
    - E = 1.0 if the point doesn't escape (inside/on boundary)

    Note: This doesn't distinguish between points near boundary vs deep inside,
    so velocity scaling stops completely inside the set. Use mandelbrot_distance_hybrid instead.
    """
    z = 0j

    for i in range(max_iter):
        if abs(z) > 2.0:
            # Escaped - normalize iteration count to [0, 1]
            return i / max_iter
        z = z * z + c

    # Didn't escape (inside or on boundary): use 1.0 to indicate maximum proximity
    return 1.0


def _generate_distance_field_opengl(
    resolution: int,
    real_range: Tuple[float, float],
    imag_range: Tuple[float, float],
    max_iter: int,
) -> np.ndarray:
    """
    GPU-accelerated escape-time field using OpenGL compute shaders.

    Produces the same semantics as `mandelbrot_escape_time` for each grid point.
    """
    assert moderngl is not None, "ModernGL is required for GPU generation"

    # Create headless OpenGL context
    ctx = moderngl.create_standalone_context()
    logger.info(f"  Using OpenGL: {ctx.info['GL_RENDERER']}")

    real_min, real_max = real_range
    imag_min, imag_max = imag_range

    # Compute shader source - simple escape time
    compute_shader = """
    #version 430
    
    layout(local_size_x = 16, local_size_y = 16) in;
    
    layout(std430, binding = 0) buffer OutputBuffer {
        float escape_times[];
    };
    
    uniform int resolution;
    uniform float real_min;
    uniform float real_max;
    uniform float imag_min;
    uniform float imag_max;
    uniform int max_iter;
    
    void main() {
        ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
        
        if (pixel.x >= resolution || pixel.y >= resolution) {
            return;
        }
        
        // Map pixel to complex plane
        float real = real_min + (real_max - real_min) * float(pixel.x) / float(resolution - 1);
        float imag = imag_min + (imag_max - imag_min) * float(pixel.y) / float(resolution - 1);
        
        vec2 c = vec2(real, imag);
        vec2 z = vec2(0.0, 0.0);
        
        int iter = 0;
        for (iter = 0; iter < max_iter; iter++) {
            // Check escape: |z|^2 > 4.0
            if (dot(z, z) > 4.0) {
                break;
            }
            
            // Mandelbrot iteration: z = z^2 + c
            float z_real = z.x * z.x - z.y * z.y + c.x;
            float z_imag = 2.0 * z.x * z.y + c.y;
            z = vec2(z_real, z_imag);
        }
        
        // Normalize: escaped points get iter/max_iter, non-escaped get 1.0
        float escape_time;
        if (iter < max_iter) {
            escape_time = float(iter) / float(max_iter);
        } else {
            escape_time = 1.0;
        }
        
        int index = pixel.y * resolution + pixel.x;
        escape_times[index] = escape_time;
    }
    """

    # Create compute shader program
    compute_prog = ctx.compute_shader(compute_shader)

    # Create output buffer
    buffer_size = resolution * resolution * 4  # 4 bytes per float32
    output_buffer = ctx.buffer(reserve=buffer_size)

    # Bind buffer and set uniforms
    output_buffer.bind_to_storage_buffer(0)
    compute_prog["resolution"].value = resolution
    compute_prog["real_min"].value = real_min
    compute_prog["real_max"].value = real_max
    compute_prog["imag_min"].value = imag_min
    compute_prog["imag_max"].value = imag_max
    compute_prog["max_iter"].value = max_iter

    # Dispatch compute shader
    # Work groups of 16x16 threads each
    groups_x = (resolution + 15) // 16
    groups_y = (resolution + 15) // 16
    compute_prog.run(groups_x, groups_y, 1)

    # Read back results
    result_data = np.frombuffer(output_buffer.read(), dtype=np.float32)
    distance_field = result_data.reshape((resolution, resolution))

    # Cleanup
    output_buffer.release()
    ctx.release()

    return distance_field


def apply_interior_distance_transform(
    field: np.ndarray, max_distance_pixels: int = 100
) -> np.ndarray:
    """
    Apply distance transform to interior points (where field == 1.0).

    For each interior pixel, find the distance to the nearest boundary pixel
    and normalize to [0, 1] where:
    - 0 = far from boundary (deep interior)
    - 1 = on boundary

    Args:
        field: Input field where 1.0 = interior, <1.0 = exterior
        max_distance_pixels: Maximum distance to search (normalizes result)

    Returns:
        Modified field with interior points having gradient based on distance to boundary
    """
    from scipy.ndimage import distance_transform_edt

    logger.info("  Applying distance transform to interior points...")

    # Create binary mask: interior (1.0) vs exterior (<1.0)
    interior_mask = field >= 0.99  # Threshold to handle floating point

    # Compute distance transform: distance from each interior pixel to nearest exterior pixel
    distances = distance_transform_edt(interior_mask)

    # Normalize distances: [0, max_distance_pixels] → [1.0, 0.0]
    # Near boundary → 1.0, deep inside → 0.0
    normalized_distances = np.clip(distances / max_distance_pixels, 0.0, 1.0)
    inverted_distances = 1.0 - normalized_distances

    # Replace interior values with distance-based values
    result = field.copy()
    result[interior_mask] = inverted_distances[interior_mask]

    interior_count = interior_mask.sum()
    logger.info(f"  Processed {interior_count} interior pixels")

    return result


def generate_distance_field(
    resolution: int = 2048,
    real_range: Tuple[float, float] = (-2.5, 1.0),
    imag_range: Tuple[float, float] = (-1.5, 1.5),
    max_iter: int = 1024,
    max_distance: float = 0.5,
    slowdown_threshold: float = 0.02,
    use_gpu: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Generate an escape-time field for the Mandelbrot set.

    Uses simple escape time iteration (vectorized). Points that escape quickly
    are far from boundary (small value), and points that do not escape are near
    or inside the boundary (value 1.0).

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

    # Prefer OpenGL GPU when available
    use_opengl = use_gpu and (moderngl is not None)
    if use_opengl:
        try:
            distance_field = _generate_distance_field_opengl(
                resolution, (real_min, real_max), (imag_min, imag_max), max_iter
            )
        except Exception as e:
            logger.warning(f"OpenGL generation failed ({e}), falling back to numpy")
            use_opengl = False

    if not use_opengl:
        # CPU fallback using numpy loops (slower but reliable)
        real_vals = np.linspace(real_min, real_max, resolution)
        imag_vals = np.linspace(imag_min, imag_max, resolution)
        distance_field = np.zeros((resolution, resolution), dtype=np.float32)
        for i, imag in enumerate(imag_vals):
            if i % 250 == 0:
                logger.info(f"  Row {i}/{resolution}")
            for j, real in enumerate(real_vals):
                et = mandelbrot_escape_time(complex(real, imag), max_iter)
                distance_field[i, j] = et

    # Apply distance transform to interior points
    distance_field = apply_interior_distance_transform(
        distance_field, max_distance_pixels=100
    )

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
    frac_non_escape = float(np.sum(distance_field == 1.0)) / float(
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
            # We want boundary/inside (1.0) to be black and far outside (~0) white.
            # So we map E → (1 - E) for visualization.
            img_data = ((1.0 - distance_field) * 255.0).astype(np.uint8)
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
            Escape-time value (0.0..1.0). Returns 1.0 if c is outside the field bounds.
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

    logger.info("Generating default Mandelbrot escape-time field (GPU if available)...")
    logger.info("This may take a minute depending on resolution...")

    distance_field, metadata = generate_distance_field(
        resolution=8192,
        real_range=(-2.5, 1.0),
        imag_range=(-1.5, 1.5),
        max_iter=512,  # higher for better dynamic range; still quick on GPU
        max_distance=0.5,  # Unused but kept for compatibility
        use_gpu=True,
    )

    save_distance_field(distance_field, metadata, str(output_path), save_png=True)

    logger.info("Done! Distance field ready for use.")


if __name__ == "__main__":
    # Generate default field when run as script
    generate_and_save_default_field()
