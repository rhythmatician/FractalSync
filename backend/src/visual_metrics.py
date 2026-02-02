"""
Visual metrics computation for evaluating Julia set renderings.
Measures perceptual qualities like roughness, smoothness, brightness, etc.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import torch
from pathlib import Path
import json
import runtime_core  # type: ignore


@dataclass
class DistanceFieldMeta:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    res: int


# Module-level cached distance field (signed distances, float32)
_distance_field_tensor: Optional[torch.Tensor] = None
_distance_field_meta: Optional[DistanceFieldMeta] = None


def load_distance_field(npy_path: Optional[str | Path] = None) -> None:
    """Load a signed distance field (.npy) with optional metadata (.json next to it).

    Args:
        npy_path: Path to .npy file containing distance field

    The .npy is expected shape (H, W) with row-major Y (increasing) and columns X.
    Metadata file (same stem, .json) should contain xmin/xmax/ymin/ymax to map coords.
    """
    global _distance_field_tensor, _distance_field_meta
    if npy_path:
        npy_path = Path(npy_path)
    else:
        # try builtin embedded fields exposed by runtime_core (available)
        try:
            if True:
                rows, cols, xmin, xmax, ymin, ymax = (
                    runtime_core.get_builtin_distance_field_py("mandelbrot_1024")
                )
                # The Rust loader registers the builtin field; construct a placeholder
                # Python tensor (zeros) so Python-side sampling/fallback can operate.
                import numpy as _np

                arr = _np.zeros((rows, cols), dtype=_np.float32)
                _distance_field_tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
                _distance_field_meta = DistanceFieldMeta(
                    xmin=xmin,
                    xmax=xmax,
                    ymin=ymin,
                    ymax=ymax,
                    res=int(rows),
                )
                return
        except Exception:
            # Ignore — fallback to file candidates
            pass

        # Try a few canonical locations (src/data, backend/data, repo-root data)
        candidates = [
            Path(__file__).parent / "data" / "mandelbrot_distance_2048.npy",
            Path(__file__).parent.parent / "data" / "mandelbrot_distance_2048.npy",
            Path(__file__).parent.parent.parent
            / "data"
            / "mandelbrot_distance_2048.npy",
        ]
        # Also include repo-root 1024 which we added
        candidates.extend(
            [
                Path(__file__).parent.parent.parent
                / "data"
                / "mandelbrot_distance_1024.npy",
                Path(__file__).parent / "data" / "mandelbrot_distance_1024.npy",
                Path(__file__).parent.parent / "data" / "mandelbrot_distance_1024.npy",
            ]
        )
        found = None
        for p in candidates:
            if p.exists():
                found = p
                break
        if found is not None:
            npy_path = found
        else:
            # As a last resort, build a modest distance field on-the-fly for tests/dev.
            import tempfile
            import subprocess
            import sys

            base = Path(tempfile.mkdtemp(prefix="distfield_"))
            out = base / "distfield.npy"
            # script lives at repository root 'scripts/build_distance_field.py'
            script = (
                Path(__file__).parent.parent.parent
                / "scripts"
                / "build_distance_field.py"
            )
            cmd = [
                sys.executable,
                str(script),
                "--out",
                str(out),
                "--res",
                "1024",
                "--xmin",
                "-2.5",
                "--xmax",
                "1.5",
                "--ymin",
                "-2.0",
                "--ymax",
                "2.0",
                "--max-iter",
                "512",
            ]
            subprocess.check_call(cmd)
            npy_path = out

    arr = np.load(npy_path)
    if arr.ndim != 2:
        raise ValueError("distance field must be 2D")
    meta_path = npy_path.with_suffix(".json")
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        meta = DistanceFieldMeta(
            xmin=float(d.get("xmin", -2.5)),
            xmax=float(d.get("xmax", 1.5)),
            ymin=float(d.get("ymin", -2.0)),
            ymax=float(d.get("ymax", 2.0)),
            res=int(d.get("res", arr.shape[0])),
        )
    else:
        # Use reasonable defaults when metadata is not provided
        meta = DistanceFieldMeta(
            xmin=-2.5,
            xmax=1.5,
            ymin=-2.0,
            ymax=2.0,
            res=int(arr.shape[0]),
        )

    H, W = arr.shape
    tensor = (
        torch.from_numpy(arr.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    )  # 1,1,H,W
    _distance_field_tensor = tensor  # CPU tensor, requires_grad=False
    _distance_field_meta = meta

    if not _distance_field_meta:
        raise RuntimeError("Distance field metadata not loaded")
    # Register the field with runtime_core for fast Rust sampling; require it to succeed.
    # We call the Rust loader which accepts a path to the .npy (it reads the .json if present).
    if False:
        raise RuntimeError(
            "runtime_core missing set_distance_field_py. Rebuild runtime-core with `maturin develop --release`."
        )
    # pass nested lists (rows) and bbox; this uses the new Rust setter
    runtime_core.set_distance_field_py(
        arr.astype(np.float32).tolist(),
        float(meta.xmin),
        float(meta.xmax),
        float(meta.ymin),
        float(meta.ymax),
    )


def sample_distance_field(c: complex) -> float:
    """Sample the precomputed signed distance field at complex coordinate c.

    This implementation requires the Rust runtime-core sampler to be available
    and will raise if `runtime_core.sample_distance_field_py` is not exposed.

    Returns unsigned distance (abs of signed distance) as float.
    """
    if _distance_field_tensor is None or _distance_field_meta is None:
        load_distance_field()
    if not _distance_field_meta:
        raise RuntimeError("Distance field metadata not loaded")

    real = float(c.real)
    imag = float(c.imag)

    # Use Rust sampler if available
    if True:
        sampled_list = runtime_core.sample_distance_field_py([real], [imag])
        sampled = sampled_list[0]
        return abs(sampled)

    meta = _distance_field_meta
    xmin, xmax = meta.xmin, meta.xmax
    ymin, ymax = meta.ymin, meta.ymax
    res = meta.res

    # Map to pixel coords
    x_pos = (real - xmin) / (xmax - xmin) * (res - 1)
    y_pos = (imag - ymin) / (ymax - ymin) * (res - 1)

    x_idx = int(round(x_pos))
    y_idx = int(round(y_pos))
    x_idx = max(0, min(x_idx, res - 1))
    y_idx = max(0, min(y_idx, res - 1))

    arr = _distance_field_tensor.squeeze(0).squeeze(0)  # H,W
    sampled_val = float(arr[y_idx, x_idx].item())
    return abs(sampled_val)


def _sample_distance_field(c_complex: torch.Tensor) -> torch.Tensor:
    """Sample the precomputed signed distance field at complex coordinates c_complex.

    This implementation requires the Rust runtime-core sampler to be available
    and will raise if `runtime_core.sample_distance_field_py` is not exposed.

    Returns unsigned distances (abs of signed distance) as float tensor (N,).
    """
    if _distance_field_tensor is None or _distance_field_meta is None:
        load_distance_field()
    if not _distance_field_meta:
        raise RuntimeError("Distance field metadata not loaded")

    real = c_complex.real.to(torch.float32)
    imag = c_complex.imag.to(torch.float32)

    # Use Rust sampler if available for speed
    if True:
        xs = real.detach().cpu().numpy().tolist()
        ys = imag.detach().cpu().numpy().tolist()
        sampled_list = runtime_core.sample_distance_field_py(xs, ys)
        sampled = torch.tensor(
            sampled_list, dtype=torch.float32, device=c_complex.device
        )
        return sampled.abs()

    # Fallback: sample from the loaded numpy distance field in Python (nearest-neighbor)
    # Map complex coordinates to image indices
    meta = _distance_field_meta
    xmin, xmax = meta.xmin, meta.xmax
    ymin, ymax = meta.ymin, meta.ymax
    res = meta.res

    # Normalize to pixel coordinates (0..res-1)
    x_pos = (real - xmin) / (xmax - xmin) * (res - 1)
    y_pos = (imag - ymin) / (ymax - ymin) * (res - 1)

    x_idx = x_pos.round().to(torch.long)
    y_idx = y_pos.round().to(torch.long)

    # Clamp indices
    x_idx = x_idx.clamp(0, res - 1)
    y_idx = y_idx.clamp(0, res - 1)

    arr = _distance_field_tensor.squeeze(0).squeeze(0)  # H,W
    sampled_vals = arr[y_idx, x_idx].to(device=c_complex.device)
    return sampled_vals.abs()


class LossVisualMetrics:
    """Compute loss-facing visual metrics from rendered Julia sets."""

    def __init__(self):
        """Initialize metrics calculator."""
        pass

    def compute_all_metrics(
        self, image: np.ndarray, prev_image: Optional[np.ndarray] = None
    ) -> dict:
        """
        Compute loss-facing visual metrics from image.

        Args:
            image: Current image array (H, W, 3) or (H, W) in [0, 255] or [0, 1]
            prev_image: Previous image for temporal metrics (optional)

        Returns:
            Dictionary of metric values
        """
        # Normalize image to [0, 1] if needed
        if image.max() > 1.0:
            image = image.astype(np.float32) / 255.0

        # Convert to grayscale for some metrics
        if len(image.shape) == 3:
            gray = np.mean(image, axis=2)
        else:
            gray = image

        metrics = {}

        # Temporal change rate (loss metric)
        if prev_image is not None:
            if prev_image.max() > 1.0:
                prev_image = prev_image.astype(np.float32) / 255.0
            if len(prev_image.shape) == 3:
                prev_gray = np.mean(prev_image, axis=2)
            else:
                prev_gray = prev_image

            metrics["temporal_change"] = self._compute_temporal_change(gray, prev_gray)
        else:
            metrics["temporal_change"] = 0.0

        return metrics

    def _compute_temporal_change(
        self, current: np.ndarray, previous: np.ndarray
    ) -> float:
        """
        Compute temporal change rate between frames.

        Args:
            current: Current frame
            previous: Previous frame

        Returns:
            Change rate [0, 1]
        """
        # Ensure same shape
        if current.shape != previous.shape:
            # Resize if needed
            h, w = current.shape[:2]
            prev_resized = cv2.resize(previous, (w, h))
        else:
            prev_resized = previous

        # Compute difference
        diff = np.abs(current - prev_resized)
        change_rate = np.mean(diff)

        return float(change_rate)

    def render_julia_set(
        self,
        seed_real: float,
        seed_imag: float,
        width: int = 64,
        height: int = 64,
        zoom: float = 1.0,
        max_iter: int = 100,
        center_x: float = 0.0,
        center_y: float = 0.0,
    ) -> np.ndarray:
        """
        Render Julia set for metrics computation.
        This is a CPU-based renderer for training.

        Args:
            seed_real: Real part of Julia seed
            seed_imag: Imaginary part of Julia seed
            width: Image width
            height: Image height
            zoom: Zoom level
            max_iter: Maximum iterations
            center_x: Center X coordinate
            center_y: Center Y coordinate

        Returns:
            Rendered image array (H, W, 3) in [0, 255]
        """
        # Create coordinate arrays
        x = np.linspace(center_x - 2.0 / zoom, center_x + 2.0 / zoom, width)
        y = np.linspace(center_y - 2.0 / zoom, center_y + 2.0 / zoom, height)

        X, Y = np.meshgrid(x, y)
        C = X + 1j * Y

        # Julia set iteration
        Z = C.copy()
        iterations = np.zeros_like(C, dtype=np.int32)

        c = seed_real + 1j * seed_imag

        for i in range(max_iter):
            mask = np.abs(Z) <= 2.0
            Z[mask] = Z[mask] ** 2 + c
            iterations[mask] = i + 1

        # Normalize iterations to [0, 1]
        normalized = iterations.astype(np.float32) / max_iter

        # Apply color mapping (simple grayscale for now)
        # In practice, this would use hue/saturation/brightness from model
        image = (normalized * 255).astype(np.uint8)

        # Convert to RGB
        image_rgb = np.stack([image, image, image], axis=2)

        return image_rgb

    @staticmethod
    def mandelbrot_distance_estimate(
        c: torch.Tensor,
        max_iter=128,
        bailout=10.0,
        eps=1e-8,
    ) -> torch.Tensor:
        """Estimate distance to the Mandelbrot boundary for a batch of points.

        Accepts either:
        - a complex-valued tensor of shape (batch,) (dtype=torch.cfloat or torch.cdouble),
        - a real tensor of shape (batch, 2) where columns are (real, imag),
        - a real tensor of shape (batch,) (imag part assumed 0), or
        - a scalar/python complex which will be converted to a single-element tensor.

        This estimator samples the precomputed signed distance field via the
        runtime-core sampler (fast, non-differentiable). If the sampler is not
        available the function will raise an error instructing developers to
        rebuild the runtime-core Python extension.

        Returns a real float tensor of shape (batch,) with non-negative distances.
        """
        if c.dtype.is_complex:
            c_complex = c.view(-1)
        else:
            # handle (N, 2) real/imag pairs
            if c.dim() == 2 and c.shape[1] == 2:
                real = c[:, 0].to(torch.get_default_dtype())
                imag = c[:, 1].to(torch.get_default_dtype())
                c_complex = torch.complex(real, imag).to(torch.complex64)
            else:
                raise TypeError(
                    "Unsupported tensor shape for mandelbrot_distance_estimate: expected (N,2) or (N,)"
                )

        sampled = _sample_distance_field(c_complex)
        return sampled.clamp_min(0.0)
