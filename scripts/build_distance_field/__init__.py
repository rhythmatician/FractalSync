"""
Build a signed distance field (SDF) for the Mandelbrot set.

Outputs:
- <out>.npy  : float32 signed distance field (positive outside, negative inside)
- <out>.bin  : raw little-endian float32 array (row-major), same as .npy
- <out>.json : metadata (bounds, res, dx/dy, max_iter, bailout, etc.)
- <out>.png  : visualization preview (8-bit), optional

Distance definition:
- Inside mask is defined by "did not escape by max_iter" under z_{n+1} = z_n^2 + c, z_0 = 0.
- Signed distance computed by Euclidean distance transforms with correct anisotropic sampling:
  signed = dist_to_inside(outside points) - dist_to_outside(inside points)
  => positive outside, negative inside, 0 on boundary (up to pixelization).

GPU acceleration:
- Uses OpenGL compute shaders via moderngl (OpenGL 4.3+ required) to compute the inside/outside mask.
- Tiles large resolutions using batches up to 2048x2048 (configurable).
- Falls back to CPU if GPU path fails or --cpu is supplied.

Dependencies:
- numpy
- scipy
- pillow (for PNG output)
- moderngl (for GPU path)

Example:
  python scripts/build_distance_field.py \
    --out data/mandelbrot_distance_512 \
    --res 512 \
    --xmin -2.5 --xmax 1.5 --ymin -2.0 --ymax 2.0 \
    --max-iter 2048 \
    --bailout 2.0 \
    --png

Notes / caveats:
- The "inside" mask is an approximation (iteration-limited). Near the boundary, classification errors can occur.
  If you need more reliable distances near the boundary, raise --max-iter and/or consider generating an
  uncertainty band (not implemented here).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage


# -------------------------
# Args / CLI
# -------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out", type=Path, required=True, help="Output base path (suffix optional)."
    )
    p.add_argument(
        "--res", type=int, default=2048, help="Square resolution (res x res)."
    )

    p.add_argument("--xmin", type=float, default=-2.5)
    p.add_argument("--xmax", type=float, default=1.5)
    p.add_argument("--ymin", type=float, default=-2.0)
    p.add_argument("--ymax", type=float, default=2.0)

    p.add_argument("--max-iter", type=int, default=2048)
    p.add_argument("--bailout", type=float, default=2.0)

    # GPU controls
    p.add_argument(
        "--cpu", action="store_true", help="Force CPU mask generation (no OpenGL)."
    )
    p.add_argument(
        "--batch",
        type=int,
        default=2048,
        help="Max GPU batch/tile size (square). Must be <= 2048 per request.",
    )
    p.add_argument(
        "--local-size",
        type=int,
        default=16,
        help="Compute shader local workgroup size.",
    )

    # PNG output
    p.add_argument("--png", action="store_true", help="Also output a preview PNG.")
    p.add_argument(
        "--png-scale",
        type=float,
        default=0.15,
        help=(
            "Controls contrast in PNG preview: maps signed distance via tanh(signed/png_scale). "
            "Smaller -> more contrast near boundary."
        ),
    )
    p.add_argument(
        "--supersample",
        type=int,
        default=1,
        help=(
            "Supersample factor for mask generation (1 = no supersampling). "
            "When >1 the mask is rendered at res*supersample, EDT is computed at that "
            "resolution and the resulting SDF is downsampled to the target res."
        ),
    )
    return p.parse_args()


def normalize_out_base(out: Path) -> Path:
    # If user supplies "foo.npy", treat base as "foo"
    return out.with_suffix("") if out.suffix else out


# -------------------------
# Core math: mask + SDF
# -------------------------


def build_mask_cpu(
    res: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    max_iter: int,
    bailout: float,
) -> np.ndarray:
    w = res
    h = res
    xs = np.linspace(xmin, xmax, w, dtype=np.float64)
    ys = np.linspace(ymin, ymax, h, dtype=np.float64)
    X, Y = np.meshgrid(xs, ys)
    C = X + 1j * Y

    Z = np.zeros_like(C)
    escaped = np.zeros(C.shape, dtype=bool)

    bailout2 = float(bailout) * float(bailout)

    for _ in range(max_iter):
        mask = ~escaped
        if not mask.any():
            break
        Zm = Z[mask]
        Cm = C[mask]
        Zm = Zm * Zm + Cm
        Z[mask] = Zm

        # Avoid sqrt: abs(Z)^2 = re^2 + im^2
        zz = (Zm.real * Zm.real + Zm.imag * Zm.imag) > bailout2
        escaped_now = np.zeros_like(escaped)
        escaped_now[mask] = zz
        escaped |= escaped_now

    inside = ~escaped
    return inside.astype(np.bool_)


def build_signed_distance(
    inside_mask: np.ndarray,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> Tuple[np.ndarray, float, float]:
    """
    Returns:
      signed: float32 SDF (positive outside, negative inside), in complex-plane units
      dx, dy: pixel spacing in x and y (complex-plane units per pixel)
    """
    h, w = inside_mask.shape
    if w <= 1 or h <= 1:
        raise ValueError("res must be > 1")

    dx = (xmax - xmin) / float(w - 1)
    dy = (ymax - ymin) / float(h - 1)

    # distance_transform_edt: for each True element, distance to nearest False
    dist_to_outside = ndimage.distance_transform_edt(inside_mask, sampling=(dy, dx))
    dist_to_inside = ndimage.distance_transform_edt(~inside_mask, sampling=(dy, dx))

    signed = (dist_to_inside - dist_to_outside).astype(np.float32)
    return signed, dx, dy


# -------------------------
# GPU: OpenGL compute shader mask generation (moderngl)
# -------------------------
shader_path = Path(__file__).parent / "mandelbrot_mask.comp"
with open(shader_path, "r") as f:
    _COMPUTE_SRC = f.read()


def _make_compute_src(local_size: int) -> str:
    if local_size <= 0:
        raise ValueError("--local-size must be > 0")
    return _COMPUTE_SRC.replace("LOCAL_SIZE", str(local_size))


def _try_build_mask_gpu(
    res: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    max_iter: int,
    bailout: float,
    batch: int,
    local_size: int,
) -> Optional[np.ndarray]:
    """
    Returns inside mask as bool array if successful, else None.

    Notes:
    - Requires moderngl and OpenGL 4.3+ (compute shaders).
    - Uses tiled batches for memory and dispatch sizing.
    """
    if batch <= 0 or batch > 2048:
        raise ValueError("--batch must be in [1, 2048]")

    try:
        import moderngl  # type: ignore
    except Exception:
        return None

    try:
        # Create a standalone context (headless). On Windows this usually works via WGL.
        # On some headless Linux environments you may need EGL/OSMesa setup.
        ctx = moderngl.create_standalone_context(require=430)
    except Exception:
        return None

    comp_src = _make_compute_src(local_size)
    try:
        prog = ctx.compute_shader(comp_src)
    except Exception:
        return None

    bailout2 = float(bailout) * float(bailout)

    inside = np.zeros((res, res), dtype=np.bool_)

    # Tile over full image
    for y0 in range(0, res, batch):
        tile_h = min(batch, res - y0)
        for x0 in range(0, res, batch):
            tile_w = min(batch, res - x0)

            # SSBO for tile mask (uint32)
            out_count = tile_w * tile_h
            ssbo = ctx.buffer(reserve=out_count * 4)

            # Bind SSBO at binding = 0
            ssbo.bind_to_storage_buffer(binding=0)

            # Set uniforms
            prog["u_tile_w"].value = int(tile_w)
            prog["u_tile_h"].value = int(tile_h)
            prog["u_tile_x0"].value = int(x0)
            prog["u_tile_y0"].value = int(y0)
            prog["u_res"].value = int(res)

            # doubles: moderngl supports float uniforms; for double uniforms it depends on driver.
            # Many drivers accept .value for double uniforms; if not, GPU path will fail.
            prog["u_xmin"].value = xmin  # FIXME: "float" is not assignable to "int"
            prog["u_xmax"].value = xmax  # FIXME: "float" is not assignable to "int"
            prog["u_ymin"].value = ymin  # FIXME: "float" is not assignable to "int"
            prog["u_ymax"].value = ymax  # FIXME: "float" is not assignable to "int"

            prog["u_max_iter"].value = int(max_iter)
            prog["u_bailout2"].value = bailout2  # FIXME: float not assignable to int

            # Dispatch
            groups_x = (tile_w + local_size - 1) // local_size
            groups_y = (tile_h + local_size - 1) // local_size
            prog.run(group_x=groups_x, group_y=groups_y, group_z=1)

            # Read back
            raw = ssbo.read()
            arr = np.frombuffer(raw, dtype=np.uint32, count=out_count).reshape(
                (tile_h, tile_w)
            )
            inside[y0 : y0 + tile_h, x0 : x0 + tile_w] = arr != 0

            ssbo.release()

    ctx.release()
    return inside


def build_mask(
    res: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    max_iter: int,
    bailout: float,
    force_cpu: bool,
    batch: int,
    local_size: int,
) -> Tuple[np.ndarray, str]:
    if force_cpu:
        return (
            build_mask_cpu(res, xmin, xmax, ymin, ymax, max_iter, bailout),
            "cpu",
        )

    inside = _try_build_mask_gpu(
        res, xmin, xmax, ymin, ymax, max_iter, bailout, batch, local_size
    )
    if inside is not None:
        return inside, "gpu"

    return (
        build_mask_cpu(res, xmin, xmax, ymin, ymax, max_iter, bailout),
        "cpu(fallback)",
    )


# -------------------------
# PNG preview
# -------------------------


def save_preview_png(path: Path, signed: np.ndarray, png_scale: float) -> None:
    """
    Absolute-distance visualization (grayscale):
      v = clamp(|signed| / png_scale, 0..1)
      0 (boundary) -> black
      far from boundary -> white
    """
    if png_scale <= 0:
        raise ValueError("--png-scale must be > 0")

    d = np.abs(signed.astype(np.float32))
    v = np.clip(d / float(png_scale), 0.0, 1.0)
    img = (v * 255.0).astype(np.uint8)
    Image.fromarray(img, mode="L").save(path)


# -------------------------
# Main
# -------------------------


def main() -> None:
    args = parse_args()
    out_base = normalize_out_base(args.out)

    res = int(args.res)
    if res <= 1:
        raise ValueError("--res must be > 1")

    max_batch_size = 1024 / int(args.supersample)
    if args.batch > max_batch_size:
        print(f"Warning: Reducing --batch from {args.batch} to {int(max_batch_size)} ")
        args.batch = int(max_batch_size)

    print(
        f"Building inside mask (res={res}) in box "
        f"x=[{args.xmin},{args.xmax}] y=[{args.ymin},{args.ymax}] "
        f"max_iter={args.max_iter} bailout={args.bailout}"
    )

    inside, mode = build_mask(
        res=res,
        xmin=args.xmin,
        xmax=args.xmax,
        ymin=args.ymin,
        ymax=args.ymax,
        max_iter=args.max_iter,
        bailout=args.bailout,
        force_cpu=bool(args.cpu),
        batch=int(args.batch),
        local_size=int(args.local_size),
    )
    print(f"Mask generation mode: {mode}")

    # If supersampling is enabled, render mask at higher resolution, compute EDT, then downsample the SDF
    if args.supersample <= 1:
        print("Computing signed distance transform (CPU, scipy)...")
        signed, dx, dy = build_signed_distance(
            inside, args.xmin, args.xmax, args.ymin, args.ymax
        )
    else:
        ss = int(args.supersample)
        high_res = res * ss
        print(f"Supersampling mask at {ss}x => high_res={high_res}")
        high_mask, mode_high = build_mask(
            res=high_res,
            xmin=args.xmin,
            xmax=args.xmax,
            ymin=args.ymin,
            ymax=args.ymax,
            max_iter=args.max_iter,
            bailout=args.bailout,
            force_cpu=bool(args.cpu),
            batch=int(args.batch),
            local_size=int(args.local_size),
        )
        print(f"Mask generation mode (high-res): {mode_high}")
        print("Computing signed distance transform at high resolution (CPU, scipy)...")
        signed_high, dx_high, dy_high = build_signed_distance(
            high_mask, args.xmin, args.xmax, args.ymin, args.ymax
        )
        # Downsample signed_high to target res using cubic interpolation
        zoom_factor = 1.0 / float(ss)
        signed = ndimage.zoom(signed_high, (zoom_factor, zoom_factor), order=3)
        # Ensure final shape matches (res,res)
        signed = signed[:res, :res]
        dx = (args.xmax - args.xmin) / float(res - 1)
        dy = (args.ymax - args.ymin) / float(res - 1)
        mode = f"supersampled_{mode_high}"

    out_base.parent.mkdir(parents=True, exist_ok=True)

    npy_path = out_base.with_suffix(".npy")
    bin_path = out_base.with_suffix(".bin")
    json_path = out_base.with_suffix(".json")
    png_path = out_base.with_suffix(".png")

    np.save(npy_path, signed)
    with open(bin_path, "wb") as f:
        f.write(signed.astype("<f4").tobytes())

    meta = {
        "xmin": float(args.xmin),
        "xmax": float(args.xmax),
        "ymin": float(args.ymin),
        "ymax": float(args.ymax),
        "res": int(res),
        "dx": float(dx),
        "dy": float(dy),
        "max_iter": int(args.max_iter),
        "bailout": float(args.bailout),
        "mask_mode": mode,
        "layout": "row-major; signed[y][x]; y increases with row index (ymin->ymax)",
        "sign_convention": "positive outside, negative inside",
        "note": "Inside mask is iteration-limited; near-boundary classification errors are possible.",
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if args.png:
        print("Writing PNG preview...")
        save_preview_png(png_path, signed, png_scale=float(args.png_scale))

    print(f"Saved: {npy_path}")
    print(f"Saved: {bin_path}")
    print(f"Saved: {json_path}")
    if args.png:
        print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
