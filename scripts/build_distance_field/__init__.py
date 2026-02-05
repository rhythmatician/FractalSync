"""
Build a signed distance field (SDF) for the Mandelbrot set.

Outputs:
- <out>.bin  : raw little-endian float32 array (row-major) — canonical runtime artifact
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
  python -m scripts.build_distance_field \
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
- Optional DEM exterior replacement is available and enabled by default; use `--no-dem` to disable.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from numpy.typing import NDArray

import numpy as np
from PIL import Image
from scipy import ndimage

# -------------------------
# Args / CLI
# -------------------------


@dataclass
class Coords:
    xmin: float
    xmax: float
    ymin: float
    ymax: float


@dataclass
class DemParams:
    enabled: bool
    bailout: float
    max_iter: int
    eps: float
    blend: float
    band: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--out",
        type=Path,
        required=False,
        help="Output base path (suffix optional). Defaults to 'runtime-core/data'.",
        default=Path("runtime-core/data/mandelbrot_distance"),
    )
    p.add_argument(
        "--res", type=int, default=1024, help="Square resolution (res x res)."
    )
    # Real (x) range: \([-2.0,\ 0.4711]\).Imaginary (y) range: Approximately \([-1.122,\ 1.122]\).Center: \((-0.875,\ 0)\).Dimensions: \(2.25\times 2.245\).

    p.add_argument("--max-iter", type=int, default=2048)
    p.add_argument("--bailout", type=float, default=2.0)

    # GPU controls
    p.add_argument(
        "--cpu", action="store_true", help="Force CPU mask generation (no OpenGL)."
    )
    p.add_argument(
        "--batch",
        type=int,
        default=1024,
        help="Max GPU batch/tile size (square). Must be <= 2048 per request.",
    )
    p.add_argument(
        "--local-size",
        type=int,
        default=16,
        help="Compute shader local workgroup size.",
    )

    # PNG output
    p.add_argument(
        "--png", action="store_true", help="Also output a preview PNG.", default=True
    )
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
        default=2,
        help=(
            "Supersample factor for mask generation (1 = no supersampling). "
            "When >1 the mask is rendered at res*supersample, EDT is computed at that "
            "resolution and the resulting SDF is downsampled to the target res."
        ),
    )

    # DEM exterior replacement (enabled by default; use --no-dem to disable)
    p.add_argument(
        "--no-dem",
        action="store_true",
        help="Disable DEM exterior replacement (enabled by default).",
    )
    p.add_argument(
        "--dem-bailout",
        type=float,
        default=1e4,
        help="Escape radius to use for DEM exterior computation (default: 1e4).",
    )
    p.add_argument(
        "--dem-max-iter",
        type=int,
        default=None,
        help=(
            "Max iterations for DEM derivative loop. If omitted, falls back to --max-iter. "
            "Increase for higher-fidelity DEM."
        ),
    )
    p.add_argument(
        "--dem-eps",
        type=float,
        default=1e-12,
        help="Derivative magnitude threshold below which DEM is considered unstable.",
    )
    p.add_argument(
        "--dem-blend",
        type=float,
        default=0.5,
        help=(
            "Blend factor between DEM and EDT for exterior pixels (0..1). "
            "0 = keep EDT, 1 = pure DEM."
        ),
    )
    p.add_argument(
        "--dem-band",
        type=float,
        default=0.5,
        help=(
            "Only compute DEM for exterior pixels whose EDT <= dem_band (in plane units). "
            "Smaller -> less work and less noise near far exterior points."
        ),
    )

    p.add_argument(
        "--dem-gpu",
        action="store_true",
        help="Use GPU-based DEM compute shader (experimental).",
        default=True,
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
    coords: Coords,
    max_iter: int,
    bailout: float,
) -> NDArray:
    w = res
    h = res
    xs = np.linspace(coords.xmin, coords.xmax, w, dtype=np.float64)
    ys = np.linspace(coords.ymin, coords.ymax, h, dtype=np.float64)
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
    inside_mask: NDArray,
    coords: Coords,
) -> Tuple[NDArray[np.float32], float, float]:
    """
    Returns:
      signed: float32 SDF (positive outside, negative inside), in complex-plane units
      dx, dy: pixel spacing in x and y (complex-plane units per pixel)
    """
    h, w = inside_mask.shape
    if w <= 1 or h <= 1:
        raise ValueError("res must be > 1")

    dx = (coords.xmax - coords.xmin) / float(w - 1)
    dy = (coords.ymax - coords.ymin) / float(h - 1)

    # distance_transform_edt: for each True element, distance to nearest False
    dist_to_outside = ndimage.distance_transform_edt(inside_mask, sampling=(dy, dx))
    dist_to_inside = ndimage.distance_transform_edt(~inside_mask, sampling=(dy, dx))

    signed = (dist_to_inside - dist_to_outside).astype(np.float32)  # type: ignore
    return signed, dx, dy


# -------------------------
# DEM: CPU exterior distance estimation helpers
# -------------------------


def compute_dem_for_point(c: complex, dem_params: DemParams) -> Optional[float]:
    """Compute the exterior DEM (approximate unsigned distance) for c.

    Returns DEM in complex-plane units or None if DEM is unstable or c is interior.
    Uses derivative iteration: dz_{n+1} = 2*z_n*dz_n + 1, starting with z=0, dz=1.
    """
    z = 0 + 0j
    dz = 1 + 0j
    for _ in range(dem_params.max_iter):
        # derivative update uses current z_n
        dz = 2.0 * z * dz + 1.0
        z = z * z + c
        absz = abs(z)
        if absz > dem_params.bailout:
            absdz = abs(dz)
            if absdz < dem_params.eps or absdz == 0.0:
                return None
            # distance estimate (exterior DEM formula)
            try:
                de = 2.0 * absz * math.log(absz) / absdz
            except Exception:
                return None
            if not math.isfinite(de) or de <= 0.0:
                return None
            return float(de)
    # did not escape -> interior or unknown
    return None


def apply_dem_to_sdf(
    signed: NDArray,
    coords: Coords,
    dx: float,
    dy: float,
    dem_params: DemParams,
) -> NDArray:
    """Apply DEM exterior replacement using per-point CPU DEM computations.

    Only processes exterior pixels with EDT <= dem_params.band. Returns new signed array.
    """
    h, w = signed.shape
    out = signed.copy()

    # clamp interior to be <= 0
    out[out < 0.0] = np.minimum(out[out < 0.0], 0.0)

    # Iterate over candidate exterior pixels
    for y in range(h):
        yr = coords.ymin + y * dy
        for x in range(w):
            v = signed[y, x]
            if v <= 0.0:
                continue  # interior
            if v > dem_params.band:
                continue  # skip far exterior points
            xr = coords.xmin + x * dx
            c = complex(xr, yr)
            de = compute_dem_for_point(c, dem_params)
            if de is None:
                continue  # fallback to EDT
            # blend DEM and EDT
            new_val = dem_params.blend * float(de) + (1.0 - dem_params.blend) * float(v)
            out[y, x] = float(new_val)

    return out


def apply_dem_from_dem_array(
    signed: NDArray,
    dem_arr: NDArray,
    dem_params: DemParams,
) -> NDArray:
    """Apply DEM replacement from a precomputed dem array (shader output).

    dem_arr uses sentinel values:
      >0.0 : DEM distance (plane units)
      -1.0 : failure/unstable (leave EDT unchanged)
      -2.0 : interior / did not escape (leave EDT unchanged)

    Only replaces EDT where dem_arr > 0 and EDT <= dem_params.band.
    """
    h, w = signed.shape
    if dem_arr.shape != (h, w):
        raise ValueError("dem_arr shape must match signed shape")

    out = signed.copy()

    for y in range(h):
        for x in range(w):
            edt = signed[y, x]
            if edt <= 0.0:
                continue
            if edt > dem_params.band:
                continue
            de = dem_arr[y, x]
            if de > 0.0:
                out[y, x] = dem_params.blend * float(de) + (
                    1.0 - dem_params.blend
                ) * float(edt)
            else:
                # sentinel or interior: do not change (fallback to EDT)
                continue
    return out


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


def _alloc_and_set_tile(
    ctx,
    prog,
    res: int,
    x0: int,
    y0: int,
    tile_w: int,
    tile_h: int,
    coords: Coords,
    max_iter: int,
    bailout2: float,
    local_size: int,
    extra_uniforms: Optional[dict] = None,
):
    """Allocate SSBO for a tile, bind it and set common uniforms.

    Returns (ssbo, out_count, groups_x, groups_y).
    """
    out_count = tile_w * tile_h
    ssbo = ctx.buffer(reserve=out_count * 4)
    ssbo.bind_to_storage_buffer(binding=0)

    prog["u_tile_w"].value = int(tile_w)
    prog["u_tile_h"].value = int(tile_h)
    prog["u_tile_x0"].value = int(x0)
    prog["u_tile_y0"].value = int(y0)
    prog["u_res"].value = int(res)

    # Coordinate bounds (float/double uniforms)
    prog["u_xmin"].value = coords.xmin  # type: ignore
    prog["u_xmax"].value = coords.xmax  # type: ignore
    prog["u_ymin"].value = coords.ymin  # type: ignore
    prog["u_ymax"].value = coords.ymax  # type: ignore

    prog["u_max_iter"].value = int(max_iter)
    prog["u_bailout2"].value = bailout2  # type: ignore

    if extra_uniforms:
        for k, v in extra_uniforms.items():
            prog[k].value = v  # type: ignore

    groups_x = (tile_w + local_size - 1) // local_size
    groups_y = (tile_h + local_size - 1) // local_size
    return ssbo, out_count, groups_x, groups_y


def _try_build_mask_gpu(
    res: int,
    coords: Coords,
    max_iter: int,
    bailout: float,
    batch: int,
    local_size: int,
) -> Optional[NDArray]:
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

            ssbo, out_count, groups_x, groups_y = _alloc_and_set_tile(
                ctx,
                prog,
                res,
                x0,
                y0,
                tile_w,
                tile_h,
                coords,
                max_iter,
                bailout2,
                local_size,
            )

            # Dispatch
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


# -------------------------
# DEM GPU compute shader path (mirrors mask tile dispatch)
# -------------------------

dem_shader_path = Path(__file__).parent / "mandelbrot_dem.comp"
with open(dem_shader_path, "r") as f:
    _DEM_COMPUTE_SRC = f.read()


def _make_dem_compute_src(local_size: int) -> str:
    if local_size <= 0:
        raise ValueError("--local-size must be > 0")
    return _DEM_COMPUTE_SRC.replace("LOCAL_SIZE", str(local_size))


def _try_compute_dem_gpu(
    res: int,
    coords: Coords,
    max_iter: int,
    dem_bailout: float,
    dem_eps: float,
    batch: int,
    local_size: int,
) -> Optional[NDArray]:
    """
    Returns a float32 array with per-pixel DEM estimates if successful.

    The shader writes:
      -2.0 -> interior (did not escape)
      -1.0 -> failure/unstable
      >0.0 -> computed DEM distance

    Returns None on failure to run (moderngl unavailable or compilation errors).
    """
    if batch <= 0 or batch > 2048:
        raise ValueError("--batch must be in [1, 2048]")

    try:
        import moderngl  # type: ignore
    except Exception:
        return None

    try:
        ctx = moderngl.create_standalone_context(require=430)
    except Exception:
        return None

    dem_src = _make_dem_compute_src(local_size)
    try:
        prog = ctx.compute_shader(dem_src)
    except Exception:
        return None

    bailout2 = float(dem_bailout) * float(dem_bailout)

    dem_out = np.full((res, res), -2.0, dtype=np.float32)

    # Tile over full image
    for y0 in range(0, res, batch):
        tile_h = min(batch, res - y0)
        for x0 in range(0, res, batch):
            tile_w = min(batch, res - x0)

            ssbo, out_count, groups_x, groups_y = _alloc_and_set_tile(
                ctx,
                prog,
                res,
                x0,
                y0,
                tile_w,
                tile_h,
                coords,
                max_iter,
                bailout2,
                local_size,
                extra_uniforms={"u_dem_eps": float(dem_eps)},
            )

            # Dispatch
            prog.run(group_x=groups_x, group_y=groups_y, group_z=1)

            raw = ssbo.read()
            arr = np.frombuffer(raw, dtype=np.float32, count=out_count).reshape(
                (tile_h, tile_w)
            )
            dem_out[y0 : y0 + tile_h, x0 : x0 + tile_w] = arr

            ssbo.release()

    ctx.release()
    return dem_out


def build_mask(
    res: int,
    coords: Coords,
    max_iter: int,
    bailout: float,
    force_cpu: bool,
    batch: int,
    local_size: int,
) -> Tuple[NDArray, str]:
    if force_cpu:
        return (
            build_mask_cpu(res, coords, max_iter, bailout),
            "cpu",
        )

    inside = _try_build_mask_gpu(res, coords, max_iter, bailout, batch, local_size)
    if inside is not None:
        return inside, "gpu"

    return (
        build_mask_cpu(res, coords, max_iter, bailout),
        "cpu(fallback)",
    )


# -------------------------
# PNG preview
# -------------------------


def save_preview_png(path: Path, signed: NDArray, png_scale: float) -> None:
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
    coords = Coords(
        xmin=-2.0,
        xmax=0.4711,
        ymin=-1.122,
        ymax=1.122,
    )
    args = parse_args()
    out_base = normalize_out_base(Path(f"{args.out}_{args.res}"))

    res = int(args.res)
    if res <= 1:
        raise ValueError("--res must be > 1")

    supersample = int(args.supersample)
    max_batch_size = max(1, 2048 // supersample)
    if int(args.batch) > max_batch_size:
        print(f"Warning: Reducing --batch from {args.batch} to {max_batch_size}")
        args.batch = max_batch_size

    print(
        f"Building inside mask (res={res}) in box "
        f"x=[{coords.xmin},{coords.xmax}] y=[{coords.ymin},{coords.ymax}] "
        f"max_iter={args.max_iter} bailout={args.bailout}"
    )

    inside, mode = build_mask(
        res=res,
        coords=coords,
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
        signed, dx, dy = build_signed_distance(inside, coords)
    else:
        ss = int(args.supersample)
        high_res = res * ss
        print(f"Supersampling mask at {ss}x => high_res={high_res}")
        high_mask, mode_high = build_mask(
            res=high_res,
            coords=coords,
            max_iter=args.max_iter,
            bailout=args.bailout,
            force_cpu=bool(args.cpu),
            batch=int(args.batch),
            local_size=int(args.local_size),
        )
        print(f"Mask generation mode (high-res): {mode_high}")
        print("Computing signed distance transform at high resolution (CPU, scipy)...")
        signed_high, _, _ = build_signed_distance(high_mask, coords)
        # Downsample signed_high to target res using cubic interpolation
        zoom_factor = 1.0 / float(ss)

        signed: NDArray[np.float32] = ndimage.zoom(  # type: ignore
            signed_high, (zoom_factor, zoom_factor), order=3
        )  # pyright: ignore[reportAssignmentType]
        # Ensure final shape matches (res,res)
        signed = signed[:res, :res]
        dx = (coords.xmax - coords.xmin) / float(res - 1)
        dy = (coords.ymax - coords.ymin) / float(res - 1)
        mode = f"supersampled_{mode_high}"

    # Apply DEM exterior replacement by default unless explicitly disabled
    dem_enabled = not bool(args.no_dem)
    dem_params = DemParams(
        enabled=dem_enabled,
        bailout=float(args.dem_bailout),
        max_iter=(
            int(args.dem_max_iter)
            if args.dem_max_iter is not None
            else int(args.max_iter)
        ),
        eps=float(args.dem_eps),
        blend=float(args.dem_blend),
        band=float(args.dem_band),
    )
    if dem_enabled:
        if getattr(args, "dem_gpu", False):
            print("Attempting GPU DEM (experimental)...")
            dem_out = _try_compute_dem_gpu(
                res=res,
                coords=coords,
                max_iter=dem_params.max_iter,
                dem_bailout=dem_params.bailout,
                dem_eps=dem_params.eps,
                batch=int(args.batch),
                local_size=int(args.local_size),
            )
            if dem_out is not None:
                print("Applying DEM from GPU output...")
                signed = apply_dem_from_dem_array(signed, dem_out, dem_params)
                mode = f"{mode}+dem_gpu"
            else:
                print("GPU DEM unavailable; falling back to CPU DEM")
                signed = apply_dem_to_sdf(signed, coords, dx, dy, dem_params)
                mode = f"{mode}+dem_cpu"
        else:
            print("Applying DEM exterior replacement (CPU)...")
            signed = apply_dem_to_sdf(
                signed,
                coords,
                dx,
                dy,
                dem_params,
            )
            mode = f"{mode}+dem"

    out_base.parent.mkdir(parents=True, exist_ok=True)

    bin_path = out_base.with_suffix(".bin")
    json_path = out_base.with_suffix(".json")
    png_path = out_base.with_suffix(".png")

    # Write canonical runtime binary (.bin)
    with open(bin_path, "wb") as f:
        f.write(signed.astype("<f4").tobytes())

    meta = {
        "xmin": float(coords.xmin),
        "xmax": float(coords.xmax),
        "ymin": float(coords.ymin),
        "ymax": float(coords.ymax),
        "res": int(res),
        "dx": float(dx),
        "dy": float(dy),
        "max_iter": int(args.max_iter),
        "bailout": float(args.bailout),
        "mask_mode": mode,
        "dem": dem_params.__dict__,
        "layout": "row-major; signed[y][x]; y increases with row index (ymin->ymax)",
        "sign_convention": "positive outside, negative inside",
        "note": "Inside mask is iteration-limited; near-boundary classification errors are possible.",
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if args.png:
        print("Writing PNG preview...")
        save_preview_png(png_path, signed, png_scale=float(args.png_scale))

    print(f"Saved: {bin_path}")
    print(f"Saved: {json_path}")
    if args.png:
        print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
