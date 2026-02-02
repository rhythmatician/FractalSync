"""Build a signed distance field for the Mandelbrot set.

Generates a signed distance field (float32) where positive values indicate
outside distance to the boundary and negative values indicate inside depth.
Saves a .npy file and a metadata .json alongside it.

Usage (from repo root):
    python scripts/build_distance_field.py --out data/mandelbrot_distance_512.npy --res 512 --xmin -2.5 --xmax 1.5 --ymin -2.0 --ymax 2.0 --max-iter 1024

For high-resolution builds, GPU acceleration (OpenGL) is recommended:
    python scripts/build_distance_field.py --out data/mandelbrot_distance_2048.npy --res 2048 --max-iter 1024 --use-gpu

This script depends on numpy and scipy. GPU acceleration requires moderngl.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--res", type=int, default=2048)
    p.add_argument("--xmin", type=float, default=-2.5)
    p.add_argument("--xmax", type=float, default=1.5)
    p.add_argument("--ymin", type=float, default=-2.0)
    p.add_argument("--ymax", type=float, default=2.0)
    p.add_argument("--max-iter", type=int, default=1024)
    p.add_argument("--bailout", type=float, default=4.0)
    p.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU acceleration (OpenGL) for high-resolution builds",
    )
    return p.parse_args()


def build_mask(
    res: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    max_iter: int,
    bailout: float,
):
    w = res
    h = res
    xs = np.linspace(xmin, xmax, w)
    ys = np.linspace(ymin, ymax, h)
    X, Y = np.meshgrid(xs, ys)
    C = X + 1j * Y

    Z = np.zeros_like(C)
    escaped = np.zeros(C.shape, dtype=bool)

    for i in range(max_iter):
        mask = ~escaped
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        escaped_now = np.abs(Z) > bailout
        escaped = escaped | escaped_now
        if escaped.all():
            break

    inside = ~escaped
    return inside.astype(np.bool_)


def build_mask_gpu(
    res: int,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    max_iter: int,
    bailout: float,
):
    """
    GPU-accelerated Mandelbrot mask generation using OpenGL.
    
    Returns a boolean mask where True indicates points inside the Mandelbrot set.
    Falls back to CPU implementation if GPU is unavailable.
    """
    try:
        import moderngl
    except ImportError:
        logger.warning("moderngl not available, falling back to CPU implementation")
        return build_mask(res, xmin, xmax, ymin, ymax, max_iter, bailout)

    try:
        # Create headless OpenGL context
        ctx = moderngl.create_context(standalone=True)
        logger.info(f"GPU context initialized: {ctx.info.get('GL_VENDOR', 'unknown')}")

        # Vertex shader (fullscreen quad)
        vertex_shader = """
        #version 330

        out VS_OUTPUT {
            vec2 uv;
        } vs_out;

        void main() {
            vec2 pos = vec2(gl_VertexID & 1, (gl_VertexID >> 1) & 1) * 2.0 - 1.0;
            vs_out.uv = pos * 0.5 + 0.5;
            gl_Position = vec4(pos, 0.0, 1.0);
        }
        """

        # Fragment shader (Mandelbrot computation)
        fragment_shader = """
        #version 330

        in VS_OUTPUT {
            vec2 uv;
        } fs_in;

        out vec4 color;

        uniform vec2 bounds_min;  // (xmin, ymin)
        uniform vec2 bounds_max;  // (xmax, ymax)
        uniform int max_iter;
        uniform float bailout;

        void main() {
            // Map UV [0,1] to complex plane bounds
            vec2 c = mix(bounds_min, bounds_max, fs_in.uv);
            vec2 z = vec2(0.0, 0.0);

            int iter = 0;
            for (int i = 0; i < 4096; i++) {
                if (i >= max_iter) break;
                
                // Check if escaped
                float zx2 = z.x * z.x;
                float zy2 = z.y * z.y;
                if (zx2 + zy2 > bailout) {
                    break;
                }
                
                // z = z^2 + c
                z = vec2(zx2 - zy2, 2.0 * z.x * z.y) + c;
                iter++;
            }

            // Output 1.0 if inside (didn't escape), 0.0 if outside
            float inside = (iter >= max_iter) ? 1.0 : 0.0;
            color = vec4(inside, inside, inside, 1.0);
        }
        """

        program = ctx.program(
            vertex_shader=vertex_shader, fragment_shader=fragment_shader
        )

        # Set uniforms
        program["bounds_min"] = (xmin, ymin)
        program["bounds_max"] = (xmax, ymax)
        program["max_iter"] = max_iter
        program["bailout"] = bailout

        # Create VAO for rendering fullscreen quad
        vao = ctx.vertex_array(program, [])

        # Create framebuffer with single-channel texture
        texture = ctx.texture((res, res), 4)  # RGBA
        fbo = ctx.framebuffer([texture])

        # Render to framebuffer
        fbo.use()
        ctx.viewport = (0, 0, res, res)
        ctx.clear(0.0, 0.0, 0.0, 1.0)
        vao.render(mode=ctx.TRIANGLE_STRIP, vertices=4)

        # Read back data
        data = fbo.read(components=4)
        image = np.frombuffer(data, dtype=np.uint8).reshape((res, res, 4))

        # Extract first channel and flip (OpenGL origin is bottom-left)
        inside_float = np.flip(image[:, :, 0], axis=0)
        inside = inside_float > 127  # Threshold at 0.5

        # Cleanup
        fbo.release()
        texture.release()
        vao.release()
        program.release()
        ctx.release()

        return inside.astype(np.bool_)

    except Exception as e:
        logger.warning(f"GPU rendering failed ({e}), falling back to CPU")
        return build_mask(res, xmin, xmax, ymin, ymax, max_iter, bailout)


def build_signed_distance(inside_mask: np.ndarray, pixel_scale: float):
    # distance to outside for interior points
    dist_out = ndimage.distance_transform_edt(~inside_mask) * pixel_scale
    # distance to inside for outside points
    dist_in = ndimage.distance_transform_edt(inside_mask) * pixel_scale
    # signed distance: positive outside, negative inside
    signed = dist_out.astype(np.float32) - dist_in.astype(np.float32)
    return signed


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    args = parse_args()
    out = args.out
    res = args.res

    print(
        f"Building mask for res={res} in box x=[{args.xmin},{args.xmax}] y=[{args.ymin},{args.ymax}] max_iter={args.max_iter}"
    )
    
    if args.use_gpu:
        print("Using GPU acceleration (OpenGL)")
    
    start_time = time.time()
    
    if args.use_gpu:
        inside = build_mask_gpu(
            res, args.xmin, args.xmax, args.ymin, args.ymax, args.max_iter, args.bailout
        )
    else:
        inside = build_mask(
            res, args.xmin, args.xmax, args.ymin, args.ymax, args.max_iter, args.bailout
        )
    
    mask_time = time.time() - start_time
    print(f"Mask generation completed in {mask_time:.2f}s")

    # pixel scale: approximate Euclidean distance per pixel; assume square pixels
    pixel_scale = math.hypot(args.xmax - args.xmin, args.ymax - args.ymin) / float(res)

    print("Computing signed distance transform...")
    distance_start = time.time()
    signed = build_signed_distance(inside, pixel_scale)
    distance_time = time.time() - distance_start
    print(f"Distance transform completed in {distance_time:.2f}s")

    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out.with_suffix(".npy"), signed)

    meta = {
        "xmin": args.xmin,
        "xmax": args.xmax,
        "ymin": args.ymin,
        "ymax": args.ymax,
        "res": res,
        "max_iter": args.max_iter,
        "used_gpu": args.use_gpu,
    }
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.2f}s")
    print(
        f"Saved signed distance field to {out.with_suffix('.npy')} and metadata {out.with_suffix('.json')}"
    )


if __name__ == "__main__":
    main()
