/**
 * Cockpit minimap panel renderer (issue #111 Phase A).
 *
 * Reuses the CANONICAL Rust mip pyramid via the wasm
 * `minimapShoreProximityBatch` binding — there is deliberately no TypeScript
 * Mandelbrot formula here (ADR 0001; issue #111: "Reuse the canonical Rust
 * Map/minimap machinery from #88/#106 rather than introducing another
 * frontend geometry authority").
 *
 * The panel shows: the Shore/realm field (S), the recent trajectory trail,
 * the current c, and the 3D viewport footprint (the terrain patch extent).
 * All geometry comes from the pyramid extent + authoritative snapshot data.
 */

/** Panel canvas size in pixels (square). */
export const MINIMAP_SIZE = 192;

/** Wire shape of the pyramid extent [re_min, re_max, im_min, im_max]. */
export type MinimapExtent = [number, number, number, number];

/** Input for one minimap paint. */
export interface MinimapPaintInput {
  /** Canonical pyramid extent. */
  extent: MinimapExtent;
  /** Recent trajectory points in c-space (oldest first). */
  trail: Array<[number, number]>;
  /** Current rider position in c-space. */
  currentC: [number, number];
  /** Half-extent of the 3D terrain patch in c-space (viewport footprint). */
  footprintHalf: number;
  /** Rider heading in radians (0 = +X/east). Default 0. */
  heading?: number;
  /** Camera vertical FOV in degrees. Default 55 (the cockpit camera). */
  fovDeg?: number;
}

/** A zoomed minimap sampling window. */
export interface MinimapWindow {
  center: [number, number];
  half: number;
}

/**
 * Compute the zoomed sampling window for the minimap (issue #111 feedback:
 * "the minimap should zoom with scale"). The window follows the rider and
 * tracks the LOD patch half-extent — at deep scale the map magnifies the
 * crest neighborhood, in the valley it shows the broad view. Clamped so
 * the window never leaves the canonical pyramid extent.
 */
export function minimapWindowFor(
  c: [number, number],
  footprintHalf: number,
  extent: MinimapExtent
): MinimapWindow {
  const [reMin, reMax, imMin, imMax] = extent;
  const rawHalf = Math.max(footprintHalf, 1e-6);
  // Window spans 4x the terrain patch for orientation context, clamped to
  // a quarter of the extent so the zoomed map never exceeds the bake.
  const half = Math.min(rawHalf * 4, Math.min(reMax - reMin, imMax - imMin) / 4);
  const cx = Math.min(Math.max(c[0], reMin + half), reMax - half);
  const cy = Math.min(Math.max(c[1], imMin + half), imMax - half);
  return { center: [cx, cy], half };
}

/**
 * The ground footprint of the third-person camera FOV as a triangle in
 * c-space (issue #111 feedback: "a triangle accurately signifying the FOV
 * rather than a square").
 *
 * Apex at the rider; the base spans the visible ground ahead. Geometry:
 * the camera sits `camBack` scene units behind the rider looking down at
 * ~34 deg (back 3.2, up 2.2), vertical FOV `fovDeg`. Projecting the frustum
 * onto the ground plane gives a near-half-width at the rider's feet and a
 * far reach of roughly one patch diagonal; the triangle is the honest
 * simple approximation of that region in c-space units.
 */
export function fovTriangleC(
  c: [number, number],
  footprintHalf: number,
  heading: number,
  fovDeg: number
): Array<[number, number]> {
  // The triangle spans about one patch diagonal ahead of the rider.
  const reach = Math.max(footprintHalf * 2 * Math.SQRT2, 1e-6);
  // Half-width of the base at the far end, from the vertical FOV.
  const halfFov = ((fovDeg / 2) * Math.PI) / 180;
  const halfBase = reach * Math.tan(halfFov);
  // Heading unit vector (0 = +X/east, +PI/2 = +Y/north).
  const hx = Math.cos(heading);
  const hy = Math.sin(heading);
  // Perpendicular (left of heading).
  const px = -hy;
  const py = hx;
  const apexX = c[0];
  const apexY = c[1];
  const baseX = apexX + hx * reach;
  const baseY = apexY + hy * reach;
  return [
    [apexX, apexY],
    [baseX + px * halfBase, baseY + py * halfBase],
    [baseX - px * halfBase, baseY - py * halfBase],
  ];
}

/** Shape of the wasm module surface this module needs. */
interface WasmMinimapSurface {
  minimapShoreProximityBatch?: (
    re: number[],
    im: number[],
    level: number
  ) => Float32Array | number[];
  /** Resolution-unlimited deep-zoom distance field (issue #111 feedback). */
  deepZoomField?: (re: number[], im: number[]) => Float32Array | number[];
}

let wasmSurface: WasmMinimapSurface | null = null;

/** Test seam: inject the wasm module surface. */
export function setMinimapWasmSurface(surface: WasmMinimapSurface | null): void {
  wasmSurface = surface;
}

/**
 * Paint one minimap frame. Returns true when the canonical pyramid was
 * available (field painted); false when it was not (caller should show the
 * panel's "pyramid unavailable" state rather than a blank lie).
 */
export function paintMinimap(
  canvas: HTMLCanvasElement,
  input: MinimapPaintInput
): boolean {
  const surface = wasmSurface;
  if (!surface || typeof surface.minimapShoreProximityBatch !== 'function') {
    return false;
  }

  // Zoomed window: track the rider and the LOD patch scale (issue #111
  // feedback: "the minimap should zoom with scale"). The canonical extent
  // only bounds the window via minimapWindowFor.
  const window = minimapWindowFor(input.currentC, input.footprintHalf, input.extent);
  const [reMin, reMax, imMin, imMax] = [
    window.center[0] - window.half,
    window.center[0] + window.half,
    window.center[1] - window.half,
    window.center[1] + window.half,
  ];
  // The canonical bake is 2048x2048 over the extent — its texel size sets
  // the pyramid's resolution limit (see deep-zoom handoff below).
  const [extentReMin, extentReMax] = input.extent;
  const pyramidTexel = (extentReMax - extentReMin) / 2048;
  const size = canvas.width;
  const ctx = canvas.getContext('2d');
  if (!ctx) return false;

  // Sample the S field on a size x size grid over the extent via the
  // canonical batch binding (level 0 = finest).
  const re: number[] = [];
  const im: number[] = [];
  for (let row = 0; row < size; row++) {
    // Row 0 = north edge (im_max), matching the pyramid's row convention.
    const y = imMax - ((imMax - imMin) * row) / (size - 1);
    for (let col = 0; col < size; col++) {
      const x = reMin + ((reMax - reMin) * col) / (size - 1);
      re.push(x);
      im.push(y);
    }
  }
  let field: Float32Array | number[];
  // Deep-zoom handoff (issue #111 feedback: "think about our minimap like
  // a Mandelbrot deep zoom ... zoom level depends on the player location"):
  // the baked pyramid has finite texel resolution. When the window spans
  // fewer than ~8 pyramid texels across, the S field degenerates to a
  // smear — switch to the resolution-unlimited escape-iteration distance
  // estimator, which resolves structure at ANY zoom depth.
  const windowSpan = reMax - reMin;
  const deepZoom = windowSpan < pyramidTexel * 8;
  try {
    if (deepZoom && typeof surface.deepZoomField === 'function') {
      field = surface.deepZoomField(re, im);
    } else {
      field = surface.minimapShoreProximityBatch(re, im, 0);
    }
  } catch {
    return false;
  }
  if (!field || field.length !== re.length) return false;

  // Paint the field. Two sources, one visual language:
  // - S field (pyramid): [0,1] proximity, 1 = at Shore.
  // - Deep-zoom DEM: unsigned distance to the boundary (0 inside). Convert
  //   to a proximity-like ramp via exp(-d / zoomScale) so the Shore band
  //   stays bright and structure fades with distance, matching the S look.
  const image = ctx.createImageData(size, size);
  const zoomScale = Math.max(windowSpan * 0.08, 1e-9);
  for (let i = 0; i < field.length; i++) {
    const raw = field[i];
    const v = deepZoom
      ? Math.max(0, Math.min(1, Math.exp(-raw / zoomScale)))
      : Math.max(0, Math.min(1, raw));
    const r = Math.round(30 + 200 * v * v);
    const g = Math.round(40 + 190 * v);
    const b = Math.round(90 + 60 * (1 - v));
    image.data[i * 4] = r;
    image.data[i * 4 + 1] = g;
    image.data[i * 4 + 2] = b;
    image.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);

  // World -> pixel transform (row 0 = im_max).
  const toPx = (x: number, y: number): [number, number] => [
    ((x - reMin) / (reMax - reMin)) * (size - 1),
    ((imMax - y) / (imMax - imMin)) * (size - 1),
  ];

  // Viewport footprint: the camera FOV ground triangle (issue #111
  // feedback: a triangle accurately signifying the FOV, not a square).
  const [cx, cy] = input.currentC;
  const triangle = fovTriangleC(
    [cx, cy],
    input.footprintHalf,
    input.heading ?? 0,
    input.fovDeg ?? 55
  );
  ctx.strokeStyle = 'rgba(255,255,255,0.75)';
  ctx.fillStyle = 'rgba(255,255,255,0.12)';
  ctx.lineWidth = 1;
  ctx.beginPath();
  triangle.forEach(([wx, wy], i) => {
    const [px, py] = toPx(wx, wy);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  });
  ctx.closePath();
  ctx.fill();
  ctx.stroke();

  // Trail: recent trajectory in c-space.
  if (input.trail.length > 1) {
    ctx.strokeStyle = '#66ff99';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    input.trail.forEach(([x, y], i) => {
      const [px, py] = toPx(x, y);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
  }

  // Current c: rider dot.
  const [px, py] = toPx(cx, cy);
  ctx.fillStyle = '#ff5555';
  ctx.beginPath();
  ctx.arc(px, py, 3, 0, Math.PI * 2);
  ctx.fill();

  return true;
}
