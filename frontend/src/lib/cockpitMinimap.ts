/**
 * Cockpit minimap panel renderer (issue #111 Phase A).
 *
 * Renders a Mandelbrot deep zoom centered on the player, with the zoom
 * level matching the in-game horizontal zoom. Self-contained TypeScript
 * escape-time renderer — no wasm, no Python backend dependency.
 *
 * The panel shows: the Mandelbrot set (Shore = boundary), the recent
 * trajectory trail, the current c, and the 3D viewport footprint.
 */

/** Panel canvas size in pixels (square). */
export const MINIMAP_SIZE = 192;

/** Input for one minimap paint. */
export interface MinimapPaintInput {
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

/**
 * Paint one minimap frame. Renders a Mandelbrot deep zoom centered on the
 * player, with the zoom level matching the in-game horizontal zoom ratio
 * (the terrain patch half-extent). Self-contained TypeScript escape-time
 * renderer — no wasm, no Python backend dependency.
 */
export function paintMinimap(
  canvas: HTMLCanvasElement,
  input: MinimapPaintInput
): boolean {
  const size = canvas.width;
  const ctx = canvas.getContext('2d');
  if (!ctx) return false;

  // Zoom window: centered on the player, spanning 4x the terrain patch
  // half-extent so the minimap zooms with the in-game horizontal zoom.
  const [cx, cy] = input.currentC;
  const half = Math.max(input.footprintHalf, 1e-6) * 4;
  const reMin = cx - half;
  const reMax = cx + half;
  const imMin = cy - half;
  const imMax = cy + half;

  // Render the Mandelbrot set via escape-time iteration.
  const image = ctx.createImageData(size, size);
  const maxIter = 128;
  const bailout = 4.0;
  for (let row = 0; row < size; row++) {
    const y = imMax - ((imMax - imMin) * row) / (size - 1);
    for (let col = 0; col < size; col++) {
      const x = reMin + ((reMax - reMin) * col) / (size - 1);
      let zx = 0;
      let zy = 0;
      let zx2 = 0;
      let zy2 = 0;
      let iter = 0;
      while (zx2 + zy2 < bailout && iter < maxIter) {
        zy = 2 * zx * zy + y;
        zx = zx2 - zy2 + x;
        zx2 = zx * zx;
        zy2 = zy * zy;
        iter++;
      }
      const i = image.data;
      const idx = (row * size + col) * 4;
      if (iter >= maxIter) {
        // Inside the set: dark blue.
        i[idx] = 10;
        i[idx + 1] = 20;
        i[idx + 2] = 60;
      } else {
        // Outside: proximity ramp — brighter near the Shore boundary.
        const proximity = 1.0 - iter / maxIter;
        const v = proximity * proximity;
        i[idx] = Math.round(30 + 200 * v);
        i[idx + 1] = Math.round(40 + 190 * v);
        i[idx + 2] = Math.round(90 + 60 * (1 - v));
      }
      i[idx + 3] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);

  // World -> pixel transform (row 0 = im_max).
  const toPx = (x: number, y: number): [number, number] => [
    ((x - reMin) / (reMax - reMin)) * (size - 1),
    ((imMax - y) / (imMax - imMin)) * (size - 1),
  ];

  // Viewport footprint: the camera FOV ground triangle.
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
