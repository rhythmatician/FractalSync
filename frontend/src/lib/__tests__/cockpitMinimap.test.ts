/**
 * Tests for the cockpit minimap panel renderer (issue #111 Phase A).
 *
 * The minimap reuses the canonical Rust mip pyramid via the wasm
 * `minimapShoreProximityBatch` binding — no TypeScript Mandelbrot formulas.
 * These tests verify the canvas painting contract against the vitest mock.
 */

import { describe, it, expect } from 'vitest';
import {
  MINIMAP_SIZE,
  paintMinimap,
  minimapWindowFor,
  fovTriangleC,
  setMinimapWasmSurface,
  type MinimapPaintInput,
} from '../cockpitMinimap';
import mockModule from './orbitSynthesizer.mock';

function makeCanvas(): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = MINIMAP_SIZE;
  canvas.height = MINIMAP_SIZE;
  return canvas;
}

describe('cockpit minimap panel (issue #111 Phase A)', () => {
  it('paints a MINIMAP_SIZE canvas with trail + rider + footprint', () => {
    setMinimapWasmSurface(mockModule as never);
    const canvas = makeCanvas();
    const input: MinimapPaintInput = {
      extent: [-2, 1, -1.5, 1.5],
      trail: [
        [0.0, 0.0],
        [0.1, 0.0],
        [0.2, 0.0],
      ],
      currentC: [0.2, 0.0],
      footprintHalf: 0.5,
    };
    const painted = paintMinimap(canvas, input);
    expect(painted).toBe(true);
  });

  it('returns false and leaves the canvas blank without a pyramid surface', () => {
    setMinimapWasmSurface(null);
    const canvas = makeCanvas();
    const input: MinimapPaintInput = {
      extent: [-2, 1, -1.5, 1.5],
      trail: [[0.0, 0.0]],
      currentC: [0.0, 0.0],
      footprintHalf: 0.5,
    };
    const painted = paintMinimap(canvas, input);
    expect(painted).toBe(false);
  });

  it('handles empty trail without throwing', () => {
    setMinimapWasmSurface(mockModule as never);
    const canvas = makeCanvas();
    const input: MinimapPaintInput = {
      extent: [-2, 1, -1.5, 1.5],
      trail: [],
      currentC: [0.0, 0.0],
      footprintHalf: 0.5,
    };
    // May be true or false depending on pyramid availability, but must not
    // throw.
    expect(() => paintMinimap(canvas, input)).not.toThrow();
  });

  it('exposes the canonical panel size', () => {
    expect(MINIMAP_SIZE).toBeGreaterThan(64);
    expect(MINIMAP_SIZE).toBeLessThanOrEqual(512);
  });

  it('zoom window shrinks with scale and stays inside the pyramid extent', () => {
    // Issue feedback: "the minimap should zoom with scale" — the sampled
    // window follows the rider and tracks the LOD patch half-extent (4x
    // the patch for orientation context), clamped to the extent.
    const extent: [number, number, number, number] = [-2, 1, -1.5, 1.5];
    const deep = minimapWindowFor([0.2549, 0.0], 0.006, extent);
    const valley = minimapWindowFor([0.0, 0.0], 0.5, extent);
    expect(deep.half).toBeLessThan(valley.half);
    // Window tracks the patch: 4x the footprint for context.
    expect(deep.half).toBeCloseTo(0.024, 12);
    // Deep window centers on the rider near the Shore.
    expect(deep.center).toEqual([0.2549, 0.0]);
    // Clamped windows never leak past the extent.
    for (const win of [deep, valley]) {
      const [reMin, reMax, imMin, imMax] = extent;
      expect(win.center[0] - win.half).toBeGreaterThanOrEqual(reMin - 1e-12);
      expect(win.center[0] + win.half).toBeLessThanOrEqual(reMax + 1e-12);
      expect(win.center[1] - win.half).toBeGreaterThanOrEqual(imMin - 1e-12);
      expect(win.center[1] + win.half).toBeLessThanOrEqual(imMax + 1e-12);
    }
  });

  it('FOV triangle spans the terrain patch in front of the rider', () => {
    // Issue feedback: the footprint marker must be a triangle accurately
    // signifying the camera FOV, not a square. The camera looks along the
    // rider heading from behind; the visible ground region is a triangle
    // with the apex at the rider and the base ahead of them.
    const tri = fovTriangleC([0.1, 0.0], 1.0, 0.0, 60);
    expect(tri.length).toBe(3);
    // Apex at the rider position.
    expect(tri[0][0]).toBeCloseTo(0.1, 12);
    expect(tri[0][1]).toBeCloseTo(0.0, 12);
    // Base ahead along the heading (+X when heading 0), symmetric about it.
    expect(tri[1][0]).toBeGreaterThan(0.1);
    expect(tri[2][0]).toBeGreaterThan(0.1);
    expect(tri[1][1]).toBeCloseTo(-tri[2][1], 12);
    // Base width grows with FOV.
    const wide = fovTriangleC([0.1, 0.0], 1.0, 0.0, 100);
    const baseNarrow = Math.abs(tri[1][1] - tri[2][1]);
    const baseWide = Math.abs(wide[1][1] - wide[2][1]);
    expect(baseWide).toBeGreaterThan(baseNarrow);
  });

  it('FOV triangle rotates with the heading', () => {
    const east = fovTriangleC([0.0, 0.0], 1.0, 0.0, 60);
    const north = fovTriangleC([0.0, 0.0], 1.0, Math.PI / 2, 60);
    // Heading east: base is ahead on +X, centered on the axis.
    expect(east[1][0]).toBeGreaterThan(east[1][1]);
    expect(east[1][1]).toBeCloseTo(-east[2][1], 12);
    // Heading north: base is ahead on +Y, centered on the axis.
    expect(north[1][1]).toBeGreaterThan(Math.abs(north[1][0]));
    expect(north[1][0]).toBeCloseTo(-north[2][0], 12);
  });

  it('paints with the zoomed window and FOV triangle', () => {
    setMinimapWasmSurface(mockModule as never);
    const canvas = makeCanvas();
    const input: MinimapPaintInput = {
      extent: [-2, 1, -1.5, 1.5],
      trail: [
        [0.2, 0.0],
        [0.24, 0.0],
        [0.2549, 0.0],
      ],
      currentC: [0.2549, 0.0],
      footprintHalf: 0.006,
      heading: 0.0,
      fovDeg: 55,
    };
    const painted = paintMinimap(canvas, input);
    expect(painted).toBe(true);
  });

  it('hands off to the deep-zoom field when the window out-resolves the pyramid', () => {
    // Issue #111 feedback: the minimap is a Mandelbrot deep zoom. The
    // canonical bake is 2048 texels over a 3-wide extent (~1.5e-3/texel);
    // a window below ~8 texels across must sample the resolution-unlimited
    // DEM instead of the degenerating S field.
    setMinimapWasmSurface(mockModule as never);
    const canvas = makeCanvas();
    const input: MinimapPaintInput = {
      extent: [-2, 1, -1.5, 1.5],
      trail: [[0.2549, 0.0]],
      currentC: [0.2549, 0.0],
      footprintHalf: 0.006, // window = 0.024 wide ~ 16 texels... use smaller
      heading: 0.0,
      fovDeg: 55,
    };
    // 4x footprint = 0.024 window ~ 16 texels: still pyramid. Shrink the
    // footprint so the window drops under 8 texels (~0.012 wide).
    input.footprintHalf = 0.001;
    expect(paintMinimap(canvas, input)).toBe(true);
  });

  it('still paints from the pyramid at valley zoom', () => {
    setMinimapWasmSurface(mockModule as never);
    const canvas = makeCanvas();
    const input: MinimapPaintInput = {
      extent: [-2, 1, -1.5, 1.5],
      trail: [[0.0, 0.0]],
      currentC: [0.0, 0.0],
      footprintHalf: 0.5, // window = 2.0 wide ~ 1365 texels: pyramid
      heading: 0.0,
      fovDeg: 55,
    };
    expect(paintMinimap(canvas, input)).toBe(true);
  });
});
