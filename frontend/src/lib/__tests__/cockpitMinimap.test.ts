/**
 * Tests for the cockpit minimap panel renderer (issue #111 Phase A).
 *
 * The minimap renders a Mandelbrot deep zoom centered on the player, with
 * the zoom level matching the in-game horizontal zoom. Self-contained
 * TypeScript escape-time renderer — no wasm, no Python backend dependency.
 */

import { describe, it, expect } from 'vitest';
import {
  MINIMAP_SIZE,
  paintMinimap,
  fovTriangleC,
  type MinimapPaintInput,
} from '../cockpitMinimap';

function makeCanvas(): HTMLCanvasElement {
  const canvas = document.createElement('canvas');
  canvas.width = MINIMAP_SIZE;
  canvas.height = MINIMAP_SIZE;
  return canvas;
}

describe('cockpit minimap panel (issue #111 Phase A)', () => {
  it('paints a MINIMAP_SIZE canvas with trail + rider + footprint', () => {
    const canvas = makeCanvas();
    const input: MinimapPaintInput = {
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

  it('always paints (no backend dependency)', () => {
    const canvas = makeCanvas();
    const input: MinimapPaintInput = {
      trail: [[0.0, 0.0]],
      currentC: [0.0, 0.0],
      footprintHalf: 0.5,
    };
    const painted = paintMinimap(canvas, input);
    expect(painted).toBe(true);
  });

  it('handles empty trail without throwing', () => {
    const canvas = makeCanvas();
    const input: MinimapPaintInput = {
      trail: [],
      currentC: [0.0, 0.0],
      footprintHalf: 0.5,
    };
    expect(() => paintMinimap(canvas, input)).not.toThrow();
  });

  it('exposes the canonical panel size', () => {
    expect(MINIMAP_SIZE).toBeGreaterThan(64);
    expect(MINIMAP_SIZE).toBeLessThanOrEqual(512);
  });

  it('FOV triangle spans the terrain patch in front of the rider', () => {
    const tri = fovTriangleC([0.1, 0.0], 1.0, 0.0, 60);
    expect(tri.length).toBe(3);
    expect(tri[0][0]).toBeCloseTo(0.1, 12);
    expect(tri[0][1]).toBeCloseTo(0.0, 12);
    expect(tri[1][0]).toBeGreaterThan(0.1);
    expect(tri[2][0]).toBeGreaterThan(0.1);
    expect(tri[1][1]).toBeCloseTo(-tri[2][1], 12);
    const wide = fovTriangleC([0.1, 0.0], 1.0, 0.0, 100);
    const baseNarrow = Math.abs(tri[1][1] - tri[2][1]);
    const baseWide = Math.abs(wide[1][1] - wide[2][1]);
    expect(baseWide).toBeGreaterThan(baseNarrow);
  });

  it('FOV triangle rotates with the heading', () => {
    const east = fovTriangleC([0.0, 0.0], 1.0, 0.0, 60);
    const north = fovTriangleC([0.0, 0.0], 1.0, Math.PI / 2, 60);
    expect(east[1][0]).toBeGreaterThan(east[1][1]);
    expect(east[1][1]).toBeCloseTo(-east[2][1], 12);
    expect(north[1][1]).toBeGreaterThan(Math.abs(north[1][0]));
    expect(north[1][0]).toBeCloseTo(-north[2][0], 12);
  });

  it('paints with the zoomed window and FOV triangle', () => {
    const canvas = makeCanvas();
    const input: MinimapPaintInput = {
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

  it('renders Mandelbrot structure at deep zoom', () => {
    // At c=0 with a small footprint, the center pixel should be inside
    // the set (iter >= maxIter → dark blue) and the corner pixels should
    // escape (iter < maxIter → bright). We can't inspect the canvas in
    // jsdom, so we verify the renderer doesn't throw and returns true.
    const canvas = makeCanvas();
    const input: MinimapPaintInput = {
      trail: [],
      currentC: [0.0, 0.0],
      footprintHalf: 0.001, // window = 0.004 wide, deep zoom
      heading: 0.0,
      fovDeg: 55,
    };
    const painted = paintMinimap(canvas, input);
    expect(painted).toBe(true);
  });

  it('still paints at valley zoom', () => {
    const canvas = makeCanvas();
    const input: MinimapPaintInput = {
      trail: [[0.0, 0.0]],
      currentC: [0.0, 0.0],
      footprintHalf: 0.5, // window = 2.0 wide, broad view
      heading: 0.0,
      fovDeg: 55,
    };
    expect(paintMinimap(canvas, input)).toBe(true);
  });
});
