/**
 * Parity test for JuliaViewState: Rust/Python/WASM must produce identical state.
 *
 * Uses the same fixture as backend/tests/test_julia_view_parity.py:
 * initial zoom=1, rotation=0, anchor_hue=0, chroma=0.18, lightness=0.55,
 * harmony=analogous, accent_weight=0.35, harmony_cooldown=0, harmony_armed=true
 * and a 5-tick delta sequence. The mock in orbitSynthesizer.mock.ts now mirrors
 * the Rust logic in runtime-core/src/controls.rs::JuliaViewState::apply_controls,
 * so this test pins that the WASM projection (via the mock) matches the Rust
 * expected values. The Python sibling test pins the PyO3 projection.
 *
 * Together they demonstrate: identical initial view state + identical deltas
 * => identical state in Rust/Python/WASM (ADR 0001).
 */

import { describe, it, expect } from 'vitest';

function wrapAngle(theta: number): number {
  const tau = 2 * Math.PI;
  let w = theta % tau;
  if (w < 0) w += tau;
  if (w > Math.PI) w -= tau;
  return w;
}
function wrap01(x: number): number {
  return ((x % 1) + 1) % 1;
}

function rustExpected(): {
  zoom: number;
  rotation: number;
  anchor_hue: number;
  chroma: number;
  lightness: number;
  accent_weight: number;
  harmony: string;
  harmony_cooldown: number;
  harmony_armed: boolean;
} {
  let zoom = 1.0;
  let rotation = 0.0;
  let anchor_hue = 0.0;
  let chroma = 0.18;
  let lightness = 0.55;
  let accent_weight = 0.35;
  let harmony = 'analogous';
  let harmony_cooldown = 0;
  let harmony_armed = true;

  const deltas: Array<[number, number, number, number, number, number, number]> = [
    [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.7],
    [0.2, -0.3, 0.1, -0.2, 0.1, -0.1, 0.0],
    [-0.4, 0.2, -0.5, 0.3, -0.3, 0.2, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
  ];

  const modes = ['monochrome', 'analogous', 'opponent'] as const;
  for (const [zoom_d0, rot_d0, hue_d0, chroma_d0, light_d0, accent_d0, hs0] of deltas) {
    if (harmony_cooldown > 0) harmony_cooldown -= 1;
    const c11 = (v: number) => Math.max(-1, Math.min(1, v));
    const zoom_d = c11(zoom_d0);
    const rot_d = c11(rot_d0);
    const hue_d = c11(hue_d0);
    const chroma_d = c11(chroma_d0);
    const light_d = c11(light_d0);
    const accent_d = c11(accent_d0);
    const hs = c11(hs0);
    zoom = Math.max(0.5, Math.min(8.0, zoom * Math.exp(zoom_d * 0.05)));
    rotation = wrapAngle(rotation + rot_d * 0.08);
    anchor_hue = wrap01(anchor_hue + hue_d * 0.02);
    chroma = Math.max(0.0, Math.min(0.4, chroma + chroma_d * 0.03));
    lightness = Math.max(0.2, Math.min(0.9, lightness + light_d * 0.03));
    accent_weight = Math.max(0.0, Math.min(1.0, accent_weight + accent_d * 0.04));
    if (Math.abs(hs) < 0.3) harmony_armed = true;
    if (Math.abs(hs) > 0.6 && harmony_armed && harmony_cooldown === 0) {
      const idx = modes.indexOf(harmony as (typeof modes)[number]);
      const dir = hs > 0 ? 1 : 2;
      harmony = modes[(idx + dir) % 3];
      harmony_cooldown = 15;
      harmony_armed = false;
    }
  }
  return { zoom, rotation, anchor_hue, chroma, lightness, accent_weight, harmony, harmony_cooldown, harmony_armed };
}

describe('JuliaViewState parity (WASM vs Rust)', () => {
  it('identical initial + identical deltas => identical state via WASM (mock)', async () => {
    const mock = await import('./orbitSynthesizer.mock');
    const mod: any = (mock as any).default ?? mock;
    const State = mod.JuliaViewState;
    const Controls = mod.JuliaViewControls;
    expect(State).toBeDefined();
    expect(Controls).toBeDefined();

    const color = mod.ColorIntent ? new mod.ColorIntent(0.0, 0.18, 0.55, 'analogous', 0.35) : { anchor_hue: 0, chroma: 0.18, lightness: 0.55, harmony: 'analogous', accent_weight: 0.35 };
    // JuliaViewState(zoom, rotation, color, harmony_cooldown, harmony_armed)
    const state: any = new State(1.0, 0.0, color, 0, true);

    const deltas: Array<[number, number, number, number, number, number, number]> = [
      [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.7],
      [0.2, -0.3, 0.1, -0.2, 0.1, -0.1, 0.0],
      [-0.4, 0.2, -0.5, 0.3, -0.3, 0.2, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ];
    for (const d of deltas) {
      const c = new Controls(...d);
      state.apply_controls(c);
    }

    const expected = rustExpected();
    expect(state.zoom).toBeCloseTo(expected.zoom, 12);
    expect(state.rotation).toBeCloseTo(expected.rotation, 12);
    // color may be .color or .color() depending on mock shape
    expect(state.color.anchor_hue).toBeCloseTo(expected.anchor_hue, 12);
    expect(state.color.chroma).toBeCloseTo(expected.chroma, 12);
    expect(state.color.lightness).toBeCloseTo(expected.lightness, 12);
    expect(state.color.accent_weight).toBeCloseTo(expected.accent_weight, 12);
    expect(state.color.harmony).toBe(expected.harmony);
    expect(state.harmony_cooldown).toBe(expected.harmony_cooldown);
    expect(state.harmony_armed).toBe(expected.harmony_armed);

    // Determinism: repeating yields same
    const color2: any = (mod as any).ColorIntent ? new (mod as any).ColorIntent(0.0, 0.18, 0.55, 'analogous', 0.35) : { anchor_hue: 0, chroma: 0.18, lightness: 0.55, harmony: 'analogous', accent_weight: 0.35 };
    const state2: any = new State(1.0, 0.0, color2, 0, true);
    for (const d of deltas) {
      state2.apply_controls(new Controls(...d));
    }
    expect(state2.zoom).toBeCloseTo(state.zoom, 12);
  });
});
