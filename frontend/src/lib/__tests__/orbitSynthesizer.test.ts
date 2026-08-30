/**
 * Tests for the wasm-backed orbit synthesizer adapter.
 *
 * Uses the deterministic mock wasm module (same math as runtime-core's
 * `PlayerState` c-space integrator for lobe=1) so these run without the
 * compiled binary.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  initOrbitSynth,
  OrbitSynthesizer,
  createInitialState,
} from '../orbitSynthesizer';

beforeAll(async () => {
  (globalThis as any).__vitest = true;
  await initOrbitSynth();
});

describe('OrbitSynthesizer (wasm adapter)', () => {
  it('requires initialization', async () => {
    // A fresh module instance would throw; here we just assert construction
    // works after init in beforeAll.
    expect(() => new OrbitSynthesizer(6)).not.toThrow();
  });

  it('positions c from (s, alpha) on the first step', () => {
    const synth = new OrbitSynthesizer(6, { s: 1.0, alpha: 0.25 });
    const c = synth.step(1 / 60, [0, 0, 0, 0, 0, 0]);
    // May's mandelbrotBoundary(1.0, 0.25): theta=pi/2, r=0.25, scale=1.
    const theta = 0.25 * 2 * Math.PI;
    const r = 0.25 * (1 - Math.cos(theta));
    const expectedRe = r * Math.cos(theta / 2);
    const expectedIm = r * Math.sin(theta / 2);
    expect(c.real).toBeCloseTo(expectedRe, 10);
    expect(c.imag).toBeCloseTo(expectedIm, 10);
  });

  it('moves c toward the model-driven target (no closed loop)', () => {
    const synth = new OrbitSynthesizer(6, { s: 1.0, alpha: 0.0 });
    const startRe = synth.cRe;
    const startIm = synth.cIm;
    // Move the target to alpha=0.5 (opposite side of the cardioid).
    synth.applyControls({ sTarget: 1.0, alpha: 0.5, omegaScale: 1.0, bandGates: [] });
    const c = synth.step(1 / 60, [1, 1, 1, 1, 1, 1]);
    const moved = Math.hypot(c.real - startRe, c.imag - startIm);
    expect(moved).toBeGreaterThan(1e-6);
  });

  it('wobbles around the model-driven position but never leaves it far', () => {
    // May semantics: with FIXED controls, c stays near the boundary point
    // (residuals are only ±0.05·k amplitude), unlike the old closed loop
    // which traced the whole cardioid regardless of audio.
    const synth = new OrbitSynthesizer(6, { s: 1.0, alpha: 0.5 });
    let maxDist = 0;
    for (let i = 0; i < 600; i++) {
      const c = synth.step(1 / 60, [1, 1, 1, 1, 1, 1]);
      const theta = 0.5 * 2 * Math.PI;
      const r = 0.25 * (1 - Math.cos(theta));
      const baseRe = r * Math.cos(theta / 2);
      const baseIm = r * Math.sin(theta / 2);
      maxDist = Math.max(maxDist, Math.hypot(c.real - baseRe, c.imag - baseIm));
    }
    // Residuals bounded by sum of 0.05*gate ≈ 0.3 worst case; verify c hugs base.
    expect(maxDist).toBeLessThan(0.35);
  });

  it('applyControls updates s, alpha and omega from model output', () => {
    const base = new OrbitSynthesizer(6, { s: 1.0, alpha: 0.3 });
    const before = base.step(1 / 60, [1, 1, 1, 1, 1, 1]);

    const scaled = new OrbitSynthesizer(6, { s: 1.0, alpha: 0.3 });
    scaled.applyControls({ sTarget: 1.5, alpha: 0.9, omegaScale: 2.0, bandGates: [] });
    const after = scaled.step(1 / 60, [1, 1, 1, 1, 1, 1]);

    const dist = Math.hypot(after.real - before.real, after.imag - before.imag);
    expect(dist).toBeGreaterThan(1e-6);
  });

  it('keeps wandering when the model wiggles controls (momentum, no parking)', () => {
    // Regression guard for the frozen-c bug: with saturated, slowly-varying
    // controls (as the epoch-10 model produces), the momentum integrator
    // must keep c moving. Constant controls legitimately settle at an
    // equilibrium offset; real model output never holds still.
    const synth = new OrbitSynthesizer(6, { s: 2.7, alpha: 0.95 });
    let path = 0;
    let prev = { real: synth.cRe, imag: synth.cIm };
    for (let i = 0; i < 600; i++) {
      const sT = 2.7 + 0.03 * Math.sin(i * 0.05);
      const aT = Math.min(1, Math.max(0, 0.95 + 0.002 * Math.cos(i * 0.03)));
      synth.applyControls({ sTarget: sT, alpha: aT, omegaScale: 4.0, bandGates: [] });
      const c = synth.step(1 / 60, [0.95, 0.95, 0.95, 0.95, 0.95, 0.95]);
      path += Math.hypot(c.real - prev.real, c.imag - prev.imag);
      prev = c;
    }
    expect(path).toBeGreaterThan(0.05);
  });

  it('setLobe is accepted (no-op for cardioid-only May controller)', () => {
    const synth = new OrbitSynthesizer(6);
    expect(() => synth.setLobe(2)).not.toThrow();
  });

  it('clamps band gates to [0, 1]', () => {
    const a = new OrbitSynthesizer(6, { s: 1.0, alpha: 0.5 });
    const ca = a.step(0.016, [2.0, -1.0, 1, 1, 1, 1]);
    const b = new OrbitSynthesizer(6, { s: 1.0, alpha: 0.5 });
    const cb = b.step(0.016, [1.0, 0.0, 1, 1, 1, 1]);
    expect(ca.real).toBeCloseTo(cb.real, 12);
    expect(ca.imag).toBeCloseTo(cb.imag, 12);
  });

  it('createInitialState returns the documented defaults', () => {
    const state = createInitialState({ kResiduals: 6 });
    expect(state).toEqual({
      lobe: 1,
      subLobe: 0,
      s: 0.5,
      alpha: 0.5,
      omega: 1.0,
      theta: 0.0,
    });
  });
});
