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

  it('starts on the boundary at (s, alpha)', () => {
    const synth = new OrbitSynthesizer(6, { s: 1.0, alpha: 0.25 });
    // c = mu/2 - mu^2/4 with mu = s * e^{i * alpha * 2π}
    const theta = 0.25 * 2 * Math.PI;
    const muRe = Math.cos(theta);
    const muIm = Math.sin(theta);
    const expectedRe = 0.5 * muRe - 0.25 * (muRe * muRe - muIm * muIm);
    const expectedIm = 0.5 * muIm - 0.25 * (2 * muRe * muIm);
    expect(synth.cRe).toBeCloseTo(expectedRe, 10);
    expect(synth.cIm).toBeCloseTo(expectedIm, 10);
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

  it('settles at a fixed target instead of tracing a loop', () => {
    const synth = new OrbitSynthesizer(6, { s: 1.0, alpha: 0.5 });
    // Hold the target fixed; c should converge and stay (no perpetual orbit).
    for (let i = 0; i < 600; i++) {
      synth.step(1 / 60, [1, 1, 1, 1, 1, 1]);
    }
    const beforeRe = synth.cRe;
    const beforeIm = synth.cIm;
    for (let i = 0; i < 60; i++) {
      synth.step(1 / 60, [1, 1, 1, 1, 1, 1]);
    }
    const drift = Math.hypot(synth.cRe - beforeRe, synth.cIm - beforeIm);
    expect(drift).toBeLessThan(1e-3);
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

  it('setLobe switches the active bulb', () => {
    const synth = new OrbitSynthesizer(6);
    expect(synth.lobe).toBe(1);
    synth.setLobe(2);
    expect(synth.lobe).toBe(2);
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
