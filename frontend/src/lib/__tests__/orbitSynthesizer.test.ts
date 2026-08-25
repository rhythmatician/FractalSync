/**
 * Tests for the wasm-backed orbit synthesizer adapter.
 *
 * Uses the deterministic mock wasm module (same math as runtime-core's
 * controller for lobe=1) so these run without the compiled binary.
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

  it('advances theta by omega * dt each step', () => {
    const synth = new OrbitSynthesizer(6, { theta: 0.0, omega: 1.0 });
    synth.step(1 / 60, [1, 1, 1, 1, 1, 1]);
    expect(synth.theta).toBeCloseTo(1 / 60, 10);
    synth.step(1 / 60, [1, 1, 1, 1, 1, 1]);
    expect(synth.theta).toBeCloseTo(2 / 60, 10);
  });

  it('wraps theta into [0, 2π)', () => {
    const synth = new OrbitSynthesizer(6, { theta: 6.28, omega: 1.0 });
    synth.step(0.1, [1, 1, 1, 1, 1, 1]);
    expect(synth.theta).toBeGreaterThanOrEqual(0);
    expect(synth.theta).toBeLessThan(2 * Math.PI);
  });

  it('returns carrier when alpha is zero', () => {
    const synth = new OrbitSynthesizer(6, { s: 1.0, alpha: 0.0, theta: 0.5 });
    const c = synth.step(0.016, [1, 1, 1, 1, 1, 1]);
    // Carrier at theta=0.5+dt, s=1: c = mu/2 - mu^2/4
    const theta = 0.5 + 0.016;
    const muRe = Math.cos(theta);
    const muIm = Math.sin(theta);
    const expectedRe = 0.5 * muRe - 0.25 * (muRe * muRe - muIm * muIm);
    const expectedIm = 0.5 * muIm - 0.25 * (2 * muRe * muIm);
    expect(c.real).toBeCloseTo(expectedRe, 10);
    expect(c.imag).toBeCloseTo(expectedIm, 10);
  });

  it('adds gated residuals when alpha > 0', () => {
    const closed = new OrbitSynthesizer(6, { s: 1.0, alpha: 0.8 });
    const cClosed = closed.step(0.016, [0, 0, 0, 0, 0, 0]);

    const open = new OrbitSynthesizer(6, { s: 1.0, alpha: 0.8 });
    const cOpen = open.step(0.016, [1, 1, 1, 1, 1, 1]);

    const dist = Math.hypot(cOpen.real - cClosed.real, cOpen.imag - cClosed.imag);
    expect(dist).toBeGreaterThan(1e-6);
  });

  it('applyControls updates s, alpha and omega from model output', () => {
    const base = new OrbitSynthesizer(6, { s: 1.0, alpha: 0.3, theta: 0.25 });
    const before = base.step(0.0, [1, 1, 1, 1, 1, 1]);

    const scaled = new OrbitSynthesizer(6, { s: 1.0, alpha: 0.3, theta: 0.25 });
    scaled.applyControls({ sTarget: 1.5, alpha: 0.9, omegaScale: 2.0, bandGates: [] });
    const after = scaled.step(0.0, [1, 1, 1, 1, 1, 1]);

    const dist = Math.hypot(after.real - before.real, after.imag - before.imag);
    expect(dist).toBeGreaterThan(1e-6);
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
