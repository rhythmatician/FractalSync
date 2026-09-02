/**
 * Tests for the debug cockpit's typed adapter over the wasm DebugSnapshot
 * seam (issue #111 Phase A).
 *
 * These tests run against the vitest wasm mock, so they verify the SEAM
 * CONTRACT (snapshot shape, provenance echo, determinism, crossing
 * bookkeeping) — not manifold math, which is Rust-owned and verified by the
 * runtime-core and backend suites. Real-physics crossing behavior is
 * verified in the browser against the actual wasm build.
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  initOrbitSynth,
  setWasmModuleForTesting,
  getWasmModule,
} from '../orbitSynthesizer';
import mockModule from './orbitSynthesizer.mock';
import {
  CockpitRecorder,
  DEFAULT_TERRAIN_GRID,
  replayVariantAsTrajectory,
  sampleTerrainPatch,
  type DebugSnapshot,
} from '../debugCockpit';
import { baselineVariants, expandActions } from '../shoreCrossingVariants';

describe('debug cockpit adapter (issue #111 Phase A)', () => {
  beforeAll(async () => {
    setWasmModuleForTesting(mockModule as never);
    await initOrbitSynth();
  });

  it('exposes the Rust DebugSnapshot contract version', () => {
    const meta = (
      getWasmModule() as unknown as {
        debugSnapshotMeta?: () => { version: string; canonicalDt: number };
      }
    ).debugSnapshotMeta?.();
    expect(meta?.version).toBe('debug-snapshot/1');
    expect(meta?.canonicalDt).toBeCloseTo(1024 / 48000, 15);
  });

  it('records a trajectory of snapshots keyed to destination steps', () => {
    const recorder = new CockpitRecorder();
    const variant = baselineVariants()[0];
    const trajectory = recorder.recordVariant(variant);

    const actions = expandActions(variant);
    expect(trajectory.snapshots.length).toBe(actions.length);
    // Snapshot i corresponds to state AFTER step i (0-indexed action applied).
    expect(trajectory.snapshots[0].physics.c[0]).not.toBe(0);
    // Time is the canonical destination-step clock, not frames.
    expect(trajectory.snapshots[0].timeSeconds).toBeCloseTo(1024 / 48000, 12);
  });

  it('records raw-vs-effective provenance for every step', () => {
    const recorder = new CockpitRecorder();
    const trajectory = recorder.recordVariant(baselineVariants()[2]);
    const withAction = trajectory.snapshots.filter((s) => s.action);
    expect(withAction.length).toBe(trajectory.snapshots.length);

    // The drip cadence emits commit (0.8) and settle (0.4) frames; raw must
    // echo exactly what was emitted, effective must be the clamped value.
    const rawThrottles = new Set(withAction.map((s) => s.action!.raw.throttle));
    expect(rawThrottles.has(0.8)).toBe(true);
    expect(rawThrottles.has(0.4)).toBe(true);
    for (const s of withAction) {
      expect(s.action!.effective.throttle).toBeLessThanOrEqual(1.0);
      expect(s.action!.effective.brake).toBeLessThanOrEqual(1.0);
      expect(s.action!.effective.grip).toBeLessThanOrEqual(1.0);
      expect(s.action!.effective.throttle).toBe(s.action!.raw.throttle);
    }
  });

  it('captures friction evidence (friction power nonpositive)', () => {
    const recorder = new CockpitRecorder();
    const trajectory = recorder.recordVariant(baselineVariants()[3]);
    for (const s of trajectory.snapshots) {
      if (s.action) {
        expect(s.action.frictionPower).toBeLessThanOrEqual(1e-12);
      }
    }
  });

  it('crossing bookkeeping derives from authoritative D(c) only', () => {
    const recorder = new CockpitRecorder();
    const trajectory = recorder.recordVariant(baselineVariants()[3]);
    if (trajectory.crossed) {
      // crossingStep is the FIRST snapshot with D > 0.
      const atCrossing = trajectory.snapshots[trajectory.crossingStep!];
      expect(atCrossing.physics.signedDistance).toBeGreaterThan(0);
      for (let i = 0; i < trajectory.crossingStep!; i++) {
        expect(trajectory.snapshots[i].physics.signedDistance).toBeLessThanOrEqual(0);
      }
    } else {
      expect(trajectory.crossingStep).toBeNull();
    }
  });

  it('maxPotential equals the max snapshot potential (crest evidence)', () => {
    const recorder = new CockpitRecorder();
    const trajectory = recorder.recordVariant(baselineVariants()[3]);
    const maxU = Math.max(...trajectory.snapshots.map((s) => s.physics.potential));
    expect(trajectory.maxPotential).toBeCloseTo(maxU, 14);
    expect(trajectory.crestedRidge).toBe(trajectory.maxPotential > 8.9);
  });

  it('snapshots are deterministic for identical inputs', () => {
    const recorder = new CockpitRecorder();
    const a = recorder.recordVariant(baselineVariants()[1]);
    const b = recorder.recordVariant(baselineVariants()[1]);
    expect(a.snapshots.length).toBe(b.snapshots.length);
    for (let i = 0; i < a.snapshots.length; i++) {
      expect(a.snapshots[i].physics.c).toEqual(b.snapshots[i].physics.c);
      expect(a.snapshots[i].physics.kinetic).toBe(b.snapshots[i].physics.kinetic);
    }
  });

  it('replay helper builds trajectories from variant specs', async () => {
    const variants = baselineVariants();
    const trajectories = await Promise.all(
      variants.map((v) => replayVariantAsTrajectory(v))
    );
    expect(trajectories.length).toBe(variants.length);
    expect(trajectories.map((t) => t.spec.name)).toEqual(variants.map((v) => v.name));
  });

  it('exposes the canonical terrain grid size', () => {
    expect(DEFAULT_TERRAIN_GRID).toBeGreaterThanOrEqual(2);
    expect(DEFAULT_TERRAIN_GRID).toBeLessThanOrEqual(512);
  });

  it('terrain patch passthrough preserves the Rust wire shape', () => {
    const patch = sampleTerrainPatch(0.0, 0.0, 0.5, 9);
    expect(patch.n).toBe(9);
    expect(patch.positions.length).toBe(9 * 9 * 3);
    expect(patch.signed.length).toBe(9 * 9);
    expect(patch.realm.length).toBe(9 * 9);
    // Row 0 is the north edge (im = +half).
    expect(patch.positions[1]).toBeCloseTo(0.5, 14);
  });

  it('trajectory snapshots carry all Phase-A diagnostic groups', () => {
    const recorder = new CockpitRecorder();
    const trajectory = recorder.recordVariant(baselineVariants()[3]);
    const s: DebugSnapshot = trajectory.snapshots[10];
    expect(s.version).toBe('debug-snapshot/1');
    // Phase-A groups only: observation arrives with #108 (Phase B).
    expect(s).toHaveProperty('physics');
    expect(s).toHaveProperty('action');
    expect(s).toHaveProperty('map');
    expect(s).toHaveProperty('diagnostics');
    expect(s).not.toHaveProperty('observation');
  });
});
