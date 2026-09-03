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
  CANONICAL_DT,
  DEFAULT_TERRAIN_GRID,
  planTerrainLod,
  replayVariantAsTrajectory,
  riderSurfaceHeight,
  sampleTerrainPatch,
  type DebugSnapshot,
} from '../debugCockpit';
import { baselineVariants, expandActions, explorationVariants } from '../shoreCrossingVariants';

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

  it('rider height comes from the authoritative embedding at the rider position', () => {
    // The frozen fixed-view plan (zoom 2.5 => BASE_SPAN/zoom span) resolves
    // near-Shore structure to ~1e-4 c-space; the height source must resolve
    // at least as finely so the rider never floats off the surface.
    const snap = sampleSnapshotWith(0.2549, 0.0, 0.9);
    const h = riderSurfaceHeight(snap, snap.physics.c[0], snap.physics.c[1]);
    // At the rider's own position the height IS the authoritative embedding.
    expect(h).toBeCloseTo(snap.physics.sigma, 12);
  });

  it('rider height resolves steep near-Shore gradients (no flying)', () => {
    // The old patch-center height diverged from the surface whenever the
    // rider moved within the patch: at |grad sigma| ~ 760 near the Shore,
    // a 5e-4 c-space offset is a >3-sigma height error. The authoritative
    // sampler must stay on the surface at the same offsets.
    const snap = sampleSnapshotWith(0.2549, 0.0, 0.9);
    const atRider = riderSurfaceHeight(snap, snap.physics.c[0], snap.physics.c[1]);
    const offset = 5e-4;
    const h1 = riderSurfaceHeight(snap, snap.physics.c[0] + offset, snap.physics.c[1]);
    const h2 = riderSurfaceHeight(snap, snap.physics.c[0] - offset, snap.physics.c[1]);
    // Heights at +/-5e-4 must DIFFER by roughly |grad sigma| * 1e-3 (the
    // surface has real slope there), proving per-position sampling.
    const expectedDelta = Math.hypot(
      snap.physics.scaleGradient[0],
      snap.physics.scaleGradient[1]
    ) * 2 * offset;
    expect(Math.abs(h1 - h2)).toBeGreaterThan(expectedDelta * 0.5);
    // And the rider's own height must sit between them (continuous surface).
    expect(Math.max(h1, h2)).toBeGreaterThan(atRider);
    expect(Math.min(h1, h2)).toBeLessThan(atRider);
  });

  it('LOD tightens the patch and raises resolution as scale rises', () => {
    // Deep scale (rho tiny): small patch, fine grid, short render distance.
    const deep = planTerrainLod(1e-4);
    // Valley (rho large): wide patch, coarse grid, long render distance.
    const valley = planTerrainLod(0.25);
    expect(deep.half).toBeLessThan(valley.half);
    expect(deep.grid).toBeGreaterThanOrEqual(valley.grid);
    expect(deep.renderDistance).toBeLessThan(valley.renderDistance);
    expect(deep.fogNear).toBeLessThan(deep.fogFar);
    expect(valley.fogNear).toBeLessThan(valley.fogFar);
  });

  it('exploration variants drive the rider off the real number line', () => {
    // Issue feedback: the baseline family drives [1,0] only, so the rider
    // never leaves y=0 and the interesting 2D topography stays unseen.
    const exploration = explorationVariants();
    expect(exploration.length).toBeGreaterThan(0);
    for (const variant of exploration) {
      const trajectory = new CockpitRecorder().recordVariant(variant);
      // Somewhere in the trajectory the imaginary part must be substantial
      // relative to the patch scale (the rider genuinely left the axis).
      const maxAbsIm = Math.max(...trajectory.snapshots.map((s) => Math.abs(s.physics.c[1])));
      expect(maxAbsIm).toBeGreaterThan(0.01);
    }
  });

  it('baseline crossing family still drives along the real line', () => {
    // Protect the #82-measured family: directions stay on the axis.
    for (const variant of baselineVariants()) {
      for (const a of expandActions(variant)) {
        expect(a.direction[1]).toBe(0);
      }
    }
  });

  it('LOD stays within the Rust terrain_patch grid bounds', () => {
    for (const rho of [1e-9, 1e-4, 1e-2, 0.25, 2.0]) {
      const lod = planTerrainLod(rho);
      expect(lod.grid).toBeGreaterThanOrEqual(2);
      expect(lod.grid).toBeLessThanOrEqual(512);
      expect(lod.half).toBeGreaterThan(0);
      expect(Number.isFinite(lod.half)).toBe(true);
      expect(lod.renderDistance).toBeGreaterThan(0);
    }
  });

  it('LOD patch always covers the rider footprint between rebuilds', () => {
    // The patch must exceed the planar ground the rider can cover between
    // terrain rebuilds (rebuild triggers at 0.6*half drift from center).
    // Planar speeds come from RECORDED snapshots — data, not restated
    // physics (metric speed != planar speed; G >= I suppresses planar
    // motion near the Shore, which is exactly where the patch tightens).
    const recorder = new CockpitRecorder();
    let maxPlanar = 0;
    for (const variant of baselineVariants()) {
      for (const s of recorder.recordVariant(variant).snapshots) {
        maxPlanar = Math.max(maxPlanar, Math.hypot(s.physics.velocity[0], s.physics.velocity[1]));
      }
    }
    expect(maxPlanar).toBeGreaterThan(0);
    for (const rho of [1e-4, 1e-2, 0.25]) {
      const lod = planTerrainLod(rho, maxPlanar);
      // The rebuild triggers when the rider drifts 0.6*half from the patch
      // center, so the remaining edge clearance is 0.4*half. That must
      // exceed ONE canonical tick of travel at the max recorded planar
      // speed — the rider can then never step off the mesh between rebuilds.
      const edgeDistance = 0.4 * lod.half;
      expect(edgeDistance).toBeGreaterThan(maxPlanar * CANONICAL_DT);
    }
  });
});

/**
 * Helper: build a snapshot-shaped object with overridden physics fields.
 * sigma is derived from the position with the same relation the wasm MOCK
 * embedding uses, so the fixture is internally consistent (a snapshot
 * whose c and sigma disagree would test nothing).
 */
function sampleSnapshotWith(x: number, y: number, metricSpeed: number): DebugSnapshot {
  const recorder = new CockpitRecorder();
  const trajectory = recorder.recordVariant(baselineVariants()[3]);
  const base = trajectory.snapshots[Math.floor(trajectory.snapshots.length / 2)];
  const d = 0.25 - x;
  const rho = Math.sqrt(d * d + 1e-8);
  const sigma = Math.log2(0.1 / rho);
  return {
    ...base,
    physics: {
      ...base.physics,
      c: [x, y],
      sigma,
      metricSpeed,
    },
  };
}
