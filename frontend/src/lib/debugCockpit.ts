/**
 * Debug cockpit adapter over the read-only DebugSnapshot seam (issue #111
 * Phase A).
 *
 * This module is a THIN recorder/replayer: it drives hand-authored Controls
 * v2 through the wasm destination physics seam and stores the Rust-owned
 * DebugSnapshot stream. It contains ZERO manifold math — every physics,
 * map, and diagnostics value comes from the canonical Rust seam (ADR 0001).
 *
 * TypeScript owns only:
 * - the recording/replay loop (driving stepWithControls per canonical dt),
 * - crossing bookkeeping derived from the authoritative D(c) in snapshots,
 * - UI-facing types for the cockpit components.
 */

import {
  OrbitSynthesizer,
  initOrbitSynth,
  getWasmModule,
} from './orbitSynthesizer';
import type { CrossingVariantSpec } from './shoreCrossingVariants';
import { expandActions } from './shoreCrossingVariants';

/** Canonical physics timestep (HOP_LENGTH / SAMPLE_RATE, Rust timebase). */
export const CANONICAL_DT = 1024 / 48000;

/** Terrain patch grid resolution for the 3D skate park (n x n vertices). */
export const DEFAULT_TERRAIN_GRID = 65;

/** Terrain patch half-extent in c-space units around the rider. */
export const DEFAULT_TERRAIN_HALF = 0.5;

/** Wire shape of the Rust DebugSnapshot (camelCase, debug-snapshot/1). */
export interface DebugSnapshot {
  version: string;
  timeSeconds: number;
  action: {
    raw: { direction: [number, number]; throttle: number; brake: number; grip: number; impulse: number };
    effective: { direction: [number, number]; throttle: number; brake: number; grip: number; impulse: number };
    driveCovector: [number, number];
    frictionBeta: number;
    frictionPower: number;
  } | null;
  map: {
    pyramidLoaded: boolean;
    shoreProximity: number | null;
    minimapWindow: number[] | null;
    extent: [number, number, number, number] | null;
  };
  physics: {
    c: [number, number];
    velocity: [number, number];
    signedDistance: number;
    realm: number;
    rho: number;
    sigma: number;
    sigmaDot: number;
    scaleGradient: [number, number];
    metric: [number, number, number];
    metricSpeed: number;
    kinetic: number;
    potential: number;
    total: number;
    geodesicAccel: [number, number];
    potentialForce: [number, number];
    netAccel: [number, number];
    derivativeValid: boolean;
  };
  diagnostics: {
    derivativeStep: number;
    valid: boolean;
    lastError: string | null;
    lastDeltaTotal: number | null;
    /** Rust-owned regularized crest potential kappa*log2(d_ref/epsilon). */
    crestPotential: number;
  };
}

/** Wire shape of the Rust TerrainPatch (camelCase). */
export interface TerrainPatch {
  n: number;
  center: [number, number];
  half: number;
  positions: number[];
  signed: number[];
  realm: number[];
}

/** A recorded trajectory: the snapshot stream plus derived crossing evidence. */
export interface CockpitTrajectory {
  spec: CrossingVariantSpec;
  snapshots: DebugSnapshot[];
  /** Step index where D(c) first became > 0 (started inside), or null. */
  crossingStep: number | null;
  crossed: boolean;
  /** Max potential U reached over the trajectory (crest evidence). */
  maxPotential: number;
  /** Whether the trajectory reached the Rust-owned crest neighborhood. */
  crestedRidge: boolean;
}

/**
 * Records DebugSnapshot streams by replaying Controls v2 variants through
 * the destination physics seam. Each recording uses a FRESH synthesizer so
 * no momentum carries across variants (same discipline as the #82 page).
 */
export class CockpitRecorder {
  recordVariant(spec: CrossingVariantSpec): CockpitTrajectory {
    const synth = new OrbitSynthesizer(6);
    const actions = expandActions(spec);
    const snapshots: DebugSnapshot[] = [];
    let crossingStep: number | null = null;
    let maxPotential = -Infinity;
    let crestPotential = Infinity;

    for (const a of actions) {
      synth.stepWithControls(CANONICAL_DT, {
        direction: a.direction,
        throttle: a.throttle,
        brake: a.brake,
        grip: a.grip,
        impulse: a.impulse,
      });
      const snap = currentSnapshot(synth);
      snapshots.push(snap);
      maxPotential = Math.max(maxPotential, snap.physics.potential);
      crestPotential = snap.diagnostics.crestPotential;
      if (crossingStep === null && snap.physics.signedDistance > 0) {
        crossingStep = snapshots.length - 1;
      }
    }

    return {
      spec,
      snapshots,
      crossingStep,
      crossed: crossingStep !== null,
      maxPotential,
      // Crest evidence is judged against the RUST-OWNED crest potential
      // carried by every snapshot (within 1.0 of the ceiling), never a
      // TypeScript-restated constant.
      crestedRidge: maxPotential > crestPotential - 1.0,
    };
  }
}

/**
 * Read the current DebugSnapshot from the synthesizer's wasm controller.
 * The controller's debugSnapshot() is a pure function of its state.
 */
export function currentSnapshot(synth: OrbitSynthesizer): DebugSnapshot {
  const controller = synth.debugSnapshotController();
  if (typeof controller?.debugSnapshot !== 'function') {
    throw new Error(
      '[debugCockpit] wasm build has no debugSnapshot binding; rebuild wasm-orbit'
    );
  }
  return controller.debugSnapshot() as DebugSnapshot;
}

/**
 * Sample a terrain patch of the canonical embedding around a center point.
 * Pure passthrough to the Rust seam — no TS geometry.
 */
export function sampleTerrainPatch(
  cx: number,
  cy: number,
  half: number = DEFAULT_TERRAIN_HALF,
  n: number = DEFAULT_TERRAIN_GRID
): TerrainPatch {
  const m = getWasmModule() as unknown as {
    ManifoldConfig: new (d: number, e: number, l: number, k: number) => unknown;
    debugTerrainPatch: (
      cx: number,
      cy: number,
      half: number,
      n: number,
      config: unknown
    ) => TerrainPatch;
  };
  if (typeof m.debugTerrainPatch !== 'function') {
    throw new Error(
      '[debugCockpit] wasm build has no debugTerrainPatch binding; rebuild wasm-orbit'
    );
  }
  // Controller-default manifold config (kappa=1.0, lambda^2=1.0, eps=1e-4).
  const config = new m.ManifoldConfig(0.1, 1e-4, 1.0, 1.0);
  return m.debugTerrainPatch(cx, cy, half, n, config);
}

/** Convenience: record a trajectory in one call (initializes wasm). */
export async function replayVariantAsTrajectory(
  spec: CrossingVariantSpec
): Promise<CockpitTrajectory> {
  await initOrbitSynth();
  return new CockpitRecorder().recordVariant(spec);
}
