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
  loadMipPyramid,
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
    // Honor non-default starting c/v ("approach from outside" trajectories
    // that begin at a specific point without paying the launch cost of
    // crossing the cardioid ridge from c=0). When neither is set the
    // synth is left at the controller default (c=0, v=0), preserving the
    // #82 fresh-synth-per-variant discipline for all existing variants.
    if (spec.initialC || spec.initialV) {
      const cPair = spec.initialC ? { re: spec.initialC[0], im: spec.initialC[1] } : undefined;
      const vPair = spec.initialV ? { re: spec.initialV[0], im: spec.initialV[1] } : undefined;
      synth.seed(cPair, vPair);
    }
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
    ManifoldConfig: new (d: number, e: number, l: number, k: number, mu: number) => unknown;
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
  // Controller-default manifold config (kappa=1.0, lambda^2=1.0, eps=1e-4,
  // mu=1/pi for the p=8 secant wall — must mirror Rust ManifoldConfig::default()).
  const config = new m.ManifoldConfig(0.1, 1e-4, 1.0, 1.0, 1.0 / Math.PI);
  return m.debugTerrainPatch(cx, cy, half, n, config);
}

/**
 * Authoritative surface height under ANY (x, y), sampled per-call through
 * the Rust embedding seam — never interpolated from a patch grid.
 *
 * This is the fix for the rider "flying off the mountains": the previous
 * height source returned the PATCH-CENTER sigma, which diverges from the
 * surface at the rider's actual position wherever |grad sigma| is large
 * (near the Shore it reaches ~1e3, so a 5e-4 c-space offset is a >3-sigma
 * height error). Sampling the embedding at the rider's exact position keeps
 * the rider glued to the surface at every scale.
 */
export function riderSurfaceHeight(snap: DebugSnapshot, x: number, y: number): number {
  const m = getWasmModule() as unknown as {
    ManifoldConfig: new (d: number, e: number, l: number, k: number, mu: number) => unknown;
    manifold_embedding: (re: number, im: number, config: unknown) => [number, number, number];
  };
  if (typeof m.manifold_embedding !== 'function') {
    // Mock/test builds: fall back to the snapshot's sigma (the rider's own
    // position height), which keeps vitest honest about the seam shape.
    return snap.physics.sigma;
  }
  const config = new m.ManifoldConfig(0.1, 1e-4, 1.0, 1.0, 1.0 / Math.PI);
  const [, , sigma] = m.manifold_embedding(x, y, config);
  return sigma;
}

/** Terrain level-of-detail plan for one terrain rebuild. */
export interface TerrainLod {
  /** Patch half-extent in c-space (tighter at deep scale). */
  half: number;
  /** Grid resolution n x n (finer at deep scale). */
  grid: number;
  /** Camera far plane in scene units. */
  renderDistance: number;
  /** Fog start distance in scene units. */
  fogNear: number;
  /** Fog full-occlusion distance in scene units. */
  fogFar: number;
}

/** Grid bounds accepted by the Rust terrain_patch seam. */
const LOD_GRID_MIN = 2;
const LOD_GRID_MAX = 512;

/** Scene units per c-space unit (must match cockpitScene.SCENE_SCALE). */
const LOD_SCENE_SCALE = 10.0;

/**
 * Plan terrain fidelity vs performance as the Mandelbrot scale shifts
 * (issue #111: "adjust resolution and render distance as scale shifts").
 *
 * Two constraints, whichever is tighter:
 * 1. SCALE: the local regularized distance rho is the natural LOD signal —
 *    at deep scale (rho -> epsilon ~ 1e-4) interesting structure lives in a
 *    tiny c-space neighborhood, so the patch tightens and the grid refines;
 *    in the valley (rho ~ 0.25) structure is broad, so the patch widens.
 * 2. MOTION: the patch must always exceed the planar ground the rider can
 *    cover between rebuilds (rebuild triggers at 0.6*half drift), or the
 *    surface ends mid-air under a fast rider.
 *
 * Render distance and fog follow the patch's SCENE-space size so the mesh
 * edge always hides inside the fog instead of ending mid-air.
 *
 * Pure function — no hidden state, trivially testable.
 */
export function planTerrainLod(rho: number, planarSpeed = 0): TerrainLod {
  const r = Math.max(rho, 1e-9);
  // Scale-driven half-extent: track rho so the patch spans a roughly
  // constant number of "scale features":
  //   deep scale (1e-4): half ~ 6e-3 (resolves the crest neighborhood)
  //   valley    (0.25):  half ~ 0.5  (broad view)
  const scaleHalf = Math.min(0.5, Math.max(6e-3, r * 2.5));
  // Motion floor: reach the mesh edge (0.4*half before a rebuild) must
  // take many ticks; use a 200-tick margin at the current planar speed.
  const motionHalf = Math.max(scaleHalf, planarSpeed * CANONICAL_DT * 200 / 0.4);
  const half = Math.min(0.5, motionHalf);
  // Grid: keep vertex DENSITY roughly constant as the patch shrinks, so
  // deep-scale patches get MORE resolution per c-space unit without the
  // vertex count (and thus the cost) exploding. Clamped to the Rust seam's
  // accepted grid bounds.
  const grid = Math.round(
    Math.min(
      LOD_GRID_MAX,
      Math.max(LOD_GRID_MIN, 65 * Math.sqrt(0.5 / half), LOD_GRID_MIN)
    )
  );
  // Render distance: the patch diagonal in scene units, padded so the far
  // edge is always beyond the fog wall (never a visible mesh edge).
  const patchScene = half * 2 * LOD_SCENE_SCALE;
  const diagonal = patchScene * Math.SQRT2;
  const renderDistance = diagonal * 1.6;
  // Fog wall sits just inside the render distance, hiding the mesh edge.
  const fogFar = diagonal * 1.15;
  const fogNear = fogFar * 0.45;
  return { half, grid, renderDistance, fogNear, fogFar };
}

/** Convenience: record a trajectory in one call (initializes wasm). */
export async function replayVariantAsTrajectory(
  spec: CrossingVariantSpec
): Promise<CockpitTrajectory> {
  await initOrbitSynth();
  return new CockpitRecorder().recordVariant(spec);
}

/**
 * Load the canonical mip pyramid into the wasm runtime (best-effort) and
 * report whether it is available for the minimap panel. Delegates to the
 * shared loader — no separate artifact path here.
 */
export async function ensureMinimapPyramid(): Promise<boolean> {
  await initOrbitSynth();
  return loadMipPyramid();
}

/**
 * The canonical pyramid extent from the last snapshot's map section, or
 * null when no pyramid is loaded.
 */
export function pyramidExtent(snap: DebugSnapshot): [number, number, number, number] | null {
  return snap.map.extent;
}
